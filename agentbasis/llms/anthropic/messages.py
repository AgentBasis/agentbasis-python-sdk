from typing import Any, Generator, AsyncGenerator
import functools
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span


def _get_tracer():
    """
    Get the tracer lazily at runtime.
    This ensures the tracer is retrieved after agentbasis.init() has configured the provider.
    """
    return trace.get_tracer("agentbasis.llms.anthropic")


def _set_request_attributes(span: Span, model: str, messages: list, is_streaming: bool = False):
    """
    Set common request attributes on a span.
    """
    span.set_attribute("llm.system", "anthropic")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.request.messages", str(messages))
    if is_streaming:
        span.set_attribute("llm.request.streaming", True)


def _set_response_attributes(span: Span, response):
    """
    Set common response attributes on a span (for non-streaming responses).
    
    Anthropic response structure:
    - response.content: list of content blocks (e.g., [{"type": "text", "text": "..."}])
    - response.usage.input_tokens
    - response.usage.output_tokens
    - response.model
    - response.stop_reason
    """
    # Extract text content from response
    if response.content:
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
        content = "".join(text_parts)
        span.set_attribute("llm.response.content", content)
    
    # Set stop reason
    if hasattr(response, 'stop_reason') and response.stop_reason:
        span.set_attribute("llm.response.stop_reason", response.stop_reason)
    
    # Set token usage (Anthropic uses input_tokens/output_tokens)
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'input_tokens', 0)
        output_tokens = getattr(response.usage, 'output_tokens', 0)
        span.set_attribute("llm.usage.prompt_tokens", input_tokens)
        span.set_attribute("llm.usage.completion_tokens", output_tokens)
        span.set_attribute("llm.usage.total_tokens", input_tokens + output_tokens)


def _wrap_sync_stream(stream, span: Span, start_time: float) -> Generator:
    """
    Wrap a synchronous Anthropic streaming response to track chunks and finalize span.
    
    Anthropic streaming events:
    - message_start: Contains initial message info
    - content_block_start: Start of a content block
    - content_block_delta: Text delta with 'delta.text'
    - content_block_stop: End of content block
    - message_delta: Final message info with usage stats
    - message_stop: Stream complete
    """
    content_parts = []
    chunk_count = 0
    first_token_time = None
    input_tokens = 0
    output_tokens = 0
    
    try:
        for event in stream:
            chunk_count += 1
            
            # Track time to first content
            if first_token_time is None and hasattr(event, 'type'):
                if event.type == 'content_block_delta':
                    first_token_time = time.time()
                    span.set_attribute("llm.response.first_token_ms", 
                                       int((first_token_time - start_time) * 1000))
            
            # Extract content from delta events
            if hasattr(event, 'type'):
                if event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        content_parts.append(event.delta.text)
                
                # Capture usage from message_start or message_delta
                elif event.type == 'message_start':
                    if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                        input_tokens = getattr(event.message.usage, 'input_tokens', 0)
                
                elif event.type == 'message_delta':
                    if hasattr(event, 'usage'):
                        output_tokens = getattr(event.usage, 'output_tokens', 0)
            
            yield event
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        
        if input_tokens or output_tokens:
            span.set_attribute("llm.usage.prompt_tokens", input_tokens)
            span.set_attribute("llm.usage.completion_tokens", output_tokens)
            span.set_attribute("llm.usage.total_tokens", input_tokens + output_tokens)
        
        span.set_status(Status(StatusCode.OK))
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


async def _wrap_async_stream(stream, span: Span, start_time: float) -> AsyncGenerator:
    """
    Wrap an asynchronous Anthropic streaming response to track chunks and finalize span.
    """
    content_parts = []
    chunk_count = 0
    first_token_time = None
    input_tokens = 0
    output_tokens = 0
    
    try:
        async for event in stream:
            chunk_count += 1
            
            # Track time to first content
            if first_token_time is None and hasattr(event, 'type'):
                if event.type == 'content_block_delta':
                    first_token_time = time.time()
                    span.set_attribute("llm.response.first_token_ms", 
                                       int((first_token_time - start_time) * 1000))
            
            # Extract content from delta events
            if hasattr(event, 'type'):
                if event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        content_parts.append(event.delta.text)
                
                elif event.type == 'message_start':
                    if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                        input_tokens = getattr(event.message.usage, 'input_tokens', 0)
                
                elif event.type == 'message_delta':
                    if hasattr(event, 'usage'):
                        output_tokens = getattr(event.usage, 'output_tokens', 0)
            
            yield event
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        
        if input_tokens or output_tokens:
            span.set_attribute("llm.usage.prompt_tokens", input_tokens)
            span.set_attribute("llm.usage.completion_tokens", output_tokens)
            span.set_attribute("llm.usage.total_tokens", input_tokens + output_tokens)
        
        span.set_status(Status(StatusCode.OK))
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


def instrument_messages(anthropic_module: Any):
    """
    Instruments the synchronous Anthropic Messages API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    try:
        from anthropic.resources.messages import Messages
    except ImportError:
        return

    original_create = Messages.create

    @functools.wraps(original_create)
    def wrapped_create(self, *args, **kwargs):
        tracer = _get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        is_streaming = kwargs.get("stream", False)

        span_name = f"anthropic.messages.create {model}"
        
        if is_streaming:
            # For streaming, we need to manually manage the span lifecycle
            span = tracer.start_span(span_name)
            start_time = time.time()
            _set_request_attributes(span, model, messages, is_streaming=True)
            
            try:
                stream = original_create(self, *args, **kwargs)
                # Return wrapped generator that will finalize span when exhausted
                return _wrap_sync_stream(stream, span, start_time)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise
        else:
            # Non-streaming: use context manager
            with tracer.start_as_current_span(span_name) as span:
                _set_request_attributes(span, model, messages)

                try:
                    response = original_create(self, *args, **kwargs)
                    _set_response_attributes(span, response)
                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    Messages.create = wrapped_create


def instrument_async_messages(anthropic_module: Any):
    """
    Instruments the asynchronous Anthropic Messages API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    try:
        from anthropic.resources.messages import AsyncMessages
    except ImportError:
        return

    original_async_create = AsyncMessages.create

    @functools.wraps(original_async_create)
    async def wrapped_async_create(self, *args, **kwargs):
        tracer = _get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        is_streaming = kwargs.get("stream", False)

        span_name = f"anthropic.messages.create {model}"
        
        if is_streaming:
            # For streaming, we need to manually manage the span lifecycle
            span = tracer.start_span(span_name)
            start_time = time.time()
            span.set_attribute("llm.request.async", True)
            _set_request_attributes(span, model, messages, is_streaming=True)
            
            try:
                stream = await original_async_create(self, *args, **kwargs)
                # Return wrapped async generator that will finalize span when exhausted
                return _wrap_async_stream(stream, span, start_time)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise
        else:
            # Non-streaming: use context manager
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("llm.request.async", True)
                _set_request_attributes(span, model, messages)

                try:
                    response = await original_async_create(self, *args, **kwargs)
                    _set_response_attributes(span, response)
                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    AsyncMessages.create = wrapped_async_create
