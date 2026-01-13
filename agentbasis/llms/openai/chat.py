from typing import Any, Generator, AsyncGenerator
import functools
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

from agentbasis.context import inject_context_to_span


def _get_tracer():
    """
    Get the tracer lazily at runtime.
    This ensures the tracer is retrieved after agentbasis.init() has configured the provider.
    """
    return trace.get_tracer("agentbasis.llms.openai")


def _set_request_attributes(span: Span, model: str, messages: list, is_streaming: bool = False):
    """
    Set common request attributes on a span.
    """
    # Inject user/session context
    inject_context_to_span(span)
    
    span.set_attribute("llm.system", "openai")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.request.messages", str(messages))
    if is_streaming:
        span.set_attribute("llm.request.streaming", True)


def _set_response_attributes(span: Span, response):
    """
    Set common response attributes on a span (for non-streaming responses).
    """
    if response.choices:
        content = response.choices[0].message.content
        span.set_attribute("llm.response.content", str(content))

    if response.usage:
        span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
        span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
        span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)


def _wrap_sync_stream(stream, span: Span, start_time: float) -> Generator:
    """
    Wrap a synchronous streaming response to track chunks and finalize span.
    """
    content_parts = []
    chunk_count = 0
    first_token_time = None
    
    try:
        for chunk in stream:
            chunk_count += 1
            
            # Track time to first token
            if first_token_time is None:
                first_token_time = time.time()
                span.set_attribute("llm.response.first_token_ms", 
                                   int((first_token_time - start_time) * 1000))
            
            # Extract content from chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content_parts.append(delta.content)
            
            yield chunk
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        span.set_status(Status(StatusCode.OK))
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


async def _wrap_async_stream(stream, span: Span, start_time: float) -> AsyncGenerator:
    """
    Wrap an asynchronous streaming response to track chunks and finalize span.
    """
    content_parts = []
    chunk_count = 0
    first_token_time = None
    
    try:
        async for chunk in stream:
            chunk_count += 1
            
            # Track time to first token
            if first_token_time is None:
                first_token_time = time.time()
                span.set_attribute("llm.response.first_token_ms", 
                                   int((first_token_time - start_time) * 1000))
            
            # Extract content from chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content_parts.append(delta.content)
            
            yield chunk
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        span.set_status(Status(StatusCode.OK))
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


def instrument_chat(openai_module: Any):
    """
    Instruments the synchronous OpenAI Chat Completions API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return

    original_create = Completions.create

    @functools.wraps(original_create)
    def wrapped_create(self, *args, **kwargs):
        tracer = _get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        is_streaming = kwargs.get("stream", False)

        span_name = f"openai.chat.completions.create {model}"
        
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
            # Non-streaming: use context manager as before
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

    Completions.create = wrapped_create


def instrument_async_chat(openai_module: Any):
    """
    Instruments the asynchronous OpenAI Chat Completions API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return

    original_async_create = AsyncCompletions.create

    @functools.wraps(original_async_create)
    async def wrapped_async_create(self, *args, **kwargs):
        tracer = _get_tracer()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        is_streaming = kwargs.get("stream", False)

        span_name = f"openai.chat.completions.create {model}"
        
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
            # Non-streaming: use context manager as before
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

    AsyncCompletions.create = wrapped_async_create
