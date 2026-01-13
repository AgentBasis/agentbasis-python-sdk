from typing import Any, Generator, AsyncGenerator
import functools
import json
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

from agentbasis.context import inject_context_to_span


def _get_tracer():
    """
    Get the tracer lazily at runtime.
    This ensures the tracer is retrieved after agentbasis.init() has configured the provider.
    """
    return trace.get_tracer("agentbasis.llms.gemini")


def _set_request_attributes(span: Span, model: str, contents: Any, tools: list, is_streaming: bool = False):
    """
    Set common request attributes on a span.
    """
    # Inject user/session context
    inject_context_to_span(span)
    
    span.set_attribute("llm.system", "gemini")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.request.messages", str(contents))
    
    if is_streaming:
        span.set_attribute("llm.request.streaming", True)
    
    # Track tools if provided to agent
    if tools:
        try:
            tools_str = json.dumps(tools) if isinstance(tools, (list, dict)) else str(tools)
        except (TypeError, ValueError):
            tools_str = str(tools)
        span.set_attribute("llm.request.tools", tools_str)
        if isinstance(tools, list):
            span.set_attribute("llm.request.tool_count", len(tools))


def _set_response_attributes(span: Span, response):
    """
    Set common response attributes on a span (for non-streaming responses).
    """
    # Record text response
    if hasattr(response, 'text') and response.text:
        span.set_attribute("llm.response.content", str(response.text))

    # Track function/tool calls if used in the agent
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'function_calls') and candidate.function_calls:
            function_calls_data = []
            for func_call in candidate.function_calls:
                func_data = {
                    'name': getattr(func_call, 'name', None),
                }
                # Handle arguments
                arguments = getattr(func_call, 'arguments', None)
                if arguments is not None:
                    if isinstance(arguments, (dict, list)):
                        try:
                            func_data['arguments'] = json.dumps(arguments)
                        except (TypeError, ValueError):
                            func_data['arguments'] = str(arguments)
                    else:
                        func_data['arguments'] = str(arguments)
                else:
                    func_data['arguments'] = None
                
                if hasattr(func_call, 'response') and func_call.response:
                    func_data['response'] = str(func_call.response)
                if hasattr(func_call, 'error') and func_call.error:
                    func_data['error'] = str(func_call.error)
                function_calls_data.append(func_data)
            
            try:
                function_calls_str = json.dumps(function_calls_data)
            except (TypeError, ValueError):
                function_calls_str = str(function_calls_data)
            span.set_attribute("llm.response.function_calls", function_calls_str)
            span.set_attribute("llm.response.function_call_count", len(candidate.function_calls))

    # Track token usage
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        if hasattr(usage, 'prompt_token_count'):
            span.set_attribute("llm.usage.prompt_tokens", usage.prompt_token_count)
        if hasattr(usage, 'candidates_token_count'):
            span.set_attribute("llm.usage.completion_tokens", usage.candidates_token_count)
        if hasattr(usage, 'total_token_count'):
            span.set_attribute("llm.usage.total_tokens", usage.total_token_count)


def _wrap_sync_stream(stream, span: Span, start_time: float) -> Generator:
    """
    Wrap a synchronous streaming response to track chunks and finalize span.
    """
    content_parts = []
    chunk_count = 0
    first_token_time = None
    last_chunk = None
    
    try:
        for chunk in stream:
            chunk_count += 1
            last_chunk = chunk
            
            # Track time to first token
            if first_token_time is None:
                first_token_time = time.time()
                span.set_attribute("llm.response.first_token_ms", 
                                   int((first_token_time - start_time) * 1000))
            
            # Extract content from chunk (Gemini chunks have .text property)
            if hasattr(chunk, 'text') and chunk.text:
                content_parts.append(chunk.text)
            
            yield chunk
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        
        # Try to get final token counts from last chunk
        if last_chunk and hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
            usage = last_chunk.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                span.set_attribute("llm.usage.prompt_tokens", usage.prompt_token_count)
            if hasattr(usage, 'candidates_token_count'):
                span.set_attribute("llm.usage.completion_tokens", usage.candidates_token_count)
            if hasattr(usage, 'total_token_count'):
                span.set_attribute("llm.usage.total_tokens", usage.total_token_count)
        
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
    last_chunk = None
    
    try:
        async for chunk in stream:
            chunk_count += 1
            last_chunk = chunk
            
            # Track time to first token
            if first_token_time is None:
                first_token_time = time.time()
                span.set_attribute("llm.response.first_token_ms", 
                                   int((first_token_time - start_time) * 1000))
            
            # Extract content from chunk
            if hasattr(chunk, 'text') and chunk.text:
                content_parts.append(chunk.text)
            
            yield chunk
        
        # Stream complete - finalize span
        full_content = "".join(content_parts)
        span.set_attribute("llm.response.content", full_content)
        span.set_attribute("llm.response.chunk_count", chunk_count)
        
        # Try to get final token counts from last chunk
        if last_chunk and hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
            usage = last_chunk.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                span.set_attribute("llm.usage.prompt_tokens", usage.prompt_token_count)
            if hasattr(usage, 'candidates_token_count'):
                span.set_attribute("llm.usage.completion_tokens", usage.candidates_token_count)
            if hasattr(usage, 'total_token_count'):
                span.set_attribute("llm.usage.total_tokens", usage.total_token_count)
        
        span.set_status(Status(StatusCode.OK))
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


def _instrument_grpc():
    """
    Optionally instrument gRPC client (Gemini uses gRPC under the hood).
    """
    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
        grpc_instrumentor = GrpcInstrumentorClient()
        grpc_instrumentor.instrument()
    except Exception:
        # gRPC instrumentation is optional, don't fail if it doesn't work
        pass


def instrument_chat(gemini_module: Any):
    """
    Instruments the synchronous Google Gemini Chat API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    _instrument_grpc()
    
    try:
        from google.generativeai import GenerativeModel
    except ImportError:
        return

    original_generate_content = GenerativeModel.generate_content

    @functools.wraps(original_generate_content)
    def wrapped_generate_content(self, *args, **kwargs):
        tracer = _get_tracer()
        model = getattr(self, "model_name", "unknown")
        contents = args[0] if args else kwargs.get('contents', [])
        tools = kwargs.get('tools', [])
        is_streaming = kwargs.get('stream', False)

        span_name = f"gemini.generate_content {model}"

        if is_streaming:
            # For streaming, manually manage span lifecycle
            span = tracer.start_span(span_name)
            start_time = time.time()
            _set_request_attributes(span, model, contents, tools, is_streaming=True)
            
            try:
                stream = original_generate_content(self, *args, **kwargs)
                return _wrap_sync_stream(stream, span, start_time)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise
        else:
            # Non-streaming: use context manager
            with tracer.start_as_current_span(span_name) as span:
                _set_request_attributes(span, model, contents, tools)

                try:
                    response = original_generate_content(self, *args, **kwargs)
                    _set_response_attributes(span, response)
                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    GenerativeModel.generate_content = wrapped_generate_content


def instrument_async_chat(gemini_module: Any):
    """
    Instruments the asynchronous Google Gemini Chat API with OpenTelemetry.
    Handles both regular and streaming responses.
    """
    _instrument_grpc()
    
    try:
        from google.generativeai import GenerativeModel
    except ImportError:
        return

    original_generate_content_async = GenerativeModel.generate_content_async

    @functools.wraps(original_generate_content_async)
    async def wrapped_generate_content_async(self, *args, **kwargs):
        tracer = _get_tracer()
        model = getattr(self, "model_name", "unknown")
        contents = args[0] if args else kwargs.get('contents', [])
        tools = kwargs.get('tools', [])
        is_streaming = kwargs.get('stream', False)

        span_name = f"gemini.generate_content {model}"

        if is_streaming:
            # For streaming, manually manage span lifecycle
            span = tracer.start_span(span_name)
            start_time = time.time()
            span.set_attribute("llm.request.async", True)
            _set_request_attributes(span, model, contents, tools, is_streaming=True)
            
            try:
                stream = await original_generate_content_async(self, *args, **kwargs)
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
                _set_request_attributes(span, model, contents, tools)

                try:
                    response = await original_generate_content_async(self, *args, **kwargs)
                    _set_response_attributes(span, response)
                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    GenerativeModel.generate_content_async = wrapped_generate_content_async
