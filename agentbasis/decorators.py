import functools
import asyncio
from typing import Callable
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

from agentbasis.context import inject_context_to_span


def trace(func: Callable) -> Callable:
    """
    Decorator to track the execution of a function as an OTel Span.
    
    Works with both sync and async functions. Automatically injects
    user/session context from agentbasis.context.
    
    Example:
        @agentbasis.trace
        def my_function():
            ...
        
        @agentbasis.trace
        async def my_async_function():
            ...
    """
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Get the tracer at runtime, so it uses the configured provider
        tracer = otel_trace.get_tracer("agentbasis")
        
        with tracer.start_as_current_span(func.__name__) as span:
            # Inject user/session context
            inject_context_to_span(span)
            
            # Record Inputs
            span.set_attribute("code.function", func.__name__)
            span.set_attribute("input.args", str(args))
            span.set_attribute("input.kwargs", str(kwargs))

            try:
                result = func(*args, **kwargs)
                span.set_attribute("output", str(result))
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Get the tracer at runtime
        tracer = otel_trace.get_tracer("agentbasis")
        
        with tracer.start_as_current_span(func.__name__) as span:
            # Inject user/session context
            inject_context_to_span(span)
            
            # Record Inputs
            span.set_attribute("code.function", func.__name__)
            span.set_attribute("input.args", str(args))
            span.set_attribute("input.kwargs", str(kwargs))

            try:
                result = await func(*args, **kwargs)
                span.set_attribute("output", str(result))
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
