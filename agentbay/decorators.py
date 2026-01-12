import functools
from typing import Callable
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

# Create a global tracer for the SDK
# tracer = otel_trace.get_tracer("agentbasis") 
# We shouldn't create it globaly at import time because the provider might not be set yet.

def trace(func: Callable) -> Callable:
    """
    Decorator to track the execution of a function as an OTel Span.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the tracer at runtime, so it uses the configured provider
        tracer = otel_trace.get_tracer("agentbasis")
        
        # Start a new span. 'start_as_current_span' automatically handles 
        # parent/child relationships if one function calls another.
        with tracer.start_as_current_span(func.__name__) as span:
            
            # Record Inputs
            # We convert to string to ensure it fits in a span attribute
            span.set_attribute("code.function", func.__name__)
            span.set_attribute("input.args", str(args))
            span.set_attribute("input.kwargs", str(kwargs))

            try:
                # Run user function
                result = func(*args, **kwargs)
                
                # Record Output
                span.set_attribute("output", str(result))
                span.set_status(Status(StatusCode.OK))
                
                return result

            except Exception as e:
                # Record Error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise e

    return wrapper
