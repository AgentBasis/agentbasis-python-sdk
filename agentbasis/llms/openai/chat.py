from typing import Any
import functools
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


def _get_tracer():
    """
    Get the tracer lazily at runtime.
    This ensures the tracer is retrieved after agentbasis.init() has configured the provider.
    """
    return trace.get_tracer("agentbasis.llms.openai")


def _set_request_attributes(span, model: str, messages: list):
    """
    Set common request attributes on a span.
    """
    span.set_attribute("llm.system", "openai")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.request.messages", str(messages))


def _set_response_attributes(span, response):
    """
    Set common response attributes on a span.
    """
    if response.choices:
        content = response.choices[0].message.content
        span.set_attribute("llm.response.content", str(content))

    if response.usage:
        span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
        span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
        span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)


def instrument_chat(openai_module: Any):
    """
    Instruments the synchronous OpenAI Chat Completions API with OpenTelemetry.
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

        span_name = f"openai.chat.completions.create {model}"
        
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
    This handles `AsyncOpenAI().chat.completions.create()` calls.
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

        span_name = f"openai.chat.completions.create {model}"
        
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
