from typing import Any
import functools
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Get our tracer
tracer = trace.get_tracer("agentbasis.llms.openai")

def instrument_chat(openai_module: Any):
    """
    Instruments the OpenAI Chat Completions API with OpenTelemetry.
    """
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return

    original_create = Completions.create

    @functools.wraps(original_create)
    def wrapped_create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Semantic Convention: "chat.completions" or "llm.openai.chat_completions"
        span_name = f"openai.chat.completions.create {model}"
        
        with tracer.start_as_current_span(span_name) as span:
            # 1. Record Input Attributes (Semantic Conventions)
            span.set_attribute("llm.system", "openai")
            span.set_attribute("llm.request.model", model)
            
            # We can serialize complex objects (messages) to string for now
            # In future, we might map them to specific OTel semantic events
            span.set_attribute("llm.request.messages", str(messages))

            try:
                response = original_create(self, *args, **kwargs)

                # 2. Record Response Attributes
                if response.choices:
                    content = response.choices[0].message.content
                    span.set_attribute("llm.response.content", str(content))

                if response.usage:
                    span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
                    span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
                    span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise e

    Completions.create = wrapped_create
