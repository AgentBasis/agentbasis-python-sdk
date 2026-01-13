from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Try to import BaseCallbackHandler. If not available, we create a dummy class
# so the code doesn't crash on import (though instrument() will check this).
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    class BaseCallbackHandler:
        pass
    LLMResult = Any

tracer = trace.get_tracer("agentbasis.frameworks.langchain")

class AgentBasisCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for LangChain that sends telemetry to AgentBasis via OpenTelemetry.
    """
    def __init__(self):
        super().__init__()
        # We need to track active spans by run_id to close them later
        self.spans: Dict[UUID, trace.Span] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        run_id = kwargs.get("run_id")
        name = "langchain.llm"
        
        # Start Span
        span = tracer.start_span(name)
        span.set_attribute("llm.system", "langchain")
        span.set_attribute("llm.prompts", str(prompts))
        
        if run_id:
            self.spans[run_id] = span

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            # Record Output
            # LangChain response structure is complex, we simplify it for now
            span.set_attribute("llm.output", str(response))
            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when LLM errors."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        run_id = kwargs.get("run_id")
        # Use the chain class name if available
        name = serialized.get("name", "langchain.chain")
        
        span = tracer.start_span(name)
        span.set_attribute("langchain.inputs", str(inputs))
        
        if run_id:
            self.spans[run_id] = span

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.set_attribute("langchain.outputs", str(outputs))
            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when chain errors."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    # We can add on_tool_start / on_tool_end similarly

