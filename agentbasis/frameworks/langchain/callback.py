from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
import json

from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Status, StatusCode, Span

from agentbasis.context import inject_context_to_span

# Try to import LangChain types. If not available, we create dummy classes
# so the code doesn't crash on import (though instrument() will check this).
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
except ImportError:
    class BaseCallbackHandler:
        pass
    LLMResult = Any
    Document = Any
    BaseMessage = Any


def _get_tracer():
    """Get the tracer lazily at runtime to ensure it uses the configured provider."""
    return trace.get_tracer("agentbasis.frameworks.langchain")


def _safe_json_dumps(obj: Any) -> str:
    """Safely serialize an object to JSON string, falling back to str() if needed."""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _extract_llm_info(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model name and other LLM info from serialized data."""
    info = {}
    
    # Try various paths where model name might be stored
    if "kwargs" in serialized:
        kwargs = serialized["kwargs"]
        if "model_name" in kwargs:
            info["model"] = kwargs["model_name"]
        elif "model" in kwargs:
            info["model"] = kwargs["model"]
    
    # Get the class name
    if "name" in serialized:
        info["class_name"] = serialized["name"]
    elif "id" in serialized and isinstance(serialized["id"], list):
        info["class_name"] = serialized["id"][-1]
    
    return info


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from LLM response."""
    usage = {}
    
    if hasattr(response, "llm_output") and response.llm_output:
        llm_output = response.llm_output
        if isinstance(llm_output, dict):
            # OpenAI-style token usage
            if "token_usage" in llm_output:
                token_usage = llm_output["token_usage"]
                usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
                usage["completion_tokens"] = token_usage.get("completion_tokens", 0)
                usage["total_tokens"] = token_usage.get("total_tokens", 0)
            # Direct usage field
            elif "usage" in llm_output:
                token_usage = llm_output["usage"]
                usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
                usage["completion_tokens"] = token_usage.get("completion_tokens", 0)
                usage["total_tokens"] = token_usage.get("total_tokens", 0)
    
    return usage


def _extract_response_content(response: Any) -> str:
    """Extract the text content from an LLM response."""
    if hasattr(response, "generations") and response.generations:
        # Get first generation from first prompt
        first_gen_list = response.generations[0]
        if first_gen_list:
            first_gen = first_gen_list[0]
            if hasattr(first_gen, "text"):
                return first_gen.text
            elif hasattr(first_gen, "message") and hasattr(first_gen.message, "content"):
                return first_gen.message.content
    return str(response)


class AgentBasisCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for LangChain that sends telemetry to AgentBasis via OpenTelemetry.
    
    Supports:
    - LLM calls (on_llm_start/end/error)
    - Chain execution (on_chain_start/end/error)
    - Tool invocations (on_tool_start/end/error)
    - Retriever operations (on_retriever_start/end/error)
    - Parent-child span relationships (nested traces)
    
    Usage:
        from agentbasis.frameworks.langchain import AgentBasisCallbackHandler
        
        handler = AgentBasisCallbackHandler()
        chain.invoke({"query": "..."}, config={"callbacks": [handler]})
    
    Trace Structure Example:
        └─ langchain.chain.RetrievalQA
           ├─ langchain.retriever.VectorStoreRetriever
           └─ langchain.llm.ChatOpenAI
    """
    
    def __init__(self):
        super().__init__()
        # Track active spans by run_id to close them later
        self.spans: Dict[UUID, Span] = {}
        # Track span contexts for parent-child relationships
        self.span_contexts: Dict[UUID, otel_context.Context] = {}
    
    def _start_span(self, span_name: str, parent_run_id: Optional[UUID] = None) -> Span:
        """
        Start a new span, optionally as a child of a parent span.
        
        Automatically injects user/session context (user_id, session_id, 
        conversation_id, metadata) from agentbasis.context.
        
        Args:
            span_name: Name for the new span
            parent_run_id: The run_id of the parent operation (if any)
            
        Returns:
            The newly created span
        """
        tracer = _get_tracer()
        
        # Check if we have a parent span to nest under
        if parent_run_id and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id]
            # Create a context with the parent span
            parent_context = trace.set_span_in_context(parent_span)
            # Start the new span as a child
            span = tracer.start_span(span_name, context=parent_context)
        else:
            # No parent, start a root span
            span = tracer.start_span(span_name)
        
        # Inject user/session context (user_id, session_id, etc.)
        inject_context_to_span(span)
        
        return span
    
    def _store_span(self, run_id: Optional[UUID], span: Span) -> None:
        """Store a span for later retrieval and for use as a parent."""
        if run_id:
            self.spans[run_id] = span
    
    def _end_span(self, run_id: Optional[UUID], status: Status) -> Optional[Span]:
        """End a span and remove it from tracking."""
        span = self.spans.pop(run_id, None) if run_id else None
        if span:
            span.set_status(status)
            span.end()
        return span

    # ==================== LLM Callbacks ====================
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        
        # Extract model info
        llm_info = _extract_llm_info(serialized)
        model = llm_info.get("model", "unknown")
        class_name = llm_info.get("class_name", "LLM")
        
        span_name = f"langchain.llm.{class_name}"
        span = self._start_span(span_name, parent_run_id)
        
        # Set attributes
        span.set_attribute("llm.system", "langchain")
        span.set_attribute("llm.request.model", model)
        span.set_attribute("llm.request.prompts", _safe_json_dumps(prompts))
        span.set_attribute("llm.request.prompt_count", len(prompts))
        
        self._store_span(run_id, span)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            # Extract and set response content
            content = _extract_response_content(response)
            span.set_attribute("llm.response.content", content)
            
            # Extract and set token usage
            usage = _extract_token_usage(response)
            if usage:
                span.set_attribute("llm.usage.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("llm.usage.completion_tokens", usage.get("completion_tokens", 0))
                span.set_attribute("llm.usage.total_tokens", usage.get("total_tokens", 0))
            
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

    # ==================== Chain Callbacks ====================
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        
        # Get chain name from serialized data
        chain_name = serialized.get("name")
        if not chain_name and "id" in serialized and isinstance(serialized["id"], list):
            chain_name = serialized["id"][-1]
        chain_name = chain_name or "Chain"
        
        span_name = f"langchain.chain.{chain_name}"
        span = self._start_span(span_name, parent_run_id)
        
        span.set_attribute("langchain.chain.name", chain_name)
        span.set_attribute("langchain.chain.inputs", _safe_json_dumps(inputs))
        
        self._store_span(run_id, span)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.set_attribute("langchain.chain.outputs", _safe_json_dumps(outputs))
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

    # ==================== Tool Callbacks ====================
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        
        # Get tool name
        tool_name = serialized.get("name")
        if not tool_name and "id" in serialized and isinstance(serialized["id"], list):
            tool_name = serialized["id"][-1]
        tool_name = tool_name or "Tool"
        
        span_name = f"langchain.tool.{tool_name}"
        span = self._start_span(span_name, parent_run_id)
        
        span.set_attribute("langchain.tool.name", tool_name)
        span.set_attribute("langchain.tool.input", input_str)
        
        # Get tool description if available
        if "description" in serialized:
            span.set_attribute("langchain.tool.description", serialized["description"])
        
        self._store_span(run_id, span)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.set_attribute("langchain.tool.output", str(output))
            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when tool errors."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    # ==================== Retriever Callbacks ====================
    
    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> Any:
        """Run when retriever starts running."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        
        # Get retriever name
        retriever_name = serialized.get("name")
        if not retriever_name and "id" in serialized and isinstance(serialized["id"], list):
            retriever_name = serialized["id"][-1]
        retriever_name = retriever_name or "Retriever"
        
        span_name = f"langchain.retriever.{retriever_name}"
        span = self._start_span(span_name, parent_run_id)
        
        span.set_attribute("langchain.retriever.name", retriever_name)
        span.set_attribute("langchain.retriever.query", query)
        
        self._store_span(run_id, span)

    def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> Any:
        """Run when retriever ends running."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            # Record document count
            span.set_attribute("langchain.retriever.document_count", len(documents))
            
            # Extract document metadata and content summaries
            doc_summaries = []
            for i, doc in enumerate(documents[:10]):  # Limit to first 10 docs
                doc_info = {"index": i}
                if hasattr(doc, "page_content"):
                    # Truncate content for span attribute
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    doc_info["content_preview"] = content
                if hasattr(doc, "metadata") and doc.metadata:
                    doc_info["metadata"] = doc.metadata
                doc_summaries.append(doc_info)
            
            span.set_attribute("langchain.retriever.documents", _safe_json_dumps(doc_summaries))
            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when retriever errors."""
        run_id = kwargs.get("run_id")
        span = self.spans.pop(run_id, None)
        
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

