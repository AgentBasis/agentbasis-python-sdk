import unittest
from unittest.mock import MagicMock
from uuid import uuid4
import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentbasis.frameworks.langchain.callback import AgentBasisCallbackHandler
from agentbasis.frameworks.langchain import (
    instrument,
    get_callback_handler,
    get_callback_config,
)
import agentbasis.context as ctx


# Setup OTel at module level
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_processor = SimpleSpanProcessor(_exporter)
_provider.add_span_processor(_processor)
trace._set_tracer_provider(_provider, log=False)


class TestLangChainLLMCallbacks(unittest.TestCase):
    """Tests for LLM callbacks."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_llm_start_end_creates_span(self):
        """Test on_llm_start and on_llm_end create a span with correct attributes."""
        run_id = uuid4()
        
        # Create mock LLM response
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Hello Human")]]
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        # 1. Start LLM
        self.handler.on_llm_start(
            serialized={"name": "ChatOpenAI", "kwargs": {"model": "gpt-4"}}, 
            prompts=["Hello AI"], 
            run_id=run_id
        )
        
        # 2. End LLM
        self.handler.on_llm_end(
            response=mock_response, 
            run_id=run_id
        )
        
        # 3. Verify Span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "langchain.llm.ChatOpenAI")
        self.assertEqual(span.attributes["llm.system"], "langchain")
        self.assertEqual(span.attributes["llm.request.model"], "gpt-4")
        self.assertIn("Hello AI", span.attributes["llm.request.prompts"])
        self.assertEqual(span.attributes["llm.request.prompt_count"], 1)
        self.assertEqual(span.attributes["llm.response.content"], "Hello Human")
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 15)

    def test_llm_error_records_exception(self):
        """Test that LLM errors are properly recorded."""
        run_id = uuid4()
        
        self.handler.on_llm_start(
            serialized={"name": "ChatOpenAI"}, 
            prompts=["Hello"], 
            run_id=run_id
        )
        
        self.handler.on_llm_error(
            error=Exception("API Error"),
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        
        # Check exception was recorded
        exception_events = [e for e in span.events if e.name == "exception"]
        self.assertGreaterEqual(len(exception_events), 1)


class TestLangChainChainCallbacks(unittest.TestCase):
    """Tests for Chain callbacks."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_chain_start_end_creates_span(self):
        """Test on_chain_start and on_chain_end create a span."""
        run_id = uuid4()
        
        self.handler.on_chain_start(
            serialized={"name": "RetrievalQA"}, 
            inputs={"query": "What is AI?"}, 
            run_id=run_id
        )
        
        self.handler.on_chain_end(
            outputs={"result": "AI is..."}, 
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "langchain.chain.RetrievalQA")
        self.assertEqual(span.attributes["langchain.chain.name"], "RetrievalQA")
        
        # Verify JSON serialization
        inputs = json.loads(span.attributes["langchain.chain.inputs"])
        self.assertEqual(inputs["query"], "What is AI?")

    def test_chain_error_records_exception(self):
        """Test that chain errors are properly recorded."""
        run_id = uuid4()
        
        self.handler.on_chain_start(
            serialized={"name": "MyChain"}, 
            inputs={}, 
            run_id=run_id
        )
        
        self.handler.on_chain_error(
            error=ValueError("Chain failed"),
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)


class TestLangChainToolCallbacks(unittest.TestCase):
    """Tests for Tool callbacks."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_tool_start_end_creates_span(self):
        """Test on_tool_start and on_tool_end create a span."""
        run_id = uuid4()
        
        self.handler.on_tool_start(
            serialized={"name": "search", "description": "Search the web"}, 
            input_str="python tutorials", 
            run_id=run_id
        )
        
        self.handler.on_tool_end(
            output="Found 10 results", 
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "langchain.tool.search")
        self.assertEqual(span.attributes["langchain.tool.name"], "search")
        self.assertEqual(span.attributes["langchain.tool.input"], "python tutorials")
        self.assertEqual(span.attributes["langchain.tool.description"], "Search the web")
        self.assertEqual(span.attributes["langchain.tool.output"], "Found 10 results")

    def test_tool_error_records_exception(self):
        """Test that tool errors are properly recorded."""
        run_id = uuid4()
        
        self.handler.on_tool_start(
            serialized={"name": "calculator"}, 
            input_str="1/0", 
            run_id=run_id
        )
        
        self.handler.on_tool_error(
            error=ZeroDivisionError("division by zero"),
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)


class TestLangChainRetrieverCallbacks(unittest.TestCase):
    """Tests for Retriever callbacks."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_retriever_start_end_creates_span(self):
        """Test on_retriever_start and on_retriever_end create a span."""
        run_id = uuid4()
        
        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Document 1 content"
        mock_doc1.metadata = {"source": "file1.txt"}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Document 2 content"
        mock_doc2.metadata = {"source": "file2.txt"}
        
        self.handler.on_retriever_start(
            serialized={"name": "VectorStoreRetriever"}, 
            query="What is Python?", 
            run_id=run_id
        )
        
        self.handler.on_retriever_end(
            documents=[mock_doc1, mock_doc2], 
            run_id=run_id
        )
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "langchain.retriever.VectorStoreRetriever")
        self.assertEqual(span.attributes["langchain.retriever.name"], "VectorStoreRetriever")
        self.assertEqual(span.attributes["langchain.retriever.query"], "What is Python?")
        self.assertEqual(span.attributes["langchain.retriever.document_count"], 2)
        
        # Verify documents are serialized
        docs = json.loads(span.attributes["langchain.retriever.documents"])
        self.assertEqual(len(docs), 2)
        self.assertIn("Document 1 content", docs[0]["content_preview"])


class TestLangChainParentChildSpans(unittest.TestCase):
    """Tests for parent-child span relationships."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_llm_nested_under_chain(self):
        """Test that LLM spans are children of chain spans."""
        chain_run_id = uuid4()
        llm_run_id = uuid4()
        
        # Start chain
        self.handler.on_chain_start(
            serialized={"name": "QAChain"}, 
            inputs={"query": "test"}, 
            run_id=chain_run_id
        )
        
        # Start LLM (with parent_run_id)
        self.handler.on_llm_start(
            serialized={"name": "ChatOpenAI"}, 
            prompts=["Hello"], 
            run_id=llm_run_id,
            parent_run_id=chain_run_id
        )
        
        # End LLM
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Response")]]
        mock_response.llm_output = None
        
        self.handler.on_llm_end(response=mock_response, run_id=llm_run_id)
        
        # End chain
        self.handler.on_chain_end(outputs={"result": "done"}, run_id=chain_run_id)
        
        # Verify spans
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        
        # Find the spans
        chain_span = next(s for s in spans if "chain" in s.name)
        llm_span = next(s for s in spans if "llm" in s.name)
        
        # Verify parent-child relationship
        self.assertEqual(llm_span.parent.span_id, chain_span.context.span_id)

    def test_tool_nested_under_chain(self):
        """Test that tool spans are children of chain spans."""
        chain_run_id = uuid4()
        tool_run_id = uuid4()
        
        # Start chain
        self.handler.on_chain_start(
            serialized={"name": "AgentExecutor"}, 
            inputs={}, 
            run_id=chain_run_id
        )
        
        # Start tool (with parent_run_id)
        self.handler.on_tool_start(
            serialized={"name": "calculator"}, 
            input_str="2+2", 
            run_id=tool_run_id,
            parent_run_id=chain_run_id
        )
        
        # End tool
        self.handler.on_tool_end(output="4", run_id=tool_run_id)
        
        # End chain
        self.handler.on_chain_end(outputs={}, run_id=chain_run_id)
        
        # Verify parent-child relationship
        spans = _exporter.get_finished_spans()
        chain_span = next(s for s in spans if "chain" in s.name)
        tool_span = next(s for s in spans if "tool" in s.name)
        
        self.assertEqual(tool_span.parent.span_id, chain_span.context.span_id)


class TestLangChainContextInjection(unittest.TestCase):
    """Tests for context injection (user_id, session_id, etc.)."""
    
    def setUp(self):
        _exporter.clear()
        self.handler = AgentBasisCallbackHandler()
        # Clear any existing context
        ctx.set_user(None)
        ctx.set_session(None)
        ctx.set_conversation(None)

    def tearDown(self):
        # Clean up context after each test
        ctx.set_user(None)
        ctx.set_session(None)
        ctx.set_conversation(None)

    def test_user_context_injected_into_spans(self):
        """Test that user context is injected into spans."""
        run_id = uuid4()
        
        # Set user context
        ctx.set_user("user-123")
        ctx.set_session("session-456")
        
        self.handler.on_llm_start(
            serialized={"name": "ChatOpenAI"}, 
            prompts=["Hello"], 
            run_id=run_id
        )
        
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Hi")]]
        mock_response.llm_output = None
        
        self.handler.on_llm_end(response=mock_response, run_id=run_id)
        
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.attributes["agentbasis.user.id"], "user-123")
        self.assertEqual(span.attributes["agentbasis.session.id"], "session-456")


class TestLangChainModuleFunctions(unittest.TestCase):
    """Tests for the module-level functions in __init__.py."""
    
    def test_get_callback_handler_returns_handler(self):
        """Test that get_callback_handler returns a new handler."""
        handler = get_callback_handler()
        self.assertIsInstance(handler, AgentBasisCallbackHandler)
        
        # Should return different instances
        handler2 = get_callback_handler()
        self.assertIsNot(handler, handler2)

    def test_instrument_returns_singleton(self):
        """Test that instrument() returns the same handler."""
        handler1 = instrument()
        handler2 = instrument()
        self.assertIs(handler1, handler2)

    def test_get_callback_config_returns_dict(self):
        """Test that get_callback_config returns a valid config dict."""
        config = get_callback_config()
        self.assertIsInstance(config, dict)
        self.assertIn("callbacks", config)
        self.assertEqual(len(config["callbacks"]), 1)
        self.assertIsInstance(config["callbacks"][0], AgentBasisCallbackHandler)


if __name__ == "__main__":
    unittest.main()
