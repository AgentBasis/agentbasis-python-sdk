import unittest
from uuid import uuid4
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentbasis.frameworks.langchain.callback import AgentBasisCallbackHandler

class TestLangChainCallback(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup OTel
        cls.exporter = InMemorySpanExporter()
        cls.provider = TracerProvider()
        processor = SimpleSpanProcessor(cls.exporter)
        cls.provider.add_span_processor(processor)
        trace._set_tracer_provider(cls.provider, log=False)

    def setUp(self):
        self.exporter.clear()
        self.handler = AgentBasisCallbackHandler()

    def test_llm_lifecycle(self):
        """Test on_llm_start and on_llm_end create a span."""
        run_id = uuid4()
        
        # 1. Start LLM
        self.handler.on_llm_start(
            serialized={}, 
            prompts=["Hello AI"], 
            run_id=run_id
        )
        
        # 2. End LLM
        self.handler.on_llm_end(
            response="Hello Human", 
            run_id=run_id
        )
        
        # 3. Verify Span
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "langchain.llm")
        self.assertEqual(span.attributes["llm.system"], "langchain")
        self.assertIn("Hello AI", span.attributes["llm.prompts"])
        self.assertIn("Hello Human", span.attributes["llm.output"])

    def test_chain_lifecycle(self):
        """Test on_chain_start and on_chain_end."""
        run_id = uuid4()
        
        # 1. Start Chain
        self.handler.on_chain_start(
            serialized={"name": "MyChain"}, 
            inputs={"query": "test"}, 
            run_id=run_id
        )
        
        # 2. End Chain
        self.handler.on_chain_end(
            outputs={"text": "result"}, 
            run_id=run_id
        )
        
        # 3. Verify Span
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.name, "MyChain")
        self.assertIn("test", span.attributes["langchain.inputs"])

if __name__ == "__main__":
    unittest.main()

