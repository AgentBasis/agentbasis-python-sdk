import unittest
from unittest.mock import MagicMock, patch
import sys
import types

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# --- MOCK SETUP START ---
# We must setup the mocks BEFORE importing the module under test
# This simulates 'openai' being installed on the system

mock_openai = types.ModuleType("openai")
mock_resources = types.ModuleType("openai.resources")
mock_chat = types.ModuleType("openai.resources.chat")
mock_completions_module = types.ModuleType("openai.resources.chat.completions")

# 2. Create the Completions Class and create method
class MockCompletions:
    def create(self, *args, **kwargs):
        pass

mock_completions_module.Completions = MockCompletions

# 3. Connect them
mock_openai.resources = mock_resources
mock_resources.chat = mock_chat
mock_chat.completions = mock_completions_module

# 4. Register them in sys.modules so 'from openai...' works
sys.modules["openai"] = mock_openai
sys.modules["openai.resources"] = mock_resources
sys.modules["openai.resources.chat"] = mock_chat
sys.modules["openai.resources.chat.completions"] = mock_completions_module
# --- MOCK SETUP END ---

from agentbasis.llms.openai import instrument
from agentbasis.llms.openai.chat import instrument_chat

class TestOpenAIChat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup OTel for testing
        cls.exporter = InMemorySpanExporter()
        cls.provider = TracerProvider()
        processor = SimpleSpanProcessor(cls.exporter)
        cls.provider.add_span_processor(processor)
        trace._set_tracer_provider(cls.provider, log=False)

    def setUp(self):
        self.exporter.clear()
        self.mock_create = MagicMock()
        # Update the mock on the class method
        sys.modules["openai.resources.chat.completions"].Completions.create = self.mock_create
        
    def test_instrumentation_wraps_create(self):
        """Test that calling create() triggers our wrapper and OTel span."""
        
        # 1. Run instrumentation
        instrument_chat(mock_openai)
        
        # 2. Setup a fake response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="AI Response"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        self.mock_create.return_value = mock_response
        
        # 3. Call the method (simulating user code)
        # We need to instantiate the class first, because 'create' expects 'self'
        completions_instance = sys.modules["openai.resources.chat.completions"].Completions()
        response = completions_instance.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        self.assertEqual(response, mock_response)
        
        # 4. Verify OTel Span
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("gpt-4", span.name)
        self.assertEqual(span.attributes["llm.system"], "openai")
        self.assertEqual(span.attributes["llm.request.model"], "gpt-4")
        self.assertEqual(span.attributes["llm.response.content"], "AI Response")
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 15)

if __name__ == "__main__":
    unittest.main()
