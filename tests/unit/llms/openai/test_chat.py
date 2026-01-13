import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import types
import asyncio

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

# Create the sync Completions Class
class MockCompletions:
    def create(self, *args, **kwargs):
        pass

# Create the async AsyncCompletions Class
class MockAsyncCompletions:
    async def create(self, *args, **kwargs):
        pass

mock_completions_module.Completions = MockCompletions
mock_completions_module.AsyncCompletions = MockAsyncCompletions

# Connect them
mock_openai.resources = mock_resources
mock_resources.chat = mock_chat
mock_chat.completions = mock_completions_module

# Register them in sys.modules so 'from openai...' works
sys.modules["openai"] = mock_openai
sys.modules["openai.resources"] = mock_resources
sys.modules["openai.resources.chat"] = mock_chat
sys.modules["openai.resources.chat.completions"] = mock_completions_module
# --- MOCK SETUP END ---

# --- OTEL SETUP (Module level - shared across all tests) ---
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_processor = SimpleSpanProcessor(_exporter)
_provider.add_span_processor(_processor)
trace._set_tracer_provider(_provider, log=False)
# --- OTEL SETUP END ---

from agentbasis.llms.openai import instrument
from agentbasis.llms.openai.chat import instrument_chat, instrument_async_chat


class TestOpenAIChat(unittest.TestCase):
    """Tests for synchronous OpenAI Chat Completions instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the Completions.create to a fresh mock before each test
        self.mock_create = MagicMock()
        MockCompletions.create = self.mock_create
        
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
        completions_instance = MockCompletions()
        response = completions_instance.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        self.assertEqual(response, mock_response)
        
        # 4. Verify OTel Span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("gpt-4", span.name)
        self.assertEqual(span.attributes["llm.system"], "openai")
        self.assertEqual(span.attributes["llm.request.model"], "gpt-4")
        self.assertEqual(span.attributes["llm.response.content"], "AI Response")
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 15)

    def test_sync_error_handling(self):
        """Test that errors in sync calls are properly recorded."""
        
        # 1. Run instrumentation
        instrument_chat(mock_openai)
        
        # 2. Setup the mock to raise an exception
        self.mock_create.side_effect = Exception("Sync API Error")
        
        # 3. Call the method and expect an error
        completions_instance = MockCompletions()
        
        with self.assertRaises(Exception) as context:
            completions_instance.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        self.assertEqual(str(context.exception), "Sync API Error")
        
        # 4. Verify Span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        
        # Check that we recorded an exception event
        exception_events = [e for e in span.events if e.name == "exception"]
        self.assertGreaterEqual(len(exception_events), 1)


class TestOpenAIAsyncChat(unittest.TestCase):
    """Tests for async OpenAI Chat Completions instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the AsyncCompletions.create to a fresh mock before each test
        self.mock_async_create = AsyncMock()
        MockAsyncCompletions.create = self.mock_async_create
        
    def test_async_instrumentation_wraps_create(self):
        """Test that calling async create() triggers our wrapper and OTel span."""
        
        # 1. Run instrumentation
        instrument_async_chat(mock_openai)
        
        # 2. Setup a fake response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Async AI Response"))]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        
        self.mock_async_create.return_value = mock_response
        
        # 3. Call the async method
        async def run_async_test():
            completions_instance = MockAsyncCompletions()
            response = await completions_instance.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": "Hello async"}]
            )
            return response
        
        response = asyncio.run(run_async_test())
        
        self.assertEqual(response, mock_response)
        
        # 4. Verify OTel Span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("gpt-4-turbo", span.name)
        self.assertEqual(span.attributes["llm.system"], "openai")
        self.assertEqual(span.attributes["llm.request.model"], "gpt-4-turbo")
        self.assertEqual(span.attributes["llm.request.async"], True)
        self.assertEqual(span.attributes["llm.response.content"], "Async AI Response")
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 30)

    def test_async_error_handling(self):
        """Test that errors in async calls are properly recorded."""
        
        # 1. Run instrumentation
        instrument_async_chat(mock_openai)
        
        # 2. Setup the mock to raise an exception
        self.mock_async_create.side_effect = Exception("API Error")
        
        # 3. Call the async method and expect an error
        async def run_async_error_test():
            completions_instance = MockAsyncCompletions()
            await completions_instance.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        with self.assertRaises(Exception) as context:
            asyncio.run(run_async_error_test())
        
        self.assertEqual(str(context.exception), "API Error")
        
        # 4. Verify Span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        
        # Check that we recorded an exception event
        exception_events = [e for e in span.events if e.name == "exception"]
        self.assertGreaterEqual(len(exception_events), 1)


if __name__ == "__main__":
    unittest.main()
