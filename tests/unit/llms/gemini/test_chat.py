import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import types
import json
import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# --- MOCK SETUP START ---
# 1. Mock Google Generative AI
# IMPORTANT: Do NOT overwrite 'google' module because other libraries (like protobuf) depend on it.
# We only inject 'google.generativeai' into sys.modules.

mock_genai = types.ModuleType("google.generativeai")

class MockGenerativeModel:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

    def generate_content(self, *args, **kwargs):
        pass
    
    async def generate_content_async(self, *args, **kwargs):
        pass

mock_genai.GenerativeModel = MockGenerativeModel
sys.modules["google.generativeai"] = mock_genai

# 2. Mock OTel GRPC Instrumentation
mock_otel_instr = types.ModuleType("opentelemetry.instrumentation")
mock_otel_grpc = types.ModuleType("opentelemetry.instrumentation.grpc")

class MockGrpcInstrumentorClient:
    def instrument(self):
        pass

mock_otel_grpc.GrpcInstrumentorClient = MockGrpcInstrumentorClient
mock_otel_instr.grpc = mock_otel_grpc

sys.modules["opentelemetry.instrumentation"] = mock_otel_instr
sys.modules["opentelemetry.instrumentation.grpc"] = mock_otel_grpc
# --- MOCK SETUP END ---

# --- OTEL SETUP (Module level - shared across all tests) ---
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_processor = SimpleSpanProcessor(_exporter)
_provider.add_span_processor(_processor)
trace._set_tracer_provider(_provider, log=False)
# --- OTEL SETUP END ---

from agentbasis.llms.gemini.chat import instrument_chat, instrument_async_chat


class TestGeminiChat(unittest.TestCase):
    """Tests for synchronous Gemini Chat instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the mock before each test
        self.mock_generate_content = MagicMock()
        MockGenerativeModel.generate_content = self.mock_generate_content
        
    def test_basic_chat_flow(self):
        """Test basic text generation and response."""
        instrument_chat(mock_genai)

        # Setup a fake response
        mock_response = MagicMock()
        mock_response.text = "Gemini Response"
        mock_response.candidates = []
        mock_response.usage_metadata = None 
        self.mock_generate_content.return_value = mock_response

        # Create a model and call generate_content
        model = MockGenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="Hello, Gemini!")

        # Verify the span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("gemini-2.5-flash", span.name)
        self.assertEqual(span.attributes.get("llm.system"), "gemini")
        self.assertEqual(span.attributes.get("llm.response.content"), "Gemini Response")

    def test_tools_function_calls(self):
        """Test that tool function calls are traced correctly."""
        instrument_chat(mock_genai)

        # Setup a fake response with tool calls
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = None

        # Mock the candidate structure for tool calls
        mock_func_call = MagicMock()
        mock_func_call.name = "get_weather"
        mock_func_call.arguments = {"location": "New York"}

        mock_candidate = MagicMock()
        mock_candidate.function_calls = [mock_func_call]
        mock_response.candidates = [mock_candidate]

        self.mock_generate_content.return_value = mock_response

        # Call with tools
        tools_def = [{"name": "get_weather", "description": "Get weather info"}]
        model = MockGenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="What's the weather?", tools=tools_def)

        # Verify the spans
        spans = _exporter.get_finished_spans()
        span = spans[0]

        # Check request tools (input)
        self.assertIn("get_weather", span.attributes.get("llm.request.tools"))
        self.assertEqual(span.attributes.get("llm.request.tool_count"), 1)

        # Check tool call details (output)
        self.assertIn("get_weather", span.attributes.get("llm.response.function_calls"))
        self.assertIn("New York", span.attributes.get("llm.response.function_calls"))
        self.assertEqual(span.attributes.get("llm.response.function_call_count"), 1)

    def test_token_usage_tracking(self):
        """Test that token usage is tracked correctly."""
        instrument_chat(mock_genai)

        mock_response = MagicMock()
        mock_response.text = "Token usage response"
        mock_response.candidates = []

        # Mock Usage Metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 20
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 30
        mock_response.usage_metadata = mock_usage

        self.mock_generate_content.return_value = mock_response

        model = MockGenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="Track my tokens")

        spans = _exporter.get_finished_spans()
        span = spans[0]

        self.assertEqual(span.attributes.get("llm.usage.prompt_tokens"), 20)
        self.assertEqual(span.attributes.get("llm.usage.completion_tokens"), 10)
        self.assertEqual(span.attributes.get("llm.usage.total_tokens"), 30)

    def test_sync_error_handling(self):
        """Test that errors in sync calls are properly recorded."""
        instrument_chat(mock_genai)
        
        # Setup the mock to raise an exception
        self.mock_generate_content.side_effect = Exception("Gemini API Error")
        
        model = MockGenerativeModel("gemini-2.5-flash")
        
        with self.assertRaises(Exception) as context:
            model.generate_content(contents="Hello")
        
        self.assertEqual(str(context.exception), "Gemini API Error")
        
        # Verify Span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)


class TestGeminiAsyncChat(unittest.TestCase):
    """Tests for async Gemini Chat instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the async mock before each test
        self.mock_async_generate_content = AsyncMock()
        MockGenerativeModel.generate_content_async = self.mock_async_generate_content
        
    def test_async_basic_chat_flow(self):
        """Test async text generation and response."""
        instrument_async_chat(mock_genai)

        # Setup a fake response
        mock_response = MagicMock()
        mock_response.text = "Async Gemini Response"
        mock_response.candidates = []
        mock_response.usage_metadata = None 
        self.mock_async_generate_content.return_value = mock_response

        # Run async test
        async def run_async_test():
            model = MockGenerativeModel("gemini-2.5-flash")
            return await model.generate_content_async(contents="Hello async Gemini!")
        
        response = asyncio.run(run_async_test())

        # Verify the span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("gemini-2.5-flash", span.name)
        self.assertEqual(span.attributes.get("llm.system"), "gemini")
        self.assertEqual(span.attributes.get("llm.request.async"), True)
        self.assertEqual(span.attributes.get("llm.response.content"), "Async Gemini Response")

    def test_async_token_usage_tracking(self):
        """Test that token usage is tracked correctly in async calls."""
        instrument_async_chat(mock_genai)

        mock_response = MagicMock()
        mock_response.text = "Async token response"
        mock_response.candidates = []

        # Mock Usage Metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 25
        mock_usage.candidates_token_count = 15
        mock_usage.total_token_count = 40
        mock_response.usage_metadata = mock_usage

        self.mock_async_generate_content.return_value = mock_response

        async def run_async_test():
            model = MockGenerativeModel("gemini-2.5-flash")
            return await model.generate_content_async(contents="Track async tokens")
        
        asyncio.run(run_async_test())

        spans = _exporter.get_finished_spans()
        span = spans[0]

        self.assertEqual(span.attributes.get("llm.usage.prompt_tokens"), 25)
        self.assertEqual(span.attributes.get("llm.usage.completion_tokens"), 15)
        self.assertEqual(span.attributes.get("llm.usage.total_tokens"), 40)

    def test_async_error_handling(self):
        """Test that errors in async calls are properly recorded."""
        instrument_async_chat(mock_genai)
        
        # Setup the mock to raise an exception
        self.mock_async_generate_content.side_effect = Exception("Async Gemini API Error")
        
        async def run_async_error_test():
            model = MockGenerativeModel("gemini-2.5-flash")
            await model.generate_content_async(contents="Hello")
        
        with self.assertRaises(Exception) as context:
            asyncio.run(run_async_error_test())
        
        self.assertEqual(str(context.exception), "Async Gemini API Error")
        
        # Verify Span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        
        # Check that we recorded an exception event
        exception_events = [e for e in span.events if e.name == "exception"]
        self.assertGreaterEqual(len(exception_events), 1)


if __name__ == "__main__":
    unittest.main()
