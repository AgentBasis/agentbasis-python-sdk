import unittest
from unittest.mock import MagicMock
import sys
import types
import json

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

from agentbay.llms.gemini import instrument_chat

class TestGeminiChat(unittest.TestCase):
    
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
        self.mock_generate_content = MagicMock()
        # Update the mock on the class method
        sys.modules["google.generativeai"].GenerativeModel.generate_content = self.mock_generate_content
        
        # Reset the gRPC mock
        self.mock_grpc_instrument = MagicMock()
        sys.modules["opentelemetry.instrumentation.grpc"].GrpcInstrumentorClient.instrument = self.mock_grpc_instrument
        
    def test_instrumentation_wraps_generate_content(self):
        """Test that GRPC triggers our wrapper and OTel span."""
        instrument_chat(mock_genai)
        self.mock_grpc_instrument.assert_called_once()

    def test_basic_chat_flow(self):
        '''Test basic text generation and response'''
        instrument_chat(mock_genai)

        # Setup a fake response
        mock_response = MagicMock()
        mock_response.text = "Gemini Response"
        mock_response.candidates = []
        # Explicitly set usage_metadata to None to prevent MagicMock from creating it
        mock_response.usage_metadata = None 
        self.mock_generate_content.return_value = mock_response

        # Create a model and call generate_content
        model = sys.modules["google.generativeai"].GenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="Hello, Gemini!")

        # Verify the response
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes.get("llm.response.content"), "Gemini Response")

    def test_tools_function_calls(self):
        '''Test that tool function calls are traced correctly'''
        instrument_chat(mock_genai)

        # Setup a fake response with tool calls
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = None # Explicitly set to None

        # Mock the candidate structure for tool calls
        mock_func_call = MagicMock()
        mock_func_call.name = "get_weather"
        mock_func_call.arguments = {"location": "New York"}

        mock_candidate = MagicMock()
        # Gemini uses 'function_calls', not 'tool_calls'
        mock_candidate.function_calls = [mock_func_call]
        mock_response.candidates = [mock_candidate]

        self.mock_generate_content.return_value = mock_response

        # Call with tools
        tools_def = [{"name": "get_weather", "description": "Get weather info"}]
        model = sys.modules["google.generativeai"].GenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="What's the weather?", tools=tools_def)

        # Verify the spans
        spans = self.exporter.get_finished_spans()
        span = spans[0]

        # Check request tools (input)
        self.assertIn("get_weather", span.attributes.get("llm.request.tools"))
        self.assertEqual(span.attributes.get("llm.request.tool_count"), 1)

        # Check tool call details (output)
        self.assertIn("get_weather", span.attributes.get("llm.response.function_calls"))
        self.assertIn("New York", span.attributes.get("llm.response.function_calls"))
        self.assertEqual(span.attributes.get("llm.response.function_call_count"), 1)

    def test_token_usage_tracking(self):
        '''Test that token usage is tracked correctly'''
        instrument_chat(mock_genai)

        mock_response = MagicMock()
        mock_response.text = "Token usage response"
        mock_response.candidates = []

        # Mock Usage Metadata (Gemini uses usage_metadata, not usage.metadata)
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 20
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 30
        mock_response.usage_metadata = mock_usage

        self.mock_generate_content.return_value = mock_response

        model = sys.modules["google.generativeai"].GenerativeModel("gemini-2.5-flash")
        model.generate_content(contents="Track my tokens")

        spans = self.exporter.get_finished_spans()
        span = spans[0]

        self.assertEqual(span.attributes.get("llm.usage.prompt_tokens"), 20)
        self.assertEqual(span.attributes.get("llm.usage.completion_tokens"), 10)
        self.assertEqual(span.attributes.get("llm.usage.total_tokens"), 30)

if __name__ == "__main__":
    unittest.main()