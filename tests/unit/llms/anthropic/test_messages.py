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
# This simulates 'anthropic' being installed on the system

mock_anthropic = types.ModuleType("anthropic")
mock_resources = types.ModuleType("anthropic.resources")
mock_messages_module = types.ModuleType("anthropic.resources.messages")
mock_streaming_module = types.ModuleType("anthropic.lib.streaming")


# Mock streaming classes
class MockMessageStreamManager:
    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def __iter__(self):
        return iter(self._events)
    
    @property
    def text_stream(self):
        for event in self._events:
            if event.type == 'content_block_delta' and hasattr(event, 'delta'):
                yield event.delta.text
    
    def get_final_message(self):
        return self._final_message
    
    def get_final_text(self):
        parts = []
        for block in self._final_message.content:
            if hasattr(block, 'text'):
                parts.append(block.text)
        return "".join(parts)


class MockAsyncMessageStreamManager:
    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    
    async def __aiter__(self):
        for event in self._events:
            yield event
    
    @property
    def text_stream(self):
        async def gen():
            for event in self._events:
                if event.type == 'content_block_delta' and hasattr(event, 'delta'):
                    yield event.delta.text
        return gen()
    
    async def get_final_message(self):
        return self._final_message
    
    async def get_final_text(self):
        parts = []
        for block in self._final_message.content:
            if hasattr(block, 'text'):
                parts.append(block.text)
        return "".join(parts)


# Create the sync Messages Class
class MockMessages:
    def create(self, *args, **kwargs):
        pass
    
    def stream(self, *args, **kwargs):
        pass


# Create the async AsyncMessages Class
class MockAsyncMessages:
    async def create(self, *args, **kwargs):
        pass
    
    async def stream(self, *args, **kwargs):
        pass


mock_messages_module.Messages = MockMessages
mock_messages_module.AsyncMessages = MockAsyncMessages
mock_streaming_module.MessageStreamManager = MockMessageStreamManager
mock_streaming_module.AsyncMessageStreamManager = MockAsyncMessageStreamManager

# Connect them
mock_anthropic.resources = mock_resources
mock_resources.messages = mock_messages_module

# Register them in sys.modules so 'from anthropic...' works
sys.modules["anthropic"] = mock_anthropic
sys.modules["anthropic.resources"] = mock_resources
sys.modules["anthropic.resources.messages"] = mock_messages_module
sys.modules["anthropic.lib"] = types.ModuleType("anthropic.lib")
sys.modules["anthropic.lib.streaming"] = mock_streaming_module
# --- MOCK SETUP END ---

# --- OTEL SETUP (Module level - shared across all tests) ---
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_processor = SimpleSpanProcessor(_exporter)
_provider.add_span_processor(_processor)
trace._set_tracer_provider(_provider, log=False)
# --- OTEL SETUP END ---

from agentbasis.llms.anthropic import instrument
from agentbasis.llms.anthropic.messages import instrument_messages, instrument_async_messages


class TestAnthropicMessages(unittest.TestCase):
    """Tests for synchronous Anthropic Messages instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the Messages.create to a fresh mock before each test
        self.mock_create = MagicMock()
        MockMessages.create = self.mock_create
        
    def test_instrumentation_wraps_create(self):
        """Test that calling create() triggers our wrapper and OTel span."""
        
        # 1. Run instrumentation
        instrument_messages(mock_anthropic)
        
        # 2. Setup a fake response matching Anthropic's structure
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "AI Response from Claude"
        mock_response.content = [mock_content_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        
        self.mock_create.return_value = mock_response
        
        # 3. Call the method (simulating user code)
        messages_instance = MockMessages()
        response = messages_instance.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        self.assertEqual(response, mock_response)
        
        # 4. Verify OTel Span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("claude-3-opus", span.name)
        self.assertEqual(span.attributes["llm.system"], "anthropic")
        self.assertEqual(span.attributes["llm.request.model"], "claude-3-opus-20240229")
        self.assertEqual(span.attributes["llm.response.content"], "AI Response from Claude")
        self.assertEqual(span.attributes["llm.usage.prompt_tokens"], 10)
        self.assertEqual(span.attributes["llm.usage.completion_tokens"], 5)
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 15)

    def test_sync_error_handling(self):
        """Test that errors in sync calls are properly recorded."""
        
        # 1. Run instrumentation
        instrument_messages(mock_anthropic)
        
        # 2. Setup the mock to raise an exception
        self.mock_create.side_effect = Exception("Sync API Error")
        
        # 3. Call the method and expect an error
        messages_instance = MockMessages()
        
        with self.assertRaises(Exception) as context:
            messages_instance.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
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


class TestAnthropicAsyncMessages(unittest.TestCase):
    """Tests for async Anthropic Messages instrumentation."""

    def setUp(self):
        _exporter.clear()
        # Reset the AsyncMessages.create to a fresh mock before each test
        self.mock_async_create = AsyncMock()
        MockAsyncMessages.create = self.mock_async_create
        
    def test_async_instrumentation_wraps_create(self):
        """Test that calling async create() triggers our wrapper and OTel span."""
        
        # 1. Run instrumentation
        instrument_async_messages(mock_anthropic)
        
        # 2. Setup a fake response
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = "Async AI Response from Claude"
        mock_response.content = [mock_content_block]
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_response.stop_reason = "end_turn"
        
        self.mock_async_create.return_value = mock_response
        
        # 3. Call the async method
        async def run_async_test():
            messages_instance = MockAsyncMessages()
            response = await messages_instance.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello async"}]
            )
            return response
        
        response = asyncio.run(run_async_test())
        
        self.assertEqual(response, mock_response)
        
        # 4. Verify OTel Span
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertIn("claude-3-sonnet", span.name)
        self.assertEqual(span.attributes["llm.system"], "anthropic")
        self.assertEqual(span.attributes["llm.request.model"], "claude-3-sonnet-20240229")
        self.assertEqual(span.attributes["llm.request.async"], True)
        self.assertEqual(span.attributes["llm.response.content"], "Async AI Response from Claude")
        self.assertEqual(span.attributes["llm.usage.total_tokens"], 30)

    def test_async_error_handling(self):
        """Test that errors in async calls are properly recorded."""
        
        # 1. Run instrumentation
        instrument_async_messages(mock_anthropic)
        
        # 2. Setup the mock to raise an exception
        self.mock_async_create.side_effect = Exception("Async API Error")
        
        # 3. Call the async method and expect an error
        async def run_async_error_test():
            messages_instance = MockAsyncMessages()
            await messages_instance.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        with self.assertRaises(Exception) as context:
            asyncio.run(run_async_error_test())
        
        self.assertEqual(str(context.exception), "Async API Error")
        
        # 4. Verify Span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        self.assertEqual(span.status.status_code, trace.StatusCode.ERROR)
        
        # Check that we recorded an exception event
        exception_events = [e for e in span.events if e.name == "exception"]
        self.assertGreaterEqual(len(exception_events), 1)


class TestAnthropicStreamingMessages(unittest.TestCase):
    """Tests for synchronous Anthropic streaming Messages instrumentation."""

    def setUp(self):
        _exporter.clear()
        self.mock_create = MagicMock()
        MockMessages.create = self.mock_create

    def _create_mock_event(self, event_type: str, text: str = None, 
                           input_tokens: int = None, output_tokens: int = None):
        """Helper to create mock streaming events."""
        event = MagicMock()
        event.type = event_type
        
        if event_type == 'content_block_delta' and text is not None:
            event.delta = MagicMock()
            event.delta.text = text
        
        if event_type == 'message_start' and input_tokens is not None:
            event.message = MagicMock()
            event.message.usage = MagicMock()
            event.message.usage.input_tokens = input_tokens
        
        if event_type == 'message_delta' and output_tokens is not None:
            event.usage = MagicMock()
            event.usage.output_tokens = output_tokens
        
        return event

    def test_sync_streaming_basic(self):
        """Test that sync streaming responses are traced correctly."""
        instrument_messages(mock_anthropic)

        # Create mock events (Anthropic's streaming format)
        events = [
            self._create_mock_event('message_start', input_tokens=10),
            self._create_mock_event('content_block_start'),
            self._create_mock_event('content_block_delta', text="Hello"),
            self._create_mock_event('content_block_delta', text=" "),
            self._create_mock_event('content_block_delta', text="World"),
            self._create_mock_event('content_block_delta', text="!"),
            self._create_mock_event('content_block_stop'),
            self._create_mock_event('message_delta', output_tokens=5),
            self._create_mock_event('message_stop'),
        ]
        
        # Return an iterator of events
        self.mock_create.return_value = iter(events)

        # Call with stream=True
        messages_instance = MockMessages()
        stream = messages_instance.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Consume the stream
        collected_content = []
        for event in stream:
            if event.type == 'content_block_delta' and hasattr(event, 'delta'):
                collected_content.append(event.delta.text)

        self.assertEqual("".join(collected_content), "Hello World!")

        # Verify span after stream is consumed
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertIn("claude-3-opus", span.name)
        self.assertEqual(span.attributes["llm.system"], "anthropic")
        self.assertEqual(span.attributes["llm.request.streaming"], True)
        self.assertEqual(span.attributes["llm.response.content"], "Hello World!")
        self.assertEqual(span.attributes["llm.usage.prompt_tokens"], 10)
        self.assertEqual(span.attributes["llm.usage.completion_tokens"], 5)
        self.assertIn("llm.response.first_token_ms", span.attributes)

    def test_sync_streaming_error(self):
        """Test that errors during streaming are recorded."""
        instrument_messages(mock_anthropic)

        # Create a generator that raises an error mid-stream
        def error_generator():
            yield self._create_mock_event('message_start', input_tokens=10)
            yield self._create_mock_event('content_block_delta', text="Hello")
            raise Exception("Stream Error")

        self.mock_create.return_value = error_generator()

        messages_instance = MockMessages()
        stream = messages_instance.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Consume the stream and expect an error
        with self.assertRaises(Exception) as context:
            for event in stream:
                pass

        self.assertEqual(str(context.exception), "Stream Error")

        # Verify span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)


class TestAnthropicAsyncStreamingMessages(unittest.TestCase):
    """Tests for async Anthropic streaming Messages instrumentation."""

    def setUp(self):
        _exporter.clear()
        self.mock_async_create = AsyncMock()
        MockAsyncMessages.create = self.mock_async_create

    def _create_mock_event(self, event_type: str, text: str = None,
                           input_tokens: int = None, output_tokens: int = None):
        """Helper to create mock streaming events."""
        event = MagicMock()
        event.type = event_type
        
        if event_type == 'content_block_delta' and text is not None:
            event.delta = MagicMock()
            event.delta.text = text
        
        if event_type == 'message_start' and input_tokens is not None:
            event.message = MagicMock()
            event.message.usage = MagicMock()
            event.message.usage.input_tokens = input_tokens
        
        if event_type == 'message_delta' and output_tokens is not None:
            event.usage = MagicMock()
            event.usage.output_tokens = output_tokens
        
        return event

    def test_async_streaming_basic(self):
        """Test that async streaming responses are traced correctly."""
        instrument_async_messages(mock_anthropic)

        # Create mock events
        events = [
            self._create_mock_event('message_start', input_tokens=15),
            self._create_mock_event('content_block_delta', text="Async"),
            self._create_mock_event('content_block_delta', text=" "),
            self._create_mock_event('content_block_delta', text="Stream"),
            self._create_mock_event('message_delta', output_tokens=8),
        ]

        # Create an async iterator
        async def async_event_iterator():
            for event in events:
                yield event

        self.mock_async_create.return_value = async_event_iterator()

        async def run_async_stream_test():
            messages_instance = MockAsyncMessages()
            stream = await messages_instance.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )

            collected_content = []
            async for event in stream:
                if event.type == 'content_block_delta' and hasattr(event, 'delta'):
                    collected_content.append(event.delta.text)
            return "".join(collected_content)

        result = asyncio.run(run_async_stream_test())
        self.assertEqual(result, "Async Stream")

        # Verify span after stream is consumed
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertIn("claude-3-sonnet", span.name)
        self.assertEqual(span.attributes["llm.request.async"], True)
        self.assertEqual(span.attributes["llm.request.streaming"], True)
        self.assertEqual(span.attributes["llm.response.content"], "Async Stream")
        self.assertEqual(span.attributes["llm.usage.prompt_tokens"], 15)
        self.assertEqual(span.attributes["llm.usage.completion_tokens"], 8)

    def test_async_streaming_error(self):
        """Test that errors during async streaming are recorded."""
        instrument_async_messages(mock_anthropic)

        # Create an async generator that raises an error
        async def error_async_generator():
            yield self._create_mock_event('message_start', input_tokens=10)
            yield self._create_mock_event('content_block_delta', text="Start")
            raise Exception("Async Stream Error")

        self.mock_async_create.return_value = error_async_generator()

        async def run_async_error_test():
            messages_instance = MockAsyncMessages()
            stream = await messages_instance.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )
            async for event in stream:
                pass

        with self.assertRaises(Exception) as context:
            asyncio.run(run_async_error_test())

        self.assertEqual(str(context.exception), "Async Stream Error")

        # Verify span recorded the error
        spans = _exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)


if __name__ == "__main__":
    unittest.main()
