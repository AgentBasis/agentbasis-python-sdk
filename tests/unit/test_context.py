import unittest
import asyncio
from agentbasis.context import (
    AgentBasisContext,
    context,
    set_user,
    set_session,
    set_conversation,
    set_metadata,
    get_user,
    get_session,
    get_conversation,
    get_metadata,
    get_context_attributes,
    with_context,
    inject_context_to_span,
)
from unittest.mock import MagicMock


class TestContextManager(unittest.TestCase):
    """Tests for the AgentBasisContext context manager."""

    def setUp(self):
        # Clear any existing context
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)

    def test_context_manager_sets_user_id(self):
        """Test that context manager sets user_id."""
        self.assertIsNone(get_user())
        
        with AgentBasisContext(user_id="user-123"):
            self.assertEqual(get_user(), "user-123")
        
        # After exiting, should be reset
        self.assertIsNone(get_user())

    def test_context_manager_sets_session_id(self):
        """Test that context manager sets session_id."""
        self.assertIsNone(get_session())
        
        with AgentBasisContext(session_id="sess-456"):
            self.assertEqual(get_session(), "sess-456")
        
        self.assertIsNone(get_session())

    def test_context_manager_sets_conversation_id(self):
        """Test that context manager sets conversation_id."""
        self.assertIsNone(get_conversation())
        
        with AgentBasisContext(conversation_id="conv-789"):
            self.assertEqual(get_conversation(), "conv-789")
        
        self.assertIsNone(get_conversation())

    def test_context_manager_sets_metadata(self):
        """Test that context manager sets metadata."""
        self.assertIsNone(get_metadata())
        
        metadata = {"plan": "pro", "version": "1.0"}
        with AgentBasisContext(metadata=metadata):
            self.assertEqual(get_metadata(), metadata)
        
        self.assertIsNone(get_metadata())

    def test_context_manager_sets_all_fields(self):
        """Test that context manager sets all fields at once."""
        with AgentBasisContext(
            user_id="user-123",
            session_id="sess-456",
            conversation_id="conv-789",
            metadata={"key": "value"}
        ):
            self.assertEqual(get_user(), "user-123")
            self.assertEqual(get_session(), "sess-456")
            self.assertEqual(get_conversation(), "conv-789")
            self.assertEqual(get_metadata(), {"key": "value"})

    def test_nested_context_managers(self):
        """Test that nested context managers work correctly."""
        with AgentBasisContext(user_id="outer-user"):
            self.assertEqual(get_user(), "outer-user")
            
            with AgentBasisContext(user_id="inner-user"):
                self.assertEqual(get_user(), "inner-user")
            
            # After inner exits, should restore outer
            self.assertEqual(get_user(), "outer-user")
        
        self.assertIsNone(get_user())

    def test_context_function_alias(self):
        """Test the context() convenience function."""
        with context(user_id="user-via-function"):
            self.assertEqual(get_user(), "user-via-function")
        
        self.assertIsNone(get_user())


class TestGlobalSetters(unittest.TestCase):
    """Tests for the global setter functions."""

    def setUp(self):
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)

    def test_set_user(self):
        """Test set_user() function."""
        set_user("global-user")
        self.assertEqual(get_user(), "global-user")
        
        set_user(None)
        self.assertIsNone(get_user())

    def test_set_session(self):
        """Test set_session() function."""
        set_session("global-session")
        self.assertEqual(get_session(), "global-session")

    def test_set_conversation(self):
        """Test set_conversation() function."""
        set_conversation("global-conv")
        self.assertEqual(get_conversation(), "global-conv")

    def test_set_metadata(self):
        """Test set_metadata() function."""
        set_metadata({"global": True})
        self.assertEqual(get_metadata(), {"global": True})


class TestGetContextAttributes(unittest.TestCase):
    """Tests for get_context_attributes()."""

    def setUp(self):
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)

    def test_empty_context(self):
        """Test that empty context returns empty dict."""
        attrs = get_context_attributes()
        self.assertEqual(attrs, {})

    def test_partial_context(self):
        """Test with only some fields set."""
        set_user("user-123")
        
        attrs = get_context_attributes()
        self.assertEqual(attrs, {"agentbasis.user.id": "user-123"})

    def test_full_context(self):
        """Test with all fields set."""
        set_user("user-123")
        set_session("sess-456")
        set_conversation("conv-789")
        set_metadata({"key": "value"})
        
        attrs = get_context_attributes()
        
        self.assertEqual(attrs["agentbasis.user.id"], "user-123")
        self.assertEqual(attrs["agentbasis.session.id"], "sess-456")
        self.assertEqual(attrs["agentbasis.conversation.id"], "conv-789")
        self.assertEqual(attrs["agentbasis.metadata"], '{"key": "value"}')


class TestInjectContextToSpan(unittest.TestCase):
    """Tests for inject_context_to_span()."""

    def setUp(self):
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)

    def test_inject_empty_context(self):
        """Test injecting empty context does nothing."""
        mock_span = MagicMock()
        
        inject_context_to_span(mock_span)
        
        mock_span.set_attribute.assert_not_called()

    def test_inject_user_context(self):
        """Test injecting user context."""
        mock_span = MagicMock()
        set_user("user-123")
        
        inject_context_to_span(mock_span)
        
        mock_span.set_attribute.assert_called_once_with(
            "agentbasis.user.id", "user-123"
        )

    def test_inject_full_context(self):
        """Test injecting full context."""
        mock_span = MagicMock()
        set_user("user-123")
        set_session("sess-456")
        set_conversation("conv-789")
        
        inject_context_to_span(mock_span)
        
        # Check all attributes were set
        calls = mock_span.set_attribute.call_args_list
        self.assertEqual(len(calls), 3)


class TestWithContextDecorator(unittest.TestCase):
    """Tests for the @with_context decorator."""

    def setUp(self):
        set_user(None)
        set_session(None)

    def test_decorator_on_sync_function(self):
        """Test decorator works on sync functions."""
        captured_user = None
        
        @with_context(user_id="decorator-user")
        def my_function():
            nonlocal captured_user
            captured_user = get_user()
            return "result"
        
        result = my_function()
        
        self.assertEqual(result, "result")
        self.assertEqual(captured_user, "decorator-user")
        # After function exits, context should be cleared
        self.assertIsNone(get_user())

    def test_decorator_on_async_function(self):
        """Test decorator works on async functions."""
        captured_user = None
        
        @with_context(user_id="async-decorator-user")
        async def my_async_function():
            nonlocal captured_user
            captured_user = get_user()
            return "async-result"
        
        result = asyncio.run(my_async_function())
        
        self.assertEqual(result, "async-result")
        self.assertEqual(captured_user, "async-decorator-user")
        self.assertIsNone(get_user())

    def test_decorator_with_all_fields(self):
        """Test decorator with all context fields."""
        captured_context = {}
        
        @with_context(
            user_id="dec-user",
            session_id="dec-session",
            conversation_id="dec-conv",
            metadata={"decorated": True}
        )
        def my_function():
            captured_context['user'] = get_user()
            captured_context['session'] = get_session()
            captured_context['conversation'] = get_conversation()
            captured_context['metadata'] = get_metadata()
        
        my_function()
        
        self.assertEqual(captured_context['user'], "dec-user")
        self.assertEqual(captured_context['session'], "dec-session")
        self.assertEqual(captured_context['conversation'], "dec-conv")
        self.assertEqual(captured_context['metadata'], {"decorated": True})


if __name__ == "__main__":
    unittest.main()
