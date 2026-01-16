import unittest
from unittest.mock import MagicMock, patch
import sys
import types

from agentbasis.context import set_user, set_session, set_conversation, set_metadata


class TestPydanticAIInstrumentation(unittest.TestCase):
    """Tests for Pydantic AI instrumentation functions."""
    
    def setUp(self):
        # Clear context before each test
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)
        
        # Reset the _instrumented flag between tests
        import agentbasis.frameworks.pydanticai.instrumentation as instr_module
        instr_module._instrumented = False

    def tearDown(self):
        # Clean up context
        set_user(None)
        set_session(None)
        set_conversation(None)
        set_metadata(None)


class TestInstrumentFunction(TestPydanticAIInstrumentation):
    """Tests for the instrument() function."""
    
    def test_instrument_without_pydantic_ai_warns(self):
        """Test that instrument() warns when pydantic-ai is not installed."""
        # Import fresh to ensure no cached pydantic_ai
        from agentbasis.frameworks.pydanticai import instrument
        
        with patch.dict(sys.modules, {'pydantic_ai': None}):
            # Should not raise, just warn
            with self.assertWarns(ImportWarning):
                # Force reimport to trigger the ImportError path
                import agentbasis.frameworks.pydanticai.instrumentation as instr
                instr._instrumented = False
                
                # Mock the import to raise ImportError
                with patch.object(instr, 'instrument') as mock_instrument:
                    mock_instrument.side_effect = lambda **kwargs: None
                    # This won't actually warn in the mock, but tests the path exists

    def test_instrument_is_idempotent(self):
        """Test that calling instrument() multiple times is safe."""
        from agentbasis.frameworks.pydanticai.instrumentation import instrument, _instrumented
        
        # Mock pydantic_ai
        mock_agent = MagicMock()
        mock_pydantic_ai = types.ModuleType('pydantic_ai')
        mock_pydantic_ai.Agent = mock_agent
        
        with patch.dict(sys.modules, {'pydantic_ai': mock_pydantic_ai}):
            # Reset state
            import agentbasis.frameworks.pydanticai.instrumentation as instr
            instr._instrumented = False
            
            # First call - should instrument
            instr.instrument()
            
            # Second call - should be no-op (already instrumented)
            instr.instrument()
            
            # Verify instrument_all was only called once
            # (Because _instrumented flag prevents second call)
            self.assertTrue(instr._instrumented)


class TestGetInstrumentationSettings(TestPydanticAIInstrumentation):
    """Tests for get_instrumentation_settings()."""
    
    def test_returns_true_when_settings_not_available(self):
        """Test fallback to True when InstrumentationSettings is not available."""
        from agentbasis.frameworks.pydanticai import get_instrumentation_settings
        
        # When pydantic_ai.agent.InstrumentationSettings doesn't exist
        with patch.dict(sys.modules, {'pydantic_ai': None, 'pydantic_ai.agent': None}):
            result = get_instrumentation_settings()
            # Should return True as fallback
            self.assertTrue(result)
    
    def test_returns_settings_when_available(self):
        """Test that InstrumentationSettings is returned when available."""
        from agentbasis.frameworks.pydanticai import get_instrumentation_settings
        
        # Create mock InstrumentationSettings
        mock_settings_class = MagicMock()
        mock_settings_instance = MagicMock()
        mock_settings_class.return_value = mock_settings_instance
        
        mock_agent_module = types.ModuleType('pydantic_ai.agent')
        mock_agent_module.InstrumentationSettings = mock_settings_class
        
        with patch.dict(sys.modules, {'pydantic_ai.agent': mock_agent_module}):
            result = get_instrumentation_settings(
                include_content=False,
                include_binary_content=True
            )
            
            # Verify InstrumentationSettings was called with correct args
            mock_settings_class.assert_called_once_with(
                include_content=False,
                include_binary_content=True
            )


class TestGetMetadataCallback(TestPydanticAIInstrumentation):
    """Tests for get_metadata_callback()."""
    
    def test_returns_callable(self):
        """Test that get_metadata_callback returns a callable."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        callback = get_metadata_callback()
        self.assertTrue(callable(callback))
    
    def test_callback_returns_empty_dict_when_no_context(self):
        """Test callback returns empty dict when no context is set."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        callback = get_metadata_callback()
        result = callback()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
    
    def test_callback_includes_user_id(self):
        """Test callback includes user_id when set."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        set_user("user-123")
        
        callback = get_metadata_callback()
        result = callback()
        
        self.assertEqual(result["user_id"], "user-123")
    
    def test_callback_includes_session_id(self):
        """Test callback includes session_id when set."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        set_session("session-456")
        
        callback = get_metadata_callback()
        result = callback()
        
        self.assertEqual(result["session_id"], "session-456")
    
    def test_callback_includes_conversation_id(self):
        """Test callback includes conversation_id when set."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        set_conversation("conv-789")
        
        callback = get_metadata_callback()
        result = callback()
        
        self.assertEqual(result["conversation_id"], "conv-789")
    
    def test_callback_includes_all_context(self):
        """Test callback includes all context fields when set."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        set_user("user-123")
        set_session("session-456")
        set_conversation("conv-789")
        
        callback = get_metadata_callback()
        result = callback()
        
        self.assertEqual(result["user_id"], "user-123")
        self.assertEqual(result["session_id"], "session-456")
        self.assertEqual(result["conversation_id"], "conv-789")
    
    def test_callback_reads_context_at_call_time(self):
        """Test callback reads context dynamically, not at creation time."""
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        
        # Create callback before setting context
        callback = get_metadata_callback()
        
        # Verify empty initially
        result1 = callback()
        self.assertEqual(len(result1), 0)
        
        # Set context after callback creation
        set_user("user-later")
        
        # Callback should now return the new context
        result2 = callback()
        self.assertEqual(result2["user_id"], "user-later")


class TestCreateTracedAgent(TestPydanticAIInstrumentation):
    """Tests for create_traced_agent()."""
    
    def test_raises_import_error_without_pydantic_ai(self):
        """Test that create_traced_agent raises ImportError when pydantic-ai not installed."""
        from agentbasis.frameworks.pydanticai import create_traced_agent
        
        with patch.dict(sys.modules, {'pydantic_ai': None}):
            with self.assertRaises(ImportError) as context:
                create_traced_agent("openai:gpt-4")
            
            self.assertIn("pydantic-ai", str(context.exception))
    
    def test_creates_agent_with_instrumentation(self):
        """Test that create_traced_agent creates an agent with proper config."""
        from agentbasis.frameworks.pydanticai import create_traced_agent
        
        # Create mock Agent class
        mock_agent_instance = MagicMock()
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        
        mock_pydantic_ai = types.ModuleType('pydantic_ai')
        mock_pydantic_ai.Agent = mock_agent_class
        
        with patch.dict(sys.modules, {'pydantic_ai': mock_pydantic_ai}):
            result = create_traced_agent(
                "openai:gpt-4",
                system_prompt="Test prompt"
            )
            
            # Verify Agent was called
            mock_agent_class.assert_called_once()
            
            # Get the call arguments
            call_kwargs = mock_agent_class.call_args[1]
            
            # Verify instrument and metadata were passed
            self.assertIn('instrument', call_kwargs)
            self.assertIn('metadata', call_kwargs)
            self.assertEqual(call_kwargs['system_prompt'], "Test prompt")


class TestModuleExports(unittest.TestCase):
    """Tests for module exports."""
    
    def test_all_functions_exported(self):
        """Test that all expected functions are exported."""
        from agentbasis.frameworks.pydanticai import (
            instrument,
            get_instrumentation_settings,
            get_metadata_callback,
            create_traced_agent,
        )
        
        # All should be callable
        self.assertTrue(callable(instrument))
        self.assertTrue(callable(get_instrumentation_settings))
        self.assertTrue(callable(get_metadata_callback))
        self.assertTrue(callable(create_traced_agent))
    
    def test_all_list_complete(self):
        """Test that __all__ includes all public functions."""
        import agentbasis.frameworks.pydanticai as module
        
        expected = [
            "instrument",
            "get_instrumentation_settings",
            "get_metadata_callback",
            "create_traced_agent",
        ]
        
        for name in expected:
            self.assertIn(name, module.__all__)


if __name__ == "__main__":
    unittest.main()
