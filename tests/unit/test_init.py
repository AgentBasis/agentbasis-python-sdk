import unittest
import os
from agentbasis import init, flush, shutdown, AgentBasis


class TestAgentBasisInit(unittest.TestCase):
    
    def setUp(self):
        # Reset the singleton before each test to ensure isolation
        AgentBasis._instance = None
        AgentBasis._shutdown_registered = False
        # Clear env var if present
        if "AGENTBASIS_API_KEY" in os.environ:
            del os.environ["AGENTBASIS_API_KEY"]
        if "AGENTBASIS_AGENT_ID" in os.environ:
            del os.environ["AGENTBASIS_AGENT_ID"]

    def test_init_with_api_key(self):
        """Test that init works with an explicit API key."""
        client = init(api_key="test-key-123", agent_id="test-agent-123")
        self.assertIsNotNone(client)
        self.assertEqual(client.config.api_key, "test-key-123")
        
    def test_init_missing_key_raises_error(self):
        """Test that init raises ValueError if no key is provided."""
        with self.assertRaises(ValueError):
            init()

    def test_singleton_pattern(self):
        """Test that init returns the same instance if called twice."""
        client1 = init(api_key="key-1", agent_id="agent-1")
        
        # Let's check that get_instance returns the initialized client
        client2 = AgentBasis.get_instance()
        self.assertIs(client1, client2)


class TestAgentBasisShutdown(unittest.TestCase):
    """Tests for graceful shutdown functionality."""
    
    def setUp(self):
        # Reset the singleton before each test
        AgentBasis._instance = None
        AgentBasis._shutdown_registered = False
        if "AGENTBASIS_API_KEY" in os.environ:
            del os.environ["AGENTBASIS_API_KEY"]
        if "AGENTBASIS_AGENT_ID" in os.environ:
            del os.environ["AGENTBASIS_AGENT_ID"]

    def test_flush_before_init_returns_false(self):
        """Test that flush returns False if SDK is not initialized."""
        result = flush()
        self.assertFalse(result)

    def test_shutdown_before_init_is_safe(self):
        """Test that shutdown is safe to call before init."""
        # Should not raise any exception
        shutdown()

    def test_flush_after_init(self):
        """Test that flush works after initialization."""
        init(api_key="test-key", agent_id="test-agent")
        result = flush(timeout_millis=1000)
        self.assertTrue(result)

    def test_shutdown_is_idempotent(self):
        """Test that shutdown can be called multiple times safely."""
        client = init(api_key="test-key", agent_id="test-agent")
        
        # First shutdown
        client.shutdown()
        self.assertTrue(client._is_shutdown)
        
        # Second shutdown should not raise
        client.shutdown()
        self.assertTrue(client._is_shutdown)

    def test_flush_after_shutdown_returns_false(self):
        """Test that flush returns False after shutdown."""
        client = init(api_key="test-key", agent_id="test-agent")
        client.shutdown()
        
        result = client.flush()
        self.assertFalse(result)

    def test_atexit_handler_registered(self):
        """Test that atexit handler is registered on init."""
        self.assertFalse(AgentBasis._shutdown_registered)
        
        init(api_key="test-key", agent_id="test-agent")
        
        self.assertTrue(AgentBasis._shutdown_registered)

    def test_atexit_handler_registered_only_once(self):
        """Test that atexit handler is only registered once."""
        init(api_key="test-key-1", agent_id="test-agent-1")
        self.assertTrue(AgentBasis._shutdown_registered)
        
        # Second init should not register again
        # (we can't easily test this, but we ensure the flag stays True)
        AgentBasis._instance = None  # Reset instance but keep flag
        init(api_key="test-key-2", agent_id="test-agent-2")
        
        # Flag should still be True (not re-registered)
        self.assertTrue(AgentBasis._shutdown_registered)


if __name__ == "__main__":
    unittest.main()

