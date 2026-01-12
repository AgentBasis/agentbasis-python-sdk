# IMPORTANT NOTE: This test is deprecated and will be removed in the future.
# We are using the OpenTelemetry SDK to send data to the AgentBasis backend.
# The OpenTelemetry SDK is a more robust and feature-rich SDK that is more suited for this purpose.
# This test was a part of transport.py

import unittest
from unittest.mock import patch, MagicMock
import time
from agentbasis.transport import Transport
from agentbasis.config import Config

class TestTransport(unittest.TestCase):
    
    def setUp(self):
        self.config = Config(api_key="test-key", api_url="http://test-url")
        # Use a small batch size and fast interval for testing
        self.transport = Transport(self.config, batch_size=2, flush_interval=0.1)
        self.transport.start()

    def tearDown(self):
        self.transport.stop()

    @patch("requests.post")
    def test_send_batch(self, mock_post):
        """Test that items are batched and sent."""
        # Configure the mock to return a fake 200 OK response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Send 2 items (matching our batch_size)
        self.transport.send({"event": 1})
        self.transport.send({"event": 2})

        # Wait a moment for the thread to wake up and process
        time.sleep(0.3)

        # Verify requests.post was called
        self.assertTrue(mock_post.called)
        
        # Verify arguments (URL, headers, data)
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://test-url/v1/telemetry")
        self.assertEqual(len(kwargs["json"]), 2)
        self.assertEqual(kwargs["json"][0], {"event": 1})
        self.assertEqual(kwargs["json"][1], {"event": 2})
        
    @patch("requests.post")
    def test_flush_on_interval(self, mock_post):
        """Test that items are sent even if batch isn't full, after time limit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Send just 1 item (less than batch_size of 2)
        self.transport.send({"event": "lonely"})

        # Wait longer than flush_interval (0.1s)
        time.sleep(0.3)

        # Should have sent anyway
        self.assertTrue(mock_post.called)
        args, kwargs = mock_post.call_args
        self.assertEqual(len(kwargs["json"]), 1)

if __name__ == "__main__":
    unittest.main()

