# IMPORTANT NOTE: This session layer is deprecated and will be removed in the future.
# We are using the OpenTelemetry SDK to send data to the AgentBasis backend.
# The OpenTelemetry SDK is a more robust and feature-rich SDK that is more suited for this purpose.

import uuid
import time
from typing import Optional, Dict, Any

class Session:
    """
    Represents a single tracking session (e.g., a conversation or a task).
    """
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = "active"  # active, completed, failed

    def end(self, status: str = "completed"):
        """
        Ends the session.
        """
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the session to a dictionary for API transmission.
        """
        return {
            "id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status
        }
