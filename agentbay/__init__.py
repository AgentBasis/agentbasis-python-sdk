from typing import Optional
from .client import AgentBay
from .decorators import trace

def init(api_key: Optional[str] = None, api_url: Optional[str] = None, agent_id: Optional[str] = None) -> AgentBay:
    """
    Initialize the AgentBay SDK.
    
    Args:
        api_key: Your AgentBay API Key. If not provided, reads from AGENTBAY_API_KEY env var.
        api_url: Optional URL for the AgentBay backend (mostly for testing/on-prem).
        agent_id: The ID of the agent to track. If not provided, reads from AGENTBAY_AGENT_ID env var.
    Returns:
        The initialized AgentBay client instance.
    """
    return AgentBay.initialize(api_key=api_key, api_url=api_url, agent_id=agent_id)

__all__ = ["init", "AgentBay", "trace"]
