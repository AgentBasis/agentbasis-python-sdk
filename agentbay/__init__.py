from typing import Optional
from .client import AgentBasis
from .decorators import trace

def init(api_key: Optional[str] = None, api_url: Optional[str] = None, agent_id: Optional[str] = None) -> AgentBasis:
    """
    Initialize the AgentBasis SDK.
    
    Args:
        api_key: Your AgentBasis API Key. If not provided, reads from AGENTBASIS_API_KEY env var.
        api_url: Optional URL for the AgentBasis backend (mostly for testing/on-prem).
        agent_id: The ID of the agent to track. If not provided, reads from AGENTBASIS_AGENT_ID env var.
    Returns:
        The initialized AgentBasis client instance.
    """
    return AgentBasis.initialize(api_key=api_key, api_url=api_url, agent_id=agent_id)

__all__ = ["init", "AgentBasis", "trace"]
