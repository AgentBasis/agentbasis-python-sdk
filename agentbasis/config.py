import os
from typing import Optional

class Config:
    """
    Configuration settings for the AgentBasis SDK.
    Handles API keys and Agent IDs.
    """
    def __init__(self, api_key: Optional[str] = None, agent_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AGENTBASIS_API_KEY")
        self.agent_id = agent_id or os.environ.get("AGENTBASIS_AGENT_ID")
        
        # Backend endpoint - not user-configurable, but can be overridden via env var for testing
        self.api_url = os.environ.get("AGENTBASIS_API_URL", "https://api.agentbasis.co")

    def validate(self):
        """
        Checks if the configuration is valid (i.e., has an API key and Agent ID).
        Raises a ValueError if the key is missing.
        """
        if not self.api_key:
            raise ValueError(
                "AgentBasis API Key is missing. "
                "Please provide it via `agentbasis.init(api_key='...')` "
                "or set the `AGENTBASIS_API_KEY` environment variable."
            )
            
        if not self.agent_id:
            raise ValueError(
                "AgentBasis Agent ID is missing. "
                "Please provide it via `agentbasis.init(agent_id='...')` "
                "or set the `AGENTBASIS_AGENT_ID` environment variable."
            )
