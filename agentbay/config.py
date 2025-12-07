import os
from typing import Optional

class Config:
    """
    Configuration settings for the AgentBay SDK.
    Handles API keys and endpoint URLs.
    """
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, agent_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AGENTBAY_API_KEY") # Get API Key from the arguments first, if not provided, try to get it from the environment variables.
        self.agent_id = agent_id or os.environ.get("AGENTBAY_AGENT_ID") # Get Agent ID from the arguments first, if not provided, try to get it from the environment variables.
        self.api_url = api_url or os.environ.get("AGENTBAY_API_URL", "https://api.agentbay.co") # Set the API URL (defaulting to the hosted version if not changed, https://api.agentbay.co is the hosted version)

    def validate(self):
        """
        Checks if the configuration is valid (i.e., has an API key and Agent ID).
        Raises a ValueError if the key is missing.
        """
        if not self.api_key:
            raise ValueError(
                "AgentBay API Key is missing. "
                "Please provide it via `agentbay.init(api_key='...')` "
                "or set the `AGENTBAY_API_KEY` environment variable."
            )
            
        if not self.agent_id:
            raise ValueError(
                "AgentBay Agent ID is missing. "
                "Please provide it via `agentbay.init(agent_id='...')` "
                "or set the `AGENTBAY_AGENT_ID` environment variable."
            )
