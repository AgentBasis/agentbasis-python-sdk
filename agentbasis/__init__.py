from typing import Optional
from .client import AgentBasis
from .decorators import trace
from .context import (
    context,
    set_user,
    set_session,
    set_conversation,
    set_metadata,
    with_context,
    AgentBasisContext,
)


def init(api_key: Optional[str] = None, agent_id: Optional[str] = None) -> AgentBasis:
    """
    Initialize the AgentBasis SDK.
    
    Args:
        api_key: Your AgentBasis API Key. If not provided, reads from AGENTBASIS_API_KEY env var.
        agent_id: The ID of the agent to track. If not provided, reads from AGENTBASIS_AGENT_ID env var.
    Returns:
        The initialized AgentBasis client instance.
    """
    return AgentBasis.initialize(api_key=api_key, agent_id=agent_id)


def flush(timeout_millis: int = 30000) -> bool:
    """
    Force flush all pending telemetry data.
    
    This is useful when you want to ensure all traces are sent before
    a critical operation, at specific checkpoints, or before exiting.
    
    Note: The SDK automatically flushes on normal Python exit via atexit.
    
    Args:
        timeout_millis: Maximum time to wait for flush (default 30 seconds).
        
    Returns:
        True if flush completed successfully, False if timed out or not initialized.
        
    Example:
        >>> agentbasis.init(api_key="...", agent_id="...")
        >>> # ... your agent code ...
        >>> agentbasis.flush()  # Ensure all data is sent
    """
    try:
        client = AgentBasis.get_instance()
        return client.flush(timeout_millis)
    except RuntimeError:
        # SDK not initialized
        return False


def shutdown():
    """
    Manually shut down the SDK and flush all pending data.
    
    This is automatically called on Python exit, but can be called
    manually if you need to shut down the SDK before the process ends.
    
    This method is idempotent - calling it multiple times is safe.
    """
    try:
        client = AgentBasis.get_instance()
        client.shutdown()
    except RuntimeError:
        # SDK not initialized
        pass


__all__ = [
    "init",
    "AgentBasis",
    "trace",
    "flush",
    "shutdown",
    # Context management
    "context",
    "set_user",
    "set_session",
    "set_conversation",
    "set_metadata",
    "with_context",
    "AgentBasisContext",
]
