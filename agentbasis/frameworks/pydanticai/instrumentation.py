"""
Pydantic AI instrumentation for AgentBasis.

This module provides integration with Pydantic AI's built-in OpenTelemetry
instrumentation, ensuring traces flow to AgentBasis.

Pydantic AI already emits OTel traces via its Logfire integration. Since
AgentBasis uses OTel as its backbone, we configure Pydantic AI to use
AgentBasis's tracer provider.
"""

from typing import Any, Callable, Dict, Optional

from agentbasis.context import get_context_attributes


# Placeholder for the instrumentation implementation
# Will be implemented in Step 2.2

def instrument():
    """
    Enable AgentBasis instrumentation for all Pydantic AI agents.
    
    This configures Pydantic AI to send traces to AgentBasis via OpenTelemetry.
    
    Example:
        import agentbasis
        from agentbasis.frameworks.pydanticai import instrument
        
        agentbasis.init(api_key="...", agent_id="...")
        instrument()
        
        # Now all Pydantic AI agents will be traced
        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4")
        result = agent.run_sync("Hello!")
    """
    pass  # Will be implemented in Step 2.2


def get_instrumentation_settings():
    """
    Get instrumentation settings for a specific Pydantic AI agent.
    
    Use this when you want to configure instrumentation per-agent
    rather than globally.
    
    Returns:
        Instrumentation settings to pass to Agent(..., instrument=...)
        
    Example:
        from agentbasis.frameworks.pydanticai import get_instrumentation_settings
        from pydantic_ai import Agent
        
        agent = Agent(
            "openai:gpt-4",
            instrument=get_instrumentation_settings()
        )
    """
    pass  # Will be implemented in Step 2.2


def get_metadata_callback() -> Callable[..., Dict[str, Any]]:
    """
    Get a metadata callback that injects AgentBasis context.
    
    This callback can be passed to Pydantic AI's metadata parameter
    to automatically include user_id, session_id, etc. in traces.
    
    Returns:
        A callable that returns context metadata
        
    Example:
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        from pydantic_ai import Agent
        
        agent = Agent(
            "openai:gpt-4",
            metadata=get_metadata_callback()
        )
    """
    pass  # Will be implemented in Step 2.3
