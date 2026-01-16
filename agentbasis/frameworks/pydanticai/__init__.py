"""
Pydantic AI integration for AgentBasis.

This module provides instrumentation for Pydantic AI agents, enabling
automatic tracing and observability via OpenTelemetry.

Usage:
    import agentbasis
    from agentbasis.frameworks.pydanticai import instrument
    
    agentbasis.init(api_key="...", agent_id="...")
    instrument()
    
    # All Pydantic AI agents are now traced
    from pydantic_ai import Agent
    agent = Agent("openai:gpt-4")
    result = agent.run_sync("Hello!")
"""

from .instrumentation import (
    instrument,
    get_instrumentation_settings,
    get_metadata_callback,
    create_traced_agent,
)

__all__ = [
    "instrument",
    "get_instrumentation_settings",
    "get_metadata_callback",
    "create_traced_agent",
]
