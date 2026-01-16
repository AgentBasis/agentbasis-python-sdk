"""
Pydantic AI instrumentation for AgentBasis.

This module provides integration with Pydantic AI's built-in OpenTelemetry
instrumentation, ensuring traces flow to AgentBasis.

Pydantic AI already emits OTel traces via its instrumentation system. Since
AgentBasis sets up a global TracerProvider in agentbasis.init(), we just need
to enable Pydantic AI's instrumentation and the traces will automatically
flow to AgentBasis.
"""

from typing import Any, Callable, Dict, Optional, Union
import warnings

from agentbasis.context import get_context_attributes


# Track if we've already instrumented globally
_instrumented = False


def instrument(
    include_content: bool = True,
    include_binary_content: bool = False,
) -> None:
    """
    Enable AgentBasis instrumentation for all Pydantic AI agents.
    
    This configures Pydantic AI to send traces to AgentBasis via OpenTelemetry.
    Call this after agentbasis.init() and before creating any agents.
    
    Args:
        include_content: Whether to include prompts and completions in traces.
                        Set to False for privacy (default: True)
        include_binary_content: Whether to include binary content like images.
                               Can increase trace size significantly (default: False)
    
    Example:
        import agentbasis
        from agentbasis.frameworks.pydanticai import instrument
        
        agentbasis.init(api_key="...", agent_id="...")
        instrument()
        
        # Now all Pydantic AI agents will be traced
        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4")
        result = agent.run_sync("Hello!")
        
    Example with privacy controls:
        instrument(include_content=False)  # Don't log prompts/completions
    """
    global _instrumented
    
    if _instrumented:
        return
    
    try:
        from pydantic_ai import Agent
        
        # Get instrumentation settings
        settings = get_instrumentation_settings(
            include_content=include_content,
            include_binary_content=include_binary_content,
        )
        
        # Enable instrumentation globally for all agents
        Agent.instrument_all(settings)
        
        _instrumented = True
        
    except ImportError:
        warnings.warn(
            "pydantic-ai is not installed. Install it with: pip install pydantic-ai",
            ImportWarning
        )
    except AttributeError:
        # Older versions of pydantic-ai might not have instrument_all
        warnings.warn(
            "Your version of pydantic-ai does not support global instrumentation. "
            "Please upgrade to the latest version or use per-agent instrumentation.",
            UserWarning
        )


def get_instrumentation_settings(
    include_content: bool = True,
    include_binary_content: bool = False,
) -> Any:
    """
    Get instrumentation settings for a specific Pydantic AI agent.
    
    Use this when you want to configure instrumentation per-agent
    rather than globally, or when you need custom settings.
    
    Args:
        include_content: Whether to include prompts and completions in traces
        include_binary_content: Whether to include binary content like images
        
    Returns:
        InstrumentationSettings object to pass to Agent(..., instrument=...)
        Returns True if InstrumentationSettings is not available (basic mode)
        
    Example:
        from agentbasis.frameworks.pydanticai import get_instrumentation_settings
        from pydantic_ai import Agent
        
        # Basic usage
        agent = Agent(
            "openai:gpt-4",
            instrument=get_instrumentation_settings()
        )
        
        # With privacy controls (no prompt/completion logging)
        agent = Agent(
            "openai:gpt-4",
            instrument=get_instrumentation_settings(include_content=False)
        )
    """
    try:
        from pydantic_ai.agent import InstrumentationSettings
        
        return InstrumentationSettings(
            include_content=include_content,
            include_binary_content=include_binary_content,
        )
    except ImportError:
        # If InstrumentationSettings is not available, return True for basic instrumentation
        return True


def get_metadata_callback() -> Callable[..., Dict[str, Any]]:
    """
    Get a metadata callback that injects AgentBasis context.
    
    This callback can be passed to Pydantic AI's metadata parameter
    to automatically include user_id, session_id, etc. in traces.
    
    The callback reads from AgentBasis context at runtime, so it will
    pick up any context set via agentbasis.set_user(), set_session(), etc.
    
    Returns:
        A callable that returns context metadata
        
    Example:
        import agentbasis
        from agentbasis.frameworks.pydanticai import get_metadata_callback
        from pydantic_ai import Agent
        
        agent = Agent(
            "openai:gpt-4",
            metadata=get_metadata_callback()
        )
        
        # Set context before running
        agentbasis.set_user("user-123")
        agentbasis.set_session("session-456")
        
        # This run will include user_id and session_id in metadata
        result = agent.run_sync("Hello!")
    """
    def _get_agentbasis_metadata(*args, **kwargs) -> Dict[str, Any]:
        """
        Callback function that retrieves AgentBasis context attributes.
        
        This is called by Pydantic AI before each agent run to get metadata.
        """
        # Get current context from AgentBasis
        context_attrs = get_context_attributes()
        
        # Convert to a simpler format for Pydantic AI metadata
        metadata = {}
        
        if "agentbasis.user.id" in context_attrs:
            metadata["user_id"] = context_attrs["agentbasis.user.id"]
        
        if "agentbasis.session.id" in context_attrs:
            metadata["session_id"] = context_attrs["agentbasis.session.id"]
        
        if "agentbasis.conversation.id" in context_attrs:
            metadata["conversation_id"] = context_attrs["agentbasis.conversation.id"]
        
        if "agentbasis.metadata" in context_attrs:
            # Include custom metadata (already JSON serialized)
            metadata["custom"] = context_attrs["agentbasis.metadata"]
        
        return metadata
    
    return _get_agentbasis_metadata


def create_traced_agent(
    model: str,
    include_content: bool = True,
    **agent_kwargs
) -> Any:
    """
    Convenience function to create a Pydantic AI agent with AgentBasis tracing.
    
    This creates an agent with instrumentation and metadata callback
    pre-configured for AgentBasis.
    
    Args:
        model: The model to use (e.g., "openai:gpt-4", "anthropic:claude-3-opus")
        include_content: Whether to include prompts/completions in traces
        **agent_kwargs: Additional arguments to pass to Agent()
        
    Returns:
        A configured Pydantic AI Agent
        
    Example:
        from agentbasis.frameworks.pydanticai import create_traced_agent
        
        agent = create_traced_agent(
            "openai:gpt-4",
            system_prompt="You are a helpful assistant."
        )
        result = agent.run_sync("Hello!")
    """
    try:
        from pydantic_ai import Agent
        
        return Agent(
            model,
            instrument=get_instrumentation_settings(include_content=include_content),
            metadata=get_metadata_callback(),
            **agent_kwargs
        )
    except ImportError:
        raise ImportError(
            "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
        )
