from .callback import AgentBasisCallbackHandler

# Module-level handler instance for convenience
_handler_instance = None


def get_callback_handler() -> AgentBasisCallbackHandler:
    """
    Get a new AgentBasis callback handler for LangChain.
    
    Use this to get a handler instance that you can pass to your
    LangChain chains, agents, or LLMs.
    
    Returns:
        A new AgentBasisCallbackHandler instance
        
    Example:
        from agentbasis.frameworks.langchain import get_callback_handler
        
        handler = get_callback_handler()
        
        # Pass to chain
        chain.invoke({"query": "..."}, config={"callbacks": [handler]})
        
        # Or pass to LLM
        llm = ChatOpenAI(callbacks=[handler])
        
        # Or pass to agent
        agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
    """
    return AgentBasisCallbackHandler()


def instrument() -> AgentBasisCallbackHandler:
    """
    Get the global AgentBasis callback handler for LangChain.
    
    This returns a singleton handler instance that can be reused across
    your application. For most use cases, you should pass this handler
    explicitly to your LangChain components.
    
    Returns:
        The global AgentBasisCallbackHandler instance
        
    Example - Explicit callbacks (recommended):
        from agentbasis.frameworks.langchain import instrument
        
        handler = instrument()
        
        # Option 1: Pass to invoke/run
        chain.invoke({"query": "..."}, config={"callbacks": [handler]})
        
        # Option 2: Pass to constructor
        llm = ChatOpenAI(model="gpt-4", callbacks=[handler])
        
        # Option 3: Pass to agent executor
        agent = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=[handler]
        )
        
    Example - With RunnableConfig:
        from langchain_core.runnables import RunnableConfig
        
        handler = instrument()
        config = RunnableConfig(callbacks=[handler])
        
        result = chain.invoke({"query": "..."}, config=config)
    
    Note:
        Unlike OpenAI/Anthropic instrumentation which patches the SDK globally,
        LangChain requires explicit callback passing. This is because LangChain
        has its own callback system that doesn't support global monkey-patching.
    """
    global _handler_instance
    
    if _handler_instance is None:
        _handler_instance = AgentBasisCallbackHandler()
    
    return _handler_instance


def get_callback_config():
    """
    Get a LangChain RunnableConfig with AgentBasis callbacks pre-configured.
    
    This is a convenience function for users who prefer working with
    RunnableConfig objects.
    
    Returns:
        A dict that can be passed as the `config` parameter to invoke/batch/stream
        
    Example:
        from agentbasis.frameworks.langchain import get_callback_config
        
        config = get_callback_config()
        result = chain.invoke({"query": "..."}, config=config)
    """
    handler = instrument()
    return {"callbacks": [handler]}


__all__ = [
    "AgentBasisCallbackHandler",
    "instrument",
    "get_callback_handler",
    "get_callback_config",
]
