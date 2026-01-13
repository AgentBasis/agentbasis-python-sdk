from .callback import AgentBasisCallbackHandler

def instrument():
    """
    Auto-instruments LangChain by setting a global callback handler.
    Note: This requires the user to have 'langchain' installed.
    """
    try:
        # Try importing the global handler configuration
        # Note: This path changes often in LangChain versions.
        # We support the modern 'langchain_core' or 'langchain' approach.
        import langchain
        
        # Create our handler
        handler = AgentBasisCallbackHandler()
        
        # Add to global handlers if it's a list, or create a list
        # This is a best-effort attempt to inject ourselves globally
        if not hasattr(langchain, "callbacks"):
            langchain.callbacks = []
            
        # If it's already a list, append
        if isinstance(langchain.callbacks, list):
            langchain.callbacks.append(handler)
        else:
            # If it's something else (manager?), we might just print a warning
            # For now, we assume standard usage.
            pass
            
    except ImportError:
        pass

__all__ = ["AgentBasisCallbackHandler", "instrument"]
