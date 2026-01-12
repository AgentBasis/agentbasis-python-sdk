from .chat import instrument_chat

def instrument():
    """
    Auto-instruments the OpenAI SDK.
    Call this function after `agentbasis.init()` and before using `openai`.
    """
    # We try to import openai here to ensure it's available
    try:
        import openai
        instrument_chat(openai)
    except ImportError:
        # If openai is not installed, we simply do nothing or could log a warning
        pass
