from .chat import instrument_chat, instrument_async_chat

def instrument():
    """
    Auto-instruments the OpenAI SDK (both sync and async).
    Call this function after `agentbasis.init()` and before using `openai`.
    
    This instruments:
    - OpenAI().chat.completions.create() (sync)
    - AsyncOpenAI().chat.completions.create() (async)
    """
    try:
        import openai
        instrument_chat(openai)
        instrument_async_chat(openai)
    except ImportError:
        # If openai is not installed, we simply do nothing or could log a warning
        pass
