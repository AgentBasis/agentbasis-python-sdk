from .messages import instrument_messages, instrument_async_messages


def instrument():
    """
    Auto-instruments the Anthropic SDK (both sync and async).
    Call this function after `agentbasis.init()` and before using `anthropic`.
    
    This instruments:
    - Anthropic().messages.create() (sync)
    - AsyncAnthropic().messages.create() (async)
    """
    try:
        import anthropic
        instrument_messages(anthropic)
        instrument_async_messages(anthropic)
    except ImportError:
        pass
