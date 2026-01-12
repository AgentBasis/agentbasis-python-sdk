from .chat import instrument_chat

def instrument():
    """
    Auto-instruments the Google Gemini SDK.
    Call this function after `agentbasis.init()` and before using `google.generativeai`.
    """
    # We try to import google.generativeai here to ensure it's available
    try:
        import google.generativeai as genai
        instrument_chat(genai)
    except ImportError:
        # If google.generativeai is not installed, we simply do nothing or could log a warning
        pass
