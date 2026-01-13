from .chat import instrument_chat, instrument_async_chat

def instrument():
    """
    Auto-instruments the Google Gemini SDK (both sync and async).
    Call this function after `agentbasis.init()` and before using `google.generativeai`.
    
    This instruments:
    - GenerativeModel.generate_content() (sync)
    - GenerativeModel.generate_content_async() (async)
    """
    try:
        import google.generativeai as genai
        instrument_chat(genai)
        instrument_async_chat(genai)
    except ImportError:
        # If google.generativeai is not installed, we simply do nothing or could log a warning
        pass
