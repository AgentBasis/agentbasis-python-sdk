"""
Context management for AgentBasis.

This module provides context tracking for user sessions, allowing developers
to associate traces with specific users, sessions, and conversations.
"""

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Dict, Any
import functools
import json


# Context variables for storing user/session info
# These are thread-safe and async-safe
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
_conversation_id: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)
_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar('metadata', default=None)


class AgentBasisContext:
    """
    Context manager for setting user/session context.
    
    All spans created within this context will automatically include
    the user_id, session_id, and other context attributes.
    
    Example:
        with AgentBasisContext(user_id="user-123", session_id="sess-456"):
            response = client.chat.completions.create(...)
            # This span will have user_id and session_id attributes
    """ 
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.metadata = metadata
        
        # Store tokens to reset on exit
        self._tokens = []
    
    def __enter__(self):
        # Set context variables and store tokens for cleanup
        if self.user_id is not None:
            self._tokens.append(('user_id', _user_id.set(self.user_id)))
        if self.session_id is not None:
            self._tokens.append(('session_id', _session_id.set(self.session_id)))
        if self.conversation_id is not None:
            self._tokens.append(('conversation_id', _conversation_id.set(self.conversation_id)))
        if self.metadata is not None:
            self._tokens.append(('metadata', _metadata.set(self.metadata)))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables to their previous values
        for name, token in self._tokens:
            if name == 'user_id':
                _user_id.reset(token)
            elif name == 'session_id':
                _session_id.reset(token)
            elif name == 'conversation_id':
                _conversation_id.reset(token)
            elif name == 'metadata':
                _metadata.reset(token)
        return False  # Don't suppress exceptions


# Convenience function (alias for AgentBasisContext)
@contextmanager
def context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for setting user/session context.
    
    Example:
        with agentbasis.context(user_id="user-123"):
            response = client.chat.completions.create(...)
    """
    with AgentBasisContext(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        metadata=metadata
    ):
        yield


# Global setters for simpler usage
def set_user(user_id: Optional[str]):
    """
    Set the current user ID globally.
    
    This affects all subsequent spans until changed or cleared.
    For scoped context, use the context() context manager instead.
    
    Example:
        agentbasis.set_user("user-123")
        response = client.chat.completions.create(...)  # Has user_id
        agentbasis.set_user(None)  # Clear
    """
    _user_id.set(user_id)


def set_session(session_id: Optional[str]):
    """
    Set the current session ID globally.
    
    Example:
        agentbasis.set_session("sess-456")
    """
    _session_id.set(session_id)


def set_conversation(conversation_id: Optional[str]):
    """
    Set the current conversation ID globally.
    
    Example:
        agentbasis.set_conversation("conv-789")
    """
    _conversation_id.set(conversation_id)


def set_metadata(metadata: Optional[Dict[str, Any]]):
    """
    Set custom metadata globally.
    
    Example:
        agentbasis.set_metadata({"plan": "pro", "feature_flag": "new_ui"})
    """
    _metadata.set(metadata)


# Getters for internal use by instrumentations
def get_user() -> Optional[str]:
    """Get the current user ID from context."""
    return _user_id.get()


def get_session() -> Optional[str]:
    """Get the current session ID from context."""
    return _session_id.get()


def get_conversation() -> Optional[str]:
    """Get the current conversation ID from context."""
    return _conversation_id.get()


def get_metadata() -> Optional[Dict[str, Any]]:
    """Get the current metadata from context."""
    return _metadata.get()


def get_context_attributes() -> Dict[str, Any]:
    """
    Get all context attributes as a dictionary.
    
    This is used internally by instrumentations to inject context into spans.
    Only returns attributes that are set (not None).
    """
    attributes = {}
    
    user_id = get_user()
    if user_id:
        attributes["agentbasis.user.id"] = user_id
    
    session_id = get_session()
    if session_id:
        attributes["agentbasis.session.id"] = session_id
    
    conversation_id = get_conversation()
    if conversation_id:
        attributes["agentbasis.conversation.id"] = conversation_id
    
    metadata = get_metadata()
    if metadata:
        attributes["agentbasis.metadata"] = json.dumps(metadata)
    
    return attributes


# Decorator for function-level context
def with_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator to set context for a function.
    
    Example:
        @agentbasis.with_context(user_id="user-123")
        def my_agent_function():
            response = client.chat.completions.create(...)
            return response
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with AgentBasisContext(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                metadata=metadata
            ):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with AgentBasisContext(
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                metadata=metadata
            ):
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if _is_async_function(func):
            return async_wrapper
        return wrapper
    
    return decorator


def _is_async_function(func) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def inject_context_to_span(span) -> None:
    """
    Inject current context attributes into a span.
    
    This is called by LLM instrumentations to automatically add
    user/session context to every span.
    
    Args:
        span: An OpenTelemetry Span object
    """
    attributes = get_context_attributes()
    for key, value in attributes.items():
        span.set_attribute(key, value)
