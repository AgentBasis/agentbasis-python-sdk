# AgentBasis Python SDK

**Management & Observability SDK for AI Agents in production**

The **AgentBasis Python SDK** provides a simple, lightweight way to track the performance, traces, sessions, and behavior of AI agents. It sends data using the **OpenTelemetry (OTel)** standard, making it compatible with AgentBasis and other observability backends.

This is the **foundation SDK** that enables deep observability for coded agents built with:
- Pure Python
- LLM Providers: 
    - OpenAI
    - Anthropic
    - Gemini
- Frameworks
    - LangChain
    - Pydantic AI

## Installation

```bash
pip install agentbasis
```

## Quick Start

### 1. Initialize the SDK
Start by initializing the SDK with your API key and Agent ID. This usually goes at the top of your main application file.

```python
import agentbasis

# Initialize with your API Key and Agent ID
agentbasis.init(
    api_key="your-api-key-here", 
    agent_id="your-agent-id-here"
)
```

### 2. Manual Tracking (Decorators)
Use the `@trace` decorator to automatically track any function.

```python
from agentbasis import trace

@trace
def chat_with_user(query):
    # Your agent logic here
    return "Response to: " + query

# When you call this, data is automatically sent to AgentBasis
chat_with_user("Hello world")
```

### 3. OpenAI Integration
Automatically track all your OpenAI calls (models, tokens, prompts) with one line of code.

```python
from agentbasis.llms.openai import instrument

# Enable OpenAI instrumentation
instrument()

# Now just use the OpenAI client as normal
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 4. Anthropic Integration
Automatically track all your Anthropic Claude calls.

```python
from agentbasis.llms.anthropic import instrument

# Enable Anthropic instrumentation
instrument()

# Now just use the Anthropic client as normal
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 5. LangChain Integration
Track chains, tools, retrievers, and LLM calls in LangChain with full parent-child span relationships.

```python
from agentbasis.frameworks.langchain import get_callback_handler

# Create a callback handler
handler = get_callback_handler()

# Pass it to your LangChain calls
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello world", config={"callbacks": [handler]})
```

For chains and agents, pass the callback handler in the config:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from agentbasis.frameworks.langchain import get_callback_config

# Use get_callback_config() for convenience
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{query}"))
result = chain.invoke({"query": "What is AI?"}, config=get_callback_config())
```

### 6. Pydantic AI Integration
Track Pydantic AI agents with built-in OpenTelemetry support.

```python
from agentbasis.frameworks.pydanticai import instrument

# Enable global instrumentation for all Pydantic AI agents
instrument()

# Your agents are now automatically traced
from pydantic_ai import Agent
agent = Agent("openai:gpt-4")
result = agent.run_sync("Hello!")
```

For per-agent control with user context:

```python
from agentbasis.frameworks.pydanticai import create_traced_agent

# Create an agent pre-configured with tracing and context
agent = create_traced_agent(
    "openai:gpt-4",
    system_prompt="You are a helpful assistant."
)

# Set user context - it will be included in traces
agentbasis.set_user("user-123")
result = agent.run_sync("Hello!")
```

### 7. Track Users & Sessions (Optional)
Associate traces with specific users and sessions to debug issues and see per-user analytics.

```python
# Set the current user (from your auth system)
agentbasis.set_user(current_user.id)

# Optionally set session and conversation IDs
agentbasis.set_session("session-abc")
agentbasis.set_conversation("conv-123")

# All subsequent LLM calls will be tagged with this context
response = client.chat.completions.create(...)
```

Or use the context manager for scoped context:

```python
from agentbasis import context

with context(user_id="user-123", session_id="session-abc"):
    # All traces in this block include the context
    response = client.chat.completions.create(...)
```

## Core Concepts

- **OpenTelemetry**: We use OTel under the hood for maximum compatibility.
- **Spans**: Every action (function call, LLM request) is recorded as a Span.
- **Transport**: Data is batched and sent asynchronously to AgentBasis Backend service

## Notes:
After every version update: python -m build (to build the latest version and update)

Install the sdk for testing:  `pip install git+https://github.com/AgentBasis/agentbasis-python-sdk.git`
