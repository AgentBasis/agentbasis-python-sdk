# AgentBasis Python SDK

**Management & Observability SDK for AI Agents**

The **AgentBasis Python SDK** provides a simple, lightweight way to track the performance, traces, sessions, and behavior of AI agents. It sends data using the **OpenTelemetry (OTel)** standard, making it compatible with AgentBasis and other observability backends.

This is the **foundation SDK** that enables deep observability for coded agents built with:
- Pure Python
- LLM Providers: 
    - OpenAI
    - Anthropic
    - Gemini
- Frameworks
    - LangChain

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
Automatically track chains, tools, and LLM calls in LangChain.

```python
from agentbasis.frameworks.langchain import instrument

# Enable LangChain instrumentation
instrument()

# Your existing LangChain code...
from langchain.llms import OpenAI
llm = OpenAI()
llm.predict("Hello world")
```

### 6. Track Users (Optional)
Associate traces with specific users to debug issues and see per-user analytics.

```python
# Set the current user (from your auth system)
agentbasis.set_user(current_user.id)

# All subsequent LLM calls will be tagged with this user
response = client.chat.completions.create(...)
```

## Core Concepts

- **OpenTelemetry**: We use OTel under the hood for maximum compatibility.
- **Spans**: Every action (function call, LLM request) is recorded as a Span.
- **Transport**: Data is batched and sent asynchronously to AgentBasis Backend service

## Notes:
After every version update: python -m build (to build the latest version and update)

Install the sdk for testing:  `pip install git+https://github.com/AgentBasis/agentbasis-python-sdk.git`
