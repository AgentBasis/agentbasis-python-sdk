# AgentBasis Python SDK

**Management & Observability SDK for AI Agents**

The **AgentBasis Python SDK** provides a simple, lightweight way to track the performance, traces, sessions, and behavior of AI agents. It sends data using the **OpenTelemetry (OTel)** standard, making it compatible with AgentBasis and other observability backends.

This is the **foundation SDK** that enables deep observability for coded agents built with:
- Pure Python
- LLM Providers: 
    - Open AI
    - Gemini
- Frameworks
    - LangChain

## Installation

```bash
pip install agentbay
```

## Quick Start

### 1. Initialize the SDK
Start by initializing the SDK with your API key and Agent ID. This usually goes at the top of your main application file.

```python
import agentbay

# Initialize with your API Key and Agent ID
agentbay.init(
    api_key="your-api-key-here", 
    agent_id="your-agent-id-here"
)
```

### 2. Manual Tracking (Decorators)
Use the `@trace` decorator to automatically track any function.

```python
from agentbay import trace

@trace
def chat_with_user(query):
    # Your agent logic here
    return "Response to: " + query

# When you call this, data is automatically sent to AgentBay
chat_with_user("Hello world")
```

### 3. OpenAI Integration
Automatically track all your OpenAI calls (models, tokens, prompts) with one line of code.

```python
from agentbay.llms import openai

# Enable OpenAI instrumentation
openai.instrument()

# Now just use the OpenAI client as normal
import openai as oa
client = oa.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 4. LangChain Integration
Automatically track chains, tools, and LLM calls in LangChain.

```python
from agentbay.frameworks import langchain

# Enable LangChain instrumentation
langchain.instrument()

# Your existing LangChain code...
from langchain.llms import OpenAI
llm = OpenAI()
llm.predict("Hello world")
```

## Core Concepts

- **OpenTelemetry**: We use OTel under the hood for maximum compatibility.
- **Spans**: Every action (function call, LLM request) is recorded as a Span.
- **Transport**: Data is batched and sent asynchronously to AgentBay Backend service

## Notes:
After every version update: python -m build (to build the latest version and update)

Install the sdk for testing:  `pip install git+https://github.com/AgentBay-AI/agentbay-python-sdk.git`

