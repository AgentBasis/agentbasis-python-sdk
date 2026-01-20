# Changelog

All notable changes to the AgentBasis Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-19

### Added

- **Core SDK**: Initialize with `agentbasis.init(api_key, agent_id)`
- **OpenTelemetry Integration**: Full OTel-based tracing with OTLP HTTP exporter
- **Graceful Shutdown**: Automatic flush on exit via `atexit`

#### LLM Integrations
- **OpenAI**: Sync, async, and streaming support for Chat Completions
- **Anthropic**: Sync, async, and streaming support for Messages API
- **Gemini**: Sync, async, and streaming support for GenerativeModel

#### Framework Integrations
- **LangChain**: Callback handler with full parent-child span relationships
- **Pydantic AI**: Integration via native OpenTelemetry support

#### Context Management
- User/session/conversation tracking with `set_user()`, `set_session()`, `set_conversation()`
- Context manager pattern with `agentbasis.context()`
- Decorator support with `@agentbasis.with_context()`

#### Decorators
- `@agentbasis.trace` for tracking any function (sync and async)

### Documentation
- Full documentation available at [docs.agentbasis.co](https://docs.agentbasis.co)
