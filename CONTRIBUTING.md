# Contributing to AgentBasis Python SDK

Thank you for your interest in contributing to the AgentBasis Python SDK!

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation for Development

1. Clone the repository:
```bash
git clone https://github.com/AgentBasis/agentbasis-python-sdk.git
cd agentbasis-python-sdk
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install development dependencies:
```bash
pip install pytest pytest-asyncio
```

## Building the Package

After making changes, rebuild the package:

```bash
python -m build
```

This updates the distribution files in `dist/`.

## Versioning & Releasing to PyPI

We use [Semantic Versioning](https://semver.org/):

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fix | Patch | `0.1.0` → `0.1.1` |
| New feature (backward compatible) | Minor | `0.1.0` → `0.2.0` |
| Breaking change | Major | `0.1.0` → `1.0.0` |

### To Release a New Version

1. Update the version in `pyproject.toml`:
   ```toml
   version = "0.1.1"
   ```

2. Update `CHANGELOG.md` with the changes

3. Build the package:
   ```bash
   python -m build
   ```

4. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

**Note:** PyPI does not allow overwriting existing versions. Once a version is published, you must increment the version number for any changes or fixes.

## Testing Locally

### Install from Local Build

```bash
pip install dist/agentbasis-0.1.0-py3-none-any.whl
```

### Install from GitHub (for testing before release)

```bash
pip install git+https://github.com/AgentBasis/agentbasis-python-sdk.git
```

### Run Unit Tests

```bash
python -m pytest tests/
```

## Project Structure

```
agentbasis/
├── __init__.py          # Main entry point, init() function
├── client.py            # AgentBasis client with OTel setup
├── config.py            # Configuration handling
├── context.py           # Context management (user, session, etc.)
├── decorators.py        # @trace decorator
├── llms/                # LLM provider integrations
│   ├── openai/
│   ├── anthropic/
│   └── gemini/
└── frameworks/          # Framework integrations
    ├── langchain/
    └── pydanticai/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to public functions
- Keep functions focused and single-purpose

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/my-feature`)
7. Open a Pull Request

## Questions?

If you have questions, please reach out to support@agentbasis.co
