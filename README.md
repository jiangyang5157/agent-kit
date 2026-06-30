# Agent-Kit

Protocol contracts for AI agents. ~300 lines.

## Install

```bash
# Development (editable)
pip install -e .

# From GitHub
pip install "agent-kit @ git+https://github.com/your-org/agent-kit.git"

# From PyPI (future)
pip install agent-kit
```

## Quick Start

```python
from agent_kit import OpenAICompatibleAdapter, AIClient

# Write a provider in 6 lines
class DeepSeekAdapter(OpenAICompatibleAdapter):
    def __init__(self, api_key):
        super().__init__(
            api_key=api_key,
            default_model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            provider_label="DeepSeek",
        )

# Use it
client = DeepSeekAdapter(api_key="sk-...")
response = client.generate_content(
    model="deepseek-v4-flash",
    contents=[{"role": "user", "text": "Hello!"}],
)
print(response.text)
```

## Public API (8 symbols)

| Symbol | Kind | Purpose |
|--------|------|---------|
| `AIClient` | ABC | Protocol -- one abstract method |
| `AIResponse` | dataclass | Provider-agnostic response |
| `ToolCall` | dataclass | Function call (name + args) |
| `ToolDeclaration` | dataclass | Tool signature (JSON Schema) |
| `Usage` | dataclass | Token usage (total/input/output) |
| `AIError` | Exception | Base for all agent-kit errors |
| `OpenAICompatibleAdapter` | class | Base class for OpenAI-protocol providers |
| `ClientFactory` | class | Stateless config->adapter resolver |

## Adapter Extension Pattern

Override hooks for custom behavior:

```python
class MyAdapter(OpenAICompatibleAdapter):
    def __init__(self, api_key):
        super().__init__(
            api_key=api_key,
            default_model="my-model",
            base_url="https://my.api.com",
            provider_label="MyProvider",
        )

    def _create_client(self):
        """Custom OpenAI client (proxy, headers, etc.)."""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=...,
        )

    def _build_messages(self, system_instruction, contents, *, response_json=False):
        """Custom message conversion (e.g. multimodal)."""
        return super()._build_messages(...)

    def _parse_response(self, response, response_json):
        """Custom response parsing (e.g. reasoning_content)."""
        return super()._parse_response(...)
```

## Write a Provider-Protocol Adapter

Providers not speaking OpenAI protocol (Gemini, Anthropic)
implement `AIClient` directly:

```python
from agent_kit import AIClient, AIResponse

class GeminiAdapter(AIClient):
    def generate_content(self, model, contents, *, system_instruction=None,
                         tools=None, temperature=0.5, response_json=False,
                         http_timeout=None) -> AIResponse:
        # Full custom implementation with google-genai SDK
        ...

    # Provider-specific features exposed on the adapter, not the protocol
    def create_cache(self, **kwargs): ...
    def delete_cache(self, name): ...
```

## Factory

```python
from agent_kit import ClientFactory

client = ClientFactory.from_dict(
    {"provider": "deepseek", "api_key": "sk-..."},
    adapters={
        "deepseek": DeepSeekAdapter,
        "gemini": GeminiAdapter,
    },
)
```

## Contents Format

The `contents` parameter uses provider-agnostic dicts:

```python
[
    {"role": "user", "text": "Hello"},
    {"role": "model", "text": "Hi, how can I help?"},
    {"role": "model", "tool_calls": [
        {"id": "call_1", "name": "search", "args": {"q": "..."}},
    ]},
    {"role": "user", "tool_responses": [
        {"id": "call_1", "name": "search", "result": "..."},
    ]},
]
```

## License

Apache 2.0
