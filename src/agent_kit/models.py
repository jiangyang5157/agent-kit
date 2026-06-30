"""Provider-agnostic data models for agent-kit."""
from dataclasses import dataclass, field


class AIError(Exception):
    """Base exception for all agent-kit errors."""
    pass


@dataclass
class ToolCall:
    """Function call from model — name + arguments."""
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage snapshot.

    Only common-denominator fields. Provider-specific fields (cache
    tokens, reasoning tokens) are not included — consumers access
    those via the adapter's raw response.
    """
    total_token_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0


@dataclass
class AIResponse:
    """Provider-agnostic model response."""
    text: str
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None


@dataclass
class ToolDeclaration:
    """Tool signature in function-calling format.

    Compatible with OpenAI, Gemini, and Anthropic tool declarations.

    Parameters
    ----------
    name : str
        Function name the model uses to call.
    description : str
        What the function does — helps the model decide when to call.
    parameters : dict
        JSON Schema object describing the function's parameters:
        ``{"type": "object", "properties": {...}, "required": [...]}``
    """
    name: str
    description: str
    parameters: dict
