"""agent-kit: Minimal AI provider abstraction for agent harnesses."""

__version__ = "0.1.0"

from agent_kit.models import (
    AIError,
    AIResponse,
    ToolCall,
    ToolDeclaration,
    Usage,
)
from agent_kit.ai_client import AIClient
from agent_kit.openai_adapter import OpenAICompatibleAdapter
from agent_kit.factory import ClientFactory

__all__ = [
    "AIClient",
    "AIError",
    "AIResponse",
    "ClientFactory",
    "OpenAICompatibleAdapter",
    "ToolCall",
    "ToolDeclaration",
    "Usage",
]
