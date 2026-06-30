"""Verify the public API surface matches the spec."""
import agent_kit


def test_version():
    assert agent_kit.__version__ == "0.1.0"


def test_all_exports():
    expected = [
        "AIClient",
        "AIError",
        "AIResponse",
        "OpenAICompatibleAdapter",
        "ClientFactory",
        "ToolCall",
        "ToolDeclaration",
        "Usage",
    ]
    for name in expected:
        assert hasattr(agent_kit, name), f"Missing: {name}"
    assert sorted(agent_kit.__all__) == sorted(expected)


def test_aiclient_is_abstract():
    from agent_kit import AIClient
    import inspect
    assert inspect.isabstract(AIClient)


def test_can_import_top_level():
    """All 8 symbols importable from agent_kit directly."""
    from agent_kit import (
        AIClient,
        AIError,
        AIResponse,
        OpenAICompatibleAdapter,
        ClientFactory,
        ToolCall,
        ToolDeclaration,
        Usage,
    )
    # If we got here without ImportError, it works
    assert AIClient is not None
    assert AIError is not None
    assert AIResponse is not None
    assert OpenAICompatibleAdapter is not None
    assert ClientFactory is not None
    assert ToolCall is not None
    assert ToolDeclaration is not None
    assert Usage is not None


def test_can_import_from_submodules():
    """Granular imports also work."""
    from agent_kit.models import AIResponse, ToolCall, Usage, ToolDeclaration, AIError
    from agent_kit.ai_client import AIClient
    from agent_kit.openai_adapter import OpenAICompatibleAdapter
    from agent_kit.factory import ClientFactory
    assert True
