"""Tests for agent-kit data models."""
import pytest
from agent_kit.models import AIError, AIResponse, ToolCall, Usage, ToolDeclaration


class TestToolCall:
    def test_default_args_is_empty_dict(self):
        tc = ToolCall(name="search")
        assert tc.args == {}
        assert isinstance(tc.args, dict)

    def test_with_args(self):
        tc = ToolCall(name="search", args={"q": "hello"})
        assert tc.name == "search"
        assert tc.args == {"q": "hello"}

    def test_immutable_name(self):
        """ToolCall.name is a string — it should be assignable but
        dataclass default_factory isolates args across instances."""
        tc1 = ToolCall(name="a")
        tc2 = ToolCall(name="b", args={"x": 1})
        assert tc1.args == {}
        assert tc2.args == {"x": 1}
        # Verify args dicts are independent (default_factory creates new dicts)
        tc1.args["y"] = 2
        assert ToolCall(name="c").args == {}


class TestUsage:
    def test_defaults_are_zero(self):
        u = Usage()
        assert u.total_token_count == 0
        assert u.input_token_count == 0
        assert u.output_token_count == 0

    def test_with_values(self):
        u = Usage(total_token_count=100, input_token_count=60, output_token_count=40)
        assert u.total_token_count == 100
        assert u.input_token_count == 60
        assert u.output_token_count == 40


class TestAIResponse:
    def test_text_only(self):
        r = AIResponse(text="Hello")
        assert r.text == "Hello"
        assert r.tool_calls is None
        assert r.usage is None

    def test_with_tool_calls(self):
        tc = ToolCall(name="search", args={"q": "x"})
        r = AIResponse(text="", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"

    def test_with_usage(self):
        u = Usage(total_token_count=50)
        r = AIResponse(text="Hi", usage=u)
        assert r.usage is not None
        assert r.usage.total_token_count == 50


class TestToolDeclaration:
    def test_minimal_declaration(self):
        td = ToolDeclaration(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )
        assert td.name == "get_weather"
        assert td.description == "Get current weather"
        assert td.parameters["type"] == "object"
        assert td.parameters["required"] == ["city"]
        assert td.parameters["properties"]["city"]["type"] == "string"

    def test_no_required_params(self):
        td = ToolDeclaration(
            name="ping",
            description="Check status",
            parameters={
                "type": "object",
                "properties": {},
            },
        )
        assert td.parameters.get("required") is None


class TestAIError:
    def test_raise_and_catch(self):
        with pytest.raises(AIError, match="test error"):
            raise AIError("test error")

    def test_is_exception(self):
        assert issubclass(AIError, Exception)

    def test_can_be_caught_as_exception(self):
        try:
            raise AIError("fail")
        except Exception as e:
            assert isinstance(e, AIError)
            assert str(e) == "fail"
