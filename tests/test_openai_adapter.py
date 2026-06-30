"""Tests for OpenAICompatibleAdapter."""
import threading
from unittest.mock import MagicMock
import pytest
from agent_kit.openai_adapter import OpenAICompatibleAdapter
from agent_kit.models import AIResponse, ToolCall, Usage, ToolDeclaration


class FakeAdapter(OpenAICompatibleAdapter):
    """Concrete adapter for testing."""
    def __init__(self, api_key="test-key"):
        super().__init__(
            api_key=api_key,
            default_model="test-model",
            base_url="https://test.api.com",
            provider_label="Test",
        )


@pytest.fixture
def adapter():
    return FakeAdapter()


# ── _build_messages ──────────────────────────────────────────────────

class TestBuildMessages:
    def test_simple_user_message(self, adapter):
        contents = [{"role": "user", "text": "Hello"}]
        result = adapter._build_messages(None, contents)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_with_system_instruction(self, adapter):
        contents = [{"role": "user", "text": "Hello"}]
        result = adapter._build_messages("You are helpful", contents)
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_json_mode_adds_hint(self, adapter):
        contents = [{"role": "user", "text": "List items"}]
        result = adapter._build_messages("Be concise", contents, response_json=True)
        system_content = result[0]["content"]
        assert "JSON" in system_content

    def test_json_mode_without_system_instruction(self, adapter):
        contents = [{"role": "user", "text": "List items"}]
        result = adapter._build_messages(None, contents, response_json=True)
        system_content = result[0]["content"]
        assert "JSON" in system_content

    def test_model_message(self, adapter):
        contents = [{"role": "model", "text": "Hi there"}]
        result = adapter._build_messages(None, contents)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_tool_calls_message(self, adapter):
        contents = [{
            "role": "model",
            "tool_calls": [{"id": "c1", "name": "search", "args": {"q": "x"}}],
        }]
        result = adapter._build_messages(None, contents)
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "c1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'

    def test_tool_responses_message(self, adapter):
        contents = [{
            "role": "user",
            "tool_responses": [{"id": "c1", "name": "search", "result": "found"}],
        }]
        result = adapter._build_messages(None, contents)
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "c1"
        assert msg["content"] == "found"

    def test_string_content(self, adapter):
        contents = ["plain string"]
        result = adapter._build_messages(None, contents)
        assert result == [{"role": "user", "content": "plain string"}]


# ── _convert_tools ───────────────────────────────────────────────────

class TestConvertTools:
    def test_empty_list(self, adapter):
        assert adapter._convert_tools([]) == []

    def test_single_tool(self, adapter):
        tools = [
            ToolDeclaration(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            )
        ]
        result = adapter._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        func = result[0]["function"]
        assert func["name"] == "get_weather"
        assert func["description"] == "Get weather"
        assert func["parameters"]["type"] == "object"
        assert func["parameters"]["required"] == ["city"]
        assert func["parameters"]["properties"]["city"]["type"] == "string"

    def test_tool_with_enum_and_minimum(self, adapter):
        tools = [
            ToolDeclaration(
                name="set_mode",
                description="Set mode",
                parameters={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "description": "Mode",
                            "enum": ["a", "b"],
                        },
                        "count": {"type": "integer", "minimum": 0},
                    },
                },
            )
        ]
        result = adapter._convert_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert props["mode"]["enum"] == ["a", "b"]
        assert props["count"]["minimum"] == 0

    def test_tool_with_items(self, adapter):
        tools = [
            ToolDeclaration(
                name="search",
                description="Search",
                parameters={
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "description": "Tags",
                            "items": {"type": "string"},
                        },
                    },
                },
            )
        ]
        result = adapter._convert_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert props["tags"]["items"]["type"] == "string"


# ── _parse_response ──────────────────────────────────────────────────

class TestParseResponse:
    def test_text_response(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = "Hello world"
        mock_msg.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.text == "Hello world"
        assert result.tool_calls is None
        assert result.usage is None

    def test_tool_call_response(self, adapter):
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"q": "hello"}'
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = [mock_tc]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].args == {"q": "hello"}

    def test_tool_call_with_malformed_json_args(self, adapter):
        mock_tc = MagicMock()
        mock_tc.function.name = "bad"
        mock_tc.function.arguments = "not json"
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = [mock_tc]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.tool_calls[0].args == {}

    def test_usage_response(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_msg.tool_calls = None
        mock_usage = MagicMock()
        mock_usage.total_tokens = 50
        mock_usage.prompt_tokens = 30
        mock_usage.completion_tokens = 20
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = mock_usage

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.usage is not None
        assert result.usage.total_token_count == 50
        assert result.usage.input_token_count == 30
        assert result.usage.output_token_count == 20

    def test_json_response_cleans_markdown_fence(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = '```json\n{"key": "value"}\n```'
        mock_msg.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=True)
        assert result.text == '{"key": "value"}'

    def test_json_response_cleans_plain_fence(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = '```\n{"x": 1}\n```'
        mock_msg.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=True)
        assert result.text == '{"x": 1}'

    def test_json_response_no_fence(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = '{"x": 1}'
        mock_msg.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=True)
        assert result.text == '{"x": 1}'

    def test_none_content(self, adapter):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = None
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.text == ""

    def test_empty_choices(self, adapter):
        """Response with choices=[], verify AIResponse(text="") returned."""
        mock_resp = MagicMock()
        mock_resp.choices = []
        mock_resp.usage = None

        result = adapter._parse_response(mock_resp, response_json=False)
        assert result.text == ""
        assert result.tool_calls is None
        assert result.usage is None


# ── generate_content integration ─────────────────────────────────────

class TestGenerateContent:
    def test_calls_openai_sdk(self, adapter):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "response text"
        mock_choice.message.tool_calls = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=None,
        )
        adapter._client = mock_client

        result = adapter.generate_content(
            model="test-model",
            contents=[{"role": "user", "text": "Hello"}],
        )
        assert result.text == "response text"
        mock_client.chat.completions.create.assert_called_once()

    def test_uses_default_model_when_gemini_in_name(self, adapter):
        """Model name checks for 'gemini' in the provided model string
        to decide whether to fall back to default_model."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=None,
        )
        adapter._client = mock_client

        adapter.generate_content(
            model="gemini-flash",
            contents=[{"role": "user", "text": "hi"}],
        )
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "test-model"

    def test_passes_correct_model(self, adapter):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=None,
        )
        adapter._client = mock_client

        adapter.generate_content(
            model="deepseek-v4",
            contents=[{"role": "user", "text": "hi"}],
        )
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "deepseek-v4"

    def test_passes_tools_when_provided(self, adapter):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=None,
        )
        adapter._client = mock_client

        tool = ToolDeclaration(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        adapter.generate_content(
            model="test",
            contents=[{"role": "user", "text": "hi"}],
            tools=[tool],
        )
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] is not None

    def test_no_tools_when_none(self, adapter):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=None,
        )
        adapter._client = mock_client

        adapter.generate_content(
            model="test",
            contents=[{"role": "user", "text": "hi"}],
        )
        call_args = mock_client.chat.completions.create.call_args
        assert "tools" not in call_args[1]


# ── _create_client hook ──────────────────────────────────────────────

class TestCreateClient:
    def test_default_creates_openai_client(self, adapter):
        client = adapter._create_client()
        assert client is not None
        assert client.base_url == "https://test.api.com"

    def test_can_override_in_subclass(self):
        class CustomAdapter(FakeAdapter):
            def _create_client(self):
                return "custom-client"

        a = CustomAdapter()
        assert a._get_client() == "custom-client"

    def test_thread_safe(self):
        """Spawn 2 threads, call _get_client() on each, assert both return same client."""
        results = []

        def get_it(a):
            results.append(a._get_client())

        a = FakeAdapter()
        t1 = threading.Thread(target=get_it, args=(a,))
        t2 = threading.Thread(target=get_it, args=(a,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        assert results[0] is results[1]
