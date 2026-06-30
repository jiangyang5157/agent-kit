"""Contract tests for AIClient protocol."""
import pytest
from agent_kit.ai_client import AIClient
from agent_kit.models import AIResponse


class TestAIClientContract:
    """Any AIClient implementation must pass these tests."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AIClient()

    def test_missing_generate_content_fails(self):
        class Incomplete(AIClient):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_implementation_with_generate_content_succeeds(self):
        class Complete(AIClient):
            def generate_content(self, model, contents, *,
                                 system_instruction=None, tools=None,
                                 temperature=0.5, response_json=False,
                                 http_timeout=None):
                return AIResponse(text="ok")

        client = Complete()
        result = client.generate_content(
            model="test",
            contents=[{"role": "user", "text": "hi"}],
        )
        assert isinstance(result, AIResponse)
        assert result.text == "ok"

    def test_all_keyword_only_params_are_optional(self):
        """Consumer must be able to call with only model + contents."""
        class Minimal(AIClient):
            def generate_content(self, model, contents, **kwargs):
                return AIResponse(text=model)

        client = Minimal()
        result = client.generate_content("m1", [])
        assert result.text == "m1"
