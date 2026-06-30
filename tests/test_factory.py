"""Tests for ClientFactory."""
import pytest
from agent_kit.factory import ClientFactory
from agent_kit.ai_client import AIClient
from agent_kit.models import AIError, AIResponse


class FakeAdapter(AIClient):
    def __init__(self, api_key=None, model=None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.extra = kwargs

    def generate_content(self, **kwargs):
        return AIResponse(text="fake")


class TestFromDict:
    def test_creates_adapter_with_api_key(self):
        adapters = {"fake": FakeAdapter}
        client = ClientFactory.from_dict(
            {"provider": "fake", "api_key": "sk-test"},
            adapters=adapters,
        )
        assert isinstance(client, FakeAdapter)
        assert client.api_key == "sk-test"

    def test_missing_provider_key_raises_ai_error(self):
        with pytest.raises(AIError, match="missing required key"):
            ClientFactory.from_dict(
                {"api_key": "sk-test"},
                adapters={"fake": FakeAdapter},
            )

    def test_unknown_provider_raises_ai_error(self):
        with pytest.raises(AIError, match="Unknown provider"):
            ClientFactory.from_dict(
                {"provider": "unknown"},
                adapters={"fake": FakeAdapter},
            )

    def test_unknown_provider_error_includes_registered_names(self):
        with pytest.raises(AIError, match="fake"):
            ClientFactory.from_dict(
                {"provider": "unknown"},
                adapters={"fake": FakeAdapter, "real": FakeAdapter},
            )

    def test_forwards_extra_kwargs(self):
        adapters = {"fake": FakeAdapter}
        client = ClientFactory.from_dict(
            {"provider": "fake", "api_key": "sk-test", "model": "m1"},
            adapters=adapters,
        )
        assert client.model == "m1"

    def test_does_not_mutate_config_dict(self):
        adapters = {"fake": FakeAdapter}
        config = {"provider": "fake", "api_key": "sk-test"}
        original_keys = list(config.keys())
        ClientFactory.from_dict(config, adapters=adapters)
        assert list(config.keys()) == original_keys

    def test_multiple_registered_adapters(self):
        class AdapterA(FakeAdapter):
            pass
        class AdapterB(FakeAdapter):
            pass

        adapters = {"a": AdapterA, "b": AdapterB}
        client = ClientFactory.from_dict(
            {"provider": "b", "api_key": "k"},
            adapters=adapters,
        )
        assert isinstance(client, AdapterB)
