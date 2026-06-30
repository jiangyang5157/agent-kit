"""ClientFactory — stateless resolver from config to AIClient instance."""
from agent_kit.ai_client import AIClient
from agent_kit.models import AIError


class ClientFactory:
    """Stateless factory — consumer supplies the adapter registry."""

    @staticmethod
    def from_dict(
        config: dict,
        adapters: dict[str, type[AIClient]],
    ) -> AIClient:
        """Create an AIClient from a config dict.

        Parameters
        ----------
        config : dict
            Must contain a ``"provider"`` key matching a key in
            ``adapters``. All other keys are forwarded to the adapter
            constructor as keyword arguments.

        adapters : dict[str, type[AIClient]]
            Registry mapping provider name -> adapter class. Each class
            must implement ``AIClient``.

        Returns
        -------
        AIClient
            Configured adapter instance.

        Raises
        ------
        AIError
            If ``config`` is missing the ``"provider"`` key, or the
            provider value does not match any key in ``adapters``.
        """
        if "provider" not in config:
            raise AIError("Config missing required key: 'provider'")
        provider = config["provider"]
        adapter_class = adapters.get(provider)
        if adapter_class is None:
            raise AIError(
                f"Unknown provider: '{provider}'. "
                f"Registered: {list(adapters.keys())}"
            )
        kwargs = {k: v for k, v in config.items() if k != "provider"}
        return adapter_class(**kwargs)
