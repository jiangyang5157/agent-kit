"""AIClient protocol — the single interface for AI provider clients."""
from abc import ABC, abstractmethod
from typing import Any


class AIClient(ABC):
    """Protocol for AI provider clients.

    One abstract method. Provider-specific capabilities (cache, vision,
    streaming, etc.) belong on the concrete adapter, not here.
    """

    @abstractmethod
    def generate_content(
        self,
        model: str,
        contents: list[Any],
        *,
        system_instruction: str | None = None,
        tools: list | None = None,
        temperature: float = 0.5,
        response_json: bool = False,
        http_timeout: int | None = None,
    ):
        """Send contents and return a provider-agnostic response.

        Parameters
        ----------
        model : str
            Model identifier understood by the provider.

        contents : list
            Conversation in provider-agnostic dict format::

                {"role": "user", "text": "Hello"}
                {"role": "model", "text": "Hi, how can I help?"}
                {"role": "model", "tool_calls": [
                    {"id": "call_1", "name": "search", "args": {"q": "..."}}
                ]}
                {"role": "user", "tool_responses": [
                    {"id": "call_1", "name": "search", "result": "..."}
                ]}

            Text-only dicts are the base contract. Adapters may accept
            additional fields (e.g. visual_parts) and interpret what
            they support.

        system_instruction : str | None
            System-level instruction for the model. Supported by all
            major providers.

        tools : list[ToolDeclaration] | None
            Function-calling declarations. The response may contain
            ToolCall objects in ``AIResponse.tool_calls``.

        temperature : float
            Sampling temperature (0.0–1.0+). Default 0.5.

        response_json : bool
            Request JSON-formatted text output (not structured output,
            not function calling). Some adapters inject a JSON hint
            into the system message. For strict structured output or
            function calling, use the ``tools`` parameter instead.

        http_timeout : int | None
            Per-request timeout in seconds. Adapter default if None.

        Returns
        -------
        AIResponse
            Standardized response with text, optional tool_calls, and
            optional usage metadata.
        """
        ...
