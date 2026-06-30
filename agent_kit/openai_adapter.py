"""OpenAI-compatible adapter base class."""
import json
import logging
from typing import Any

from agent_kit.models import AIResponse, ToolCall, Usage, ToolDeclaration
from agent_kit.ai_client import AIClient

logger = logging.getLogger(__name__)

JSON_HINT = (
    "IMPORTANT: You MUST respond ONLY with a valid JSON object. "
    "Do NOT include markdown blocks, preamble, or explanations."
)


class OpenAICompatibleAdapter(AIClient):
    """Base class for providers that speak the OpenAI chat completions
    protocol (DeepSeek, Qwen, Grok, Fireworks, Together, Ollama, etc.).

    Consumer extends this with their own ``__init__`` to set api_key,
    base_url, and default_model. For custom behavior (proxy, vision,
    extra headers), override the hook methods.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str,
        base_url: str,
        provider_label: str,
        *,
        http_timeout: int = 240,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = base_url
        self.provider_label = provider_label
        self._http_timeout = http_timeout
        self._client = None

    # ── Hook: create OpenAI client ──────────────────────────────────

    def _create_client(self):
        """Create the OpenAI SDK client.

        Override for proxy, custom transport, or extra headers.
        """
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self._http_timeout,
        )

    def _get_client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    # ── Core ────────────────────────────────────────────────────────

    def generate_content(
        self,
        model: str,
        contents: list[Any],
        *,
        system_instruction: str | None = None,
        tools: list[ToolDeclaration] | None = None,
        temperature: float = 0.5,
        response_json: bool = False,
        http_timeout: int | None = None,
    ) -> AIResponse:
        target_model = self.default_model if "gemini" in model.lower() else model
        messages = self._build_messages(
            system_instruction, contents, response_json=response_json,
        )
        openai_tools = self._convert_tools(tools) if tools else None

        params: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"
        if response_json:
            params["response_format"] = {"type": "json_object"}
        if http_timeout is not None:
            params["timeout"] = http_timeout

        response = self._get_client().chat.completions.create(**params)
        return self._parse_response(response, response_json)

    # ── Hook: build messages ────────────────────────────────────────

    def _build_messages(
        self,
        system_instruction: str | None,
        contents: list,
        *,
        response_json: bool = False,
    ) -> list[dict]:
        """Convert agent-kit contents to OpenAI message format.

        Override for multimodal (image_url, etc.).
        """
        json_instruction = f"\n\n{JSON_HINT}" if response_json else ""
        system_content = (
            f"{system_instruction}{json_instruction}"
            if system_instruction
            else json_instruction or None
        )
        messages: list[dict] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if "text" in item:
                    messages.append({
                        "role": "assistant" if role == "model" else role,
                        "content": item["text"],
                    })
                elif "tool_responses" in item:
                    for tr in item["tool_responses"]:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tr["id"],
                            "content": json.dumps(tr.get("result", {})),
                        })
                elif "tool_calls" in item:
                    tcs = [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    } for tc in item["tool_calls"]]
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tcs,
                    })
        return messages

    # ── Hook: convert tools ─────────────────────────────────────────

    def _convert_tools(
        self, tools: list[ToolDeclaration],
    ) -> list[dict]:
        """Convert ToolDeclaration list to OpenAI function-calling format."""
        result = []
        for tool in tools:
            props, required = {}, []
            params = tool.parameters
            for pn, ps in params.get("properties", {}).items():
                prop = {
                    "type": ps.get("type", "string").lower(),
                    "description": ps.get("description", ""),
                }
                for key in ("enum", "minimum", "maximum"):
                    if key in ps:
                        prop[key] = ps[key]
                if "items" in ps:
                    prop["items"] = {
                        "type": ps["items"].get("type", "string").lower(),
                    }
                props[pn] = prop
            required = list(params.get("required", []) or [])
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            })
        return result

    # ── Hook: parse response ────────────────────────────────────────

    def _parse_response(
        self, response, response_json: bool,
    ) -> AIResponse:
        """Parse OpenAI API response into AIResponse.

        Override for custom fields (reasoning_content, etc.).
        """
        msg = response.choices[0].message
        text = (
            self._clean_json_text(msg.content or "")
            if response_json
            else (msg.content or "")
        )
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=tc.function.name, args=args))
        usage = None
        if response.usage:
            usage = Usage(
                total_token_count=response.usage.total_tokens or 0,
                input_token_count=response.usage.prompt_tokens or 0,
                output_token_count=response.usage.completion_tokens or 0,
            )
        return AIResponse(text=text, tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        """Remove markdown code fences from JSON response text."""
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
        return text
