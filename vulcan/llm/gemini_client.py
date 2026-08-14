"""Google Gemini adapter.

Requires the ``gemini`` extra::

    pip install "vulcan-datagen[gemini]"

Gemini diverges from chat completions more than the other providers:

* Messages are ``contents`` with role ``user`` / ``model``; the system prompt is
  a separate ``system_instruction``.
* Tool calls and results are *parts*, and **function calls carry no id** —
  results are matched back by function *name*. This adapter mints deterministic
  ids so the rest of the pipeline (which resolves ``tool_call_id`` to a tool
  name when building training records) keeps working, and maps them back to
  names on the way in.
* Function-call arguments are objects, where the pipeline expects a JSON string.
* Declaration schemas accept a strict subset of JSON Schema, so schemas are
  sanitised before being sent.
* Thinking is on by default on 2.5-series models; thought *summaries* are
  returned only when explicitly requested.
"""

from __future__ import annotations

from typing import Any

from ..core.exceptions import LLMError
from .base import LLMClient
from .capabilities import ModelCapabilities, detect

# JSON Schema keywords Gemini's function-declaration schema accepts. Anything
# else (notably additionalProperties and $schema) causes a 400.
_ALLOWED_SCHEMA_KEYS = frozenset(
    {
        "type", "format", "title", "description", "nullable", "enum",
        "items", "properties", "required", "minimum", "maximum",
        "minItems", "maxItems", "minLength", "maxLength", "pattern", "example",
        "anyOf", "propertyOrdering",
    }
)

_EFFORT_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}

_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
}


class GeminiClient(LLMClient):
    """Generative Language API against Google Gemini."""

    provider = "gemini"

    def __init__(self, *, capabilities: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "the google-genai package is required for provider 'gemini'. "
                'Install it with: pip install "vulcan-datagen[gemini]"',
                provider=self.provider,
            ) from exc

        self._client = genai.Client(api_key=self.api_key or None)
        self._capability_overrides = capabilities or {}

    def _caps(self, model: str) -> ModelCapabilities:
        return detect(self.provider, model).merged(self._capability_overrides)

    async def _request(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float | None,
        max_tokens: int,
        tools: list[dict] | None,
        tool_choice: str,
        reasoning: dict | None,
        stop: list[str] | None,
    ) -> dict:
        caps = self._caps(model)
        system, rest = self._split_system(messages)

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config["system_instruction"] = system
        if temperature is not None and caps.supports_temperature:
            config["temperature"] = temperature
        if stop and caps.supports_stop:
            config["stop_sequences"] = stop
        if tools:
            config["tools"] = [{"function_declarations": self._to_declarations(tools)}]
            config["tool_config"] = {
                "function_calling_config": {"mode": self._to_mode(tool_choice)}
            }
        if reasoning and caps.supports_reasoning:
            budget = reasoning.get("budget_tokens")
            if budget is None:
                budget = _EFFORT_BUDGETS.get(reasoning.get("effort", "medium"), 8192)
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": int(budget),
            }
        config.update(self.extra_body)

        contents, id_to_name = self._to_contents(rest)

        try:
            response = await self._client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            raise self._as_llm_error(exc) from exc

        return self._normalise(response, id_to_name)

    # ── Request translation ──────────────────────────────────────────────

    def _to_contents(self, messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
        """Translate OpenAI-shaped messages into Gemini ``contents``.

        Returns the contents alongside a ``tool_call_id -> function name`` map,
        needed because Gemini matches function responses by name.
        """
        contents: list[dict] = []
        id_to_name: dict[str, str] = {}

        for message in messages:
            role = message.get("role")

            if role == "tool":
                call_id = message.get("tool_call_id") or ""
                name = id_to_name.get(call_id, call_id)
                part = {
                    "function_response": {
                        "name": name,
                        # Gemini requires an object; wrap scalar tool output.
                        "response": {
                            "result": self._content_to_text(message.get("content"))
                        },
                    }
                }
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue

            if role == "assistant":
                parts: list[dict] = []
                text = self._content_to_text(message.get("content"))
                if text:
                    parts.append({"text": text})
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    if call.get("id"):
                        id_to_name[call["id"]] = name
                    parts.append(
                        {
                            "function_call": {
                                "name": name,
                                "args": self._load_arguments(function.get("arguments")),
                            }
                        }
                    )
                if not parts:
                    continue
                if contents and contents[-1]["role"] == "model":
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": "model", "parts": parts})
                continue

            text = self._content_to_text(message.get("content"))
            if not text:
                continue
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append({"text": text})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        return contents, id_to_name

    @classmethod
    def _to_declarations(cls, tools: list[dict]) -> list[dict]:
        declarations = []
        for tool in tools:
            function = tool.get("function") or tool
            declarations.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", "") or "",
                    "parameters": cls._sanitise_schema(
                        function.get("parameters") or {"type": "object", "properties": {}}
                    ),
                }
            )
        return declarations

    @classmethod
    def _sanitise_schema(cls, schema: Any) -> Any:
        """Strip JSON Schema keywords Gemini's declaration parser rejects."""
        if not isinstance(schema, dict):
            return schema

        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in _ALLOWED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: cls._sanitise_schema(v) for k, v in value.items()}
            elif key in ("items",):
                cleaned[key] = cls._sanitise_schema(value)
            elif key == "anyOf" and isinstance(value, list):
                cleaned[key] = [cls._sanitise_schema(v) for v in value]
            else:
                cleaned[key] = value

        # Gemini requires an explicit type on object schemas that carry properties.
        if "properties" in cleaned and "type" not in cleaned:
            cleaned["type"] = "object"
        return cleaned

    @staticmethod
    def _to_mode(tool_choice: str) -> str:
        return {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice, "AUTO")

    # ── Response translation ─────────────────────────────────────────────

    def _normalise(self, response: Any, id_to_name: dict[str, str]) -> dict:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return self._envelope(finish_reason="error")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []

        for index, part in enumerate(parts):
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                name = getattr(function_call, "name", "") or ""
                # Gemini supplies no call id; mint a stable one and register it
                # so the matching tool result can be routed back by name.
                call_id = f"call_{len(tool_calls)}_{name}" if name else f"call_{index}"
                id_to_name[call_id] = name
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": self._dump_arguments(
                                getattr(function_call, "args", {}) or {}
                            ),
                        },
                    }
                )
                continue

            text = getattr(part, "text", None)
            if not text:
                continue
            # Thought summaries arrive as ordinary text parts flagged `thought`.
            if getattr(part, "thought", False):
                reasoning_parts.append(text)
            else:
                text_parts.append(text)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        # Thinking tokens are billed as output but reported separately.
        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason = getattr(finish_reason, "name", None) or str(finish_reason or "STOP")

        return self._envelope(
            content="".join(text_parts),
            reasoning_content="\n".join(reasoning_parts),
            tool_calls=tool_calls or None,
            finish_reason=_FINISH_REASON_MAP.get(finish_reason, "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens + thoughts_tokens,
            total_tokens=getattr(usage, "total_token_count", None),
        )

    def _as_llm_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if not isinstance(status, int):
            status = None
        return LLMError(str(exc), status_code=status, body=str(exc)[:2000], provider=self.provider)
