"""OpenAI adapter (GPT).

Requires the ``openai`` extra::

    pip install "vulcan-datagen[openai]"

This is the reference adapter: OpenAI's chat-completions shape *is* VULCAN's
normalised shape, so translation is limited to capability handling — reasoning
models take ``max_completion_tokens`` and reject a custom ``temperature``.

Reasoning text: public OpenAI reasoning models bill for reasoning tokens but do
not return the reasoning text through chat completions, so
``reasoning_content`` comes back empty. Trajectories generated with
``generate_trajectories.tool_call_format: thinking`` therefore need Claude, Gemini, or a self-hosted
reasoning model — see docs/providers.md.
"""

from __future__ import annotations

from typing import Any

from ..core.exceptions import LLMError
from .base import LLMClient
from .capabilities import ModelCapabilities, detect


class OpenAIClient(LLMClient):
    """Chat completions against the OpenAI API."""

    provider = "openai"

    def __init__(self, *, capabilities: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "the openai package is required for provider 'openai'. "
                'Install it with: pip install "vulcan-datagen[openai]"',
                provider=self.provider,
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.api_key or None}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # Retries are owned by LLMClient.call so that backoff, error typing and
        # logging stay identical across providers.
        self._client = AsyncOpenAI(max_retries=0, **client_kwargs)
        self._capability_overrides = capabilities or {}

    def _caps(self, model: str) -> ModelCapabilities:
        return detect(self.provider, model).merged(self._capability_overrides)

    async def aclose(self) -> None:
        await self._client.close()

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

        request: dict[str, Any] = {"model": model, "messages": messages}

        if caps.uses_max_completion_tokens:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens

        if temperature is not None and caps.supports_temperature:
            request["temperature"] = temperature

        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        if stop and caps.supports_stop:
            request["stop"] = stop

        if reasoning and caps.supports_reasoning:
            effort = reasoning.get("effort")
            if effort:
                request["reasoning_effort"] = effort

        request.update(self.extra_body)

        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as exc:
            raise self._as_llm_error(exc) from exc

        return self._normalise(response)

    # ── Translation ──────────────────────────────────────────────────────

    def _normalise(self, response: Any) -> dict:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return self._envelope(finish_reason="error")

        choice = choices[0]
        message = choice.message
        content = getattr(message, "content", "") or ""

        # vLLM, SGLang and some proxies expose reasoning under this field. The
        # OpenAI SDK keeps unknown fields on the model, so read it defensively.
        reasoning_content = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        if not isinstance(reasoning_content, str):
            reasoning_content = ""

        tool_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            tool_calls.append(
                {
                    "id": getattr(call, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(function, "name", "") or "",
                        "arguments": self._dump_arguments(
                            getattr(function, "arguments", "") or "{}"
                        ),
                    },
                }
            )

        usage = getattr(response, "usage", None)
        return self._envelope(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
            finish_reason=getattr(choice, "finish_reason", "") or "stop",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def _as_llm_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "message", None) or str(exc)
        return LLMError(str(exc), status_code=status, body=str(body)[:2000], provider=self.provider)
