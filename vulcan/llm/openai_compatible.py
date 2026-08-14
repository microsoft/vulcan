"""Adapter for open-source and self-hosted models.

Talks plain HTTP to any server that implements ``POST /v1/chat/completions``:
vLLM, SGLang, Ollama, llama.cpp, text-generation-inference, LM Studio, LiteLLM
proxy, or a hosted OpenAI-compatible service such as Together or Fireworks.

Deliberately built on ``aiohttp`` rather than the OpenAI SDK so that running an
open-weight model needs no optional dependency at all.

Two portability details this adapter handles that a naive OpenAI client does not:

* **Ollama returns tool-call ``arguments`` as a parsed object**, where OpenAI
  returns a JSON string. Both are normalised to a string.
* **Reasoning text lands in different fields.** vLLM and SGLang use
  ``reasoning_content`` (with ``--reasoning-parser``), Ollama uses ``thinking``.
  Both are read.

Native tool calling requires the server to be started for it — for vLLM that is
``--enable-auto-tool-choice --tool-call-parser <parser>``. Without it the model
still emits tool calls as plain text, which VULCAN's ``prompting`` and
``thinking`` trajectory styles parse anyway; those styles are the safe default
for self-hosted models.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.exceptions import LLMError
from .base import LLMClient
from .capabilities import ModelCapabilities, detect


class OpenAICompatibleClient(LLMClient):
    """Chat completions against any OpenAI-compatible HTTP endpoint."""

    provider = "openai_compatible"

    def __init__(self, *, capabilities: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.base_url:
            raise LLMError(
                "provider 'openai_compatible' requires llm.base_url "
                "(for example http://localhost:8000/v1)",
                provider=self.provider,
            )
        self._session = None
        self._capability_overrides = capabilities or {}

    def _caps(self, model: str) -> ModelCapabilities:
        return detect(self.provider, model).merged(self._capability_overrides)

    @property
    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

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

        # max_tokens, never max_completion_tokens: older vLLM, SGLang, Ollama
        # and llama.cpp reject or ignore the newer name.
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None and caps.supports_temperature:
            request["temperature"] = temperature
        if tools and caps.supports_tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice
        if stop and caps.supports_stop:
            request["stop"] = stop
        request.update(self.extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = await self._get_session()
        async with session.post(self._endpoint, json=request, headers=headers) as response:
            body = await response.text()
            if response.status != 200:
                raise LLMError(
                    f"server returned {response.status}",
                    status_code=response.status,
                    body=body[:2000],
                    provider=self.provider,
                )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"server returned non-JSON body: {body[:200]}",
                body=body[:2000],
                provider=self.provider,
            ) from exc

        return self._normalise(payload)

    # ── Translation ──────────────────────────────────────────────────────

    def _normalise(self, payload: dict) -> dict:
        choices = payload.get("choices") or []
        if not choices:
            # Some servers report errors with HTTP 200 and an "error" key.
            error = payload.get("error")
            if error:
                raise LLMError(
                    str(error)[:500], body=json.dumps(payload)[:2000], provider=self.provider
                )
            return self._envelope(finish_reason="error")

        choice = choices[0]
        message = choice.get("message") or {}

        reasoning_content = (
            message.get("reasoning_content")   # vLLM, SGLang
            or message.get("reasoning")        # some proxies
            or message.get("thinking")          # Ollama wire key — do not rename
            or ""
        )
        if not isinstance(reasoning_content, str):
            reasoning_content = ""

        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            tool_calls.append(
                {
                    # Ollama and llama.cpp frequently omit the id; downstream
                    # code maps tool_call_id back to a tool name, so mint one.
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        # Ollama returns a parsed object here, OpenAI a string.
                        "arguments": self._dump_arguments(function.get("arguments", "{}")),
                    },
                }
            )

        usage = payload.get("usage") or {}
        return self._envelope(
            content=self._content_to_text(message.get("content")),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens"),
        )
