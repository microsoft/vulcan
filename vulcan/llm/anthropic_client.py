"""Anthropic adapter (Claude).

Requires the ``anthropic`` extra::

    pip install "vulcan-datagen[anthropic]"

Anthropic's Messages API differs from chat completions in four ways that matter
here, all handled below:

* The system prompt is a top-level ``system`` field, not a message role.
* Tool calls and tool results are *content blocks*, not a separate ``tool``
  role. Consecutive tool results must be merged into a single user message.
* Tool-call inputs are objects, where the pipeline expects a JSON string.
* Thinking returns ``thinking`` blocks carrying a ``signature`` that must be
  echoed back on the following turn when tool use is in play.

The thinking request shape differs by model generation, which this adapter
resolves from :mod:`vulcan.llm.capabilities`: Claude 4.6 and later take
``{"type": "adaptive"}`` plus ``output_config.effort``, while the older
fixed-budget form (``{"type": "enabled", "budget_tokens": N}``) returns 400 on
them and is sent only to Claude 4.5 and earlier. Current models also reject
``temperature`` outright, so it is omitted rather than passed through.

That last point is the subtle one. VULCAN's trajectory loop rebuilds each
assistant message from ``content`` + ``tool_calls`` alone, so signatures would
be dropped between turns and the API would reject the next request. This
adapter keeps a bounded cache of thinking blocks keyed by tool-call id and
re-attaches them on the way back in, which makes extended thinking work through
the existing plumbing without changing it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ..core.exceptions import LLMError
from .base import LLMClient
from .capabilities import ModelCapabilities, detect

# Legacy fixed-budget thinking (Claude 4.5 and older). Anthropic rejects a
# budget below this and requires max_tokens to exceed it.
_MIN_THINKING_BUDGET = 1024
_DEFAULT_THINKING_BUDGET = 4096
_EFFORT_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384}

# Effort levels accepted by output_config on adaptive-thinking models.
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

# How many turns of thinking blocks to retain. Trajectories are bounded by
# generate_trajectories.max_turns, so a few hundred entries covers concurrent workers with
# room to spare.
_THINKING_CACHE_SIZE = 512

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


class AnthropicClient(LLMClient):
    """Messages API against Anthropic."""

    provider = "anthropic"

    def __init__(self, *, capabilities: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "the anthropic package is required for provider 'anthropic'. "
                'Install it with: pip install "vulcan-datagen[anthropic]"',
                provider=self.provider,
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.api_key or None, "max_retries": 0}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = AsyncAnthropic(**client_kwargs)
        self._capability_overrides = capabilities or {}
        # tool_call_id -> the thinking blocks that accompanied it.
        self._thinking_cache: OrderedDict[str, list[dict]] = OrderedDict()

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
        system, rest = self._split_system(messages)

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._to_anthropic_messages(rest),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = self._to_anthropic_tools(tools)
            request["tool_choice"] = self._to_anthropic_tool_choice(tool_choice)
        if stop and caps.supports_stop:
            request["stop_sequences"] = stop

        thinking = self._thinking_config(reasoning, max_tokens, caps)
        if thinking:
            request["thinking"] = thinking          # Anthropic wire key — do not rename
            if caps.uses_adaptive_thinking:
                # Depth is controlled by effort, not a token budget.
                effort = (reasoning or {}).get("effort")
                if effort in _EFFORT_LEVELS:
                    request["output_config"] = {"effort": effort}

        # Sampling parameters are rejected on current Claude models, and
        # extended thinking requires the default temperature on older ones.
        if temperature is not None and caps.supports_temperature and not thinking:
            request["temperature"] = temperature

        request.update(self.extra_body)

        try:
            response = await self._client.messages.create(**request)
        except Exception as exc:
            raise self._as_llm_error(exc) from exc

        return self._normalise(response)

    # ── Request translation ──────────────────────────────────────────────

    @staticmethod
    def _thinking_config(
        reasoning: dict | None, max_tokens: int, caps: ModelCapabilities
    ) -> dict | None:
        """Build the ``thinking`` request field for this model generation.

        Claude 4.6 and later take adaptive thinking; the fixed-budget form was
        removed and returns 400 there. Older models still take a budget.

        ``display`` is set to ``summarized`` because the API default is
        ``omitted`` — without it, thinking blocks come back with empty text and
        every reasoning-style trajectory this pipeline produces would carry an
        empty ``<think>`` block.
        """
        if not reasoning or not caps.supports_reasoning:
            return None

        if caps.uses_adaptive_thinking:
            return {"type": "adaptive", "display": "summarized"}

        budget = reasoning.get("budget_tokens")
        if budget is None:
            budget = _EFFORT_BUDGETS.get(reasoning.get("effort", "medium"), _DEFAULT_THINKING_BUDGET)
        budget = int(budget)
        if budget <= 0:
            return None
        budget = max(budget, _MIN_THINKING_BUDGET)
        # max_tokens must leave room for a response beyond the thinking budget.
        if budget >= max_tokens:
            budget = max_tokens - 512
        if budget < _MIN_THINKING_BUDGET:
            return None
        return {"type": "enabled", "budget_tokens": budget}

    def _to_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """Translate OpenAI-shaped messages into Anthropic content blocks.

        Consecutive ``tool`` messages collapse into one user message, and
        consecutive same-role messages are merged, because Anthropic requires
        strictly alternating roles.
        """
        out: list[dict] = []

        for message in messages:
            role = message.get("role")

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") or "",
                    "content": self._content_to_text(message.get("content")) or "(no output)",
                }
                # Attach to the open user turn when there is one.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                blocks: list[dict] = []

                # Re-attach cached thinking blocks so signatures survive the
                # round trip; Anthropic rejects tool results otherwise.
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    cached = self._thinking_cache.get(tool_calls[0].get("id", ""))
                    if cached:
                        blocks.extend(cached)

                text = self._content_to_text(message.get("content"))
                if text:
                    blocks.append({"type": "text", "text": text})

                for call in tool_calls:
                    function = call.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id") or "",
                            "name": function.get("name", ""),
                            "input": self._load_arguments(function.get("arguments")),
                        }
                    )

                if not blocks:
                    continue
                if out and out[-1]["role"] == "assistant":
                    out[-1]["content"].extend(blocks)
                else:
                    out.append({"role": "assistant", "content": blocks})
                continue

            # user (and anything unrecognised, treated as user input)
            text = self._content_to_text(message.get("content"))
            if not text:
                continue
            block = {"type": "text", "text": text}
            if out and out[-1]["role"] == "user":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

        # The conversation must open with a user turn.
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    @staticmethod
    def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
        """OpenAI ``{"type": "function", "function": {...}}`` -> Anthropic flat form."""
        converted = []
        for tool in tools:
            function = tool.get("function") or tool
            converted.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", "") or "",
                    "input_schema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return converted

    @staticmethod
    def _to_anthropic_tool_choice(tool_choice: str) -> dict:
        return {
            "auto": {"type": "auto"},
            "required": {"type": "any"},
            "none": {"type": "none"},
        }.get(tool_choice, {"type": "auto"})

    # ── Response translation ─────────────────────────────────────────────

    def _normalise(self, response: Any) -> dict:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        thinking_blocks: list[dict] = []
        tool_calls: list[dict] = []

        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", "")

            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")

            elif block_type == "thinking":   # Anthropic wire type — do not rename
                thinking = getattr(block, "thinking", "") or ""
                reasoning_parts.append(thinking)
                # Preserved verbatim, signature included, for the next turn.
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": getattr(block, "signature", "") or "",
                    }
                )

            elif block_type == "redacted_thinking":
                thinking_blocks.append(
                    {"type": "redacted_thinking", "data": getattr(block, "data", "") or ""}
                )

            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", "") or "",
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", "") or "",
                            "arguments": self._dump_arguments(getattr(block, "input", {})),
                        },
                    }
                )

        if thinking_blocks and tool_calls:
            self._remember_thinking(tool_calls[0]["id"], thinking_blocks)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

        return self._envelope(
            content="".join(text_parts),
            reasoning_content="\n".join(p for p in reasoning_parts if p),
            tool_calls=tool_calls or None,
            finish_reason=_STOP_REASON_MAP.get(getattr(response, "stop_reason", "") or "", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _remember_thinking(self, tool_call_id: str, blocks: list[dict]) -> None:
        if not tool_call_id:
            return
        self._thinking_cache[tool_call_id] = blocks
        self._thinking_cache.move_to_end(tool_call_id)
        while len(self._thinking_cache) > _THINKING_CACHE_SIZE:
            self._thinking_cache.popitem(last=False)

    def _as_llm_error(self, exc: Exception) -> LLMError:
        status = getattr(exc, "status_code", None)
        return LLMError(str(exc), status_code=status, body=str(exc)[:2000], provider=self.provider)
