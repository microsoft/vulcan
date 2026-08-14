"""Provider-neutral LLM client contract.

Every LLM call in VULCAN goes through :meth:`LLMClient.call`. Adapters translate
to and from their provider's native wire format and return a *normalised*
response with the OpenAI chat-completions shape::

    {
      "choices": [
        {
          "message": {
            "content":           str,           # always present, "" when absent
            "reasoning_content": str,           # "" when the model exposes none
            "tool_calls":        [ ... ] | None # OpenAI shape, see below
          },
          "finish_reason": str,
        }
      ],
      "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
    }

Tool calls are always normalised to the OpenAI form, with ``arguments`` as a
**JSON string** even for providers that return a parsed object::

    {"id": str, "type": "function",
     "function": {"name": str, "arguments": "{\\"k\\": \\"v\\"}"}}

Downstream code re-embeds these objects verbatim into the outgoing message list,
so ids must be stable and non-empty. Adapters for providers that omit call ids
(Gemini) mint deterministic ones.

Adapters accept messages in that same OpenAI shape — including
``{"role": "assistant", "tool_calls": [...]}`` and ``{"role": "tool",
"tool_call_id": ...}`` — and translate them inbound. Callers never construct
provider-native messages.
"""

from __future__ import annotations

import abc
import asyncio
import json
import math
import random
from typing import Any, Iterable, Sequence

from ..core.exceptions import LLMError

# Exponential-backoff defaults, overridable per client.
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 2.0    # seconds before the first retry
DEFAULT_RETRY_MAX_DELAY = 60.0    # ceiling on any single backoff
DEFAULT_REQUEST_TIMEOUT = 600.0   # seconds for one attempt, end to end

# Status codes worth retrying. Everything else (400, 401, 403, 404, 422) is a
# request the provider will reject identically next time, so retrying it only
# burns quota and delays the real error.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class LLMClient(abc.ABC):
    """Base class for provider adapters.

    Subclasses implement :meth:`_request`; this class owns argument
    normalisation, retry/backoff, and error typing so every provider behaves
    identically from the pipeline's point of view.
    """

    #: Short provider id used in config, logs and errors.
    provider: str = "unknown"

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_RETRY_ATTEMPTS,
        extra_body: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        self.default_model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.max_retries = max(1, int(max_retries))
        self.extra_body = dict(extra_body or {})
        #: Applied to every call that does not request reasoning explicitly.
        self.default_reasoning = dict(reasoning) if reasoning else None

    # ── Public API ───────────────────────────────────────────────────────

    async def call(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 4096,
        thinking_model: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        reasoning: dict | None = None,
        stop: Sequence[str] | None = None,
        **_ignored: Any,
    ) -> dict:
        """Send a chat completion request and return a normalised response.

        Args:
            messages:       OpenAI-shaped message list. Roles: system, user,
                            assistant (optionally with ``tool_calls``), tool.
            model:          overrides the client's default model.
            temperature:    sampling temperature, or None to let the provider
                            decide. Silently dropped for models that reject it.
            max_tokens:     cap on generated tokens (output only).
            thinking_model: hint that this model reasons before answering. Its
                            only portable effect is enabling reasoning output
                            where the provider supports it.
            tools:          OpenAI tool definitions, ``[{"type": "function",
                            "function": {...}}]``.
            tool_choice:    "auto", "none", or "required".
            reasoning:      provider-neutral reasoning request, e.g.
                            ``{"effort": "medium"}`` or ``{"budget_tokens": 4096}``.
            stop:           stop sequences; omitted from the request when empty.

        Returns:
            A normalised response dict (see module docstring). Always has at
            least one entry in ``choices``.

        Raises:
            LLMError: when every attempt fails.

        ``**_ignored`` absorbs arguments retired from the original single-endpoint
        client (``tier``, ``n``) so older call sites do not break; they have no
        effect.
        """
        model = model or self.default_model
        if not model:
            raise LLMError(
                "no model configured: set llm.model in your config",
                provider=self.provider,
            )

        # Precedence: the per-call argument, then the client's configured
        # `llm.reasoning`, then a default effort when the step asked for a
        # thinking model without saying how much to think.
        if reasoning is None:
            reasoning = self.default_reasoning
        if reasoning is None and thinking_model:
            reasoning = {"effort": "medium"}

        stop_list = [s for s in (stop or []) if s] or None

        last_error = "unknown error"
        last_status: int | None = None

        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(
                    self._request(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice,
                        reasoning=reasoning,
                        stop=stop_list,
                    ),
                    timeout=self.request_timeout,
                )
            except asyncio.TimeoutError:
                last_error = f"request exceeded {self.request_timeout}s"
                last_status = None
            except LLMError as exc:
                last_error = exc.body or str(exc)
                last_status = exc.status_code
                # A request the provider will reject identically next time.
                if last_status is not None and last_status not in RETRYABLE_STATUS:
                    raise
            except Exception as exc:  # transport / SDK-level failure
                last_error = f"{type(exc).__name__}: {exc}"
                last_status = None

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self._backoff(attempt))

        raise LLMError(
            f"all {self.max_retries} attempts failed: {last_error}",
            status_code=last_status,
            body=last_error[:2000],
            provider=self.provider,
        )

    async def aclose(self) -> None:
        """Release any transport resources. Safe to call more than once."""

    # ── Subclass hook ────────────────────────────────────────────────────

    @abc.abstractmethod
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
        """Perform one attempt. Raise LLMError on a provider error response."""

    # ── Helpers shared by adapters ───────────────────────────────────────

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter, so parallel workers do not sync up."""
        base = min(DEFAULT_RETRY_BASE_DELAY * math.pow(2, attempt), DEFAULT_RETRY_MAX_DELAY)
        return base * (0.5 + random.random() / 2.0)

    @staticmethod
    def _envelope(
        *,
        content: str = "",
        reasoning_content: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> dict:
        """Build the normalised response every adapter returns."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
            "reasoning_content": reasoning_content or "",
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    total_tokens
                    if total_tokens is not None
                    else prompt_tokens + completion_tokens
                ),
            },
        }

    @staticmethod
    def _split_system(messages: Iterable[dict]) -> tuple[str, list[dict]]:
        """Split leading system messages out of an OpenAI-shaped message list.

        Anthropic and Gemini take the system prompt as a separate top-level
        field rather than a message role.
        """
        system_parts: list[str] = []
        rest: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                text = msg.get("content") or ""
                if text:
                    system_parts.append(text if isinstance(text, str) else str(text))
            else:
                rest.append(msg)
        return "\n\n".join(system_parts), rest

    @staticmethod
    def _dump_arguments(value: Any) -> str:
        """Coerce tool-call arguments to the JSON *string* the pipeline expects."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value if value is not None else {})
        except (TypeError, ValueError):
            return "{}"

    @staticmethod
    def _load_arguments(value: Any) -> dict:
        """Coerce tool-call arguments to the dict providers expect."""
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        try:
            loaded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Flatten a message ``content`` field to plain text.

        Accepts the plain strings VULCAN uses internally, and tolerates the
        content-block lists a caller might pass through from another provider.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out: list[str] = []
            for block in content:
                if isinstance(block, str):
                    out.append(block)
                elif isinstance(block, dict):
                    out.append(block.get("text") or block.get("content") or "")
            return "".join(out)
        return str(content)
