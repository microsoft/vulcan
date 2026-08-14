"""Exception hierarchy for VULCAN.

All pipeline-level errors derive from VulcanError so callers can
catch broadly (VulcanError) or narrowly (LLMError, StepError, …).
"""

from __future__ import annotations

from typing import Any


class VulcanError(Exception):
    """Base class for all VULCAN pipeline errors."""


class StepError(VulcanError):
    """Raised when a pipeline step fails to process an item.

    Attributes:
        step_name: name of the step that failed (e.g. "generate_states")
        item_id:   _item_id of the offending input item
        message:   human-readable description
        cause:     original exception, if any
    """

    def __init__(
        self,
        step_name: str,
        item_id: str,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        self.step_name = step_name
        self.item_id = item_id
        self.message = message
        self.cause = cause
        super().__init__(f"[{step_name}] item={item_id}: {message}")

    def __repr__(self) -> str:
        return (
            f"StepError(step_name={self.step_name!r}, item_id={self.item_id!r}, "
            f"message={self.message!r}, cause={self.cause!r})"
        )


class LLMError(VulcanError):
    """Raised when an LLM provider fails after all retries are exhausted.

    Covers non-2xx responses, transport failures, and request timeouts, for
    every provider (OpenAI, Anthropic, Gemini, or an OpenAI-compatible server).

    Attributes:
        status_code: HTTP status code, or None for transport/timeout errors
        body:        response body / error text, truncated by the caller
        provider:    which provider raised it, for multi-provider runs
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str = "",
        provider: str = "",
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.provider = provider
        bits = []
        if provider:
            bits.append(provider)
        if status_code:
            bits.append(f"HTTP {status_code}")
        detail = f" ({', '.join(bits)})" if bits else ""
        super().__init__(f"LLMError{detail}: {message}")


class ValidationError(VulcanError):
    """Raised when a generated output fails schema or content validation.

    Attributes:
        field:   which field/key failed validation (optional)
        value:   the offending value (optional)
    """

    def __init__(
        self,
        message: str,
        *,
        field: str = "",
        value: Any = None,
    ) -> None:
        self.field = field
        self.value = value
        super().__init__(message)


class EnvironmentError(VulcanError):
    """Raised when generated environment code fails to execute, import, or
    produce correct tool outputs.

    Attributes:
        env_name:  name of the environment category
        tool_name: name of the tool being called, if applicable
    """

    def __init__(
        self,
        message: str,
        *,
        env_name: str = "",
        tool_name: str = "",
    ) -> None:
        self.env_name = env_name
        self.tool_name = tool_name
        super().__init__(message)


class ParseError(VulcanError):
    """Raised when LLM output cannot be parsed into the expected format.

    Attributes:
        raw:  the raw string that could not be parsed (truncated to 500 chars
              in the repr to keep logs readable)
    """

    def __init__(self, message: str, *, raw: str = "") -> None:
        self.raw = raw
        super().__init__(message)

    def __repr__(self) -> str:
        snippet = self.raw[:200] + "…" if len(self.raw) > 200 else self.raw
        return f"ParseError({self.args[0]!r}, raw={snippet!r})"
