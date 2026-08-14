"""Per-model capability detection.

Providers disagree about which request parameters a given model accepts. The
OpenAI reasoning families, for example, want ``max_completion_tokens`` instead
of ``max_tokens`` and reject any ``temperature`` other than 1.

Detection here is a *default*, never a hard rule: every field can be pinned
explicitly in config, which is what you want for a self-hosted model whose name
matches none of these patterns::

    llm:
      provider: openai_compatible
      model: my-org/my-finetune
      capabilities:
        uses_max_completion_tokens: false
        supports_temperature: true
        supports_reasoning: true
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

# OpenAI families that take max_completion_tokens and fix temperature at 1.
# Matches o1/o3/o4-series and the gpt-5 line, including dated and -mini/-nano
# variants, without matching gpt-4o (hence the explicit boundary after "o1").
_OPENAI_REASONING = re.compile(
    r"^(?:o[134](?:[-_]|$)|gpt-5)",
    re.IGNORECASE,
)

# Claude families that take adaptive thinking (4.6 and later, plus the 5 line).
_ANTHROPIC_ADAPTIVE = re.compile(
    r"(opus-(?:5|4-6|4-7|4-8)|sonnet-(?:5|4-6)|fable-5|mythos)",
    re.IGNORECASE,
)

# Claude families that reject temperature / top_p / top_k outright.
_ANTHROPIC_NO_SAMPLING = re.compile(
    r"(opus-(?:5|4-7|4-8)|sonnet-5|fable-5|mythos)",
    re.IGNORECASE,
)

# Open-weight reasoning models commonly served through an OpenAI-compatible
# endpoint. They emit reasoning under `reasoning_content` when the server is
# started with a reasoning parser.
_OPEN_REASONING = re.compile(
    r"(deepseek[-_]?r1|qwq|qwen3|kimi|glm-?4\.?[56]|minimax-?m|magistral|"
    r"exaone.*deep|phi-4-reasoning)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model will accept and produce."""

    #: Send ``max_completion_tokens`` rather than ``max_tokens``.
    uses_max_completion_tokens: bool = False
    #: Model accepts a ``temperature`` other than its default. Current Claude
    #: models reject sampling parameters outright.
    supports_temperature: bool = True
    #: Model can be asked to reason, and may return reasoning text.
    supports_reasoning: bool = False
    #: Model can return reasoning *text* (not just a token count). Public OpenAI
    #: reasoning models bill for reasoning tokens but never return the text.
    returns_reasoning_text: bool = False
    #: Model supports native tool/function calling.
    supports_tools: bool = True
    #: Model accepts stop sequences.
    supports_stop: bool = True
    #: Anthropic only: use ``thinking={"type": "adaptive"}`` plus
    #: ``output_config.effort``. False means the retired
    #: ``{"type": "enabled", "budget_tokens": N}`` form, which returns 400 on
    #: current models and is only valid on Claude 4.5 and older.
    uses_adaptive_thinking: bool = False

    def merged(self, overrides: dict[str, Any] | None) -> "ModelCapabilities":
        """Return a copy with any explicitly configured fields applied."""
        if not overrides:
            return self
        known = {
            k: bool(v)
            for k, v in overrides.items()
            if k in self.__dataclass_fields__ and v is not None
        }
        return replace(self, **known) if known else self


def detect(provider: str, model: str) -> ModelCapabilities:
    """Best-effort capabilities for *model* on *provider*."""
    name = model or ""

    if provider == "anthropic":
        # Claude 4.6 and later take adaptive thinking; the fixed-budget form
        # returns 400 there. Sampling parameters are removed on the newest
        # families (Opus 5/4.8/4.7, Sonnet 5, Fable 5, Mythos 5) and sending
        # one is likewise a 400 — Opus 4.6 and Sonnet 4.6 still accept them.
        adaptive = bool(_ANTHROPIC_ADAPTIVE.search(name))
        return ModelCapabilities(
            supports_reasoning=True,
            returns_reasoning_text=True,
            uses_adaptive_thinking=adaptive,
            supports_temperature=not _ANTHROPIC_NO_SAMPLING.search(name),
        )

    if provider == "gemini":
        # Gemini 2.5+ thinks by default and can return thought summaries.
        return ModelCapabilities(
            supports_reasoning=True,
            returns_reasoning_text=True,
        )

    if provider == "openai":
        if _OPENAI_REASONING.match(name):
            return ModelCapabilities(
                uses_max_completion_tokens=True,
                supports_temperature=False,
                supports_reasoning=True,
                # Reasoning text is summarised at best and not returned through
                # chat completions; treat it as unavailable.
                returns_reasoning_text=False,
            )
        return ModelCapabilities()

    # openai_compatible: vLLM, SGLang, Ollama, llama.cpp, TGI, LM Studio.
    # These accept max_tokens universally; max_completion_tokens is only
    # understood by recent vLLM builds, so never assume it.
    if _OPEN_REASONING.search(name):
        return ModelCapabilities(
            supports_reasoning=True,
            # True only when the server runs with --reasoning-parser; harmless
            # when it does not, since the field is simply absent.
            returns_reasoning_text=True,
        )
    return ModelCapabilities()
