"""Build LLM clients from configuration.

Config lives under a single ``llm:`` block::

    llm:
      provider: openai              # openai | anthropic | gemini | openai_compatible
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}    # or omit and let the provider default apply
      base_url: null                # required for openai_compatible
      request_timeout: 600
      max_retries: 3
      reasoning: null               # {effort: low|medium|high} or {budget_tokens: N}
      capabilities: {}              # pin detection results, see capabilities.py
      extra_body: {}                # passed through to the provider verbatim

Any step may override ``provider``, ``model``, ``base_url`` or ``reasoning``,
which is how you run generation on one model and judging on another::

    steps:
      generate_trajectories:
        provider: openai_compatible
        base_url: http://localhost:8000/v1
        model: Qwen/Qwen3-32B
      judge_trajectories:
        provider: anthropic
        model: claude-sonnet-4-5

Clients are cached per distinct (provider, model, base_url) so a pipeline run
opens one connection pool per backend rather than one per step.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..core.exceptions import LLMError
from .base import LLMClient

#: Environment variable consulted for each provider when the config sets no key.
DEFAULT_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai_compatible": ("VULCAN_API_KEY", "OPENAI_API_KEY"),
}

#: Environment variable consulted for a provider's base URL.
DEFAULT_URL_ENV = {
    "openai": ("OPENAI_BASE_URL",),
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "gemini": (),
    "openai_compatible": ("VULCAN_BASE_URL", "OPENAI_BASE_URL"),
}

PROVIDERS = ("openai", "anthropic", "gemini", "openai_compatible")

# (provider, model, base_url, reasoning) -> client
_CLIENT_CACHE: dict[tuple[str, str, str, str], LLMClient] = {}


def load_dotenv(path: str | None = None) -> None:
    """Load ``KEY=value`` pairs from a .env file into the environment.

    Deliberately minimal — no dependency, no interpolation, no export syntax.
    Existing environment variables always win, so an explicit ``export`` in the
    shell overrides the file. Missing files are ignored.
    """
    candidates = [path] if path else [os.path.join(os.getcwd(), ".env")]
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        with open(candidate, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def _expand(value: Any) -> Any:
    """Expand ``${VAR}`` in a string, treating unresolved placeholders as unset."""
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    return "" if expanded.startswith("${") else expanded


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def resolve_llm_config(config: dict, step_name: str | None = None) -> dict:
    """Merge the global ``llm:`` block with any per-step overrides."""
    if "llm" not in config and "gateway" in config:
        raise LLMError(
            "config uses the retired 'gateway:' block. Rename it to 'llm:' and set "
            "'provider:' to one of: " + ", ".join(PROVIDERS)
        )

    settings = dict(config.get("llm") or {})

    if step_name:
        step_config = (config.get("steps") or {}).get(step_name) or {}
        for key in ("provider", "model", "base_url", "reasoning", "capabilities", "extra_body"):
            if step_config.get(key) is not None:
                settings[key] = step_config[key]
        # `defaults.model` and `steps.<step>.model` remain the documented way to
        # choose a model per step; treat them as the model for this client.
        if step_config.get("model"):
            settings["model"] = step_config["model"]

    if not settings.get("model"):
        settings["model"] = (config.get("defaults") or {}).get("model", "")

    provider = (settings.get("provider") or "openai").strip()
    if provider not in PROVIDERS:
        raise LLMError(
            f"unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}"
        )
    settings["provider"] = provider

    settings["api_key"] = _expand(settings.get("api_key")) or _first_env(
        DEFAULT_KEY_ENV[provider]
    )
    settings["base_url"] = _expand(settings.get("base_url")) or _first_env(
        DEFAULT_URL_ENV[provider]
    )
    return settings


def create_client(config: dict, step_name: str | None = None) -> LLMClient:
    """Return a cached client for *config*, honouring per-step overrides."""
    settings = resolve_llm_config(config, step_name)
    provider = settings["provider"]
    model = settings.get("model") or ""
    base_url = settings.get("base_url") or ""

    # Reasoning is part of the key: two steps on the same model but different
    # reasoning settings must not silently share one client.
    reasoning = settings.get("reasoning")
    cache_key = (
        provider,
        model,
        base_url,
        json.dumps(reasoning, sort_keys=True) if reasoning else "",
    )
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not model:
        raise LLMError(
            "no model configured. Set llm.model (or defaults.model) in your config."
        )
    if provider != "openai_compatible" and not settings.get("api_key"):
        env_names = " or ".join(DEFAULT_KEY_ENV[provider])
        raise LLMError(
            f"no API key found for provider {provider!r}. "
            f"Set {env_names} in your environment or in a .env file "
            f"(see .env.example), or set llm.api_key in your config."
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": settings.get("api_key", ""),
        "base_url": base_url or None,
        "request_timeout": float(settings.get("request_timeout", 600)),
        "max_retries": int(settings.get("max_retries", 3)),
        "extra_body": settings.get("extra_body") or {},
        "capabilities": settings.get("capabilities") or {},
        "reasoning": settings.get("reasoning") or None,
    }

    if provider == "openai":
        from .openai_client import OpenAIClient

        client: LLMClient = OpenAIClient(**kwargs)
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(**kwargs)
    elif provider == "gemini":
        from .gemini_client import GeminiClient

        client = GeminiClient(**kwargs)
    else:
        from .openai_compatible import OpenAICompatibleClient

        client = OpenAICompatibleClient(**kwargs)

    _CLIENT_CACHE[cache_key] = client
    return client


async def close_clients() -> None:
    """Close every cached client. Call once at the end of a run."""
    for client in list(_CLIENT_CACHE.values()):
        try:
            await client.aclose()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    _CLIENT_CACHE.clear()
