"""Tests for the provider layer.

These cover the translation and normalisation logic — the part most likely to
break silently, because a wrong translation produces a plausible-looking request
that the provider accepts and answers badly.

Everything here runs without the optional provider SDKs installed: the adapters
that need one are constructed with ``object.__new__`` so their translation
methods can be exercised without a network or a vendor package.
"""

import json

import pytest

from vulcan.core.exceptions import LLMError
from vulcan.llm.base import LLMClient
from vulcan.llm.capabilities import ModelCapabilities, detect
from vulcan.llm.factory import PROVIDERS, resolve_llm_config
from vulcan.llm.openai_compatible import OpenAICompatibleClient


# ── Shared helpers on LLMClient ───────────────────────────────────────────────


class TestArgumentCoercion:
    """Tool-call arguments cross the boundary as a JSON string, always."""

    def test_dump_passes_through_a_string(self):
        assert LLMClient._dump_arguments('{"a": 1}') == '{"a": 1}'

    def test_dump_serialises_a_dict(self):
        assert json.loads(LLMClient._dump_arguments({"a": 1})) == {"a": 1}

    def test_dump_handles_none(self):
        assert LLMClient._dump_arguments(None) == "{}"

    def test_dump_survives_unserialisable_input(self):
        assert LLMClient._dump_arguments({"a": object()}) == "{}"

    def test_load_parses_a_string(self):
        assert LLMClient._load_arguments('{"a": 1}') == {"a": 1}

    def test_load_passes_through_a_dict(self):
        assert LLMClient._load_arguments({"a": 1}) == {"a": 1}

    def test_load_survives_malformed_json(self):
        assert LLMClient._load_arguments("{not json") == {}

    def test_load_rejects_non_object_json(self):
        assert LLMClient._load_arguments("[1, 2]") == {}


class TestSplitSystem:
    def test_extracts_and_joins_system_messages(self):
        system, rest = LLMClient._split_system(
            [
                {"role": "system", "content": "a"},
                {"role": "user", "content": "q"},
                {"role": "system", "content": "b"},
            ]
        )
        assert system == "a\n\nb"
        assert [m["role"] for m in rest] == ["user"]

    def test_no_system_message_yields_empty_string(self):
        system, rest = LLMClient._split_system([{"role": "user", "content": "q"}])
        assert system == ""
        assert len(rest) == 1


class TestContentToText:
    def test_plain_string(self):
        assert LLMClient._content_to_text("hi") == "hi"

    def test_none_becomes_empty(self):
        assert LLMClient._content_to_text(None) == ""

    def test_flattens_a_block_list(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert LLMClient._content_to_text(blocks) == "ab"


class TestEnvelope:
    def test_always_has_one_choice_and_usage(self):
        env = LLMClient._envelope(content="hello")
        assert len(env["choices"]) == 1
        assert env["choices"][0]["message"]["content"] == "hello"
        assert env["choices"][0]["message"]["reasoning_content"] == ""
        assert set(env["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    def test_total_tokens_defaults_to_the_sum(self):
        env = LLMClient._envelope(prompt_tokens=3, completion_tokens=4)
        assert env["usage"]["total_tokens"] == 7

    def test_explicit_total_wins(self):
        env = LLMClient._envelope(prompt_tokens=3, completion_tokens=4, total_tokens=99)
        assert env["usage"]["total_tokens"] == 99

    def test_tool_calls_omitted_when_absent(self):
        assert "tool_calls" not in LLMClient._envelope(content="x")["choices"][0]["message"]


# ── Capability detection ──────────────────────────────────────────────────────


class TestCapabilityDetection:
    @pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"])
    def test_openai_reasoning_models_use_completion_tokens(self, model):
        caps = detect("openai", model)
        assert caps.uses_max_completion_tokens
        assert not caps.supports_temperature
        # Public OpenAI reasoning models bill for reasoning but never return it.
        assert not caps.returns_reasoning_text

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    def test_openai_chat_models_take_max_tokens_and_temperature(self, model):
        caps = detect("openai", model)
        assert not caps.uses_max_completion_tokens
        assert caps.supports_temperature

    def test_gpt_4o_is_not_mistaken_for_an_o_series_model(self):
        """The o-series pattern must not swallow gpt-4o."""
        assert not detect("openai", "gpt-4o").uses_max_completion_tokens

    @pytest.mark.parametrize(
        "model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-6"]
    )
    def test_current_claude_uses_adaptive_thinking(self, model):
        caps = detect("anthropic", model)
        assert caps.uses_adaptive_thinking
        assert caps.returns_reasoning_text

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-7"])
    def test_current_claude_rejects_sampling_params(self, model):
        assert not detect("anthropic", model).supports_temperature

    def test_older_claude_still_takes_a_budget_and_temperature(self):
        caps = detect("anthropic", "claude-3-5-sonnet-20241022")
        assert not caps.uses_adaptive_thinking
        assert caps.supports_temperature

    def test_gemini_returns_reasoning_text(self):
        assert detect("gemini", "gemini-2.5-pro").returns_reasoning_text

    def test_open_weight_reasoning_model_detected(self):
        assert detect("openai_compatible", "deepseek-r1-distill-32b").supports_reasoning

    def test_plain_open_weight_model_is_not_a_reasoning_model(self):
        assert not detect("openai_compatible", "meta-llama/Llama-3-8B").supports_reasoning

    def test_self_hosted_never_uses_max_completion_tokens(self):
        """Older vLLM, SGLang, Ollama and llama.cpp only accept max_tokens."""
        for model in ("gpt-5-lookalike", "o1-lookalike", "Qwen/Qwen3-32B"):
            assert not detect("openai_compatible", model).uses_max_completion_tokens

    def test_overrides_win_over_detection(self):
        caps = detect("openai", "gpt-4o").merged({"supports_temperature": False})
        assert not caps.supports_temperature

    def test_unknown_override_keys_are_ignored(self):
        caps = detect("openai", "gpt-4o").merged({"not_a_field": True})
        assert isinstance(caps, ModelCapabilities)

    def test_none_overrides_are_ignored(self):
        base = detect("openai", "gpt-4o")
        assert base.merged({"supports_temperature": None}) == base


# ── openai_compatible normalisation ───────────────────────────────────────────


def _compat_client() -> OpenAICompatibleClient:
    """Build the adapter without running __init__ (which needs a base_url)."""
    client = object.__new__(OpenAICompatibleClient)
    client._capability_overrides = {}
    return client


class TestOpenAICompatibleNormalise:
    def test_plain_text_response(self):
        out = _compat_client()._normalise(
            {
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
        assert out["choices"][0]["message"]["content"] == "hello"
        assert out["usage"]["total_tokens"] == 3

    def test_ollama_object_arguments_are_stringified(self):
        """Ollama returns parsed args; the pipeline json.loads() them downstream."""
        out = _compat_client()._normalise(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "f", "arguments": {"a": 1}}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        call = out["choices"][0]["message"]["tool_calls"][0]
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"a": 1}

    def test_missing_tool_call_id_is_minted(self):
        """Downstream code maps tool_call_id back to a tool name; it must exist."""
        out = _compat_client()._normalise(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        assert out["choices"][0]["message"]["tool_calls"][0]["id"]

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "reasoning"])
    def test_reasoning_read_from_every_known_field(self, field):
        """vLLM/SGLang use reasoning_content, Ollama uses thinking."""
        out = _compat_client()._normalise(
            {"choices": [{"message": {"content": "a", field: "because"}}]}
        )
        assert out["choices"][0]["message"]["reasoning_content"] == "because"

    def test_no_choices_yields_an_error_envelope(self):
        out = _compat_client()._normalise({"choices": []})
        assert out["choices"][0]["finish_reason"] == "error"

    def test_http_200_error_body_raises(self):
        """Some servers report errors with a 200 and an "error" key."""
        with pytest.raises(LLMError):
            _compat_client()._normalise({"error": "model not found"})


# ── Config resolution ─────────────────────────────────────────────────────────


class TestResolveLLMConfig:
    def test_defaults_to_openai(self):
        assert resolve_llm_config({"llm": {"model": "m"}})["provider"] == "openai"

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError):
            resolve_llm_config({"llm": {"provider": "not-a-provider", "model": "m"}})

    def test_retired_gateway_block_raises_a_helpful_error(self):
        with pytest.raises(LLMError, match="gateway"):
            resolve_llm_config({"gateway": {"url": "http://example"}})

    def test_step_overrides_provider_and_model(self):
        config = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "steps": {"judge_trajectories": {"provider": "anthropic", "model": "claude-sonnet-5"}},
        }
        resolved = resolve_llm_config(config, "judge_trajectories")
        assert resolved["provider"] == "anthropic"
        assert resolved["model"] == "claude-sonnet-5"

    def test_step_without_overrides_inherits_the_global_block(self):
        config = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "steps": {"generate_trajectories": {"max_tokens": 100}},
        }
        resolved = resolve_llm_config(config, "generate_trajectories")
        assert resolved["provider"] == "openai"
        assert resolved["model"] == "gpt-4o"

    def test_defaults_model_is_the_fallback_when_llm_model_is_absent(self):
        config = {"llm": {"provider": "openai"}, "defaults": {"model": "fallback"}}
        assert resolve_llm_config(config)["model"] == "fallback"

    def test_api_key_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        config = {"llm": {"provider": "anthropic", "model": "claude-sonnet-5"}}
        assert resolve_llm_config(config)["api_key"] == "from-env"

    def test_unresolved_placeholder_counts_as_unset(self, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"llm": {"provider": "openai", "model": "m", "api_key": "${NOT_SET_ANYWHERE}"}}
        assert resolve_llm_config(config)["api_key"] == ""

    def test_every_provider_has_a_key_and_url_env_mapping(self):
        from vulcan.llm.factory import DEFAULT_KEY_ENV, DEFAULT_URL_ENV

        for provider in PROVIDERS:
            assert provider in DEFAULT_KEY_ENV
            assert provider in DEFAULT_URL_ENV


# ── Retry policy ──────────────────────────────────────────────────────────────


class TestRetryPolicy:
    def test_client_errors_are_not_retried(self):
        from vulcan.llm.base import RETRYABLE_STATUS

        for status in (400, 401, 403, 404, 422):
            assert status not in RETRYABLE_STATUS

    def test_transient_errors_are_retried(self):
        from vulcan.llm.base import RETRYABLE_STATUS

        for status in (408, 429, 500, 502, 503, 529):
            assert status in RETRYABLE_STATUS

    def test_backoff_grows_and_stays_bounded(self):
        delays = [LLMClient._backoff(attempt) for attempt in range(5)]
        assert all(d > 0 for d in delays)
        assert max(delays) <= 60.0

    def test_llm_error_carries_provider_and_status(self):
        err = LLMError("boom", status_code=503, body="b", provider="openai")
        assert err.status_code == 503
        assert err.provider == "openai"
        assert "openai" in str(err)


# ── Provider wire formats ─────────────────────────────────────────────────────


class TestProviderWireFormat:
    """Protocol literals must survive refactors.

    These are the vendors' names, not VULCAN's vocabulary. A repo-wide rename of
    VULCAN's own `thinking` terminology once rewrote them — silently disabling
    Anthropic extended thinking and Ollama reasoning, with every test still
    green because nothing asserted on the wire format. These tests are that
    assertion.
    """

    def _source(self, module) -> str:
        import inspect

        return inspect.getsource(module)

    def test_anthropic_sends_the_thinking_request_key(self):
        from vulcan.llm import anthropic_client

        src = self._source(anthropic_client)
        assert 'request["thinking"]' in src, (
            "Anthropic's request key is `thinking`; `reasoning` is not accepted"
        )

    def test_anthropic_reads_thinking_response_blocks(self):
        from vulcan.llm import anthropic_client

        src = self._source(anthropic_client)
        assert 'block_type == "thinking"' in src
        assert 'getattr(block, "thinking"' in src
        assert 'block_type == "redacted_thinking"' in src

    def test_anthropic_replays_thinking_blocks_in_wire_shape(self):
        """Cached blocks are echoed back on the next turn; the shape must match."""
        from vulcan.llm import anthropic_client

        src = self._source(anthropic_client)
        assert '"type": "thinking"' in src
        assert '"signature"' in src

    def test_anthropic_adaptive_thinking_shape(self):
        from vulcan.llm import anthropic_client

        src = self._source(anthropic_client)
        assert '"type": "adaptive"' in src
        assert '"display": "summarized"' in src, (
            "without display the API returns thinking blocks with empty text"
        )

    def test_openai_compatible_reads_every_reasoning_field(self):
        """vLLM/SGLang use reasoning_content, some proxies reasoning, Ollama thinking."""
        from vulcan.llm import openai_compatible

        src = self._source(openai_compatible)
        for field in ('"reasoning_content"', '"reasoning"', '"thinking"'):
            assert f"message.get({field})" in src, f"stopped reading {field}"

    def test_tool_call_format_values_are_not_wire_names(self):
        """VULCAN's own vocabulary — these are ours to rename, the above are not."""
        from vulcan.cli import _resolve_output_label

        assert _resolve_output_label({"defaults": {"thinking_model": True}}, None) == "reasoning"
        assert _resolve_output_label({"defaults": {"thinking_model": False}}, None) == "native"

    def test_step_level_thinking_model_is_honoured(self):
        """Otherwise reasoning data lands in a folder labelled `native`."""
        from vulcan.cli import _resolve_output_label

        config = {"steps": {"generate_trajectories": {"thinking_model": True}}}
        assert _resolve_output_label(config, None) == "reasoning"
