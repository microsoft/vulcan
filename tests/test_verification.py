"""Tests for verification/verifier.py and verification/criteria.py."""

import json
from types import SimpleNamespace
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from vulcan.verification.criteria import VERIFICATION_CRITERIA
from vulcan.verification.verifier import StageVerifier


def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ── VERIFICATION_CRITERIA ─────────────────────────────────────────────────────

class TestVerificationCriteria:
    def test_is_dict(self):
        assert isinstance(VERIFICATION_CRITERIA, dict)

    def test_has_generate_tools_entry(self):
        assert "generate_tools" in VERIFICATION_CRITERIA

    def test_has_verify_tools_entry(self):
        assert "verify_tools" in VERIFICATION_CRITERIA

    def test_has_generate_trajectories_entry(self):
        assert "generate_trajectories" in VERIFICATION_CRITERIA

    def test_has_judge_trajectories_entry(self):
        assert "judge_trajectories" in VERIFICATION_CRITERIA

    def test_has_generate_states_entry(self):
        assert "generate_states" in VERIFICATION_CRITERIA

    def test_has_sample_arguments_entry(self):
        assert "sample_arguments" in VERIFICATION_CRITERIA

    def test_has_build_graph_entry(self):
        assert "build_graph" in VERIFICATION_CRITERIA

    def test_has_score_sequences_entry(self):
        assert "score_sequences" in VERIFICATION_CRITERIA

    def test_has_generate_queries_entry(self):
        assert "generate_queries" in VERIFICATION_CRITERIA

    def test_each_entry_has_check_items(self):
        for step_name, criteria in VERIFICATION_CRITERIA.items():
            assert "check_items" in criteria, (
                f"Step '{step_name}' criteria missing 'check_items'"
            )

    def test_check_items_are_non_empty_lists(self):
        for step_name, criteria in VERIFICATION_CRITERIA.items():
            items = criteria["check_items"]
            assert isinstance(items, list), f"check_items for '{step_name}' is not a list"
            assert len(items) > 0, f"check_items for '{step_name}' is empty"

    def test_check_items_are_strings(self):
        for step_name, criteria in VERIFICATION_CRITERIA.items():
            for item in criteria["check_items"]:
                assert isinstance(item, str), (
                    f"check_item in '{step_name}' is not a string: {item!r}"
                )

    def test_generate_tools_checks_code_quality(self):
        checks = VERIFICATION_CRITERIA["generate_tools"]["check_items"]
        joined = " ".join(checks).lower()
        assert "code" in joined or "implement" in joined

    def test_generate_trajectories_checks_chat_template(self):
        checks = VERIFICATION_CRITERIA["generate_trajectories"]["check_items"]
        joined = " ".join(checks).lower()
        assert "chat" in joined or "template" in joined or "trajectory" in joined


# ── StageVerifier.__init__ ────────────────────────────────────────────────────

class TestStageVerifierInit:
    def test_reads_model_from_config(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {"model": "gpt-4o"})
        assert verifier.model == "gpt-4o"

    def test_no_model_when_config_omits_it(self):
        """With no `model` key the verifier defers to the client's own model.

        It must NOT fall back to a hardcoded vendor model id — that id would be
        sent verbatim to whichever provider is configured.
        """
        llm = AsyncMock()
        verifier = StageVerifier(llm, {})
        assert verifier.model is None

    def test_reads_max_tokens_from_config(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {"max_tokens": 1024})
        assert verifier.max_tokens == 1024

    def test_default_max_tokens(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {})
        assert verifier.max_tokens == 512

    def test_reads_temperature_from_config(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {"temperature": 0.5})
        assert verifier.temperature == pytest.approx(0.5)

    def test_default_temperature(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {})
        assert verifier.temperature == pytest.approx(0.0)

    def test_stores_llm(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {})
        assert verifier.llm is llm


# ── StageVerifier._extract_relevant_fields ────────────────────────────────────

class TestExtractRelevantFields:
    def _make_verifier(self):
        return StageVerifier(AsyncMock(), {"model": "gpt-4o"})

    def test_generate_tools_extracts_tool_code(self):
        verifier = self._make_verifier()
        item = {"tool_code": "def foo(): pass", "decision": "accepted", "irrelevant": "data"}
        result = verifier._extract_relevant_fields("generate_tools", item)
        assert "tool_code" in result
        assert "irrelevant" not in result

    def test_generate_tools_extracts_decision(self):
        verifier = self._make_verifier()
        item = {"tool_code": "def foo(): pass", "decision": "accepted", "extra": "x"}
        result = verifier._extract_relevant_fields("generate_tools", item)
        assert "decision" in result

    def test_generate_trajectories_extracts_conversation(self):
        verifier = self._make_verifier()
        item = {
            "conversation": [{"role": "user", "content": "hi"}],
            "irrelevant_key": "noise",
        }
        result = verifier._extract_relevant_fields("generate_trajectories", item)
        assert "conversation" in result
        assert "irrelevant_key" not in result

    def test_generate_states_extracts_initial_state(self):
        verifier = self._make_verifier()
        item = {"initial_state": {"balance": 100}, "noise": "noise"}
        result = verifier._extract_relevant_fields("generate_states", item)
        assert "initial_state" in result
        assert "noise" not in result

    def test_unknown_step_returns_partial_item(self):
        verifier = self._make_verifier()
        item = {f"key_{i}": i for i in range(20)}
        result = verifier._extract_relevant_fields("unknown_step", item)
        # Should return at most 10 items (first 10)
        assert len(result) <= 10

    def test_returns_subset_of_item(self):
        verifier = self._make_verifier()
        item = {
            "tool_code": "def foo(): pass",
            "decision": "accepted",
            "category_name": "retail",
            "extra1": "a",
            "extra2": "b",
        }
        result = verifier._extract_relevant_fields("generate_tools", item)
        # All keys in result must be from item
        for k in result:
            assert k in item

    def test_long_string_fields_are_truncated(self):
        verifier = self._make_verifier()
        long_code = "x = 1\n" * 1000  # very long code
        item = {"tool_code": long_code, "decision": "accepted"}
        result = verifier._extract_relevant_fields("generate_tools", item)
        # Truncated value should be shorter than the original
        if isinstance(result.get("tool_code"), str):
            assert len(result["tool_code"]) <= 3100  # 3000 + some margin


# ── StageVerifier.verify (async) ──────────────────────────────────────────────

class TestStageVerifierVerify:
    def _make_verifier_with_response(self, response_text):
        # Mock the REAL LLMClient contract: .call(messages, ...) -> response dict.
        # (An earlier version of this helper mocked the llm as a plain callable
        # returning a string, which StageRunner never provides — so these tests passed
        # while StageVerifier was broken in production.)
        llm = SimpleNamespace(call=AsyncMock(return_value={
            "choices": [{"message": {"content": response_text}}]
        }))
        return StageVerifier(llm, {"model": "gpt-4o"})

    def test_returns_true_when_passed(self):
        response = json.dumps({"passed": True, "issues": [], "confidence": 0.95})
        verifier = self._make_verifier_with_response(response)
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert ok is True

    def test_returns_false_when_failed(self):
        response = json.dumps({"passed": False, "issues": ["missing param"], "confidence": 0.8})
        verifier = self._make_verifier_with_response(response)
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert ok is False

    def test_reason_includes_issues_when_failed(self):
        response = json.dumps({"passed": False, "issues": ["missing param"], "confidence": 0.8})
        verifier = self._make_verifier_with_response(response)
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert "missing param" in reason

    def test_unknown_step_returns_true_no_criteria(self):
        llm = AsyncMock()
        verifier = StageVerifier(llm, {})
        ok, reason = run(verifier.verify("step_without_criteria", {"any": "data"}))
        assert ok is True
        assert "no criteria" in reason.lower()

    def test_llm_error_returns_false(self):
        llm = AsyncMock(side_effect=RuntimeError("llm down"))
        verifier = StageVerifier(llm, {})
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert ok is False
        assert "verifier error" in reason.lower() or "error" in reason.lower()

    def test_empty_response_returns_false(self):
        verifier = self._make_verifier_with_response("")
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert ok is False

    def test_markdown_fenced_json_is_parsed(self):
        response = "```json\n" + json.dumps({"passed": True, "issues": [], "confidence": 1.0}) + "\n```"
        verifier = self._make_verifier_with_response(response)
        ok, reason = run(verifier.verify("generate_tools", {"tool_code": "def f(): pass", "decision": "accepted"}))
        assert ok is True


# ── StageVerifier._parse_verdict ─────────────────────────────────────────────

class TestParseVerdict:
    def _verifier(self):
        return StageVerifier(AsyncMock(), {})

    def test_parses_passed_true(self):
        v = self._verifier()
        ok, reason = v._parse_verdict(json.dumps({"passed": True, "issues": [], "confidence": 1.0}))
        assert ok is True

    def test_parses_passed_false(self):
        v = self._verifier()
        ok, reason = v._parse_verdict(json.dumps({"passed": False, "issues": ["x"], "confidence": 0.5}))
        assert ok is False

    def test_empty_string_fails(self):
        v = self._verifier()
        ok, reason = v._parse_verdict("")
        assert ok is False

    def test_invalid_json_fails(self):
        v = self._verifier()
        ok, reason = v._parse_verdict("this is not json")
        assert ok is False

    def test_json_embedded_in_text_parsed(self):
        v = self._verifier()
        text = 'Here is my verdict: {"passed": true, "issues": [], "confidence": 0.9} done.'
        ok, reason = v._parse_verdict(text)
        assert ok is True

    def test_confidence_included_in_reason(self):
        v = self._verifier()
        ok, reason = v._parse_verdict(json.dumps({"passed": True, "issues": [], "confidence": 0.88}))
        assert "0.88" in reason
