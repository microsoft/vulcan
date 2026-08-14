"""Tests for output/select.py and output/reconstruct.py."""

import json
import pytest

from vulcan.steps.output.select import (
    STRIP_FIELDS,
    TRAJ_FIELDS,
    VERIFY_FIELDS,
    _passes_criteria,
)
from vulcan.steps.output.reconstruct import (
    check_chat_template,
    check_tool_call_response_pairs,
    check_no_empty_assistant,
    validate_reconstructed,
)


# ── STRIP_FIELDS contents ─────────────────────────────────────────────────────

class TestStripFields:
    def test_conversation_in_strip_fields(self):
        assert "conversation" in STRIP_FIELDS

    def test_thinking_mode_in_strip_fields(self):
        assert "thinking_mode" in STRIP_FIELDS

    def test_judgment_in_strip_fields(self):
        assert "judgment" in STRIP_FIELDS

    def test_strip_fields_is_frozenset(self):
        assert isinstance(STRIP_FIELDS, frozenset)

    def test_strip_fields_is_union_of_traj_and_verify(self):
        assert STRIP_FIELDS == TRAJ_FIELDS | VERIFY_FIELDS

    def test_traj_fields_subset_of_strip_fields(self):
        assert TRAJ_FIELDS.issubset(STRIP_FIELDS)

    def test_verify_fields_subset_of_strip_fields(self):
        assert VERIFY_FIELDS.issubset(STRIP_FIELDS)


# ── _passes_criteria ──────────────────────────────────────────────────────────

class TestPassesCriteria:
    def test_passes_with_high_score_and_completed(self):
        judgment = {"score": 9, "task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is True

    def test_passes_with_exact_threshold_score(self):
        judgment = {"score": 8, "task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is True

    def test_fails_with_low_score(self):
        judgment = {"score": 6, "task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is False

    def test_fails_with_task_not_completed(self):
        judgment = {"score": 10, "task_completed": False}
        assert _passes_criteria(judgment, min_score=8) is False

    def test_fails_with_none_judgment(self):
        assert _passes_criteria(None, min_score=8) is False

    def test_fails_with_missing_score(self):
        judgment = {"task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is False

    def test_score_zero_fails(self):
        judgment = {"score": 0, "task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is False

    def test_works_with_json_string_judgment(self):
        judgment_str = json.dumps({"score": 9, "task_completed": True})
        assert _passes_criteria(judgment_str, min_score=8) is True

    def test_invalid_json_string_fails(self):
        assert _passes_criteria("{not valid json}", min_score=8) is False

    def test_passes_with_min_score_zero(self):
        judgment = {"score": 0, "task_completed": True}
        assert _passes_criteria(judgment, min_score=0) is True

    def test_task_completed_none_fails(self):
        judgment = {"score": 9, "task_completed": None}
        assert _passes_criteria(judgment, min_score=8) is False

    def test_score_below_boundary(self):
        judgment = {"score": 7, "task_completed": True}
        assert _passes_criteria(judgment, min_score=8) is False


# ── check_chat_template ────────────────────────────────────────────────────────

class TestCheckChatTemplate:
    def test_empty_messages_fails(self):
        ok, reason = check_chat_template([])
        assert ok is False
        assert reason

    def test_system_then_user_then_assistant_passes(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ok, reason = check_chat_template(messages)
        assert ok is True

    def test_user_then_assistant_passes(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        ok, reason = check_chat_template(messages)
        assert ok is True

    def test_user_user_consecutive_fails(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ]
        ok, reason = check_chat_template(messages)
        assert ok is False
        assert "consecutive" in reason.lower() or "user" in reason.lower()

    def test_assistant_assistant_consecutive_fails(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "assistant", "content": "Also this"},
        ]
        ok, reason = check_chat_template(messages)
        assert ok is False

    def test_system_only_fails(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        ok, reason = check_chat_template(messages)
        assert ok is False

    def test_first_non_system_must_be_user(self):
        messages = [
            {"role": "assistant", "content": "Starting without a user"},
        ]
        ok, reason = check_chat_template(messages)
        assert ok is False

    def test_unexpected_role_fails(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "tool output"},  # unexpected non-user/assistant
        ]
        ok, reason = check_chat_template(messages)
        assert ok is False


# ── check_tool_call_response_pairs ────────────────────────────────────────────

class TestCheckToolCallResponsePairs:
    def test_no_tool_calls_passes(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        ok, reason = check_tool_call_response_pairs(messages)
        assert ok is True

    def test_matching_call_and_response_passes(self):
        messages = [
            {"role": "user", "content": "Check my balance"},
            {"role": "assistant", "content": "<tool_call>get_balance()</tool_call>"},
            {"role": "user", "content": "<tool_response>100</tool_response>"},
        ]
        ok, reason = check_tool_call_response_pairs(messages)
        assert ok is True

    def test_tool_call_without_response_fails(self):
        messages = [
            {"role": "user", "content": "Check my balance"},
            {"role": "assistant", "content": "<tool_call>get_balance()</tool_call>"},
            # No following user message with tool_response
        ]
        ok, reason = check_tool_call_response_pairs(messages)
        assert ok is False

    def test_mismatched_count_fails(self):
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": "<tool_call>foo()</tool_call><tool_call>bar()</tool_call>"
            },
            {"role": "user", "content": "<tool_response>result1</tool_response>"},
            # Only 1 response for 2 calls
        ]
        ok, reason = check_tool_call_response_pairs(messages)
        assert ok is False

    def test_multiple_matching_calls_passes(self):
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": "<tool_call>foo()</tool_call><tool_call>bar()</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>r1</tool_response><tool_response>r2</tool_response>"
            },
        ]
        ok, reason = check_tool_call_response_pairs(messages)
        assert ok is True


# ── check_no_empty_assistant ──────────────────────────────────────────────────

class TestCheckNoEmptyAssistant:
    def test_non_empty_assistant_passes(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ok, reason = check_no_empty_assistant(messages)
        assert ok is True

    def test_empty_assistant_fails(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]
        ok, reason = check_no_empty_assistant(messages)
        assert ok is False

    def test_whitespace_only_assistant_fails(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   \n\t  "},
        ]
        ok, reason = check_no_empty_assistant(messages)
        assert ok is False

    def test_user_empty_is_ok(self):
        # check_no_empty_assistant only checks assistant role
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Hi!"},
        ]
        ok, reason = check_no_empty_assistant(messages)
        assert ok is True

    def test_error_message_includes_index(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]
        ok, reason = check_no_empty_assistant(messages)
        assert ok is False
        assert "1" in reason  # index 1 is the empty assistant message


# ── validate_reconstructed ────────────────────────────────────────────────────

class TestValidateReconstructed:
    def test_valid_messages_pass_all_checks(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        ok, reasons = validate_reconstructed(messages)
        assert ok is True
        assert reasons == []

    def test_returns_tuple(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        result = validate_reconstructed(messages)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_invalid_messages_return_reasons(self):
        # Empty messages fail the chat_template check
        ok, reasons = validate_reconstructed([])
        assert ok is False
        assert len(reasons) > 0

    def test_reasons_are_strings(self):
        ok, reasons = validate_reconstructed([])
        assert all(isinstance(r, str) for r in reasons)

    def test_consecutive_users_captured_in_reasons(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Still here"},
        ]
        ok, reasons = validate_reconstructed(messages)
        assert ok is False
        chat_template_failure = any("chat_template" in r for r in reasons)
        assert chat_template_failure
