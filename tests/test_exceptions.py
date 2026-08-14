"""Tests for the VULCAN exception hierarchy."""

import pytest

from vulcan.core.exceptions import (
    VulcanError,
    StepError,
    LLMError,
    ValidationError,
    EnvironmentError,
    ParseError,
)


# ── VulcanError base ──────────────────────────────────────────────────────────

class TestVulcanError:
    def test_is_exception(self):
        err = VulcanError("base error")
        assert isinstance(err, Exception)

    def test_message(self):
        err = VulcanError("something went wrong")
        assert "something went wrong" in str(err)


# ── StepError ────────────────────────────────────────────────────────────────

class TestStepError:
    def test_extends_vulcan_error(self):
        err = StepError("generate_tools", "item_001", "code gen failed")
        assert isinstance(err, VulcanError)

    def test_stores_step_name(self):
        err = StepError("generate_states", "item_42", "state failed")
        assert err.step_name == "generate_states"

    def test_stores_item_id(self):
        err = StepError("generate_trajectories", "item_99", "trajectory failed")
        assert err.item_id == "item_99"

    def test_stores_message(self):
        err = StepError("judge_trajectories", "item_5", "verification error")
        assert err.message == "verification error"

    def test_cause_defaults_to_none(self):
        err = StepError("build_graph", "item_1", "graph failed")
        assert err.cause is None

    def test_stores_cause(self):
        original = ValueError("original error")
        err = StepError("score_sequences", "item_7", "eval failed", cause=original)
        assert err.cause is original

    def test_str_includes_step_name_and_item_id(self):
        err = StepError("generate_tools", "item_001", "code gen failed")
        s = str(err)
        assert "generate_tools" in s
        assert "item_001" in s

    def test_repr(self):
        err = StepError("generate_tools", "item_001", "code gen failed")
        r = repr(err)
        assert "StepError" in r
        assert "generate_tools" in r

    def test_can_be_caught_as_vulcan_error(self):
        with pytest.raises(VulcanError):
            raise StepError("generate_tools", "item_1", "failed")


# ── LLMError ─────────────────────────────────────────────────────────────

class TestLLMError:
    def test_extends_vulcan_error(self):
        err = LLMError("timeout")
        assert isinstance(err, VulcanError)

    def test_status_code_defaults_to_none(self):
        err = LLMError("network failure")
        assert err.status_code is None

    def test_stores_status_code(self):
        err = LLMError("not found", status_code=404)
        assert err.status_code == 404

    def test_stores_body(self):
        err = LLMError("server error", status_code=500, body="Internal Server Error")
        assert err.body == "Internal Server Error"

    def test_body_defaults_to_empty(self):
        err = LLMError("timeout")
        assert err.body == ""

    def test_str_includes_status_code_when_set(self):
        err = LLMError("not found", status_code=404)
        assert "404" in str(err)

    def test_str_no_status_code_when_none(self):
        err = LLMError("network failure")
        s = str(err)
        assert "None" not in s
        assert "network failure" in s

    def test_can_be_caught_as_vulcan_error(self):
        with pytest.raises(VulcanError):
            raise LLMError("llm down")


# ── ValidationError ───────────────────────────────────────────────────────────

class TestValidationError:
    def test_extends_vulcan_error(self):
        err = ValidationError("invalid field")
        assert isinstance(err, VulcanError)

    def test_field_defaults_to_empty(self):
        err = ValidationError("validation failed")
        assert err.field == ""

    def test_stores_field(self):
        err = ValidationError("bad value", field="category_name")
        assert err.field == "category_name"

    def test_value_defaults_to_none(self):
        err = ValidationError("bad value")
        assert err.value is None

    def test_stores_value(self):
        err = ValidationError("bad value", field="score", value=999)
        assert err.value == 999

    def test_can_be_caught_as_vulcan_error(self):
        with pytest.raises(VulcanError):
            raise ValidationError("invalid")


# ── EnvironmentError ──────────────────────────────────────────────────────────

class TestEnvironmentError:
    def test_extends_vulcan_error(self):
        err = EnvironmentError("env exec failed")
        assert isinstance(err, VulcanError)

    def test_env_name_defaults_to_empty(self):
        err = EnvironmentError("failed")
        assert err.env_name == ""

    def test_stores_env_name(self):
        err = EnvironmentError("failed", env_name="retail")
        assert err.env_name == "retail"

    def test_tool_name_defaults_to_empty(self):
        err = EnvironmentError("failed")
        assert err.tool_name == ""

    def test_stores_tool_name(self):
        err = EnvironmentError("tool crash", env_name="airline", tool_name="book_flight")
        assert err.tool_name == "book_flight"

    def test_can_be_caught_as_vulcan_error(self):
        with pytest.raises(VulcanError):
            raise EnvironmentError("env error")


# ── ParseError ────────────────────────────────────────────────────────────────

class TestParseError:
    def test_extends_vulcan_error(self):
        err = ParseError("parse failed")
        assert isinstance(err, VulcanError)

    def test_raw_defaults_to_empty(self):
        err = ParseError("could not parse")
        assert err.raw == ""

    def test_stores_raw(self):
        err = ParseError("bad json", raw='{"broken": }')
        assert err.raw == '{"broken": }'

    def test_repr_truncates_long_raw(self):
        long_raw = "x" * 300
        err = ParseError("parse failed", raw=long_raw)
        r = repr(err)
        assert "ParseError" in r
        assert len(r) < 500  # Verify truncation is happening

    def test_repr_keeps_short_raw_intact(self):
        err = ParseError("parse failed", raw="short text")
        r = repr(err)
        assert "short text" in r

    def test_can_be_caught_as_vulcan_error(self):
        with pytest.raises(VulcanError):
            raise ParseError("parse error")


# ── All as VulcanError ────────────────────────────────────────────────────────

class TestAllExceptionsCatchableAsVulcanError:
    @pytest.mark.parametrize("exc", [
        VulcanError("base"),
        StepError("step", "id", "msg"),
        LLMError("llm"),
        ValidationError("validation"),
        EnvironmentError("env"),
        ParseError("parse"),
    ])
    def test_catchable_as_vulcan_error(self, exc):
        with pytest.raises(VulcanError):
            raise exc
