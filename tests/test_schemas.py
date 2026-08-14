"""Tests for the pydantic step-IO schemas."""

import pytest

try:
    import pydantic
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYDANTIC_AVAILABLE,
    reason="pydantic not installed"
)

from vulcan.schemas.step_io import (
    BaseStepInput,
    GenerateToolsInput,
    GenerateToolsOutput,
    GenerateTrajectoriesInput,
    GenerateTrajectoriesOutput,
    JudgeTrajectoriesOutput,
)


# ── BaseStepInput ─────────────────────────────────────────────────────────────

class TestBaseStepInput:
    def test_requires_category_name(self):
        with pytest.raises(Exception):
            BaseStepInput()

    def test_valid_with_category_name(self):
        item = BaseStepInput(category_name="retail")
        assert item.category_name == "retail"

    def test_item_id_optional(self):
        item = BaseStepInput(category_name="retail")
        assert item.item_id is None

    def test_item_id_via_alias(self):
        item = BaseStepInput.model_validate({"category_name": "retail", "_item_id": "abc123"})
        assert item.item_id == "abc123"

    def test_extra_fields_allowed(self):
        # model_config extra=allow means extra keys pass through
        item = BaseStepInput.model_validate({"category_name": "retail", "extra_key": "extra_val"})
        assert item.category_name == "retail"

    def test_model_config_extra_allow(self):
        from pydantic import BaseModel
        cfg = BaseStepInput.model_config
        assert cfg.get("extra") == "allow"


# ── GenerateTrajectoriesInput ──────────────────────────────────────────────────────────────

class TestGenerateTrajectoriesInput:
    def _make_valid(self, **overrides):
        base = {
            "category_name": "retail",
            "tool_code": "def get_balance(): return 100",
            "initial_state": {"balance": 100},
            "query": "What is my balance?",
            "mode": "tools_only",
        }
        base.update(overrides)
        return base

    def test_valid_input(self):
        item = GenerateTrajectoriesInput.model_validate(self._make_valid())
        assert item.category_name == "retail"
        assert item.mode == "tools_only"

    def test_requires_tool_code(self):
        data = self._make_valid()
        del data["tool_code"]
        with pytest.raises(Exception):
            GenerateTrajectoriesInput.model_validate(data)

    def test_requires_initial_state(self):
        data = self._make_valid()
        del data["initial_state"]
        with pytest.raises(Exception):
            GenerateTrajectoriesInput.model_validate(data)

    def test_requires_query(self):
        data = self._make_valid()
        del data["query"]
        with pytest.raises(Exception):
            GenerateTrajectoriesInput.model_validate(data)

    def test_requires_mode(self):
        data = self._make_valid()
        del data["mode"]
        with pytest.raises(Exception):
            GenerateTrajectoriesInput.model_validate(data)

    def test_api_list_defaults_to_empty(self):
        item = GenerateTrajectoriesInput.model_validate(self._make_valid())
        assert item.api_list == []

    def test_api_list_accepts_list_of_dicts(self):
        item = GenerateTrajectoriesInput.model_validate(self._make_valid(api_list=[{"name": "get_balance"}]))
        assert len(item.api_list) == 1

    def test_extra_fields_allowed(self):
        data = self._make_valid(extra_field="surprise")
        item = GenerateTrajectoriesInput.model_validate(data)
        assert item.category_name == "retail"

    def test_inherits_from_base_step_input(self):
        item = GenerateTrajectoriesInput.model_validate(self._make_valid())
        assert isinstance(item, BaseStepInput)


# ── GenerateTrajectoriesOutput ─────────────────────────────────────────────────────────────

class TestGenerateTrajectoriesOutput:
    def test_requires_conversation(self):
        with pytest.raises(Exception):
            GenerateTrajectoriesOutput.model_validate({})

    def test_valid_output(self):
        output = GenerateTrajectoriesOutput.model_validate({
            "conversation": [{"role": "user", "content": "hello"}]
        })
        assert len(output.conversation) == 1

    def test_conversation_is_list(self):
        output = GenerateTrajectoriesOutput.model_validate({
            "conversation": [{"role": "user", "content": "hi"}]
        })
        assert isinstance(output.conversation, list)

    def test_thinking_mode_defaults_to_false(self):
        output = GenerateTrajectoriesOutput.model_validate({
            "conversation": []
        })
        assert output.thinking_mode is False

    def test_thinking_mode_can_be_set(self):
        output = GenerateTrajectoriesOutput.model_validate({
            "conversation": [],
            "thinking_mode": True,
        })
        assert output.thinking_mode is True

    def test_extra_fields_allowed(self):
        output = GenerateTrajectoriesOutput.model_validate({
            "conversation": [],
            "extra_field": "extra_val"
        })
        assert output.thinking_mode is False


# ── GenerateToolsInput ───────────────────────────────────────────────────────────

class TestGenerateToolsInput:
    def test_requires_category_name_and_function(self):
        with pytest.raises(Exception):
            GenerateToolsInput.model_validate({"category_name": "retail"})

    def test_valid_input(self):
        item = GenerateToolsInput.model_validate({
            "category_name": "retail",
            "function": {"name": "get_balance", "parameters": {}}
        })
        assert item.category_name == "retail"
        assert item.function["name"] == "get_balance"

    def test_constraints_optional(self):
        item = GenerateToolsInput.model_validate({
            "category_name": "retail",
            "function": {"name": "foo"}
        })
        assert item.constraints is None

    def test_constraints_can_be_set(self):
        item = GenerateToolsInput.model_validate({
            "category_name": "retail",
            "function": {"name": "foo"},
            "constraints": "must be positive"
        })
        assert item.constraints == "must be positive"


# ── GenerateToolsOutput ──────────────────────────────────────────────────────────

class TestGenerateToolsOutput:
    def test_requires_tool_code_and_decision(self):
        with pytest.raises(Exception):
            GenerateToolsOutput.model_validate({})

    def test_valid_output(self):
        output = GenerateToolsOutput.model_validate({
            "tool_code": "def foo(): pass",
            "decision": "accepted"
        })
        assert output.tool_code == "def foo(): pass"
        assert output.decision == "accepted"

    def test_normalized_function_defaults_to_none(self):
        output = GenerateToolsOutput.model_validate({
            "tool_code": "def foo(): pass",
            "decision": "accepted"
        })
        assert output.normalized_function is None

    def test_extra_fields_allowed(self):
        output = GenerateToolsOutput.model_validate({
            "tool_code": "def foo(): pass",
            "decision": "modified",
            "extra": "value"
        })
        assert output.decision == "modified"


# ── JudgeTrajectoriesOutput ──────────────────────────────────────────────────────────

class TestJudgeTrajectoriesOutput:
    def test_requires_judgment(self):
        with pytest.raises(Exception):
            JudgeTrajectoriesOutput.model_validate({})

    def test_valid_output(self):
        output = JudgeTrajectoriesOutput.model_validate({"judgment": "looks good"})
        assert output.judgment == "looks good"

    def test_judgment_score_defaults_to_none(self):
        output = JudgeTrajectoriesOutput.model_validate({"judgment": "ok"})
        assert output.score is None

    def test_judgment_score_can_be_set(self):
        output = JudgeTrajectoriesOutput.model_validate({"judgment": "ok", "score": 9})
        assert output.score == 9

    def test_task_completed_defaults_to_none(self):
        output = JudgeTrajectoriesOutput.model_validate({"judgment": "ok"})
        assert output.task_completed is None

    def test_task_completed_can_be_set(self):
        output = JudgeTrajectoriesOutput.model_validate({"judgment": "ok", "task_completed": True})
        assert output.task_completed is True

    def test_extra_fields_allowed(self):
        output = JudgeTrajectoriesOutput.model_validate({
            "judgment": "ok",
            "extra_key": "extra_val"
        })
        assert output.judgment == "ok"
