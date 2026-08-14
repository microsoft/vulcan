"""Tests for the step registration system."""

import pytest

from vulcan.steps import (
    STEP_REGISTRY,
    PIPELINE_ORDER,
    register_step,
    get_step_class,
    list_steps,
)


# ── PIPELINE_ORDER completeness ───────────────────────────────────────────────

class TestPipelineOrder:
    def test_has_15_steps(self):
        assert len(PIPELINE_ORDER) == 15

    def test_contains_normalize_specs(self):
        assert "normalize_specs" in PIPELINE_ORDER

    def test_contains_generate_tools(self):
        assert "generate_tools" in PIPELINE_ORDER

    def test_contains_verify_tools(self):
        assert "verify_tools" in PIPELINE_ORDER

    def test_contains_assemble_environment(self):
        assert "assemble_environment" in PIPELINE_ORDER

    def test_contains_debug_environment(self):
        assert "debug_environment" in PIPELINE_ORDER

    def test_contains_generate_states(self):
        assert "generate_states" in PIPELINE_ORDER

    def test_contains_sample_arguments(self):
        assert "sample_arguments" in PIPELINE_ORDER

    def test_contains_build_graph(self):
        assert "build_graph" in PIPELINE_ORDER

    def test_contains_plan_sequences(self):
        assert "plan_sequences" in PIPELINE_ORDER

    def test_contains_score_sequences(self):
        assert "score_sequences" in PIPELINE_ORDER

    def test_contains_execute_sequences(self):
        assert "execute_sequences" in PIPELINE_ORDER

    def test_contains_generate_queries(self):
        assert "generate_queries" in PIPELINE_ORDER

    def test_contains_generate_trajectories(self):
        assert "generate_trajectories" in PIPELINE_ORDER

    def test_contains_judge_trajectories(self):
        assert "judge_trajectories" in PIPELINE_ORDER

    def test_contains_export_dataset(self):
        assert "export_dataset" in PIPELINE_ORDER

    def test_order_preprocess_before_generate_tools(self):
        assert PIPELINE_ORDER.index("normalize_specs") < PIPELINE_ORDER.index("generate_tools")

    def test_order_generate_tools_before_verify_tools(self):
        assert PIPELINE_ORDER.index("generate_tools") < PIPELINE_ORDER.index("verify_tools")

    def test_order_generate_trajectories_before_judge_trajectories(self):
        assert PIPELINE_ORDER.index("generate_trajectories") < PIPELINE_ORDER.index("judge_trajectories")

    def test_order_judge_trajectories_before_export_dataset(self):
        assert PIPELINE_ORDER.index("judge_trajectories") < PIPELINE_ORDER.index("export_dataset")


# ── All PIPELINE_ORDER steps are registered ───────────────────────────────────

class TestAllStepsRegistered:
    @pytest.mark.parametrize("step_name", PIPELINE_ORDER)
    def test_step_is_registered(self, step_name):
        assert step_name in STEP_REGISTRY, (
            f"Step '{step_name}' in PIPELINE_ORDER but not in STEP_REGISTRY"
        )


# ── register_step decorator ───────────────────────────────────────────────────

class TestRegisterStepDecorator:
    def test_decorator_adds_to_registry(self):
        # Use a unique name to avoid collisions with real steps
        test_name = "_test_step_unique_12345"

        @register_step(test_name)
        class _TestStep:
            pass

        assert test_name in STEP_REGISTRY
        assert STEP_REGISTRY[test_name] is _TestStep

        # Cleanup
        del STEP_REGISTRY[test_name]

    def test_decorator_sets_name_attribute(self):
        test_name = "_test_step_name_attr_99"

        @register_step(test_name)
        class _TestStep:
            pass

        assert _TestStep.name == test_name

        # Cleanup
        del STEP_REGISTRY[test_name]

    def test_decorator_returns_class(self):
        test_name = "_test_returns_class_abc"

        @register_step(test_name)
        class _TestStep:
            pass

        assert isinstance(_TestStep, type)

        # Cleanup
        del STEP_REGISTRY[test_name]


# ── get_step_class ────────────────────────────────────────────────────────────

class TestGetStepClass:
    def test_returns_correct_class_for_generate_tools(self):
        cls = get_step_class("generate_tools")
        assert cls is STEP_REGISTRY["generate_tools"]

    def test_returns_correct_class_for_generate_trajectories(self):
        cls = get_step_class("generate_trajectories")
        assert cls is STEP_REGISTRY["generate_trajectories"]

    def test_returns_correct_class_for_export_dataset(self):
        cls = get_step_class("export_dataset")
        assert cls is STEP_REGISTRY["export_dataset"]

    def test_raises_value_error_for_unknown_step(self):
        with pytest.raises(ValueError, match="Unknown step"):
            get_step_class("this_step_does_not_exist")

    def test_error_message_lists_available_steps(self):
        with pytest.raises(ValueError) as exc_info:
            get_step_class("bogus_step_xyz")
        assert "Available" in str(exc_info.value) or "available" in str(exc_info.value).lower()

    @pytest.mark.parametrize("step_name", PIPELINE_ORDER)
    def test_get_step_class_for_all_pipeline_steps(self, step_name):
        cls = get_step_class(step_name)
        assert cls is not None


# ── list_steps ────────────────────────────────────────────────────────────────

class TestListSteps:
    def test_returns_list(self):
        result = list_steps()
        assert isinstance(result, list)

    def test_contains_all_pipeline_steps(self):
        steps = list_steps()
        for name in PIPELINE_ORDER:
            assert name in steps

    def test_length_matches_registry(self):
        assert len(list_steps()) == len(STEP_REGISTRY)

    def test_returns_strings(self):
        steps = list_steps()
        assert all(isinstance(s, str) for s in steps)

    def test_contains_no_duplicates(self):
        steps = list_steps()
        assert len(steps) == len(set(steps))
