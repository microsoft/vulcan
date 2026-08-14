"""The spec contract between `normalize_specs` and `build_graph`.

`normalize_specs` always writes `normalized_schema`, and `build_graph` reads that
field and nothing else. The raw catalog `function` must never reach the
dependency prompt: it carries a `response` schema only if the catalog author
supplied one, and without `response` the model cannot detect an `explicit`
(output→input) edge — so a fallback would silently degrade the graph to
ordering-only edges instead of failing.

This is regression cover for a real defect. `build_graph` reads rows from
`env_grouped`, which writes every key unconditionally via `.get()`. A
presence-based selection (`if "normalized_schema" in api`) therefore always
matched, and any catalog that already shipped `response` schemas serialised
`null` into the prompt and produced an empty graph — a full-price run yielding
no dependencies at all.
"""

import pytest

from vulcan.core.exceptions import StepError
from vulcan.steps.graph_construction.build_graph import BuildGraphStep


def _step() -> BuildGraphStep:
    """A step instance without __init__ (which would build an LLM client)."""
    step = object.__new__(BuildGraphStep)
    step.name = "build_graph"
    return step


INP = {"category_name": "ticket_api"}

WITH_RESPONSE = {"name": "search_users", "response": {"users": "array"}}


class TestBuildGraphReadsNormalizedSchemaOnly:
    def test_returns_normalized_schema(self):
        api = {"function": {"name": "search_users"}, "normalized_schema": WITH_RESPONSE}
        assert _step()._get_api_spec(api, INP) == WITH_RESPONSE

    def test_normalized_schema_wins_over_every_other_field(self):
        api = {
            "function": {"name": "raw"},
            "normalized_function": {"name": "corrected"},
            "normalized_schema": WITH_RESPONSE,
        }
        assert _step()._get_api_spec(api, INP) == WITH_RESPONSE

    def test_never_falls_back_to_the_raw_catalog_spec(self):
        """The whole point: a missing normalization must fail, not silently degrade."""
        api = {"function": {"name": "search_users", "response": {"users": "array"}}}
        with pytest.raises(StepError):
            _step()._get_api_spec(api, INP)

    def test_never_falls_back_to_normalized_function(self):
        api = {
            "function": {"name": "raw"},
            "normalized_function": {"name": "corrected"},
        }
        with pytest.raises(StepError):
            _step()._get_api_spec(api, INP)

    def test_present_but_null_is_treated_as_missing(self):
        """`env_grouped` writes every key with .get(), so null is the common case."""
        api = {"function": {"name": "search_users"}, "normalized_schema": None}
        with pytest.raises(StepError):
            _step()._get_api_spec(api, INP)

    def test_error_names_the_api_and_the_remedy(self):
        api = {"function": {"name": "search_users"}, "normalized_schema": None}
        with pytest.raises(StepError) as exc:
            _step()._get_api_spec(api, INP)
        message = str(exc.value)
        assert "search_users" in message
        assert "normalize_specs" in message


class TestNormalizeSpecsAlwaysWritesTheContract:
    """`normalized_schema` must be written on every path, not just the
    response-was-missing one — that gap is what produced the null specs."""

    def test_source_writes_normalized_schema_unconditionally(self):
        import inspect

        from vulcan.steps.env_simulation import normalize_specs

        source = inspect.getsource(normalize_specs.NormalizeSpecsStep.run)
        assert 'inp["normalized_schema"] = api_function' in source, (
            "normalize_specs must assign normalized_schema on every path"
        )
        # It must not sit inside the `if not ... response` branch that only fires
        # when the catalog omitted a response schema.
        assignment_line = next(
            i for i, line in enumerate(source.splitlines())
            if 'inp["normalized_schema"] = api_function' in line
        )
        indent = len(source.splitlines()[assignment_line]) - len(
            source.splitlines()[assignment_line].lstrip()
        )
        assert indent <= 12, "normalized_schema assignment appears to be nested in a branch"


# ── generate_queries parse-failure fallbacks ──────────────────────────────────


class TestParseFailureFallbacksAreKept:
    """A value set on the JSONDecodeError path must survive the method.

    `_classify_complexity` and `_gen_ground_truth_plan` share a shape: parse the
    tagged block, keep the raw text if it will not parse, and fall through to
    `None` only when nothing was extracted at all. `_classify_complexity` was
    missing the `return` on the parse-failure branch, so the loop continued to
    the unconditional `= None` and the salvaged text was always discarded.
    """

    import_path = "vulcan.steps.task_generation.generate_queries"

    def _method_source(self, name: str) -> str:
        import importlib
        import inspect

        module = importlib.import_module(self.import_path)
        return inspect.getsource(getattr(module.GenerateQueriesStep, name))

    def test_classify_complexity_returns_after_keeping_raw(self):
        src = self._method_source("_classify_complexity")
        after = src.split('inp["query_complexity"] = raw', 1)[1]
        # The next statement on that branch must be `return`, not a fall-through.
        assert after.lstrip().startswith("return"), (
            "the JSONDecodeError branch must return, or the value is wiped below"
        )

    def test_ground_truth_plan_has_the_same_shape(self):
        """The sibling this one is modelled on — they must not drift apart."""
        src = self._method_source("_gen_ground_truth_plan")
        after = src.split('inp["ground_truth_plan"] = raw', 1)[1]
        assert after.lstrip().startswith("return")

    def test_both_still_fall_through_to_none_when_nothing_parsed(self):
        """The unconditional reset is correct when no block was ever extracted."""
        for method, field in (
            ("_classify_complexity", "query_complexity"),
            ("_gen_ground_truth_plan", "ground_truth_plan"),
        ):
            src = self._method_source(method)
            assert f'inp["{field}"] = None' in src


# ── plan_sequences output artifacts ───────────────────────────────────────────


class TestPlanSequencesWritesDistinctFiles:
    """The two `multi_turn/` writes must not collide.

    `plan_sequences` writes a sequences-only artifact and, at the end, a merged
    one (sequences + multipath). `save_jsonl` opens mode "w", so if both names
    are equal the merge silently truncates the first. The rename that produced
    this repo mapped the merged file onto a name the sequences-only file already
    had, which destroyed the inspection artifact whenever multipath was enabled.
    """

    def _source(self) -> str:
        import inspect

        from vulcan.steps.graph_construction import plan_sequences

        return inspect.getsource(plan_sequences.PlanSequencesStep)

    def test_the_two_writes_use_different_filenames(self):
        src = self._source()
        assert '"sequences_only.jsonl"' in src
        assert '"all_sequences.jsonl"' in src
        assert src.count('os.path.join(mt_dir, "all_sequences.jsonl")') == 1, (
            "the merged file must be written exactly once, to its own name"
        )

    def test_the_next_stage_reads_the_merged_file(self):
        from vulcan.core.stage_runner import _PLAN_SEQUENCES_OUT, _STANDARD_INPUT

        assert _PLAN_SEQUENCES_OUT.endswith("all_sequences.jsonl")
        assert _STANDARD_INPUT["score_sequences"] == ("plan_sequences", _PLAN_SEQUENCES_OUT)

    def test_plan_sequences_makes_no_model_call(self):
        """Subgraph selection is deterministic — it must stay that way."""
        from vulcan.steps import STEP_REGISTRY

        assert STEP_REGISTRY["plan_sequences"].requires_llm is False
        src = self._source()
        for forbidden in ("Agent(", "self.llm", "await "):
            assert forbidden not in src, f"plan_sequences must not use {forbidden!r}"
