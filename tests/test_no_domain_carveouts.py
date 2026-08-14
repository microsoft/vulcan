"""VULCAN must behave the same whatever your categories and tools are called.

The pipeline was built inside a group that ran it against specific benchmarks,
and one of those benchmarks left a filter behind:

    if category in ["retail", "airline", "telecom"]:
        api_list = [a for a in api_list if a["function"]["name"] != "calculate"]

A user whose catalog happened to contain a category named `retail` and a tool
named `calculate` silently lost that tool from the dependency graph — and so
from every sequence, every conversation and their whole training set — with
nothing logged. These tests make sure that class of thing does not come back.
"""

import ast
import asyncio
import inspect
import pathlib
from unittest.mock import Mock

import pytest

from vulcan.steps import STEP_REGISTRY
from vulcan.steps.graph_construction.build_graph import BuildGraphStep

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "vulcan"

# Names that only ever appeared because of a specific benchmark. This repo's own
# fixtures (ticket_api, hotel_booking, ecommerce) are deliberately absent — those
# are example data, not behaviour.
BENCHMARK_NAMES = {
    "retail", "airline", "telecom", "bfcl", "appworld",
    "tau2bench", "tau_bench", "vitabench", "vita_bench", "gorilla",
}


def _python_sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


class TestNoBenchmarkNamesDriveBehaviour:
    """A benchmark name may appear in prose. It may not appear in a comparison."""

    def test_no_comparison_against_a_benchmark_name(self):
        offenders = []
        for path in _python_sources():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for operand in [node.left, *node.comparators]:
                    for const in ast.walk(operand):
                        if isinstance(const, ast.Constant) and isinstance(const.value, str):
                            if const.value.lower() in BENCHMARK_NAMES:
                                offenders.append(
                                    f"{path.relative_to(PACKAGE.parent)}:{node.lineno} "
                                    f"compares against {const.value!r}"
                                )
        assert not offenders, "benchmark-specific branching:\n  " + "\n  ".join(offenders)

    def test_no_membership_test_against_a_literal_category_list(self):
        """Catches `if category in ["retail", "airline", "telecom"]` directly.

        Only a *raw* category value counts. Comparing the result of a call —
        `get_environment_type(cat) in ("with_user_and_policy", "with_user")` —
        is checking VULCAN's own vocabulary and is exactly right.
        """
        category_names = {"category", "cat", "category_name"}
        offenders = []
        for path in _python_sources():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Compare) and node.ops
                        and isinstance(node.ops[0], ast.In)):
                    continue

                left = node.left
                if isinstance(left, ast.Name):
                    is_category = left.id in category_names
                elif isinstance(left, ast.Subscript):  # inp["category_name"]
                    key = getattr(left.slice, "value", None)
                    is_category = key in category_names
                else:
                    is_category = False  # a call result is not a raw category
                if not is_category:
                    continue

                for comp in node.comparators:
                    if isinstance(comp, (ast.List, ast.Tuple, ast.Set)):
                        offenders.append(
                            f"{path.relative_to(PACKAGE.parent)}:{node.lineno}"
                        )
        assert not offenders, (
            "a category compared against a literal collection:\n  " + "\n  ".join(offenders)
        )


class TestBuildGraphKeepsEveryTool:
    """The behavioural proof, not just a source scan."""

    def _step(self):
        step = object.__new__(BuildGraphStep)
        step.name = "build_graph"
        step.graph_agent = Mock()
        return step

    def _catalog(self, *names):
        return [
            {
                "function": {"name": n},
                "normalized_schema": {"name": n, "response": {"ok": "boolean"}},
            }
            for n in names
        ]

    @pytest.mark.parametrize("category", ["retail", "airline", "telecom", "anything_else"])
    def test_a_tool_named_calculate_survives_in_every_category(self, category):
        step = self._step()
        seen = []

        async def fake_run(messages, progress=None):
            seen.append(messages)
            return {"parsed_content": ["[]"], "content": "<graph>[]</graph>"}

        step.graph_agent.run = fake_run
        item = {"category_name": category, "api_list": self._catalog("calculate", "search", "book")}
        out = asyncio.run(step.run(item))

        sources = {entry["source"] for entry in out["graph"]}
        assert "calculate" in sources, (
            f"'calculate' was dropped from the graph for category {category!r}"
        )
        assert sources == {"calculate", "search", "book"}
        # 3 tools -> 3*2 ordered pairs, no tool skipped
        assert len(seen) == 6

    def test_graph_shape_is_identical_across_category_names(self):
        """Rename the category, get the same graph."""
        results = {}
        for category in ("retail", "my_domain"):
            step = self._step()

            async def fake_run(messages, progress=None):
                return {"parsed_content": ["[]"], "content": "<graph>[]</graph>"}

            step.graph_agent.run = fake_run
            item = {"category_name": category,
                    "api_list": self._catalog("calculate", "search")}
            out = asyncio.run(step.run(item))
            results[category] = [
                (e["source"], [t["target"] for t in e["all_edges"]]) for e in out["graph"]
            ]
        assert results["retail"] == results["my_domain"]


class TestTerminateToolsAreNotHardcoded:
    """`terminate_tools` is user-configurable, so nothing may assume its contents."""

    def test_no_step_hardcodes_the_escalation_tool_name(self):
        offenders = []
        for path in _python_sources():
            if path.name == "base.py" and path.parent.name == "environment":
                continue  # the constant is defined there, legitimately
            if '"transfer_to_human_agents"' in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PACKAGE.parent)))
        assert not offenders, (
            "hardcoded escalation tool name outside its definition:\n  " + "\n  ".join(offenders)
        )

    def test_the_default_references_the_constant(self):
        """So the default cannot drift from the tool it names."""
        from vulcan.environment.base import EnvironmentConfig

        src = inspect.getsource(EnvironmentConfig.__init__)
        assert 'self.TRANSFER_TO_HUMAN_API["name"]' in src

    def test_terminate_response_payload_is_tool_agnostic(self):
        """Every terminate branch must emit the same tool-neutral payload."""
        from vulcan.steps.trajectory import generate_trajectories

        src = inspect.getsource(generate_trajectories)
        branches = src.count("if name in self.terminate_tools:")
        payloads = src.count('json.dumps({"status": "transferred"})')
        assert branches >= 6
        assert payloads >= 1
        assert '"content": "transfer_to_human_agents"' not in src


class TestEveryStepIsDomainNeutral:
    @pytest.mark.parametrize("step_name", sorted(STEP_REGISTRY))
    def test_step_module_has_no_benchmark_branching(self, step_name):
        module = inspect.getmodule(STEP_REGISTRY[step_name])
        try:
            tree = ast.parse(inspect.getsource(module))
        except (OSError, SyntaxError):  # pragma: no cover
            pytest.skip("source unavailable")
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    for const in ast.walk(operand):
                        if isinstance(const, ast.Constant) and isinstance(const.value, str):
                            assert const.value.lower() not in BENCHMARK_NAMES, (
                                f"{step_name} branches on {const.value!r} at line {node.lineno}"
                            )


class TestSampleBankFiltersOnSchemaNotOnName:
    """`_resolve_inputs` used to blacklist the literal key `database_index`.

    That name came from the original research group's mock-data format. Any
    catalog with a genuine parameter of that name — plausible for a database,
    analytics or sharding API — silently never had it filled from the sample
    bank, so the tool was invoked without it and the whole chain degraded with
    nothing naming the dropped argument. The tool's own schema is the authority.
    """

    def _resolve(self, spec, bank):
        from vulcan.steps.task_generation.execute_sequences import _resolve_inputs

        return _resolve_inputs("search", [], {}, bank, spec)

    BANK = {"search": [{"database_index": 3, "query": "abc", "not_a_param": "x"}]}

    def test_a_declared_parameter_named_database_index_survives(self):
        spec = {"parameters": {"properties": {"database_index": {}, "query": {}}}}
        assert self._resolve(spec, self.BANK)["database_index"] == 3

    def test_an_undeclared_key_is_not_passed_to_invoke(self):
        """Undeclared kwargs make invoke() raise TypeError."""
        spec = {"parameters": {"properties": {"database_index": {}, "query": {}}}}
        assert "not_a_param" not in self._resolve(spec, self.BANK)

    def test_a_key_the_tool_does_not_declare_is_dropped(self):
        spec = {"parameters": {"properties": {"query": {}}}}
        resolved = self._resolve(spec, self.BANK)
        assert resolved == {"query": "abc"}

    def test_missing_schema_falls_back_to_accepting_the_bank(self):
        """Parameterless or absent spec: keep the old permissive behaviour."""
        assert self._resolve({}, self.BANK) == {
            "database_index": 3, "query": "abc", "not_a_param": "x"
        }

    def test_no_parameter_name_is_hardcoded(self):
        import inspect

        from vulcan.steps.task_generation import execute_sequences

        src = inspect.getsource(execute_sequences._resolve_inputs)
        assert "database_index" not in src


class TestThePackageNamesNoBenchmark:
    """Prose counts too, once the behaviour is clean.

    The package used to describe its own environment types by naming the
    benchmarks they came from — "Static env (BFCL, appworld)", "Dynamic
    (tau2bench)". Nothing branched on those names, but a reader had to know
    four external datasets to learn what `with_user_and_policy` means, and the
    names implied a coupling that no longer exists. The vocabulary in
    `EnvironmentType` is the one VULCAN actually uses; docstrings should use it.

    Example catalogs may still cite benchmarks as *sources* of input data — that
    is why this scans the package, not `examples/` or the docs.
    """

    def test_no_source_file_mentions_a_benchmark(self):
        offenders = []
        for path in _python_sources():
            text = path.read_text(encoding="utf-8").lower()
            for name in sorted(BENCHMARK_NAMES):
                if name in text:
                    line = next(
                        i for i, l in enumerate(text.splitlines(), 1) if name in l
                    )
                    offenders.append(
                        f"{path.relative_to(PACKAGE.parent)}:{line} mentions {name!r}"
                    )
        assert not offenders, (
            "benchmark names in the package — use the EnvironmentType vocabulary "
            "(tools_only / with_user / with_user_and_policy):\n  " + "\n  ".join(offenders)
        )

    def test_the_environment_type_vocabulary_is_what_docstrings_use(self):
        from vulcan.environment.types import EnvironmentType

        vocab = {t.value for t in EnvironmentType}
        assert vocab == {"tools_only", "with_user", "with_user_and_policy"}
        text = (PACKAGE / "environment" / "types.py").read_text(encoding="utf-8")
        for term in vocab:
            assert term in text
