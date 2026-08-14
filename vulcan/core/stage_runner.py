"""StageRunner — declarative orchestration for a single pipeline stage.

The CLI (``python -m vulcan.cli``) constructs one :class:`StageRunner` and calls
:meth:`StageRunner.run` once per stage. StageRunner owns all input/output path
resolution and dispatch:

  * standard per-item stages run through :class:`PipelineRunner`
    (incremental save, retry, crash-recovery);
  * deterministic batch stages (``plan_sequences``, ``execute_sequences``) call the
    step's ``run_batch``;
  * grouping, iteration-aware, and cross-category stages have dedicated handlers.

Output layout (under ``config["output_base"]``)::

    <output_base>/<category>/<step>/success.jsonl              # most steps
    <output_base>/<category>/env_grouped/success.jsonl         # APIs regrouped after verify_tools
    <output_base>/<category>/plan_sequences/multi_turn/all_sequences.jsonl
    <output_base>/<category>/execute_sequences/pre_query_all_modes.jsonl
    <output_base>/<category>/iter_<k>/trajectories/<label>/success.jsonl
    <output_base>/<category>/iter_<k>/judged/<label>/success.jsonl
    <output_base>/<category>/iter_<k+1>/input.jsonl            # rejected items, next iteration
    <output_base>/training_dataset/{json,xml}/<category>_iter_<k>.jsonl

Usage (exactly as the CLI calls it)::

    runner = StageRunner()
    await runner.run(stage="generate_tools", config=config, llm=gw,
                     environment=env_config, categories=["ticket_api"],
                     iteration=0, output_label="native")
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from .io import load_jsonl, save_jsonl, ensure_dir
from .logger import PipelineLogger
from .runner import PipelineRunner
from .exceptions import VulcanError


# Special output locations (relative to the step directory).
_PLAN_SEQUENCES_OUT = os.path.join("multi_turn", "all_sequences.jsonl")
_EXECUTE_SEQUENCES_OUT = "pre_query_all_modes.jsonl"

# Standard input->output stages: step_name -> (previous step dir, file within it).
# Grouping (env_grouped), deterministic batch, iteration-aware, and cross-category
# stages are handled by dedicated methods and are intentionally absent here.
_STANDARD_INPUT: dict[str, tuple[str, str]] = {
    "generate_tools":    ("normalize_specs", "success.jsonl"),
    "verify_tools": ("generate_tools", "success.jsonl"),
    "assemble_environment":   ("env_grouped", "success.jsonl"),
    "debug_environment":    ("assemble_environment", "success.jsonl"),
    "generate_states":  ("debug_environment", "success.jsonl"),
    "sample_arguments": ("generate_states", "success.jsonl"),
    "build_graph":       ("env_grouped", "success.jsonl"),
    "score_sequences":      ("plan_sequences", _PLAN_SEQUENCES_OUT),
    "generate_queries":  ("execute_sequences", _EXECUTE_SEQUENCES_OUT),
}


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


class StageRunner:
    """Run one pipeline stage for one or more categories.

    Stateless: the CLI builds a single instance and reuses it across stages.
    """

    def __init__(self) -> None:
        pass

    # ── Public entry point (called by cli.py) ─────────────────────────────────

    async def run(
        self,
        *,
        stage: str,
        config: dict,
        llm: Any,
        environment: Any,
        categories: list[str] | None = None,
        iteration: int = 0,
        output_label: str = "native",
    ) -> bool:
        """Run *stage* for the selected categories.

        Args:
            stage:       registered step name (see ``--list-steps``).
            config:      merged config dict from ``load_config``.
            llm:     a ``LLMClient``.
            environment: an ``EnvironmentConfig``.
            categories:        categories to process; defaults to ``config["categories"]``
                         (or every category in the input catalog).
            iteration:   iteration index for traj/verify/select stages.
            output_label:   ``"native"``, ``"reasoning"``, or ``"tagged"`` — selects the
                         traj/verify subdirectory.

        Returns:
            True if every processed category succeeded, else False.
        """
        from ..steps import STEP_REGISTRY  # lazy: avoid a core<->steps import cycle

        if stage not in STEP_REGISTRY:
            print(f"Error: unknown stage {stage!r}")
            return False

        output_base = config.get("output_base", ".")
        log_dir = os.path.join(output_base, "logs")
        ensure_dir(log_dir)
        logger = PipelineLogger(log_dir, getattr(environment, "name", ""))

        categories = (
            list(categories) if categories
            else (config.get("categories") or environment.get_categories())
        )
        if not categories:
            print("Error: no categories to process (empty 'categories' and empty input catalog)")
            return False

        defaults = config.get("defaults", {})
        workers = defaults.get("parallel_workers", 20)
        max_retries = defaults.get("max_retries", 10)

        # export_dataset aggregates across all categories in one pass.
        if stage == "export_dataset":
            return await self._run_export_dataset(
                config, llm, environment, categories, iteration, logger
            )

        all_ok = True
        for cat in categories:
            try:
                ok = await self._run_for_category(
                    stage, cat, config, llm, environment,
                    iteration, output_label, workers, max_retries, logger,
                )
            except VulcanError as exc:
                logger.error(f"[{stage}/{cat}] {exc}")
                ok = False
            except Exception as exc:  # noqa: BLE001 — one bad category shouldn't abort the rest
                logger.error(f"[{stage}/{cat}] unexpected error: {exc}\n{traceback.format_exc()}")
                ok = False
            all_ok = all_ok and ok
        return all_ok

    # ── Per-category dispatch ─────────────────────────────────────────────────

    async def _run_for_category(
        self, stage, cat, config, llm, environment,
        iteration, output_label, workers, max_retries, logger,
    ) -> bool:
        if stage == "normalize_specs":
            return await self._run_normalize_specs(
                cat, config, llm, environment, workers, max_retries, logger
            )
        if stage == "plan_sequences":
            return self._run_plan_sequences(cat, config, llm, environment, logger)
        if stage == "execute_sequences":
            return self._run_execute_sequences(cat, config, llm, environment, logger)
        if stage == "generate_trajectories":
            return await self._run_generate_trajectories(
                cat, config, llm, environment, iteration, output_label, workers, logger
            )
        if stage == "judge_trajectories":
            return await self._run_judge_trajectories(
                cat, config, llm, environment, iteration, output_label, workers, logger
            )

        # Standard input -> output stage.
        ok = await self._run_standard(
            stage, cat, config, llm, environment, workers, max_retries, logger
        )
        # verify_tools's per-API rows are regrouped into one record per category
        # so assemble_environment and build_graph receive a unified ``api_list``.
        if stage == "verify_tools":
            self._build_env_grouped(cat, config, logger)
        return ok

    # ── Path helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _step_dir(output_base: str, cat: str, step: str) -> str:
        return os.path.join(output_base, cat, step)

    @staticmethod
    def _success(output_base: str, cat: str, step: str) -> str:
        return os.path.join(output_base, cat, step, "success.jsonl")

    # ── Runner / verifier construction ────────────────────────────────────────

    def _make_runner(self, stage, step, workers, max_retries, logger, config, llm) -> PipelineRunner:
        return PipelineRunner(
            step,
            parallel_workers=workers,
            max_retries=max_retries,
            logger=logger,
            verifier=self._make_verifier(stage, config, llm),
        )

    @staticmethod
    def _make_verifier(stage, config, llm):
        ver_cfg = config.get("verification", {})
        if ver_cfg.get("enabled") and stage in ver_cfg.get("verify_steps", []):
            from ..verification import StageVerifier  # lazy
            # Pass the verification block, not the whole config: StageVerifier
            # reads config["model"]/["max_tokens"]/["temperature"], so handing it
            # the full config silently ignored verification.model and fell back
            # to the verifier's own default model.
            return StageVerifier(llm, ver_cfg)
        return None

    # ── Standard stage ────────────────────────────────────────────────────────

    async def _run_standard(
        self, stage, cat, config, llm, environment, workers, max_retries, logger
    ) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        prev_step, prev_file = _STANDARD_INPUT[stage]
        input_path = os.path.join(output_base, cat, prev_step, prev_file)
        output_path = self._success(output_base, cat, stage)

        if not os.path.exists(input_path):
            logger.error(f"[{stage}/{cat}] input not found: {input_path}")
            return False

        step = STEP_REGISTRY[stage](config, llm, environment)
        runner = self._make_runner(stage, step, workers, max_retries, logger, config, llm)
        stats = await runner.run(input_path, output_path)
        logger.info(f"[{stage}/{cat}] {stats}")
        return stats.get("success", 0) > 0

    # ── Specialised stages ────────────────────────────────────────────────────

    async def _run_normalize_specs(
        self, cat, config, llm, environment, workers, max_retries, logger
    ) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        api_catalog = environment.get_api_catalog(cat)
        if not api_catalog:
            logger.error(f"[normalize_specs/{cat}] no APIs found in the input catalog")
            return False

        cat_record: dict = {"category_name": cat, "api_list": api_catalog}
        rules = environment.get_domain_rules(cat)
        if rules:
            cat_record["policy_rules"] = rules

        step_dir = self._step_dir(output_base, cat, "normalize_specs")
        ensure_dir(step_dir)
        input_path = os.path.join(step_dir, "input.jsonl")
        save_jsonl([cat_record], input_path)
        output_path = os.path.join(step_dir, "success.jsonl")

        step = STEP_REGISTRY["normalize_specs"](config, llm, environment)
        runner = self._make_runner("normalize_specs", step, workers, max_retries, logger, config, llm)
        stats = await runner.run(input_path, output_path)
        logger.info(f"[normalize_specs/{cat}] {stats}")
        return stats.get("success", 0) > 0

    def _build_env_grouped(self, cat, config, logger) -> None:
        """Regroup verify_tools's per-API rows into one record per category.

        Keeps only APIs that produced a ``tool_code``. Written to
        ``<category>/env_grouped/success.jsonl``, which is the input for both
        ``assemble_environment`` and ``build_graph``.

        All four per-API keys are written **unconditionally** via ``.get()``, so
        every key is present on every row whatever its value —
        ``normalized_function`` is ``None`` for an API that needed no correction.
        Consumers must therefore select on the **value**, not on key presence: a
        ``"normalized_schema" in api`` test matches these rows even when the
        value is ``None``.
        """
        output_base = config["output_base"]
        verified_path = self._success(output_base, cat, "verify_tools")
        if not os.path.exists(verified_path):
            return

        api_list = [
            {
                "function": v.get("function"),
                "tool_code": v.get("tool_code"),
                "normalized_function": v.get("normalized_function"),
                "normalized_schema": v.get("normalized_schema"),
            }
            for v in load_jsonl(verified_path)
            if v.get("tool_code")
        ]
        if not api_list:
            logger.error(f"[verify_tools/{cat}] no verified APIs to group — env_grouped skipped")
            return

        grouped_path = self._success(output_base, cat, "env_grouped")
        save_jsonl([{"category_name": cat, "api_list": api_list}], grouped_path)
        logger.info(f"[env_grouped/{cat}] grouped {len(api_list)} APIs")

    def _run_plan_sequences(self, cat, config, llm, environment, logger) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        input_path = self._success(output_base, cat, "build_graph")
        if not os.path.exists(input_path):
            logger.error(f"[plan_sequences/{cat}] input not found: {input_path}")
            return False

        output_dir = self._step_dir(output_base, cat, "plan_sequences")
        ensure_dir(output_dir)
        step = STEP_REGISTRY["plan_sequences"](config, llm, environment)
        step.run_batch(input_path, output_dir)

        out = os.path.join(output_dir, _PLAN_SEQUENCES_OUT)
        n = _count_lines(out)
        logger.info(f"[plan_sequences/{cat}] {n} samples → {out}")
        return n > 0

    def _run_execute_sequences(self, cat, config, llm, environment, logger) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        graph_path = os.path.join(self._step_dir(output_base, cat, "plan_sequences"), _PLAN_SEQUENCES_OUT)
        env_data_path = self._success(output_base, cat, "sample_arguments")
        if not os.path.exists(graph_path):
            logger.error(f"[execute_sequences/{cat}] graph input not found: {graph_path}")
            return False

        output_dir = self._step_dir(output_base, cat, "execute_sequences")
        ensure_dir(output_dir)
        step = STEP_REGISTRY["execute_sequences"](config, llm, environment)
        step.run_batch(graph_path, env_data_path, output_dir)

        out = os.path.join(output_dir, _EXECUTE_SEQUENCES_OUT)
        n = _count_lines(out)
        logger.info(f"[execute_sequences/{cat}] {n} samples → {out}")
        return n > 0

    async def _run_generate_trajectories(
        self, cat, config, llm, environment, iteration, output_label, workers, logger
    ) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        if iteration == 0:
            input_path = self._success(output_base, cat, "generate_queries")
        else:
            input_path = os.path.join(output_base, cat, f"iter_{iteration}", "input.jsonl")

        if not os.path.exists(input_path):
            logger.error(f"[generate_trajectories/{cat}] input not found: {input_path}")
            return False

        output_dir = os.path.join(output_base, cat, f"iter_{iteration}", "trajectories", output_label)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "success.jsonl")

        step = STEP_REGISTRY["generate_trajectories"](config, llm, environment)
        runner = self._make_runner("generate_trajectories", step, workers, 0, logger, config, llm)
        stats = await runner.run(input_path, output_path)
        logger.info(f"[generate_trajectories/{cat}] iter={iteration} type={output_label} {stats}")
        return stats.get("success", 0) > 0

    async def _run_judge_trajectories(
        self, cat, config, llm, environment, iteration, output_label, workers, logger
    ) -> bool:
        from ..steps import STEP_REGISTRY

        output_base = config["output_base"]
        input_path = os.path.join(
            output_base, cat, f"iter_{iteration}", "trajectories", output_label, "success.jsonl"
        )
        if not os.path.exists(input_path):
            logger.error(f"[judge_trajectories/{cat}] input not found: {input_path}")
            return False

        output_dir = os.path.join(output_base, cat, f"iter_{iteration}", "judged", output_label)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "success.jsonl")

        step = STEP_REGISTRY["judge_trajectories"](config, llm, environment)
        runner = self._make_runner("judge_trajectories", step, workers, 0, logger, config, llm)
        stats = await runner.run(input_path, output_path)
        logger.info(f"[judge_trajectories/{cat}] iter={iteration} type={output_label} {stats}")
        return stats.get("success", 0) > 0

    async def _run_export_dataset(
        self, config, llm, environment, categories, iteration, logger
    ) -> bool:
        """Filter verified trajectories into training records; route rejects to the next iter."""
        from ..steps import STEP_REGISTRY
        from ..steps.output.select import STRIP_FIELDS

        output_base = config["output_base"]
        json_dir = os.path.join(output_base, "training_dataset", "json")
        xml_dir = os.path.join(output_base, "training_dataset", "xml")
        ensure_dir(json_dir)
        ensure_dir(xml_dir)

        # Annotations added by the step are dropped before an item feeds the next iteration.
        drop = set(STRIP_FIELDS) | {
            "_selected", "_training_json", "_training_xml", "_validation_reasons",
        }
        step = STEP_REGISTRY["export_dataset"](config, llm, environment)

        any_selected = False
        for cat in categories:
            judged_root = os.path.join(output_base, cat, f"iter_{iteration}", "judged")
            items: list[dict] = []
            if os.path.isdir(judged_root):
                for output_label in sorted(os.listdir(judged_root)):
                    vp = os.path.join(judged_root, output_label, "success.jsonl")
                    if os.path.exists(vp):
                        items.extend(load_jsonl(vp))
            if not items:
                logger.info(f"[export_dataset/{cat}] no verified items for iter_{iteration}")
                continue

            json_records: list[dict] = []
            xml_records: list[dict] = []
            rejected: list[dict] = []
            for it in items:
                out = await step.run(it)
                if out.get("_selected"):
                    if out.get("_training_json"):
                        json_records.append(out["_training_json"])
                    if out.get("_training_xml"):
                        xml_records.append(out["_training_xml"])
                else:
                    rejected.append({k: v for k, v in out.items() if k not in drop})

            if json_records:
                save_jsonl(json_records, os.path.join(json_dir, f"{cat}_iter_{iteration}.jsonl"))
            if xml_records:
                save_jsonl(xml_records, os.path.join(xml_dir, f"{cat}_iter_{iteration}.jsonl"))
            if rejected:
                save_jsonl(
                    rejected,
                    os.path.join(output_base, cat, f"iter_{iteration + 1}", "input.jsonl"),
                )

            logger.info(
                f"[export_dataset/{cat}] selected={len(json_records)} "
                f"rejected={len(rejected)} → iter_{iteration + 1}"
            )
            any_selected = any_selected or bool(json_records)

        return any_selected
