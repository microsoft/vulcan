"""VULCAN CLI — single entry point for running pipeline stages.

Usage:
    python -m vulcan.cli --env <config_name> --stage <step_name> [--categories cat1,cat2] [--workers N]
    python -m vulcan.cli --list-steps
    python -m vulcan.cli --env <config_name> --pipeline [--start <step>] [--end <step>]

Examples:
    python -m vulcan.cli --list-steps
    python -m vulcan.cli --env mock_test --stage generate_tools --categories ticket_api
    python -m vulcan.cli --env mock_test --pipeline --start generate_trajectories --end judge_trajectories
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

from .config import load_config
from .core.exceptions import LLMError
from .llm import create_client, load_dotenv
from .core.logger import PipelineLogger
from .core.stage_runner import StageRunner
from .environment import get_environment
from .steps import STEP_REGISTRY, PIPELINE_ORDER, list_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VULCAN — Synthetic Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all registered pipeline steps and exit",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment config name (e.g., 'mock_test')",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Run a single stage by name",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the full pipeline (or a range with --start/--end)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start from this stage (inclusive, with --pipeline)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End at this stage (inclusive, with --pipeline)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of environments to process (default: all from config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override parallel worker count",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="Iteration number for trajectory/verify/select stages",
    )
    parser.add_argument(
        "--output-label",
        type=str,
        default=None,
        choices=["native", "reasoning", "tagged"],
        help="Output folder name for the trajectory and judged directories. Defaults to steps.generate_trajectories.tool_call_format, "
             "so the folder always describes what is in it. Pass this only to "
             "override the folder name; it does not change how trajectories are "
             "generated.",
    )

    return parser.parse_args()


def _resolve_output_label(config: dict, requested: str | None) -> str:
    """Pick the traj/verify output folder name.

    The folder is named after the tool-call style that actually produced the
    trajectories (``steps.generate_trajectories.tool_call_format``) unless the
    caller overrides it, so a directory called ``native`` can no longer hold
    ``tagged`` data.

    The ``thinking_model`` fallback checks the step first and then ``defaults``,
    matching ``GenerateTrajectoriesStep._resolve_style`` — otherwise a step-level
    ``thinking_model`` would write reasoning data into a folder named ``native``.
    """
    if requested:
        return requested
    style = ((config.get("steps") or {}).get("generate_trajectories") or {}).get("tool_call_format")
    if style in ("native", "reasoning", "tagged"):
        return style
    step = (config.get("steps") or {}).get("generate_trajectories") or {}
    thinking = step.get(
        "thinking_model", (config.get("defaults") or {}).get("thinking_model", False)
    )
    return "reasoning" if thinking else "native"


def main() -> None:
    args = parse_args()

    if args.list_steps:
        print("VULCAN pipeline steps:")
        print("=" * 50)
        for i, name in enumerate(PIPELINE_ORDER, 1):
            registered = "✓" if name in STEP_REGISTRY else "✗"
            step_cls = STEP_REGISTRY.get(name)
            llm = "LLM" if step_cls and getattr(step_cls, "requires_llm", True) else "deterministic"
            print(f"  {i:2d}. [{registered}] {name:25s} ({llm})")
        print(f"\nTotal: {len(STEP_REGISTRY)} registered, {len(PIPELINE_ORDER)} in pipeline")
        return

    if not args.env:
        print("Error: --env is required (or use --list-steps)")
        sys.exit(1)

    # Read provider keys from a local .env before the config resolves ${VARS}.
    load_dotenv()

    # Load config
    config = load_config(args.env)

    # Apply CLI overrides
    if args.workers:
        config.setdefault("defaults", {})["parallel_workers"] = args.workers

    output_label = _resolve_output_label(config, args.output_label)

    # Fail fast on a missing key or an unknown provider, before any work starts.
    # Each stage then gets a client built from its own overrides; clients are
    # cached per (provider, model, base_url), so a run opens one connection pool
    # per distinct backend rather than one per stage.
    try:
        create_client(config)
    except LLMError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # Setup environment
    environment = get_environment(args.env, config)

    # Setup logger
    output_base = config.get("output_base", ".")
    log_dir = os.path.join(output_base, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Get environments to process
    categories = None
    if args.categories:
        categories = [e.strip() for e in args.categories.split(",")]

    # Create stage runner
    runner = StageRunner()

    if args.stage:
        # Single stage
        if args.stage not in STEP_REGISTRY:
            print(f"Error: unknown stage '{args.stage}'")
            print(f"Available: {', '.join(PIPELINE_ORDER)}")
            sys.exit(1)

        asyncio.run(
            runner.run(
                stage=args.stage,
                config=config,
                llm=create_client(config, args.stage),
                environment=environment,
                categories=categories,
                iteration=args.iter,
                output_label=output_label,
            )
        )

    elif args.pipeline:
        # Pipeline range
        stages = PIPELINE_ORDER[:]
        if args.start:
            try:
                start_idx = stages.index(args.start)
                stages = stages[start_idx:]
            except ValueError:
                print(f"Error: unknown start stage '{args.start}'")
                sys.exit(1)
        if args.end:
            try:
                end_idx = stages.index(args.end)
                stages = stages[: end_idx + 1]
            except ValueError:
                print(f"Error: unknown end stage '{args.end}'")
                sys.exit(1)

        print(f"Running pipeline stages: {' → '.join(stages)}")
        for stage in stages:
            if stage not in STEP_REGISTRY:
                print(f"  Skipping {stage} (not registered)")
                continue
            print(f"\n{'=' * 60}")
            print(f"  STAGE: {stage}")
            print(f"{'=' * 60}")
            asyncio.run(
                runner.run(
                    stage=stage,
                    config=config,
                    llm=create_client(config, stage),
                    environment=environment,
                    categories=categories,
                    iteration=args.iter,
                    output_label=output_label,
                )
            )

    else:
        print("Error: specify --stage <name> or --pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
