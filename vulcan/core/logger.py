"""Pipeline logger — tracks all activity across steps and categories.

Writes to:
1. Global log:       {output_base}/pipeline.log
2. Per-category log: {output_base}/{category}/pipeline.log
3. Per-step log:     {output_base}/{category}/{step_name}.log
                 or  {output_base}/steps/{step_name}.log  (no category)

Log levels:
- Step start/end with timing
- Per-item success/failure/retry
- LLM call counts and token usage
- Error tracebacks
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional


class PipelineLogger:
    """Centralized logger for the VULCAN pipeline."""

    def __init__(self, output_base: str, domain_name: str = "") -> None:
        self.output_base = output_base
        self.domain_name = domain_name
        self.start_time = time.time()

        os.makedirs(output_base, exist_ok=True)

        # Global logger (writes to console + file)
        self.global_logger = self._create_logger(
            name="vulcan.pipeline",
            log_file=os.path.join(output_base, "pipeline.log"),
        )

        self._cat_loggers: dict[str, logging.Logger] = {}
        self._step_loggers: dict[str, logging.Logger] = {}

        # Aggregated stats
        self._step_stats: dict[str, dict] = {}
        self._total_llm_calls: int = 0
        self._total_tokens: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

        self.info(f"Pipeline started | domain={domain_name} | output={output_base}")

    # ── Logger factory ────────────────────────────────────────────────────

    def _create_logger(self, name: str, log_file: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console output only for the global logger
        if name == "vulcan.pipeline":
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger

    def get_category_logger(self, category: str) -> logging.Logger:
        """Get (or create) a per-category logger."""
        if category not in self._cat_loggers:
            cat_dir = os.path.join(self.output_base, category)
            os.makedirs(cat_dir, exist_ok=True)
            self._cat_loggers[category] = self._create_logger(
                name=f"vulcan.{category}",
                log_file=os.path.join(cat_dir, "pipeline.log"),
            )
        return self._cat_loggers[category]

    def get_step_logger(self, step_name: str, category: Optional[str] = None) -> logging.Logger:
        """Get (or create) a per-step logger."""
        key = f"{category or '_global'}:{step_name}"
        if key not in self._step_loggers:
            if category:
                log_dir = os.path.join(self.output_base, category)
            else:
                log_dir = os.path.join(self.output_base, "steps")
            os.makedirs(log_dir, exist_ok=True)
            self._step_loggers[key] = self._create_logger(
                name=f"vulcan.step.{key}",
                log_file=os.path.join(log_dir, f"{step_name}.log"),
            )
        return self._step_loggers[key]

    # ── Convenience write methods ─────────────────────────────────────────

    def info(self, msg: str, category: Optional[str] = None) -> None:
        self.global_logger.info(msg)
        if category:
            self.get_category_logger(category).info(msg)

    def debug(self, msg: str, category: Optional[str] = None) -> None:
        self.global_logger.debug(msg)
        if category:
            self.get_category_logger(category).debug(msg)

    def warning(self, msg: str, category: Optional[str] = None) -> None:
        self.global_logger.warning(msg)
        if category:
            self.get_category_logger(category).warning(msg)

    def error(self, msg: str, category: Optional[str] = None) -> None:
        self.global_logger.error(msg)
        if category:
            self.get_category_logger(category).error(msg)

    # ── Step lifecycle ────────────────────────────────────────────────────

    def step_start(
        self,
        step_name: str,
        category: Optional[str] = None,
        num_items: int = 0,
    ) -> None:
        self._step_stats[step_name] = {
            "start_time": time.time(),
            "category": category,
            "num_items": num_items,
            "success": 0,
            "failed": 0,
            "retries": 0,
            "llm_calls": 0,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
        }
        msg = f"STEP START | {step_name} | category={category or 'all'} | items={num_items}"
        self.info(msg, category)
        self.get_step_logger(step_name, category).info(msg)

    def step_end(
        self,
        step_name: str,
        success: int = 0,
        failed: int = 0,
        retries: int = 0,
    ) -> None:
        stats = self._step_stats.get(step_name, {})
        elapsed = time.time() - stats.get("start_time", time.time())
        category = stats.get("category")
        stats["success"] = success
        stats["failed"] = failed
        stats["retries"] = retries

        msg = (
            f"STEP END   | {step_name} | "
            f"success={success} failed={failed} retries={retries} | "
            f"llm_calls={stats.get('llm_calls', 0)} | "
            f"tokens={stats.get('tokens', {}).get('total', 0)} | "
            f"time={elapsed:.1f}s"
        )
        self.info(msg, category)
        self.get_step_logger(step_name, category).info(msg)

    # ── Item-level tracking ───────────────────────────────────────────────

    def item_success(
        self,
        step_name: str,
        item_id: str = "",
        category: Optional[str] = None,
        details: str = "",
    ) -> None:
        msg = f"  PASS | {item_id} | {details}" if details else f"  PASS | {item_id}"
        self.debug(msg, category)
        self.get_step_logger(step_name, category).info(msg)
        if step_name in self._step_stats:
            self._step_stats[step_name]["success"] += 1

    def item_fail(
        self,
        step_name: str,
        item_id: str = "",
        category: Optional[str] = None,
        reason: str = "",
    ) -> None:
        msg = f"  FAIL | {item_id} | {reason}"
        self.warning(msg, category)
        self.get_step_logger(step_name, category).warning(msg)
        if step_name in self._step_stats:
            self._step_stats[step_name]["failed"] += 1

    def item_retry(
        self,
        step_name: str,
        item_id: str = "",
        category: Optional[str] = None,
        attempt: int = 0,
    ) -> None:
        msg = f"  RETRY | {item_id} | attempt={attempt}"
        self.debug(msg, category)
        self.get_step_logger(step_name, category).info(msg)
        if step_name in self._step_stats:
            self._step_stats[step_name]["retries"] += 1

    # ── LLM call tracking ────────────────────────────────────────────────

    def llm_call(
        self,
        step_name: str,
        agent_name: str = "",
        category: Optional[str] = None,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self._total_llm_calls += 1
        self._total_tokens["prompt"] += prompt_tokens
        self._total_tokens["completion"] += completion_tokens
        self._total_tokens["total"] += total_tokens

        if step_name in self._step_stats:
            self._step_stats[step_name]["llm_calls"] += 1
            self._step_stats[step_name]["tokens"]["prompt"] += prompt_tokens
            self._step_stats[step_name]["tokens"]["completion"] += completion_tokens
            self._step_stats[step_name]["tokens"]["total"] += total_tokens

        msg = f"  LLM  | {agent_name} | model={model} | tokens={total_tokens}"
        self.debug(msg, category)
        self.get_step_logger(step_name, category).debug(msg)

    # ── Error tracking ────────────────────────────────────────────────────

    def exception(
        self,
        step_name: str,
        error: str,
        category: Optional[str] = None,
        traceback: str = "",
    ) -> None:
        msg = f"  ERROR | {step_name} | {error}"
        self.error(msg, category)
        step_log = self.get_step_logger(step_name, category)
        step_log.error(msg)
        if traceback:
            self.debug(f"  TRACEBACK | {step_name} | {traceback}", category)
            step_log.error(f"  TRACEBACK | {traceback}")

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print and write a pipeline summary table. Returns the text."""
        elapsed = time.time() - self.start_time
        lines = [
            "",
            "=" * 70,
            f"PIPELINE SUMMARY | domain={self.domain_name} | total_time={elapsed:.1f}s",
            "=" * 70,
            f"Total LLM calls: {self._total_llm_calls}",
            (
                f"Total tokens: prompt={self._total_tokens['prompt']} "
                f"completion={self._total_tokens['completion']} "
                f"total={self._total_tokens['total']}"
            ),
            "",
            f"{'Step':<25s} {'Success':>8s} {'Failed':>8s} {'Retries':>8s} "
            f"{'LLM':>6s} {'Tokens':>10s} {'Time':>8s}",
            "-" * 70,
        ]

        for step_name, stats in self._step_stats.items():
            elapsed_step = stats.get("start_time", 0)
            elapsed_step = time.time() - elapsed_step if elapsed_step else 0.0
            lines.append(
                f"{step_name:<25s} "
                f"{stats.get('success', 0):>8d} "
                f"{stats.get('failed', 0):>8d} "
                f"{stats.get('retries', 0):>8d} "
                f"{stats.get('llm_calls', 0):>6d} "
                f"{stats.get('tokens', {}).get('total', 0):>10d} "
                f"{elapsed_step:>7.1f}s"
            )

        lines.append("=" * 70)
        summary_text = "\n".join(lines)
        self.global_logger.info(summary_text)

        summary_path = os.path.join(self.output_base, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")

        return summary_text
