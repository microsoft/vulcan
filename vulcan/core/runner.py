"""Async pipeline runner with incremental saving, retry, and crash recovery.

Output files per stage:
  processing.jsonl — temp file; each completed item written immediately
  success.jsonl    — items that passed validation (appended after each batch)
  failed.jsonl     — items that failed after all retries

Flow:
  1. Each completed item is written to processing.jsonl immediately.
  2. After the batch, validate each → split to success.jsonl + failed.jsonl.
  3. Delete processing.jsonl.

Resume on restart:
  - Finalize any stale processing.jsonl (dedup against success.jsonl).
  - Skip items whose _item_id is already in success.jsonl.
  - Retry items in failed.jsonl.

Quality hooks:
  - Optional *verifier* hook: if provided and the step passes validation,
    verifier.verify(result) is also called; failure demotes the item to failed.
  - Step failures surface as StepError, which the worker catches per item so a
    single bad item cannot abort the batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import traceback
from typing import Any

import aiofiles
from tqdm.asyncio import tqdm

from .io import load_jsonl, ensure_dir
from .exceptions import LLMError, StepError


class PipelineRunner:
    """Runs a pipeline step with incremental saving, retry, and resume.

    Args:
        step:             the step object (must implement run(), validate_output(),
                          preprocess_input(), and name attribute).
        parallel_workers: number of concurrent asyncio workers.
        max_retries:      how many times to retry failed items.
        logger:           optional PipelineLogger instance.
        verifier:         optional object with a verify(result) -> (bool, str) method.
                          Called after the step's own validate_output() passes.
                          If it returns (False, reason) the item is treated as failed.
    """

    def __init__(
        self,
        step: Any,
        *,
        parallel_workers: int = 10,
        max_retries: int = 0,
        logger: Any = None,
        verifier: Any = None,
    ) -> None:
        self.step = step
        self.parallel_workers = parallel_workers
        self.max_retries = max_retries
        self.logger = logger
        self.verifier = verifier

    # ── Public entry point ────────────────────────────────────────────────

    async def run(
        self,
        input_path: str,
        output_path: str,
        *,
        first_k: int | None = None,
    ) -> dict:
        """Run the step with incremental saving, retry, and resume.

        Returns a stats dict:
            {total, success, failed, retries_used, success_path, failed_path}
        """
        raw_data = load_jsonl(input_path)
        data = self.step.preprocess_input(raw_data)
        if first_k:
            data = data[:first_k]

        # Assign stable _item_id based on content hash
        for item in data:
            if "_item_id" not in item:
                key = json.dumps(item, sort_keys=True)
                item["_item_id"] = hashlib.md5(key.encode()).hexdigest()[:12]

        out_dir = os.path.dirname(os.path.abspath(output_path))
        ensure_dir(out_dir)
        self.step.out_dir = out_dir

        if hasattr(self.step, "_propagate_config_to_agents"):
            self.step._propagate_config_to_agents()

        success_path = os.path.join(out_dir, "success.jsonl")
        failed_path = os.path.join(out_dir, "failed.jsonl")
        processing_path = os.path.join(out_dir, "processing.jsonl")

        _id_pattern = re.compile(r'"_item_id"\s*:\s*"([^"]+)"')

        # Recover from any stale processing.jsonl (previous crash)
        if os.path.exists(processing_path):
            self._finalize_processing(processing_path, success_path, failed_path)
            print(f"[{self.step.name}] Finalized stale processing.jsonl → success/failed")

        # Collect already-done IDs (use the top-level _item_id, not a nested one
        # — records may embed sub-items such as api_list that carry their own ids)
        done_ids: set[str] = set()
        if os.path.exists(success_path):
            with open(success_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        iid = json.loads(line).get("_item_id")
                    except json.JSONDecodeError:
                        m = _id_pattern.search(line)
                        iid = m.group(1) if m else None
                    if iid:
                        done_ids.add(iid)

        # Collect previous failures (will be retried)
        previous_failures: list[dict] = []
        if os.path.exists(failed_path):
            with open(failed_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            item.pop("_validation_passed", None)
                            item.pop("_validation_reason", None)
                            previous_failures.append(item)
                        except json.JSONDecodeError:
                            pass

        total = len(data)
        if done_ids or previous_failures:
            print(
                f"[{self.step.name}] Resuming: {len(done_ids)} succeeded, "
                f"{len(previous_failures)} failed from previous run"
            )

        category: str | None = None
        if data and isinstance(data[0], dict):
            category = data[0].get("category_name")

        if self.logger and hasattr(self.step, "set_logger"):
            self.step.set_logger(self.logger, category)
        if self.logger:
            self.logger.step_start(self.step.name, category=category, num_items=total)

        write_lock = asyncio.Lock()

        if previous_failures:
            items_to_process = previous_failures
            # Clear failed file (will be repopulated after this batch)
            async with aiofiles.open(failed_path, "w") as f:
                pass
        else:
            items_to_process = [item for item in data if item.get("_item_id") not in done_ids]

        retries_used = 0

        for attempt in range(1 + self.max_retries):
            if not items_to_process:
                break

            label = (
                f"[{self.step.name}]"
                if attempt == 0
                else f"[{self.step.name} retry {attempt}]"
            )
            print(f"\n{label} Processing {len(items_to_process)} items…")

            successes, failures = await self._run_batch(
                items_to_process, processing_path, write_lock, label, category
            )
            self._finalize_processing(processing_path, success_path, failed_path)

            if not failures:
                print(f"{label} All {len(successes)} items passed!")
                break

            print(f"{label} {len(successes)} passed, {len(failures)} failed")

            if attempt < self.max_retries:
                retries_used += 1
                for item in failures:
                    item.pop("_validation_passed", None)
                    item.pop("_validation_reason", None)
                items_to_process = failures

        # Final counts
        final_success = 0
        final_failed = 0
        if os.path.exists(success_path):
            with open(success_path) as f:
                final_success = sum(1 for line in f if line.strip())
        if os.path.exists(failed_path):
            with open(failed_path) as f:
                final_failed = sum(1 for line in f if line.strip())

        stats = {
            "total": total,
            "success": final_success,
            "failed": final_failed,
            "retries_used": retries_used,
            "success_path": success_path,
            "failed_path": failed_path,
        }
        print(
            f"\n[{self.step.name}] Final: {final_success} success, "
            f"{final_failed} failed out of {total} ({retries_used} retries)"
        )

        if self.logger:
            self.logger.step_end(
                self.step.name, success=final_success, failed=final_failed, retries=retries_used
            )

        return stats

    # ── Internal helpers ──────────────────────────────────────────────────

    def _finalize_processing(
        self,
        processing_path: str,
        success_path: str,
        failed_path: str,
    ) -> None:
        """Split processing.jsonl → success.jsonl + failed.jsonl, then delete it.

        Deduplicates against existing success.jsonl to prevent double-appending
        after a crash recovery.
        """
        if not os.path.exists(processing_path):
            return

        _id_pat = re.compile(r'"_item_id"\s*:\s*"([^"]+)"')
        existing_ids: set[str] = set()
        if os.path.exists(success_path):
            with open(success_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        iid = json.loads(line).get("_item_id")
                    except json.JSONDecodeError:
                        m = _id_pat.search(line)
                        iid = m.group(1) if m else None
                    if iid:
                        existing_ids.add(iid)

        items: list[dict] = []
        with open(processing_path) as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        with open(success_path, "a") as sf, open(failed_path, "a") as ff:
            for item in items:
                iid = item.get("_item_id")
                if iid and iid in existing_ids:
                    continue
                if item.get("_validation_passed", False):
                    sf.write(json.dumps(item) + "\n")
                    if iid:
                        existing_ids.add(iid)
                else:
                    ff.write(json.dumps(item) + "\n")

        os.remove(processing_path)

    async def _run_batch(
        self,
        data: list[dict],
        processing_path: str,
        write_lock: asyncio.Lock,
        label: str,
        category: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """Run one batch; write each item to processing.jsonl immediately."""
        queue: asyncio.Queue = asyncio.Queue()
        for item in data:
            queue.put_nowait(item)

        progress_main = tqdm(total=len(data), desc=f"{label} Items")
        progress_api = tqdm(desc=f"{label} API calls")

        successes: list[dict] = []
        failures: list[dict] = []
        results_lock = asyncio.Lock()
        written_ids: set[str] = set()

        async def worker(worker_id: int) -> None:
            local_successes: list[dict] = []
            local_failures: list[dict] = []

            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                item_id = item.get("_item_id", "unknown")

                try:
                    result = await self.step.run(item, progress_api)

                    passed, reason = self.step.validate_output(result)

                    # Optional verifier hook — only applied when step validation passes
                    if passed and self.verifier is not None:
                        try:
                            v_passed, v_reason = await self.verifier.verify(
                                self.step.name, result)
                            if not v_passed:
                                passed = False
                                reason = f"verifier: {v_reason}"
                        except LLMError:
                            # The provider failed, so the verifier never returned a
                            # verdict about this item. Marking it failed here would
                            # silently reject every item in the run during an outage,
                            # which is exactly what StageVerifier's re-raise exists to
                            # prevent — so let it propagate.
                            raise
                        except Exception as verifier_exc:
                            passed = False
                            reason = f"verifier exception: {verifier_exc}"

                    result["_validation_passed"] = passed
                    if not passed:
                        result["_validation_reason"] = reason

                    async with write_lock:
                        rid = result.get("_item_id", item_id)
                        if rid not in written_ids:
                            written_ids.add(rid)
                            async with aiofiles.open(processing_path, "a") as f:
                                await f.write(json.dumps(result) + "\n")

                    if passed:
                        local_successes.append(result)
                        if self.logger:
                            self.logger.item_success(self.step.name, str(item_id), category)
                    else:
                        local_failures.append(result)
                        if self.logger:
                            self.logger.item_fail(self.step.name, str(item_id), category, reason)

                except StepError as step_err:
                    tb = traceback.format_exc()
                    item["_validation_passed"] = False
                    item["_validation_reason"] = str(step_err)

                    async with write_lock:
                        rid = item.get("_item_id", item_id)
                        if rid not in written_ids:
                            written_ids.add(rid)
                            async with aiofiles.open(processing_path, "a") as f:
                                await f.write(json.dumps(item) + "\n")

                    if self.logger:
                        self.logger.exception(self.step.name, str(step_err), category, tb)
                    local_failures.append(item)

                except (Exception, SystemExit):
                    tb = traceback.format_exc()
                    traceback.print_exc()
                    item["_validation_passed"] = False
                    item["_validation_reason"] = tb

                    async with write_lock:
                        rid = item.get("_item_id", item_id)
                        if rid not in written_ids:
                            written_ids.add(rid)
                            async with aiofiles.open(processing_path, "a") as f:
                                await f.write(json.dumps(item) + "\n")

                    if self.logger:
                        self.logger.exception(self.step.name, tb[:200], category, tb)
                    local_failures.append(item)

                finally:
                    queue.task_done()
                    progress_main.update(1)

            async with results_lock:
                successes.extend(local_successes)
                failures.extend(local_failures)

        tasks = [
            asyncio.create_task(worker(i))
            for i in range(min(self.parallel_workers, len(data)))
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        progress_main.close()
        progress_api.close()

        return successes, failures


async def run_step(
    step: Any,
    input_path: str,
    output_path: str,
    *,
    parallel_workers: int = 10,
    max_retries: int = 0,
    first_k: int | None = None,
    logger: Any = None,
    verifier: Any = None,
) -> dict:
    """Convenience wrapper: create a PipelineRunner and call run()."""
    runner = PipelineRunner(
        step,
        parallel_workers=parallel_workers,
        max_retries=max_retries,
        logger=logger,
        verifier=verifier,
    )
    return await runner.run(input_path, output_path, first_k=first_k)
