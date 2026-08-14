"""Selection logic for trajectory output step.

Public API:
    STRIP_FIELDS                    - set of keys added by generate_trajectories/verify (stripped for next iter)
    select_and_reject(path, score)  - split JSONL into selected vs rejected inputs
"""

from __future__ import annotations

import json

# ── Field sets ────────────────────────────────────────────────────────────────

# Fields added by generate_trajectories step
TRAJ_FIELDS: frozenset[str] = frozenset({"conversation", "thinking_mode"})

# Fields added by judge_trajectories step
VERIFY_FIELDS: frozenset[str] = frozenset({"judgment"})

# All fields to strip when creating next-iteration input
STRIP_FIELDS: frozenset[str] = TRAJ_FIELDS | VERIFY_FIELDS


# ── Selection criteria ────────────────────────────────────────────────────────

def _passes_criteria(judgment: dict | None, min_score: int) -> bool:
    """Return True if a judgment dict meets selection criteria.

    Criteria:
        score >= min_score AND task_completed == True
    """
    if judgment is None:
        return False
    if isinstance(judgment, str):
        try:
            judgment = json.loads(judgment)
        except (json.JSONDecodeError, TypeError):
            return False
    try:
        score = int(judgment.get("score", 0))
        task_completed = judgment.get("task_completed", False)
        return score >= min_score and bool(task_completed)
    except (TypeError, ValueError):
        return False


# ── Main selection function ───────────────────────────────────────────────────

def select_and_reject(
    input_path: str,
    min_score: int = 8,
) -> tuple[list[dict], list[dict], int]:
    """Split verified items into selected (high quality) and rejected.

    Reads a JSONL file (typically a verify/fc/success.jsonl or
    verify/thinking/success.jsonl).

    Args:
        input_path: path to the JSONL file.
        min_score:  minimum score for selection (inclusive).

    Returns:
        (selected, rejected_inputs, total)
        - selected:        list of full items passing criteria
        - rejected_inputs: list of items stripped back to pre-traj format
                           (STRIP_FIELDS removed) for the next iteration
        - total:           total number of lines read
    """
    selected: list[dict] = []
    rejected: list[dict] = []
    total = 0

    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            judgment = item.get("judgment")
            if _passes_criteria(judgment, min_score):
                selected.append(item)
            else:
                stripped = {k: v for k, v in item.items() if k not in STRIP_FIELDS}
                rejected.append(stripped)

    return selected, rejected, total
