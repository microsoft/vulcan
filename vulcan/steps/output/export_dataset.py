"""export_dataset step: Select high-quality trajectories and reconstruct training format.

Registered as the final pipeline step. Deterministic (no LLM calls).

Reads verified trajectories, filters by score/task_completed,
reconstructs into JSON/XML training format, and writes rejected
items as input for the next iteration.
"""

from __future__ import annotations

import json
import os

from .. import register_step
from ..base import BaseStep
from .select import STRIP_FIELDS, _passes_criteria
from .reconstruct import (
    reconstruct_fc_json,
    reconstruct_fc_xml,
    reconstruct_policy_fc_json,
    reconstruct_policy_fc_xml,
    reconstruct_reasoning,
    reconstruct_reasoning_xml,
    reconstruct_prompting,
    prepend_policy,
    validate_reconstructed,
)


def _is_prompting(inp: dict) -> bool:
    """Detect a prompting-style trajectory.

    Priority: explicit per-record 'tool_call_format' == 'prompting', else heuristic on the
    conversation's system message (text <tool_call> + <answer> convention).
    """
    style = inp.get("tool_call_format")
    if isinstance(style, str) and style.lower() == "tagged":
        return True
    if isinstance(style, str) and style.lower() in ("native", "reasoning"):
        return False
    conv = inp.get("conversation", [])
    if conv and isinstance(conv[0], dict) and conv[0].get("role") == "system":
        sys_txt = conv[0].get("content", "") or ""
        return "<answer>" in sys_txt and "<tool_call>" in sys_txt
    return False


@register_step("export_dataset")
class ExportDatasetStep(BaseStep):
    """Select high-quality trajectories and reconstruct into training format.

    This step is deterministic (requires_llm = False).
    It reads from judge_trajectories output and produces:
    - training_dataset/{json,xml}/<name>_iter_<k>.jsonl
    - iter_{k+1}/fc_input.jsonl (rejected items for next iteration)
    """

    requires_llm = False

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.min_score = self.step_config.get("min_score", 8)
        self.require_task_completed = self.step_config.get("require_task_completed", True)

    async def run(self, inp: dict, progress=None) -> dict:
        """Process a single verified trajectory item.

        Adds '_selected': True/False and '_training_record' if selected.
        """
        j = inp.get("judgment", {})
        if isinstance(j, str):
            try:
                j = json.loads(j)
            except json.JSONDecodeError:
                j = {}

        score = 0
        tc_bool = False
        if isinstance(j, dict):
            score = int(j.get("score", j.get("judgment_score", 0)))
            tc = j.get("task_completed", j.get("Task_Completed"))
            tc_bool = tc is True or str(tc).lower() == "true"

        passes = score >= self.min_score
        if self.require_task_completed:
            passes = passes and tc_bool

        inp["_selected"] = passes

        if passes:
            # A `with_user_and_policy` category was driven under its policy and judged
            # against it, so the exported record must carry that policy. Resolved from
            # the environment, the same source generate_trajectories and
            # judge_trajectories read.
            cat_name = inp.get("category_name", "")
            domain_rules = ""
            if cat_name and self.env.get_environment_type(cat_name) == "with_user_and_policy":
                domain_rules = self.env.get_domain_rules(cat_name) or ""

            if _is_prompting(inp):
                msgs = prepend_policy(reconstruct_prompting(inp), domain_rules)
                valid, reasons = validate_reconstructed(msgs)
                if valid:
                    item_id = inp.get("id", inp.get("_item_id", ""))
                    cat = inp.get("category_name", "")
                    record = {"id": item_id, "category": cat, "messages": msgs}
                    inp["_training_json"] = record
                    inp["_training_xml"] = None
                else:
                    inp["_selected"] = False
                    inp["_validation_reasons"] = reasons
                return inp

            thinking = inp.get("thinking_mode", False)
            if thinking:
                # No policy-aware reasoning template exists, so the policy is
                # prepended to the system message instead.
                msgs_json = prepend_policy(reconstruct_reasoning(inp), domain_rules)
                msgs_xml = prepend_policy(reconstruct_reasoning_xml(inp), domain_rules)
            elif domain_rules:
                # Purpose-built templates that interpolate the policy themselves.
                msgs_json = reconstruct_policy_fc_json(inp, domain_rules)
                msgs_xml = reconstruct_policy_fc_xml(inp, domain_rules)
            else:
                msgs_json = reconstruct_fc_json(inp)
                msgs_xml = reconstruct_fc_xml(inp)

            valid, reasons = validate_reconstructed(msgs_json)
            if valid:
                item_id = inp.get("_item_id", inp.get("id", ""))
                cat = inp.get("category_name", "")
                inp["_training_json"] = {"id": item_id, "category_name": cat, "messages": msgs_json}
                inp["_training_xml"] = {"id": item_id, "category_name": cat, "messages": msgs_xml}
            else:
                inp["_selected"] = False
                inp["_validation_reasons"] = reasons

        return inp
