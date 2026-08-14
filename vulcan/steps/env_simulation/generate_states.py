"""Step: generate_states — Generate initial state data for environments.

Each processed item corresponds to one LLM call producing one initial state.
``preprocess_input`` explodes one env item into N items (N = ``num_states``),
so the runner can parallelize state generation across items.

Temperature > 0 (controlled by step config / defaults) ensures variety between calls.

Output fields:
  - ``initial_state``: parsed state dict, or ``None`` if generation/parsing failed.
  - ``_filter_reason``: set when ``initial_state`` is None due to a quality check.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError
from ...prompts.generate_states import (
    INIT_STATE_SYSTEM, INIT_STATE_USER,
    JSON_REPAIR_SYSTEM, JSON_REPAIR_USER,
)


@register_step("generate_states")
class GenerateStatesStep(BaseStep):
    """Generate initial state data for environment categories."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.num_states: int = self.step_config.get("num_states", 500)
        self.min_list_length: int = self.step_config.get("min_list_length", 5)

        init_system = INIT_STATE_SYSTEM.replace("{MIN_LIST_VALUES}", str(self.min_list_length))
        self.init_agent = Agent(
            "init_state", init_system, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("initial_states"),
        )
        self.repair_agent = Agent(
            "json_repair", JSON_REPAIR_SYSTEM, llm,
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("repaired_json"),
        )

    def preprocess_input(self, raw_data: list[dict]) -> list[dict]:
        """Explode one env item into N items — one per initial state to generate.

        Filters out items without ``tool_code``.
        """
        items: list[dict] = []
        for env_item in raw_data:
            py_code = env_item.get("tool_code", "")
            if not py_code:
                continue
            for i in range(self.num_states):
                items.append({
                    "state_index": i,
                    "category_name": env_item.get("category_name", ""),
                    "tool_code": py_code,
                    "api_list": env_item.get("api_list", []),
                })
        return items

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            py_code = inp.get("tool_code", "")
            if not py_code:
                return inp

            msgs = [{"role": "user", "content": INIT_STATE_USER.format(PY_CODE=py_code)}]
            res = await self.init_agent.run(msgs, progress)
            state = await self._parse_json_with_repair(res, "initial_states", progress)

            # Handle list response (model may return [state] instead of state dict)
            if isinstance(state, list) and state:
                state = state[0]

            if isinstance(state, dict) and state:
                if self._check_state_values(state):
                    inp["initial_state"] = state
                else:
                    inp["initial_state"] = None
                    inp["_filter_reason"] = "failed _check_state_values"
            else:
                inp["initial_state"] = None

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    def _check_state_values(self, state: dict) -> bool:
        """Check that top-level list fields have at least ``min_list_length`` entries
        and that no top-level field is empty (None or "").

        Only the TOP level is checked — nested lists inside objects (e.g.
        ``user.roles`` with 1 item) are fine and realistic.
        """
        if not isinstance(state, dict):
            return True
        for val in state.values():
            if val is None:
                return False
            if isinstance(val, str) and val == "":
                return False
            if isinstance(val, list) and len(val) < self.min_list_length:
                return False
        return True

    async def _parse_json_with_repair(self, res: dict, tag: str, progress) -> dict | list | None:
        """Try to parse JSON from *res*, falling back to a repair agent if needed."""
        from ...core.json_utils import try_parse_json

        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], (dict, list)):
                return parsed[0]
            obj, _ = try_parse_json(str(parsed[0]))
            if obj is not None:
                return obj

        raw = TagParser.extract_first(tag, res["content"])
        if raw:
            obj, _ = try_parse_json(raw)
            if obj is not None:
                return obj

        obj, _ = try_parse_json(res["content"])
        if obj is not None:
            return obj

        # Fall back to LLM-based JSON repair
        raw_for_repair = raw or res["content"]
        repair_msgs = [{"role": "user", "content": JSON_REPAIR_USER.format(BROKEN_JSON=raw_for_repair)}]
        repair_res = await self.repair_agent.run(repair_msgs, progress)
        repaired = TagParser.extract_first("repaired_json", repair_res["content"])
        if repaired:
            obj, _ = try_parse_json(repaired)
            if obj is not None:
                return obj

        return None
