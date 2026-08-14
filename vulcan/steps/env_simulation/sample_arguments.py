"""Step: sample_arguments — Generate function call samples per initial state.

Each initial state gets its own function samples via a separate LLM call,
since different states have different values that affect expected outputs.

Input: items from generate_states (one item per initial state).
``preprocess_input`` supports both:
  - New format: each item has ``initial_state`` directly (from generate_states).
  - Legacy format: one item with ``mock_data.initial_data`` list.

Output field:
  - ``functions``: list of function-call sample dicts for every API in the category.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError
from ...prompts.sample_arguments import FUNC_SAMPLE_SYSTEM, FUNC_SAMPLE_USER
from ...prompts.generate_states import JSON_REPAIR_SYSTEM, JSON_REPAIR_USER


@register_step("sample_arguments")
class SampleArgumentsStep(BaseStep):
    """Generate function call samples for each initial state."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.gen_agent = Agent(
            "func_sample", FUNC_SAMPLE_SYSTEM, llm,
            model=self.model, temperature=0.7, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("function_samples"),
        )
        self.repair_agent = Agent(
            "json_repair", JSON_REPAIR_SYSTEM, llm,
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("repaired_json"),
        )

    def preprocess_input(self, data: list[dict]) -> list[dict]:
        """Pass items through, exploding legacy ``mock_data`` format if present.

        Supports two input formats:
          - New:    each item already has ``initial_state`` (from generate_states).
          - Legacy: one item with ``mock_data.initial_data`` list.
        """
        exploded: list[dict] = []
        for item in data:
            if "initial_state" in item and "mock_data" not in item:
                # New format: already one item per state
                if not item.get("state_id") and item.get("state_index") is not None:
                    item = {**item, "state_id": item["state_index"]}
                exploded.append(item)
            elif "mock_data" in item:
                # Legacy format: explode mock_data.initial_data
                py_code = item.get("tool_code", "")
                cat_name = item.get("category_name", "")
                initial_data_list = item.get("mock_data", {}).get("initial_data", [])
                if not isinstance(initial_data_list, list) or not initial_data_list:
                    exploded.append(item)
                    continue
                for state_item in initial_data_list:
                    exploded.append({
                        "category_name": cat_name,
                        "tool_code": py_code,
                        "state_id": state_item.get("id", 0),
                        "initial_state": state_item.get("initial_state", state_item),
                    })
            else:
                exploded.append(item)
        return exploded

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            py_code = inp.get("tool_code", "")
            initial_state = inp.get("initial_state", {})
            cat_name = inp.get("category_name", "")

            # Resolve API names for this category from the environment config
            api_specs = self.env.get_api_catalog(cat_name)
            api_names = [
                a["function"]["name"]
                for a in api_specs
                if a.get("function", {}).get("parameters", {}).get("properties")
            ]
            api_list_str = json.dumps(api_names, indent=2)

            msgs = [{"role": "user", "content": FUNC_SAMPLE_USER.format(
                PY_CODE=py_code,
                INITIAL_STATE=json.dumps(initial_state, indent=2),
                API_LIST=api_list_str,
            )}]

            best_result: list[dict] | None = None

            for attempt in range(3):
                res = await self.gen_agent.run(msgs, progress)
                functions = await self._parse_functions(res, progress)

                if functions and isinstance(functions, list):
                    generated_names = {f.get("function_name") for f in functions}
                    if generated_names >= set(api_names):
                        best_result = functions
                        break
                    elif best_result is None or len(generated_names) > len(
                        {f.get("function_name") for f in best_result}
                    ):
                        best_result = functions

                    missing = set(api_names) - generated_names
                    msgs.extend([
                        {"role": "assistant", "content": res["content"]},
                        {"role": "user", "content": f"Missing functions: {list(missing)}. Generate samples for all APIs."},
                    ])

            inp["functions"] = best_result

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    async def _parse_functions(self, res: dict, progress) -> list[dict] | None:
        """Parse function samples from *res*, falling back to repair agent if needed."""
        from ...core.json_utils import try_parse_json, try_parse_json_list

        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], list):
                return parsed[0]
            obj, _ = try_parse_json(str(parsed[0]))
            if isinstance(obj, list):
                return obj

        raw = TagParser.extract_first("function_samples", res["content"])
        if raw:
            obj, _ = try_parse_json_list(raw)
            if obj is not None:
                return obj

        obj, _ = try_parse_json_list(res["content"])
        if obj is not None:
            return obj

        # Last-resort: try as plain JSON
        raw_fallback = raw or res["content"]
        obj, _ = try_parse_json(raw_fallback)
        if isinstance(obj, list):
            return obj

        return None
