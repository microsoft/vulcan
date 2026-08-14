"""Step: assemble_environment — Generate unified Python environment classes.

Takes a category-level item with ``api_list`` (each entry having ``tool_code``
from verify_tools) and generates a single unified class that exposes all API
methods through a common ``invoke()`` interface and ``function_name_mapping()``.

Sub-steps per item:
1. Generate the unified class from individual API code snippets + tool specs.
2. Verify the class against tool specs; apply any suggested fixes.
3. Attempt to execute the class; if it fails, ask the executor agent to fix it.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.code_exec import execute_class_code, strip_get_info
from ...core.exceptions import StepError
from ...prompts.assemble_environment import (
    CLASS_GEN_SYSTEM, CLASS_GEN_USER,
    CLASS_VERIFY_SYSTEM, CLASS_VERIFY_USER,
)
from ...prompts.generate_tools import EXECUTOR_SYSTEM, EXECUTOR_USER


@register_step("assemble_environment")
class AssembleEnvironmentStep(BaseStep):
    """Generate a unified Python environment class from individual API implementations."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.gen_agent = Agent(
            "class_gen", "", llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_pair_parser("decision", "generated_code"),
        )
        self.verify_agent = Agent(
            "class_verify", "", llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_pair_parser("decision", "generated_code"),
        )
        self.executor_agent = Agent(
            "executor", EXECUTOR_SYSTEM, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("verified_code"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            # Collect individual API code snippets and tool specifications
            api_codes: list[str] = []
            tool_specs: list[dict] = []
            for api in inp.get("api_list", []):
                code = api.get("tool_code", "")
                if code:
                    stripped = strip_get_info(code)
                    api_codes.append(
                        f"# API: {api.get('function', {}).get('name', 'unknown')}\n{stripped}"
                    )
                # Use best available spec (preprocessed > original)
                spec = (
                    api.get("normalized_schema")
                    or api.get("normalized_function")
                    or api.get("function", {})
                )
                tool_specs.append(spec)

            if not api_codes:
                inp["decision"] = "No API code available"
                return inp

            combined_code = "\n\n".join(api_codes)
            tool_specs_str = json.dumps(tool_specs, indent=2)

            # Sub-step 1: Generate unified class
            messages = [
                {"role": "system", "content": CLASS_GEN_SYSTEM},
                {"role": "user", "content": CLASS_GEN_USER.format(
                    API_CODES=combined_code, TOOL_SPECS=tool_specs_str,
                )},
            ]
            res = await self.gen_agent.run(messages, progress)
            py_code = self._extract_code(res)
            inp["tool_code"] = py_code

            # Sub-step 2: Verify against tool specs
            verify_msgs = [
                {"role": "system", "content": CLASS_VERIFY_SYSTEM},
                {"role": "user", "content": CLASS_VERIFY_USER.format(
                    PY_CODE=py_code, TOOL_SPECS=tool_specs_str,
                )},
            ]
            res_v = await self.verify_agent.run(verify_msgs, progress)
            decision_v = TagParser.extract_first("decision", res_v["content"])
            if decision_v and "Problem" in decision_v:
                fixed = self._extract_code(res_v)
                if fixed:
                    py_code = fixed

            # Sub-step 3: Execute the class; fix if it fails
            try:
                execute_class_code(py_code)
                inp["tool_code"] = py_code
            except Exception:
                error_msg = traceback.format_exc()
                fix_msgs = [{"role": "user", "content": EXECUTOR_USER.format(
                    API_INFO="environment class", PY_CODE=py_code, ERROR_MESSAGE=error_msg,
                )}]
                res_fix = await self.executor_agent.run(fix_msgs, progress)
                fixed = TagParser.extract_first("verified_code", res_fix["content"])
                if fixed:
                    try:
                        execute_class_code(fixed)
                        inp["tool_code"] = fixed
                    except Exception as exec_err:
                        inp["decision"] = f"Execution Error: {exec_err}"

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    def _extract_code(self, res: dict) -> str:
        """Extract generated_code from an agent response dict."""
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            p = parsed[0]
            if isinstance(p, dict):
                return p.get("generated_code", res["content"])
        return TagParser.extract_first("generated_code", res["content"]) or res["content"]
