"""Step: generate_tools — Generate Python API implementations from JSON specs.

EXPECTS: Input already preprocessed by normalize_specs (name/description
checked, response schema generated).  Uses the best available spec:
``normalized_schema``, falling back to ``normalized_function`` then ``function``.
normalize_specs now writes ``normalized_schema`` for every API, so the fallbacks
below are a legacy safety net and should not normally engage.  Unlike
build_graph, this step tolerates their absence rather than failing.

Three sub-steps per item:
1. Generate a Python function from the API spec.
2. Review/modify the generated code (modifier agent).
3. Verify the code executes; fix with executor agent if it fails.

Constraint awareness is **per API, not per environment**: sub-step 1 uses the
constraint-aware system/user prompt variants when the item itself carries a
``constraints`` field (copied from the catalog entry by ``normalize_specs``).
The environment-level ``has_constraints`` flag is informational and is not read
here. Category-level ``policy_rules`` never reaches this step.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.code_exec import load_api_class
from ...core.exceptions import StepError
from ...prompts.generate_tools import (
    FG_SYSTEM, FG_USER, FG_SYSTEM_CONSTRAINTS, FG_USER_CONSTRAINTS,
    FM_SYSTEM, FM_USER,
    EXECUTOR_SYSTEM, EXECUTOR_USER,
)


@register_step("generate_tools")
class GenerateToolsStep(BaseStep):
    """Generate Python API implementations from preprocessed JSON specs."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.gen_agent = Agent(
            "func_gen", "", llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_pair_parser("decision", "generated_code"),
        )
        self.modifier_agent = Agent(
            "func_modifier", "", llm,
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
            # Use best available spec (preprocessed > original)
            api_function = self._get_api_spec(inp)

            # Sub-step 1: Generate Python function
            has_constraints = "constraints" in inp
            if has_constraints:
                system = FG_SYSTEM_CONSTRAINTS
                user = FG_USER_CONSTRAINTS.format(
                    API_INFO=json.dumps(api_function, indent=2),
                    POLICY_CONSTRAINTS=inp["constraints"],
                )
            else:
                system = FG_SYSTEM
                user = FG_USER.format(API_INFO=json.dumps(api_function, indent=2))

            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            res = await self.gen_agent.run(messages, progress)
            py_code, decision = self._extract_code_decision(res)
            inp["tool_code"] = py_code
            inp["decision"] = decision

            # Sub-step 2: Review and optionally fix the generated code
            mod_msgs = [
                {"role": "system", "content": FM_SYSTEM},
                {"role": "user", "content": FM_USER.format(PY_CODE=py_code)},
            ]
            res_mod = await self.modifier_agent.run(mod_msgs, progress)
            mod_code, mod_decision = self._extract_code_decision(res_mod, code_tag="generated_code")
            if mod_decision and "Problem" in mod_decision and mod_code:
                py_code = mod_code

            # Sub-step 3: Verify the code actually executes
            try:
                load_api_class(py_code)
                inp["tool_code"] = py_code
            except Exception:
                error_msg = traceback.format_exc()
                fix_prompt = EXECUTOR_USER.format(
                    API_INFO=json.dumps(api_function, indent=2),
                    PY_CODE=py_code,
                    ERROR_MESSAGE=error_msg,
                )
                fix_msgs = [{"role": "user", "content": fix_prompt}]
                res_fix = await self.executor_agent.run(fix_msgs, progress)
                fixed = TagParser.extract_first("verified_code", res_fix["content"])
                if fixed:
                    inp["tool_code"] = fixed
                    try:
                        load_api_class(fixed)
                    except Exception as exec_err:
                        inp["decision"] = (inp.get("decision") or "") + f"\nExecution Error: {exec_err}"
                else:
                    inp["tool_code"] = res_fix["content"]

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    def _get_api_spec(self, inp: dict) -> dict:
        """Return the API spec: normalized_schema, else normalized_function, else function.

        normalize_specs always writes ``normalized_schema``, so the fallbacks are a
        legacy safety net. This step tolerates their absence; ``build_graph`` does
        not, because dependency detection needs the ``response`` schema.
        """
        if "normalized_schema" in inp:
            return inp["normalized_schema"]
        if "normalized_function" in inp:
            return inp["normalized_function"]
        return inp["function"]

    def _extract_code_decision(self, res: dict, code_tag: str = "generated_code") -> tuple[str, str | None]:
        """Extract (code, decision) from an agent response."""
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            p = parsed[0]
            if isinstance(p, dict):
                return p.get(code_tag, res["content"]), p.get("decision")
        code = TagParser.extract_first(code_tag, res["content"])
        decision = TagParser.extract_first("decision", res["content"])
        return code or res["content"], decision
