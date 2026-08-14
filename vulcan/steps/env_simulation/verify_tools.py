"""Step: verify_tools — Verify Python API implementations against their JSON specs.

Iterative verification loop (up to ``max_fix_iterations``, default 5):
  1. LLM verifier checks code against the spec.
  2. If PROBLEM found → LLM fixes → execute to confirm the fix compiles.
  3. Programmatic check for unused parameters in invoke() → fix if found.
  4. Store iteration result (decision + flags).
  5. If FINE and no unused params → stop early.

After all iterations: run name/description check (informational only).

Output fields:
  - ``tool_code``:             the final (best) version of the code.
  - ``verification_iterations``:     list of per-iteration result dicts.
  - ``py_verification_decision``:    decision from the last iteration.
  - ``verification_iteration_count``: number of iterations performed.
  - ``check_name_description``:      informational name/desc consistency result.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.code_exec import load_api_class, find_unused_invoke_params
from ...core.exceptions import StepError
from ...prompts.verify_tools import (
    VERIFY_SYSTEM, VERIFY_USER, UNUSED_PARAMS_USER,
    CHECK_NAME_DESC_SYSTEM, CHECK_NAME_DESC_USER,
)
from ...prompts.generate_tools import EXECUTOR_SYSTEM, EXECUTOR_USER


@register_step("verify_tools")
class VerifyToolsStep(BaseStep):
    """Iteratively verify and fix Python API implementations against their specs."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.max_fix_iterations: int = self.step_config.get("max_fix_iterations", 5)

        self.verify_agent = Agent(
            "verifier", "", llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_pair_parser("decision", "modified_code"),
        )
        self.check_agent = Agent(
            "check_name_desc", "", llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        self.executor_agent = Agent(
            "executor", EXECUTOR_SYSTEM, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("verified_code"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            py_code = inp["tool_code"]
            api_function = self._get_api_spec(inp)
            api_str = json.dumps({"type": "function", "function": api_function}, indent=2)

            iterations: list[dict] = []

            for iteration in range(1, self.max_fix_iterations + 1):
                iter_result: dict = {"iteration": iteration}

                # Step 1: LLM verify code against spec
                messages = [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": VERIFY_USER.format(PY_CODE=py_code, INP_DICT=api_str)},
                ]
                res = await self.verify_agent.run(messages, progress)
                decision, modified_code = self._extract_decision_code(res)
                iter_result["decision"] = decision

                # Step 2: If PROBLEM, apply the suggested fix
                if decision and "PROBLE" in decision and modified_code:
                    py_code = await self._try_execute_or_fix(modified_code, api_str, inp, progress)
                    iter_result["code_modified"] = True
                else:
                    iter_result["code_modified"] = False

                # Step 3: Programmatic unused-params check
                unused = find_unused_invoke_params(py_code)
                iter_result["unused_params"] = unused if unused else []

                if unused:
                    unused_str = ", ".join(unused)
                    recheck_msgs = [
                        {"role": "system", "content": VERIFY_SYSTEM},
                        {"role": "user", "content": UNUSED_PARAMS_USER.format(
                            PY_CODE=py_code, INP_DICT=api_str, UNUSED_PARAMS=unused_str,
                        )},
                    ]
                    res2 = await self.verify_agent.run(recheck_msgs, progress)
                    _, recheck_code = self._extract_decision_code(res2)
                    if recheck_code and recheck_code.strip().lower() != "none":
                        try:
                            load_api_class(recheck_code)
                            py_code = recheck_code
                            iter_result["unused_params_fixed"] = True
                        except Exception:
                            iter_result["unused_params_fixed"] = False
                    else:
                        iter_result["unused_params_fixed"] = False

                # Persist best code after this iteration
                inp["tool_code"] = py_code
                iterations.append(iter_result)

                # Early exit: FINE and no unused params
                if decision and "FINE" in decision and not unused:
                    break

            # Persist iteration metadata
            inp["verification_iterations"] = iterations
            inp["py_verification_decision"] = iterations[-1]["decision"] if iterations else None
            inp["verification_iteration_count"] = len(iterations)

            # Name/description check (informational — runs after all verify iterations)
            initial_api = inp.get("normalized_function", inp.get("function", {}))
            check_msgs = [
                {"role": "system", "content": CHECK_NAME_DESC_SYSTEM},
                {"role": "user", "content": CHECK_NAME_DESC_USER.format(
                    PY_CODE=py_code, INP_DICT=initial_api,
                )},
            ]
            res3 = await self.check_agent.run(check_msgs, progress)

            if "PASS" in res3["content"]:
                inp["check_name_description"] = {"decision": True, "description": ""}
            elif "PROBLEM" in res3["content"]:
                inp["check_name_description"] = {"decision": False, "description": res3["content"]}
            else:
                inp["check_name_description"] = {"decision": None, "description": "no clear response"}

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    def _get_api_spec(self, inp: dict) -> dict:
        """Return the API spec: normalized_schema, else normalized_function, else function.

        normalize_specs always writes ``normalized_schema``, so the fallbacks are a
        legacy safety net rather than an expected path.
        """
        if "normalized_schema" in inp:
            return inp["normalized_schema"]
        if "normalized_function" in inp:
            return inp["normalized_function"]
        return inp["function"]

    def _extract_decision_code(self, res: dict) -> tuple[str | None, str | None]:
        """Extract (decision, modified_code) from an agent response."""
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            p = parsed[0]
            if isinstance(p, dict):
                return p.get("decision"), p.get("modified_code")
        decision = TagParser.extract_first("decision", res["content"])
        code = TagParser.extract_first("modified_code", res["content"])
        return decision, code

    async def _try_execute_or_fix(
        self, py_code: str, api_str: str, inp: dict, progress
    ) -> str:
        """Try to execute *py_code*.  If it fails, ask the executor agent to fix it."""
        try:
            load_api_class(py_code)
            return py_code
        except Exception:
            error_msg = traceback.format_exc()
            fix_msgs = [{"role": "user", "content": EXECUTOR_USER.format(
                API_INFO=api_str, PY_CODE=py_code, ERROR_MESSAGE=error_msg,
            )}]
            res = await self.executor_agent.run(fix_msgs, progress)
            fixed = TagParser.extract_first("verified_code", res["content"])
            if fixed:
                try:
                    load_api_class(fixed)
                    return fixed
                except Exception as exec_err:
                    inp["decision"] = (inp.get("decision") or "") + f"\nExecution Error: {exec_err}"
            return py_code
