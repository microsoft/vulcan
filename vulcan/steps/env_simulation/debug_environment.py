"""Step: debug_environment — Test and iteratively fix environment code.

Flow per iteration:
  0. Validate parameter names/types match between tool specs and Python methods
     (fix if mismatched, fail early if still mismatched after max retries).
  0.5. Generate a test initial state using the init_state prompt.
  1. Generate test cases for the environment class.
  2. Execute tests in an isolated namespace.
  3. Analyze test results + review initial state design (single agent).
  4. If failures: fix code using the combined analysis.
  5. Repeat until all tests pass or max iterations reached.

Output fields:
  - ``environment_ready``:      True if all tests passed in any iteration.
  - ``tool_code``:       the final (best) version of the code.
  - ``test_iterations``:       list of per-iteration result dicts.
  - ``param_validation_failed``: True if param validation failed after max retries.
"""

from __future__ import annotations

import ast
import copy
import inspect
import io
import json
import sys
import traceback
import types

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.code_exec import execute_class_code
from ...core.exceptions import StepError
from ...prompts.debug_environment import (
    TEST_GEN_SYSTEM, TEST_GEN_USER,
    ANALYZE_SYSTEM, ANALYZE_USER,
    FIX_SYSTEM, FIX_USER,
)
from ...prompts.generate_states import INIT_STATE_SYSTEM, INIT_STATE_USER


@register_step("debug_environment")
class DebugEnvironmentStep(BaseStep):
    """Test environment code and iteratively fix failures."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.max_fix_iterations: int = self.step_config.get("max_fix_iterations", 3)
        self.max_test_generation_retries: int = self.step_config.get("max_test_generation_retries", 3)

        init_system = INIT_STATE_SYSTEM.replace("{MIN_LIST_VALUES}", "5")
        self.init_state_agent = Agent(
            "test_init_state", init_system, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("initial_states"),
        )
        self.test_gen_agent = Agent(
            "test_gen", TEST_GEN_SYSTEM, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        self.analyze_agent = Agent(
            "analyze", ANALYZE_SYSTEM, llm,
            model=self.model, temperature=0, max_tokens=self.max_tokens,
        )
        self.fix_agent = Agent(
            "fixer", FIX_SYSTEM, llm,
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("modified_code"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            py_code = inp.get("modified_env_code") or inp.get("tool_code", "")
            if not py_code:
                inp["environment_ready"] = False
                return inp

            # Build tool specs string for analysis
            tool_specs: list[dict] = []
            for api in inp.get("api_list", []):
                spec = (
                    api.get("normalized_schema")
                    or api.get("normalized_function")
                    or api.get("function", {})
                )
                tool_specs.append(spec)
            tool_specs_str = json.dumps(tool_specs, indent=2)

            # Step 0: Validate parameter names/types against tool specs
            py_code, param_valid = await self._validate_and_fix_params(
                py_code, tool_specs, tool_specs_str, progress,
            )
            if not param_valid:
                inp["environment_ready"] = False
                inp["tool_code"] = py_code
                inp["param_validation_failed"] = True
                return inp

            # Step 0.5: Generate a test initial state
            test_init_state = await self._generate_test_init_state(
                py_code, tool_specs_str, progress,
            )

            iterations_log: list[dict] = []
            prev_test_code: str | None = None
            same_test_fails = 0
            _MAX_SAME_TEST_RETRIES = 3

            for iteration in range(self.max_fix_iterations):
                # Decide: reuse previous tests or generate new ones
                reuse = (
                    prev_test_code is not None
                    and same_test_fails < _MAX_SAME_TEST_RETRIES
                )

                test_code: str | None = None
                test_output: str | None = None
                gen_attempt = 0

                if reuse:
                    test_code = prev_test_code
                    test_output = self._execute_tests(py_code, test_code)
                    if not self._tests_actually_ran(test_output):
                        reuse = False

                if not reuse:
                    same_test_fails = 0
                    for gen_attempt in range(self.max_test_generation_retries):
                        test_msgs = [{"role": "user", "content": TEST_GEN_USER.format(
                            PY_CODE=py_code,
                            TOOL_SPECS=tool_specs_str,
                            INIT_STATE=test_init_state,
                        )}]
                        test_res = await self.test_gen_agent.run(test_msgs, progress)
                        test_code = test_res["content"]
                        test_output = self._execute_tests(py_code, test_code)
                        if self._tests_actually_ran(test_output):
                            break

                # Step 3: Analyze results + state review (single combined agent)
                analyze_msgs = [{"role": "user", "content": ANALYZE_USER.format(
                    TEST_CODE=test_code,
                    TEST_OUTPUT=test_output,
                    PY_CODE=py_code,
                    TOOL_SPECS=tool_specs_str,
                )}]
                analyze_res = await self.analyze_agent.run(analyze_msgs, progress)
                analysis = analyze_res["content"]

                # Extract verdict from <verdict> tag
                verdict = (TagParser.extract_first("verdict", analysis) or "").strip().upper()

                # Step 4: Validate code executes and mapping is populated
                try:
                    cls = execute_class_code(py_code)
                    obj = cls()
                    mapping = obj.function_name_mapping()
                    validation_passed = isinstance(mapping, dict) and len(mapping) > 0
                except Exception:
                    validation_passed = False

                # Step 5: Check if all tests passed
                if verdict == "ALL_PASSED":
                    # Gate: state-awareness check (deterministic)
                    sa_issues = self._check_state_awareness(py_code, tool_specs)
                    confirmed = []
                    if sa_issues:
                        confirmed = self._run_differential_test(
                            py_code, sa_issues, test_init_state, tool_specs,
                        )
                    if confirmed:
                        report = self._build_state_awareness_report(confirmed)
                        fix_msgs = [{"role": "user", "content": FIX_USER.format(
                            ANALYSIS=report, PY_CODE=py_code,
                        )}]
                        fix_res = await self.fix_agent.run(fix_msgs, progress)
                        fixed = TagParser.extract_first("modified_code", fix_res["content"])
                        sa_fixed = False
                        if fixed and fixed.strip():
                            fixed = self._strip_markdown_fences(fixed)
                            try:
                                execute_class_code(fixed)
                                re_issues = self._check_state_awareness(fixed, tool_specs)
                                re_confirmed = (
                                    self._run_differential_test(
                                        fixed, re_issues, test_init_state, tool_specs,
                                    )
                                    if re_issues else []
                                )
                                if len(re_confirmed) < len(confirmed):
                                    py_code = fixed
                                    sa_fixed = True
                            except Exception:
                                pass

                        prev_test_code = None
                        same_test_fails = 0
                        iterations_log.append({
                            "iteration": iteration + 1,
                            "status": "STATE_AWARENESS_FIX",
                            "validation_passed": validation_passed,
                            "state_awareness_issues": len(sa_issues),
                            "state_awareness_confirmed": len(confirmed),
                            "state_awareness_fixed": sa_fixed,
                            "test_gen_retries": gen_attempt,
                            "test_code": test_code[:2000],
                            "test_output": test_output[:1000],
                            "analysis": report,
                            "tool_code": py_code[:5000],
                        })
                        continue  # re-run LLM tests on the fixed code

                    # No state-awareness issues — genuine pass
                    iterations_log.append({
                        "iteration": iteration + 1,
                        "status": "ALL_PASSED",
                        "validation_passed": validation_passed,
                        "test_gen_retries": gen_attempt,
                        "test_code": test_code[:2000],
                        "test_output": test_output[:1000],
                        "analysis": analysis,
                        "tool_code": py_code[:5000],
                    })
                    inp["environment_ready"] = True
                    inp["tool_code"] = py_code
                    inp["test_iterations"] = iterations_log
                    return inp

                # Step 6: Fix code using the combined analysis
                fix_msgs = [{"role": "user", "content": FIX_USER.format(
                    ANALYSIS=analysis,
                    PY_CODE=py_code,
                )}]
                fix_res = await self.fix_agent.run(fix_msgs, progress)
                fixed = TagParser.extract_first("modified_code", fix_res["content"])
                code_updated = False
                if fixed and fixed.strip():
                    fixed = self._strip_markdown_fences(fixed)
                    try:
                        execute_class_code(fixed)
                        py_code = fixed
                        code_updated = True
                    except Exception:
                        pass

                iter_status = "TESTS_FAILED" if code_updated else "FIX_FAILED"
                prev_test_code = test_code
                same_test_fails += 1
                iterations_log.append({
                    "iteration": iteration + 1,
                    "status": iter_status,
                    "validation_passed": validation_passed,
                    "test_gen_retries": gen_attempt,
                    "test_code": test_code[:2000],
                    "test_output": test_output[:1000],
                    "analysis": analysis,
                    "code_updated": code_updated,
                    "tool_code": py_code[:5000],
                })

            # Max iterations reached without full pass
            inp["environment_ready"] = False
            inp["tool_code"] = py_code
            inp["test_iterations"] = iterations_log

        except (StepError, KeyError):
            raise
        except (Exception, SystemExit):
            traceback.print_exc()
            inp["environment_ready"] = False

        return inp

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _generate_test_init_state(
        self, py_code: str, tool_specs_str: str, progress
    ) -> str:
        """Generate a realistic initial state for testing, reusing the init_state prompt."""
        try:
            msgs = [{"role": "user", "content": INIT_STATE_USER.format(PY_CODE=py_code)}]
            res = await self.init_state_agent.run(msgs, progress)

            parsed = res.get("parsed_content")
            if parsed and isinstance(parsed, list) and parsed:
                raw = parsed[0]
            else:
                raw = TagParser.extract_first("initial_states", res["content"])

            if raw and raw.strip():
                raw = raw.strip()
                try:
                    state = json.loads(raw)
                    if isinstance(state, dict):
                        return json.dumps(state, indent=2)
                except json.JSONDecodeError:
                    pass
                try:
                    state = eval(raw)  # noqa: S307
                    if isinstance(state, dict):
                        return json.dumps(state, indent=2)
                except Exception:
                    pass

            return "{}"
        except Exception:
            return "{}"

    @staticmethod
    def _check_param_mismatches(py_code: str, tool_specs: list[dict]) -> list[dict]:
        """Check parameter name/type mismatches between tool specs and Python methods.

        Returns a list of mismatch dicts:
          ``[{"tool": name, "spec_only": [...], "method_only": [...]}]``
        """
        try:
            cls = execute_class_code(py_code)
            obj = cls()
            mapping = obj.function_name_mapping()
        except Exception:
            return []  # Cannot check if code doesn't compile

        mismatches: list[dict] = []
        for spec in tool_specs:
            spec_name = spec.get("name", "")
            spec_params = spec.get("parameters", {})

            if isinstance(spec_params, dict):
                props = spec_params.get("properties", spec_params)
                if "properties" in props and "type" in props:
                    props = props["properties"]
            else:
                props = {}

            spec_param_names = set(props.keys())

            if spec_name not in mapping:
                continue

            method = mapping[spec_name]
            try:
                sig = inspect.signature(method)
                method_param_names = set(
                    p for p in sig.parameters
                    if p != "self"
                    and sig.parameters[p].kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                )
            except (ValueError, TypeError):
                continue

            spec_only = spec_param_names - method_param_names
            method_only = method_param_names - spec_param_names

            if spec_only or method_only:
                mismatches.append({
                    "tool": spec_name,
                    "spec_only": sorted(spec_only),
                    "method_only": sorted(method_only),
                })

        return mismatches

    @staticmethod
    def _check_state_awareness(py_code: str, tool_specs: list[dict]) -> list[dict]:
        """AST-based check: flag methods that don't read self.state but should.

        Returns list of issue dicts for methods that appear to use hardcoded data.
        Detects: inline dicts/lists, nested functions returning hardcoded collections,
        and methods that only write to self.state but never read domain data from it.
        """
        try:
            tree = ast.parse(py_code)
        except SyntaxError:
            return []

        cls_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_node = node
                break
        if cls_node is None:
            return []

        skip = {"__init__", "function_name_mapping"}
        data_verbs = {"search", "query", "fetch", "retrieve", "lookup", "browse"}
        output_names = {"result", "response", "output", "error_response",
                        "ret", "resp", "payload", "reply", "filtered", "matched"}
        write_only_patterns = {"last_results", "setdefault"}

        spec_descs = {}
        for s in tool_specs:
            name = s.get("name", "")
            desc = s.get("description", "")
            spec_descs[name] = desc

        issues: list[dict] = []

        for method_node in cls_node.body:
            if not isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method_node.name in skip or method_node.name.startswith("_"):
                continue

            state_reads = 0
            state_writes_only = 0
            hardcoded: list[str] = []

            for child in ast.walk(method_node):
                if not isinstance(child, ast.Subscript):
                    continue
                val = child.value
                if not (isinstance(val, ast.Attribute) and val.attr == "state"
                        and isinstance(val.value, ast.Name) and val.value.id == "self"):
                    continue
                key = None
                if isinstance(child.slice, ast.Constant):
                    key = child.slice.value
                if key and key in write_only_patterns:
                    state_writes_only += 1
                else:
                    state_reads += 1

            for child in ast.walk(method_node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if (isinstance(func, ast.Attribute)
                            and func.attr in ("setdefault", "get")
                            and isinstance(func.value, ast.Attribute)
                            and func.value.attr == "state"
                            and isinstance(func.value.value, ast.Name)
                            and func.value.value.id == "self"):
                        if child.args:
                            arg0 = child.args[0]
                            if isinstance(arg0, ast.Constant) and arg0.value == "last_results":
                                state_writes_only += 1
                            else:
                                state_reads += 1

            def _is_mostly_constants(node: ast.AST) -> bool:
                """Check if a dict/list literal has mostly constant values (hardcoded data)."""
                if isinstance(node, ast.Dict):
                    vals = [v for v in node.values if v is not None]
                    if not vals:
                        return False
                    const_count = sum(1 for v in vals if isinstance(v, (ast.Constant, ast.Dict, ast.List)))
                    return const_count / len(vals) > 0.5
                if isinstance(node, ast.List):
                    if not node.elts:
                        return False
                    const_count = sum(1 for e in node.elts if isinstance(e, (ast.Constant, ast.Dict, ast.List)))
                    return const_count / len(node.elts) > 0.5
                return False

            for child in ast.walk(method_node):
                if isinstance(child, ast.Assign):
                    val = child.value
                    targets = [
                        t.id for t in child.targets if isinstance(t, ast.Name)
                    ]
                    target_name = targets[0] if targets else ""
                    if target_name.lower() in output_names:
                        continue
                    if isinstance(val, ast.Dict) and len(val.keys) >= 3 and _is_mostly_constants(val):
                        hardcoded.append(f"{target_name} (dict, {len(val.keys)} entries)")
                    if isinstance(val, ast.List) and len(val.elts) >= 3 and _is_mostly_constants(val):
                        hardcoded.append(f"{target_name} (list, {len(val.elts)} items)")

            for child in ast.walk(method_node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child is method_node:
                        continue
                    for inner in ast.walk(child):
                        if isinstance(inner, ast.Return) and inner.value:
                            if isinstance(inner.value, ast.List) and len(inner.value.elts) >= 3:
                                hardcoded.append(
                                    f"{child.name}() return (list, {len(inner.value.elts)} items)"
                                )
                            if isinstance(inner.value, ast.Dict) and len(inner.value.keys) >= 3:
                                hardcoded.append(
                                    f"{child.name}() return (dict, {len(inner.value.keys)} entries)"
                                )

            desc = spec_descs.get(method_node.name, "")
            desc_lower = desc.lower()
            method_lower = method_node.name.lower()
            data_nouns = {"database", "stored", "storage", "repository",
                         "catalog", "inventory", "registry"}
            is_data_method = (
                any(v in method_lower or v in desc_lower for v in data_verbs)
                or any(n in desc_lower for n in data_nouns)
            )

            compute_verbs = {"calculate", "compute", "convert", "format", "parse",
                             "validate", "encode", "decode", "generate", "hash",
                             "fibonacci", "prime", "factorial", "sqrt", "average",
                             "sum", "count", "length", "palindrome", "anagram",
                             "even", "odd", "absolute", "round", "ceil", "floor",
                             "min", "max", "sort", "reverse", "upper", "lower",
                             "replace", "split", "join", "strip", "trim",
                             "create", "add", "insert", "set", "update", "delete",
                             "remove", "move", "copy", "rename", "write"}
            is_pure_compute = (
                not is_data_method
                and any(v in method_lower or v in desc_lower for v in compute_verbs)
            )

            if hardcoded and state_reads >= 1:
                import re as _re
                large_hardcoded = []
                for h in hardcoded:
                    m = _re.search(r'(\d+)\s+(?:entries|items)', h)
                    size = int(m.group(1)) if m else 0
                    if size >= 6:
                        large_hardcoded.append(h)
                hardcoded = large_hardcoded

            if hardcoded:
                issues.append({
                    "method": method_node.name,
                    "issue_type": "HARDCODED_DATA",
                    "detail": (
                        f"Method has {state_reads} state reads, "
                        f"{state_writes_only} write-only refs, "
                        f"hardcoded: {hardcoded}"
                        + (f". Tool description: '{desc[:120]}'" if desc else "")
                    ),
                    "state_refs": state_reads,
                    "hardcoded_collections": hardcoded,
                })
            elif state_reads == 0 and is_data_method and not is_pure_compute:
                issues.append({
                    "method": method_node.name,
                    "issue_type": "NO_STATE_READ",
                    "detail": (
                        f"Method has 0 state reads ({state_writes_only} write-only refs). "
                        f"A data-retrieval method should read from self.state"
                        + (f". Tool description: '{desc[:120]}'" if desc else "")
                    ),
                    "state_refs": 0,
                    "hardcoded_collections": [],
                })

        return issues

    @staticmethod
    def _run_differential_test(
        py_code: str,
        suspects: list[dict],
        test_init_state: str,
        tool_specs: list[dict],
    ) -> list[dict]:
        """Run each suspect method with two different states; confirm if output is identical."""
        try:
            state_a = json.loads(test_init_state) if test_init_state.strip() else {}
        except (json.JSONDecodeError, ValueError):
            return suspects

        state_b = copy.deepcopy(state_a)
        for k, v in state_b.items():
            if isinstance(v, list):
                state_b[k] = list(reversed(v)) + [{"__DIFF__": True}]
            elif isinstance(v, dict):
                state_b[k] = {**v, "__DIFF__": True}
            elif isinstance(v, str):
                state_b[k] = v + "_MODIFIED"
            elif isinstance(v, (int, float)):
                state_b[k] = v * 2 + 1
            elif isinstance(v, bool):
                state_b[k] = not v

        spec_params: dict[str, dict] = {}
        for s in tool_specs:
            props = s.get("parameters", {}).get("properties", {})
            required = set(s.get("parameters", {}).get("required", []))
            spec_params[s.get("name", "")] = {"props": props, "required": required}

        def _make_args(method_name: str) -> dict:
            sp = spec_params.get(method_name, {})
            props = sp.get("props", {})
            required = sp.get("required", set())
            args = {}
            type_defaults = {
                "string": "test",
                "integer": 1,
                "number": 1.0,
                "boolean": True,
                "array": [],
                "object": {},
            }
            for pname, pdef in props.items():
                if pname not in required:
                    continue
                ptype = pdef.get("type", "string")
                enum = pdef.get("enum")
                if enum:
                    args[pname] = enum[0]
                else:
                    args[pname] = type_defaults.get(ptype, "test")
            return args

        write_prefixes = ("create_", "add_", "insert_", "set_", "update_",
                          "delete_", "remove_", "move_", "copy_", "rename_",
                          "write_", "save_", "send_", "post_", "put_",
                          "mark_", "archive_", "pin_", "unpin_", "toggle_")

        confirmed: list[dict] = []
        try:
            cls = execute_class_code(py_code)
            if cls is None:
                return suspects
        except Exception:
            return suspects

        for suspect in suspects:
            method_name = suspect["method"]
            is_write_op = any(method_name.startswith(p) for p in write_prefixes)
            if is_write_op and suspect["issue_type"] == "HARDCODED_DATA":
                hc = suspect.get("hardcoded_collections", [])
                all_small_response = all(
                    "entries)" in h and int(h.split(", ")[1].split(" ")[0]) <= 5
                    for h in hc if "dict" in h
                )
                if all_small_response:
                    continue

            try:
                env_a = cls(initial_state=copy.deepcopy(state_a))
                env_b = cls(initial_state=copy.deepcopy(state_b))
                mapping_a = env_a.function_name_mapping()
                mapping_b = env_b.function_name_mapping()
                if method_name not in mapping_a or method_name not in mapping_b:
                    continue
                args = _make_args(method_name)
                out_a = mapping_a[method_name](**args)
                out_b = mapping_b[method_name](**args)
                if out_a == out_b:
                    suspect["confirmed"] = True
                    suspect["detail"] += " [CONFIRMED: identical output with different states]"
                    confirmed.append(suspect)
            except Exception:
                continue

        return confirmed

    @staticmethod
    def _build_state_awareness_report(confirmed_issues: list[dict]) -> str:
        """Format confirmed state-awareness issues into a fix report."""
        lines = [
            "STATE-AWARENESS VIOLATIONS: Methods using hardcoded data instead of self.state\n",
            "The following methods ignore self.state and use inline hardcoded data.",
            "This is a CRITICAL bug: during trajectory generation, these methods will",
            "return the same output regardless of the initial state provided.\n",
        ]
        for i, issue in enumerate(confirmed_issues, 1):
            lines.append(f"{i}. [{issue['issue_type']}] {issue['method']}:")
            lines.append(f"   - self.state references: {issue['state_refs']}")
            if issue.get("hardcoded_collections"):
                lines.append(f"   - Hardcoded collections: {', '.join(issue['hardcoded_collections'])}")
            lines.append(f"   - {issue['detail']}")
            lines.append(
                f"   - Required fix: Move the data into self.state in __init__ "
                f"and read from self.state in {issue['method']}(). "
                f"The method output MUST change when self.state changes."
            )
            lines.append("")
        lines.append(
            "IMPORTANT: Each method must read its data from self.state, NOT from inline "
            "constants or hardcoded dictionaries/lists. The __init__ default state should "
            "include realistic default values for all data these methods need."
        )
        return "\n".join(lines)

    async def _validate_and_fix_params(
        self,
        py_code: str,
        tool_specs: list[dict],
        tool_specs_str: str,
        progress,
    ) -> tuple[str, bool]:
        """Validate parameter names against specs; fix with the fixer agent if mismatched.

        Returns ``(fixed_code, success_bool)``.
        """
        for _ in range(self.max_fix_iterations):
            mismatches = self._check_param_mismatches(py_code, tool_specs)
            if not mismatches:
                return py_code, True

            report = "PARAMETER NAME MISMATCHES between tool specs and Python methods:\n\n"
            for m in mismatches:
                report += f"Tool: {m['tool']}\n"
                if m["spec_only"]:
                    report += f"  In spec but NOT in method: {m['spec_only']}\n"
                if m["method_only"]:
                    report += f"  In method but NOT in spec: {m['method_only']}\n"
            report += (
                "\nThe Python method signatures MUST match the tool spec parameter names exactly. "
                "Rename the method parameters and update all references in the method body accordingly."
            )

            fix_msgs = [{"role": "user", "content": FIX_USER.format(
                ANALYSIS=report, PY_CODE=py_code,
            )}]
            fix_res = await self.fix_agent.run(fix_msgs, progress)
            fixed = TagParser.extract_first("modified_code", fix_res["content"])

            if fixed and fixed.strip():
                fixed = self._strip_markdown_fences(fixed)
                try:
                    execute_class_code(fixed)
                    py_code = fixed
                except Exception:
                    pass  # Fix didn't compile; try again next iteration

        # Final check after max iterations
        mismatches = self._check_param_mismatches(py_code, tool_specs)
        return py_code, len(mismatches) == 0

    @staticmethod
    def _tests_actually_ran(test_output: str) -> bool:
        """Return True if *test_output* contains real PASS/FAIL lines (not crash output)."""
        if not test_output or not test_output.strip():
            return False
        out = test_output.upper()
        has_results = "PASS:" in out or "FAIL:" in out or "SUMMARY:" in out
        is_crash = (
            "TEST EXECUTION ERROR:" in out
            or "WARNING: NO TEST OUTPUT" in out
            or "ENVIRONMENT CODE EXECUTION ERROR:" in out
        )
        return has_results and not is_crash

    @staticmethod
    def _strip_markdown_fences(code: str) -> str:
        """Strip all markdown code fences from LLM-generated code."""
        import re
        code = re.sub(r'^```(?:python|py)?\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        return code.strip()

    def _execute_tests(self, env_code: str, test_code: str) -> str:
        """Execute *test_code* in an isolated namespace; capture and return stdout/stderr."""
        try:
            test_code = self._strip_markdown_fences(test_code)
            env_code = self._strip_markdown_fences(env_code)

            module = types.ModuleType("test_env")
            exec(env_code, module.__dict__)  # noqa: S102

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured = io.StringIO()
            sys.stderr = captured

            try:
                exec(test_code, module.__dict__)  # noqa: S102

                # Auto-discover and call test functions if nothing was printed
                test_funcs = [
                    (name, obj)
                    for name, obj in module.__dict__.items()
                    if callable(obj)
                    and (
                        name.startswith("test_")
                        or name in ("run_tests", "main")
                    )
                    and isinstance(obj, types.FunctionType)
                ]
                if test_funcs and not captured.getvalue().strip():
                    for name, func in test_funcs:
                        try:
                            sig = inspect.signature(func)
                            params = [
                                p for p in sig.parameters.values()
                                if p.default is inspect.Parameter.empty
                                and p.kind not in (
                                    inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD,
                                )
                            ]
                            if len(params) == 0:
                                func()
                            else:
                                captured.write(
                                    f"\nFAIL: {name} - test function requires "
                                    f"{len(params)} argument(s) but must take none"
                                )
                        except SystemExit:
                            captured.write(f"\nFAIL: {name}\nTest called sys.exit()")
                        except Exception:
                            captured.write(f"\nFAIL: {name}\n{traceback.format_exc()}")

            except SystemExit as exc:
                captured.write(f"\nTest execution error: test code called sys.exit({exc.code})")
            except Exception:
                captured.write(f"\nTest execution error:\n{traceback.format_exc()}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            output = captured.getvalue()
            if not output.strip():
                output = "WARNING: No test output produced. Tests may not have executed."
            return output

        except Exception:
            return f"Environment code execution error:\n{traceback.format_exc()}"
