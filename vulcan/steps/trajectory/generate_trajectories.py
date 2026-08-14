"""Unified trajectory generation step.

A single GenerateTrajectoriesStep that covers static, dynamic, and thinking trajectory
generation, routing based on environment type and model config.

Routing strategy:
    env_type  = self.env.get_environment_type(category)  → "tools_only" | "with_user_and_policy" | "with_user"
    thinking  = step_config.thinking_model or inp.thinking_mode
    → six execution paths: tools_only/with_user_and_policy/with_user x native/reasoning
"""

from __future__ import annotations

import copy
import json
import re
import traceback
from datetime import datetime

from vulcan.steps import register_step
from vulcan.steps.base import BaseStep
from vulcan.core.code_exec import execute_class_code
from vulcan.core.tool_exec import execute_tool, normalize_schema
from vulcan.core.exceptions import EnvironmentError as VulcanEnvironmentError
from vulcan.prompts.generate_trajectories import (
    DYNAMIC_POLICY_SYSTEM,
    DYNAMIC_NO_POLICY_SYSTEM,
    PROMPTING_SYSTEM,
    SINGLE_TURN_SYSTEM,
    USER_AGENT_SYSTEM,
)

from ._tool_call_parser import parse_tool_calls, parse_prompting_tool_calls


MAX_ACTIONS_PER_TURN = 30


@register_step("generate_trajectories")
class GenerateTrajectoriesStep(BaseStep):
    """Unified trajectory generation for static, dynamic, and thinking models.

    Reads from inp:
        tool_code   - environment class source code
        initial_state     - dict passed to environment constructor
        query             - user query / scenario dict or list of turn queries
        query_type        - optional hint ("fixed", "scenario")
        category_name     - used to look up environment_type and api catalog
        mode              - "class_multi_turn" (default) or "class_single_turn"

    Writes to inp:
        conversation     - trajectory as list of role/content dicts
        thinking_mode     - True if reasoning model was used (thinking path)
    """

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.max_turns = self.step_config.get("max_turns", 100)
        self.max_tool_calls_per_turn = self.step_config.get("max_tool_calls_per_turn", MAX_ACTIONS_PER_TURN)
        self.terminate_tools = env.get_terminate_tools()

    # ── Entry point ──────────────────────────────────────────────────────────

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            mode = inp.get("mode", "class_multi_turn")
            cat = inp.get("category_name", "")

            if mode != "class_multi_turn":
                await self._run_single_turn(inp, mode, progress)
                return inp

            env_type = self.env.get_environment_type(cat)
            style = self._resolve_style(inp)
            inp["tool_call_format"] = style  # stamp for downstream verification routing

            if env_type == "tools_only":
                if style == "tagged":
                    await self._run_static_prompting(inp, progress)
                elif style == "reasoning":
                    await self._run_static_thinking(inp, progress)
                else:
                    await self._run_static_fc(inp, progress)
            elif env_type == "with_user_and_policy":
                if style == "reasoning":
                    await self._run_dynamic_thinking(inp, cat, progress)
                else:
                    await self._run_dynamic_fc(inp, cat, progress)
            else:  # with_user
                if style == "reasoning":
                    await self._run_with_user_reasoning(inp, cat, progress)
                else:
                    await self._run_with_user_native(inp, cat, progress)

        except Exception:
            traceback.print_exc()

        return inp

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _init_environment(self, inp: dict):
        """Instantiate environment from tool_code source.

        Returns app instance, or raises on failure (caller writes error msg).
        """
        py_code = inp.get("tool_code", "")
        init_state = inp.get("initial_state", {})
        cls = execute_class_code(py_code)
        try:
            app = cls(initial_state=copy.deepcopy(init_state))
        except TypeError:
            app = cls()
        return app

    def _init_environment_dynamic(self, inp: dict):
        """Instantiate environment for dynamic mode (may have _load_scenario)."""
        py_code = inp.get("tool_code", "")
        init_state = inp.get("initial_state", {})
        query = inp.get("query", {})
        cls = execute_class_code(py_code)
        try:
            app = cls(initial_state=copy.deepcopy(init_state))
        except TypeError:
            app = cls()
        if hasattr(app, "_load_scenario"):
            scenario_text = (
                query.get("scenario", query.get("instruction", ""))
                if isinstance(query, dict)
                else str(query)
            )
            app._load_scenario(scenario_text, init_state)
        return app

    def _build_tool_specs(self, cat: str, include_transfer: bool = False) -> list[dict]:
        """Build OpenAI-style tool spec list from environment API catalog."""
        api_specs = self.env.get_api_catalog(cat)
        tool_specs = [{"type": "function", "function": a["function"]} for a in api_specs]
        if include_transfer:
            tool_specs.append({"type": "function", "function": self.env.TRANSFER_TO_HUMAN_API})
        return tool_specs

    def _build_native_tool_specs(self, cat: str) -> list[dict]:
        """Build native OpenAI FC tool specs with schema normalization."""
        api_specs = self.env.get_api_catalog(cat)
        native_tools = []
        for a in api_specs:
            func = copy.deepcopy(a["function"])
            normalize_schema(func)
            native_tools.append({"type": "function", "function": func})
        return native_tools

    def _build_thinking_tool_section(self, tool_specs: list[dict]) -> str:
        """Build the text-based tool section for thinking-model system prompts."""
        fun_str = json.dumps([t["function"] for t in tool_specs], indent=2)
        return (
            "You are provided with function signatures within <tool> tags.\n"
            f"<tool>\n{fun_str}\n</tool>\n\n"
            "When using tools, return calls in a JSON array:\n"
            "<tool_call>\n"
            '[{name: "tool_name", arguments: {arg1: "value1"}}]\n'
            "</tool_call>"
        )

    def _build_prompting_tool_section(self, tool_specs: list[dict]) -> str:
        """Render the available tools for the prompting-style system prompt.

        One compact JSON object per line (name / description / parameters), matching
        the prompting sample format inside the <tools> block.
        """
        lines = []
        for t in tool_specs:
            fn = t.get("function", {})
            compact = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
            lines.append(json.dumps(compact, ensure_ascii=False))
        return "\n".join(lines)

    def _resolve_style(self, inp: dict) -> str:
        """Resolve the trajectory generation style: 'native' | 'reasoning' | 'tagged'.

        Priority: explicit ``tool_call_format`` on this step's config or on the
        item > ``thinking_model`` / ``thinking_mode`` (-> 'reasoning') > 'native'.

        Note that ``vulcan.cli._resolve_output_label`` re-implements this
        fallback but consults only ``defaults.thinking_model``, so setting
        ``thinking_model`` under ``steps.generate_trajectories`` selects the
        reasoning path here while the output folder is still named 'native'.
        """
        style = self.step_config.get("tool_call_format") or inp.get("tool_call_format")
        if style:
            return style
        if self.thinking_model or inp.get("thinking_mode", False):
            return "reasoning"
        return "native"

    @staticmethod
    def _normalize_turn_queries(query) -> list[str]:
        """Normalize a query (str | dict | list) into a list of per-turn query strings."""
        if isinstance(query, list):
            return [q.get("query", str(q)) if isinstance(q, dict) else str(q) for q in query]
        if isinstance(query, dict):
            if "instruction" in query:
                return [query["instruction"]]
            if "query" in query:
                return [query["query"]]
            return [str(query)]
        return [str(query)]

    @staticmethod
    def _clean_prompting_content(content: str) -> str:
        """Strip hallucinated environment output from a prompting-style response.

        Prompting models (which are not trained for this protocol) sometimes
        continue past their own turn and fabricate ``<tool_response>`` blocks. We
        remove any fully-formed ``<tool_response>...</tool_response>`` blocks while
        preserving the model's real text/tool-calls that may follow them, then drop
        a trailing unclosed ``<tool_response>`` (output that ran into fake results).
        """
        if not content:
            return ""
        content = re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)
        cut = content.find("<tool_response>")
        if cut != -1:
            content = content[:cut]
        return content.strip()

    async def _exec_tool(self, app, name: str, args: dict) -> str:
        """Execute a tool, returning JSON string result. Never raises."""
        try:
            return await execute_tool(app, name, args)
        except VulcanEnvironmentError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Static FC ────────────────────────────────────────────────────────────

    async def _run_static_fc(self, inp: dict, progress) -> None:
        """Static multi-turn with native OpenAI function calling."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})
        cat = inp.get("category_name", "")

        if not py_code or not query:
            return

        try:
            app = self._init_environment(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        native_tools = self._build_native_tool_specs(cat)
        system = (
            "You are a helpful assistant with access to tools. Use them to fulfill the user's request.\n\n"
            "You may call one or more functions to assist with the user query.\n"
            "At each turn, you should try your best to complete the tasks requested by the user within "
            "the current turn. Continue to output functions to call until you have fulfilled the user's "
            "request to the best of your ability. Once you have no more functions to call, the system will "
            "consider the current turn complete and proceed to the next turn or task."
        )

        # Normalize queries into a list of turn queries
        if isinstance(query, list):
            turn_queries = [q.get("query", str(q)) if isinstance(q, dict) else str(q) for q in query]
        elif isinstance(query, dict):
            if "instruction" in query:
                turn_queries = [query["instruction"]]
            elif "query" in query:
                turn_queries = [query["query"]]
            else:
                turn_queries = [str(query)]
        else:
            turn_queries = [str(query)]

        agent = self.make_agent("traj_agent", system)

        messages = [{"role": "system", "content": system}]
        trajectory = [{"role": "system", "content": system}]

        for turn_idx, turn_query in enumerate(turn_queries):
            messages.append({"role": "user", "content": turn_query})
            trajectory.append({"role": "user", "content": turn_query})

            for _action in range(self.max_turns):
                res = await agent.run(messages, progress, tools=native_tools)

                if res.get("tool_calls"):
                    assistant_msg = {
                        "role": "assistant",
                        "content": res.get("content", ""),
                        "tool_calls": res["tool_calls"],
                    }
                    trajectory.append(assistant_msg)
                    messages.append(assistant_msg)

                    for tc in res["tool_calls"]:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        call_id = tc.get("id", "")

                        if name in self.terminate_tools:
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps({"status": "transferred"}),
                            }
                            trajectory.append(tool_msg)
                            messages.append(tool_msg)
                            inp["conversation"] = trajectory
                            return

                        result = await self._exec_tool(app, name, args)
                        tool_msg = {"role": "tool", "tool_call_id": call_id, "content": result}
                        trajectory.append(tool_msg)
                        messages.append(tool_msg)

                    continue  # more tool calls possible this turn

                content = res.get("content", "")
                if content:
                    trajectory.append({"role": "assistant", "content": content})
                    messages.append({"role": "assistant", "content": content})
                break

        inp["conversation"] = trajectory

    # ── Static thinking ──────────────────────────────────────────────────────

    async def _run_static_thinking(self, inp: dict, progress) -> None:
        """Static multi-turn with text-based tool calls (reasoning model)."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})
        query_type = inp.get("query_type", "")
        cat = inp.get("category_name", "")

        if not py_code or not query:
            return

        try:
            app = self._init_environment(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        tool_specs = self._build_tool_specs(cat, include_transfer=False)
        tool_section = self._build_thinking_tool_section(tool_specs)
        system = (
            "You are a helpful assistant. You may call one or more functions to assist "
            "with the user query.\n\n"
            f"{tool_section}\n\n"
            "Complete the user's request. Call all needed functions before responding."
        )

        agent = self.make_agent("thinking_agent", system)

        # Route sub-path: fixed queries vs single query
        if "fixed" in query_type or (
            isinstance(query, list) and all(isinstance(q, dict) and "query" in q for q in query)
        ):
            trajectory = await self._run_fixed_queries(agent, app, query, system, progress)
        else:
            query_text = (
                query.get("query", query.get("instruction", "")) if isinstance(query, dict) else str(query)
            )
            trajectory = await self._run_single_query(agent, app, query_text, system, progress)

        inp["conversation"] = trajectory
        inp["thinking_mode"] = True

    # ── Static prompting (text-based tool calls, no reasoning) ───────────────

    async def _run_static_prompting(self, inp: dict, progress) -> None:
        """Static multi-turn with text-based tool calls in the prompting style.

        Differs from the thinking path: no reasoning model and no reasoning_content
        — every tool call is parsed straight from the assistant's text output. The
        system prompt and trajectory format follow the prompting sample format:
        the assistant writes a short explanation then emits one or more
        ``<tool_call>{...}</tool_call>`` tags; tool results are fed back as a *user*
        message wrapped in ``<tool_response>...</tool_response>``; the final answer is
        wrapped by the model in ``<answer>...</answer>``.
        """
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})
        cat = inp.get("category_name", "")

        if not py_code or not query:
            return

        try:
            app = self._init_environment(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        tool_specs = self._build_tool_specs(cat, include_transfer=False)
        system = PROMPTING_SYSTEM.format(TOOLS=self._build_prompting_tool_section(tool_specs))

        # Plain (non-thinking) agent: everything is returned in `content`.
        agent = self.make_agent("prompting_agent", system)

        turn_queries = self._normalize_turn_queries(query)

        messages = [{"role": "system", "content": system}]
        trajectory = [{"role": "system", "content": system}]

        for turn_query in turn_queries:
            messages.append({"role": "user", "content": turn_query})
            trajectory.append({"role": "user", "content": turn_query})

            answered = False
            empty_retries = 0
            for _action in range(self.max_tool_calls_per_turn):
                res = await agent.run(messages, progress)
                content = res.get("content", "")

                # Strip hallucinated environment output (some prompting models continue
                # past their turn and fabricate <tool_response> blocks), while keeping any
                # real text/tool-calls. Some models reject `stop`, so this is done here.
                content = self._clean_prompting_content(content)

                if not content:
                    empty_retries += 1
                    if empty_retries <= self.empty_response_retries:
                        continue
                    break
                empty_retries = 0

                # Record the assistant's explanation + tool_call tags verbatim.
                trajectory.append({"role": "assistant", "content": content})
                messages.append({"role": "assistant", "content": content})

                tool_calls = parse_prompting_tool_calls(content)

                if not tool_calls:
                    # Terminal text response: this is how a task ends in the prompting
                    # style — the model stops calling tools and replies directly. Ensure
                    # the final message is wrapped in <answer>...</answer>; if the model
                    # omitted the tags, wrap its text so the trajectory always terminates
                    # with an explicit answer.
                    if not ("<answer>" in content and "</answer>" in content):
                        wrapped = f"<answer>\n{content.strip()}\n</answer>"
                        trajectory[-1]["content"] = wrapped
                        messages[-1]["content"] = wrapped
                    answered = True
                    break

                # The model emitted tool calls, so it is not done yet. Drop any premature
                # <answer> it produced before seeing results — the real final answer comes
                # once the tool outputs are returned — so answers only appear as the final,
                # tool-less message.
                if "<answer>" in content:
                    stripped = re.sub(r"<answer>.*?</answer>", "", content, flags=re.DOTALL).rstrip()
                    trajectory[-1]["content"] = stripped
                    messages[-1]["content"] = stripped

                response_blocks = []
                terminated = False
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}

                    if name in self.terminate_tools:
                        terminated = True
                        response_blocks.append("<tool_response>\ntransferred\n</tool_response>")
                        break

                    result = await self._exec_tool(app, name, args)
                    response_blocks.append(f"<tool_response>\n{result}\n</tool_response>")

                resp_text = "\n".join(response_blocks)
                trajectory.append({"role": "user", "content": resp_text})
                messages.append({"role": "user", "content": resp_text})

                # A hand-off tool ends the task immediately.
                if terminated:
                    answered = True
                    break

            # Finalization: never leave a turn ending on a tool response. If no final
            # answer was produced (e.g. the model exhausted retries or hit the action
            # cap), nudge it once for a closing answer and wrap it in <answer> tags.
            if not answered:
                messages.append({
                    "role": "user",
                    "content": "Provide your final response to the user between "
                               "<answer> and </answer> tags.",
                })
                res = await agent.run(messages, progress)
                content = self._clean_prompting_content(res.get("content", ""))
                if content:
                    if not ("<answer>" in content and "</answer>" in content):
                        content = f"<answer>\n{content.strip()}\n</answer>"
                    trajectory.append({"role": "assistant", "content": content})

        inp["conversation"] = trajectory

    # ── Dynamic FC ────────────────────────────────────────────────────────────

    async def _run_dynamic_fc(self, inp: dict, cat: str, progress) -> None:
        """with_user_and_policy, multi-turn: tool calls + simulated user + policy."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})

        if not py_code or not query:
            return

        try:
            app = self._init_environment_dynamic(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        tool_specs = self._build_tool_specs(cat, include_transfer=True)
        domain_rules = self.env.get_domain_rules(cat) or ""
        system = DYNAMIC_POLICY_SYSTEM.format(DOMAIN_RULES=domain_rules)

        assistant_model = self.step_config.get("model", self.model)
        assistant_temp = self.step_config.get("temperature", self.temperature)
        simulated_user_model = self.step_config.get("simulated_user_model", self.model)
        user_temp = self.step_config.get("simulated_user_temperature", self.temperature)
        simulated_user_name = self.step_config.get("simulated_user_name", "user_sim")
        simulated_user_max_tokens = self.step_config.get("simulated_user_max_tokens", self.max_tokens)

        agent = self.make_agent(
            "service_agent", system, model=assistant_model, temperature=assistant_temp
        )

        scenario_text = (
            query.get("scenario", query.get("instruction", "")) if isinstance(query, dict) else str(query)
        )
        user_system = USER_AGENT_SYSTEM.format(SCENARIO=scenario_text)
        user_agent = self.make_agent(
            simulated_user_name, user_system, model=simulated_user_model,
            temperature=user_temp, max_tokens=simulated_user_max_tokens,
        )

        user_messages = [
            {"role": "system", "content": user_system},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        first_res = await user_agent.run(user_messages, progress)
        first_user_msg = first_res.get("content", "")
        user_messages.append({"role": "assistant", "content": first_user_msg})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]
        trajectory = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]

        for _turn in range(self.max_turns):
            res = await agent.run(messages, progress, tools=tool_specs)
            content = res.get("content", "")
            fc_tool_calls = res.get("tool_calls", None)

            if fc_tool_calls:
                asst_msg = {"role": "assistant", "content": content or "", "tool_calls": fc_tool_calls}
                messages.append(asst_msg)
                trajectory.append(asst_msg)

                terminated = False
                for tc in fc_tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_call_id = tc.get("id", "")

                    if name in self.terminate_tools:
                        # Tool-agnostic, matching the other terminate branches:
                        # `terminate_tools` is user-configurable, so the response
                        # must not name one particular tool.
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"status": "transferred"}),
                        }
                        messages.append(tool_msg)
                        trajectory.append(tool_msg)
                        terminated = True
                        break

                    result = await self._exec_tool(app, name, args)
                    tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                    messages.append(tool_msg)
                    trajectory.append(tool_msg)

                if terminated:
                    break

            elif content:
                asst_msg = {"role": "assistant", "content": content}
                messages.append(asst_msg)
                trajectory.append(asst_msg)

                user_messages.append({"role": "user", "content": content})
                user_res = await user_agent.run(user_messages, progress)
                user_content = user_res.get("content", "")
                user_messages.append({"role": "assistant", "content": user_content})

                if "###STOP###" in user_content:
                    trajectory.append({"role": "user", "content": user_content})
                    break

                user_msg = {"role": "user", "content": user_content}
                messages.append(user_msg)
                trajectory.append(user_msg)
            else:
                break

        inp["conversation"] = trajectory

    # ── Dynamic no-policy FC (uses DYNAMIC_NO_POLICY_SYSTEM) ─────────────────

    async def _run_with_user_native(self, inp: dict, cat: str, progress) -> None:
        """with_user, multi-turn: tool calls + simulated user, no policy."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})

        if not py_code or not query:
            return

        try:
            app = self._init_environment_dynamic(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        # with_user: NO transfer tool added
        tool_specs = self._build_tool_specs(cat, include_transfer=False)
        system = DYNAMIC_NO_POLICY_SYSTEM.format(TIME=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        assistant_model = self.step_config.get("model", self.model)
        assistant_temp = self.step_config.get("temperature", self.temperature)
        simulated_user_model = self.step_config.get("simulated_user_model", self.model)
        user_temp = self.step_config.get("simulated_user_temperature", self.temperature)
        simulated_user_name = self.step_config.get("simulated_user_name", "user_sim")
        simulated_user_max_tokens = self.step_config.get("simulated_user_max_tokens", self.max_tokens)

        agent = self.make_agent(
            "service_agent", system, model=assistant_model, temperature=assistant_temp
        )

        scenario_text = (
            query.get("scenario", query.get("instruction", "")) if isinstance(query, dict) else str(query)
        )
        user_system = USER_AGENT_SYSTEM.format(SCENARIO=scenario_text)
        user_agent = self.make_agent(
            simulated_user_name, user_system, model=simulated_user_model,
            temperature=user_temp, max_tokens=simulated_user_max_tokens,
        )

        user_messages = [
            {"role": "system", "content": user_system},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        first_res = await user_agent.run(user_messages, progress)
        first_user_msg = first_res.get("content", "")
        user_messages.append({"role": "assistant", "content": first_user_msg})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]
        trajectory = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]

        for _turn in range(self.max_turns):
            res = await agent.run(messages, progress, tools=tool_specs)
            content = res.get("content", "")
            fc_tool_calls = res.get("tool_calls", None)

            if fc_tool_calls:
                asst_msg = {"role": "assistant", "content": content or "", "tool_calls": fc_tool_calls}
                messages.append(asst_msg)
                trajectory.append(asst_msg)

                for tc in fc_tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_call_id = tc.get("id", "")

                    result = await self._exec_tool(app, name, args)
                    tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                    messages.append(tool_msg)
                    trajectory.append(tool_msg)

            elif content:
                if "###STOP###" in content:
                    trajectory.append({"role": "assistant", "content": content})
                    break

                asst_msg = {"role": "assistant", "content": content}
                messages.append(asst_msg)
                trajectory.append(asst_msg)

                user_messages.append({"role": "user", "content": content})
                user_res = await user_agent.run(user_messages, progress)
                user_content = user_res.get("content", "")
                user_messages.append({"role": "assistant", "content": user_content})

                if "###STOP###" in user_content:
                    trajectory.append({"role": "user", "content": user_content})
                    break

                user_msg = {"role": "user", "content": user_content}
                messages.append(user_msg)
                trajectory.append(user_msg)
            else:
                break

        inp["conversation"] = trajectory

    # ── Dynamic thinking ──────────────────────────────────────────────────────

    async def _run_dynamic_thinking(self, inp: dict, cat: str, progress) -> None:
        """with_user_and_policy, thinking model: policy-governed + simulated user."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})

        if not py_code or not query:
            return

        try:
            app = self._init_environment(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        tool_specs = self._build_tool_specs(cat, include_transfer=True)
        tool_section = self._build_thinking_tool_section(tool_specs)
        domain_rules = self.env.get_domain_rules(cat) or ""
        system = (
            "<instructions>\n"
            "You are a customer service agent. Follow the policy below strictly.\n\n"
            "Each turn you must do exactly ONE of:\n"
            "- Send a message to the user (ask for info, confirm details, provide updates)\n"
            "- Make a tool call (look up account, process request, update records)\n\n"
            "Never do both in the same turn. Always follow the policy rules.\n"
            "</instructions>\n\n"
            f"<policy>\n{domain_rules}\n</policy>\n\n"
            f"{tool_section}"
        )

        agent = self.make_agent("thinking_agent", system)

        scenario_text = (
            query.get("scenario", query.get("instruction", "")) if isinstance(query, dict) else str(query)
        )
        trajectory = await self._run_user_scenario(agent, app, scenario_text, system, inp, progress)
        inp["conversation"] = trajectory
        inp["thinking_mode"] = True

    # ── Dynamic no-policy thinking ────────────────────────────────────────────

    async def _run_with_user_reasoning(self, inp: dict, cat: str, progress) -> None:
        """with_user, thinking model: simulated user, no policy."""
        py_code = inp.get("tool_code", "")
        query = inp.get("query", {})

        if not py_code or not query:
            return

        try:
            app = self._init_environment(inp)
        except Exception as e:
            inp["conversation"] = [{"role": "error", "content": f"Env init failed: {e}"}]
            return

        tool_specs = self._build_tool_specs(cat, include_transfer=False)
        tool_section = self._build_thinking_tool_section(tool_specs)
        system = (
            f"# Environment\n"
            f"- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "# Tool Usage Guidelines:\n"
            "- When the user's needs require using tools to complete, first determine whether all parameter information is known. "
            "If it is known, extract the corresponding parameters, otherwise ask the user for the relevant parameter values\n"
            "- When the user cannot provide relevant information, first obtain relevant information through tools\n"
            "- Complete tasks based on Precondition and Postcondition\n\n"
            "# Conversation Guidelines\n"
            "- Only use information from the above context, prohibit constructing information without basis and replying to users\n"
            "- Focus on completing user needs, prohibit divergent guidance to users to propose new needs\n"
            "- After completing the user's task requirements, ask if there are any other needs. "
            "If the user indicates no, generate '###STOP###' mark to end the conversation\n\n"
            f"{tool_section}"
        )

        agent = self.make_agent("thinking_agent", system)

        scenario_text = (
            query.get("scenario", query.get("instruction", "")) if isinstance(query, dict) else str(query)
        )
        trajectory = await self._run_user_scenario(agent, app, scenario_text, system, inp, progress)
        inp["conversation"] = trajectory
        inp["thinking_mode"] = True

    # ── Single-turn ──────────────────────────────────────────────────────────

    async def _run_single_turn(self, inp: dict, mode: str, progress) -> None:
        """Single-turn trajectory without environment execution."""
        api_info = inp.get("api_info", [])
        fun_str = json.dumps(api_info, indent=2)
        system = SINGLE_TURN_SYSTEM.format(FUN=fun_str)
        query = inp.get("query", "")

        agent = self.make_agent("traj_agent", system)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(query)},
        ]
        res = await agent.run(messages, progress)

        trajectory = list(messages)
        trajectory.append({"role": "assistant", "content": res.get("content", "")})
        inp["conversation"] = trajectory

    # ── Thinking sub-routines ────────────────────────────────────────────────

    async def _run_fixed_queries(self, agent, app, queries, system: str, progress) -> list[dict]:
        """Multi-turn with fixed queries for thinking model. One query per turn."""
        if isinstance(queries, list):
            turn_queries = [q.get("query", str(q)) if isinstance(q, dict) else str(q) for q in queries]
        elif isinstance(queries, dict) and "instruction" in queries:
            turn_queries = [queries["instruction"]]
        else:
            turn_queries = [str(queries)]

        messages = [{"role": "system", "content": system}]
        trajectory = [{"role": "system", "content": system}]

        for turn_idx, turn_query in enumerate(turn_queries):
            messages.append({"role": "user", "content": turn_query})
            trajectory.append({"role": "user", "content": turn_query, "turn": turn_idx + 1})

            empty_retries = 0
            for action in range(self.max_tool_calls_per_turn):
                res = await agent.run(messages, progress)
                content = res.get("content", "")
                reasoning = res.get("reasoning_content", "")

                if not content:
                    empty_retries += 1
                    if empty_retries <= self.empty_response_retries:
                        continue
                    break
                empty_retries = 0

                traj_entry: dict = {"role": "assistant", "content": content, "turn": turn_idx + 1}
                if reasoning:
                    traj_entry["reasoning_content"] = reasoning
                trajectory.append(traj_entry)

                tool_calls = parse_tool_calls(content)

                if tool_calls:
                    tool_results = []
                    terminated = False

                    for tc in tool_calls:
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})

                        if name in self.terminate_tools:
                            terminated = True
                            tool_results.append({"tool": name, "result": "transferred"})
                            break

                        result = await self._exec_tool(app, name, args)
                        tool_results.append({"tool": name, "result": result})

                    results_str = json.dumps(tool_results, indent=2)
                    trajectory.append({"role": "tool", "content": results_str, "turn": turn_idx + 1})
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Output of tool calls:\n{results_str}"})

                    if terminated:
                        return trajectory

                    if action == self.max_tool_calls_per_turn - 1:
                        break
                else:
                    messages.append({"role": "assistant", "content": content})
                    break

        return trajectory

    async def _run_user_scenario(self, agent, app, scenario_text: str, system: str, inp: dict, progress) -> list[dict]:
        """Dynamic with user agent simulating the scenario (thinking model)."""
        user_system = USER_AGENT_SYSTEM.format(SCENARIO=scenario_text)
        simulated_user_model = self.step_config.get("simulated_user_model", self.model)
        user_temp = self.step_config.get("simulated_user_temperature", self.temperature)
        simulated_user_name = self.step_config.get("simulated_user_name", "user_sim")
        simulated_user_max_tokens = self.step_config.get("simulated_user_max_tokens", self.max_tokens)
        user_agent = self.make_agent(
            simulated_user_name, user_system, model=simulated_user_model,
            temperature=user_temp, max_tokens=simulated_user_max_tokens,
        )

        user_messages = [
            {"role": "system", "content": user_system},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        first_res = await user_agent.run(user_messages, progress)
        first_user_msg = first_res.get("content", "")
        user_messages.append({"role": "assistant", "content": first_user_msg})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]
        trajectory = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_msg},
        ]

        max_turns = self.step_config.get("max_turns", 80)
        for _action in range(max_turns):
            res = await agent.run(messages, progress)
            content = res.get("content", "")
            reasoning = res.get("reasoning_content", "")

            if not content:
                break

            if "###STOP###" in content:
                traj_entry: dict = {"role": "assistant", "content": content}
                if reasoning:
                    traj_entry["reasoning_content"] = reasoning
                trajectory.append(traj_entry)
                break

            traj_entry = {"role": "assistant", "content": content}
            if reasoning:
                traj_entry["reasoning_content"] = reasoning
            trajectory.append(traj_entry)

            tool_calls = parse_tool_calls(content)

            if tool_calls:
                tool_results = []
                terminated = False

                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})

                    if name in self.terminate_tools:
                        terminated = True
                        tool_results.append({"tool": name, "result": "transferred"})
                        break

                    result = await self._exec_tool(app, name, args)
                    tool_results.append({"tool": name, "result": result})

                results_str = json.dumps(tool_results, indent=2)
                trajectory.append({"role": "tool", "content": results_str})
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Output of tool calls:\n{results_str}"})

                if terminated:
                    break
            else:
                user_messages.append({"role": "user", "content": content})
                user_res = await user_agent.run(user_messages, progress)
                user_content = user_res.get("content", "")
                user_messages.append({"role": "assistant", "content": user_content})

                if "###STOP###" in user_content:
                    trajectory.append({"role": "user", "content": user_content})
                    break

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": user_content})
                trajectory.append({"role": "user", "content": user_content})

        return trajectory

    async def _run_single_query(self, agent, app, query_text: str, system: str, progress) -> list[dict]:
        """Single query for thinking model. Up to max_tool_calls_per_turn rounds."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query_text},
        ]
        trajectory = list(messages)

        for _action in range(self.max_tool_calls_per_turn):
            res = await agent.run(messages, progress)
            content = res.get("content", "")
            reasoning = res.get("reasoning_content", "")

            if not content:
                break

            traj_entry: dict = {"role": "assistant", "content": content}
            if reasoning:
                traj_entry["reasoning_content"] = reasoning
            trajectory.append(traj_entry)

            tool_calls = parse_tool_calls(content)

            if tool_calls:
                tool_results = []
                terminated = False

                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})

                    if name in self.terminate_tools:
                        terminated = True
                        tool_results.append({"tool": name, "result": "transferred"})
                        break

                    result = await self._exec_tool(app, name, args)
                    tool_results.append({"tool": name, "result": result})

                results_str = json.dumps(tool_results, indent=2)
                trajectory.append({"role": "tool", "content": results_str})
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Output of tool calls:\n{results_str}"})

                if terminated:
                    break
            else:
                break  # Text response, done

        return trajectory
