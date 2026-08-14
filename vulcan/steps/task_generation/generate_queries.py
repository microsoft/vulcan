"""generate_queries step: Generate user queries.

Routes based on TWO dimensions:
  graph type:  sequence (multi-turn chain) vs multipath (fork-merge diamond)
  domain mode: static (fixed queries) vs dynamic (user scenario)

Produces 4 possible flows:
  sequence + static   → fixed queries, one per turn, logically connected
  sequence + dynamic  → user scenario with per-subgraph motivation
  multipath + static  → single declarative query, multiple valid paths
  multipath + dynamic → user scenario allowing multiple paths
"""

from __future__ import annotations

import json
import os
import random
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError
from ...prompts.generate_queries import (
    # Sequence prompts
    FIXED_QUERIES_SYSTEM,
    FIXED_QUERIES_USER,
    SCENARIO_WITH_POLICY_SYSTEM,
    SCENARIO_WITH_POLICY_USER,
    SCENARIO_NO_POLICY_SYSTEM,
    SCENARIO_NO_POLICY_USER,
    # Multipath prompts
    MULTIPATH_FIXED_SYSTEM,
    MULTIPATH_FIXED_USER,
    MULTIPATH_SCENARIO_SYSTEM,
    MULTIPATH_SCENARIO_USER,
    MULTIPATH_SCENARIO_WITH_POLICY_USER,
    # Few-shot system variants
    FIXED_QUERIES_SYSTEM_FEWSHOT,
    SCENARIO_WITH_POLICY_SYSTEM_FEWSHOT,
    SCENARIO_NO_POLICY_SYSTEM_FEWSHOT,
    MULTIPATH_FIXED_SYSTEM_FEWSHOT,
    MULTIPATH_SCENARIO_SYSTEM_FEWSHOT,
    # Planning
    PLANNING_SYSTEM,
    PLANNING_USER,
    # Shared
    COMPLEXITY_SYSTEM,
    COMPLEXITY_USER,
)


@register_step("generate_queries")
class GenerateQueriesStep(BaseStep):
    """Generate user queries for combined graph + env data items.

    Supports four query types depending on graph type and environment type:
    - sequence/static:    fixed multi-turn queries        (tools_only)
    - sequence/dynamic:   scenario with policy            (with_user_and_policy)
    - sequence/no_policy: scenario without policy         (with_user)
    - multipath/*:        single ambiguous query covering multiple paths
    """

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.max_try: int = self.step_config.get("max_attempts", 5)

        self.query_agent = Agent(
            "query_gen", "", llm,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("query"),
        )
        self.complexity_agent = Agent(
            "complexity", "", llm,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("scenario_complexity"),
        )
        self.planning_agent = Agent(
            "planner", PLANNING_SYSTEM, llm,
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("ground_truth_plan"),
        )

        # Build domain tools lookup: {category: [api_spec_dicts]}
        self.domain_tools_lookup: dict = {}
        for cat in env.get_categories():
            apis = env.get_api_catalog(cat)
            self.domain_tools_lookup[cat] = [
                a.get("normalized_schema") or a.get("normalized_function") or a.get("function", {})
                for a in apis
            ]

        # Few-shot configuration (opt-in). When disabled (default), generation is
        # zero-shot and behaves identically to before — no examples are loaded.
        fs_cfg = self.step_config.get("few_shot", {}) or {}
        self.few_shot_enabled: bool = bool(fs_cfg.get("enabled", False))
        self.few_shot_num: int = int(fs_cfg.get("num_examples", 0) or 0)  # 0 = all
        self.few_shot_examples: dict[str, list] = {}
        if self.few_shot_enabled:
            self.few_shot_examples = self._load_few_shot_examples(fs_cfg)

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            cat = inp.get("category_name", "")
            graph_type = inp.get("type", "sequence")  # "sequence" or "multipath"
            environment_type = self.env.get_environment_type(cat)
            init_state = inp.get("initial_state", {})

            inp["mode"] = "class_multi_turn"
            inp["query_type"] = f"{graph_type}_{environment_type}"

            if graph_type == "multipath":
                await self._handle_multipath(inp, cat, environment_type, init_state, progress)
            else:
                await self._handle_sequence(inp, cat, environment_type, init_state, progress)

            # Generate ground truth plan after query is set
            if inp.get("query") and inp["query"] != "Error":
                await self._gen_ground_truth_plan(inp, cat, init_state, progress)

            # Classify complexity
            if inp.get("query") and inp["query"] != "Error":
                await self._classify_complexity(inp, init_state, progress)

        except StepError:
            raise
        except Exception:
            traceback.print_exc()

        return inp

    # ── Sequence flows ────────────────────────────────────────────────────────

    async def _handle_sequence(
        self,
        inp: dict,
        cat: str,
        environment_type: str,
        init_state,
        progress,
    ) -> None:
        data_list = inp.get("data_list", [])
        for turn in data_list:
            g = turn.get("graph", {})
            turn["graph"] = {"nodes": g.get("nodes", []), "edges": g.get("edges", [])}

        scenario = json.dumps(
            {"number_turns": inp.get("number_turns"), "data_list": data_list},
            indent=2,
        )

        if environment_type == "tools_only":
            await self._gen_fixed_queries(inp, init_state, scenario, progress)
        elif environment_type == "with_user_and_policy":
            await self._gen_scenario_with_policy(inp, cat, init_state, scenario, progress)
        else:
            await self._gen_scenario_no_policy(inp, cat, init_state, scenario, progress)

    async def _gen_fixed_queries(
        self, inp: dict, init_state, scenario: str, progress
    ) -> None:
        """Fixed queries for `tools_only` categories — no simulated user."""
        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    FIXED_QUERIES_SYSTEM, FIXED_QUERIES_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": FIXED_QUERIES_USER.format(
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    SCENARIO=scenario,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error":
                if isinstance(parsed, list) and all(
                    isinstance(q, dict) and "query" in q for q in parsed
                ):
                    inp["query"] = parsed
                    return
                if isinstance(parsed, dict) and "query" in parsed:
                    inp["query"] = [parsed]
                    return
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    async def _gen_scenario_with_policy(
        self, inp: dict, cat: str, init_state, scenario: str, progress
    ) -> None:
        """Scenario for `with_user_and_policy` categories — simulated user plus policy."""
        domain_rules = self.env.get_domain_rules(cat) or ""
        domain_tools = json.dumps(self.domain_tools_lookup.get(cat, []), indent=2)

        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    SCENARIO_WITH_POLICY_SYSTEM, SCENARIO_WITH_POLICY_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": SCENARIO_WITH_POLICY_USER.format(
                    DOMAIN_NAME=cat,
                    DOMAIN_RULES=domain_rules,
                    DOMAIN_TOOLS=domain_tools,
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    SCENARIO=scenario,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error" and isinstance(parsed, dict) and "scenario" in parsed:
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    async def _gen_scenario_no_policy(
        self, inp: dict, cat: str, init_state, scenario: str, progress
    ) -> None:
        """Scenario for `with_user` categories — simulated user, no policy."""
        domain_tools = json.dumps(self.domain_tools_lookup.get(cat, []), indent=2)

        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    SCENARIO_NO_POLICY_SYSTEM, SCENARIO_NO_POLICY_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": SCENARIO_NO_POLICY_USER.format(
                    DOMAIN_NAME=cat,
                    DOMAIN_TOOLS=domain_tools,
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    SCENARIO=scenario,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error" and isinstance(parsed, dict) and "scenario" in parsed:
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    # ── Multipath flows ───────────────────────────────────────────────────────

    async def _handle_multipath(
        self,
        inp: dict,
        cat: str,
        environment_type: str,
        init_state,
        progress,
    ) -> None:
        multipath_info = json.dumps({
            "nodes": inp.get("nodes", []),
            "edges": inp.get("edges", []),
            "multipath_pairs": inp.get("multipath_pairs", []),
            "multipath_strength": inp.get("multipath_strength", 0),
        }, indent=2)

        if environment_type == "tools_only":
            await self._gen_multipath_fixed(inp, init_state, multipath_info, progress)
        elif environment_type == "with_user_and_policy":
            await self._gen_multipath_scenario_with_policy(
                inp, cat, init_state, multipath_info, progress
            )
        else:
            await self._gen_multipath_scenario(inp, init_state, multipath_info, progress)

    async def _gen_multipath_fixed(
        self, inp: dict, init_state, multipath_info: str, progress
    ) -> None:
        """Single declarative query — multiple valid execution paths."""
        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    MULTIPATH_FIXED_SYSTEM, MULTIPATH_FIXED_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": MULTIPATH_FIXED_USER.format(
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    MULTIPATH_INFO=multipath_info,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error" and isinstance(parsed, dict) and "query" in parsed:
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    async def _gen_multipath_scenario(
        self, inp: dict, init_state, multipath_info: str, progress
    ) -> None:
        """User scenario for multipath — multiple valid paths, no policy."""
        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    MULTIPATH_SCENARIO_SYSTEM, MULTIPATH_SCENARIO_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": MULTIPATH_SCENARIO_USER.format(
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    MULTIPATH_INFO=multipath_info,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error" and isinstance(parsed, dict) and "scenario" in parsed:
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    async def _gen_multipath_scenario_with_policy(
        self, inp: dict, cat: str, init_state, multipath_info: str, progress
    ) -> None:
        """User scenario for multipath — multiple valid paths, with policy."""
        domain_rules = self.env.get_domain_rules(cat) or ""
        domain_tools = json.dumps(self.domain_tools_lookup.get(cat, []), indent=2)

        for _ in range(self.max_try):
            msgs = [
                {"role": "system", "content": self._select_system(
                    MULTIPATH_SCENARIO_SYSTEM, MULTIPATH_SCENARIO_SYSTEM_FEWSHOT, inp)},
                {"role": "user", "content": MULTIPATH_SCENARIO_WITH_POLICY_USER.format(
                    DOMAIN_NAME=cat,
                    DOMAIN_RULES=domain_rules,
                    DOMAIN_TOOLS=domain_tools,
                    INITIAL_STATE=json.dumps(init_state, indent=2),
                    MULTIPATH_INFO=multipath_info,
                )},
            ]
            res = await self.query_agent.run(msgs, progress)
            parsed = self._parse_query(res)
            if parsed != "Error" and isinstance(parsed, dict) and "scenario" in parsed:
                inp["query"] = parsed
                return
        inp["query"] = "Error"

    # ── Ground truth planning ─────────────────────────────────────────────────

    async def _gen_ground_truth_plan(
        self, inp: dict, cat: str, init_state, progress
    ) -> None:
        """Generate ground truth plan: exact tool calls with reasoning."""
        query = inp.get("query", "")
        if isinstance(query, (dict, list)):
            query_str = json.dumps(query, indent=2)
        else:
            query_str = str(query)

        tool_specs = json.dumps(self.domain_tools_lookup.get(cat, []), indent=2)

        for _ in range(self.max_try):
            msgs = [{"role": "user", "content": PLANNING_USER.format(
                USER_QUERY=query_str,
                INITIAL_STATE=json.dumps(init_state, indent=2),
                TOOL_SPECS=tool_specs,
            )}]
            res = await self.planning_agent.run(msgs, progress)

            raw = TagParser.extract_first("ground_truth_plan", res["content"])
            if raw:
                try:
                    inp["ground_truth_plan"] = json.loads(raw)
                    if isinstance(inp["ground_truth_plan"], dict):
                        return
                except json.JSONDecodeError:
                    inp["ground_truth_plan"] = raw
                    return

        inp["ground_truth_plan"] = None

    # ── Complexity classification ─────────────────────────────────────────────

    async def _classify_complexity(
        self, inp: dict, init_state, progress
    ) -> None:
        query = inp.get("query", "")
        if isinstance(query, (dict, list)):
            query = json.dumps(query, indent=2)

        if inp.get("data_list"):
            scenario_str = json.dumps(
                {"number_turns": inp.get("number_turns"), "data_list": inp["data_list"]},
                indent=2,
            )
        elif inp.get("nodes"):
            scenario_str = json.dumps(
                {"nodes": inp["nodes"], "multipath_pairs": inp.get("multipath_pairs", [])},
                indent=2,
            )
        else:
            scenario_str = ""

        msgs = [
            {"role": "system", "content": COMPLEXITY_SYSTEM},
            {"role": "user", "content": COMPLEXITY_USER.format(
                USER_SCENARIO=str(query),
                INITIAL_STATE=json.dumps(init_state, indent=2),
                SCENARIO=scenario_str,
            )},
        ]

        for _ in range(self.max_try):
            res = await self.complexity_agent.run(msgs, progress)
            raw = TagParser.extract_first("scenario_complexity", res["content"])
            if raw:
                try:
                    inp["query_complexity"] = json.loads(raw)
                    if isinstance(inp["query_complexity"], dict):
                        return
                except json.JSONDecodeError:
                    # Keep the unparsed text rather than discarding it, and stop —
                    # matching _gen_ground_truth_plan. Without the return, the loop
                    # falls through to the unconditional None below and the value
                    # set here is always thrown away.
                    inp["query_complexity"] = raw
                    return

        inp["query_complexity"] = None

    # ── Few-shot helpers ──────────────────────────────────────────────────────

    def _load_few_shot_examples(self, fs_cfg: dict) -> dict:
        """Load + normalize few-shot examples into {key: [examples]}.

        Keys may be category names ("ticket_api", "hotel_booking", ...), flow keys
        ("sequence_tools_only", "multipath_with_user_and_policy", ...), env-type keys ("tools_only",
        "with_user_and_policy", "with_user"), or "*" (applies to every flow).
        ``examples_path`` may point to a single file (JSON/JSONL) or a directory
        of per-category files (filename stem = category name). Examples stay raw
        (str or dict) until formatted.
        """
        raw = None
        path = fs_cfg.get("examples_path")
        if path:
            raw = self._read_examples_file(path)
        if raw is None:
            raw = fs_cfg.get("examples")
        return self._normalize_examples(raw)

    @staticmethod
    def _read_examples_file(path: str):
        """Load few-shot examples from a file OR a directory.

        - File: whole-file JSON (object/array) or JSONL (one example per line).
        - Directory: each *.json / *.jsonl file becomes a key = filename stem
          (intended to be a category_name, e.g. ``ticket_api.json``).
          A file containing a dict merges its keys directly.
        """
        if not path or not os.path.exists(path):
            if path:
                print(f"[generate_queries] few_shot examples_path not found: {path}")
            return None
        if os.path.isdir(path):
            out: dict = {}
            for fn in sorted(os.listdir(path)):
                fp = os.path.join(path, fn)
                if not os.path.isfile(fp):
                    continue
                data = GenerateQueriesStep._read_single_file(fp)
                if not data:
                    continue
                stem = os.path.splitext(fn)[0]
                if isinstance(data, dict):
                    for k, v in data.items():
                        out.setdefault(str(k), []).extend(
                            v if isinstance(v, list) else [v]
                        )
                else:
                    out.setdefault(stem, []).extend(
                        data if isinstance(data, list) else [data]
                    )
            return out or None
        return GenerateQueriesStep._read_single_file(path)

    @staticmethod
    def _read_single_file(path: str):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        try:
            return json.loads(text)  # whole-file JSON (object or array)
        except json.JSONDecodeError:
            pass
        items = []  # fall back to JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append(line)
        return items

    @staticmethod
    def _normalize_examples(raw) -> dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            out = {}
            for k, v in raw.items():
                if v is None:
                    continue
                out[str(k)] = v if isinstance(v, list) else [v]
            return out
        if isinstance(raw, list):
            out: dict = {}
            for item in raw:
                key = "*"
                if isinstance(item, dict):
                    key = str(
                        item.get("type") or item.get("env_type")
                        or item.get("environment_type") or item.get("query_type") or "*"
                    )
                out.setdefault(key, []).append(item)
            return out
        return {"*": [raw]}

    def _collect_examples(self, inp: dict) -> list:
        """Build the few-shot pool for one record, then randomly sample it.

        Examples are gathered from the most general to the most specific key:
        ``"*"`` (all) -> env-type -> exact flow -> ``category_name``. After
        de-duplication, if ``num_examples`` (> 0) is smaller than the pool, a
        random subset of that size is drawn. Sampling is seeded by a stable
        per-record id so it is reproducible across runs but varies per record.
        """
        if not self.few_shot_examples:
            return []
        flow_key = inp.get("query_type", "")
        env_key = flow_key.split("_", 1)[1] if "_" in flow_key else flow_key
        cat = inp.get("category_name", "")

        pool: list = []
        for key in ("*", env_key, flow_key, cat):
            if key:
                pool.extend(self.few_shot_examples.get(key, []))

        # De-duplicate while preserving order.
        seen: set = set()
        uniq: list = []
        for ex in pool:
            h = ex if isinstance(ex, str) else json.dumps(ex, sort_keys=True, default=str)
            if h not in seen:
                seen.add(h)
                uniq.append(ex)
        pool = uniq

        if self.few_shot_num > 0 and len(pool) > self.few_shot_num:
            seed_src = inp.get("_item_id") or f"{inp.get('id', '')}|{cat}|{flow_key}"
            rng = random.Random(str(seed_src))
            pool = rng.sample(pool, self.few_shot_num)
        return pool

    @staticmethod
    def _format_examples(examples: list) -> str:
        blocks = []
        for i, ex in enumerate(examples, 1):
            if isinstance(ex, str):
                body = ex.strip()
            elif isinstance(ex, dict):
                core = ex.get("scenario") or ex.get("query")
                if isinstance(core, str) and set(ex.keys()) <= {"scenario", "query"}:
                    body = core.strip()
                else:
                    body = json.dumps(ex, indent=2, ensure_ascii=False)
            else:
                body = str(ex)
            blocks.append(f"Example {i}:\n{body}")
        return "\n\n".join(blocks)

    def _select_system(self, zero_system: str, fewshot_system: str, inp: dict) -> str:
        """Few-shot system prompt (examples injected) when enabled and examples
        exist for this flow; otherwise the zero-shot system prompt."""
        if not self.few_shot_enabled:
            return zero_system
        examples = self._collect_examples(inp)
        if not examples:
            return zero_system
        if "{FEW_SHOT_EXAMPLES}" not in fewshot_system:
            raise ValueError(
                "Few-shot system prompt is missing the {FEW_SHOT_EXAMPLES} marker."
            )
        return fewshot_system.replace(
            "{FEW_SHOT_EXAMPLES}", self._format_examples(examples)
        )

    # ── Parsing helper ────────────────────────────────────────────────────────

    def _parse_query(self, res: dict):
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            try:
                return json.loads(parsed[0])
            except (json.JSONDecodeError, TypeError):
                return parsed[0] if parsed[0] else "Error"
        raw = TagParser.extract_first("query", res["content"])
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return "Error"
