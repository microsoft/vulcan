"""execute_sequences step: Combine graphs + env data with environment execution.

For each (graph, initial_state) pair:
1. Load environment class, initialize with initial_state
2. Topological sort the graph nodes
3. Execute tools in order:
   - Explicit edge dependencies: use upstream tool's actual output
   - Independent params: pick from function sample bank
4. Store real executed input/output chains

Handles both graph types:
- sequence:  per-turn execution, bridge edges thread between turns
- multipath: execute ALL paths, store each path's chain
"""

from __future__ import annotations

import json
import os
import random
import traceback
from collections import defaultdict
from copy import deepcopy

from .. import register_step
from ..base import BaseStep
from ...core.io import load_jsonl, save_jsonl, ensure_dir
from ...core.code_exec import execute_class_code, safe_execute_tool
from ...core.exceptions import StepError


def _topological_sort(nodes: list, edges: list) -> list:
    """Kahn's algorithm for topological sort within a subgraph."""
    node_set = set(nodes)
    adj: dict = defaultdict(list)
    in_degree: dict = defaultdict(int)
    for n in nodes:
        in_degree[n] = 0

    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src in node_set and tgt in node_set:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    queue = [n for n in nodes if in_degree[n] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes sorted (cycle), append remaining
    remaining = [n for n in nodes if n not in result]
    result.extend(remaining)
    return result


def _resolve_inputs(
    tool_name: str,
    edges: list,
    executed_outputs: dict,
    sample_bank: dict,
    api_spec: dict,
) -> dict:
    """Build input arguments for a tool.

    Priority:
    1. Explicit edge: upstream tool's output field → this tool's input param
    2. Sample bank: random sample from function sample pool
    3. Default: leave empty (tool may have defaults)
    """
    args = {}

    for edge in edges:
        if edge.get("target") != tool_name or edge.get("type") != "explicit":
            continue
        source = edge.get("source")
        source_output = executed_outputs.get(source, {})

        arguments = edge.get("arguments", {})
        source_args = edge.get("source_arguments", [])
        target_args = edge.get("target_arguments", [])

        if arguments and isinstance(arguments, dict):
            for target_param, source_field in arguments.items():
                if isinstance(source_output, dict) and source_field in source_output:
                    args[target_param] = source_output[source_field]
        elif source_args and target_args:
            for src_field, tgt_param in zip(source_args, target_args):
                if isinstance(source_output, dict) and src_field in source_output:
                    args[tgt_param] = source_output[src_field]

    # Fill remaining from the sample bank, accepting only keys this tool
    # actually declares. The tool's own schema is the authority here — a
    # name-based blacklist would silently drop a real parameter from any
    # catalog that happened to use the blacklisted name, and passing an
    # undeclared key through would make invoke() raise TypeError.
    declared = set((api_spec.get("parameters") or {}).get("properties") or {})
    if tool_name in sample_bank:
        samples = sample_bank[tool_name]
        if samples:
            sample = random.choice(samples) if isinstance(samples, list) else samples
            if isinstance(sample, dict):
                for param, value in sample.items():
                    if param in args:
                        continue
                    # No declared properties (spec absent or parameterless):
                    # fall back to accepting whatever the bank supplies.
                    if declared and param not in declared:
                        continue
                    args[param] = value

    return args


@register_step("execute_sequences")
class ExecuteSequencesStep(BaseStep):
    """Combine graphs + env data with actual environment execution.

    Produces pre_query_all_modes.jsonl: one item per (graph, initial_state) pair
    with real executed tool inputs/outputs.
    """

    requires_llm = False

    async def run(self, inp: dict, progress=None) -> dict:
        """No-op: this step uses run_batch() instead."""
        return inp

    def run_batch(self, graph_path: str, env_data_path: str, output_dir: str) -> str:
        """Main entry point.

        Args:
            graph_path: path to plan_sequences's all_sequences.jsonl
            env_data_path: path to sample_arguments success.jsonl
            output_dir: directory to write outputs

        Returns:
            path to pre_query_all_modes.jsonl
        """
        ensure_dir(output_dir)

        graphs = load_jsonl(graph_path) if os.path.exists(graph_path) else []
        env_data = load_jsonl(env_data_path) if os.path.exists(env_data_path) else []
        max_per_cat = self.step_config.get("max_samples_per_category", 50000)
        states_per_sequence = self.step_config.get("states_per_sequence", 2)
        shuffle_seed = self.step_config.get("shuffle_seed", 42)

        # Index env data by category — support both new and legacy formats
        env_by_cat: dict = {}
        states_by_cat: dict = defaultdict(list)
        for item in env_data:
            cat = item.get("category_name")
            if not cat:
                continue
            if "initial_state" in item and "mock_data" not in item:
                # New format: individual state items
                states_by_cat[cat].append(item)
                if cat not in env_by_cat:
                    env_by_cat[cat] = item
            else:
                env_by_cat[cat] = item

        # Group graphs by category
        graphs_by_cat: dict = defaultdict(list)
        for g in graphs:
            cat = g.get("category_name")
            if cat:
                graphs_by_cat[cat].append(g)

        all_samples = []
        failed_samples = []

        for cat, cat_graphs in graphs_by_cat.items():
            env = env_by_cat.get(cat)
            if not env:
                print(f"[execute_sequences] Skipping {cat}: no env data")
                continue

            py_code = env.get("tool_code", "")
            mock_data = env.get("mock_data", {})
            api_list = env.get("api_list", [])

            if not py_code:
                continue

            try:
                cls = execute_class_code(py_code)
                test_obj = cls()
                look_up = test_obj.function_name_mapping()
                inverted_look_up = {
                    (v.__name__ if callable(v) else str(v)): k
                    for k, v in look_up.items()
                }
            except Exception as e:
                print(f"[execute_sequences] Skipping {cat}: {e}")
                continue

            # Build API spec lookup
            api_by_name: dict = {}
            for api in api_list:
                func = (
                    api.get("normalized_schema")
                    or api.get("normalized_function")
                    or api.get("function", {})
                )
                name = func.get("name", "")
                if name:
                    api_by_name[name] = func

            # Collect initial states
            initial_data_list = mock_data.get("initial_data", [])
            if cat in states_by_cat:
                initial_states = [
                    {"id": i, "initial_state": s.get("initial_state")}
                    for i, s in enumerate(states_by_cat[cat])
                    if s.get("initial_state") is not None
                ]
            else:
                initial_states = [
                    d for d in initial_data_list
                    if isinstance(d, dict) and d.get("initial_state") is not None
                ]

            if not initial_states:
                print(f"[execute_sequences] Skipping {cat}: no initial states")
                continue

            # Build sample bank
            if cat in states_by_cat:
                sample_bank = self._build_sample_bank(states_by_cat[cat])
            else:
                sample_bank = self._build_sample_bank(initial_data_list)

            cat_count = 0
            state_cursor = 0

            # Shuffle before the per-category cap so samples are picked uniformly
            # across turn-counts. The input is ordered in turn blocks (all 1-turn,
            # then 2-turn, then 3-turn); without shuffling, max_samples_per_category
            # is exhausted on the early turn blocks and higher-turn samples (e.g.
            # 3-turn) get starved. Seeded for reproducibility.
            cat_graphs = list(cat_graphs)
            random.Random(shuffle_seed).shuffle(cat_graphs)

            for graph_item in cat_graphs:
                if cat_count >= max_per_cat:
                    break

                graph_type = graph_item.get("type", "sequence")

                n = min(states_per_sequence, len(initial_states))
                selected_states = []
                for _ in range(n):
                    selected_states.append(initial_states[state_cursor % len(initial_states)])
                    state_cursor += 1

                for state_item in selected_states:
                    if cat_count >= max_per_cat:
                        break

                    init_state = deepcopy(state_item.get("initial_state"))
                    state_id = state_item.get("id", 0)

                    try:
                        if graph_type == "multipath":
                            sample = self._process_multipath(
                                graph_item, init_state, state_id, cat, py_code, cls,
                                look_up, inverted_look_up, api_by_name, sample_bank, api_list,
                            )
                        else:
                            sample = self._process_sequence(
                                graph_item, init_state, state_id, cat, py_code, cls,
                                look_up, inverted_look_up, api_by_name, sample_bank, api_list,
                            )
                    except Exception:
                        traceback.print_exc()
                        sample = None

                    if sample:
                        all_samples.append(sample)
                        cat_count += 1
                    else:
                        failed_samples.append({
                            "category_name": cat,
                            "graph_id": graph_item.get("id"),
                            "graph_type": graph_type,
                            "database_index": state_id,
                            "reason": (
                                "empty executed_paths" if graph_type == "multipath"
                                else "empty data_list"
                            ),
                        })

            print(f"[execute_sequences] {cat}: {cat_count} samples")

        out_path = os.path.join(output_dir, "pre_query_all_modes.jsonl")
        save_jsonl(all_samples, out_path)
        print(f"[execute_sequences] Total: {len(all_samples)} samples -> {out_path}")

        if failed_samples:
            failed_path = os.path.join(output_dir, "failed.jsonl")
            save_jsonl(failed_samples, failed_path)
            print(f"[execute_sequences] Failed: {len(failed_samples)} -> {failed_path}")

        return out_path

    # ── Sequence processing ──────────────────────────────────────────────────

    def _process_sequence(
        self,
        graph_item: dict,
        init_state,
        state_id: int,
        cat: str,
        py_code: str,
        cls,
        look_up: dict,
        inverted_look_up: dict,
        api_by_name: dict,
        sample_bank: dict,
        api_list: list,
    ) -> dict | None:
        """Process a sequence-type graph: execute per-turn, thread bridge edges."""
        sequence = graph_item.get("sequence", [])
        if not sequence:
            return None

        try:
            env = cls(initial_state=deepcopy(init_state))
        except TypeError:
            env = cls()

        all_turn_data = []
        all_executed_outputs: dict = {}

        for turn_id, turn_info in enumerate(sequence):
            gi = turn_info.get("graph_info", turn_info)
            nodes = gi.get("nodes", [])
            edges = gi.get("edges", [])

            sorted_nodes = _topological_sort(nodes, edges)

            all_edges = list(edges) + list(self._get_bridge_edges(graph_item, turn_id))

            turn_mock: dict = {}
            for tool_name in sorted_nodes:
                args = _resolve_inputs(
                    tool_name, all_edges, all_executed_outputs,
                    sample_bank, api_by_name.get(tool_name, {}),
                )
                result = safe_execute_tool(env, tool_name, args)

                turn_mock[tool_name] = {
                    "input": args,
                    "output": result.get("output") if result["success"] else None,
                    "success": result["success"],
                    "error": result.get("error"),
                }

                if result["success"]:
                    all_executed_outputs[tool_name] = (
                        result["output"] if isinstance(result["output"], dict) else {}
                    )

            nodes_api_info = [api_by_name[n] for n in nodes if n in api_by_name]

            all_turn_data.append({
                "turn": turn_id,
                "graph": {"nodes": nodes, "edges": edges},
                "api_info": nodes_api_info,
                "mock_data_apis": turn_mock,
                "executed": True,
            })

        if not all_turn_data:
            return None

        return {
            "mode": "class_multi_turn",
            "type": "sequence",
            "id": graph_item.get("id"),
            "initial_state": init_state,
            "database_index": state_id,
            "category_name": cat,
            "number_turns": graph_item.get("number_turns", len(sequence)),
            "data_list": all_turn_data,
            "tool_code": py_code,
            "api_list": api_list,
        }

    # ── Multipath processing ─────────────────────────────────────────────────

    def _process_multipath(
        self,
        graph_item: dict,
        init_state,
        state_id: int,
        cat: str,
        py_code: str,
        cls,
        look_up: dict,
        inverted_look_up: dict,
        api_by_name: dict,
        sample_bank: dict,
        api_list: list,
    ) -> dict | None:
        """Process a multipath-type graph: execute ALL paths."""
        from ..graph_construction._multipath_extractor import _find_all_paths

        nodes = graph_item.get("nodes", [])
        edges = graph_item.get("edges", [])
        multipath_pairs = graph_item.get("multipath_pairs", [])

        adj: dict = defaultdict(set)
        for e in edges:
            adj[e["source"]].add(e["target"])

        executed_paths = []

        for pair in multipath_pairs:
            source = pair["source"]
            target = pair["target"]

            paths = _find_all_paths(source, target, adj, max_paths=10, max_depth=len(nodes))

            for path in paths:
                try:
                    env = cls(initial_state=deepcopy(init_state))
                except TypeError:
                    env = cls()

                path_steps = []
                path_outputs: dict = {}

                for tool_name in path:
                    args = _resolve_inputs(
                        tool_name, edges, path_outputs,
                        sample_bank, api_by_name.get(tool_name, {}),
                    )
                    result = safe_execute_tool(env, tool_name, args)

                    path_steps.append({
                        "tool": tool_name,
                        "input": args,
                        "output": result.get("output") if result["success"] else None,
                        "success": result["success"],
                        "error": result.get("error"),
                    })

                    if result["success"]:
                        path_outputs[tool_name] = (
                            result["output"] if isinstance(result["output"], dict) else {}
                        )

                executed_paths.append({
                    "path": path,
                    "source": source,
                    "target": target,
                    "steps": path_steps,
                    "all_succeeded": all(s["success"] for s in path_steps),
                })

        nodes_api_info = [api_by_name[n] for n in nodes if n in api_by_name]

        if not executed_paths:
            return None

        return {
            "mode": "class_multi_turn",
            "type": "multipath",
            "id": graph_item.get("id"),
            "initial_state": init_state,
            "database_index": state_id,
            "category_name": cat,
            "number_turns": 1,
            "nodes": nodes,
            "edges": edges,
            "multipath_pairs": multipath_pairs,
            "data_list": [{
                "turn": 0,
                "graph": {"nodes": nodes, "edges": edges},
                "api_info": nodes_api_info,
                "mock_data_apis": {
                    s["tool"]: {"input": s["input"], "output": s["output"]}
                    for p in executed_paths
                    for s in p["steps"]
                },
                "executed": True,
            }],
            "executed_paths": executed_paths,
            "tool_code": py_code,
            "api_list": api_list,
        }

    # ── Bridge edges ──────────────────────────────────────────────────────────

    def _get_bridge_edges(self, graph_item: dict, current_turn_id: int) -> list:
        """Get edges from previous turns that bridge into current turn."""
        bridge_edges = []
        sequence = graph_item.get("sequence", [])
        if current_turn_id == 0:
            return bridge_edges

        gi = sequence[current_turn_id].get("graph_info", sequence[current_turn_id])
        current_nodes = set(gi.get("nodes", []))

        for prev_turn_id in range(current_turn_id):
            prev_gi = sequence[prev_turn_id].get("graph_info", sequence[prev_turn_id])
            prev_nodes = set(prev_gi.get("nodes", []))
            prev_edges = prev_gi.get("edges", [])

            for e in prev_edges:
                if e.get("source") in prev_nodes and e.get("target") in current_nodes:
                    bridge_edges.append(e)

        return bridge_edges

    # ── Sample bank ───────────────────────────────────────────────────────────

    def _build_sample_bank(self, initial_data_list: list) -> dict:
        """Build a sample bank: {tool_name: [sample_inputs]}."""
        bank: dict = defaultdict(list)
        for pool in initial_data_list:
            if not isinstance(pool, dict):
                continue
            functions = pool.get("functions")
            if not functions or not isinstance(functions, list):
                continue
            for func_entry in functions:
                name = func_entry.get("function_name") or func_entry.get("name", "")
                cases = func_entry.get("cases", [])
                for case in cases:
                    if isinstance(case, dict):
                        details = case.get("details") or case.get("input") or case
                        if isinstance(details, dict):
                            bank[name].append(details)
        return dict(bank)
