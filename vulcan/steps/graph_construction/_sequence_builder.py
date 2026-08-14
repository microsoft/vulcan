"""Meta-graph based multi-turn sequence builder.

Sequences are found by deterministic search over a meta-graph of compatible
subgraphs, so every emitted sequence is valid by construction.

Algorithm:
1. Pre-compute compatibility between all good subgraph pairs
   - No node overlap
   - Bridge edges from A's leaf → B's nodes
   - Weight = explicit bridge count + implicit_edge_weight * implicit bridge count
2. Build a directed meta-graph: nodes = subgraphs, edges = compatibility
3. Enumerate all k-length paths in the meta-graph (DFS)
   - Each path = a valid multi-turn sequence (guaranteed valid by construction)
4. Score each path by total bridge weight
5. Select diverse top-N sequences (coverage-aware)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple


def _get_nodes(subgraph: dict) -> Set[str]:
    return set(subgraph.get("nodes", []))


def _get_edges(subgraph: dict) -> List[dict]:
    return subgraph.get("edges", [])


def _get_leaf(subgraph: dict, all_edges: List[Tuple[str, str]]) -> str | None:
    """Find the single leaf node (0 out-degree within this subgraph)."""
    nodes = _get_nodes(subgraph)
    out_deg: Dict[str, int] = defaultdict(int)
    for a, b in all_edges:
        if a in nodes and b in nodes:
            out_deg[a] += 1
    leaves = [n for n in nodes if out_deg[n] == 0]
    return leaves[0] if len(leaves) == 1 else None


def _extract_all_edge_pairs(
    graph_edges: List[dict],
) -> Tuple[List[Tuple], List[Tuple]]:
    """Extract (source, target) pairs separated by type."""
    explicit: List[Tuple] = []
    implicit: List[Tuple] = []
    for e in graph_edges:
        pair = (e.get("source"), e.get("target"))
        if None in pair:
            continue
        if e.get("type") == "explicit":
            explicit.append(pair)
        elif e.get("type") == "implicit":
            implicit.append(pair)
    return explicit, implicit


def build_meta_graph(
    good_subgraphs: List[dict],
    all_graph_edges: List[dict],
    implicit_edge_weight: float = 0.5,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build a directed meta-graph where:
    - Node i = good_subgraphs[i]
    - Edge i → j exists if subgraphs i and j are compatible
    - Weight = explicit_bridges + implicit_edge_weight * implicit_bridges

    Compatibility = no node overlap + at least 1 bridge edge from i's leaf to j's nodes.
    """
    explicit_pairs, implicit_pairs = _extract_all_edge_pairs(all_graph_edges)
    expl_set = set(explicit_pairs)
    impl_set = set(implicit_pairs)
    all_pairs = explicit_pairs + implicit_pairs

    n = len(good_subgraphs)
    leaves = [_get_leaf(sg, all_pairs) for sg in good_subgraphs]

    meta_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for i in range(n):
        nodes_i = _get_nodes(good_subgraphs[i])
        leaf_i = leaves[i]
        if leaf_i is None:
            continue

        for j in range(n):
            if i == j:
                continue
            nodes_j = _get_nodes(good_subgraphs[j])

            if nodes_i & nodes_j:
                continue  # node overlap — incompatible

            bridge_exp = sum(1 for b in nodes_j if (leaf_i, b) in expl_set)
            bridge_imp = sum(1 for b in nodes_j if (leaf_i, b) in impl_set)

            if bridge_exp + bridge_imp == 0:
                continue  # no bridge edge

            weight = bridge_exp + implicit_edge_weight * bridge_imp
            meta_adj[i].append((j, weight))

    return meta_adj


def enumerate_paths(
    meta_adj: Dict[int, List[Tuple[int, float]]],
    n_subgraphs: int,
    k: int,
    max_paths: int = 50000,
) -> List[Tuple[List[int], float]]:
    """Enumerate all k-length paths in the meta-graph via DFS.

    Returns list of (path, total_weight) sorted by weight descending.
    Each path is guaranteed valid: no node overlap, bridge edges exist.
    """
    paths: List[Tuple[List[int], float]] = []

    def dfs(
        current: int,
        path: List[int],
        visited_nodes: Set[str],
        total_weight: float,
    ) -> None:
        if len(path) == k:
            paths.append((list(path), total_weight))
            return
        if len(paths) >= max_paths:
            return
        for neighbor, weight in meta_adj.get(current, []):
            if neighbor in path:
                continue
            if len(paths) >= max_paths:
                return
            path.append(neighbor)
            dfs(neighbor, path, visited_nodes, total_weight + weight)
            path.pop()

    for start in range(n_subgraphs):
        if len(paths) >= max_paths:
            break
        dfs(start, [start], set(), 0.0)

    paths.sort(key=lambda x: x[1], reverse=True)
    return paths


def select_diverse(
    paths: List[Tuple[List[int], float]],
    good_subgraphs: List[dict],
    target_count: int,
) -> List[Tuple[List[int], float]]:
    """Select diverse sequences that maximize API coverage.

    Strategy:
    1. Group paths by the set of API nodes they cover
    2. From each group, take the highest-scoring path
    3. If still need more, take second-highest from each group, etc.
    """
    if len(paths) <= target_count:
        return paths

    groups: Dict[frozenset, List[Tuple[List[int], float]]] = defaultdict(list)
    for path, weight in paths:
        all_nodes: frozenset = frozenset()
        for idx in path:
            all_nodes = all_nodes | frozenset(good_subgraphs[idx].get("nodes", []))
        groups[all_nodes].append((path, weight))

    selected: List[Tuple[List[int], float]] = []
    round_idx = 0
    while len(selected) < target_count:
        added_this_round = False
        for node_set in sorted(groups.keys(), key=lambda s: len(s), reverse=True):
            group = groups[node_set]
            if round_idx < len(group):
                selected.append(group[round_idx])
                added_this_round = True
                if len(selected) >= target_count:
                    break
        if not added_this_round:
            break
        round_idx += 1

    return selected


def build_sequences(
    good_subgraphs: List[dict],
    all_graph_edges: List[dict],
    catalog: List[dict],
    turns_per_sequence: List[int] = None,
    target_per_k: int = 5000,
    implicit_edge_weight: float = 0.5,
) -> List[dict]:
    """Main entry point: build multi-turn sequences from good subgraphs.

    Args:
        good_subgraphs: list of subgraph dicts with 'nodes', 'edges'
        all_graph_edges: list of edge dicts from the full dependency graph
        catalog: API catalog for the category
        turns_per_sequence: number of turns to generate sequences for (default: [2, 3, 4])
        target_per_k: how many sequences to select per k value
        implicit_edge_weight: weight of implicit vs explicit bridge edges

    Returns:
        list of sequence dicts ready for downstream steps
    """
    if turns_per_sequence is None:
        turns_per_sequence = [2, 3, 4]

    if not good_subgraphs:
        return []

    meta_adj = build_meta_graph(good_subgraphs, all_graph_edges, implicit_edge_weight=implicit_edge_weight)
    n = len(good_subgraphs)

    print(
        f"  Meta-graph: {n} subgraphs, "
        f"{sum(len(v) for v in meta_adj.values())} compatibility edges"
    )

    all_sequences: List[dict] = []

    for k in turns_per_sequence:
        if k > n:
            print(f"  k={k}: skip (only {n} subgraphs available)")
            continue

        max_paths = target_per_k * 5
        paths = enumerate_paths(meta_adj, n, k, max_paths=max_paths)
        print(f"  k={k}: {len(paths)} valid paths found")

        if not paths:
            continue

        selected = select_diverse(paths, good_subgraphs, target_per_k)
        print(f"  k={k}: {len(selected)} selected (target={target_per_k})")

        for seq_idx, (path, weight) in enumerate(selected):
            sequence = []
            for turn_idx, sg_idx in enumerate(path):
                sg = good_subgraphs[sg_idx]
                sequence.append({
                    "graph_info": {
                        "nodes": sg.get("nodes", []),
                        "edges": sg.get("edges", []),
                        "popularity": sg.get("popularity", 0),
                    }
                })

            all_sequences.append({
                "number_turns": k,
                "sequence": sequence,
                "bridge_weight": weight,
                "index": len(all_sequences),
            })

    return all_sequences


def validate_sequence(
    sequence: List[dict],
    all_graph_edges: List[dict],
) -> dict:
    """Validate a multi-turn sequence.

    Returns dict with: sequence_ok, bridge_exp, bridge_imp, failed_checks.
    """
    explicit_pairs, implicit_pairs = _extract_all_edge_pairs(all_graph_edges)
    expl_set = set(explicit_pairs)
    impl_set = set(implicit_pairs)
    all_pairs = explicit_pairs + implicit_pairs

    turns = [s.get("graph_info", s) for s in sequence]
    failed: List[str] = []

    union_nodes: Set[str] = set()
    for t in turns:
        union_nodes |= set(t.get("nodes", []))

    if union_nodes:
        undirected: Dict[str, Set[str]] = defaultdict(set)
        for a, b in all_pairs:
            if a in union_nodes and b in union_nodes:
                undirected[a].add(b)
                undirected[b].add(a)
        stack, seen = [next(iter(union_nodes))], set()
        while stack:
            v = stack.pop()
            seen.add(v)
            stack.extend(undirected[v] - seen)
        if seen != union_nodes:
            failed.append("not_connected_union")

    for i in range(len(turns)):
        for j in range(i + 1, len(turns)):
            if set(turns[i].get("nodes", [])) & set(turns[j].get("nodes", [])):
                failed.append(f"node_overlap_turn{i+1}_{j+1}")

    bridge_exp = 0
    bridge_imp = 0
    for i in range(len(turns) - 1):
        nodes_i = set(turns[i].get("nodes", []))
        nodes_j = set(turns[i + 1].get("nodes", []))
        be = sum(1 for a, b in expl_set if a in nodes_i and b in nodes_j)
        bi = sum(1 for a, b in impl_set if a in nodes_i and b in nodes_j)
        bridge_exp += be
        bridge_imp += bi
        if be + bi == 0:
            failed.append(f"no_bridge_{i+1}_{i+2}")

    return {
        "sequence_ok": not failed,
        "bridge_exp": bridge_exp,
        "bridge_imp": bridge_imp,
        "failed_checks": failed,
    }
