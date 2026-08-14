"""Multipath subgraph extraction from the full dependency graph.

Independent from the good-subgraph/sequence pipeline.

Algorithm:
1. Build directed graph from ALL dependency edges
2. For every ordered pair (u, v), count distinct directed paths
3. If paths(u, v) > 1 → (u, v) is a multipath seed pair
4. For each seed pair, collect ALL nodes on ALL paths from u to v
   → this forms the minimal multipath subgraph
5. Deduplicate subgraphs (same node set = same subgraph)
6. Score by multipath_strength (max path count) and size

A multipath subgraph captures a "diamond" or "fork-merge" pattern where
the dependency graph offers multiple tool-call routes between two points.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple


def _build_directed_adj(
    graph_data: list,
) -> Tuple[Dict[str, Set[str]], List[dict], Set[str]]:
    """Build directed adjacency from build_graph output.

    Returns: (adj, all_edge_dicts, all_nodes)
    """
    adj: Dict[str, Set[str]] = defaultdict(set)
    all_edge_dicts: List[dict] = []
    all_nodes: Set[str] = set()

    for entry in graph_data:
        source = entry.get("source")
        if source:
            all_nodes.add(source)
        for edge_info in entry.get("all_edges", []):
            target = edge_info.get("target")
            edges = edge_info.get("edges")
            if not edges or not isinstance(edges, dict):
                continue
            etype = edges.get("type")
            if etype not in ("explicit", "implicit"):
                continue
            if target:
                all_nodes.add(target)
            adj[source].add(target)
            all_edge_dicts.append({
                "source": source,
                "target": target,
                "type": etype,
                "arguments": edges.get("arguments", {}),
            })

    return adj, all_edge_dicts, all_nodes


def _find_all_paths(
    source: str,
    target: str,
    adj: Dict[str, Set[str]],
    max_paths: int = 20,
    max_depth: int = 8,
) -> List[List[str]]:
    """Find all distinct directed paths from source to target.

    Uses DFS with backtracking. Bounded by max_paths and max_depth.
    Every path found has at least one intermediate node (pairs with a
    direct edge are pre-filtered by the caller).
    """
    if source == target:
        return []

    paths: List[List[str]] = []

    def dfs(current: str, path: List[str], depth: int) -> None:
        if len(paths) >= max_paths:
            return
        if depth > max_depth:
            return
        for neighbor in adj.get(current, set()):
            if neighbor in path:
                continue
            if neighbor == target:
                paths.append(path + [neighbor])
                if len(paths) >= max_paths:
                    return
            else:
                dfs(neighbor, path + [neighbor], depth + 1)

    dfs(source, [source], 0)
    return paths


def _collect_path_nodes(paths: List[List[str]]) -> Set[str]:
    """Collect all unique nodes across all paths."""
    nodes: Set[str] = set()
    for path in paths:
        nodes.update(path)
    return nodes


def extract_multipath_subgraphs(
    graph_data: list,
    *,
    min_paths: int = 2,
    max_tools_per_sequence: int = 8,
    max_paths_per_pair: int = 20,
    max_path_depth: int = 8,
) -> List[dict]:
    """Extract all multipath subgraphs from the full dependency graph.

    Scans every ordered pair (u, v) for multiple paths. When found,
    collects all nodes on all paths to form the multipath subgraph.

    Args:
        graph_data: full dependency graph from build_graph step
        min_paths: minimum paths between a pair to qualify as multipath
        max_tools_per_sequence: skip subgraphs larger than this
        max_paths_per_pair: bound on path enumeration per pair
        max_path_depth: max path length to search

    Returns:
        list of multipath subgraph dicts:
        {
            "nodes": [...],
            "edges": [...],
            "multipath_pairs": [{"source": u, "target": v, "num_paths": N}],
            "multipath_strength": max path count across all pairs,
            "subgraph_type": "multipath"
        }
    """
    adj, all_edge_dicts, all_nodes = _build_directed_adj(graph_data)

    edge_set = {(e["source"], e["target"]): e for e in all_edge_dicts}
    direct_edges = {(e["source"], e["target"]) for e in all_edge_dicts}

    print(f"    Scanning {len(all_nodes)} nodes, {len(all_edge_dicts)} edges for multipath pairs...")

    multipath_pairs = []
    for u in sorted(all_nodes):
        for v in sorted(all_nodes):
            if u == v:
                continue
            if (u, v) in direct_edges:
                continue  # skip pairs with direct edge
            paths = _find_all_paths(
                u, v, adj,
                max_paths=max_paths_per_pair,
                max_depth=max_path_depth,
            )
            if len(paths) >= min_paths:
                multipath_pairs.append({
                    "source": u,
                    "target": v,
                    "num_paths": len(paths),
                    "paths": paths,
                })

    print(f"    Found {len(multipath_pairs)} multipath pairs")

    if not multipath_pairs:
        return []

    # Build subgraphs from path nodes, deduplicate by node set
    seen_node_sets: Dict[FrozenSet[str], dict] = {}

    for pair in multipath_pairs:
        path_nodes = _collect_path_nodes(pair["paths"])

        if len(path_nodes) > max_tools_per_sequence:
            continue

        node_key = frozenset(path_nodes)

        if node_key in seen_node_sets:
            existing = seen_node_sets[node_key]
            existing["multipath_pairs"].append({
                "source": pair["source"],
                "target": pair["target"],
                "num_paths": pair["num_paths"],
            })
            existing["multipath_strength"] = max(
                existing["multipath_strength"], pair["num_paths"]
            )
        else:
            nodes = sorted(path_nodes)
            subgraph_edges = [
                e for e in all_edge_dicts
                if e["source"] in path_nodes and e["target"] in path_nodes
            ]
            seen_node_sets[node_key] = {
                "nodes": nodes,
                "edges": subgraph_edges,
                "multipath_pairs": [{
                    "source": pair["source"],
                    "target": pair["target"],
                    "num_paths": pair["num_paths"],
                }],
                "multipath_strength": pair["num_paths"],
                "subgraph_type": "multipath",
                "size": len(nodes),
                "num_edges": len(subgraph_edges),
            }

    subgraphs = list(seen_node_sets.values())
    subgraphs.sort(key=lambda s: (-s["multipath_strength"], s["size"]))

    print(f"    Multipath subgraphs: {len(subgraphs)} (deduplicated by node set)")
    for sg in subgraphs[:5]:
        pairs_str = ", ".join(
            f"{p['source']}→{p['target']}({p['num_paths']} paths)"
            for p in sg["multipath_pairs"]
        )
        print(f"      {sg['nodes']} strength={sg['multipath_strength']} [{pairs_str}]")

    return subgraphs
