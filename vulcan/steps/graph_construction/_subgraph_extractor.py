"""Fast subgraph extraction using a grow-from-roots approach.

Candidates are grown by edge-guided BFS expansion, so every subgraph is
connected by construction — no connectivity check needed.

Complexity: O(N * d^max_size) where d = average degree.

Each candidate is scored (popularity, declarative_suitability) and reduced
to an acyclic single-leaf subgraph before it is emitted.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ── Edge helpers ──────────────────────────────────────────────────────────────

def _parse_edges(
    graph_data: list,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[dict]]:
    """Parse graph data into explicit/implicit edge pairs and full edge dicts."""
    explicit: List[Tuple[str, str]] = []
    implicit: List[Tuple[str, str]] = []
    all_edge_dicts: List[dict] = []

    for entry in graph_data:
        source = entry.get("source")
        for edge_info in entry.get("all_edges", []):
            target = edge_info.get("target")
            edges = edge_info.get("edges")
            if not edges or not isinstance(edges, dict):
                continue
            etype = edges.get("type")
            if etype not in ("explicit", "implicit"):
                continue

            pair = (source, target)
            edge_dict = {
                "source": source,
                "target": target,
                "type": etype,
                "arguments": edges.get("arguments", {}),
            }
            all_edge_dicts.append(edge_dict)
            if etype == "explicit":
                explicit.append(pair)
            else:
                implicit.append(pair)

    return explicit, implicit, all_edge_dicts


def _build_adjacency(
    explicit: list,
    implicit: list,
) -> Dict[str, Set[str]]:
    """Build directed adjacency: node → set of reachable neighbors."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in explicit + implicit:
        adj[a].add(b)
    return adj


def _build_undirected_adjacency(
    explicit: list,
    implicit: list,
) -> Dict[str, Set[str]]:
    """Build undirected adjacency for BFS expansion."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in explicit + implicit:
        adj[a].add(b)
        adj[b].add(a)
    return adj


# ── Subgraph growing ──────────────────────────────────────────────────────────

def _grow_subgraphs_from_root(
    root: str,
    undirected_adj: Dict[str, Set[str]],
    min_size: int,
    max_size: int,
) -> List[frozenset]:
    """Grow connected subgraphs from a root node using BFS expansion.

    At each step, expand the frontier by one neighbor.
    Yields all connected subsets of size [min_size, max_size] reachable from root.
    """
    results: Set[frozenset] = set()

    def _expand(current_nodes: frozenset, frontier: frozenset) -> None:
        size = len(current_nodes)

        if size >= min_size:
            results.add(current_nodes)

        if size >= max_size:
            return

        for node in sorted(frontier):
            new_nodes = current_nodes | frozenset([node])
            if new_nodes in results and size + 1 >= min_size:
                continue
            new_frontier = (frontier | undirected_adj.get(node, set())) - new_nodes
            _expand(new_nodes, new_frontier)

    initial_frontier = frozenset(undirected_adj.get(root, set()))
    _expand(frozenset([root]), initial_frontier)

    return [r for r in results if len(r) >= min_size]


def extract_good_subgraphs(
    graph_data: list,
    catalog: list,
    *,
    min_size: int = 2,
    max_size: int = 5,
    implicit_edge_weight: float = 0.01,
    lambda_I: float = 2.0,
    mu_P: float = 1.0,
) -> List[dict]:
    """Extract good subgraphs using grow-from-roots approach.

    Args:
        graph_data: list of {source, all_edges: [{target, edges}]} from build_graph
        catalog: API catalog for the category
        min_size: minimum subgraph size
        max_size: maximum subgraph size
        implicit_edge_weight: weight for implicit edges in popularity scoring
        lambda_I: implicit edge penalty in declarative suitability
        mu_P: parameter penalty in declarative suitability

    Returns:
        list of good subgraph dicts with nodes, edges, and scoring metrics
    """
    explicit, implicit, all_edge_dicts = _parse_edges(graph_data)
    undirected_adj = _build_undirected_adjacency(explicit, implicit)
    expl_set = set(explicit)
    impl_set = set(implicit)

    all_nodes: Set[str] = set()
    for a, b in explicit + implicit:
        all_nodes.add(a)
        all_nodes.add(b)

    print(f"    Nodes: {len(all_nodes)}, Explicit edges: {len(explicit)}, Implicit: {len(implicit)}")

    # ── Phase 1: Grow subgraphs from each root ──────────────────────────────
    all_subsets: Set[frozenset] = set()
    for root in sorted(all_nodes):
        grown = _grow_subgraphs_from_root(root, undirected_adj, min_size, max_size)
        all_subsets.update(grown)

    print(f"    Candidates (grow-from-roots): {len(all_subsets)}")

    # ── Phase 2: Score and filter ────────────────────────────────────────────
    good: List[dict] = []

    for node_set in all_subsets:
        nodes = list(node_set)
        N = node_set

        subgraph_edges = [
            e for e in all_edge_dicts
            if e["source"] in N and e["target"] in N
        ]

        if not subgraph_edges:
            continue

        # Check single leaf (0 out-degree within subgraph)
        out_deg: Dict[str, int] = defaultdict(int)
        for a, b in expl_set | impl_set:
            if a in N and b in N:
                out_deg[a] += 1
        leaves = [n for n in N if out_deg[n] == 0]

        if len(leaves) != 1:
            continue

        E = sum(1 for a, b in expl_set if a in N and b in N)
        I = sum(1 for a, b in impl_set if a in N and b in N)

        popularity = E + I * implicit_edge_weight

        alpha = 1 + E
        beta = 1 + lambda_I * I
        declarative_prob = alpha / (alpha + beta)

        dag = _make_acyclic_single_leaf(nodes, subgraph_edges)
        if dag is None:
            continue

        good.append({
            "nodes": dag["nodes"],
            "edges": dag["edges"],
            "popularity": popularity,
            "declarative_suitability": declarative_prob,
            "explicit_len": E,
            "implicit_len": I,
            "evaluation_metrics": {
                "popularity": popularity,
                "declarative_suitability": declarative_prob,
                "explicit_len": E,
                "implicit_len": I,
            },
        })

    print(f"    Good subgraphs (acyclic, connected, single-leaf): {len(good)}")
    return good


def _make_acyclic_single_leaf(
    nodes: list,
    edges: list,
) -> Optional[dict]:
    """Remove cycle-forming edges to make a DAG with a single leaf.

    Uses NetworkX for cycle detection. Preserves edge attributes.
    Returns None if conversion fails.
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)

    edge_lookup: Dict[tuple, dict] = {}
    for e in edges:
        u, v = e["source"], e["target"]
        g.add_edge(u, v, **e)
        edge_lookup[(u, v)] = deepcopy(e)

    leaves = [n for n in g if g.out_degree(n) == 0]
    if not leaves:
        return None
    main_leaf = leaves[0]

    max_iter = len(edges) * 2
    for _ in range(max_iter):
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break

        removed = False
        for u, v, _ in cycle:
            if g.out_degree(u) > 1 and u != main_leaf:
                g.remove_edge(u, v)
                removed = True
                break
        if not removed:
            u, v, _ = cycle[0]
            g.remove_edge(u, v)

    final_leaves = [n for n in g if g.out_degree(n) == 0]
    if len(final_leaves) != 1:
        return None

    if not nx.is_weakly_connected(g):
        return None

    dag_edges = [edge_lookup[(u, v)] for u, v in g.edges() if (u, v) in edge_lookup]
    return {"nodes": list(g.nodes), "edges": dag_edges}


# ── Uniform popularity sampling ───────────────────────────────────────────────

def sample_by_popularity(
    subgraphs: list,
    target_count: int,
    implicit_edge_weight: float = 0.01,
) -> list:
    """Sample subgraphs uniformly across popularity bins.

    Guarantees at least 1 sample per unique popularity value (if budget allows).
    """
    if not subgraphs or target_count <= 0:
        return []

    if len(subgraphs) <= target_count:
        return subgraphs

    bins: Dict[int, list] = defaultdict(list)
    for sg in subgraphs:
        pop = sg.get("popularity", 0)
        pop_key = round(pop)
        bins[pop_key].append(sg)

    selected = []
    for key in sorted(bins.keys()):
        if bins[key]:
            selected.append(bins[key][0])

    if len(selected) >= target_count:
        return selected[:target_count]

    remaining = target_count - len(selected)
    selected_set = {id(s) for s in selected}
    round_idx = 1
    while remaining > 0:
        added = False
        for key in sorted(bins.keys()):
            if round_idx < len(bins[key]):
                candidate = bins[key][round_idx]
                if id(candidate) not in selected_set:
                    selected.append(candidate)
                    selected_set.add(id(candidate))
                    remaining -= 1
                    added = True
                    if remaining <= 0:
                        break
        if not added:
            break
        round_idx += 1

    return selected
