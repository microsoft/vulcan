#!/usr/bin/env python3
"""Check what `build_graph` and `plan_sequences` actually produced.

Both stages can report success while producing nothing usable, and neither
failure raises. A graph whose every edge is ``null`` passes validation, because
parse failures and genuine non-edges are indistinguishable in the output; it
then yields zero subgraphs one stage later with no error anywhere.

    python graph_report.py --env quickstart [--category ticket_api] [--json]

Exit status: 0 all clear, 1 a warning worth reading, 2 the graph is unusable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

PLAN_OUT = os.path.join("multi_turn", "all_sequences.jsonl")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def inspect(base, cat):
    d = lambda *p: os.path.join(base, cat, *p)  # noqa: E731
    r = {"category": cat, "problems": [], "warnings": []}

    rows = load_jsonl(d("build_graph", "success.jsonl"))
    if not rows:
        r["problems"].append("no build_graph/success.jsonl — the stage produced nothing")
        return r

    graph = rows[-1].get("graph") or []
    r["sources"] = len(graph)

    pairs = null_edges = 0
    types = Counter()
    out_degree = Counter()
    nodes = set()

    for entry in graph:
        src = entry.get("source")
        nodes.add(src)
        for e in entry.get("all_edges", []):
            pairs += 1
            nodes.add(e.get("target"))
            edges = e.get("edges")
            if not edges:
                null_edges += 1
                continue
            # an edge payload is a dict, or a list of them
            for ed in (edges if isinstance(edges, list) else [edges]):
                if isinstance(ed, dict):
                    t = ed.get("type")
                    types[t] += 1
                    if t in ("explicit", "implicit"):
                        out_degree[src] += 1

    r["nodes"] = len(nodes)
    r["ordered_pairs"] = pairs
    r["null_or_empty_edges"] = null_edges
    r["edge_types"] = dict(types)
    r["usable_edges"] = types.get("explicit", 0) + types.get("implicit", 0)
    r["explicit_edges"] = types.get("explicit", 0)
    r["isolated_tools"] = sorted(n for n in nodes if n and out_degree[n] == 0)

    expected = len(graph) * (len(graph) - 1) if len(graph) > 1 else 0
    r["expected_pairs"] = expected
    if expected and pairs < expected:
        r["warnings"].append(
            f"{pairs} of {expected} ordered pairs present — build_graph did not finish, "
            f"or some pairs errored. Check llm_calls= in logs/{cat}/build_graph.log"
        )

    if pairs and null_edges == pairs:
        r["problems"].append(
            "every edge is null. Downstream consumers keep only edges typed exactly "
            "'explicit' or 'implicit', so plan_sequences will yield zero subgraphs. "
            "Usually the model's <graph> block failed to parse — check the raw content "
            "in build_graph/success.jsonl and the llm_calls= count in the stage log."
        )
    elif pairs and null_edges / pairs > 0.5:
        r["warnings"].append(
            f"{null_edges}/{pairs} edges are null ({null_edges / pairs:.0%}); parse "
            f"failures and genuine non-edges look identical here"
        )

    if r["usable_edges"] and not r["explicit_edges"]:
        r["problems"].append(
            "no explicit edges — only ordering edges were found. Chains will not thread "
            "real return values, which is the point of execute_sequences. Almost always "
            "means the specs reaching build_graph carried no `response` schema."
        )

    # plan_sequences
    seqs = load_jsonl(d("plan_sequences", PLAN_OUT))
    r["sequences"] = len(seqs)
    if seqs:
        # keys are stringified: multipath records carry no `number_turns`, so the
        # counter mixes ints with None and JSON cannot sort or encode that
        r["sequence_turns"] = {str(k): v for k, v in
                               Counter(s.get("number_turns") for s in seqs).items()}
        r["multipath_records"] = sum(1 for s in seqs if s.get("type") == "multipath")
    elif os.path.exists(d("plan_sequences")):
        r["problems"].append(
            f"plan_sequences ran but wrote no sequences to {PLAN_OUT}. With a usable "
            f"graph this means max_tools_per_sequence / turns_per_sequence are too "
            f"strict for a graph this sparse."
        )
    else:
        r["warnings"].append("plan_sequences has not run yet")

    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--category")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from vulcan.config import load_config

    config = load_config(args.env)
    base = config.get("output_base", f"./outputs/{args.env}")
    if not os.path.isdir(base):
        print(f"ERROR: output_base {base!r} does not exist", file=sys.stderr)
        return 2

    cats = ([args.category] if args.category
            else sorted(d for d in os.listdir(base)
                        if os.path.isdir(os.path.join(base, d))
                        and d not in ("logs", "training_dataset")))

    reports = [inspect(base, c) for c in cats]

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for r in reports:
            print(f"\n{'=' * 66}\n{r['category']}\n{'=' * 66}")
            if "nodes" in r:
                print(f"  tools in graph     : {r['nodes']}")
                print(f"  ordered pairs      : {r['ordered_pairs']} of {r['expected_pairs']} expected")
                print(f"  usable edges       : {r['usable_edges']}  "
                      f"(explicit {r['explicit_edges']}, implicit {r['edge_types'].get('implicit', 0)})")
                print(f"  null/empty edges   : {r['null_or_empty_edges']}")
                if r["isolated_tools"]:
                    print(f"  tools with no outgoing edge: {r['isolated_tools']}")
                print(f"  sequences planned  : {r.get('sequences', 0)}"
                      + (f"  turns={r['sequence_turns']}" if r.get("sequence_turns") else "")
                      + (f"  multipath={r['multipath_records']}" if r.get("multipath_records") else ""))
            for p in r["problems"]:
                print(f"\n  PROBLEM  {p}")
            for w in r["warnings"]:
                print(f"\n  warning  {w}")
        print()

    if any(r["problems"] for r in reports):
        return 2
    if any(r["warnings"] for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
