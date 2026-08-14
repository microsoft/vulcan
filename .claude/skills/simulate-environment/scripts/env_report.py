#!/usr/bin/env python3
"""Build a review packet for one VULCAN environment class.

Runs the checks a machine can settle exactly, then prints everything a reviewer
needs to judge the ones it cannot. It never decides whether an environment is
"good" — it removes the mechanical questions so the review can spend its
attention on behaviour.

    python env_report.py --env quickstart [--category ticket_api] [--json]

Reads, under ``output_base`` from the named config:
    <category>/assemble_environment/success.jsonl   tool_code (the class)
    <category>/debug_environment/success.jsonl      environment_ready
    <category>/generate_states/success.jsonl        generated initial states
    <category>/env_grouped/success.jsonl            tool schemas

Exit status is 0 whenever the packet was produced. Findings are data, not
failures — the reviewer decides what matters.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict

# Import VULCAN from the repo this skill ships with.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# Modules whose use makes a tool's output vary between identical calls. A
# simulated environment has to be deterministic: `execute_sequences` runs a
# chain, then `generate_trajectories` runs the same tools again, and the
# trajectory is only coherent if the second run agrees with the first.
NONDETERMINISTIC = {"random", "time", "datetime", "uuid", "secrets", "os"}
MUTATING_METHODS = {"append", "extend", "insert", "pop", "remove", "clear",
                    "update", "setdefault", "popitem", "sort", "reverse", "add", "discard"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def state_keys_touched(func_node):
    """Return (reads, writes) of top-level ``self.state[...]`` keys in a method.

    A write is an assignment target, an augmented assignment, a ``del``, or a
    call to a mutating method on the subscript. Everything else is a read.
    """
    reads, writes = set(), set()

    def key_of(node):
        """`self.state["x"]` -> "x", else None."""
        if not isinstance(node, ast.Subscript):
            return None
        val = node.value
        if not (isinstance(val, ast.Attribute) and val.attr == "state"
                and isinstance(val.value, ast.Name) and val.value.id == "self"):
            return None
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            return idx.value
        return "<dynamic>"

    for node in ast.walk(func_node):
        # writes: assignment targets
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    k = key_of(sub)
                    if k:
                        writes.add(k)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            k = key_of(node.target)
            if k:
                writes.add(k)
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                k = key_of(tgt)
                if k:
                    writes.add(k)
        # writes: mutating call on a state subscript
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in MUTATING_METHODS:
                k = key_of(node.func.value)
                if k:
                    writes.add(k)
        # reads: any remaining subscript
        k = key_of(node)
        if k:
            reads.add(k)

    return reads - writes, writes


def unused_params(func_node):
    """Parameters never referenced anywhere in the body."""
    args = func_node.args
    names = [a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs) if a.arg != "self"]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    used = {n.id for n in ast.walk(func_node) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(func_node) if isinstance(n, ast.Attribute)}
    # a name appearing only in the signature's own defaults does not count
    return [n for n in names if n not in used]


def affects_return(func_node, param):
    """Whether *param* is reachable from any return value or any state write.

    Deliberately generous: it answers "is this argument connected to anything
    observable", not "is it used correctly". A False here is a strong signal;
    a True says only that the reviewer has to look.
    """
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            if any(isinstance(n, ast.Name) and n.id == param for n in ast.walk(node.value)):
                return True
        if isinstance(node, ast.Assign):
            tgt_is_state = any(
                isinstance(s, ast.Attribute) and s.attr == "state"
                for t in node.targets for s in ast.walk(t)
            )
            if tgt_is_state and any(
                isinstance(n, ast.Name) and n.id == param for n in ast.walk(node.value)
            ):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in MUTATING_METHODS:
                if any(isinstance(n, ast.Name) and n.id == param
                       for a in node.args for n in ast.walk(a)):
                    return True
    return False


def nondeterminism(tree):
    """Imports and calls that make a tool's output vary between identical runs."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in NONDETERMINISTIC:
                    hits.append((node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in NONDETERMINISTIC:
                names = ", ".join(a.name for a in node.names)
                hits.append((node.lineno, f"from {node.module} import {names}"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            val = node.func.value
            if isinstance(val, ast.Name) and val.id in NONDETERMINISTIC:
                hits.append((node.lineno, f"{val.id}.{node.func.attr}()"))
    return hits


def analyse(code, spec_names, states):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"fatal": f"tool_code does not parse: {e}"}

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        return {"fatal": "no class definition in tool_code"}
    cls = classes[-1]

    methods = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            if node.name == "function_name_mapping":
                continue
            reads, writes = state_keys_touched(node)
            params = [a.arg for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
                      if a.arg != "self"]
            unused = unused_params(node)
            inert = [p for p in params if p not in unused and not affects_return(node, p)]
            methods[node.name] = {
                "line": node.lineno,
                "params": params,
                "unused_params": unused,
                "inert_params": inert,
                "state_reads": sorted(reads),
                "state_writes": sorted(writes),
                "touches_state": bool(reads or writes),
            }

    # default state keys, from `self.state = {...}` in __init__
    default_keys = []
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and tgt.attr == "state"
                        and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
                    v = node.value
                    if isinstance(v, ast.IfExp):
                        v = v.orelse
                    if isinstance(v, ast.Dict):
                        default_keys = [k.value for k in v.keys
                                        if isinstance(k, ast.Constant) and isinstance(k.value, str)]

    # function_name_mapping() is the only dispatch surface: a method absent from
    # it can never be called, however correct it is.
    mapped = set()
    for node in ast.walk(cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "function_name_mapping":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            mapped.add(k.value)

    all_reads = {k for m in methods.values() for k in m["state_reads"]}
    all_writes = {k for m in methods.values() for k in m["state_writes"]}
    generated_keys = sorted({k for s in states for k in (s or {}).keys()})

    return {
        "class_name": cls.name,
        "methods": methods,
        "default_state_keys": sorted(default_keys),
        "generated_state_keys": generated_keys,
        "state_keys_read": sorted(all_reads),
        "state_keys_written": sorted(all_writes),
        "nondeterminism": nondeterminism(tree),
        "spec_names": sorted(spec_names),
        "mapped_names": sorted(mapped),
        "methods_without_spec": sorted(set(methods) - spec_names),
        "specs_without_method": sorted(spec_names - set(methods)),
        "specs_not_mapped": sorted(spec_names - mapped),
        "methods_not_mapped": sorted(set(methods) - mapped),
        # a key nothing ever reads is dead seed data; a key read but never
        # seeded means the tool depends on state no state file provides
        "seeded_never_read": sorted((set(default_keys) | set(generated_keys)) - all_reads - all_writes),
        "read_never_seeded": sorted(all_reads - set(default_keys) - set(generated_keys)),
        "read_only_methods": sorted(n for n, m in methods.items() if m["state_reads"] and not m["state_writes"]),
        "stateless_methods": sorted(n for n, m in methods.items() if not m["touches_state"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, help="config name, e.g. quickstart")
    ap.add_argument("--category", help="one category; default every category found")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    from vulcan.config import load_config

    config = load_config(args.env)
    base = config.get("output_base", f"./outputs/{args.env}")
    if not os.path.isdir(base):
        print(f"ERROR: output_base {base!r} does not exist — has the pipeline run?", file=sys.stderr)
        return 2

    cats = ([args.category] if args.category
            else sorted(d for d in os.listdir(base)
                        if os.path.isdir(os.path.join(base, d)) and d not in ("logs", "training_dataset")))

    report = {}
    for cat in cats:
        d = lambda *p: os.path.join(base, cat, *p)  # noqa: E731
        env_rows = load_jsonl(d("assemble_environment", "success.jsonl"))
        if not env_rows:
            report[cat] = {"fatal": "no assemble_environment/success.jsonl — stage did not produce a class"}
            continue

        code = env_rows[-1].get("tool_code", "")
        grouped = load_jsonl(d("env_grouped", "success.jsonl"))
        spec_names = set()
        for row in grouped:
            for api in row.get("api_list", []):
                spec = api.get("normalized_schema") or api.get("function") or {}
                if spec.get("name"):
                    spec_names.add(spec["name"])
        if not spec_names:
            for api in env_rows[-1].get("api_list", []):
                spec = api.get("normalized_schema") or api.get("function") or {}
                if spec.get("name"):
                    spec_names.add(spec["name"])

        states = [r.get("initial_state") for r in load_jsonl(d("generate_states", "success.jsonl"))]
        dbg = load_jsonl(d("debug_environment", "success.jsonl"))

        r = analyse(code, spec_names, [s for s in states if isinstance(s, dict)])
        r["environment_ready"] = dbg[-1].get("environment_ready") if dbg else None
        r["n_generated_states"] = len([s for s in states if isinstance(s, dict)])
        r["code_path"] = d("assemble_environment", "success.jsonl")
        r["code_lines"] = code.count("\n") + 1
        report[cat] = r

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for cat, r in report.items():
        print(f"\n{'=' * 70}\n{cat}\n{'=' * 70}")
        if "fatal" in r:
            print(f"  FATAL: {r['fatal']}")
            continue
        print(f"  class {r['class_name']}  ({r['code_lines']} lines)   "
              f"environment_ready={r['environment_ready']}   states={r['n_generated_states']}")

        def section(title, items, fmt=str):
            if items:
                print(f"\n  {title}")
                for it in items:
                    print(f"    - {fmt(it)}")

        section("[5] tool schemas with no method (missing coverage)", r["specs_without_method"])
        section("[5] tool schemas missing from function_name_mapping (unreachable)",
                r["specs_not_mapped"])
        section("[5] methods defined but not mapped (dead code)", r["methods_not_mapped"])
        section("[5] methods with no tool schema (extra)", r["methods_without_spec"])
        section("[1] methods that never touch self.state",
                r["stateless_methods"])
        section("[1] methods that only read state, never write",
                r["read_only_methods"])
        section("[2] state keys read but never seeded", r["read_never_seeded"])
        section("[2] seeded state keys nothing ever reads", r["seeded_never_read"])
        section("[4] non-deterministic constructs", r["nondeterminism"],
                fmt=lambda t: f"line {t[0]}: {t[1]}")

        unused = {n: m["unused_params"] for n, m in r["methods"].items() if m["unused_params"]}
        section("[3] parameters never referenced", sorted(unused.items()),
                fmt=lambda kv: f"{kv[0]}(): {', '.join(kv[1])}")
        inert = {n: m["inert_params"] for n, m in r["methods"].items() if m["inert_params"]}
        section("[3] parameters referenced but not reaching a return or a state write",
                sorted(inert.items()), fmt=lambda kv: f"{kv[0]}(): {', '.join(kv[1])}")

        print(f"\n  state keys — default: {r['default_state_keys']}")
        print(f"  state keys — generated: {r['generated_state_keys']}")
        print(f"  per method:")
        for name, m in sorted(r["methods"].items()):
            print(f"    {name}(line {m['line']}) params={m['params']} "
                  f"reads={m['state_reads']} writes={m['state_writes']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
