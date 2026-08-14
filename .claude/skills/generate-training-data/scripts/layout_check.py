#!/usr/bin/env python3
"""Verify a run's output layout against the documented one, stage by stage.

The layout in ``vulcan/core/stage_runner.py`` is the contract the README and
``docs/pipeline_stages.md`` describe. Three stages do not follow the usual
``<step>/success.jsonl`` pattern, and a reader who assumes they do looks in the
wrong place and concludes the stage produced nothing:

    <category>/plan_sequences/multi_turn/all_sequences.jsonl
    <category>/execute_sequences/pre_query_all_modes.jsonl
    <category>/iter_<k>/{trajectories,judged}/<label>/success.jsonl

    python layout_check.py --env quickstart [--iteration 0] [--label native] [--json]

Also reports the `llm_calls=` count per stage, which is the only honest signal
that a stage did work: steps catch per-item errors and emit the row anyway, so a
stage can report success with every model call having failed.

Exit status: 0 complete, 1 gaps, 2 nothing to check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# (stage, relative path under <output_base>/<category>, produces model calls)
PER_CATEGORY = [
    ("normalize_specs",      "normalize_specs/success.jsonl",                 True),
    ("generate_tools",       "generate_tools/success.jsonl",                  True),
    ("verify_tools",         "verify_tools/success.jsonl",                    True),
    ("env_grouped",          "env_grouped/success.jsonl",                     False),
    ("assemble_environment", "assemble_environment/success.jsonl",            True),
    ("debug_environment",    "debug_environment/success.jsonl",               True),
    ("generate_states",      "generate_states/success.jsonl",                 True),
    ("sample_arguments",     "sample_arguments/success.jsonl",                True),
    ("build_graph",          "build_graph/success.jsonl",                     True),
    ("plan_sequences",       "plan_sequences/multi_turn/all_sequences.jsonl", False),
    ("execute_sequences",    "execute_sequences/pre_query_all_modes.jsonl",   False),
    ("generate_queries",     "generate_queries/success.jsonl",                True),
]

TRAINING_RECORD_KEYS = {"id", "category_name", "messages"}
TAGGED_RECORD_KEYS = {"id", "category", "messages"}


def count_lines(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def llm_calls(log_path):
    """Last `llm_calls=<n>` in a stage log, or None."""
    if not os.path.exists(log_path):
        return None
    hits = re.findall(r"llm_calls=(\d+)", open(log_path, encoding="utf-8", errors="replace").read())
    return int(hits[-1]) if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--category")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--label", default="native",
                    help="output_label used for trajectories/judged (default: native)")
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
    if not cats:
        print(f"ERROR: no category directories under {base!r}", file=sys.stderr)
        return 2

    k, label = args.iteration, args.label
    report, missing = {}, []

    for cat in cats:
        stages = {}
        for stage, rel, calls in PER_CATEGORY:
            path = os.path.join(base, cat, rel)
            n = count_lines(path)
            entry = {"path": os.path.join(cat, rel), "rows": n, "present": n is not None}
            if calls:
                entry["llm_calls"] = llm_calls(os.path.join(base, "logs", cat, f"{stage}.log"))
            stages[stage] = entry
            if n is None:
                missing.append(f"{cat}: {stage} -> {rel}")

        for stage, rel in [
            ("generate_trajectories", f"iter_{k}/trajectories/{label}/success.jsonl"),
            ("judge_trajectories",    f"iter_{k}/judged/{label}/success.jsonl"),
        ]:
            path = os.path.join(base, cat, rel)
            n = count_lines(path)
            stages[stage] = {
                "path": os.path.join(cat, rel), "rows": n, "present": n is not None,
                "llm_calls": llm_calls(os.path.join(base, "logs", cat, f"{stage}.log")),
            }
            if n is None:
                missing.append(f"{cat}: {stage} -> {rel}")

        nxt = os.path.join(base, cat, f"iter_{k + 1}", "input.jsonl")
        stages["_rejects_next_iteration"] = {
            "path": os.path.join(cat, f"iter_{k+1}", "input.jsonl"),
            "rows": count_lines(nxt), "present": os.path.exists(nxt),
        }
        report[cat] = stages

    # exported dataset lives at the run root, not per category
    exports = {}
    for enc in ("json", "xml"):
        rel = os.path.join("training_dataset", enc, f"{{category}}_iter_{k}.jsonl")
        per_cat = {}
        for cat in cats:
            p = os.path.join(base, "training_dataset", enc, f"{cat}_iter_{k}.jsonl")
            n = count_lines(p)
            bad_keys = None
            if n:
                with open(p, encoding="utf-8") as fh:
                    first = json.loads(next(l for l in fh if l.strip()))
                keys = set(first)
                if not (TRAINING_RECORD_KEYS <= keys or TAGGED_RECORD_KEYS <= keys):
                    bad_keys = sorted(keys)
            per_cat[cat] = {"rows": n, "present": n is not None, "unexpected_keys": bad_keys}
            if n is None:
                missing.append(f"{cat}: export_dataset ({enc}) -> training_dataset/{enc}/{cat}_iter_{k}.jsonl")
        exports[enc] = {"pattern": rel, "categories": per_cat}
    report["_export"] = exports

    if args.json:
        print(json.dumps({"output_base": base, "iteration": k, "label": label,
                          "report": report, "missing": missing}, indent=2, sort_keys=True))
        return 0 if not missing else 1

    print(f"\noutput_base = {base}   iteration = {k}   label = {label}")
    for cat in cats:
        print(f"\n{'=' * 74}\n{cat}\n{'=' * 74}")
        print(f"  {'stage':<24} {'rows':>6}  {'llm_calls':>9}  path")
        for stage, e in report[cat].items():
            if stage.startswith("_"):
                continue
            rows = "—" if e["rows"] is None else e["rows"]
            calls = e.get("llm_calls")
            calls_s = "" if calls is None else str(calls)
            flag = "  " if e["present"] else "!!"
            note = ""
            if e["present"] and e["rows"] == 0:
                note = "   <- present but empty"
            if calls == 0:
                note += "   <- 0 model calls: every call failed"
            print(f"{flag}{stage:<24} {rows:>6}  {calls_s:>9}  {e['path']}{note}")
        rj = report[cat]["_rejects_next_iteration"]
        if rj["present"]:
            print(f"   {'rejects -> iter_' + str(k+1):<24} {rj['rows']:>6}             {rj['path']}")

    print(f"\n{'=' * 74}\nexported dataset\n{'=' * 74}")
    for enc, block in report["_export"].items():
        for cat, e in block["categories"].items():
            rows = "—" if e["rows"] is None else e["rows"]
            flag = "  " if e["present"] else "!!"
            note = f"   <- unexpected keys {e['unexpected_keys']}" if e["unexpected_keys"] else ""
            print(f"{flag}{enc:<6} {cat:<28} {rows:>6}  "
                  f"training_dataset/{enc}/{cat}_iter_{k}.jsonl{note}")

    if missing:
        print(f"\n{len(missing)} expected output(s) absent:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\nAll documented outputs present.")
    print()
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
