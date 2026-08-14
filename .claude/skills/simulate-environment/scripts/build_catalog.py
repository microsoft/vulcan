#!/usr/bin/env python3
"""Turn tool schemas into a VULCAN input catalog, then prove it loads.

Accepts the shapes people actually have and normalises them to the one VULCAN
reads: **one JSON object per line, one line per category, every tool inside
`api_list`**.

    python build_catalog.py --tools my_tools.json --category billing \\
        --environment-type tools_only --out examples/billing.jsonl

Recognised inputs (auto-detected):

    [{"name": ..., "parameters": {...}}, ...]              bare function objects
    [{"type": "function", "function": {...}}, ...]         OpenAI tools array
    [{"function": {...}, "constraints": "..."}, ...]       already wrapped
    {"billing": [...], "support": [...]}                   dict keyed by category
    {"category_name": ..., "api_list": [...]}              already a catalog (revalidated)
    {"category_name": ..., "function": {...}}              ungrouped — regrouped for you
    one JSON object per line of any of the above           JSONL

The last input shape is the one that silently produces an empty catalog if you
feed it to VULCAN directly: `_load_catalog` reads `item.get("api_list", [])`, so
a file with one row per *function* loads as categories containing no tools. This
script regroups it instead.

After writing, it loads the result through `EnvironmentConfig` and reports the
category and tool counts VULCAN will actually see. A catalog that writes cleanly
but loads as 0 tools is the failure this exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

VALID_ENV_TYPES = ("tools_only", "with_user_and_policy", "with_user")


def read_any(path):
    """Read JSON or JSONL into a list of objects."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n} is neither valid JSON nor valid JSONL: {e}")
    return rows


def as_api_entry(obj):
    """Normalise one tool into `{"function": {...}}`, preserving `constraints`."""
    if not isinstance(obj, dict):
        return None
    # OpenAI tools array: {"type": "function", "function": {...}}
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        entry = {"function": obj["function"]}
    # already wrapped
    elif isinstance(obj.get("function"), dict):
        entry = {"function": obj["function"]}
    # bare function object
    elif "name" in obj:
        entry = {"function": {k: v for k, v in obj.items() if k != "constraints"}}
    else:
        return None
    if obj.get("constraints"):
        entry["constraints"] = obj["constraints"]
    elif isinstance(obj.get("function"), dict) and obj["function"].get("constraints"):
        entry["constraints"] = obj["function"].pop("constraints")
    return entry if entry["function"].get("name") else None


def collect(rows, default_category):
    """Return OrderedDict {category: {"api_list": [...], "policy_rules": str|None}}."""
    cats = OrderedDict()

    def bucket(name):
        return cats.setdefault(name, {"api_list": [], "policy_rules": None,
                                      "environment_type": None})

    for row in rows:
        if not isinstance(row, dict):
            continue

        # dict keyed by category -> list of tools
        if "category_name" not in row and "name" not in row and "function" not in row \
                and row.get("type") != "function" and "api_list" not in row:
            if all(isinstance(v, list) for v in row.values()) and row:
                for cat, tools in row.items():
                    b = bucket(cat)
                    for t in tools:
                        e = as_api_entry(t)
                        if e:
                            b["api_list"].append(e)
                continue

        cat = row.get("category_name") or default_category
        b = bucket(cat)
        if row.get("policy_rules"):
            b["policy_rules"] = row["policy_rules"]
        if row.get("environment_type"):
            b["environment_type"] = row["environment_type"]

        if isinstance(row.get("api_list"), list):          # already a catalog row
            for t in row["api_list"]:
                e = as_api_entry(t)
                if e:
                    b["api_list"].append(e)
        else:                                               # a single tool
            e = as_api_entry(row)
            if e:
                b["api_list"].append(e)

    return cats


def audit(entry):
    """Per-tool warnings that affect what the pipeline can do with it."""
    fn = entry.get("function", {})
    out = []
    name = fn.get("name", "<unnamed>")
    if not fn.get("description"):
        out.append(f"{name}: no description — generate_tools has little to work from")
    if not fn.get("response"):
        # not fatal: normalize_specs generates one, at one model call per tool
        out.append(f"{name}: no `response` schema — normalize_specs will generate one "
                   f"(costs a call, and a generated schema is a guess)")
    params = fn.get("parameters") or {}
    props = params.get("properties")
    if props is None:
        out.append(f"{name}: no parameters.properties — every argument will be undeclared")
    elif props:
        missing = [p for p, s in props.items()
                   if not isinstance(s, dict) or not s.get("description")]
        if missing:
            out.append(f"{name}: parameters without a description: {', '.join(sorted(missing))}")
    req = params.get("required")
    if req and props:
        unknown = [r for r in req if r not in props]
        if unknown:
            out.append(f"{name}: `required` names parameters not in properties: {unknown}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True, help="JSON or JSONL file of tool schemas")
    ap.add_argument("--category", help="category name when the input does not carry one")
    ap.add_argument("--environment-type", choices=VALID_ENV_TYPES, default="tools_only")
    ap.add_argument("--policy", help="policy text, or a path to a file holding it "
                                     "(implies --environment-type with_user_and_policy)")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    rows = read_any(args.tools)
    if not rows:
        raise SystemExit(f"{args.tools} is empty")

    default_cat = args.category or os.path.splitext(os.path.basename(args.out))[0]
    cats = collect(rows, default_cat)
    if not cats:
        raise SystemExit("no tools recognised — see the docstring for accepted shapes")

    policy = None
    if args.policy:
        policy = (open(args.policy, encoding="utf-8").read().strip()
                  if os.path.exists(args.policy) else args.policy)

    env_type = args.environment_type
    if policy and env_type != "with_user_and_policy":
        print("note: --policy given, setting environment_type to with_user_and_policy")
        env_type = "with_user_and_policy"

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    records, warnings = [], []
    for cat, b in cats.items():
        if not b["api_list"]:
            warnings.append(f"{cat}: no tools recognised — dropped")
            continue
        seen, deduped = set(), []
        for e in b["api_list"]:
            n = e["function"]["name"]
            if n in seen:
                warnings.append(f"{cat}: duplicate tool {n!r} — kept the first")
                continue
            seen.add(n)
            deduped.append(e)
            warnings.extend(f"{cat}: {w}" for w in audit(e))

        rec = OrderedDict()
        rec["category_name"] = cat
        rec["environment_type"] = b["environment_type"] or env_type
        pol = b["policy_rules"] or (policy if len(cats) == 1 else None)
        if pol:
            rec["policy_rules"] = pol
        elif rec["environment_type"] == "with_user_and_policy":
            warnings.append(
                f"{cat}: environment_type is with_user_and_policy but no policy_rules. "
                f"The run still reports has_policy, and no policy text reaches any prompt "
                f"— the agent is judged against rules it was never shown."
            )
        rec["api_list"] = deduped
        records.append(rec)

    if not records:
        raise SystemExit("every category was empty — nothing written")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nwrote {args.out}")
    for rec in records:
        print(f"  {rec['category_name']:<24} {len(rec['api_list']):>3} tools   "
              f"{rec['environment_type']}"
              f"{'   +policy' if 'policy_rules' in rec else ''}")

    # Prove it loads. A file that writes cleanly but loads as 0 tools is exactly
    # the failure this script exists to prevent, so never skip this.
    from vulcan.environment.base import EnvironmentConfig

    env = EnvironmentConfig({"environment": "check", "input": {"catalog": args.out}})
    loaded = env.get_categories()
    print(f"\nEnvironmentConfig sees {len(loaded)} categor{'y' if len(loaded) == 1 else 'ies'}:")
    ok = True
    for c in loaded:
        n = len(env.get_api_catalog(c))
        pol = "yes" if env.get_domain_rules(c) else "no"
        print(f"  {c:<24} {n:>3} tools   type={env.get_environment_type(c):<22} policy={pol}")
        if n == 0:
            ok = False
    if len(loaded) != len(records):
        ok = False

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if not ok:
        print("\nFAILED: the catalog does not load as written.", file=sys.stderr)
        return 2

    n_tools = sum(len(env.get_api_catalog(c)) for c in loaded)
    print(f"\nOK. Point a config at it:\n\n  input:\n    catalog: {args.out}\n")
    print(f"build_graph will cost {n_tools} tools -> "
          f"{sum(len(env.get_api_catalog(c)) * (len(env.get_api_catalog(c)) - 1) for c in loaded)} "
          f"model calls (summed per category).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
