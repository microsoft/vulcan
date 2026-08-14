# Input format

The standard input catalog. One file describes every category of an environment; point a
config at it with `input.catalog` and VULCAN needs no code changes.

Loaded by `EnvironmentConfig._load_catalog`
([`vulcan/environment/base.py`](../vulcan/environment/base.py)). The catalogs in
[`../examples/`](../examples/) are the reference implementation of everything below.

## File format

**JSONL** — one JSON object per line, one line per **category**. Not a JSON array, and not
one line per function (see [the first silent failure](#two-ways-this-file-silently-fails)).

```json
{"category_name": "ticket_api",
 "api_list": [
   {"function": {"name": "create_ticket",
                 "description": "Open a new support ticket for a user.",
                 "parameters": {"type": "object",
                                "properties": {"reporter_id": {"type": "string", "description": "…"},
                                               "title": {"type": "string", "description": "…"}},
                                "required": ["reporter_id", "title"]},
                 "response": {"type": "object",
                              "properties": {"ticket_id": {"type": "string", "description": "…"},
                                             "status": {"type": "string", "description": "…"}}}},
    "constraints": "The title must be non-empty. The reporter_id must refer to an existing user."}],
 "environment_type": "tools_only"}
```

That is the shape of every line in [`examples/ticket_api.jsonl`](../examples/ticket_api.jsonl)
(8 tools, no policy). [`examples/hotel_booking.jsonl`](../examples/hotel_booking.jsonl)
adds `policy_rules` and `environment_type: with_user_and_policy`;
[`examples/ecommerce.jsonl`](../examples/ecommerce.jsonl) is the same shape with longer
dependency chains; [`examples/all_examples.jsonl`](../examples/all_examples.jsonl) is the
three of them concatenated — three lines, three categories, one file.

## Fields

| Field | Required | Meaning |
|---|:---:|---|
| `category_name` | yes | unique category id. Rows without it are **skipped silently**. It also names the output directory, `<output_base>/<category_name>/` |
| `api_list` | yes | the tools in this category. Missing → the category loads with **zero APIs** |
| `api_list[].function` | yes | OpenAI function-calling shape. A bare function object (top-level `name`) is auto-wrapped into `{"function": …}` |
| `.function.name` / `.description` | yes | keep them consistent — `normalize_specs` rewrites the description when they disagree |
| `.function.parameters` | yes | JSON Schema. Describe every property; the text drives generated code, queries and judging |
| `.function.response` | no | output schema. **Omitted → `normalize_specs` generates one**; present → that sub-step is skipped |
| `api_list[].constraints` | no | free-text behavioural rules for one API. Switches `generate_tools` to a constraint-aware prompt for that item |
| `policy_rules` | no | domain policy for the whole category. Must be **inline here** |
| `environment_type` | no | per-category override: `tools_only` \| `with_user_and_policy` \| `with_user` |

`_load_catalog` reads exactly four top-level keys — `category_name`, `api_list`,
`policy_rules`, `environment_type` — and ignores everything else on the line, so extra bookkeeping
fields are harmless but also inert.

`constraints` is the field to reach for when the schema cannot express the behaviour —
ordering requirements, state preconditions, validation rules. It rides along with the item
through `normalize_specs.preprocess_input` and `generate_tools` branches on its presence
per API, so a catalog can mix constrained and unconstrained tools freely. Loading any
constraint also flips `EnvironmentConfig.has_constraints`, which no step currently reads —
the per-item field is what does the work.

One thing about `category_name` beyond uniqueness: it is a directory name, so keep it
filesystem-safe. No name is otherwise special. Earlier versions dropped a tool called
`calculate` from three specific category names; that carve-out is gone, and
`tests/test_no_domain_carveouts.py` keeps it gone.

## Environment type

The default is declared in the **config**; the `environment_type` field on a line here
overrides it for that one category:

| `environment_type` | Simulated user | Policy |
|---|:---:|:---:|
| `tools_only` | no | no |
| `with_user_and_policy` | yes | yes |
| `with_user` | yes | no |

`get_environment_type(category)` resolves in this order:

1. `environment_type_map` in the config, or `environment_type` on the input row (the row is written into
   the same map at load time);
2. the category has `policy_rules` → `with_user_and_policy`, **even when the environment-level type
   is `tools_only`**;
3. the environment-level default from `environment_type`.

Loading any `policy_rules` also sets `has_policy` on the environment, regardless of the
configured type.

## Two ways this file silently fails

**Ungrouped catalogs load empty.** A file with one row per *function* — `category_name`
plus `function`, no `api_list` — parses without error and yields categories containing no
tools, because `_load_catalog` reads `item.get("api_list", [])`:

```text
{"category_name": "ticket_api", "function": {"name": "create_ticket", ...}}   ← wrong: 1 category, 0 APIs
{"category_name": "ticket_api", "api_list": [{"function": {...}}, ...]}       ← right: 1 category, 8 APIs
```

`normalize_specs` then logs `[normalize_specs/ticket_api] no APIs found in the input
catalog` and every later stage fails with `input not found`, because nothing was written
for them to read. If you are converting a flat list of functions, group it by category
first — one line per category, every tool inside `api_list`. The shipped examples are the
format to match.

**Policy in a separate file is never loaded.** Only inline `policy_rules` is read.
Benchmarks that ship their policy alongside the tools — a `policy_rules.jsonl` keyed by
category, say — leave `get_domain_rules()` returning `None` while the run still reports
`has_policy: True` and resolves the category to `with_user_and_policy`, because both come
from the configured `environment_type`, not from any policy text. No policy text reaches
`generate_queries` or `generate_trajectories`, and the judge scores the agent against rules
it was never shown. Merge before running:

```python
import json

cats = [json.loads(l) for l in open("my_catalog.jsonl") if l.strip()]
pol = {r["category_name"]: r["rules"]
       for r in (json.loads(l) for l in open("policy_rules.jsonl") if l.strip())}

with open("my_catalog_with_policy.jsonl", "w") as fh:
    for rec in cats:
        rec["policy_rules"] = pol.get(rec["category_name"], "")
        fh.write(json.dumps(rec) + "\n")
```

[`examples/hotel_booking.jsonl`](../examples/hotel_booking.jsonl) shows the merged result:
a numbered policy on the category record, next to the `api_list` it governs.

## Why `response` is worth writing

`normalize_specs` generates a `response` schema only when one is absent, so a catalog that
already ships them skips that sub-step entirely — cheaper, and the every-tool-has-a-response
convention in the shipped examples is deliberate. Supply one wherever you can: a generated
schema is a model's guess at the API's output shape, while a hand-written one is what
`generate_tools` implements against and what `debug_environment` checks the environment's actual
return values against.

Either way, `normalize_specs` writes `normalized_schema` on **every** API — the resolved
spec, corrected if `name` and `description` disagreed, carrying a `response` either from
your catalog or generated for it. That one field is the contract with every later step.
`normalized_function` is also written when a correction happened, but it is **diagnostic
only**: it records that the correction took place, and nothing selects on it.

`build_graph` reads `normalized_schema` and nothing else, raising `StepError` when it is
missing or null instead of falling back to the raw catalog `function`. Dependency detection
needs the `response` schema to find an `explicit` (output→input) edge at all, and the raw
catalog spec carries `response` only when you supplied one — so a fallback would silently
downgrade the graph to ordering-only `implicit` edges rather than fail. Failing loudly is the
intended behaviour; if you see that error, rerun `normalize_specs` for the category.

## Adding an environment

1. **Write the catalog** in the format above, anywhere on disk.
2. **Copy the template.** [`vulcan/config/example_environment.yaml`](../vulcan/config/example_environment.yaml)
   → `./configs/<name>.yaml`, then set `environment_type`, `input.catalog`,
   `output_base` (`./outputs/<name>`) and `categories`:

   ```yaml
   environment: hotel
   environment_type: with_user_and_policy
   input:
     catalog: examples/hotel_booking.jsonl
   output_base: ./outputs/hotel
   categories:
     - hotel_booking
   ```

   `load_config` expands `~` and resolves a relative path against the working directory
   first, then the repo root, so both keys stay portable. `./configs/<name>.yaml` wins over
   the packaged `vulcan/config/<name>.yaml` of the same name.
3. **Run it.** `vulcan --env <name> --pipeline`, or
   `python -m vulcan.cli --env <name> --pipeline`.

`categories` lists which categories in the catalog to process; omit it to run every
category the file contains. Remember that `build_graph` costs *n·(n−1)* model calls for *n*
tools in a category, so catalog size is the main cost lever — see
[`configuration.md`](configuration.md).

## Treat the catalog as trusted input

Every prompt in Phase 1 is built from this file, and the Python those prompts produce runs
in the host process with no sandbox. That makes a catalog you did not write a
code-execution vector: descriptions, constraints and policy text are instructions to a
model whose output you then execute. Only point VULCAN at catalogs you trust, and run it in
a disposable environment. See the Security section of the
[top-level README](../README.md).
