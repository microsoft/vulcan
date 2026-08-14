---
name: simulate-environment
description: Build a VULCAN input catalog from tool schemas a user shares, run the environment-simulation phase, and review the generated environment class for state/behaviour mismatches, unusable initial states, arguments with no effect, non-determinism, and missing tool coverage. Use when starting from raw tool schemas, or after debug_environment on an existing run.
---

# Simulate an environment

Covers the first phase of a VULCAN run end to end: **schemas in → executable environment
out → reviewed before anyone spends money on it**.

Pick the mode that matches where the user is:

- **[Mode A](#mode-a--build-the-input-catalog)** — they pasted or pointed at tool schemas
  and there is no catalog yet
- **[Mode B](#mode-b--run-the-phase)** — a catalog exists; run steps 1–5
- **[Mode C](#mode-c--review-the-environment)** — the phase has run; review before the graph

Mode C is the one that must not be skipped. `build_graph` costs *n·(n−1)* model calls for
*n* tools, and `environment_ready` is a much weaker signal than it looks — the tests that
set it were written by the same model that wrote the class, from the same specs, so they
agree with its mistakes.

---

## Mode A — build the input catalog

### The format, in one line

**One JSON object per line. One line per category. Every tool inside `api_list`.**

```json
{"category_name": "ticket_api", "environment_type": "tools_only", "api_list": [{"function": {"name": "search_users", "description": "...", "parameters": {...}, "response": {...}}}]}
```

| Field | Required | Notes |
|---|---|---|
| `category_name` | yes | unique; also a directory name, so keep it filesystem-safe |
| `api_list` | yes | list of `{"function": {...}}`; **the tools must be in here** |
| `environment_type` | no | `tools_only` \| `with_user_and_policy` \| `with_user`; overrides the config for this category |
| `policy_rules` | no | policy text, **inline on this record**; required in practice for `with_user_and_policy` |
| `api_list[].constraints` | no | free text about one tool, used by `generate_tools` |

Inside `function`: `name`, `description`, `parameters` (JSON Schema with `properties` and
`required`), and `response`. The file extension is conventionally `.jsonl`, but a `.json`
file containing JSONL is common in the wild and loads the same — it is the *line* structure
that matters, not the name.

Write `response` if you possibly can. `build_graph` reads it to find an `explicit` edge —
one tool's output feeding another's input — which is what makes multi-turn chains thread
real values. Without it, `normalize_specs` generates one at a model call per tool, and a
generated response schema is a guess about your API. See
[`docs/input_format.md`](../../../docs/input_format.md#why-response-is-worth-writing).

### Two failures that produce no error

Both are documented in
[`docs/input_format.md`](../../../docs/input_format.md#two-ways-this-file-silently-fails).
Read that section — do not re-derive it. In short:

1. **A file with one row per *function* loads as zero tools.** `_load_catalog` reads
   `item.get("api_list", [])`, so `{"category_name": ..., "function": {...}}` parses fine
   and yields a category with nothing in it. This is the shape most exported tool dumps
   have, and the script below regroups it for you.
2. **Policy in a separate file is never read.** Only inline `policy_rules` is loaded. A
   run with a separate policy file still reports `has_policy` and still resolves to
   `with_user_and_policy` — both come from `environment_type`, not from any policy text —
   so the agent gets judged against rules it was never shown.

### Build it

```bash
python .claude/skills/simulate-environment/scripts/build_catalog.py \
    --tools <their-file.json> \
    --category <name> \
    --environment-type tools_only \
    --out examples/<name>.jsonl
```

It auto-detects bare function objects, OpenAI `{"type": "function", ...}` arrays,
already-wrapped entries, a dict keyed by category, ungrouped one-row-per-function files,
and JSONL of any of those. Add `--policy <file-or-text>` for a policy category; that
implies `with_user_and_policy`.

If the user pasted schemas into the conversation rather than giving a file, write them to
a `.json` file first and run the script on that. Do not hand-assemble the catalog — the
script's value is the last thing it does.

**It loads the result through `EnvironmentConfig` and prints the tool count VULCAN will
actually see.** A catalog that writes cleanly but loads as 0 tools is the failure this
prevents, so treat a mismatch there as fatal. It also audits each tool for a missing
description, a missing `response`, undescribed parameters, and `required` naming a
parameter that is not in `properties`, and it ends with the `build_graph` cost.

### Then point a config at it

```yaml
# configs/<name>.yaml
environment: <name>
input:
  catalog: examples/<name>.jsonl
output_base: ./outputs/<name>
llm:
  provider: openai
  model: gpt-4o
```

Copy [`configs/quickstart.yaml`](../../../configs/quickstart.yaml) and edit it rather than
writing one from scratch. **Never set `defaults.model`** — it overrides `llm.model` for
every stage, which in a mixed-provider config sends one vendor's model name to another.

---

## Mode B — run the phase

```bash
vulcan --env <config> --pipeline --start normalize_specs --end generate_states
```

| # | Step | Does |
|---:|---|---|
| 1 | `normalize_specs` | fix name/description mismatches; generate missing `response` schemas |
| 2 | `generate_tools` | one Python `Tool` class per API; review; execute and fix |
| 3 | `verify_tools` | verify → fix → re-verify; reject unused `invoke()` params |
| 4 | `assemble_environment` | merge into one class with `function_name_mapping()` over `self.state` |
| 5 | `debug_environment` | generate tests, run them, repair, set `environment_ready` |
| 6 | `generate_states` | initial states — the pipeline's only source of data diversity |

> `generate_tools`, `debug_environment` and later stages **execute model-generated Python
> in this process with no sandbox**. That is how simulation works here, not an oversight.
> Run it somewhere disposable: a container, a VM, or a throwaway CI worker.

`env_grouped/success.jsonl` is written after `verify_tools` by `StageRunner`, not by a
registered step. If nothing survived `verify_tools` with a `tool_code`, it is skipped and
the next stages fail with `input not found`.

Then go to Mode C. Do not go straight to `build_graph`.

---

## Mode C — review the environment

This is your review, not a script's. You read the environment class and judge it. The
script in step 3 is an optional cross-check that catches what an eye skips on a long file;
the skill works without it.

### 1. Read the artifacts

| What | Where |
|---|---|
| the environment class | `<output_base>/<category>/assemble_environment/success.jsonl` → `tool_code` |
| the generated initial states | `<output_base>/<category>/generate_states/success.jsonl` → `initial_state` |
| the schemas it must satisfy | `<output_base>/<category>/env_grouped/success.jsonl` → `api_list[].normalized_schema` |

Read the whole class. The failures that matter only appear when you hold the state and the
methods together.

### 2. The five dimensions

Work through each one against the code you just read. Every one of these has a legitimate
version, so the judgement is the work: an AST can tell you a method never writes state, but
only you can tell whether it should have.

**1 · State and behaviour disagree.** For each method, ask what its name and description
promise, then check `self.state` actually changes that way. `create_ticket` returning
`{"ticket_id": "t_1"}` without appending to `self.state["tickets"]` looks fine for one call
and breaks when a later turn calls `get_ticket("t_1")` — which is exactly what multi-turn
sequences do. A method that only reads is fine when it is a genuine reader (`get_*`,
`search_*`, `list_*`) and a defect when it is not. Check the reverse too: a key written but
never read is a change nothing can observe, so it cannot support a dependency edge.

**2 · The initial state cannot exercise the tools.** Read the generated states next to the
methods. A key a method reads that no state seeds means a `KeyError` or an empty return; a
seeded key nothing reads is dead weight. Then judge sufficiency: for each tool, is there a
row that makes it return something non-empty? A `search_users` over an empty list can never
succeed, and every trajectory through it is a dead end. Check the values line up across
tools — if `get_order(order_id)` needs an id `list_orders` returns, those ids have to
match, or no chain works.

**3 · Arguments with no effect.** Two grades. An argument never mentioned in the body is
dead on arrival, and worth flagging loudly: `generate_tools` and `verify_tools` already
reject those per tool, so one surviving into the assembled class was introduced during
assembly. Subtler and more common is an argument that *is* referenced but reaches neither
the return value nor a state write — read into a local, validated, logged, then discarded.
Either way the model learns to pass an argument that does not matter, and the judge cannot
tell a right value from a wrong one.

**4 · Noisy functionality.** The environment runs twice — `execute_sequences` proves the
chain, `generate_trajectories` replays it. Output that varies between identical calls makes
the second disagree with the first. Look for `random`, `time`, `datetime`, `uuid`,
`secrets` and `os`, and judge each: a seeded `random.Random(0)` is fine, a bare
`random.randint()` inside a returned id is not. Noise is also behavioural, and this part is
purely yours: responses padded with fields nothing consumes, `except` blocks broad enough
that a wrong argument returns success anyway, tools that succeed no matter what they are
given.

**5 · Tool coverage.** Compare the tool schemas against both the methods *and*
`function_name_mapping()`. Three distinct failures: a schema with no method is uncallable; a
schema whose method exists but is missing from the mapping is unreachable, because the
mapping is the only dispatch surface the pipeline uses; a mapped name with no method is a
runtime error waiting. The middle one is what slips through, because the class looks
complete.

### 3. Cross-check with the script (optional)

```bash
python .claude/skills/simulate-environment/scripts/env_report.py --env <config>
```

`--category <name>` narrows it; `--json` is machine-readable. It reports the same five
dimensions from the AST, tagged `[1]`–`[5]`: which methods write state, which parameters
reach a return, which keys are read but never seeded, which schemas are unmapped.

Use it to catch what you missed on a long file, and to check yourself — if it flags
something you cleared, look again. Do not treat its output as findings: every item it
prints has a legitimate version it cannot distinguish from a defect, which is why the
judgement above comes first.

### 4. Report

Group by dimension, most costly first. Per finding: the method or key with its line
number, what breaks downstream **named by stage**, and the smallest fix. Separate
**confirmed** (traced in the code) from **suspected** (flagged, looks wrong, depends on
intent). Do not report a script finding you did not check — one false positive and the
reader stops trusting the rest.

End with one line: proceed to `build_graph`, or regenerate first — and if regenerating,
which stage. `generate_states` for bad seed data, `assemble_environment` for mapping and
state problems, `generate_tools` for a single broken tool. `generate_states` is by far the
cheapest.

---

## Notes

- Nothing here re-runs the pipeline except Mode B, and neither script makes a model call.
- The scripts never execute the generated environment. It is unsandboxed model-written
  Python; reading it is safe, and a review step should not add a new place it runs.
- `env_report.py` always exits 0 — findings are data, and the decision is yours.
  `build_catalog.py` exits non-zero when the catalog does not load as written.
