---
name: build-tool-graph
description: Run VULCAN's graph-construction phase correctly — build_graph and plan_sequences — with a cost estimate before you start and a check afterwards that the graph is actually usable. Use when building the tool dependency graph for a category, or when plan_sequences produced no sequences.
---

# Build the tool dependency graph

This phase turns a category's tool catalog into a directed graph, then into the multi-turn
sequences everything downstream is generated from. It is the most expensive stage in a run
and the easiest to get a silent empty result from.

Two facts to hold onto:

- **`build_graph` costs *n·(n−1)* model calls for *n* tools.** 56 at 8 tools, 462 at 22,
  1892 at 44. It asks about every *ordered* pair, so the cost is quadratic and the growth
  catches people out.
- **An empty graph is not an error.** The validator accepts a graph whose every edge is
  `null`, because a parse failure and a genuine non-edge are indistinguishable in the
  output. You get zero subgraphs one stage later with nothing logged.

## Step 1 — check the precondition

`build_graph` reads `env_grouped/success.jsonl`, which `StageRunner` writes after
`verify_tools` — it is not a registered step. If `verify_tools` produced nothing with a
`tool_code`, `env_grouped` is skipped and this stage fails with `input not found`.

```bash
wc -l <output_base>/<category>/env_grouped/success.jsonl
```

`build_graph` reads **only** `normalized_schema` and raises `StepError` naming the API if
it is missing. That is deliberate: dependency detection needs a `response` schema to find
an output→input edge at all, and falling back to the raw catalog spec would quietly
downgrade the whole graph to ordering-only edges. If it raises, re-run `normalize_specs`
for that category rather than working around it.

If you have not reviewed the environment yet, do that first — see the
`simulate-environment` skill. Every model call this stage makes is spent on a catalog nobody
checked.

## Step 2 — estimate before you spend

Count the tools and state the cost, out loud, before running:

```bash
python -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n=sum(len(r.get('api_list',[])) for r in rows)
print(f'{n} tools -> {n*(n-1)} model calls')
" <output_base>/<category>/env_grouped/success.jsonl
```

If that number is larger than the user expects, say so and stop. Options: cut the catalog,
or point `steps.build_graph.model` at a cheaper model — per-step `provider` / `model` /
`base_url` all take effect, because the CLI builds a client per stage.

## Step 3 — run

```bash
vulcan --env <config> --stage build_graph
vulcan --env <config> --stage plan_sequences
```

Or as a range, which is the same thing:

```bash
vulcan --env <config> --pipeline --start build_graph --end plan_sequences
```

**Do not run `score_sequences`.** It sits between them in `PIPELINE_ORDER`, but it is
deprecated — analysis only, nothing downstream reads its output, and it is the one stage
you can drop with no effect on the dataset. Running the range above skips it. A bare
`--pipeline` does not.

`plan_sequences` is deterministic and makes no model calls. It is cheap to re-run with
different `turns_per_sequence` or `max_tools_per_sequence` once the graph exists — and
that, not rebuilding the graph, is the right response to "too few sequences".

## Step 4 — verify

Read `build_graph/success.jsonl` and check the graph is not all-null: every entry has a
`source` and an `all_edges` list, and a usable edge is one whose payload has `type` set to
exactly `explicit` or `implicit`. Then check `plan_sequences` actually produced sequences.

The script does the counting for you, which on a 20-tool catalog is 380 pairs to eyeball:

```bash
python .claude/skills/build-tool-graph/scripts/graph_report.py --env <config>
```

Exit 0 clear, 1 warnings, 2 unusable. It reports tools in the graph, ordered pairs present
versus expected, usable versus null edges, tools with no outgoing edge, and sequences
planned.

What the results mean:

| Result | What happened | What to do |
|---|---|---|
| every edge null | the `<graph>` block did not parse on any call | check `llm_calls=` in `logs/<category>/build_graph.log`; if it is 0 every call failed — usually auth or a bad model id. If non-zero, the model's output format is the problem: try a stronger model for this step |
| pairs < expected | the stage did not finish, or some pairs errored | re-run the stage; it resumes on input hash, so completed pairs are not re-paid for |
| usable edges but no explicit ones | the specs carried no `response` schema | re-run `normalize_specs`; ordering-only edges cannot thread real values in `execute_sequences` |
| sequences = 0 with a good graph | `max_tools_per_sequence` / `turns_per_sequence` too strict for a sparse graph | loosen them and re-run `plan_sequences` only — it is free |
| many isolated tools | those tools genuinely have no relationship to the others | expected in a mixed catalog; a concern only if it is most of them |

## Step 5 — report

State the cost actually incurred (`llm_calls=` from the stage log, not your estimate),
the graph shape, and the sequence count. If anything is off, name the stage to re-run and
whether it costs model calls — `plan_sequences` is free, `build_graph` is not.

## Notes

- Resume is on the item's **input** hash, so changing a model or temperature does not
  invalidate completed items. To force a rebuild, delete the stage output directory.
- `enable_multipath` is `false` by default. It adds fork-merge subgraphs, and the extra
  records only matter if you are studying them.
- `implicit_edge_weight` (default `0.01`) controls how much an ordering-only edge counts
  toward a subgraph's popularity score. Raising it pulls more implicit-only subgraphs into
  the sample.
