# VULCAN pipeline skills

Four skills for driving a VULCAN run from a coding agent. The procedure is the skill; the
scripts are optional helpers that settle the mechanical questions exactly, so the agent's
attention goes to the judgements a script cannot make.

Start with **`run-vulcan`** unless you already know which phase you want.

| Skill | Use it | Script |
|---|---|---|
| [`run-vulcan`](run-vulcan/) | the whole pipeline from a conversation; calls the other three in order and stops before anything expensive | *(none)* |
| [`simulate-environment`](simulate-environment/) | schemas → catalog → run steps 1–6 → review, before you spend money on the graph | `build_catalog.py`, `env_report.py` |
| [`build-tool-graph`](build-tool-graph/) | building the tool dependency graph for a category | `graph_report.py` |
| [`generate-training-data`](generate-training-data/) | states → samples → chains → queries → trajectories → judged → dataset | `layout_check.py` |

Run them in that order for a new catalog. The reason for the order is cost: `build_graph`
alone is *n·(n−1)* model calls for *n* tools, and reviewing the environment first is the
cheapest way to avoid paying it on a broken one.

## Using them

**Claude Code** discovers them automatically from this directory. Invoke by name:

```
/run-vulcan
/simulate-environment
/build-tool-graph
/generate-training-data
```

Or just say what you want. `run-vulcan`'s description is written so Claude Code selects it
for "run VULCAN on these tool schemas" without being named.

**GitHub Copilot** reads the same procedures through the prompt files in
[`.github/prompts/`](../../.github/prompts/), which point back here so there is one source
of truth. In Copilot Chat: `/vulcan-simulate-environment`, and so on.

**Any other agent, or by hand** — the scripts are plain Python with no agent dependency:

```bash
python .claude/skills/simulate-environment/scripts/build_catalog.py --tools tools.json --category demo --out examples/demo.jsonl
python .claude/skills/simulate-environment/scripts/env_report.py   --env quickstart
python .claude/skills/build-tool-graph/scripts/graph_report.py     --env quickstart
python .claude/skills/generate-training-data/scripts/layout_check.py --env quickstart
```

The report scripts take `--env <config-name>` (or a path to a config), an optional
`--category`, and `--json`. They read a completed run's artifacts under `output_base`.
`build_catalog.py` instead takes `--tools` and `--out`, and is the one you run *before* a
run exists. None of them re-runs the pipeline or makes a model call.

Exit codes: `env_report.py` always returns 0 — its findings are data, and the decision is
the reviewer's. `build_catalog.py`, `graph_report.py` and `layout_check.py` return non-zero
when the result is unusable, so they can gate a script.

## What they do not do

They do not execute the generated environment. That code is model-written Python which
VULCAN runs without a sandbox — reading it is safe, running it outside the pipeline is not
something a review step should add. See the warning at the top of the [README](../../README.md).
