---
name: run-vulcan
description: Drive a complete VULCAN run from a conversation — take tool schemas or an existing catalog, build the simulated environment, review it, construct the tool dependency graph, generate and judge trajectories, and export the training set. Use when someone wants to generate tool-calling training data and does not want to run the stages by hand.
---

# Run VULCAN end to end

The entry point. Someone has tool schemas and wants training data; this takes them the whole
way, stopping only where a decision or money is at stake.

You are driving a pipeline that **spends real money on someone else's API key**. The single
most useful thing you do is refuse to spend it on a broken environment. `build_graph` alone
costs *n·(n−1)* model calls for *n* tools: 56 at 8 tools, 462 at 22, 1892 at 44. Reviewing
before that gate is the whole reason this skill is shaped the way it is.

## Step 0 — find out where they are

Do not assume. Ask, or look:

| They have | Start at |
|-|-|
| tool schemas in any shape, no catalog | [Phase 1](#phase-1--catalog-and-environment) |
| a catalog already in VULCAN's format | [Phase 1](#phase-1--catalog-and-environment), skipping the conversion |
| a finished environment (`outputs/<env>/<cat>/debug_environment/`) | [Phase 2](#phase-2--the-tool-dependency-graph) after a review |
| a graph and sequences | [Phase 3](#phase-3--training-data) |
| a run that went wrong | diagnose first: read `logs/<category>/<step>.log` and check `llm_calls=` |

Check what exists rather than trusting a description:

```bash
ls configs/ examples/
ls outputs/*/*/ 2>/dev/null | head -30
```

Confirm before anything else: **which provider and model**, and that the key is set. Every
stage calls a model, and a missing key produces a run that reports success with
`llm_calls=0` at every stage rather than failing.

```bash
python -c "import os; print({k: bool(os.getenv(k)) for k in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY')})"
```

## Phase 1 — catalog and environment

Invoke the **`simulate-environment`** skill. It covers three things: converting whatever
schemas they have into a valid catalog, running steps 1 to 6, and reviewing the result.

Do not skip the review. `environment_ready` is set by tests a model wrote from the same
specs that produced the class, so it agrees with the class's own mistakes. The review is
what catches a tool that never writes the state it claims to change, a seed state no tool
can find anything in, or an argument the implementation ignores.

**Stop here and report** if the review finds anything material. Say what it costs
downstream and offer the cheapest fix — `generate_states` to reseed is far cheaper than
regenerating tools. Let them decide before you spend graph money.

## Phase 2 — the tool dependency graph

Invoke the **`build-tool-graph`** skill.

**Before running it, state the cost and get agreement.** Count the tools, multiply, and say
the number out loud:

```bash
python -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n=sum(len(r.get('api_list',[])) for r in rows)
print(f'{n} tools -> {n*(n-1)} model calls for build_graph alone')
" outputs/<env>/<category>/env_grouped/success.jsonl
```

This is the point of no return for the budget. If the number is larger than they expect,
offer to cut the catalog or point `steps.build_graph.model` at a cheaper model before
proceeding. Do not run it and apologise afterwards.

## Phase 3 — training data

Invoke the **`generate-training-data`** skill: states, argument samples, chain execution,
queries, trajectories, judging, export.

`generate_trajectories` is the second-largest line item — one conversation per sequence,
several turns each, plus a simulated user and a judge for the `with_user*` types. Say so
before starting it.

## Step 4 — report

One funnel, so attrition is visible where it happens:

```
tools → graph edges → sequences → chains executed → queries → trajectories → judged → exported
```

Then: where the dataset is, how many records, and what it cost. Take the cost from
`llm_calls=` in the stage logs, not from your estimate.

Flag anything that looks wrong rather than presenting a number that happens to be non-zero.
The failure that matters most is a stage reporting success with `llm_calls=0` — steps catch
per-item errors and emit the row anyway, so rows can exist that are just input passed
through untouched.

## Rules

- **Never start a phase that costs money without saying what it will cost.** Estimate first,
  in the message before you run it, not in the same message.
- **Never skip the environment review to save time.** It is the cheapest step and it gates
  the most expensive one.
- **Report `llm_calls=`, not row counts,** when asked whether a stage worked.
- **Re-running is cheap; regenerating is not.** Runs resume on each item's input hash, so a
  re-run skips completed work. Changing a model or temperature does *not* invalidate it —
  only deleting a stage's output directory does.
- **Do not run `score_sequences`.** It is deprecated, analysis only, and nothing downstream
  reads it. A bare `--pipeline` includes it; the ranges in these skills do not.
- **VULCAN executes model-written Python with no sandbox.** If the user is on a machine
  that holds anything they care about, say so before the first run rather than after.
