---
name: generate-training-data
description: Run VULCAN's task-generation and trajectory phases — initial states, argument samples, chain execution, user queries, trajectories, judging, and dataset export — and verify every stage wrote to its documented path with the documented record shape. Use after the tool dependency graph exists.
---

# Generate the training set

Everything from initial states through the exported dataset. The pipeline's own ordering
in `core/stage_runner.py` is authoritative; this skill runs it in that order and checks
each stage landed where the docs say it does.

## Prerequisites

`execute_sequences` needs **both** halves of the pipeline finished for the category:

- `sample_arguments/success.jsonl` — argument samples per tool
- `plan_sequences/multi_turn/all_sequences.jsonl` — the planned sequences

`generate_states` and `sample_arguments` are steps 6 and 7, *before* `build_graph` at 8, so
a run that went straight to the graph has them already. If not, run them first — they are
in the order below.

If the graph does not exist yet, use the `build-tool-graph` skill. If the environment has
not been reviewed, use `simulate-environment` first; every stage below executes against it,
and a broken environment produces plausible-looking records that teach the wrong thing.

## The stages, in order

| # | Stage | Model calls | Writes |
|---:|---|:---:|---|
| 6 | `generate_states` | yes | `<category>/generate_states/success.jsonl` |
| 7 | `sample_arguments` | yes | `<category>/sample_arguments/success.jsonl` |
| 11 | `execute_sequences` | **no** | `<category>/execute_sequences/pre_query_all_modes.jsonl` |
| 12 | `generate_queries` | yes | `<category>/generate_queries/success.jsonl` |
| 13 | `generate_trajectories` | yes | `<category>/iter_<k>/trajectories/<label>/success.jsonl` |
| 14 | `judge_trajectories` | yes | `<category>/iter_<k>/judged/<label>/success.jsonl` |
| 15 | `export_dataset` | **no** | `training_dataset/{json,xml}/<category>_iter_<k>.jsonl` |

```bash
vulcan --env <config> --pipeline --start generate_states --end sample_arguments
# graph phase happens here — see the build-tool-graph skill
vulcan --env <config> --pipeline --start execute_sequences --end export_dataset
```

`generate_trajectories` is the most expensive stage after `build_graph`: one conversation
per sequence, many turns each, plus a simulated user and a judge for the `with_user*`
environment types.

## Output names are a contract

Three stages break the usual `<step>/success.jsonl` pattern. This is the single most
common reason someone concludes a stage produced nothing:

```
<category>/plan_sequences/multi_turn/all_sequences.jsonl     not success.jsonl
<category>/execute_sequences/pre_query_all_modes.jsonl        not success.jsonl
<category>/iter_<k>/trajectories/<label>/success.jsonl        nested under iter_ and label
<category>/iter_<k>/judged/<label>/success.jsonl              nested under iter_ and label
```

`<k>` is the CLI's `--iter` (default 0) and `<label>` is `--output-label` (default `native`).
Rejected transcripts are written to `<category>/iter_<k+1>/input.jsonl` as input for the
next iteration — that is the loop, not a failure.

The exported dataset is the one thing that lives at the run root rather than per category:

```
<output_base>/training_dataset/json/<category>_iter_<k>.jsonl
<output_base>/training_dataset/xml/<category>_iter_<k>.jsonl
```

Record shape is `{id, category_name, messages}`, except `tagged`-format records, which use
`category` rather than `category_name`.

## Verify

Check each path in the table above exists and holds rows, and read `llm_calls=` from each
stage log. The script does both across every category at once:

```bash
python .claude/skills/generate-training-data/scripts/layout_check.py --env <config>
```

Add the script's `--iteration <k>` and `--label <label>` if you did not use the defaults. Exit 0 means
every documented output is present; 1 means gaps, listed.

It prints rows and `llm_calls` per stage. **Read the `llm_calls` column.** Steps catch
per-item errors and emit the row anyway, so a stage can report success with every model
call having failed — `llm_calls=0` on a stage that should make calls means exactly that,
and the rows below it are input passed through untouched.

## Reading the results

| Symptom | Cause | Fix |
|---|---|---|
| `llm_calls=0` on a model stage | every call failed — usually auth, a bad model id, or a provider outage | check the stage log for the error; `LLMError` names the provider |
| stage present but 0 rows | every item failed validation | read `failed.jsonl` beside it — the validator's reason is on each row |
| `execute_sequences` few rows | chains failed against the live environment | expected attrition; if it is most of them, the environment is the problem — run `simulate-environment` |
| `generate_queries` empty | `execute_sequences` produced nothing | fix that stage first; this one has no input |
| `export_dataset` empty, trajectories fine | everything scored below `min_score` (default 8) or `task_completed` was false | check the judge's scores in `judged/<label>/success.jsonl`; lower `min_score` only if the transcripts really are acceptable |
| unexpected keys in a record | reconstruction wrote a shape the docs do not describe | report it — the record shape is a contract with whatever trains on it |

## Report

Give the row count at each stage as a funnel — sequences in, chains executed, queries,
trajectories, judged, exported — so attrition is visible where it happens. Then state
where the dataset is and how many records it holds.

Flag any stage whose `llm_calls` is 0 or whose output is missing, and name the stage to
re-run. Runs resume on the item's input hash, so re-running a stage does not re-pay for
items that already succeeded; to force regeneration, delete that stage's output directory.

## Notes

- `execute_sequences` and `export_dataset` make no model calls. They are free to re-run.
- Rejected transcripts accumulate at `iter_<k+1>/input.jsonl`. Running the trajectory
  stages again with `--iter <k+1>` works that backlog rather than starting over.
- For a `with_user_and_policy` category the exported record carries the category's
  `policy_rules` in its system prompt. If it does not, that is a defect — the agent was
  driven under the policy and judged against it, so the record has to contain it.
