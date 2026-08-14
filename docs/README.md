# VULCAN — extended reference

Start with [`../README.md`](../README.md). These pages go deeper where it stays brief.

| Page | Covers |
|---|---|
| [`providers.md`](providers.md) | GPT / Claude / Gemini / open-weight setup, capability differences, errors |
| [`input_format.md`](input_format.md) | the input catalog schema, field by field |
| [`pipeline_stages.md`](pipeline_stages.md) | every stage: input, process, output fields, config, gotchas |
| [`configuration.md`](configuration.md) | every config key and how the two YAML layers merge |
| [`core_components.md`](core_components.md) | `PipelineRunner`, `StageRunner`, `LLMClient`, `Agent`, path tables |
| [`environment_types.md`](environment_types.md) | how `environment_type` changes each stage's behaviour |
| [`adding_new_steps.md`](adding_new_steps.md) | writing and registering a 16th step |
| [`../.claude/skills/`](../.claude/skills/) | four agent skills: `run-vulcan`, `simulate-environment`, `build-tool-graph`, `generate-training-data` |

Runnable input catalogs, with notes on what each one exercises:
[`../examples/README.md`](../examples/README.md).

Entry points are `vulcan …` (console script) and `python -m vulcan.cli …`. There are no
run scripts.

## Things that surprise people

Collected from real runs. Each is verified against the source.

**Your schemas, not VULCAN's, go into the training data.** `normalize_specs` may reword a
`description` and add a missing `response` schema, writing the result to `normalized_schema`.
That field exists for `build_graph`, which needs a `response` to detect an output→input
dependency at all. It is deliberately **not** used by `generate_queries`,
`generate_trajectories`, or the reconstruction in `export_dataset` — those read the catalog
entry you supplied, because records meant for training on a tool set have to carry *your*
tool definitions. A model trained on a reworded schema learns to call a tool whose
description will not match the one it is given at inference. The tool `name` is never
changed, so the record's `<tools>` block always agrees with the calls in the conversation.

**VULCAN executes model-generated Python in the host process, with no sandbox.** That is
how environment simulation works: `generate_tools` asks a model for a `Tool` class and
`debug_environment` runs it. Prompt constraints and behavioural tests are correctness measures,
not a security boundary. Run it somewhere disposable — see the Security section of
[`../README.md`](../README.md).

**A stage can report success while every model call failed.** Most steps end with a broad
`except Exception`, which catches `LLMError` too, then return the item without the model's
contribution. A step with no entry in `validators/deterministic.py` counts that as a pass,
so a dead key can yield `success=N, failed=0` and N useless rows. The honest signal is
`llm_calls=` in `<output_base>/logs/<category>/<step>.log`.

**`defaults.model` overrides `llm.model` for every step.** A step resolves its model as
`steps.<step>.model` → `defaults.model` → `""`, and `""` means "whatever model this stage's
client was built with". No shipped config sets `defaults.model`, and neither should you: it
applies to every stage regardless of which provider serves that stage, so in a
mixed-provider run it hands one vendor's model ID to another. Route a single stage with
`steps.<step>.model` instead.

**A per-step `provider:` is only checked when its stage starts.** The CLI builds each
stage's client with `create_client(config, stage)`, so `steps.<step>.{provider, base_url,
reasoning, capabilities, extra_body}` all take effect — but the fail-fast check at startup
covers only the global `llm:` block. A judge configured with `provider: anthropic` and no
`ANTHROPIC_API_KEY` (or without `pip install -e ".[anthropic]"`) raises `LLMError` when
`judge_trajectories` begins, after every earlier stage has already spent its calls.

**`enable_multipath` is a different graph shape, not more of the same.** A multipath
subgraph is a fork-merge (diamond) region where two non-adjacent tools are joined by
several distinct indirect paths, which yields one query that admits several valid tool
orderings. Records are stamped `type: "multipath"` and carry `multipath_pairs` /
`multipath_strength` instead of `sequence`. It is off by default because path enumeration
is slow on dense graphs.

**`env_grouped/success.jsonl` is not a step.** `StageRunner._build_env_grouped` writes it
after `verify_tools`; `assemble_environment` and `build_graph` both read it. It writes every
per-API key unconditionally via `.get()`, so `normalized_function: null` is normal and a
key-presence test on those rows always matches — select on the value. If no API survived
with a `tool_code`, the file is skipped and both stages fail with `input not found`.

**`build_graph` fails loudly on an un-normalized spec.** It reads each API's
`normalized_schema` — which `normalize_specs` writes for every API — and raises `StepError`
when it is missing or null, rather than falling back to the raw catalog `function`. Detecting
an `explicit` (output→input) edge requires the `response` schema, and the catalog spec has one
only if the author wrote it, so a fallback would quietly turn the category into ordering-only
`implicit` edges. The error names the API; rerun `normalize_specs` for that category.

**`score_sequences` produces nothing anyone consumes.** No stage lists it as an input, so its
`judgment` field never reaches the dataset. It is deprecated, analysis only, and the one stage
you can drop from a run with no effect on the dataset. It reads each record shape correctly —
multipath records are matched on `type` and read from their top-level `nodes` / `edges`, rather
than falling through to the single-turn branch — but that only means it wastes model calls
accurately.

**`--output-label` only names the folder.** Generation is controlled by
`steps.generate_trajectories.tool_call_format`, and the flag defaults to it, so a folder
describes its contents unless you override it — pass `--output-label native` with
`tool_call_format: tagged` and tagged data lands in a folder called `native`.

**The judge writes a nested dict.** `judge_trajectories` produces `judgment.score` and
`judgment.task_completed` — not top-level fields. `steps/output/select.py` reads that same
nested path.

**Runs resume on input hash.** `_item_id` is an MD5 of the item's *input*, so a completed
item is skipped on re-run; changing a model, temperature or prompt does **not** invalidate
it. Delete the stage output to force regeneration.
