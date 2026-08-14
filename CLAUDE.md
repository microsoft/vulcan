# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

VULCAN generates multi-turn tool-calling training data from API specifications. It
simulates the software a tool catalog describes **entirely in memory**, builds a dependency
graph over the tools, drives conversations against that live environment, judges them, and
emits training records.

Public release of a codebase developed internally against a single hosted model endpoint,
accompanying *VULCAN: Where Agents Learn by Living in Simulated Tool Environments*
(DATA-FM @ ICLR 2026, <https://openreview.net/forum?id=Ht4HB4pIQq>). The paper's experiments
used internal endpoints and an internal catalog; neither ships here, and there is no
evaluation or fine-tuning code. **Do not add claims about benchmark results to the docs.**

Write the name as **VULCAN** in prose and **`vulcan`** for the package, CLI, config keys and
paths. `Vulcan` is wrong. `VulcanError` is an identifier and stays as it is.

## Commands

```bash
pip install -e ".[openai]"        # or [anthropic], [gemini], [all], [dev]

vulcan --list-steps
vulcan --env quickstart --pipeline                        # full run, bundled data
vulcan --env quickstart --stage generate_tools            # one step
vulcan --env quickstart --pipeline --start generate_trajectories --end judge_trajectories

pytest                            # 596 tests, ~1s, no network, no API key
pytest tests/test_llm.py -q       # provider layer only
```

`vulcan` and `python -m vulcan.cli` are equivalent. There are no shell run-scripts.

## Layout

```
vulcan/llm/          provider adapters behind one interface
vulcan/core/         runner, stage runner, agent, code exec, parsers
vulcan/steps/        the 15 pipeline steps, grouped by phase
vulcan/prompts/      one module per step
vulcan/config/       packaged base.yaml and example configs
vulcan/environment/  EnvironmentConfig, environment types
vulcan/validators/   deterministic per-step output validation
configs/             user configs: quickstart, hotel_policy
examples/            three input catalogs, 31 tools
docs/                reference pages, verified against the code
.claude/skills/      four agent skills for driving a run
tests/               596 tests
```

## Architecture

Two facts explain most of the codebase.

**1. Every LLM call goes through one method.** `LLMClient.call()` in `vulcan/llm/base.py`
returns a normalised **OpenAI-shaped** response (`choices[0].message.{content,
reasoning_content, tool_calls}`, `usage.{prompt,completion,total}_tokens`) whatever the
provider. Four adapters translate to and from that shape: `openai_client.py`,
`anthropic_client.py`, `gemini_client.py`, `openai_compatible.py` (vLLM, SGLang, Ollama,
llama.cpp, TGI — how open-weight models are served).

- **No pipeline step may branch on provider.** Provider-specific behaviour belongs in an
  adapter or in `capabilities.py`, never in `steps/`.
- Tool calls are always normalised to OpenAI form with `arguments` as a **JSON string**,
  even for providers that return objects, and ids must be non-empty — `reconstruct.py` maps
  `tool_call_id` back to a tool name when building records.
- `capabilities.py` decides per model whether to send `max_completion_tokens`, whether
  `temperature` is accepted, whether reasoning text comes back, and (Anthropic) adaptive vs
  fixed-budget thinking. Add a model family there, not at a call site.

**2. Fifteen registered steps, dispatched by a runner that owns ordering.** Steps subclass
`BaseStep` (`vulcan/steps/base.py`), register with `@register_step`, and implement
`async run(item, progress)`. `core/stage_runner.py` holds `PIPELINE_ORDER` and
`_STANDARD_INPUT` — **that is the authoritative input resolution**. Steps do not describe
their own predecessors.

```
1  normalize_specs        env_simulation     spec repair, missing response schemas
2  generate_tools         env_simulation     one Python Tool class per API
3  verify_tools           env_simulation     verify → fix → re-verify
4  assemble_environment   env_simulation     merge into one stateful class
5  debug_environment      env_simulation     generate tests, run, repair
6  generate_states        env_simulation     initial states; the only data diversity
7  sample_arguments       env_simulation     per-tool argument bank
8  build_graph            graph_construction explicit/implicit edges over tool pairs
9  plan_sequences         graph_construction subgraphs → sequences (deterministic)
10 score_sequences        graph_construction deprecated, analysis only
11 execute_sequences      task_generation    run chains against the environment
12 generate_queries       task_generation    chains → natural-language queries
13 generate_trajectories  trajectory         conversation against the live environment
14 judge_trajectories     trajectory         model-as-judge
15 export_dataset         output             filter and format
```

`env_grouped/success.jsonl` is written by `StageRunner._build_env_grouped` after
`verify_tools` and is **not a registered step**. Both `assemble_environment` and
`build_graph` read it.

Four private modules under `steps/` are live and imported by `plan_sequences` and
`generate_trajectories`: `_subgraph_extractor`, `_sequence_builder`, `_multipath_extractor`,
`_tool_call_parser`. There are no others; six unreachable ones were deleted.

## Configuration

Two layers, deep-merged: packaged `vulcan/config/base.yaml`, then the per-environment file.
Precedence `steps.<step>` > `defaults` > `base.yaml`. `load_config` resolves a name as an
explicit path → `./configs/<name>.yaml` → the packaged config dir.

- **Never set `defaults.model`.** A step resolves `steps.<step>.model` → `defaults.model` →
  `""`, and empty means "use the model this stage's client was built with". Setting it
  overrides `llm.model` for every stage, which in a mixed-provider config sends one vendor's
  model name to another. Same reason no step may hardcode a fallback model id.
- The CLI builds a client **per stage** via `create_client(config, stage)`, so per-step
  `provider` / `base_url` / `reasoning` take effect. Clients cache on
  `(provider, model, base_url, reasoning)`.
- API keys come from the environment or `.env`, never from a config file.

## Skills

Four skills in `.claude/skills/`, committed so a clone is the install. Each is a written
procedure; the Python helpers beside them are optional cross-checks that settle mechanical
questions exactly, and every skill works without them.

| Skill | Covers |
|-|-|
| `run-vulcan` | the whole pipeline from a conversation; calls the other three, confirms before spending |
| `simulate-environment` | schemas → catalog → steps 1 to 6, then reviews the generated environment |
| `build-tool-graph` | cost estimate, `build_graph` + `plan_sequences`, usability check |
| `generate-training-data` | trajectories through export, verifying each stage's output |

When editing them: the **model** does the judging, the script only reports facts. Do not
rewrite a skill so its value is "run this script". `.github/prompts/` mirrors them for
Copilot by pointing at the same files, so keep it thin.

## Security model

**`core/code_exec.py` executes model-generated Python in the host process with no sandbox.**
That is how environment simulation works, not an oversight: the pipeline asks a model to
write a `Tool` class and then runs it. Prompt constraints and `debug_environment` are
correctness measures, not a security boundary.

Do not "fix" this with a partial sandbox that implies a guarantee it cannot make. Keep the
README warning and the module docstring in sync.

## Conventions

- **A stage can report success while every model call failed.** Steps catch per-item errors
  and emit the row anyway. The honest signal is `llm_calls=` in
  `<output_base>/logs/<category>/<step>.log`, not row counts.
- **Runs resume on the item's *input* hash**, so changing a model or temperature does not
  invalidate completed items. Delete the stage output directory to force regeneration.
- **`build_graph` costs *n·(n−1)* model calls for *n* tools.** Usually the largest line item
  in a run; be deliberate before raising tool counts in examples or tests.
- **Docs under `docs/` are verified against the code**, not written from memory. Sharp edges
  live with the stage that has them under a ⚠ marker in `pipeline_stages.md`, and
  cross-cutting ones in `docs/README.md`. Fixing one means deleting the entry, not softening
  it; deciding the behaviour is correct means recording *why*, so it is not re-filed.
- **No hardcoded domain or benchmark names.** Nothing may branch on a category name, a tool
  name, or a benchmark. `tests/test_no_domain_carveouts.py` fails if one reappears anywhere
  in the package, including in comments.

## Renaming, and why it is dangerous here

Names are expected to say what the thing does — the release renamed all 15 steps to
`verb_noun` and config keys to state their unit (`k_values` → `turns_per_sequence`). But
blanket renames have broken this repo repeatedly, always silently:

- **No step may start with `test_`.** pytest collects anything matching that name, which is
  why the environment-repair step is `debug_environment`.
- **Never rename a provider's wire format.** `thinking`, `reasoning_content`,
  `redacted_thinking`, `signature` and `adaptive` in `vulcan/llm/` are Anthropic's, vLLM's
  and Ollama's names, not ours. A repo-wide rename of VULCAN's own `thinking` vocabulary
  once rewrote them, silently disabling extended thinking on Claude and reasoning on Ollama,
  with the whole suite green. `TestProviderWireFormat` in `tests/test_llm.py` asserts them.
- **Identifiers compound.** `test_environment_type_is_*` is not a reference to a step, and
  `VulcanError` is not the product name. A bad substitution can *un-collect* a test rather
  than fail it.
- `tests/test_shipped_configs.py` exists because the unit suite once passed while every real
  run raised `NameError`.
- After any rename: diff against the previous tree and read every changed line in
  `vulcan/llm/`. Grepping for the new name proves nothing about what the old one was
  attached to.
