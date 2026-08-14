# Configuration

Every run is one merged dict. VULCAN always loads the packaged
[`vulcan/config/base.yaml`](../vulcan/config/base.yaml), then deep-merges your
per-environment YAML on top of it. A per-environment file only needs the keys it
changes — everything else falls back to `base.yaml`, and `base.yaml` is the
authoritative list of shipped defaults.

Precedence, highest first:

```
steps.<step>.<key>   →   defaults.<key>   →   base.yaml
```

⚠ **The `defaults.<key>` rung only exists for seven keys.** `BaseStep` gives an explicit
`defaults` fallback to `model`, `temperature`, `max_tokens`, `thinking_model` and
`empty_response_retries`; `StageRunner` reads `parallel_workers` and `max_retries` from
`defaults` directly. Every *other* step key — `num_states`, `max_fix_iterations`,
`sequences_per_category`, `min_score`, `max_turns` and the rest — is read straight off
`config["steps"][<step>]` with a hardcoded literal as its only fallback, so putting one
under `defaults:` has **no effect at all**. Set those under `steps.<step>:`. Because
`base.yaml` already ships them under `steps.<step>:`, the difference only shows up when you
try to override one via `defaults:`. See [`steps:`](#steps).

The `llm:` block sits beside that chain rather than inside it; see
[The `llm:` block](#the-llm-block).

---

## Where configs live

`load_config(name)` (in [`vulcan/config/__init__.py`](../vulcan/config/__init__.py))
resolves `name` against three candidates, in order, and takes the first that exists:

1. **an explicit path** — if `name` itself points at an existing file
   (`vulcan --env ./my/run.yaml`, absolute or relative, `~` expanded);
2. **`./configs/<name>.yaml`** (or `.yml`) relative to the **current working
   directory** — this is how a `pip install`ed VULCAN picks up configs you wrote;
3. **`<package>/config/<name>.yaml`** — the configs shipped inside the wheel.

When nothing resolves you get a `FileNotFoundError` listing every path that was
tried plus the environment names that *are* resolvable right now.

| Name | File | What it is |
|---|---|---|
| `quickstart` | [`configs/quickstart.yaml`](../configs/quickstart.yaml) | full end-to-end run over the 8-tool ticket catalog, sized to finish cheaply |
| `hotel_policy` | [`configs/hotel_policy.yaml`](../configs/hotel_policy.yaml) | policy-driven domain with a simulated user |
| `mock_test` | `vulcan/config/mock_test.yaml` (packaged) | smoke test; same catalog as quickstart, loadable from any working directory |

`base.yaml` and `example_environment.yaml` are hidden from the "Available configs"
list that a failed lookup prints (`_NON_ENVIRONMENT_FILES`), because the first is
always loaded anyway and the second is a template. Neither is *blocked*, though —
`load_config` will happily resolve either name, and `--env example_environment` runs a
template whose `categories` list names `hotel_booking`, which its `input.catalog`
(`examples/ticket_api.jsonl`) does not contain. Copy
[`vulcan/config/example_environment.yaml`](../vulcan/config/example_environment.yaml)
to `./configs/<name>.yaml` to start a new environment.

Entry points are the console script and the module — there are no run scripts:

```bash
vulcan --env quickstart --pipeline
python -m vulcan.cli --env quickstart --stage generate_trajectories
```

---

## How the merge works

`_deep_merge(base, override)` recurses into dicts and **replaces everything else**.
That includes lists:

```yaml
# base.yaml
verification:
  enabled: false
  verify_steps: [generate_tools, verify_tools, generate_trajectories, judge_trajectories]

# yours
verification:
  enabled: true                 # merged into the same dict
  verify_steps: [judge_trajectories]   # REPLACES the list — it does not append
```

`vulcan/config/__init__.py` exposes `get_step_config(config, step_name)` →
`_deep_merge(defaults, steps[step_name])`, which merges the two layers the way the
precedence chain above describes. **No pipeline code calls it** — it is a helper for
callers and tests. `BaseStep` instead takes `config["steps"][<step>]` unmerged and applies
the `defaults` fallback to five keys by hand (see the warning at the top of this page).
Two further consequences worth knowing:

- **`get_step_config` does not include the `llm:` block.** Callers read
  `config["llm"]` separately.
- **A step key you invent is silently carried through.** Nothing validates the
  `steps:` tree, so a typo (`max_turn` for `max_turns`) is not an error — the step
  simply keeps its default.

### Path resolution

After merging, `_resolve_paths` rewrites exactly two things: `output_base`, and
every string value under `input`. Each gets `~` expanded and, if relative,
resolved against the current working directory when something exists there,
otherwise against the package root (the repo root in a source checkout). A path
that does not exist yet — an output directory — lands under the working directory.

```yaml
input:
  catalog: examples/ticket_api.jsonl    # → <cwd>/examples/... or <repo>/examples/...
output_base: ./outputs/quickstart       # → <cwd>/outputs/quickstart
```

Every *other* path in the config is used verbatim, resolved by whatever code reads
it — `generate_queries.few_shot.examples_path` is opened relative to the working
directory, with no package-root fallback.

---

## Top-level keys

| Key | Default | Effect |
|---|---|---|
| `environment` | the `--env` name | run name, used for log naming. `get_environment` injects the `--env` value only when **neither** `environment` nor `domain` is present |
| `environment_type` | `tools_only` (via inference) | `tools_only` \| `with_user_and_policy` \| `with_user`. An unrecognised value raises `ValueError` listing the valid ones. See [`environment_types.md`](environment_types.md) |
| `input.catalog` | — | path to the input catalog (JSONL, one category per line). See [`../examples/README.md`](../examples/README.md) |
| `output_base` | required in practice | root for every stage output. `StageRunner` reads `config["output_base"]` directly and raises `KeyError` without it (only `BaseStep` falls back to `<cwd>/outputs`) |
| `categories` | every category in the catalog | explicit list of category names to process |
| `terminate_tools` | `["transfer_to_human_agents"]` **when the key is absent** | tool names whose call ends a trajectory |
| `environment_type_map` | `{}` | per-category override: `{category: tools_only\|with_user_and_policy\|with_user}` |
| `has_policy`, `has_user_agent` | `false` | legacy booleans, consulted **only** when `environment_type` is absent |
| `has_constraints` | `false` | read directly; also flipped on automatically when the catalog carries per-API `constraints` |

Two traps in that table:

- **`terminate_tools` defaults to a tool that is not in any input catalog.** Omit the
  key and the terminator is `transfer_to_human_agents`, which `generate_trajectories` appends to
  the tool list only on the `with_user_and_policy` path. Every shipped config sets
  `terminate_tools: []` so the behaviour is stated rather than inherited.
- **`categories` is iterated, not parsed.** Setting `categories: all` makes the
  runner iterate the string and look for categories named `a`, `l`, `l`. To process
  everything, omit the key or leave the list empty.

Category-level data can override the environment-level type: a category whose
catalog record carries `policy_rules` is promoted to `with_user_and_policy` even in a
`tools_only` environment, and a record-level `environment_type` wins over both.

---

## The `llm:` block

One block selects the provider for the whole run, and any step may override it
([Per-step provider and model overrides](#per-step-provider-and-model-overrides)). The
four adapters live in [`vulcan/llm/`](../vulcan/llm/) and all return the same normalised,
OpenAI-shaped response, so no step branches on which model produced it.

```yaml
llm:
  provider: openai              # openai | anthropic | gemini | openai_compatible
  model: gpt-4o
  # base_url: http://localhost:8000/v1
  request_timeout: 600
  max_retries: 3
  # reasoning: {effort: medium}
  # capabilities: {}
  # extra_body: {}
```

| Key | Default | Effect |
|---|---|---|
| `provider` | `openai` | picks the adapter. Anything outside the four names raises `LLMError` listing them |
| `model` | `gpt-4o` | model ID sent to the provider. Falls back to `defaults.model` when unset; if both are empty, `create_client` raises `LLMError` |
| `base_url` | provider default | **required** for `openai_compatible`; also the way to point `openai`/`anthropic` at a proxy |
| `api_key` | from the environment | `${VAR}` is expanded at load; an unresolved `${...}` becomes empty. Prefer leaving it out |
| `request_timeout` | `600` | seconds for one attempt, end to end (`asyncio.wait_for`) |
| `max_retries` | `3` | attempts per request inside the client, with exponential backoff + jitter |
| `capabilities` | `{}` | pins the per-model feature detection; see below |
| `extra_body` | `{}` | merged into the outgoing provider request verbatim, last — it can overwrite anything VULCAN set |
| `reasoning` | `null` | provider-neutral reasoning request, applied to every call the client makes; see below |

Switching families is a config change and nothing else — pick one:

```yaml
llm: {provider: openai,            model: gpt-4o}
llm: {provider: anthropic,         model: claude-sonnet-5}
llm: {provider: gemini,            model: gemini-2.5-pro}
llm: {provider: openai_compatible, model: Qwen/Qwen3-32B, base_url: http://localhost:8000/v1}
```

### Credentials

Keys are read from the environment, or from a `.env` file in the working directory
loaded at CLI start (`load_dotenv`; existing environment variables win, so an
`export` always beats the file). Copy [`../.env.example`](../.env.example).

| `provider` | API key env var | Base URL env var |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| `gemini` | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` | — |
| `openai_compatible` | `VULCAN_API_KEY`, then `OPENAI_API_KEY` | `VULCAN_BASE_URL`, then `OPENAI_BASE_URL` |

`openai_compatible` is the only provider that starts without a key: the other three
raise `LLMError` naming the variable to set. Never put a real key in a config file —
`tests/test_config.py` asserts the shipped config carries no `api_key`.

### Per-step provider and model overrides

`resolve_llm_config(config, step_name)` overlays
`steps.<step>.{provider, model, base_url, reasoning, capabilities, extra_body}`
on the global block, and `vulcan.cli` builds every stage's client with
`create_client(config, stage)` — so all six keys take effect in a normal `vulcan`
run. Clients are cached per distinct `(provider, model, base_url, reasoning)`, so a
run opens one connection pool per backend rather than one per stage.

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-32B

steps:
  generate_trajectories:
    tool_call_format: tagged    # bulk generation stays on the local server
  judge_trajectories:
    provider: anthropic         # the judge gates the dataset — spend on it
    model: claude-sonnet-5
```

Three things to know before splitting a run this way:

- **Every provider you name needs its own key and its own extra.** A step-level provider
  picks its key up from that provider's environment variables (table above) and imports
  that provider's optional dependency — `pip install -e ".[anthropic]"` for the example.
- **`api_key` is not a per-step key, and it is inherited.** `steps.<step>` overrides only
  the six keys listed above, so an `llm.api_key` written in the config is handed to *every*
  stage's client, including one that switched provider. When mixing providers, leave
  `llm.api_key` out and let each provider read its own variable.
- **A per-step provider is only validated when its stage starts.** The CLI's fail-fast
  check at startup builds a client from the global block alone; a missing key or missing
  package for a step-level provider raises `LLMError` when that stage begins, after the
  earlier stages have already spent their calls.

### `capabilities:`

Detection in [`vulcan/llm/capabilities.py`](../vulcan/llm/capabilities.py) is
pattern-based on the model name and is a *default*, never a rule. Pin any field
when you serve a model whose name matches nothing — a private fine-tune, for example:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: my-org/my-finetune
  capabilities:
    supports_reasoning: true      # server started with --reasoning-parser
    supports_tools: false         # server started without tool-call parsing
```

| Field | Default | Meaning |
|---|---|---|
| `uses_max_completion_tokens` | `false` | send `max_completion_tokens` instead of `max_tokens` |
| `supports_temperature` | `true` | model accepts a custom temperature; the layer drops it rather than failing when false |
| `supports_reasoning` | `false` | model can be asked to reason |
| `returns_reasoning_text` | `false` | model returns reasoning *text*, not just billed tokens |
| `supports_tools` | `true` | native function calling |
| `supports_stop` | `true` | stop sequences |
| `uses_adaptive_thinking` | `false` | Anthropic only: adaptive thinking rather than a fixed token budget |

### `reasoning:`

`llm.reasoning` (or `steps.<step>.reasoning`) is resolved by `resolve_llm_config`,
passed to the client by `create_client`, and stored as the client's
`default_reasoning`. `LLMClient.call` then applies it to every request that does not
ask for something else. Precedence, highest first:

1. a `reasoning=` argument on the call itself (no shipped step passes one);
2. the client's `default_reasoning` — i.e. `llm.reasoning` / `steps.<step>.reasoning`;
3. `{"effort": "medium"}`, when the step has `thinking_model: true` and nothing above
   applies.

```yaml
llm:
  reasoning: {effort: medium}          # low | medium | high
  # reasoning: {budget_tokens: 4096}   # explicit budget, where supported
```

Reasoning is part of the client cache key, so two stages on the same model with
different `reasoning` get separate clients rather than silently sharing one. Which
adapters act on it: `openai` maps it to `reasoning_effort`, `anthropic` to the
`thinking` block for that model generation, `gemini` to a thinking budget — each only
when the model's `supports_reasoning` capability is true. **`openai_compatible` never
sends it**: reasoning there is a server-side launch flag (`--reasoning-parser`), and
per-request control goes through `extra_body`. See
[`providers.md`](providers.md) for which models return reasoning *text* at all —
public OpenAI reasoning models bill for reasoning tokens and return none, so
`generate_trajectories.tool_call_format: reasoning` needs Claude, Gemini, or a
self-hosted reasoning model.

---

## `defaults:`

Inherited by every step unless the step overrides the same key.

| Key | Shipped default | Read by |
|---|---|---|
| `model` | **unset** | every step, as its per-call model. Deliberately not shipped — see below |
| `temperature` | `0` | every step |
| `max_tokens` | `16000` | every step (output cap, per request) |
| `parallel_workers` | `20` | `StageRunner` — concurrent asyncio workers per stage. `--workers N` overrides it |
| `max_retries` | `10` | `PipelineRunner` — how many times a **failed item** is re-processed. Distinct from `llm.max_retries`, which retries a single HTTP request |
| `empty_response_retries` | `10` | `Agent` — re-asks when the model returns neither text nor a tool call |
| `thinking_model` | `false` | requests reasoning where supported; also decides the default `--output-label` folder name |

**Do not set `defaults.model`.** A step resolves `steps.<step>.model` →
`defaults.model` → `""`, and `""` means "send no model override — use the one this
stage's client was built with", which is `llm.model`. Setting `defaults.model` makes
every stage override that, whatever provider serves it, so in a mixed-provider config
it hands one vendor's model ID to another. No shipped config sets it; to move a single
stage to a different model, set `steps.<step>.model`.

Reasoning models (OpenAI o-series and gpt-5, current Claude, Gemini 2.5) ignore or
reject `temperature`. The provider layer drops it for those models instead of
failing the request, so leaving `temperature: 0` in place is safe.

---

## `steps:`

Every key below is read from `steps.<step>`. Apart from the five `BaseStep` keys
(`model`, `temperature`, `max_tokens`, `thinking_model`, `empty_response_retries`) these do
**not** fall back to `defaults` — setting them there does nothing. Defaults
shown are what [`base.yaml`](../vulcan/config/base.yaml) ships, except where marked **†**
— those keys are absent from `base.yaml`, so the value shown is the code's own fallback.
Stage behaviour is in [`pipeline_stages.md`](pipeline_stages.md).

### Phase 1 — environment simulation

| Step | Key | Default | Effect |
|---|---|---|---|
| `generate_tools` | `temperature` / `max_tokens` | `0` / `32000` | one `Tool` class per API; code generation needs the large output cap |
| `verify_tools` | `max_fix_iterations` | `5` | verify → fix → re-verify rounds per API |
| `assemble_environment` | `max_tokens` | `65000` | merges every API class into one env class |
| `debug_environment` | `max_fix_iterations` | `15` | test → fix rounds; the dominant Phase-1 cost |
| | `max_test_generation_retries` | `5` | retries when the generated *test* itself crashes |
| `generate_states` | `num_states` | `5000` | **number of initial states = number of model calls.** The only source of data diversity, and the first knob to raise for a real run |
| | `min_list_length` | `5` | minimum entries the model must put in list/array fields |
| | `temperature` | `0.8` | deliberately high — diversity across states |
| `sample_arguments` | `temperature` | `0.2` | per-state call samples; the argument bank `execute_sequences` draws from |

### Phase 2 — graph construction

| Step | Key | Default | Effect |
|---|---|---|---|
| `build_graph` | `max_tokens` | `2048` | one short structured verdict per ordered API pair. **Costs *n·(n−1)* model calls for *n* tools** — 56 at 8 tools, 156 at 13 |
| `plan_sequences` | `turns_per_sequence` | `"2,3,4"` | which turn counts to build, comma-separated string or list. `"1"` produces no multi-turn data while still writing `all_sequences.jsonl` |
| | `max_tools_per_sequence` | `5` | nodes per subgraph — the size knob |
| | `sequences_per_category` | `15000` | sequence budget, split `// len(turns_per_sequence)` across those turn counts |
| | `max_subgraphs_per_category` | `5000` † | cap before sequence building; over it, subgraphs are sampled by popularity |
| | `implicit_edge_weight` | `0.01` | popularity-sampling weight during subgraph extraction |
| | `enable_multipath` | `false` | see below |
| | `max_multipath_tools` | `15` | nodes allowed in a multipath subgraph; larger ones are dropped |
| `score_sequences` | `thinking_model` | `true` | scores sequence quality. Analysis only — nothing downstream reads its output |

**Multipath** (`enable_multipath: true`) is a second, independent extractor. It scans
every ordered pair of **non-adjacent** nodes — pairs with a direct edge are skipped —
and keeps those joined by two or more distinct directed paths: a fork-merge, or
"diamond", region of the tool graph. The union of all nodes on all those paths
becomes one subgraph, stamped `"type": "multipath"` with `multipath_pairs` and
`multipath_strength` (the highest path count in it), and merged into
`all_sequences.jsonl` alongside the sequences. `generate_queries` turns each
into a *single* query that admits several valid tool orderings, rather than one query
per turn. Path enumeration is bounded at 20 paths per pair and depth 8; those bounds
are not configurable from YAML. It is off by default because the pair scan is slow on
dense graphs.

### Phase 3 — task and trajectory generation

| Step | Key | Default | Effect |
|---|---|---|---|
| `execute_sequences` | `states_per_sequence` | `2` | how many initial states each sequence is executed against — a direct multiplier on sample count |
| | `max_samples_per_category` | `50000` | **caps every stage after it** |
| | `shuffle_seed` | `42` † | seed for the pre-cap shuffle; change it to draw a different subset |
| `generate_queries` | `max_attempts` | `5` | re-asks until the query parses; on exhaustion the record's query becomes the literal `"Error"` |
| | `few_shot.*` | disabled | see below |
| `generate_trajectories` | `tool_call_format` | unset → `reasoning` if the step's `thinking_model` (or the item's `thinking_mode`) is true, else `native` | `native` (provider tool calls) \| `reasoning` (text calls from a reasoning model) \| `tagged` (`<tool_call>`/`<answer>` text, works on any model) |
| | `max_turns` | `100` | conversation turns per trajectory |
| | `max_tokens` | `2048` | **per turn**, not per trajectory. Raise it for `reasoning` or `tagged` — `quickstart` and `mock_test` set `32000`; `hotel_policy` leaves it at the default |
| | `max_tool_calls_per_turn` | `30` | tool-call rounds inside one turn before the next query |
| | `temperature` | `0.7` | moderate creativity for assistant responses |
| | `simulated_user_model` | the step's `model` | simulated user's model (`with_user_and_policy` and `with_user` only) |
| | `simulated_user_temperature` | the step's `temperature` | keep it at `0` so policy violations come from the agent, not the user |
| | `simulated_user_max_tokens` | the step's `max_tokens` | |
| | `simulated_user_name` | `user_sim` | agent name in logs |
| `judge_trajectories` | `model` | unset → the stage client's model | the judge. Set it (or `steps.judge_trajectories.provider`) to judge on something other than what generated the data |
| | `tool_call_format` | from the record | judge prompt family; the record's own value, stamped by `generate_trajectories`, wins over the step key |
| | `max_tokens` | `16000` | |

The four `simulated_user_*` keys are read **flat, from `steps.generate_trajectories`** —
`GenerateTrajectoriesStep` reads
no nested sub-dict, so keep them at the top level of the step, which is how
[`base.yaml`](../vulcan/config/base.yaml) and
[`configs/hotel_policy.yaml`](../configs/hotel_policy.yaml) ship them. Each falls back
to the corresponding agent-level value when omitted:

```yaml
steps:
  generate_trajectories:
    tool_call_format: native
    max_turns: 40
    simulated_user_model: gpt-4o        # omit to reuse the step's own model
    simulated_user_temperature: 0       # deterministic, so violations come from the agent
    simulated_user_max_tokens: 32768
    simulated_user_name: user_sim
```

`max_tool_calls_per_turn` is flat on the step too: to run the reasoning path, set
`steps.generate_trajectories.tool_call_format: reasoning` and `max_tool_calls_per_turn`
beside it. **Prefer that over relying on `thinking_model`.** With `tool_call_format` unset,
`thinking_model: true` does resolve to `reasoning` and the reasoning path does run — but the
folder name is resolved separately by `_resolve_output_label` in
[`vulcan/cli.py`](../vulcan/cli.py), which consults only `defaults.thinking_model`. Set
`thinking_model` under `steps.generate_trajectories:` instead and generation takes the
reasoning path while `--output-label` still names the folder `native`. Setting
`tool_call_format` explicitly keeps the two in step. See the `generate_trajectories`
warnings in [`pipeline_stages.md`](pipeline_stages.md).

### Few-shot query generation

Opt-in, off by default (generation is zero-shot).

```yaml
steps:
  generate_queries:
    few_shot:
      enabled: true
      num_examples: 3               # 0 = use every matching example
      examples_path: ./fewshot      # JSON/JSONL file, or a directory of per-category files
      examples: {}                  # inline alternative; examples_path wins when it loads
```

Examples are keyed from most general to most specific, and a record **pools every
matching key** before sampling:

| Key | Matches |
|---|---|
| `"*"` | every flow |
| `tools_only`, `with_user_and_policy`, `with_user` | environment type |
| `sequence_tools_only`, `multipath_with_user_and_policy`, … | exact flow (`<sequence\|multipath>_<environment_type>`) |
| `<category_name>` | one category |

With `num_examples > 0` the pool is de-duplicated and randomly sampled, seeded by the
record's own id — reproducible across runs, different per record. A directory
`examples_path` uses each filename stem as its key. A missing `examples_path` prints
a warning and falls through to `examples`.

### Phase 4 — selection

| Step | Key | Default | Effect |
|---|---|---|---|
| `export_dataset` | `min_score` | `8` | keeps trajectories with `judgment.score >= N` |
| | `require_task_completed` | `true` | also require `judgment.task_completed` |

Both are the gate on the training set. Rejected records are written to
`iter_<k+1>/input.jsonl` for the next iteration.

---

## `verification:`

An optional pass/fail hook that runs **inside** the listed stages. It is not the
`judge_trajectories` judge and has nothing to do with the selection threshold.

```yaml
verification:
  enabled: false
  # model: gpt-4o        # optional; omit to reuse the stage's own model
  max_tokens: 512
  temperature: 0.0
  verify_steps: [generate_tools, verify_tools, generate_trajectories, judge_trajectories]
```

| Key | Default | Effect |
|---|---|---|
| `enabled` | `false` | master switch |
| `verify_steps` | four stages | only these get a verifier attached; the list is replaced wholesale on override |
| `model` | unset → the stage client's model | verifier model. It is not the `judge_trajectories` judge's model and does not read `steps.*` |
| `max_tokens` | `512` | cap on the verdict |
| `temperature` | `0.0` | |

Be aware of what it is before switching it on: it truncates any string field at 3000
characters and caps its verdict at 512 tokens, so its rejections are unreliable on
large items. Only 11 stages have criteria defined; a stage without them passes
automatically. ⚠ **A provider outage during verification fails the item.** `StageVerifier`
re-raises `LLMError` rather than returning a `False` verdict — deliberately, so an outage
cannot silently reject a whole run — but its only caller, `PipelineRunner`, wraps the call
in `except Exception` and records the item as failed with `verifier exception: …`. The
re-raise therefore never reaches you, and a dead key during a verified run demotes every
item to `failed.jsonl`.

---

## Keys in the shipped configs that nothing reads

Present in a shipped file, inert in the shipped code. Setting it has no effect:

| Key | Reality |
|---|---|
| `steps.judge_trajectories.temperature` | `JudgeTrajectoriesStep` builds its agent with `temperature=0` hardcoded, so the merged value is parsed and then ignored |

Output formats are not configurable either: `export_dataset` always writes JSON **and**
XML, except for `tagged` trajectories, which are JSON only, and the reasoning variant
follows each trajectory's own `tool_call_format`.

---

## Worked example: `quickstart`

[`configs/quickstart.yaml`](../configs/quickstart.yaml) is a complete `tools_only` run
over [`examples/ticket_api.jsonl`](../examples/ticket_api.jsonl) (8 tools, one category),
sized to finish rather than to produce a large dataset. Values below are the file's;
the `steps:` entries are folded into flow style for width.

```yaml
environment: quickstart
environment_type: tools_only   # no simulated user, no policy

llm:                      # swap this block to change model family; nothing else changes
  provider: openai
  model: gpt-4o

input:
  catalog: examples/ticket_api.jsonl
output_base: ./outputs/quickstart

categories:
  - ticket_api

terminate_tools: []       # this catalog has no hand-off tool

defaults:
  temperature: 0
  parallel_workers: 8
  max_retries: 10
  empty_response_retries: 10

steps:
  verify_tools:          {max_fix_iterations: 3}
  debug_environment:     {max_fix_iterations: 6}
  generate_states:       {num_states: 5, min_list_length: 2}
  plan_sequences:        {turns_per_sequence: "2,3,4", max_tools_per_sequence: 9,
                          sequences_per_category: 120, enable_multipath: false}
  execute_sequences:     {states_per_sequence: 1, max_samples_per_category: 5}
  generate_queries:      {temperature: 0.1}
  generate_trajectories: {tool_call_format: tagged, max_turns: 20, max_tokens: 32000}
  judge_trajectories:    {max_tokens: 16000}
  export_dataset:        {min_score: 8, require_task_completed: true}
```

Run it:

```bash
vulcan --env quickstart --pipeline
vulcan --env quickstart --stage build_graph            # one stage
vulcan --env quickstart --pipeline --start generate_trajectories --end judge_trajectories
```

What the numbers do, and where the cost is:

- `build_graph` is fixed by the catalog: 8 tools → **56 model calls**, no config
  involved. It is the easiest way to spend more than you meant to; 22 tools is 462.
- `generate_states.num_states: 5` is 5 model calls and 5 initial states.
  `base.yaml` ships `5000`. This is the diversity knob — raise it first.
- `execute_sequences` yields at most `states_per_sequence × sequences`, then
  `max_samples_per_category: 5` caps everything downstream. Trajectory generation is
  the expensive phase, so this cap is what keeps a first run cheap.
- `generate_trajectories.tool_call_format: tagged` needs no server-side tool-calling
  support and works on any model. Switch to `native` when your provider returns
  structured tool calls.
- `generate_trajectories.max_tokens: 32000` overrides `base.yaml`'s `2048`, which is a per-turn
  cap and too small for `tagged`.

**To scale it into a real run**, raise `generate_states.num_states`,
`plan_sequences.sequences_per_category`, `execute_sequences.states_per_sequence` and
`max_samples_per_category` — in that order — and only after one pipeline completes
end to end.

### Delta: a policy domain with a simulated user

[`configs/hotel_policy.yaml`](../configs/hotel_policy.yaml) has the same shape as the
above. The differences that change *behaviour* are these:

```yaml
environment_type: with_user_and_policy   # simulated user + policy-compliance judging

input:
  catalog: examples/hotel_booking.jsonl
output_base: ./outputs/hotel_policy
categories: [hotel_booking]

steps:
  generate_trajectories:
    tool_call_format: native
    max_turns: 40
    simulated_user_temperature: 0.0  # deterministic, so violations come from the agent
    simulated_user_name: user_sim
```

The rest of its `steps:` block is the same set of keys at slightly larger sizes, because
the catalog is bigger (10 tools, so `build_graph` costs 90 calls):
`debug_environment.max_fix_iterations: 8`, `generate_states: {num_states: 10,
min_list_length: 3}`, `plan_sequences: {max_tools_per_sequence: 6,
sequences_per_category: 200}`, `execute_sequences: {states_per_sequence: 2,
max_samples_per_category: 20}`, `generate_queries.temperature: 0.2`.

The user-simulator keys are flat on `steps.generate_trajectories`, which is what `GenerateTrajectoriesStep` reads.
`simulated_user_model` is left unset there, so the simulated user runs on the step's own model.

The policy text itself is **not** a config key: it lives inline on the category
record as `policy_rules` in
[`examples/hotel_booking.jsonl`](../examples/hotel_booking.jsonl). A policy in a
separate file is never loaded. Its presence also promotes that category to
`with_user_and_policy` on its own, which is why a `tools_only` config pointed at this
catalog would still take the user-agent path.

### Delta: an open-weight model you serve yourself

One local OpenAI-compatible server for the whole run. Only the `llm:` block changes;
every stage inherits its model from there:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1     # vLLM / SGLang / Ollama / llama.cpp / TGI
  model: Qwen/Qwen3-32B

steps:
  generate_trajectories:
    tool_call_format: tagged             # no server-side tool-call parser needed
    max_tokens: 32000
  judge_trajectories:
    model: Qwen/Qwen3-235B               # optional: a bigger local model to judge
```

Leave `defaults.model` unset, as above — it would override `llm.model` everywhere. To
put the judge on a *different provider*, add `provider:` (and `base_url:` if needed)
under `steps.judge_trajectories`; see
[Per-step provider and model overrides](#per-step-provider-and-model-overrides).
Native function calling (`tool_call_format: native`) against a self-hosted server
requires that server to be started for it — vLLM and SGLang need
`--enable-auto-tool-choice` with a matching `--tool-call-parser`. Otherwise use
`tagged`.

---

## Creating a new environment

1. Write an input catalog — JSONL, one object per category
   ([`../examples/README.md`](../examples/README.md)).
2. Copy [`vulcan/config/example_environment.yaml`](../vulcan/config/example_environment.yaml)
   to `./configs/<name>.yaml`.
3. Set `environment_type`, `input.catalog`, `output_base`, and list your
   categories under `categories`.
4. Set `terminate_tools: []` unless your catalog really has a hand-off tool.
5. Override only the step keys you need — everything else falls back to `base.yaml`.
6. Run `vulcan --env <name> --pipeline`.

---

## Reading config in code

```python
from vulcan.config import load_config, get_step_config
from vulcan.llm import create_client, resolve_llm_config

config = load_config("quickstart")        # base.yaml + configs/quickstart.yaml, paths resolved

step_cfg = get_step_config(config, "generate_trajectories")
step_cfg["temperature"]   # 0.7 from base.yaml
step_cfg["max_turns"]     # 20  from configs/quickstart.yaml
step_cfg["max_tokens"]    # 32000

resolve_llm_config(config, "judge_trajectories") # effective provider/model/key for that step
client = create_client(config, "judge_trajectories")   # cached per (provider, model, base_url, reasoning)
```

`load_config` accepts a path as well as a name, so an ad-hoc file works without
moving it into `configs/`:

```python
config = load_config("/abs/path/to/run.yaml")
```

---

## Things that surprise people

**`defaults.model` applies to every stage, whatever provider serves it.** Nothing
ships it set, and setting it yourself overrides `llm.model` for the whole run — which
in a mixed-provider config sends one vendor's model ID to another. Use
`steps.<step>.model` to move a single stage.

**The sequence budget is per category and then divided.** `StageRunner` dispatches
`plan_sequences` once per category, so each category gets the full
`sequences_per_category`, split `// len(turns_per_sequence)` across the turn counts —
ten categories on the shipped default is a 150 000-sequence budget, not 15 000.

**Lists replace, they never append.** `verify_steps`, `terminate_tools` and
`categories` are wholesale overrides.

**A stage can report success while every model call failed.** Steps catch per-item
`LLMError` and emit the row without the model's contribution, so a dead key yields
`success=N, failed=0` and N useless rows. The honest signal is `llm_calls=` in
`<output_base>/logs/<category>/<step>.log`.

**Runs resume on an input hash.** `_item_id` is an MD5 of an item's *input*, so a
completed item is skipped on re-run — **changing a model or temperature does not
invalidate it.** Delete the stage output to force regeneration.

**`--output-label` names the folder, `steps.generate_trajectories.tool_call_format`
controls generation.** They default to the same value now, but pass the flag
explicitly and `tagged` trajectories will land in a folder called `native`.

**`defaults.max_retries` and `llm.max_retries` are different budgets.** The first
re-processes a failed *item* — 1 + 10 passes by default; the second retries a single
HTTP request — 3 attempts by default. They multiply, so one stubborn item can burn
33 provider calls per model call it makes.

**The judge writes a nested dict.** `judge_trajectories` produces
`judgment.score` and `judgment.task_completed`, and
`export_dataset` reads that same nested path — not top-level fields.

---

## Security

VULCAN executes **model-generated Python in the host process, with no sandbox**.
`generate_tools` asks a model to write the environment class and `debug_environment` runs
it. No configuration key changes that. Run it in a container, VM, or disposable CI
worker with credentials scoped to nothing you care about, and do not point
`input.catalog` or `llm.base_url` at anything you do not trust. Read the
Security section of [`../README.md`](../README.md) before your first run.
