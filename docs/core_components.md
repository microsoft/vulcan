# Core Components

The moving parts under [`vulcan/llm/`](../vulcan/llm/) and [`vulcan/core/`](../vulcan/core/),
and what each one actually guarantees.

| Layer | Module | Responsibility |
|---|---|---|
| Provider | `llm/base.py` | `LLMClient.call()` — one request contract, one response shape, retries |
| Provider | `llm/{openai,anthropic,gemini,openai_compatible}_client.py` | translate to and from a vendor's wire format |
| Provider | `llm/factory.py` | build and cache clients from the `llm:` config block |
| Pipeline | `core/agent.py` | prompt assembly, empty-response retry, token accounting, output parsing |
| Pipeline | `core/runner.py` | run one step over a JSONL: parallelism, incremental save, resume |
| Pipeline | `core/stage_runner.py` | resolve every stage's input and output path, then dispatch |
| Support | `core/parsers.py`, `core/json_utils.py`, `core/io.py`, `core/logger.py` | tag extraction, JSON repair, JSONL I/O, logging |
| Support | `core/tool_exec.py`, `core/code_exec.py` | execute generated tools — **`code_exec` runs model-written Python in-process** |

Entry points are the `vulcan` console script and `python -m vulcan.cli`; both call
`cli.main()`. There are no shell wrappers.

---

## The provider layer (`vulcan/llm/`)

Every LLM call in the pipeline goes through `LLMClient`. Adapters translate to each
vendor's format and return one normalised, OpenAI-shaped response, so no step ever
branches on which model answered. Setup, keys and per-provider capability differences
live in [`providers.md`](providers.md); this section is the code contract.

### LLMClient (`llm/base.py`)

Abstract base. Subclasses implement `_request()`; the base class owns argument
normalisation, timeouts, retry/backoff and error typing, so all four providers behave
identically from the pipeline's point of view.

```python
from vulcan.llm import create_client

llm = create_client(config, "generate_trajectories")   # per-step overrides; omit the name for the global block

response = await llm.call(
    messages,                          # OpenAI-shaped: system | user | assistant | tool
    model=None,                        # None → the client's configured default
    temperature=0.0,                   # None → let the provider decide
    max_tokens=4096,
    thinking_model=False,
    tools=None,                        # [{"type": "function", "function": {...}}]
    tool_choice="auto",                # "auto" | "none" | "required"
    reasoning=None,                    # {"effort": "medium"} or {"budget_tokens": 4096}
    stop=None,
)
```

Reasoning resolves as: the `reasoning=` argument → the client's `default_reasoning`
(built from `llm.reasoning` / `steps.<step>.reasoning`) → `{"effort": "medium"}` when
`thinking_model=True` and neither of the first two applies. Empty strings in `stop` are dropped, and an empty list
is omitted from the request entirely. Unknown keyword arguments are swallowed by
`**_ignored` — that is deliberate compatibility padding for call sites written against
the retired single-endpoint client (`tier`, `n`); they have no effect.

`await llm.aclose()` releases transport resources and is safe to call twice.

#### The normalised response

Every adapter returns exactly this shape, whatever the provider sent:

```python
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "…",              # always present, "" when the model returned none
        "reasoning_content": "…",    # "" when the model exposes no reasoning text
        "tool_calls": [ … ],         # key is ABSENT when the model requested none
      },
      "finish_reason": "stop",       # stop | length | tool_calls | content_filter | error
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}
```

`choices` always has at least one entry. Tool calls are always the OpenAI form, and
`arguments` is always a **JSON string** even for providers that return a parsed object:

```python
{"id": "call_0", "type": "function",
 "function": {"name": "create_ticket", "arguments": "{\"priority\": \"high\"}"}}
```

Downstream code re-embeds these objects verbatim into the outgoing message list and maps
`tool_call_id` back to a tool name when building training records, so ids must be stable
and non-empty. Adapters for providers that omit ids mint deterministic ones.

#### Timeouts, retries, failure

| Constant (`llm/base.py`) | Default | Meaning |
|---|---|---|
| `DEFAULT_REQUEST_TIMEOUT` | `600.0` | seconds for one attempt, end to end (`llm.request_timeout`) |
| `DEFAULT_RETRY_ATTEMPTS` | `3` | **total attempts**, not extra retries (`llm.max_retries`) |
| `DEFAULT_RETRY_BASE_DELAY` | `2.0` | seconds before the first retry |
| `DEFAULT_RETRY_MAX_DELAY` | `60.0` | ceiling on any single backoff |
| `RETRYABLE_STATUS` | `{408, 409, 425, 429, 500, 502, 503, 504, 529}` | everything else is raised immediately |

Each attempt is wrapped in `asyncio.wait_for(..., timeout=request_timeout)`. Backoff is
`min(2 · 2ⁿ, 60)` seconds scaled by a random factor in `[0.5, 1.0)`, so parallel workers
do not resynchronise after a rate-limit burst. A status outside `RETRYABLE_STATUS` (400,
401, 403, 404, 422 …) is re-raised on the first attempt: the provider will reject the same
request identically next time, and retrying only burns quota and hides the real error.
When every attempt fails, `call()` raises `LLMError` carrying `status_code`, a truncated
`body`, and `provider`.

### The four adapters

| Module | `provider` | Dependency | Transport |
|---|---|---|---|
| `openai_client.py` | `openai` | `pip install -e ".[openai]"` | `AsyncOpenAI` (SDK retries disabled) |
| `anthropic_client.py` | `anthropic` | `pip install -e ".[anthropic]"` | `AsyncAnthropic` (SDK retries disabled) |
| `gemini_client.py` | `gemini` | `pip install -e ".[gemini]"` | `google.genai` async client |
| `openai_compatible.py` | `openai_compatible` | none | raw `aiohttp` to `POST {base_url}/chat/completions` |

Retries are owned by `LLMClient.call()`, which is why each SDK is constructed with
`max_retries=0`. A missing optional dependency raises `LLMError` naming the install extra,
not `ImportError`.

**`openai`** — the reference adapter: OpenAI's chat-completions shape *is* the normalised
shape. Reasoning families (`o1`/`o3`/`o4`-series, `gpt-5`-series) take
`max_completion_tokens` instead of `max_tokens` and reject a custom `temperature`, which
the adapter drops rather than failing on. `reasoning` becomes `reasoning_effort`. Public
OpenAI reasoning models bill for reasoning tokens but do not return the text, so
`reasoning_content` comes back empty.

**`anthropic`** — the Messages API differs in four ways that all matter here. The system
prompt is split out of the message list (`LLMClient._split_system`) into a top-level
`system` field. Tool calls and results are content blocks, not a `tool` role, so
consecutive tool results merge into one user message and consecutive same-role messages
merge (Anthropic requires strict alternation); a conversation that does not open with a
user turn has its leading turns dropped. Tool inputs are objects, so they are dumped back
to a JSON string. And thinking returns blocks carrying a `signature` that must be echoed
on the next turn when tool use is in play — VULCAN's trajectory loop rebuilds assistant
messages from `content` + `tool_calls` alone, so the adapter keeps a bounded
(`_THINKING_CACHE_SIZE = 512`) cache of thinking blocks keyed by tool-call id and
re-attaches them inbound. For Claude 4.6 and later the adapter builds
`{"type": "adaptive", "display": "summarized"}` plus `output_config.effort`; older models
get the fixed-budget `{"type": "enabled", "budget_tokens": n}` form. Either way the block is
sent on the request key **`reasoning`**, not `thinking`, even though the helper that builds
it is `_thinking_config`. `display: "summarized"`
is set explicitly because the API default is `omitted`, which would return empty thinking
blocks. `temperature` is omitted whenever thinking is on, and on model families that
reject sampling parameters outright.

**`gemini`** — messages become `contents` with roles `user`/`model`, the system prompt
becomes `system_instruction`, and tool calls become *parts*. Gemini supplies **no call
id**, matching results by function *name*, so the adapter mints `call_<n>_<name>` ids and
keeps a `tool_call_id → name` map to route the matching result back. Function-declaration
schemas accept only a subset of JSON Schema, so `_sanitise_schema` strips every key
outside `_ALLOWED_SCHEMA_KEYS` (notably `additionalProperties` and `$schema`, which
otherwise 400) and adds `type: object` where properties exist without one. Thought
summaries arrive as ordinary text parts flagged `thought`; thinking tokens are reported
separately and are folded into `completion_tokens`.

**`openai_compatible`** — plain HTTP against any server implementing
`POST /v1/chat/completions`: vLLM, SGLang, Ollama, llama.cpp, TGI, LM Studio, a LiteLLM
proxy, or a hosted compatible service. Built on `aiohttp` rather than the OpenAI SDK so
that running an open-weight model needs no optional dependency at all. It always sends
`max_tokens` (never `max_completion_tokens`, which older servers reject), reads reasoning
from `reasoning_content` (vLLM, SGLang) then `reasoning` (some proxies) — the third branch
is meant to be Ollama's `thinking` but is a duplicate of `reasoning`, so a server that
reports reasoning under `thinking` yields none — coerces Ollama's parsed-object
`arguments` back to a string, and mints
`call_<index>` where the server omits an id. `base_url` is required — constructing the
client without one raises `LLMError`. Some servers report errors with HTTP 200 and an
`"error"` key; that is detected and raised rather than returned as an empty completion.
Native tool calling needs server-side support (for vLLM,
`--enable-auto-tool-choice --tool-call-parser <parser>`); without it use
`generate_trajectories.tool_call_format: tagged`, which parses tool calls out of plain text.

### Capability detection (`llm/capabilities.py`)

`detect(provider, model)` returns a frozen `ModelCapabilities` dataclass —
`uses_max_completion_tokens`, `supports_temperature`, `supports_reasoning`,
`returns_reasoning_text`, `supports_tools`, `supports_stop`, `uses_adaptive_thinking` —
inferred from the model name. Detection is a *default*, never a rule: every field can be
pinned in config, which is what a self-hosted model matching none of the name patterns
needs.

```yaml
llm:
  provider: openai_compatible
  model: my-org/my-finetune
  capabilities:
    supports_reasoning: true
    supports_temperature: true
```

`ModelCapabilities.merged(overrides)` applies the config block; adapters call
`self._caps(model)` per request, so a per-step model override still gets its own detection.

### create_client / resolve_llm_config (`llm/factory.py`)

```python
from vulcan.llm import create_client, resolve_llm_config, close_clients, load_dotenv

load_dotenv()                                   # ./.env; already-exported vars win
settings = resolve_llm_config(config, "generate_trajectories")
llm      = create_client(config, "generate_trajectories")
...
await close_clients()                           # closes every cached client
```

`resolve_llm_config(config, step_name)` merges the global `llm:` block with per-step
overrides and resolves credentials:

1. `steps.<step>.{provider,model,base_url,reasoning,capabilities,extra_body}` override the
   global block when not `None`;
2. an unset `model` falls back to `defaults.model`;
3. `provider` defaults to `openai` and must be one of `openai`, `anthropic`, `gemini`,
   `openai_compatible` — anything else raises `LLMError`;
4. `api_key` and `base_url` expand `${VAR}` (an unresolved placeholder counts as unset) and
   otherwise fall back to the provider's environment variables (`DEFAULT_KEY_ENV` /
   `DEFAULT_URL_ENV`; table in [`providers.md`](providers.md#api-keys)).

`create_client()` caches on `(provider, model, base_url, reasoning)`, so a run that
generates on a local model and judges on a hosted one opens one connection pool per backend,
not one per step — while two stages that share a model but ask for different reasoning still
get their own client. The resolved `reasoning` is handed to the client as its
`default_reasoning` and applied to every call that does not pass one. Construction fails
fast — before the first request — when no model is configured, or when a
provider other than `openai_compatible` has no API key. `vulcan.cli` builds one client per
stage this way, so per-step `provider` / `base_url` / `reasoning` take effect in an ordinary
run; the startup fail-fast check, however, sees only the global `llm:` block, so a step-level
provider with no key raises when its stage starts. `load_dotenv()` is a deliberately
minimal `KEY=value` reader: no interpolation, no `export`, missing files ignored, existing
environment variables always win.

---

## Agent (`core/agent.py`)

One class for every LLM interaction in the pipeline. It owns system-prompt injection,
empty-response retries, token accounting and optional output parsing. It never learns
which vendor answered.

```python
from vulcan.core.agent import Agent
from vulcan.core.parsers import TagParser

agent = Agent(
    "func_gen",                  # name — also the `role` of the returned dict
    EXECUTOR_SYSTEM,             # system prompt ("" is fine; steps often pass per-call)
    llm,                         # an LLMClient
    model=self.model,            # "" / None → the client's configured default
    temperature=0,
    max_tokens=4096,
    thinking_model=False,
    output_parser=TagParser.make_parser("verified_code"),
)
```

`empty_response_retries` and `logger` are set by the owning step
(`BaseStep.set_logger` / `_propagate_config_to_agents`), not by the constructor.

### Single turn

```python
result = await agent.run(
    messages,                  # system message is prepended if absent
    progress=None,             # tqdm bar; advanced by one per call
    system_prompt=None,        # override this agent's system prompt for one call
    tools=None,
    stop=None,
)
```

| Key | Value |
|---|---|
| `role` | the **agent's name**, not `"assistant"` |
| `content` | `str`, `""` when the model returned nothing |
| `reasoning_content` | `str`, `""` when unavailable |
| `tool_calls` | present **only** when the model requested one |
| `finish_reason` | from the provider; `"error"` when no choices came back |
| `usage` | `{prompt_tokens, completion_tokens, total_tokens}` |
| `parsed_content` | `output_parser` result; the raw `content` if the parser raised; `None` on a tool-call turn |
| `messages` | the exact message list that was sent, system message included |

The retry loop runs up to `1 + empty_response_retries` times and stops as soon as the turn
has non-blank `content` **or** any `tool_calls` — a turn that only requests a tool call is
legitimately text-free, and counting it as empty would burn the whole retry budget and cut
a trajectory short. Retries sleep 2 s. A parser that raises is not an error: `run()` falls
back to the raw content.

`LLMError` propagates out of `Agent.run` rather than being swallowed, so a dead API key
fails loudly instead of producing rows with no model contribution. Individual steps may
still catch it themselves — `generate_tools`, for example, re-raises only `StepError` and
`KeyError` and swallows everything else, returning the item unchanged, which is why a
stage can report `success=N, failed=0` while every model call failed. The honest signal
is `llm_calls=` in the step log.

### Multi-turn

`run_conversation(initial_messages, follow_ups, progress=None)` answers the opening
messages, then replies to each follow-up **user** string in sequence, returning one
response dict per turn. It stops early as soon as a turn returns empty `content`.

```python
results = await agent.run_conversation(
    initial_messages=[{"role": "user", "content": "Open a ticket for a failed payment."}],
    follow_ups=["Now raise its priority.", "What is its status?"],
)
```

---

## PipelineRunner (`core/runner.py`)

Runs one step across a JSONL input with bounded concurrency, incremental saving, retry and
resume.

```python
from vulcan.core.runner import PipelineRunner, run_step

runner = PipelineRunner(
    step,                       # a BaseStep subclass instance
    parallel_workers=10,        # StageRunner passes defaults.parallel_workers (20)
    max_retries=0,              # StageRunner passes defaults.max_retries (10)
    logger=logger,              # optional PipelineLogger
    verifier=verifier,          # optional StageVerifier
)

stats = await runner.run(input_path, output_path, first_k=None)
# {"total": T, "success": N, "failed": M,
#  "retries_used": R, "success_path": …, "failed_path": …}
```

`run_step(step, input_path, output_path, …)` is a one-line wrapper around the same thing.

Before the batch starts, `run()` calls `step.preprocess_input(data)` (which may explode one
row into many), applies `first_k`, stamps every item lacking one with
`_item_id = md5(json.dumps(item, sort_keys=True))[:12]`, sets `step.out_dir`, and calls
`step._propagate_config_to_agents()`.

### File protocol

All three files live in the directory containing `output_path`:

```
<step_dir>/
├── processing.jsonl    # temp — every completed item written immediately
├── success.jsonl       # validation passed  (appended after each batch)
└── failed.jsonl        # validation failed, or the item exhausted its retries
```

### Resume and crash recovery

1. A stale `processing.jsonl` from a crashed run is finalised first: its rows are split
   into `success.jsonl` / `failed.jsonl`, deduplicated against the `_item_id`s already in
   `success.jsonl`, and the temp file is deleted.
2. Items whose `_item_id` is already in `success.jsonl` are skipped.
3. **If `failed.jsonl` is non-empty, this run processes only those items** — the failed
   file is truncated and re-populated, and items never attempted are not picked up until
   the failures clear.

`_item_id` hashes the item's *input*, so a completed item stays completed across runs.
Changing the model, the temperature or a prompt does **not** invalidate it; delete the
stage output to force regeneration.

The `success` / `failed` counts in the returned stats are line counts of the two files, so
after a resumed run they are cumulative totals, not the count for this invocation.

### Per-item error handling

Each worker catches per item, so one bad row cannot abort the batch. `StepError` is
recorded with its message; any other `Exception` (and `SystemExit`) is recorded with a full
traceback, printed, and the *input* item is written to `failed.jsonl` with
`_validation_reason`. `LLMError` is not a `StepError`, so an unreachable provider fails
items with a traceback rather than silently.

### Verifier hook

When a `verifier` is supplied, an item that passes the step's own
`validate_output()` is additionally sent to `verifier.verify(step_name, item)`. A
`(False, reason)` verdict demotes it to failed with `reason` prefixed `verifier:`; an
exception inside the verifier also fails the item.

---

## StageRunner (`core/stage_runner.py`)

Declarative orchestration for one stage across one or more categories. Stateless — the CLI
builds one instance and reuses it for every stage.

```python
from vulcan.core.stage_runner import StageRunner

runner = StageRunner()
ok = await runner.run(
    stage="generate_tools",
    config=config,               # merged dict from load_config()
    llm=llm,                     # an LLMClient — the CLI passes create_client(config, stage)
    environment=environment,     # an EnvironmentConfig
    categories=["ticket_api"],   # default: config["categories"], else every catalog category
    iteration=0,
    output_label="native",       # names the trajectories/ and judged/ subdirectory
)
```

Every argument is keyword-only.

`run()` iterates the selected categories itself and returns `True` only if every processed
category succeeded — where "succeeded" means *at least one item passed*, so a stage that
salvaged 1 row out of 500 still reports success. A category that raises does not abort the
others: `VulcanError` is logged and that category alone is marked failed. Workers and
retries come from `defaults.parallel_workers` / `defaults.max_retries`, and the
`PipelineLogger` is rooted at `<output_base>/logs`.

### Step order

`PIPELINE_ORDER` in [`vulcan/steps/__init__.py`](../vulcan/steps/__init__.py) is the
canonical sequence — 15 steps, which is what `vulcan --list-steps` prints and what
`--pipeline --start/--end` slices:

```
normalize_specs → generate_tools → verify_tools → assemble_environment → debug_environment
  → generate_states → sample_arguments
  → build_graph → plan_sequences → score_sequences
  → execute_sequences → generate_queries → generate_trajectories → judge_trajectories → export_dataset
```

### Input resolution

`_STANDARD_INPUT` in `core/stage_runner.py` is the authoritative table for
straightforward input→output stages. Steps do **not** describe their own predecessors.

```python
_STANDARD_INPUT: dict[str, tuple[str, str]] = {
    "generate_tools":       ("normalize_specs",      "success.jsonl"),
    "verify_tools":         ("generate_tools",       "success.jsonl"),
    "assemble_environment": ("env_grouped",          "success.jsonl"),
    "debug_environment":    ("assemble_environment", "success.jsonl"),
    "generate_states":      ("debug_environment",    "success.jsonl"),
    "sample_arguments":     ("generate_states",      "success.jsonl"),
    "build_graph":          ("env_grouped",          "success.jsonl"),
    "score_sequences":      ("plan_sequences",       "multi_turn/all_sequences.jsonl"),
    "generate_queries":     ("execute_sequences",    "pre_query_all_modes.jsonl"),
}
```

For a stage `X` in category `C`, the input is `<output_base>/C/<prev>/<file>` and the
output is `<output_base>/C/X/success.jsonl`. A missing input is logged as
`[X/C] input not found: …` and the category fails — it is not created on demand.

Stages absent from that table have dedicated handlers:

| Stage | Handler behaviour |
|---|---|
| `normalize_specs` | builds its own input from `environment.get_api_catalog(cat)` (plus `policy_rules` from `get_domain_rules`) and writes it to `<cat>/normalize_specs/input.jsonl` first |
| `verify_tools` | runs as a standard stage, then `_build_env_grouped` regroups the per-API rows into a single `<cat>/env_grouped/success.jsonl` record with a unified `api_list` |
| `plan_sequences` | synchronous `step.run_batch(build_graph/success.jsonl, <cat>/plan_sequences/)` → `multi_turn/all_sequences.jsonl` |
| `execute_sequences` | synchronous `step.run_batch(plan_sequences multi_turn, sample_arguments/success.jsonl, <cat>/execute_sequences/)` → `pre_query_all_modes.jsonl` |
| `generate_trajectories` | input is `generate_queries/success.jsonl` at `iteration=0`, else `iter_<k>/input.jsonl`; output `iter_<k>/trajectories/<label>/success.jsonl`; runs with `max_retries=0` |
| `judge_trajectories` | `iter_<k>/trajectories/<label>/success.jsonl` → `iter_<k>/judged/<label>/success.jsonl`; also `max_retries=0` |
| `export_dataset` | runs **once across all categories**, before the per-category loop |

`env_grouped` is written by the runner, not by a step. If no API survived
`verify_tools` with a `tool_code`, it is skipped entirely — and both
`assemble_environment` and `build_graph` then fail with `input not found`.

`export_dataset` reads every `iter_<k>/judged/*/success.jsonl` subdirectory, writes
selected records to `training_dataset/{json,xml}/<category>_iter_<k>.jsonl`, and routes
rejects to `<cat>/iter_<k+1>/input.jsonl` with the step's own annotations
(`STRIP_FIELDS`, `_selected`, `_training_json`, `_training_xml`, `_validation_reasons`)
removed so the row can feed the next iteration.

### Output layout

```
<output_base>/
├── <category>/
│   ├── <step>/success.jsonl                 # + failed.jsonl
│   ├── env_grouped/success.jsonl
│   ├── plan_sequences/multi_turn/all_sequences.jsonl
│   ├── execute_sequences/pre_query_all_modes.jsonl
│   ├── iter_<k>/trajectories/<label>/success.jsonl
│   ├── iter_<k>/judged/<label>/success.jsonl
│   └── iter_<k+1>/input.jsonl
├── training_dataset/{json,xml}/<category>_iter_<k>.jsonl
└── logs/
```

Configs set `output_base` to a relative path such as `./outputs/quickstart`; `load_config`
expands `~` and resolves it to an absolute path before any step sees it.

### Verifier construction

`_make_verifier` attaches a `StageVerifier` only when `verification.enabled` is true **and**
the stage is listed in `verification.verify_steps`. It is handed the `verification:` block
itself — not the whole config — because `StageVerifier` reads `model` / `max_tokens` /
`temperature` from the dict it is given.

---

## BaseStep (`steps/base.py`)

```python
from vulcan.steps import register_step
from vulcan.steps.base import BaseStep
from vulcan.core.agent import Agent

@register_step("my_step")
class MyStep(BaseStep):
    requires_llm = True                        # False for deterministic steps

    def __init__(self, config, llm, env):      # llm is an LLMClient
        super().__init__(config, llm, env)
        self.agent = Agent("my_agent", MY_SYSTEM, llm,
                           model=self.model, temperature=self.temperature,
                           max_tokens=self.max_tokens)

    async def run(self, item: dict, progress=None) -> dict:
        result = await self.agent.run(
            [{"role": "user", "content": item["query"]}], progress)
        return {**item, "new_field": result["content"]}

    def validate_output(self, item: dict) -> tuple[bool, str]:
        if not item.get("new_field"):
            return False, "new_field missing or empty"
        return True, ""
```

`__init__` takes `(config, llm, env)` and stores the client as `self.llm`; steps pass it
straight to every `Agent` they construct. Generation settings resolve
`steps.<step>.<key>` > `defaults.<key>` > built-in: `model` (`""` → whatever the client
defaults to), `temperature` (`0`), `max_tokens` (`16000`), `thinking_model` (`False`),
`empty_response_retries` (`10`).

| Method | Purpose |
|---|---|
| `run(item, progress)` | process one item — **must** override |
| `validate_output(item)` | returns `(passed, reason)`; default checks the `VALIDATORS` registry, then `output_schema` |
| `preprocess_input(data)` | static; transform the loaded list before the batch (flatten, explode) |
| `validate_input(item)` | validate against `input_schema` when set; raises `ValidationError` |
| `set_logger(logger, category)` | attach the logger to the step and every `Agent` attribute |
| `_propagate_config_to_agents()` | push `empty_response_retries` onto every `Agent` attribute |
| `get_output_dir()` / `get_output_path()` | `<output_base>/<step>/` and `final_out_<step>.jsonl` within it |

`validate_output` returning a bare `bool` is a bug: `PipelineRunner` unpacks
`(passed, reason)`. Input paths are **not** a step concern — `StageRunner` resolves them
(see `_STANDARD_INPUT` above). Full walkthrough: [`adding_new_steps.md`](adding_new_steps.md).

### Schemas

Steps may declare Pydantic models for self-documenting I/O; `validate_input` and the
default `validate_output` use them when present. **No shipped step sets either
attribute** — [`vulcan/schemas/step_io.py`](../vulcan/schemas/step_io.py) exports
`BaseStepInput`, `GenerateToolsInput`, `GenerateToolsOutput`, `GenerateTrajectoriesInput`, `GenerateTrajectoriesOutput`
and `JudgeTrajectoriesOutput` as documentation of the contracts, and validation in practice
comes from the `VALIDATORS` registry in `validators/deterministic.py`. To opt a step in:

```python
from vulcan.schemas import GenerateToolsInput, GenerateToolsOutput

@register_step("generate_tools")
class GenerateToolsStep(BaseStep):
    input_schema = GenerateToolsInput
    output_schema = GenerateToolsOutput
```

---

## TagParser (`core/parsers.py`)

Extracts XML-tagged blocks from model output — the pipeline's main structured-output
mechanism, since most prompts ask for `<decision>…</decision><generated_code>…</generated_code>`
rather than raw JSON.

```python
from vulcan.core.parsers import TagParser

blocks = TagParser.extract_all("generated_code", text)      # list[str]
code   = TagParser.extract_first("generated_code", text)    # str | None
pairs  = TagParser.extract_pairs("decision", "generated_code", text)
data   = TagParser.extract_json("json_schema", text)        # parsed | raw str | None

parse_code = TagParser.make_parser("generated_code")        # async, for Agent(output_parser=…)
parse_both = TagParser.make_pair_parser("decision", "generated_code")
parse_json = TagParser.make_json_parser("json_schema")
```

All matching is `re.DOTALL` and non-greedy. `extract_pairs` only matches tags that are
**adjacent** in the text. `extract_json` returns the raw string when the block is not valid
JSON, so callers must type-check the result; for messier output use
`core/json_utils.try_parse_json`.

---

## Exception hierarchy (`core/exceptions.py`)

```
VulcanError
├── StepError(step_name, item_id, message, cause)
├── LLMError(message, status_code=None, body="", provider="")
├── ValidationError(message, field="", value=None)
├── EnvironmentError(message, env_name="", tool_name="")
└── ParseError(message, raw="")
```

`vulcan.core.exceptions.EnvironmentError` shadows the builtin of the same name — import it
explicitly or alias it, as `generate_trajectories` does.

```python
from vulcan.core.exceptions import LLMError, StepError, VulcanError

try:
    result = await step.run(item, progress)
except StepError as e:
    print(f"step {e.step_name} failed for {e.item_id}: {e.message}")
except LLMError as e:
    print(f"{e.provider} error (HTTP {e.status_code}): {e.body}")
except VulcanError as e:
    print(f"pipeline error: {e}")
```

`LLMError` is raised either after a client has exhausted its attempts, or immediately when
the status is outside `RETRYABLE_STATUS` (400, 401, 403, 404, 422 — see
[above](#llmclient-llmbasepy)). `StageVerifier` deliberately re-raises it instead of
returning a `False` verdict, so that a provider outage cannot silently reject every item in
the run — but note that its caller `PipelineRunner` catches `Exception` around that call and
records the item as failed anyway, so the re-raise does not currently propagate.

---

## PipelineLogger (`core/logger.py`)

```python
from vulcan.core.logger import PipelineLogger

logger = PipelineLogger("./outputs/quickstart/logs", "quickstart")
cat_log  = logger.get_category_logger("ticket_api")
step_log = logger.get_step_logger("generate_tools", "ticket_api")
print(logger.summary())
```

`StageRunner` constructs it as `PipelineLogger(<output_base>/logs, environment.name)`, so
the tree below is rooted at `<output_base>/logs/`:

```
logs/
├── pipeline.log                    # global — also the only logger echoed to the console
├── summary.txt                     # written by summary()
├── <category>/
│   ├── pipeline.log
│   └── <step_name>.log
└── steps/
    └── <step_name>.log             # steps run without a category
```

Beyond `info` / `debug` / `warning` / `error`, it tracks lifecycle and cost:
`step_start` / `step_end`, `item_success` / `item_fail` / `item_retry`,
`llm_call(...)` (call counts and prompt/completion/total tokens, aggregated per step and
globally), and `exception`. `Agent` calls `llm_call` after every request, which is why
`llm_calls=` in a step log is the reliable evidence that a stage actually reached a model.

---

## Tool execution (`core/tool_exec.py`)

The one canonical implementation used by every trajectory step.

```python
from vulcan.core.tool_exec import execute_tool, normalize_schema, clean_for_json

result = await execute_tool(env_instance, "create_ticket", {"priority": "high"})
normalize_schema(raw_schema)          # MUTATES IN PLACE, returns None
safe_obj = await clean_for_json(obj)  # async; numpy scalars/arrays, complex → plain Python
```

`execute_tool` looks the name up in `env.function_name_mapping()`, calls it, cleans the
result and returns a JSON **string** (a string result is passed through unchanged). It
raises `EnvironmentError` when the mapping fails, the name is unknown, or the tool itself
raises; `generate_trajectories._exec_tool` catches that and returns `{"error": …}` as JSON so a broken
tool degrades into a tool response the model can react to instead of killing the
trajectory. **It applies no timeout** — a generated tool that blocks forever blocks the
worker.

`normalize_schema` fixes a schema for OpenAI-style function calling in place: drops
top-level `response` and `constraints`, rewrites `dict`→`object`, `float`→`number`,
`int`→`integer` recursively, and gives every `array` without `items` a default
`{"type": "string"}`.

---

## Code execution (`core/code_exec.py`)

> **This module executes model-generated Python in the host process, with no sandbox.**
> `load_api_class` and `execute_class_code` call `exec()` on source written by a language
> model; `safe_execute_tool` then calls into the objects that source defines. The code runs
> with the full privileges of the interpreter — it can read and write any file the process
> can reach, open sockets, spawn subprocesses and import anything. Nothing here inspects,
> restricts or contains it. Run VULCAN only in a disposable, isolated environment: see the
> warning at the top of the [README](../README.md).

```python
from vulcan.core.code_exec import (
    Tool, load_api_class, execute_class_code, safe_execute_tool,
    find_unused_invoke_params, strip_get_info,
)

cls  = load_api_class(code)                     # first class that looks like an API tool
cls2 = execute_class_code(code)                 # first class defined, skipping `Tool`
res  = safe_execute_tool(env, "create_ticket", args, timeout=20)
```

`load_api_class` first `exec`s each import line on its own (ignoring `ImportError`), then
`exec`s the whole snippet, then searches the resulting namespace in three passes: a
subclass of `Tool`, then any class with callable `invoke` **and** `get_info`, then any
class with a callable `invoke`. With `strict=True` (the default) it raises
`EnvironmentError` when nothing matches. `execute_class_code` parses the AST, picks the
first `ClassDef` that is not named `Tool`, and `exec`s the code into a fresh module.
**Neither applies any timeout**: the `exec` runs inline, on the calling thread.

`safe_execute_tool` **does** bound its call, and the bound is real but partial. The tool
runs on a daemon worker thread joined for `timeout` wall-clock seconds (default `20`;
`timeout <= 0` waits forever). On expiry it returns
`{"error": "tool '…' timed out after 20s (call abandoned; it may still be running)", "success": False}` —
Python cannot kill a running thread, so the call is *abandoned*, not stopped, and keeps
consuming CPU until the process exits. The worker is a daemon precisely so an abandoned
call cannot hold up interpreter shutdown. An exception raised inside the tool is re-raised
on the calling thread and converted to `{"error": str(exc), "success": False}`, so the
timed and untimed paths behave identically. A string result that parses as JSON is decoded
first.

`find_unused_invoke_params(code)` reports parameters the **last** `invoke()` definition
never references (used by `verify_tools` to catch tools that ignore their arguments);
`strip_get_info(code)` returns just the `invoke` method body.

---

## JSON utilities (`core/json_utils.py`)

Models emit almost-JSON constantly, so parsing is four strategies deep:

```python
from vulcan.core.json_utils import (
    try_parse_json, try_parse_json_list, try_parse_json_dict, safe_json_loads,
)

obj, err = try_parse_json(raw)          # (parsed, None) | (None, "reason")
rows, err = try_parse_json_list(raw)    # validates: non-empty list of dicts
data, err = try_parse_json_dict(raw)
obj = safe_json_loads(raw)              # best effort; returns the raw string on failure
```

1. `json.loads` as-is;
2. preprocess (strip markdown code fences and `//` comments, `True`/`False`/`None` →
   `true`/`false`/`null`, drop trailing commas) and retry;
3. `json_repair.repair_json` when the optional `json_repair` package is installed;
4. slice from the outermost `[`…`]` or `{`…`}` and try both again.

`try_parse_json_list` also unwraps singly-nested lists (`[[{…}]]` → `[{…}]`).

---

## I/O utilities (`core/io.py`)

```python
from vulcan.core.io import load_jsonl, save_jsonl, append_jsonl, load_json, ensure_dir

items = load_jsonl("input.jsonl")    # list[dict]; blank lines skipped
save_jsonl(items, "out.jsonl")       # overwrites; creates parent dirs
append_jsonl(items, "out.jsonl")     # appends;   creates parent dirs
obj = load_json("catalog.json")
ensure_dir("./outputs/quickstart")   # mkdir -p; returns the path
```

These are intentionally unbuffered and synchronous. `PipelineRunner` uses `aiofiles`
directly for the hot incremental-write path.

---

## StageVerifier (`verification/verifier.py`)

An optional LLM-as-judge pass *inside* a stage, distinct from the `judge_trajectories` step.

```python
from vulcan.verification import StageVerifier

verifier = StageVerifier(llm, config["verification"])
passed, reason = await verifier.verify("generate_tools", item)
```

It takes an `LLMClient` and the `verification:` block (`model` — unset means the client's
own model, never a hardcoded one; `max_tokens`, default `512`; `temperature`, default `0.0`). `verify()` looks up the
stage's `check_items` in `verification/criteria.py`, extracts only the fields relevant to
that stage, builds a checklist prompt and parses a
`{"passed": bool, "issues": [...], "confidence": float}` verdict, tolerating markdown
fences and surrounding prose.

Two limits worth knowing before enabling it: string fields are truncated at 3000
characters and verdicts are capped at `max_tokens` (512 by default), so long items are
judged on a fragment; and a stage with no entry in `VERIFICATION_CRITERIA` passes
unconditionally. `LLMError` is re-raised rather than converted to a failed verdict — any
other exception becomes `(False, "verifier error: …")`.
