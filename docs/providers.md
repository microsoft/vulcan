# Providers

Every LLM call in VULCAN goes through one interface, `LLMClient` in
[`vulcan/llm/base.py`](../vulcan/llm/base.py). Adapters translate to and from each
vendor's wire format and return a single normalised, OpenAI-shaped response, so no
pipeline step ever branches on which model answered it. Changing provider is a config
change and nothing else.

| `provider` | Serves | Install |
|---|---|---|
| `openai` | GPT models via the OpenAI API | `pip install -e ".[openai]"` |
| `anthropic` | Claude models via the Anthropic Messages API | `pip install -e ".[anthropic]"` |
| `gemini` | Gemini models via the Google Generative AI API | `pip install -e ".[gemini]"` |
| `openai_compatible` | Any OpenAI-compatible server — this is how open-weight models are served | *(no extra needed)* |

## The `llm:` block

```yaml
llm:
  provider: openai          # openai | anthropic | gemini | openai_compatible
  model: gpt-4o
  base_url: null            # required for openai_compatible
  api_key: null             # prefer the environment; see below
  request_timeout: 600      # seconds for one attempt, end to end
  max_retries: 3            # attempts before raising LLMError
  reasoning: null           # {effort: low|medium|high} or {budget_tokens: N}
  capabilities: {}          # pin autodetection, see "Capability overrides"
  extra_body: {}            # passed to the provider verbatim
```

### API keys

Keys are read from the environment, never from a config file. Copy `.env.example` to
`.env` and fill in the one you need — `vulcan` loads it at startup, and a variable
already exported in your shell always wins.

| `provider` | Key variable | Base-URL variable |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| `gemini` | `GEMINI_API_KEY`, else `GOOGLE_API_KEY` | — |
| `openai_compatible` | `VULCAN_API_KEY`, else `OPENAI_API_KEY` | `VULCAN_BASE_URL`, else `OPENAI_BASE_URL` |

`openai_compatible` is the only provider that will start without a key, because most
local servers ignore it. The others fail fast with a message naming the variable to set.

## Per-provider setup

### OpenAI (GPT)

```yaml
llm: {provider: openai, model: gpt-4o}
```

Point `base_url` at a compatible deployment (Azure OpenAI, a proxy) if you are not
using the public API.

**Reasoning models** (`o1`/`o3`/`o4`-series, `gpt-5`-series) are detected automatically
and take `max_completion_tokens` instead of `max_tokens`; they also reject a custom
`temperature`, so the adapter drops it rather than failing the request. **They do not
return reasoning text** — the API bills reasoning tokens but exposes only the count. See
*Reasoning text* below for what that rules out.

### Anthropic (Claude)

```yaml
llm: {provider: anthropic, model: claude-sonnet-5}
```

Three things the adapter handles that a naive port would get wrong:

- **The thinking request shape differs by model generation.** Claude 4.6 and later take
  adaptive thinking (`{"type": "adaptive"}`) plus an effort level; the older fixed-budget
  form returns HTTP 400 on them. The adapter picks the right one from the model ID.
- **Current Claude models reject `temperature`, `top_p` and `top_k` outright.** The
  adapter omits them for those models, so a config with `temperature: 0.8` still works.
- **Reasoning text is opt-in.** The API defaults to omitting it; the adapter asks for
  summarized thinking so `reasoning_content` is actually populated. Without that, every
  reasoning-style trajectory would carry an empty `<think>` block.

Claude also returns thinking blocks carrying a `signature` that must be echoed back on
the next turn when tool use is in play. VULCAN's trajectory loop rebuilds each assistant
message from content plus tool calls, which would drop it — so the adapter keeps a
bounded cache of thinking blocks keyed by tool-call id and re-attaches them inbound.
Extended thinking with native tool calling therefore works without any change to the
pipeline.

### Gemini

```yaml
llm: {provider: gemini, model: gemini-2.5-pro}
```

Gemini diverges the most, and the adapter absorbs all of it:

- Messages become `contents` with roles `user`/`model`; the system prompt moves to
  `system_instruction`.
- **Function calls carry no id.** Gemini matches results back by function *name*.
  VULCAN mints deterministic ids so downstream code — which resolves `tool_call_id` to a
  tool name when building training records — keeps working.
- Function-call arguments are objects; VULCAN expects a JSON string. Both directions are
  converted.
- Declaration schemas accept a strict JSON Schema subset, so schemas are sanitised before
  being sent (`additionalProperties`, `$schema` and friends are stripped).
- Thought summaries are returned only when explicitly requested, which the adapter does
  when `reasoning` is set.

### Open-weight models (`openai_compatible`)

Talks plain HTTP to anything implementing `POST /v1/chat/completions` — vLLM, SGLang,
Ollama, llama.cpp, text-generation-inference, LM Studio, a LiteLLM proxy, or a hosted
OpenAI-compatible service. Deliberately built on `aiohttp` rather than a vendor SDK, so
running an open-weight model needs no optional dependency at all.

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-32B
```

**vLLM**

```bash
vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser deepseek_r1        # only for reasoning models
```

**SGLang**

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-32B \
  --tool-call-parser qwen25 --reasoning-parser deepseek-r1
```

**Ollama**

```bash
ollama serve                            # base_url: http://localhost:11434/v1
```

Two portability details the adapter handles: Ollama returns tool-call `arguments` as a
parsed object where OpenAI returns a JSON string (both are normalised to a string), and
it reports reasoning under `thinking` rather than `reasoning_content` (both are read).

## The two capability differences that actually matter

Everything else is absorbed by the adapters. These two change what you can generate.

### 1. Native tool calling → which `generate_trajectories.tool_call_format` works

`generate_trajectories.tool_call_format` selects how the agent emits tool calls:

| `tool_call_format` | Mechanism | Needs |
|---|---|---|
| `native` | native function calling | a provider that returns structured tool calls |
| `reasoning` | text tool calls from a reasoning model | reasoning text (see below) |
| `tagged` | `<tool_call>{...}</tool_call>` tags in plain text | nothing — works everywhere |

Hosted providers all support `native`. A self-hosted server does **not** unless it was
launched for it: vLLM and SGLang need `--enable-auto-tool-choice` with a model-matched
`--tool-call-parser`. Without those flags the model still emits tool calls, just as plain
text that never populates `tool_calls` — which VULCAN's `tagged` parser recovers
anyway. **`tagged` is the safe default for a self-hosted model**, and it is what
`configs/quickstart.yaml` uses.

### 2. Reasoning text → whether `tool_call_format: reasoning` produces anything

`tool_call_format: reasoning` writes the model's reasoning into the training records as
`<think>...</think>`. That requires a provider that returns reasoning *text*, not just a
token count:

| Provider | Returns reasoning text |
|---|---|
| `anthropic` | yes — summarized thinking, requested automatically |
| `gemini` | yes — thought summaries, when `reasoning` is set |
| `openai_compatible` | yes, when the server runs with a reasoning parser |
| `openai` | **no** — reasoning tokens are billed but never returned |

Configuring `tool_call_format: reasoning` against a public OpenAI reasoning model
produces records with empty `<think>` blocks. Use Claude, Gemini, or a self-hosted
reasoning model for that dataset variant — or use `tagged`, which does not depend on
reasoning at all.

Request reasoning with:

```yaml
llm:
  reasoning: {effort: medium}     # low | medium | high
  # reasoning: {budget_tokens: 4096}   # explicit budget where supported
```

The client keeps that as its default and applies it to every call that does not ask for
something else; a step may override it with `steps.<step>.reasoning`, and a step with
`thinking_model: true` and no `reasoning` anywhere falls back to `{"effort": "medium"}`.
`openai` sends it as `reasoning_effort`, `anthropic` as the `thinking` block for that model
generation, `gemini` as a thinking budget — each only when the model's `supports_reasoning`
capability is true. **`openai_compatible` never sends it**: there, reasoning is a server-side
launch flag (`--reasoning-parser`), and anything per-request goes through `extra_body`.

## Mixing models, and mixing providers

`resolve_llm_config(config, step_name)` overlays
`steps.<step>.{provider, model, base_url, reasoning, capabilities, extra_body}` on the
global block, and the CLI builds each stage's client with `create_client(config, stage)`.
Clients are cached per distinct `(provider, model, base_url, reasoning)`, so a run opens
one connection pool per backend rather than one per stage.

**Per-step `model:`** routes one stage to a different model on the same provider:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-32B

steps:
  generate_trajectories:
    tool_call_format: tagged
  judge_trajectories:
    model: Qwen/Qwen3-235B      # a bigger local model for the judge
```

**Per-step `provider:` / `base_url:`** switch the backend outright. Generate on a local
model, judge on a hosted one:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-32B

steps:
  # The expensive bulk stages stay local.
  generate_trajectories:
    tool_call_format: tagged

  # The judge gates the whole dataset — spend on it.
  judge_trajectories:
    provider: anthropic
    model: claude-sonnet-5

  # Code generation benefits most from a strong model.
  generate_tools:
    provider: openai
    model: gpt-4o
```

Each provider you name this way needs its own key in the environment, and its extra
installed (`pip install -e ".[anthropic]"` for the example above). The startup check only
builds a client from the global `llm:` block, so a missing key or extra for a step-level
provider surfaces as an `LLMError` when that stage starts — after the stages before it have
already run.

Do **not** set `defaults.model`. A step resolves `steps.<step>.model` → `defaults.model`
→ empty, and an empty value means "use whatever model this stage's client is configured
with". Setting `defaults.model` overrides `llm.model` for every stage, which in a
mixed-provider config sends one provider's model name to another.

`build_graph` is worth a thought here too: it costs *n·(n−1)* calls for *n* tools, so it is
often the largest single line item in a run and a good candidate for a cheaper model.

## Capability overrides

Capabilities are autodetected from the model ID
([`vulcan/llm/capabilities.py`](../vulcan/llm/capabilities.py)). Detection is a default,
never a rule — pin any field when you serve a model whose name matches nothing:

```yaml
llm:
  provider: openai_compatible
  model: my-org/my-finetune
  capabilities:
    supports_temperature: true
    supports_reasoning: true
    returns_reasoning_text: true
    supports_tools: true
    supports_stop: true
    uses_max_completion_tokens: false
```

| Field | Meaning |
|---|---|
| `uses_max_completion_tokens` | send `max_completion_tokens` instead of `max_tokens` |
| `supports_temperature` | model accepts a non-default `temperature` |
| `supports_reasoning` | model can be asked to reason |
| `returns_reasoning_text` | model returns the reasoning text, not just a token count |
| `supports_tools` | model supports native tool calling |
| `supports_stop` | model accepts stop sequences |
| `uses_adaptive_thinking` | Anthropic only: adaptive thinking rather than a fixed budget |

## Errors and retries

Every provider failure surfaces as `LLMError`
([`vulcan/core/exceptions.py`](../vulcan/core/exceptions.py)) carrying `status_code`,
`body` and `provider`. Retry policy lives in `LLMClient.call` and is identical across
providers: `max_retries` attempts with exponential backoff and jitter. Requests the
provider would reject identically next time — 400, 401, 403, 404, 422 — are **not**
retried; 408, 409, 425, 429 and 5xx are.

A dead key therefore fails the run loudly instead of quietly producing rows with no model
contribution. When a stage reports success but the data looks empty, check `llm_calls=`
in `<output_base>/logs/<category>/<step>.log`.

## Troubleshooting

**`no API key found for provider 'x'`** — set the variable from the table above, or put
it in `.env`. For `openai_compatible`, set `VULCAN_API_KEY=EMPTY` if your server wants a
non-empty value it will ignore.

**`the openai package is required for provider 'openai'`** — install the extra:
`pip install -e ".[openai]"`.

**`provider 'openai_compatible' requires llm.base_url`** — set `base_url` (or
`VULCAN_BASE_URL`) to your server, including the `/v1` suffix.

**Trajectories are empty and `tool_calls` never appears** — the server is not running
with tool-calling enabled. Either add the launch flags above, or set
`steps.generate_trajectories.tool_call_format: tagged`.

**`<think>` blocks are empty in the training records** — the provider is not returning
reasoning text. See the table in *Reasoning text* above.

**Requests time out on large generations** — `assemble_environment` and `debug_environment` allow up
to 65 000 output tokens. Raise `llm.request_timeout` (default 600 s) for slower backends.

**`unknown provider 'x'`** — `llm.provider` must be exactly one of `openai`, `anthropic`,
`gemini`, `openai_compatible`.

**`no model configured`** — neither `llm.model` nor `defaults.model` is set. `create_client`
raises before the first request rather than sending an empty model ID.
