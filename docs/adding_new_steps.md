# Adding a new step

A step is one class. It takes one item, returns one dict, and the runner wraps parallelism,
retry, incremental saving and resume around it. Everything else — where the input comes
from, where the output goes, which categories run — is declared outside the class.

The contract, from [`vulcan/steps/base.py`](../vulcan/steps/base.py):

1. decorate the class with `@register_step("<name>")`;
2. call `super().__init__(config, llm, env)` — three positional arguments, in that order;
3. override `async def run(self, item: dict, progress=None) -> dict`.

Name the class after the step: the registered name in CamelCase, plus a `Step` suffix.
`normalize_specs` → `NormalizeSpecsStep`, `plan_sequences` → `PlanSequencesStep`,
`judge_trajectories` → `JudgeTrajectoriesStep` — all fifteen shipped steps follow it. Nothing
enforces the convention (`register_step` sets `cls.name` from its argument, and the registry is
keyed by that string), but breaking it makes a class impossible to find from a log line.

`llm` is an [`LLMClient`](../vulcan/llm/base.py). `vulcan.cli` builds one per stage with
`create_client(config, stage)`, so `steps.<step>.{provider, model, base_url, reasoning,
capabilities, extra_body}` all take effect; clients are cached per distinct
`(provider, model, base_url, reasoning)`, so stages that resolve to the same backend share
one. `.call()` returns the same OpenAI-shaped response whether GPT, Claude, Gemini or a
self-hosted server answered it, so a step never branches on provider. Steps hold it as
`self.llm` and normally pass it straight to an `Agent`; per-provider setup lives in
[`providers.md`](providers.md).

## 1. Write the class

Put the file in the subpackage for its phase: `env_simulation/`, `graph_construction/`,
`task_generation/`, `trajectory/`, or `output/`.
[`env_simulation/normalize_specs.py`](../vulcan/steps/env_simulation/normalize_specs.py)
is the shortest real example; this is the same shape.

```python
# vulcan/steps/env_simulation/summarize_specs.py
"""Step: summarize_specs — one line on what it adds to each item."""

from __future__ import annotations

import json

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError

SUMMARY_SYSTEM = "You are …"

SUMMARY_USER = """Summarise this API spec in one sentence.

<api_spec>
{API_SPEC}
</api_spec>

Return the result inside <summary> tags."""


@register_step("summarize_specs")
class SummarizeSpecsStep(BaseStep):
    """Adds ``spec_summary`` to each API item."""

    requires_llm = True          # False for a deterministic step

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.agent = Agent(
            "summarizer", SUMMARY_SYSTEM, llm,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_model=self.thinking_model,
            output_parser=TagParser.make_parser("summary"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            spec = inp["function"]
            prompt = SUMMARY_USER.format(API_SPEC=json.dumps(spec, indent=2))
            res = await self.agent.run([{"role": "user", "content": prompt}], progress)

            parsed = TagParser.extract_first("summary", res["content"])
            if parsed:
                inp["spec_summary"] = parsed
        except (StepError, KeyError):
            raise

        return inp
```

`Agent(name, system_prompt, llm, …)` takes those three positionally; everything after is
keyword-only (`model`, `temperature`, `max_tokens`, `thinking_model`, `output_parser`).
`agent.run()` returns a dict with `content`, `reasoning_content`, `parsed_content`,
`finish_reason`, `usage`, `messages`, plus `tool_calls` when the model requested one.

Three habits worth copying from the shipped steps:

- **Pass `progress` into every `agent.run`.** It is the tqdm bar labelled *API calls*; a
  step that drops it looks stalled.
- **Keep agents as attributes, not locals.** `set_logger` and
  `_propagate_config_to_agents` walk the step's attributes and inject the logger, the step
  name and `empty_response_retries` into anything that looks like an `Agent`. An agent
  held only in a local gets none of that: its calls never appear in `llm_calls=`, and it
  falls back to the `Agent` default of `0` empty-response retries instead of your
  configured budget — a data problem, not just a logging one.
- **When the agent cannot be built in `__init__`, use `self.make_agent(...)`.** Some
  agents need a system prompt that depends on the item, so they have to be built inside
  `run()`. `BaseStep.make_agent(name, system_prompt, …)` constructs the `Agent` and wires
  the logger, step name, category and `empty_response_retries` immediately, so a local
  agent behaves like an attribute one. Omitted keyword arguments default to the step's
  resolved config. Never construct `Agent` directly inside `run()`.

### What `BaseStep.__init__` sets up

| Attribute | Value |
|---|---|
| `self.config` | the whole merged config dict |
| `self.llm` | the `LLMClient` for this run |
| `self.env` | the `EnvironmentConfig` — `get_api_catalog`, `get_domain_rules`, `get_environment_type`, … |
| `self.step_config` | `config["steps"]["summarize_specs"]`, or `{}` |
| `self.model` | `steps.<name>.model` → `defaults.model` → `""` |
| `self.temperature` | `steps.<name>` → `defaults` → `0` |
| `self.max_tokens` | `steps.<name>` → `defaults` → `16000` |
| `self.thinking_model` | `steps.<name>` → `defaults` → `False` |
| `self.empty_response_retries` | `steps.<name>` → `defaults` → `10` |
| `self.logger`, `self.out_dir` | set by the runner before the first item |

`self.model` of `""` is deliberate: it means *whatever the client was constructed with*
(`llm.model`), and the client raises a clear `LLMError` when nothing is configured at all.

### Reshaping the input first

Override `preprocess_input(data) -> list[dict]` when the file on disk is not one row per
unit of work. `BaseStep` declares it as a `@staticmethod`, and `PipelineRunner` calls it as
`step.preprocess_input(data)`, so an override may be either static or an ordinary method
(`generate_states` uses an instance method to read `num_states`). `normalize_specs` uses it
to flatten a category record
(`{"category_name": …, "api_list": [...]}`) into one item per API; `generate_states` uses it
to explode one item into many. The default passes data through untouched.

## 2. Import it so the decorator runs

`@register_step` only fires when the module is imported. Add the class to its subpackage's
`__init__.py`:

```python
# vulcan/steps/env_simulation/__init__.py
from .summarize_specs import SummarizeSpecsStep

__all__ = [..., "SummarizeSpecsStep"]
```

The bottom of [`vulcan/steps/__init__.py`](../vulcan/steps/__init__.py) imports the five
subpackages, which is what populates `STEP_REGISTRY`.

## 3. Put it in the pipeline order

Same file — `PIPELINE_ORDER` is the sequence `--pipeline` walks, and `--start` / `--end`
index into it. Add the name to `SUBCATEGORIES` too, so the step is grouped with its phase:

```python
PIPELINE_ORDER = [
    "normalize_specs",
    "generate_tools",
    "summarize_specs",      # in the position it must run
    "verify_tools",
    ...
]

SUBCATEGORIES = {
    "env_simulation": [..., "summarize_specs", ...],
}
```

`vulcan --list-steps` prints `PIPELINE_ORDER` with a tick per registered step, so a name in
the order but not in the registry shows up as `✗` there.

## 4. Tell `StageRunner` where its input comes from

Steps do not describe their own predecessors. Ordinary input→output stages get one line in
`_STANDARD_INPUT` in [`core/stage_runner.py`](../vulcan/core/stage_runner.py), a
`(previous_step_dir, filename)` tuple:

```python
_STANDARD_INPUT = {
    ...
    "summarize_specs": ("normalize_specs", "success.jsonl"),
}
```

The step then reads `<output_base>/<category>/normalize_specs/success.jsonl` and writes
`<output_base>/<category>/summarize_specs/success.jsonl` (plus `failed.jsonl`). A missing input
file is logged as `input not found` and the category is skipped.

Anything less regular gets a dedicated method on `StageRunner` instead of a
`_STANDARD_INPUT` entry. The existing ones are worth reading before you invent a sixth
shape: `_run_normalize_specs` (builds its input from the catalog rather than from a
previous step), `_run_plan_sequences` and `_run_execute_sequences` (deterministic `run_batch`
stages with non-standard output filenames), `_run_generate_trajectories` / `_run_judge_trajectories`
(paths keyed by iteration and output label), and `_run_export_dataset` (runs once across
every category). `_build_env_grouped` shows the other pattern: a regrouping pass the runner
performs itself, with no step class at all.

## 5. Add defaults to `base.yaml`

```yaml
# vulcan/config/base.yaml
steps:
  summarize_specs:
    temperature: 0
    max_tokens: 16000
```

Only set `model:` here when the step genuinely needs a different one from the rest of the
run. Leaving it out means the step uses whatever model its client was built with
(`llm.model`), which keeps it correct on every provider; a `model:` you do write must be a
valid ID for the provider that serves this step. To put the step on a different backend
entirely, set `provider:` / `base_url:` beside it — `vulcan.cli` builds each stage's client
with `create_client(config, stage)`.

## 6. Deterministic validation (optional)

`BaseStep.validate_output(item)` returns a **`(passed, reason)` tuple** and is what decides
whether an item lands in `success.jsonl` or `failed.jsonl`. It checks two things, in order:

1. the `VALIDATORS` registry in
   [`validators/deterministic.py`](../vulcan/validators/deterministic.py), keyed by step
   name;
2. `output_schema`, if the class sets one to a Pydantic model
   (see [`vulcan/schemas/step_io.py`](../vulcan/schemas/step_io.py)).

```python
# vulcan/validators/deterministic.py
def validate_summarize_specs(item: dict) -> tuple[bool, str]:
    if not item.get("spec_summary"):
        return False, "missing spec_summary"
    return True, ""

VALIDATORS = {
    ...
    "summarize_specs": validate_summarize_specs,
}
```

With neither hook, every item that does not raise counts as a success. Failed items are
retried up to `defaults.max_retries` before the batch gives up — except in `generate_trajectories` and
`judge_trajectories`, which `StageRunner` deliberately constructs with zero retries.

## 7. LLM verification (optional)

A second, model-based gate that runs *after* `validate_output` passes. Add criteria in
[`verification/criteria.py`](../vulcan/verification/criteria.py) and list the step under
`verification.verify_steps`:

```python
VERIFICATION_CRITERIA = {
    ...
    "summarize_specs": {
        "check_items": [
            "spec_summary is present and non-empty",
            "spec_summary is consistent with the input specification",
            "No information appears that is absent from the input",
        ],
    },
}
```

```yaml
verification:
  enabled: true
  verify_steps: [summarize_specs]
```

Know what you are turning on: `StageVerifier._extract_relevant_fields` sends only the
fields listed for the step (a step with no entry falls back to the item's **first ten
keys**), truncates any string over **3000 characters**, and caps the verdict at
`verification.max_tokens` (default **512**). Its rejections are unreliable on large items,
which is why `verification.enabled` is `false` everywhere in the shipped configs.

## 8. Tests

```python
# tests/test_summarize_specs.py
from vulcan.steps import STEP_REGISTRY, PIPELINE_ORDER


def test_step_is_registered():
    assert "summarize_specs" in STEP_REGISTRY
    assert "summarize_specs" in PIPELINE_ORDER


def test_validate_output_returns_tuple():
    step = STEP_REGISTRY["summarize_specs"](config={}, llm=None, env=None)
    assert step.validate_output({"spec_summary": "ok"}) == (True, "")
    passed, reason = step.validate_output({"spec_summary": ""})
    assert not passed and reason
```

Constructing the class with `llm=None` is fine as long as `__init__` only stores the
client — `Agent` does not touch it until the first call. Run the suite with
`pip install -e ".[dev]" && pytest`.

## Deterministic steps

A step that never calls a model sets `requires_llm = False` and implements `run_batch`
instead of `run`, because it needs all the items at once:

```python
@register_step("filter_sequences")
class FilterSequencesStep(BaseStep):
    requires_llm = False

    def run_batch(self, input_path: str, output_dir: str) -> None:
        items = load_jsonl(input_path)
        save_jsonl(self._process_all(items), os.path.join(output_dir, "success.jsonl"))
```

`run_batch` is not part of the `BaseStep` contract and `PipelineRunner` never calls it — a
deterministic step needs a dedicated `StageRunner` method that calls it directly, the way
`_run_plan_sequences` and `_run_execute_sequences` do. `requires_llm` is only read by
`vulcan --list-steps` to label the step; the runner still constructs it with a client.

## Sharp edges

**A broad `except Exception` turns a dead provider into silent success.** Most shipped
steps end with `except (StepError, KeyError): raise` / `except Exception:
traceback.print_exc()`, which catches `LLMError` too. The item is then returned unmodified,
passes `validate_output` if the step has no validator, and is written to `success.jsonl`
with no model contribution. Either let `LLMError` propagate — the worker records the item as
failed with the traceback as its reason — or register a validator that rejects an item
missing the field your step is supposed to add. The honest signal after a run is `llm_calls=`
in `<output_base>/logs/<category>/<step>.log`.

**`validate_output` must return a tuple.** Returning a bare `bool` raises inside the worker
when it unpacks `passed, reason`; the item is caught by the worker's blanket handler and
recorded as failed with a `TypeError` traceback, so the step appears to fail on every item.

**Items resume by input hash.** `_item_id` is an MD5 of the item as loaded, so a completed
item is skipped on the next run — changing a model, a temperature or a prompt does **not**
invalidate it. Delete the step's output directory to force regeneration.

**Anything you execute runs unsandboxed.** `generate_tools`, `verify_tools`,
`assemble_environment`, `debug_environment`, `execute_sequences` and `generate_trajectories` all run model-written Python
in the host process through [`core/code_exec.py`](../vulcan/core/code_exec.py) and
[`core/tool_exec.py`](../vulcan/core/tool_exec.py). A new step that does the same inherits
that trust model: read the Security section of the [top-level README](../README.md) before
adding one, and run the pipeline in a disposable environment.
