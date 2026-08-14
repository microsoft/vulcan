# Contributing to VULCAN

This project welcomes contributions and suggestions.

## Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring
that you have the right to, and actually do, grant us the rights to use your contribution.
For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to
provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow
the instructions provided by the bot. You will only need to do this once across all repos
using our CLA.

## Code of Conduct

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.

## Development

Python 3.10 or later.

```bash
git clone https://github.com/microsoft/vulcan.git && cd vulcan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pytest` runs **596 tests in about a second, with no network access and no API key set** —
the provider layer is exercised against fakes rather than live endpoints. That is the inner
loop: change code, run `pytest`. Narrow it while iterating:

```bash
pytest tests/test_llm.py -q                      # provider layer only
pytest tests/test_config.py::TestLoadConfig -q   # one class
```

There is no configured formatter, linter, or CI workflow in this repository, so a green
`pytest` before you open the PR is the gate. Match the style of the file you are editing.

### Running the pipeline itself

The `dev` extra installs the test tooling, not a provider SDK. To actually run VULCAN you
also need the extra for the provider you intend to call, plus its key:

```bash
pip install -e ".[openai]"      # or [anthropic], [gemini], [all]
export OPENAI_API_KEY=sk-...    # see .env.example
vulcan --env quickstart --pipeline
```

> **VULCAN executes model-generated Python in the host process, with no sandbox.** That is
> how environment simulation works, not an oversight. Develop against it in a container, a
> VM, or a disposable CI worker — never on a workstation holding secrets. See the Security
> warning at the top of the [README](README.md).

### Where things live

| Path | Holds |
|---|---|
| `vulcan/steps/env_simulation/` | steps 1–7: spec → executable in-memory Python environment |
| `vulcan/steps/graph_construction/` | steps 8–10: tool dependency graph → multi-turn sequences |
| `vulcan/steps/task_generation/` | steps 11–12: execute sequences, generate queries |
| `vulcan/steps/trajectory/` | steps 13–14: generate and judge conversations |
| `vulcan/steps/output/` | step 15: filter and emit training records |
| `vulcan/steps/__init__.py` | `STEP_REGISTRY`, `PIPELINE_ORDER`, `SUBCATEGORIES` |
| `vulcan/core/stage_runner.py` | `_STANDARD_INPUT` — the authoritative input resolution |
| `vulcan/llm/` | `LLMClient.call()` and the four provider adapters |
| `vulcan/prompts/` | prompt templates only — 12 modules: one per step that has any (11 of them; `normalize_specs` keeps its prompts inline, and `plan_sequences` / `execute_sequences` have none), plus `training_format.py`, the system-prompt templates `export_dataset` rebuilds training records with |
| `vulcan/config/base.yaml` | packaged defaults for every step |
| `tests/` | the suite above |

Each step subclasses `BaseStep`, is registered with `@register_step`, and implements
`async run(item, progress)`. Steps do **not** describe their own predecessors — ordering and
input resolution belong to the stage runner.

**Adding a step:** [`docs/adding_new_steps.md`](docs/adding_new_steps.md) walks the full
checklist — writing the class, importing it so the decorator runs, placing it in
`PIPELINE_ORDER`, telling `StageRunner` where its input comes from, adding defaults to
`base.yaml`, optional validation, and tests.

### Two house rules that are easy to trip over

- **No pipeline step may branch on provider.** Provider-specific behaviour belongs in an
  adapter under `vulcan/llm/`, or in `vulcan/llm/capabilities.py` for per-model differences
  — never in `vulcan/steps/`. Every adapter returns the same OpenAI-shaped response.
- **Never set `defaults.model`, and never hardcode a fallback model ID in a step.** A step
  resolves `steps.<step>.model` → `defaults.model` → `""`, where `""` means "use the model
  this stage's client was built with". Setting it overrides `llm.model` for every stage,
  which in a mixed-provider config sends one vendor's model name to another.

### Documentation changes

Pages under `docs/` are expected to be **verified against the code**, not written from
memory. Several carry a "sharp edges" or "things that surprise people" section listing real,
current defects. If your change fixes one, delete the entry rather than softening it. If you
conclude the behaviour is correct as-is, move it to a "things that surprise people" section
with the reasoning, rather than deleting it silently.

## Filing issues

- **Bugs and feature requests** → [Issues](https://github.com/microsoft/vulcan/issues).
  Search first to avoid duplicates. A bug report is much easier to act on when it includes
  the config you ran (with the API key removed), the exact command, the provider and model,
  the stage that failed, and the tail of `<output_base>/logs/<category>/<step>.log`.
  A stage can report success while every model call failed, so if the output looks empty
  rather than wrong, check the `llm_calls=` count in that log first — it is the honest
  signal.
- **Questions about using VULCAN** →
  [Discussions](https://github.com/microsoft/vulcan/discussions). Most sharp edges are
  already written down in [`docs/`](docs/) — per-stage under ⚠ in
  [`pipeline_stages.md`](docs/pipeline_stages.md), cross-cutting in
  [`docs/README.md`](docs/README.md#things-that-surprise-people).
- **Security vulnerabilities** → MSRC, per [SECURITY.md](SECURITY.md). **Never** through a
  public GitHub issue. Note that VULCAN executing model-generated code without a sandbox is
  documented, intentional behaviour rather than a vulnerability — see the
  [README](README.md).
