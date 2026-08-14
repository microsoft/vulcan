# Environment types

`EnvironmentType` ([`vulcan/environment/types.py`](../vulcan/environment/types.py)) is the one
switch that decides whether a simulated user drives the conversation and whether domain policy
is enforced. It changes which prompts `generate_queries` and `generate_trajectories` use, which judge
`judge_trajectories` picks, and what the resulting trajectories look like. Nothing else in the
pipeline branches on it.

## The three types

| `environment_type` | Simulated user | Policy | Enum member |
|---|:---:|:---:|---|
| `tools_only` | no | no | `EnvironmentType.TOOLS_ONLY` |
| `with_user_and_policy` | yes | yes | `EnvironmentType.WITH_USER_AND_POLICY` |
| `with_user` | yes | no | `EnvironmentType.WITH_USER` |

### `tools_only`

The assistant answers a fixed query by calling tools. There is no second agent.

1. The system prompt carries the tool schemas.
2. `generate_queries` produced a **list** of queries, one per sequence turn; each is sent as its
   own user message.
3. The assistant calls tools against the live environment until it stops calling them.
4. The turn ends on a text-only reply, or when `max_tool_calls_per_turn` is reached.

Use it when the interesting behaviour is tool selection and argument construction, not
conversation. It is also the cheapest of the three — one model per item instead of two.

### `with_user_and_policy`

A simulated user drives the conversation and the assistant works under written domain rules.

1. `generate_queries` produced a **scenario dict**; the user simulator is seeded with it and
   speaks first.
2. The assistant's system prompt embeds the category's `policy_rules`.
3. `transfer_to_human_agents` is appended to the assistant's tool list.
4. The two agents alternate: a turn is either tool calls or a message to the user, never both.
5. The run ends when the user emits `###STOP###`, when the assistant calls a terminate tool, or
   at the turn cap.
6. `judge_trajectories` judges with the policy-aware judge, so policy violations lower `score`.

Use it when your domain has rules an agent must follow — eligibility checks, confirmation
before a mutation, refusal conditions.

### `with_user`

Same simulated user, no policy text, and **no** transfer tool. Either side may emit `###STOP###`
to end the conversation. Use it for open-ended assistant behaviour where there is no rulebook to
comply with.

No shipped example catalog uses this type. To get one, set
`environment_type: with_user` and leave `policy_rules` off every category — a
category that carries `policy_rules` is promoted to `with_user_and_policy` regardless (see below).

## The shipped catalogs

Three original catalogs in [`../examples/`](../examples/README.md) cover the types you can run
today:

| Catalog | Category | Tools | Type | Why it is there |
|---|---|---:|---|---|
| `ticket_api.jsonl` | `ticket_api` | 8 | `tools_only` | smallest useful run; `build_graph` costs 8·7 = 56 model calls |
| `hotel_booking.jsonl` | `hotel_booking` | 10 | `with_user_and_policy` (policy inline) | simulated user + policy compliance |
| `ecommerce.jsonl` | `ecommerce` | 13 | `tools_only` | longest dependency chains, deepest graphs |
| `all_examples.jsonl` | all three | 31 | mixed | one run over all of them |

Each category line sets `environment_type` explicitly, and `hotel_booking` carries its
`policy_rules` **inline on the category record** — a policy in a separate file is never loaded.
Matching configs: [`../configs/quickstart.yaml`](../configs/quickstart.yaml) (`tools_only`) and
[`../configs/hotel_policy.yaml`](../configs/hotel_policy.yaml) (`with_user_and_policy`).

## Configuration

### Explicit (preferred)

```yaml
environment_type: with_user_and_policy
```

### Boolean fallback

```yaml
has_user_agent: true
has_policy: true          # → with_user_and_policy
```

`EnvironmentType.from_config(config)`:

1. `config["environment_type"]`, if present. An unrecognised value **raises `ValueError`**
   listing the three valid strings — it does not fall back.
2. Otherwise `has_user_agent` + `has_policy`: both ⇒ `WITH_USER_AND_POLICY`, user only ⇒
   `WITH_USER`.
3. Otherwise `TOOLS_ONLY`.

Note that step 2 is gated on `has_user_agent`: `has_policy: true` on its own yields
`TOOLS_ONLY`, and the policy is never shown to anyone. Prefer the explicit key.

### Per-category override

The environment type is a *default*. The type a step actually uses is per category, resolved by
`EnvironmentConfig.get_environment_type(category)` in this order:

1. **`environment_type_map[category]`** — from the config, or from an `environment_type` field on
   the category record in the input catalog (the input record wins, because it is merged into the
   same map at load time).
2. **The category has `policy_rules`** ⇒ `"with_user_and_policy"`, *even when the environment-level
   type is `TOOLS_ONLY`*.
3. The environment-level `EnvironmentType`.

```yaml
environment_type: tools_only
environment_type_map:
  ticket_api: tools_only
  hotel_booking: with_user_and_policy
```

⚠ **Rule 2 is the one that surprises people.** Adding a `policy_rules` string to one category of
an otherwise `tools_only` catalog silently switches that category to the simulated-user path,
doubling its model calls and changing the shape of `query` from a list to a dict. That is usually
what you want — but it happens without touching `environment_type`.

⚠ Loading the catalog also flips `EnvironmentConfig.has_policy` to true when any category has
`policy_rules`, and `has_constraints` to true when any API has `constraints`. Those attributes
are informational: **no pipeline step reads them.** The steps only call `get_environment_type`,
`get_api_catalog`, `get_domain_rules`, `get_terminate_tools`, `get_categories`, and
`TRANSFER_TO_HUMAN_API`. `get_constraints` / `get_api_constraints` are part of the public API but
currently unused — `generate_tools` reads the per-item `constraints` that `normalize_specs`
copied instead.

## Properties

```python
from vulcan.environment.types import EnvironmentType

EnvironmentType.TOOLS_ONLY.has_user_agent            # False
EnvironmentType.TOOLS_ONLY.has_policy                # False

EnvironmentType.WITH_USER_AND_POLICY.has_user_agent  # True
EnvironmentType.WITH_USER_AND_POLICY.has_policy      # True

EnvironmentType.WITH_USER.has_user_agent             # True
EnvironmentType.WITH_USER.has_policy                 # False
```

The enum subclasses `str`, so `EnvironmentType.TOOLS_ONLY == "tools_only"` is true and the
value can be written straight into YAML or JSON. There is also an `.environment_type` property,
which returns the same string as `.value`; steps compare against the plain strings that
`get_environment_type` returns.

## Impact on pipeline stages

| Stage | `tools_only` | `with_user_and_policy` | `with_user` |
|---|---|---|---|
| `normalize_specs` | — | — | — |
| `generate_tools` | constraint-aware prompts iff the **API** has `constraints` | same | same |
| `build_graph` … `execute_sequences` | identical | identical | identical |
| `generate_queries` | fixed queries — `query` is a **list** of `{"query": …}`, one per turn | scenario + `policy_rules` + catalog tools — `query` is a **dict** with `scenario` | scenario without policy — `query` is a **dict** with `scenario` |
| `generate_trajectories` | assistant only | assistant + user simulator, policy in the system prompt, `transfer_to_human_agents` added | assistant + user simulator, no policy, **no** transfer tool |
| `judge_trajectories` | tools-only judge | policy-aware judge (scores compliance) | no-policy judge; the `reasoning` tool-call format reuses the **tools-only** judge prompts |
| `export_dataset` | same criteria for all three: `score >= min_score` (8) **and** `task_completed` when `require_task_completed` (true) | ← | ← |

Phase 1 and Phase 2 are entirely type-independent: the environment class, its initial states, the
dependency graph, and the executed chains are the same whichever type you pick. Only the last
three stages differ.

⚠ **`policy_rules` does not reach `generate_tools`.** The runner puts it on the category record it
writes for `normalize_specs`, but that step's `preprocess_input` keeps only `category_name`,
`function`, and `constraints`. If you want the generated tools to *enforce* a rule, express it as
per-API `constraints`; `policy_rules` governs the conversation, not the simulation.

`policy_rules` **does** reach the training record. `export_dataset` rebuilds the system prompt
from `api_list`, dropping the trajectory's own system message, then re-injects the category's
policy for a `with_user_and_policy` record: through `reconstruct_policy_fc_json` / `_xml` for
`native`, and through `prepend_policy()` for `reasoning` and `tagged`, whose templates have no
`{domain_rules}` slot. See the `export_dataset` notes in
[`pipeline_stages.md`](pipeline_stages.md).

## Terminate tools

`terminate_tools` is a top-level config key, defaulting to `["transfer_to_human_agents"]`. A tool
call whose name is in that list ends the trajectory immediately instead of being executed.

Every shipped config sets `terminate_tools: []`, because no example catalog defines a
hand-off tool.

⚠ **With `terminate_tools: []` in a `with_user_and_policy` environment, `transfer_to_human_agents`
is still offered to the assistant** — `generate_trajectories` appends it to the tool list for that
type — but calling it is no longer a terminator, so the call is executed as an ordinary tool, fails
with `Unknown tool 'transfer_to_human_agents'`, and the error JSON is fed back to the model. Either
leave `terminate_tools` at its default for policy domains, or accept the error turn.

⚠ **In `with_user` the assistant is not given the transfer tool, but the judge is told about
it** — `judge_trajectories._get_api_specs_str` appends `TRANSFER_TO_HUMAN_API` to the spec list for
both simulated-user types, and that list is what the `native`-format judge sees. The judge can
therefore fault the assistant for not escalating with a tool it never had.

## See also

- [`pipeline_stages.md`](pipeline_stages.md) — what each stage does, field by field
- [`configuration.md`](configuration.md) — every config key and how the layers merge
- [`../examples/README.md`](../examples/README.md) — the shipped catalogs and the input schema
- [`../README.md`](../README.md) — install, providers, and the Security section
