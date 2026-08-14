# Example input catalogs

Three original API catalogs, written for this repository, so VULCAN runs end to end
without downloading anything. Each file is **JSONL** — one JSON object per line, one
line per category. The schema is documented in [`../docs/input_format.md`](../docs/input_format.md).

| File | Categories | Tools | `environment_type` | Exercises |
|---|---:|---:|---|---|
| `ticket_api.jsonl` | 1 | 8 | `tools_only` | the smallest useful run — start here |
| `hotel_booking.jsonl` | 1 | 10 | `with_user_and_policy` | simulated user + domain policy |
| `ecommerce.jsonl` | 1 | 13 | `tools_only` | longer dependency chains, deeper graphs |
| `all_examples.jsonl` | 3 | 31 | mixed | all three at once |

Point a config at one of them:

```yaml
input:
  catalog: examples/ticket_api.jsonl
categories:
  - ticket_api
```

Then run:

```bash
vulcan --env quickstart --pipeline
```

## Why these three

They are chosen to cover the two environment types a shipped catalog can exercise —
`tools_only` and `with_user_and_policy` — and to produce dependency graphs of different
shapes:

- **`ticket_api`** is deliberately small. `build_graph` costs *n·(n−1)* model calls for
  *n* tools, so 8 tools is 56 calls — cheap enough to run the whole pipeline while you
  are still learning it.
- **`hotel_booking`** carries `policy_rules` inline, which promotes it to `with_user_and_policy`
  and switches on the simulated-user path plus policy-compliance judging. It is the
  one to copy when your domain has rules an agent must follow.
- **`ecommerce`** has the longest natural chains (search → cart → coupon → shipping →
  order → track → return), which is what produces interesting multi-turn sequences and
  multipath subgraphs.

No shipped catalog uses `with_user` — a simulated user without a policy. To get one, set
`environment_type: with_user` and leave `policy_rules` off every category; see
[`../docs/environment_types.md`](../docs/environment_types.md).

Every tool ships a `response` schema. That is optional in the input format, but supplying
it means `normalize_specs` does not have to invent one, and the generated environment
code is markedly better as a result.

## Writing your own

Copy the shape of any file here and read [`../docs/input_format.md`](../docs/input_format.md)
for the full field reference. Two things matter more than the rest:

1. **The `description` must match the `name`.** `normalize_specs` rewrites descriptions
   that disagree with their function name, and a vague description propagates into every
   later stage.
2. **Use `constraints` for behaviour the schema cannot express** — ordering requirements,
   state preconditions, validation rules. `generate_tools` switches to a constraint-aware
   prompt for each API that carries one, which produces environments that reject invalid
   calls instead of silently accepting them. It is per tool, so a catalog can mix
   constrained and unconstrained tools freely.

## Using a public benchmark instead

VULCAN takes any catalog in this format, so public tool-calling benchmarks (BFCL,
AppWorld, τ²-bench, VitaBench, and others) can be converted and used as input. This
repository ships no third-party data — download it from its own source, under its own
licence, and convert it to the format above.
