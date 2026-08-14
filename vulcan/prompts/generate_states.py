"""Prompts for the generate_states step (initial state generation).

Two-phase process:
1. Sample agent: analyze code → produce template initial state + function test vectors
2. Init state agent: generate N diverse variations of the template state
"""

SAMPLE_STATE_SYSTEM = """Analyze a Python environment class and generate a realistic initial state template.

## Task

1. Read the environment code carefully — identify every key that `self.state` reads from or writes to
2. Determine the correct type and realistic default value for each key
3. Produce a complete initial state that would make ALL tools work correctly

## Rules

- Every key accessed via `self.state["key"]` or `self.state.get("key")` MUST be present
- Types must match what the code expects (don't initialize a list where code expects a dict)
- Values must be realistic for the domain (real city names, reasonable fuel levels, valid user IDs)
- Nested structures must be complete — if code accesses `self.state["vehicle"]["fuel"]["level"]`, the full path must exist
- Include lookup tables, configuration constants, and reference data the code uses
- For fields that are lists/arrays: MUST contain at least {MIN_LIST_VALUES} realistic items — no empty lists
- No field should be empty: no empty strings "", no null values. Every field must have a meaningful, realistic value

## Output

Return valid JSON inside <test_vectors> tags:

<test_vectors>
{
  "initial_state": {
    "key1": "realistic_default_value",
    "key2": {"nested": "structure"},
    ...
  },
  "functions": [
    {
      "name": "method_name",
      "cases": [
        {"input": {"param1": "value"}, "output": {"field": "value"}}
      ]
    }
  ]
}
</test_vectors>

IMPORTANT: The initial_state must be a complete, self-consistent dictionary. Do NOT return partial states or placeholder values like "TODO" or "..."."""

SAMPLE_STATE_USER = """Analyze this environment class and generate a complete initial state template with function test vectors.

<code>
{PY_CODE}
</code>"""

INIT_STATE_SYSTEM = """Generate a realistic initial state for an environment by reading its code.

## Task

Read the environment code carefully and generate ONE complete initial state with realistic, varied values for ALL fields.

## Rules

1. **Complete structure**: Include EVERY key that `self.state` reads from or writes to. If code accesses `self.state["x"]["y"]`, the full path must exist.
2. **Types correct**: Types must match what the code expects (don't use a list where code expects a dict, don't use int where code expects float).
3. **Realistic varied values**: Every field must have a meaningful, realistic value:
   - Numeric values: use realistic values within domain ranges (not always 0 or defaults)
   - String values: use realistic alternatives (real names, cities, status values)
   - Boolean values: choose true or false realistically
   - Lists/arrays: MUST contain at least {MIN_LIST_VALUES} realistic items with varied content — no empty lists
   - Lookup tables and configuration: keep internally consistent
4. **No empty values**: No empty strings "", no null values. Every field must have content.
5. **Cross-key consistency**: Related fields must be consistent:
   - If engine is "running", fuel should be > 0
   - If a user is "logged_in", auth tokens should be present

## Output

Return the state as a JSON object inside <initial_states> tags:

<initial_states>
{"key1": "value1", "key2": {...}, ...}
</initial_states>
</initial_states>

IMPORTANT: Return ONLY the JSON array. No markdown fences, no explanation text."""

INIT_STATE_USER = """Generate one complete initial state for this environment. Read the code to identify all required state keys and produce realistic values.

<code>
{PY_CODE}
</code>"""

JSON_REPAIR_SYSTEM = """Fix syntax errors in a JSON document. Common issues: trailing commas, missing quotes, unescaped characters, truncated output.

Return ONLY the corrected JSON inside tags. Do not add explanation.

<repaired_json>corrected JSON here</repaired_json>"""

JSON_REPAIR_USER = """Fix this malformed JSON:

<broken_json>
{BROKEN_JSON}
</broken_json>"""
