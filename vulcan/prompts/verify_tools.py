"""Prompts for the verify_tools step.

Verifies generated code against the same 5 criteria used in generation:
1. Tool abstract pattern (invoke + get_info)
2. Minimal error handling with meaningful messages
3. External data detection (data param if needed)
4. All parameters used (required vs optional)
5. Code is executable
"""

VERIFY_SYSTEM = """You are a code verifier. Validate a Python Tool class against its JSON API definition.

## Criteria (ALL must pass)

### 1. Abstract pattern
- Class subclasses `Tool` with static `invoke()` and `get_info()` methods.
- `invoke()` returns a JSON string via `json.dumps()`.
- `get_info()` returns `{"type": "function", "function": {...}}`.

### 2. External data consistency
- If the API needs external state (database lookups, auth tokens, stored records), `invoke` MUST have `data: Dict[str, Any]` as its first parameter and use it.
- If the API does NOT need external state, `invoke` must NOT have a `data` parameter.
- The `data` parameter is internal — it must NOT appear in `get_info`.

### 3. Parameter completeness
- Every parameter in the JSON `parameters.properties` MUST appear in `invoke()` signature.
- Every `invoke()` parameter (except `self` and `data`) MUST be meaningfully used in the function body — declared but unused is a PROBLEM.
- Required parameters (from JSON `required` list) must not have default values.
- Optional parameters should have appropriate defaults.

### 4. Type and return consistency
- Parameter types in `invoke` must match the JSON spec types.
- Return value structure must match the JSON `response` section (skip if no response defined).
- Only update `get_info` types if you changed them in `invoke`.

### 5. Error handling
- try/except blocks must be MINIMAL — only around operations that can genuinely fail at runtime (dict key access, type conversion, division).
- Each except must return `json.dumps({"error": "meaningful message"})`.
- There must NOT be a single blanket try/except wrapping the whole function.

### 6. Executability
- Code must be syntactically valid Python 3.11+.
- Only standard library imports.
- Must execute without errors when loaded.

If ALL checks pass, output FINE.
If ANY check fails, fix the code and return the COMPLETE corrected class.

<decision>FINE | PROBLEM: [which criteria failed and why]</decision>
<modified_code>None | [complete corrected Python code including imports and Tool base class]</modified_code>"""

VERIFY_USER = """Validate this code against its API definition using all 6 criteria.

<code>
{PY_CODE}
</code>

<definition>
{INP_DICT}
</definition>"""

UNUSED_PARAMS_USER = """Validate this code against its API definition.

<code>
{PY_CODE}
</code>

<definition>
{INP_DICT}
</definition>

CRITICAL: These invoke() parameters are declared but NEVER used in the function body: [{UNUSED_PARAMS}].
This violates criteria #3 (parameter completeness). You MUST rewrite invoke() so that every parameter contributes meaningfully to the logic or output — not just stored in a variable that is never read.

<decision>PROBLEM: unused parameters [{UNUSED_PARAMS}]</decision>
<modified_code>[complete corrected code where all parameters are used]</modified_code>"""

CHECK_NAME_DESC_SYSTEM = ""

CHECK_NAME_DESC_USER = """Evaluate consistency between the Python code's actual behavior and the JSON definition's name and description fields.

<code>
{PY_CODE}
</code>

<definition>
{INP_DICT}
</definition>

Rules:
- If the `data` dictionary is used in invoke but not in the JSON definition, ignore it — this is internal state.
- Only evaluate whether the code's behavior matches what the name and description promise.
- If default values are not explicitly mentioned in the description, do not check them.

If consistent: respond PASS
If inconsistent: respond PROBLEM with a brief explanation."""
