"""Prompts for the generate_tools step (API implementation code generation)."""

# --- Check name/description consistency ---

CHECK_NAME_DESC_SYSTEM = ""

CHECK_NAME_DESC_USER = """Analyze this API specification. Check if the function name and description accurately describe what the API does.

<api_spec>
{INP_DICT}
</api_spec>

If consistent, respond with just: PASS
If inconsistent, respond with PROBLEM and provide a corrected JSON specification inside <json_schema> tags.

<json_schema>corrected JSON here</json_schema>"""

# --- Generate response schema ---

SCHEMA_RESPONSE_SYSTEM = """Generate a response schema for an API that lacks one. Return the complete API spec with a new "response" field added."""

SCHEMA_RESPONSE_USER = """This API specification is missing a response schema. Add one based on the API's description and parameters.

<api_spec>
{JSON_SCHEMA}
</api_spec>

Return the complete updated JSON spec inside <json_schema> tags.

<json_schema>complete JSON with response field</json_schema>"""

# --- Python function generation (without constraints) ---

FG_SYSTEM = """Generate a Python class that implements an API tool from its JSON specification.

## Required Class Structure

The class MUST follow this exact abstract pattern:

```python
import abc
from typing import Any, Dict

class Tool(abc.ABC):
    @staticmethod
    def invoke(args, kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError

class MyApiTool(Tool):
    @staticmethod
    def invoke(param1: type, param2: type, ...) -> str:
        # Implementation here
        return json.dumps({...})

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": { ... }}
```

## Criteria

1. **External data detection**: Analyze whether the API requires external state (e.g., database lookups, user sessions, authentication tokens, stored records). If YES, add `data: Dict[str, Any]` as the FIRST parameter of `invoke` and implement the logic using `data` to retrieve/store values. If NO, use only the declared parameters.

2. **Error handling**: Use try/except blocks MINIMALLY — only where a runtime error is genuinely possible (e.g., key lookup in data dict, type conversion, division). Each except must return a JSON error with a meaningful message. Do NOT wrap the entire function in a single try/except.

3. **Parameter usage**: EVERY input parameter defined in the tool specification MUST be used meaningfully in the `invoke` body. Respect which parameters are required vs optional — optional parameters should have default values and conditional logic.

4. **Executability**: The generated code must be syntactically valid Python that executes without errors. Use only standard library imports. Return all results as JSON strings via `json.dumps()`.

5. **Simulation**: This code runs in a FULLY SIMULATED environment. The function must NEVER interact with any real external system — no real filesystem, network, database, hardware, or OS operations. All data must come from the `data` parameter or be computed from the inputs. If the API would normally access external resources, simulate the behavior by reading from / writing to the `data` dict and returning realistic JSON responses.

6. **get_info**: Must return the input JSON specification exactly in OpenAI function-calling format: `{"type": "function", "function": {...}}`.

## Output

First decide whether external data is needed, then generate the code:

<decision>
"This API [does/does not] require external data because [reason]."
</decision>

<generated_code>
Complete executable Python code. No markdown fences. Include all imports.
</generated_code>"""

FG_USER = """Generate a Python Tool class for this API:

<api_spec>
{API_INFO}
</api_spec>"""

# --- Python function generation (with constraints) ---

FG_SYSTEM_CONSTRAINTS = """Generate a Python class that implements an API tool from its JSON specification and policy constraints.

## Required Class Structure

The class MUST follow this exact abstract pattern:

```python
import abc
from typing import Any, Dict

class Tool(abc.ABC):
    @staticmethod
    def invoke(args, kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError

class MyApiTool(Tool):
    @staticmethod
    def invoke(param1: type, param2: type, ...) -> str:
        # Implementation here
        return json.dumps({...})

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": { ... }}
```

## Criteria

1. **External data detection**: Analyze whether the API requires external state (database, sessions, auth tokens, records). If YES, add `data: Dict[str, Any]` as the FIRST parameter of `invoke`. If NO, use only declared parameters.

2. **Constraint enforcement**: The implementation MUST enforce ALL provided policy constraints. Derive context-sensitive fields (status, flags, outcomes) from the constraints and input data — never hardcode fixed values.

3. **Error handling**: Use try/except MINIMALLY — only where a runtime error is genuinely possible. Each except must return a JSON error with a meaningful message. No blanket try/except around the whole function.

4. **Parameter usage**: EVERY input parameter from the spec MUST be used meaningfully in `invoke`. Respect required vs optional — optional parameters get default values and conditional logic.

5. **Executability**: Valid Python, standard library only, all results as JSON strings via `json.dumps()`.

6. **Simulation**: This code runs in a FULLY SIMULATED environment. The function must NEVER interact with any real external system. All data must come from the `data` parameter or be computed. Simulate all external operations by reading from / writing to the `data` dict.

7. **get_info**: Return the spec in OpenAI function-calling format.

## Output

<decision>
"This API [does/does not] require external data because [reason]."
</decision>

<generated_code>
Complete executable Python code. No markdown fences. Include all imports.
</generated_code>"""

FG_USER_CONSTRAINTS = """Generate a Python Tool class for this API:

<api_spec>
{API_INFO}
</api_spec>

<policy_constraints>
{POLICY_CONSTRAINTS}
</policy_constraints>"""

# --- Code modifier/verifier (inline review after generation) ---

FM_SYSTEM = """Review a generated Python Tool class against ALL of these criteria:

## Checks (ALL must pass)
1. **Abstract pattern**: Class subclasses Tool with static `invoke` and `get_info` methods.
2. **External data**: If the API needs state/database, `invoke` has `data: Dict` as first param. If not, no `data` param.
3. **Parameter usage**: Every `invoke` parameter (except self, data) is meaningfully used in the body — not just declared.
4. **Required vs optional**: Required params have no defaults. Optional params have defaults and conditional logic.
5. **Error handling**: try/except used minimally, only around genuinely risky operations, with meaningful JSON error messages. NOT a blanket try/except.
6. **Executability**: Code is syntactically valid Python 3.11+ and will execute without errors.
7. **get_info**: Returns the spec in `{"type": "function", "function": {...}}` format matching the invoke signature.

If ALL checks pass: output FINE.
If ANY check fails: fix the code and return the complete corrected version.

<decision>FINE | PROBLEM: [which criteria failed and why]</decision>
<generated_code>None | [complete corrected Python code]</generated_code>"""

FM_USER = """Review this Python Tool class against all criteria:

<code>
{PY_CODE}
</code>"""

# --- Executor (fix execution errors) ---

EXECUTOR_SYSTEM = """Fix a Python Tool class that has execution errors. The class must:
- Subclass Tool with static invoke() and get_info() methods
- Be executable without errors
- Use only standard library imports
- Return JSON strings from invoke()

Return the complete corrected code inside tags:

<verified_code>corrected Python code</verified_code>"""

EXECUTOR_USER = """This API implementation has an execution error. Fix it while preserving the Tool abstract pattern.

<api_spec>
{API_INFO}
</api_spec>

<code>
{PY_CODE}
</code>

<error>
{ERROR_MESSAGE}
</error>"""
