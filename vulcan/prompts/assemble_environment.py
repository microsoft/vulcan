"""Prompts for the assemble_environment step (environment class generation).

Criteria:
1. Initial state that keeps intermediate states between tools (non-empty)
2. One method per tool + function_name_mapping() for spec-name to method-name
3. All input arguments from tool spec (required vs optional)
4. Corner case handling with try/except and meaningful error messages
5. Minimal comments
6. Shared state accessible across all tool methods via initial state
7. Initial state must not be empty - must contain all required shared data
"""

CLASS_GEN_SYSTEM = """Generate a Python environment class that unifies multiple API tools into a single stateful environment.

## CRITICAL: FULLY SIMULATED ENVIRONMENT

This environment is FULLY SIMULATED. The class must NEVER interact with any real external system — no real filesystem, network, database, hardware, or OS operations. ALL data must come from `self.state`. If a tool would normally access external resources, simulate the behavior by reading from / writing to `self.state` and returning realistic JSON responses.

For path string manipulation only (no actual filesystem access), you may use: os.path.dirname, os.path.basename, os.path.splitext, os.path.join.

## Required Structure

```python
class EnvironmentName:
    def __init__(self, initial_state: dict = None):
        # Initialize with provided state or defaults
        # State MUST NOT be empty - include all shared data structures
        self.state = initial_state if initial_state else {
            # Default non-empty state with all required shared data
        }

    def function_name_mapping(self) -> dict:
        # Maps tool spec names to actual method references
        return {
            "spec_tool_name": self.method_name,
            ...
        }

    def method_name(self, param1: type, param2: type = default, ...) -> str:
        # Implementation using self.state for shared data
        return json.dumps({...})
```

## Criteria (ALL must be satisfied)

1. **Stateful initial state**: The `__init__` must accept `initial_state: dict` and initialize `self.state` with it. The state keeps intermediate results between tool calls (e.g., user sessions, transaction records, environment data). The default state MUST NOT be empty — it must contain realistic default data structures for all shared state the tools need.

2. **One method per tool + mapping**: Every tool in the tool specification list must have a corresponding method. The `function_name_mapping()` method must return a dict mapping each tool spec name (string) to the bound method reference.

3. **Parameter names MUST EXACTLY match tool specs**: Each method's parameter names must be IDENTICAL to the parameter names in the tool specification. Do NOT rename parameters. If the spec says `folder_path`, the method must use `folder_path` — not `path`, not `directory`, not `src_folder`. Required parameters have no defaults. Optional parameters have appropriate defaults. Access shared state via `self.state` (NOT a `data` parameter).

4. **Error handling**: Use try/except around genuinely risky operations (key lookups in state, type conversions, missing data). Each except returns `json.dumps({"error": "meaningful message"})`. Do NOT wrap entire methods in a single try/except.

5. **Minimal comments**: Only add comments where logic is non-obvious. No docstrings on every method. No commenting obvious code. No blank lines between simple statements.

6. **Efficient implementation**: Write compact, direct code. No unnecessary variables, no redundant checks, no verbose patterns. Each method should be as short as possible while being correct. Avoid repeating the same validation logic across methods — extract common patterns into private helpers if used 3+ times.

7. **Shared state between tools**: Tools that depend on each other's output communicate through `self.state`. For example, an auth tool stores a token in `self.state["tokens"]`, and a booking tool reads from `self.state["tokens"]` to validate access.

8. **Non-empty default state**: The default initial state must contain realistic data structures: user records, account data, configuration, or whatever the tools need. Never `self.state = {}`.

## Output

<decision>Description of what was combined and what shared state is needed</decision>
<generated_code>Complete executable Python code with all imports</generated_code>"""

CLASS_GEN_USER = """Generate a unified environment class from these individual API invoke methods and their tool specifications.

<api_codes>
{API_CODES}
</api_codes>

<tool_specifications>
{TOOL_SPECS}
</tool_specifications>"""

CLASS_VERIFY_SYSTEM = """Verify a unified environment class against ALL criteria:

## Criteria (ALL must pass)
0. **FULLY SIMULATED (CRITICAL)**: The code must NEVER interact with any real external system. All data must come from self.state — never from real filesystem, network, database, or hardware. If ANY real external call is found, respond with Problem and provide fixed code that uses self.state instead.
1. **Stateful init**: `__init__` accepts `initial_state: dict`, `self.state` initialized with non-empty defaults
2. **Tool coverage + mapping**: Every tool spec has a method, `function_name_mapping()` maps all spec names to methods
3. **Parameter completeness**: All spec parameters present in method signatures (required=no default, optional=has default)
4. **Error handling**: try/except only around risky operations, meaningful error messages, no blanket try/except
5. **Minimal comments**: No excessive commenting, no blank lines between simple statements
6. **Efficient code**: Compact implementation, no redundant logic, use helpers for repeated patterns
7. **Shared state**: Tools communicate via `self.state`, intermediate results persisted
8. **Non-empty state**: Default `self.state` contains realistic data structures, never empty

If ALL pass: FINE.
If ANY fails: fix and return complete corrected code.

<decision>FINE | PROBLEM: [which criteria failed]</decision>
<generated_code>None | [complete corrected code]</generated_code>"""

CLASS_VERIFY_USER = """Verify this environment class against all 7 criteria:

<code>
{PY_CODE}
</code>

<tool_specifications>
{TOOL_SPECS}
</tool_specifications>"""
