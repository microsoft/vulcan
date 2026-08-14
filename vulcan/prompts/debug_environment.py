"""Prompts for the debug_environment step (environment test & fix).

Flow per iteration:
0. Validate parameter names/types match tool specs (deterministic + fix loop)
0.5. Generate test initial state (reuses the generate_states prompt)
1. Generate test cases → Execute with initial state → Analyze (failures + state review) → Fix
"""

TEST_GEN_SYSTEM = """Write Python test cases for an environment class. The tests must verify:

1. **Initial state is non-empty**: After construction, `self.state` contains data structures
2. **All tools present**: `function_name_mapping()` returns all expected tool names
3. **Each tool callable**: Call each method with valid inputs, verify JSON return
4. **State persistence**: Call one tool, verify state changed, call another tool that reads that state
5. **Error cases**: Call tools with invalid inputs, verify meaningful error JSON returned
6. **Required vs optional params**: Required params must be provided, optional params have defaults
7. **Return value correctness**: Parse the JSON return of each tool and verify:
   - All fields from the tool's response schema are present in the output
   - Field types match the response schema (string, int, float, bool, list, dict)
   - No extra unexpected fields beyond what the schema defines
8. **Business logic correctness**: Verify that tool behavior matches its description:
   - If a tool says "activates parking brake", calling it with mode="engage" must change state to engaged
   - If a tool modifies a numeric value (e.g., fill fuel), verify the value actually changed by the expected amount
   - If a tool has enum parameters, verify it rejects values outside the enum
   - If a tool has preconditions (e.g., engine must be running), verify it handles unmet preconditions
9. **Cross-tool consistency**: When tool A's output feeds into tool B:
   - Execute A, then B with A's output values
   - Verify the chain produces consistent results and state

## CRITICAL CODE FORMAT RULES
- Output ONLY raw Python code. NEVER wrap in markdown code fences (``` or ```python).
- The code will be executed directly via exec(). Any markdown syntax causes SyntaxError.
- Define a single `run_tests()` function that takes NO arguments and calls all tests.
- Inside `run_tests()`, instantiate the class directly: `env = ClassName(initial_state=INITIAL_STATE)`
- INITIAL_STATE must be defined as a Python dict literal in your code, NOT as a JSON string or placeholder.
- Copy the tool specs as a Python list literal in your code, NOT as a JSON.parse() call or placeholder.
- All test functions must take NO arguments — they access `env` from the enclosing scope.
- Call `run_tests()` at the end of the script.

## Output Format
Print results as:
  PASS: test_name - description
  FAIL: test_name - description - actual error
End with: SUMMARY: X passed, Y failed"""

TEST_GEN_USER = """Write test cases for this environment class. Verify all 9 test categories.
Use the provided initial state to instantiate the class: cls(initial_state=INITIAL_STATE)

<code>
{PY_CODE}
</code>

<tool_specifications>
{TOOL_SPECS}
</tool_specifications>

<initial_state>
{INIT_STATE}
</initial_state>"""

ANALYZE_SYSTEM = """You are an environment code analyst. You receive test results, the environment code, and tool specifications. Your job is two-fold:

1. **Analyze test failures** — categorize what went wrong
2. **Review initial state design** — check if `self.state` in `__init__` is correct for all tools

## Rules

- If the test output is empty or contains only warnings: tests did not execute. Report as [FAILURE] no_output.
- Only output <verdict>ALL_PASSED</verdict> when there are actual PASS results and zero FAILs.

## Output Format

<verdict>ALL_PASSED or FAILED</verdict>

If FAILED, continue with:

### Test Failures
List issues (max 15) as numbered items:
1. [FAILURE] method_name: general test failure
2. [STATE_ISSUE] description: state not persisted correctly across calls
3. [MISSING_TOOL] tool_name: not in function_name_mapping
4. [PARAM_ISSUE] method.param: wrong default or missing parameter
5. [ERROR_HANDLING] method_name: no error handling for invalid input
6. [RETURN_MISSING_FIELD] method_name.field: response schema field missing from output
7. [RETURN_WRONG_TYPE] method_name.field: field type doesn't match schema (expected X, got Y)
8. [RETURN_EXTRA_FIELD] method_name.field: field not in response schema
9. [LOGIC_MISMATCH] method_name: behavior contradicts tool description (explain expected vs actual)
10. [ENUM_VIOLATION] method_name.param: accepts values outside declared enum
11. [PRECONDITION_IGNORED] method_name: does not enforce preconditions described in spec
12. [CROSS_TOOL_INCONSISTENCY] tool_A → tool_B: chained execution produces inconsistent state or values

### State Review
Analyze the initial state design:
1. **Per-tool state dependency**: For each tool, what keys it reads/writes in `self.state`
2. **Cross-tool data flow**: Which tools write state that other tools read
3. **Missing state keys**: Keys accessed in code but not initialized in `__init__`
4. **Type mismatches**: Keys initialized with wrong type (e.g., list instead of dict)
5. **Suggested fixes**: Specific changes to `self.state` in `__init__` that would fix the failures"""

ANALYZE_USER = """Analyze these test results and review the initial state design.

<test_code>
{TEST_CODE}
</test_code>

<test_output>
{TEST_OUTPUT}
</test_output>

<code>
{PY_CODE}
</code>

<tool_specifications>
{TOOL_SPECS}
</tool_specifications>"""

# ── Fix Agent ──────────────────────────────────────────────────────────

FIX_SYSTEM = """Fix defects in a Python environment class. You receive a combined analysis that includes:
1. Test failure categorization (what went wrong at runtime)
2. Initial state review (what state structures are needed, missing, or wrong)

The fixed code must satisfy:

1. Non-empty initial state with ALL required data structures
2. All tools present with function_name_mapping()
3. All parameters from spec (required vs optional)
4. Minimal try/except with meaningful error JSON messages
5. Minimal comments
6. Shared state between tools via self.state — tools that depend on each other communicate through state
7. Code executes without errors
8. State-driven data: Every method that returns data (search results, lists, lookups, current values) MUST read that data from self.state — NOT from hardcoded dicts, lists, or constants defined inside the method body. If a method defines inline sample data or a lookup table, move that data into self.state in __init__ and read from self.state in the method. The method output MUST change when self.state changes.

IMPORTANT: Pay special attention to missing_keys and type_mismatches in the state review section — these are the most common root causes of test failures.
IMPORTANT: Methods that use hardcoded inline data instead of self.state are CRITICAL bugs — the method will ignore initial_state during trajectory generation, producing wrong results.

SURGICAL FIX RULES — these are critical for avoiding regressions:
1. First, list the EXACT methods you will modify (by name) and why.
2. ONLY change methods specifically identified as buggy in the analysis.
3. For every method NOT mentioned in the analysis, copy it EXACTLY byte-for-byte — do not rename variables, reformat, reorder, or "improve" working methods.
4. Do NOT change __init__ unless the analysis specifically identifies missing/wrong state keys.
5. Do NOT change function_name_mapping() unless it is explicitly broken.
6. If a test fails due to a schema mismatch (extra fields, wrong types), fix ONLY the return statement of the affected method.

Return the COMPLETE fixed code:

<modified_code>Complete corrected Python code with all imports</modified_code>"""

FIX_USER = """Fix this environment class using the analysis below.

<analysis>
{ANALYSIS}
</analysis>

<code>
{PY_CODE}
</code>"""
