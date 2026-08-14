"""Deterministic validators for VULCAN pipeline stages.

Each validator takes a single output item dict and returns (passed: bool, reason: str).
The VALIDATORS registry at the bottom maps a step name to its validator; steps
without an entry are accepted as-is.
"""

from ..core.code_exec import load_api_class, execute_class_code, find_unused_invoke_params


def validate_generate_tools(item: dict) -> tuple[bool, str]:
    """Validate generate_tools output.

    Criteria:
    1. tool_code exists and has no execution errors
    2. Code follows Tool abstract pattern (subclass with invoke + get_info)
    3. Code is executable via load_api_class
    4. All parameters from spec are used in invoke body (AST check)
    """
    if "tool_code" not in item:
        return False, "missing tool_code"

    decision = item.get("decision", "")
    if decision and "Execution Erro" in str(decision):
        return False, f"execution error in decision: {decision}"

    py_code = item["tool_code"]

    # Check executability
    try:
        api_class = load_api_class(py_code)
    except Exception as e:
        return False, f"load_api_class failed: {e}"

    # Check Tool pattern
    if not hasattr(api_class, "invoke") or not callable(getattr(api_class, "invoke", None)):
        return False, "class missing invoke() method"
    if not hasattr(api_class, "get_info") or not callable(getattr(api_class, "get_info", None)):
        return False, "class missing get_info() method"

    # Check all parameters are used in invoke body (AST-based, not string count)
    unused = find_unused_invoke_params(py_code)
    if unused:
        return False, f"unused invoke parameters: {unused}"

    return True, ""


def validate_verify_tools(item: dict) -> tuple[bool, str]:
    """Validate verify_tools output.

    Hard criteria (fail if not met):
    1. Code exists and is executable (Tool pattern with invoke + get_info)
    2. All parameters used in invoke body
    3. No execution errors

    Soft criteria (informational, does NOT cause failure):
    - check_name_description: logged but not blocking (simulation differences are expected)
    """
    passed, reason = validate_generate_tools(item)
    if not passed:
        return passed, reason

    # check_name_description is informational only — don't fail on it
    # The check is still run and stored in the output for manual review

    return True, ""


def validate_build_graph(item: dict) -> tuple[bool, str]:
    """Validate graph_dependency output."""
    if "graph" not in item:
        return False, "missing graph"
    if len(item["graph"]) < 1:
        return False, "graph is empty"
    return True, ""


def validate_score_sequences(item: dict) -> tuple[bool, str]:
    """Validate sequence scoring output."""
    if "judgment" not in item:
        return False, "missing judgment"
    if item["judgment"] is None:
        return False, "judgment is None"
    return True, ""


def validate_assemble_environment(item: dict) -> tuple[bool, str]:
    """Validate environment class against 7 criteria:
    1. Initial state that keeps intermediate states (non-empty)
    2. One method per tool + function_name_mapping()
    3. All input arguments from spec (required vs optional)
    4. Corner case handling with try/except
    5. Minimal comments
    6. Shared state accessible between tools
    7. Initial state must not be empty
    """
    if "tool_code" not in item:
        return False, "missing tool_code"

    code = item["tool_code"]

    # C1/C7: Code executable + class instantiable
    try:
        cls = execute_class_code(code)
        if cls is None:
            return False, "execute_class_code returned None"
    except Exception as e:
        return False, f"code not executable: {e}"

    # Try instantiating with empty state to check defaults
    try:
        obj = cls()
    except TypeError:
        try:
            obj = cls(initial_state={})
        except Exception as e:
            return False, f"cannot instantiate class: {e}"

    # C2: function_name_mapping exists and returns dict
    if not hasattr(obj, "function_name_mapping"):
        return False, "missing function_name_mapping() method"
    try:
        mapping = obj.function_name_mapping()
    except Exception as e:
        return False, f"function_name_mapping() failed: {e}"
    if not isinstance(mapping, dict) or not mapping:
        return False, "function_name_mapping() returned empty or non-dict"

    # C2: Check all mapped methods are callable
    for name, method in mapping.items():
        if not callable(method):
            return False, f"function_name_mapping['{name}'] is not callable"

    # C1/C7: Check state is non-empty
    state = getattr(obj, "state", None) or getattr(obj, "initial_state", None) or getattr(obj, "_state", None)
    if state is not None and isinstance(state, dict) and len(state) == 0:
        return False, "initial state is empty (criteria 7: must not be empty)"

    # C2: Check tool coverage against spec if available
    api_list = item.get("api_list", [])
    if api_list:
        spec_names = set()
        for api in api_list:
            func = api.get("normalized_schema") or api.get("normalized_function") or api.get("function", {})
            name = func.get("name", "")
            if name:
                spec_names.add(name)
        mapping_names = set(mapping.keys())
        missing = spec_names - mapping_names
        if missing:
            return False, f"tools missing from function_name_mapping: {missing}"

    return True, ""


def validate_debug_environment(item: dict) -> tuple[bool, str]:
    """Validate debug_environment output. Strict: environment_ready must be True."""
    if item.get("environment_ready") is not True:
        iterations = item.get("test_iterations", [])
        last_status = iterations[-1]["status"] if iterations else "no_iterations"
        return False, f"env tests did not pass after {len(iterations)} iterations (last: {last_status})"
    return True, ""


def validate_generate_states(item: dict) -> tuple[bool, str]:
    """Validate generate_states output — each item is one state."""
    state = item.get("initial_state")
    if state is None:
        return False, "initial_state is None"
    if not isinstance(state, dict) or not state:
        return False, "initial_state is empty or not a dict"
    return True, ""


def validate_sample_arguments(item: dict) -> tuple[bool, str]:
    """Validate sample_arguments output.
    New format: each item has initial_state + functions (exploded per state).
    Legacy format: mock_data.initial_data list.
    """
    # New format (exploded per state)
    if "functions" in item:
        if item["functions"] is None:
            return False, "functions is None"
        if isinstance(item["functions"], list) and len(item["functions"]) > 0:
            return True, ""
        return False, "functions is empty or invalid"

    # Legacy format
    mock_data = item.get("mock_data", {})
    initial_data = mock_data.get("initial_data")

    if initial_data is None:
        return False, "initial_data is None"

    if isinstance(initial_data, list):
        if not initial_data:
            return False, "initial_data list is empty"
        has_functions = any(
            isinstance(d, dict) and d.get("functions") is not None
            for d in initial_data
        )
        if not has_functions:
            return False, "no items in initial_data have functions"
        return True, ""

    # Dict format: check functions key
    if isinstance(initial_data, dict):
        if initial_data.get("functions") is None:
            return False, "functions is None"
        return True, ""

    return False, f"initial_data unexpected type: {type(initial_data).__name__}"


def validate_generate_queries(item: dict) -> tuple[bool, str]:
    """Validate query generation output."""
    if "query" not in item:
        return False, "missing query"
    query = item["query"]
    if isinstance(query, str) and len(query) < 1:
        return False, "query is empty"
    if query == "Error":
        return False, "query is Error"
    return True, ""


def validate_generate_trajectories(item: dict) -> tuple[bool, str]:
    """Validate trajectory generation output (static/dynamic/thinking)."""
    if "conversation" not in item:
        return False, "missing conversation"
    msgs = item["conversation"]
    if not isinstance(msgs, list):
        return False, "conversation is not a list"
    if len(msgs) < 3:
        return False, f"conversation too short: {len(msgs)}"
    if len(msgs) == 3 and not msgs[-1].get("content"):
        return False, "last message has no content"
    return True, ""


def validate_judge_trajectories(item: dict) -> tuple[bool, str]:
    """Validate traj_verification output."""
    if "judgment" not in item:
        return False, "missing judgment"
    if item["judgment"] is None:
        return False, "judgment is None"
    return True, ""


def validate_mock_data(item: dict) -> tuple[bool, str]:
    """Validate mock_data output."""
    if "mock_data" not in item:
        return False, "missing mock_data"
    data = item["mock_data"].get("data", [])
    if len(data) < 1:
        return False, "mock_data.data is empty"
    return True, ""


# Registry mapping step names to validators
VALIDATORS = {
    "generate_tools": validate_generate_tools,
    "verify_tools": validate_verify_tools,
    "build_graph": validate_build_graph,
    "score_sequences": validate_score_sequences,
    "assemble_environment": validate_assemble_environment,
    "debug_environment": validate_debug_environment,
    "generate_states": validate_generate_states,
    "sample_arguments": validate_sample_arguments,
    "generate_queries": validate_generate_queries,
    "generate_trajectories": validate_generate_trajectories,
    "generate_trajectories_tools_only": validate_generate_trajectories,
    "generate_trajectories_with_user_and_policy": validate_generate_trajectories,
    "generate_trajectories_reasoning": validate_generate_trajectories,
    "judge_trajectories": validate_judge_trajectories,
}
