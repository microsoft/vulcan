"""Per-stage verification criteria for VULCAN's LLM-based quality checks."""

VERIFICATION_CRITERIA = {
    "generate_tools": {
        "check_items": [
            "Code implements the function described in the tool spec",
            "All parameters from spec are used in the implementation",
            "Return values match the response schema structure",
            "Error handling uses meaningful error messages",
            "No external library imports beyond standard library + numpy",
        ],
    },
    "verify_tools": {
        "check_items": [
            "Verification iterations show improvement or passing",
            "Final code is functionally equivalent to spec",
            "Modified schemas preserve all required parameters",
        ],
    },
    "assemble_environment": {
        "check_items": [
            "Environment class has function_name_mapping() covering all APIs",
            "Initial state __init__ accepts and uses initial_state dict",
            "Shared state maintained between tool calls via self.state",
            "All public methods have reasonable error handling",
        ],
    },
    "debug_environment": {
        "check_items": [
            "All test categories pass (state persistence, error handling, return correctness)",
            "Initial state contains all keys accessed by tool methods",
            "Business logic matches tool descriptions",
            "Cross-tool state consistency is maintained",
        ],
    },
    "generate_states": {
        "check_items": [
            "Initial state includes all keys accessed by the environment code",
            "Types match what the code expects (no list where dict is needed)",
            "Values are realistic and domain-appropriate",
            "No empty strings, null values, or empty lists",
        ],
    },
    "sample_arguments": {
        "check_items": [
            "At least 3 test cases per function (typical, edge, error)",
            "Expected outputs are derived from code logic + initial state",
            "Error cases trigger the actual error handling paths",
            "All functions in the api_list are covered",
        ],
    },
    "build_graph": {
        "check_items": [
            "Explicit edges have valid source/target argument field names",
            "Implicit edges represent a logical workflow ordering",
            "Edge direction matches data flow (producer → consumer)",
        ],
    },
    "score_sequences": {
        "check_items": [
            "Judgment score is consistent with the stated rationale",
            "GoalOriented classification is justified by a clear user-facing goal",
            "Data flow and task coherence scores are accurately assessed",
        ],
    },
    "generate_trajectories": {
        "check_items": [
            "Trajectory follows correct chat template (system, user, assistant alternation)",
            "Every tool_call has a corresponding tool response",
            "Agent uses actual values from tool responses (not fabricated)",
            "Conversation reaches a natural conclusion",
        ],
    },
    "judge_trajectories": {
        "check_items": [
            "Judgment score is justified by trajectory content",
            "task_completed assessment matches actual trajectory outcome",
            "Error classification is correct (wrong_tool, missing_tool_call, etc.)",
        ],
    },
    "generate_queries": {
        "check_items": [
            "Query is declarative (states goal, not steps)",
            "Query contains only information needed to call the required tools",
            "For multi-turn queries, inter-turn dependency is natural",
            "No tool names or API references appear in the query",
        ],
    },
}
