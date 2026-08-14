"""Tests for the state-awareness detection in debug_environment."""

import json
import pytest

from vulcan.steps.env_simulation.debug_environment import DebugEnvironmentStep


HARDCODED_ENV = '''
import json

class BadEnv:
    def __init__(self, initial_state=None):
        self.state = initial_state if initial_state else {
            "users": [{"id": 1, "name": "Alice"}],
            "last_results": {},
        }

    def function_name_mapping(self):
        return {
            "search_users": self.search_users,
            "add_user": self.add_user,
            "compute_sum": self.compute_sum,
        }

    def search_users(self, query: str) -> str:
        sample_data = [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
            {"id": 3, "name": "Charlie", "role": "user"},
        ]
        results = [u for u in sample_data if query.lower() in u["name"].lower()]
        self.state.setdefault("last_results", {})["search_users"] = results
        return json.dumps({"results": results})

    def add_user(self, name: str, role: str = "user") -> str:
        uid = len(self.state.get("users", [])) + 1
        user = {"id": uid, "name": name, "role": role}
        self.state.setdefault("users", []).append(user)
        return json.dumps({"created": user})

    def compute_sum(self, numbers: list) -> str:
        return json.dumps({"sum": sum(numbers)})
'''

GOOD_ENV = '''
import json

class GoodEnv:
    def __init__(self, initial_state=None):
        self.state = initial_state if initial_state else {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
            ],
            "last_results": {},
        }

    def function_name_mapping(self):
        return {
            "search_users": self.search_users,
            "compute_sum": self.compute_sum,
        }

    def search_users(self, query: str) -> str:
        users = self.state.get("users", [])
        results = [u for u in users if query.lower() in u.get("name", "").lower()]
        self.state.setdefault("last_results", {})["search_users"] = results
        return json.dumps({"results": results, "count": len(results)})

    def compute_sum(self, numbers: list) -> str:
        return json.dumps({"sum": sum(numbers)})
'''

TOOL_SPECS = [
    {
        "name": "search_users",
        "description": "Search users in the database by name query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "add_user",
        "description": "Add a new user to the system.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "compute_sum",
        "description": "Compute the sum of a list of numbers.",
        "parameters": {
            "type": "object",
            "properties": {"numbers": {"type": "array"}},
            "required": ["numbers"],
        },
    },
]


class TestCheckStateAwareness:
    def test_detects_hardcoded_list_in_search(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        methods = [i["method"] for i in issues]
        assert "search_users" in methods

    def test_hardcoded_issue_type(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        search_issue = [i for i in issues if i["method"] == "search_users"][0]
        assert search_issue["issue_type"] == "HARDCODED_DATA"

    def test_does_not_flag_state_reading_methods(self):
        issues = DebugEnvironmentStep._check_state_awareness(GOOD_ENV, TOOL_SPECS)
        methods = [i["method"] for i in issues]
        assert "search_users" not in methods

    def test_does_not_flag_pure_computation(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        methods = [i["method"] for i in issues]
        assert "compute_sum" not in methods

    def test_does_not_flag_write_methods(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        methods = [i["method"] for i in issues]
        assert "add_user" not in methods

    def test_skips_init_and_mapping(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        methods = [i["method"] for i in issues]
        assert "__init__" not in methods
        assert "function_name_mapping" not in methods

    def test_hardcoded_collections_populated(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        search_issue = [i for i in issues if i["method"] == "search_users"][0]
        assert len(search_issue["hardcoded_collections"]) > 0

    def test_clean_env_returns_empty(self):
        issues = DebugEnvironmentStep._check_state_awareness(GOOD_ENV, TOOL_SPECS)
        assert len(issues) == 0

    def test_invalid_syntax_returns_empty(self):
        issues = DebugEnvironmentStep._check_state_awareness("def broken(:", TOOL_SPECS)
        assert issues == []

    def test_no_class_returns_empty(self):
        issues = DebugEnvironmentStep._check_state_awareness("x = 1", TOOL_SPECS)
        assert issues == []


class TestRunDifferentialTest:
    def test_confirms_hardcoded_search(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        state = json.dumps({"users": [{"id": 1, "name": "Alice"}], "last_results": {}})
        confirmed = DebugEnvironmentStep._run_differential_test(
            HARDCODED_ENV, issues, state, TOOL_SPECS,
        )
        methods = [c["method"] for c in confirmed]
        assert "search_users" in methods

    def test_good_env_not_confirmed(self):
        issues = DebugEnvironmentStep._check_state_awareness(GOOD_ENV, TOOL_SPECS)
        state = json.dumps({"users": [{"id": 1, "name": "Alice"}], "last_results": {}})
        confirmed = DebugEnvironmentStep._run_differential_test(
            GOOD_ENV, issues, state, TOOL_SPECS,
        )
        assert len(confirmed) == 0

    def test_confirmed_flag_set(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        state = json.dumps({"users": [{"id": 1, "name": "Alice"}], "last_results": {}})
        confirmed = DebugEnvironmentStep._run_differential_test(
            HARDCODED_ENV, issues, state, TOOL_SPECS,
        )
        for c in confirmed:
            assert c.get("confirmed") is True

    def test_empty_state_still_works(self):
        issues = DebugEnvironmentStep._check_state_awareness(HARDCODED_ENV, TOOL_SPECS)
        confirmed = DebugEnvironmentStep._run_differential_test(
            HARDCODED_ENV, issues, "{}", TOOL_SPECS,
        )
        assert isinstance(confirmed, list)

    def test_invalid_json_state_returns_suspects(self):
        issues = [{"method": "x", "issue_type": "NO_STATE_READ",
                   "detail": "test", "state_refs": 0, "hardcoded_collections": []}]
        confirmed = DebugEnvironmentStep._run_differential_test(
            HARDCODED_ENV, issues, "not json", TOOL_SPECS,
        )
        assert confirmed == issues


class TestBuildReport:
    def test_report_contains_method_names(self):
        issues = [
            {"method": "search_users", "issue_type": "HARDCODED_DATA",
             "state_refs": 0, "hardcoded_collections": ["sample_data (list, 3 items)"],
             "detail": "test detail"},
        ]
        report = DebugEnvironmentStep._build_state_awareness_report(issues)
        assert "search_users" in report
        assert "HARDCODED_DATA" in report
        assert "sample_data" in report

    def test_report_contains_fix_instructions(self):
        issues = [
            {"method": "get_data", "issue_type": "NO_STATE_READ",
             "state_refs": 0, "hardcoded_collections": [],
             "detail": "test"},
        ]
        report = DebugEnvironmentStep._build_state_awareness_report(issues)
        assert "self.state" in report
        assert "Required fix" in report

    def test_empty_issues_produces_header(self):
        report = DebugEnvironmentStep._build_state_awareness_report([])
        assert "STATE-AWARENESS" in report


class TestNestedFunctionDetection:
    NESTED_FUNC_ENV = '''
import json
from datetime import datetime, timedelta, timezone
class NestedEnv:
    def __init__(self, initial_state=None):
        self.state = initial_state if initial_state else {"files": {}, "last_results": {}}

    def function_name_mapping(self):
        return {"search_files": self.search_files}

    def search_files(self, query: str) -> str:
        def sample_files():
            return [
                {"id": "f1", "name": "doc.txt", "size": 100},
                {"id": "f2", "name": "img.png", "size": 200},
                {"id": "f3", "name": "data.csv", "size": 300},
            ]
        corpus = sample_files()
        results = [f for f in corpus if query.lower() in f["name"].lower()]
        self.state.setdefault("last_results", {})["search_files"] = results
        return json.dumps({"results": results})
'''

    def test_detects_nested_function_hardcoded_return(self):
        specs = [{"name": "search_files", "description": "Search files in storage",
                  "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                 "required": ["query"]}}]
        issues = DebugEnvironmentStep._check_state_awareness(self.NESTED_FUNC_ENV, specs)
        assert len(issues) > 0
        assert issues[0]["method"] == "search_files"
        assert issues[0]["issue_type"] == "HARDCODED_DATA"
        assert any("sample_files" in h for h in issues[0]["hardcoded_collections"])
