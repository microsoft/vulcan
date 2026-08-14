"""EnvironmentConfig: generic, fully config-driven environment representation.

One class serves every environment:
  - an EnvironmentType enum instead of scattered boolean flag checks
  - a single loading path, driven by YAML plus a standardized input JSONL
  - clean type hints throughout
  - the same public API for every pipeline step

To add a new environment:
  1. Create an input catalog JSONL (one object per category, see _load_catalog).
  2. Create a YAML config (see config/example_environment.yaml).
  3. Run: python -m vulcan.cli --env <name> --stage generate_tools
"""

import os
from typing import Any

from .types import EnvironmentType


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, returning a list of parsed dicts."""
    import json
    items: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class EnvironmentConfig:
    """Generic environment config.  Loads everything from YAML + standardized input files.

    No subclassing is required for new environments.  All behavior is driven by:
      - ``environment_type``: EnvironmentType enum value (or inferred from flags)
      - Standard input JSONL: one record per category containing api_list, policy_rules, etc.

    Public API (used by pipeline steps)
    ------------------------------------
    get_categories()                     -> list[str]
    get_api_catalog(category)            -> list[dict]
    get_domain_rules(category)           -> str | None
    get_constraints(category)            -> dict | None          {api_name: text}
    get_api_constraints(category, name)  -> str | None
    get_environment_type(category)              -> str
    get_terminate_tools()                -> list[str]
    """

    # Appended to the API catalog for environments that support human escalation.
    TRANSFER_TO_HUMAN_API: dict[str, Any] = {
        "name": "transfer_to_human_agents",
        "description": (
            "Transfer the user to a human agent, with a summary of the user's issue.\n\n"
            "Only transfer if\n"
            " -  the user explicitly asks for a human agent\n"
            " -  given the policy and the available tools, you cannot solve the user's issue."
        ),
        "parameters": {
            "properties": {
                "summary": {
                    "description": "A summary of the user's issue.",
                    "title": "Summary",
                    "type": "string",
                }
            },
            "required": ["summary"],
            "title": "parameters",
            "type": "object",
        },
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.name: str = config.get("environment", config.get("domain", "unknown"))

        # Resolve environment type (explicit key, or inferred from boolean flags)
        self.environment_type: EnvironmentType = EnvironmentType.from_config(config)

        # Convenience aliases matching EnvironmentType properties
        self.has_user_agent: bool = self.environment_type.has_user_agent
        self.has_policy: bool = self.environment_type.has_policy
        self.has_constraints: bool = config.get("has_constraints", False)

        # Internal data stores, populated by _load_data()
        self._api_catalog: dict[str, list[dict]] = {}
        self._policy_rules: dict[str, str] = {}
        self._constraints: dict[str, dict[str, str]] = {}
        self._environment_type_map: dict[str, str] = config.get("environment_type_map", {})
        # Defaults to the built-in escalation tool, which is the only tool this
        # class injects itself. Referenced through the constant rather than
        # spelled out, so the default cannot drift from the tool it names.
        self._terminate_tools: list[str] = config.get(
            "terminate_tools", [self.TRANSFER_TO_HUMAN_API["name"]]
        )

        self._load_data()

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load the environment's tool catalog, when one is configured."""
        input_cfg = self.config.get("input", {})
        catalog_path: str | None = input_cfg.get("catalog")

        if catalog_path and os.path.exists(catalog_path):
            self._load_catalog(catalog_path)
        # If path not specified or does not exist yet, data stays empty.
        # Steps that require data will raise at access time.

    def _load_catalog(self, path: str) -> None:
        """Load the standardized input format.

        Each JSONL line represents one environment category:

        .. code-block:: json

            {
                "category_name": "ticket_api",
                "api_list": [
                    {
                        "function": {"name": "...", "description": "...", "parameters": {}, "response": {}},
                        "constraints": "optional per-API constraint text"
                    }
                ],
                "policy_rules": "optional domain rules text for this category",
                "environment_type": "tools_only | with_user_and_policy | with_user"
            }
        """
        for item in _load_jsonl(path):
            cat: str = item.get("category_name", "")
            if not cat:
                continue

            # ── API catalog ──────────────────────────────────────────────
            raw_apis: list[dict] = item.get("api_list", [])
            normalized: list[dict] = []
            for api in raw_apis:
                if "function" in api:
                    normalized.append(api)
                elif "name" in api:
                    # Bare function object — wrap it
                    normalized.append({"function": api})
                else:
                    normalized.append(api)
            self._api_catalog[cat] = normalized

            # ── Per-API constraints ──────────────────────────────────────
            cat_constraints: dict[str, str] = {}
            for api in normalized:
                func = api.get("function", {})
                api_name: str = func.get("name", "")
                constraint_text = api.get("constraints")
                if api_name and constraint_text:
                    cat_constraints[api_name] = constraint_text
            if cat_constraints:
                self._constraints[cat] = cat_constraints

            # ── Policy rules ─────────────────────────────────────────────
            policy = item.get("policy_rules")
            if policy:
                self._policy_rules[cat] = policy

            # ── Trajectory mode override ─────────────────────────────────
            environment_type = item.get("environment_type")
            if environment_type:
                self._environment_type_map[cat] = environment_type

        # Update flags if data implies them
        if self._policy_rules and not self.has_policy:
            self.has_policy = True
        if self._constraints and not self.has_constraints:
            self.has_constraints = True

    # ── Public API (used by pipeline steps) ───────────────────────────────

    def get_categories(self) -> list[str]:
        """Return all category names present in the API catalog."""
        return list(self._api_catalog.keys())

    def get_api_catalog(self, category: str) -> list[dict]:
        """Return the list of API entries for *category*.

        Each entry has the shape ``{"function": {...}, "constraints": "..."}``
        where ``"constraints"`` is optional.
        """
        return self._api_catalog.get(category, [])

    def get_domain_rules(self, category: str) -> str | None:
        """Return domain policy / rules text for *category*, or ``None``."""
        return self._policy_rules.get(category)

    def get_constraints(self, category: str) -> dict[str, str] | None:
        """Return per-API constraints dict ``{api_name: text}`` for *category*, or ``None``."""
        return self._constraints.get(category)

    def get_api_constraints(self, category: str, api_name: str) -> str | None:
        """Return constraints text for a specific API within *category*, or ``None``."""
        return self._constraints.get(category, {}).get(api_name)

    def get_terminate_tools(self) -> list[str]:
        """Return the list of tool names that signal trajectory termination."""
        return self._terminate_tools

    def get_environment_type(self, category: str) -> str:
        """Determine trajectory generation mode for *category*.

        Priority:
        1. Explicit per-category override in ``environment_type_map`` (config or input data).
        2. Category-level inference: if policy rules exist for this category, use
           ``"with_user_and_policy"`` even if the environment is TOOLS_ONLY at the top level.
        3. Environment-level default from ``EnvironmentType.environment_type``.

        Returns:
            One of ``"tools_only"``, ``"with_user_and_policy"``, or ``"with_user"``.
        """
        # 1. Explicit override
        if category in self._environment_type_map:
            return self._environment_type_map[category]

        # 2. Category has policy rules even if env-level type is TOOLS_ONLY
        if category in self._policy_rules:
            return "with_user_and_policy"

        # 3. Environment-level default
        return self.environment_type.environment_type

    def __repr__(self) -> str:
        categories = self.get_categories()
        return (
            f"EnvironmentConfig(name={self.name!r}, "
            f"type={self.environment_type.value!r}, "
            f"categories={categories})"
        )
