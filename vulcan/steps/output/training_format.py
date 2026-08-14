"""Helper functions for training format selection.

Imports system prompt templates from vulcan.prompts.training_format and
provides a unified format_system_prompt() dispatcher used by the export_dataset
step and any callers that need to produce a training-format system message.
"""

from __future__ import annotations

import json

from vulcan.prompts.training_format import (
    JSON_SYSTEM_TEMPLATE,
    XML_SYSTEM_TEMPLATE,
    POLICY_JSON_SYSTEM_TEMPLATE,
    POLICY_XML_SYSTEM_TEMPLATE,
    REASONING_SYSTEM_TEMPLATE,
    REASONING_XML_SYSTEM_TEMPLATE,
)

__all__ = [
    "format_system_prompt",
    # Re-export templates so callers can import from here or from prompts
    "JSON_SYSTEM_TEMPLATE",
    "XML_SYSTEM_TEMPLATE",
    "POLICY_JSON_SYSTEM_TEMPLATE",
    "POLICY_XML_SYSTEM_TEMPLATE",
    "REASONING_SYSTEM_TEMPLATE",
    "REASONING_XML_SYSTEM_TEMPLATE",
]


def format_system_prompt(
    tool_specs: list[dict],
    fmt: str,
    *,
    domain_rules: str = "",
) -> str:
    """Return a formatted training-format system prompt string.

    Args:
        tool_specs:   list of tool spec dicts (function objects).
        fmt:          one of "json", "xml", "policy_json", "policy_xml",
                      "reasoning", "reasoning_xml".
        domain_rules: domain policy text — required for "policy_json" / "policy_xml".

    Returns:
        Formatted system prompt string ready to inject as role=system.

    Raises:
        ValueError: if fmt is not one of the known values.
    """
    fmt = fmt.lower()

    if fmt == "json":
        return JSON_SYSTEM_TEMPLATE.format(tool_specs=json.dumps(tool_specs, indent=2))

    if fmt == "xml":
        lines = [json.dumps(spec) for spec in tool_specs]
        return XML_SYSTEM_TEMPLATE.format(tool_specs="\n".join(lines))

    if fmt == "policy_json":
        return POLICY_JSON_SYSTEM_TEMPLATE.format(
            domain_rules=domain_rules,
            tool_specs=json.dumps(tool_specs, indent=2),
        )

    if fmt == "policy_xml":
        lines = [json.dumps(spec) for spec in tool_specs]
        return POLICY_XML_SYSTEM_TEMPLATE.format(
            domain_rules=domain_rules,
            tool_specs="\n".join(lines),
        )

    if fmt == "reasoning":
        return REASONING_SYSTEM_TEMPLATE.format(tool_specs=json.dumps(tool_specs))

    if fmt == "reasoning_xml":
        lines = [json.dumps(spec) for spec in tool_specs]
        return REASONING_XML_SYSTEM_TEMPLATE.format(tool_specs="\n".join(lines))

    raise ValueError(
        f"Unknown training format {fmt!r}. "
        "Choose from: json, xml, policy_json, policy_xml, reasoning, reasoning_xml."
    )
