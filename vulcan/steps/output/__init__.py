"""Output steps: selection and reconstruction of training data."""

from .export_dataset import ExportDatasetStep  # noqa: F401 — registers step
from .select import select_and_reject, STRIP_FIELDS, TRAJ_FIELDS, VERIFY_FIELDS
from .reconstruct import (
    reconstruct_fc_json,
    reconstruct_fc_xml,
    reconstruct_policy_fc_json,
    reconstruct_policy_fc_xml,
    reconstruct_thinking_json,
    reconstruct_thinking_xml,
    reconstruct_reasoning,
    reconstruct_reasoning_xml,
    prepend_policy,
    validate_reconstructed,
    check_chat_template,
    check_tool_call_response_pairs,
    check_no_empty_assistant,
)
from .training_format import format_system_prompt

__all__ = [
    # Selection
    "select_and_reject",
    "STRIP_FIELDS",
    "TRAJ_FIELDS",
    "VERIFY_FIELDS",
    # Reconstruction
    "reconstruct_fc_json",
    "reconstruct_fc_xml",
    "reconstruct_policy_fc_json",
    "reconstruct_policy_fc_xml",
    "reconstruct_thinking_json",
    "reconstruct_thinking_xml",
    "reconstruct_reasoning",
    "reconstruct_reasoning_xml",
    "prepend_policy",
    "validate_reconstructed",
    "check_chat_template",
    "check_tool_call_response_pairs",
    "check_no_empty_assistant",
    # Format helpers
    "format_system_prompt",
]
