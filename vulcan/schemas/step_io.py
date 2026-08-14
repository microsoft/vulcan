"""Optional Pydantic models for pipeline step I/O contracts.

These models document the expected shape of inputs and outputs for each
pipeline step.  They are *optional*: steps work without them, but you
can use them for:
  - IDE auto-complete and type checking
  - Explicit validation when debugging a new step
  - Self-documenting pipeline contracts

Usage (optional validation)::

    from vulcan.schemas import GenerateToolsInput

    item = GenerateToolsInput.model_validate(raw_dict)   # raises on bad data
    # ... or just use raw_dict directly in production code

Requires: pydantic >= 2.0
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pydantic is required for schema validation. "
        "Install with: pip install pydantic>=2.0"
    ) from exc


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseStepInput(BaseModel):
    """Fields present on every pipeline item."""

    category_name: str = Field(
        ...,
        description="Environment/category identifier (e.g. 'ticket_api', 'hotel_booking').",
    )
    item_id: str | None = Field(
        default=None,
        alias="_item_id",
        description="Optional unique ID for deduplication and progress tracking.",
    )

    model_config = {"extra": "allow"}  # unknown fields pass through unchanged


# ── Phase 1: generate_tools / verify_tools ───────────────────────────────────

class GenerateToolsInput(BaseStepInput):
    """Input to the generate_tools step."""

    function: dict[str, Any] = Field(
        ...,
        description=(
            "Function specification dict with keys: name, description, "
            "parameters (JSON Schema), response (JSON Schema)."
        ),
    )
    constraints: str | None = Field(
        default=None,
        description="Optional per-API constraint text for this function.",
    )


class GenerateToolsOutput(BaseModel):
    """Output produced by the generate_tools step."""

    tool_code: str = Field(
        ...,
        description="Generated Python function source code as a string.",
    )
    decision: str = Field(
        ...,
        description=(
            "Verification decision after optional verify_tools pass. "
            "One of: 'accepted', 'modified', 'failed'."
        ),
    )
    normalized_function: dict | None = Field(
        default=None,
        description=(
            "Carried through from normalize_specs: the corrected API spec, set "
            "only when the tool's name and description disagreed. Diagnostic — "
            "it records that a correction happened. Steps read `normalized_schema`, "
            "which normalize_specs writes for every tool."
        ),
    )

    model_config = {"extra": "allow"}


# ── Phase 3: generate_trajectories ─────────────────────────────────────────────────────────

class GenerateTrajectoriesInput(BaseStepInput):
    """Input to the generate_trajectories step."""

    tool_code: str = Field(
        ...,
        description="Executable Python environment code (output of generate_tools/verify).",
    )
    initial_state: dict[str, Any] = Field(
        ...,
        description="Initial database/state snapshot for the environment.",
    )
    query: str = Field(
        ...,
        description="User query or task description that the agent must handle.",
    )
    mode: str = Field(
        ...,
        description=(
            "Trajectory mode. One of: 'tools_only', 'with_user_and_policy', 'with_user'. "
            "Determines whether a user agent is simulated."
        ),
    )
    api_list: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of API tool definitions available to the agent.",
    )

    model_config = {"extra": "allow"}


class GenerateTrajectoriesOutput(BaseModel):
    """Output produced by the generate_trajectories step."""

    conversation: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Full conversation history as a list of message dicts "
            "(role, content, tool_calls, tool_call_id, etc.)."
        ),
    )
    thinking_mode: bool = Field(
        default=False,
        description="True if the trajectory was generated with a thinking/reasoning model.",
    )

    model_config = {"extra": "allow"}


# ── Phase 3: judge_trajectories ─────────────────────────────────────────────────────

class JudgeTrajectoriesOutput(BaseModel):
    """Output produced by the judge_trajectories step."""

    judgment: str = Field(
        ...,
        description="Raw judgment text from the verifier model.",
    )
    score: int | None = Field(
        default=None,
        description="Parsed numeric score (1–10). Used by export_dataset threshold.",
    )
    task_completed: bool | None = Field(
        default=None,
        description="Whether the agent successfully completed the task.",
    )

    model_config = {"extra": "allow"}
