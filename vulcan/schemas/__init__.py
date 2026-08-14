"""Schemas package for VULCAN.

Contains optional Pydantic models for step I/O contracts.
These models are used for documentation and optional validation;
pipeline steps work without them.

Modules
-------
step_io   -- Input/output models for pipeline steps
"""

from .step_io import (
    BaseStepInput,
    GenerateToolsInput,
    GenerateToolsOutput,
    GenerateTrajectoriesInput,
    GenerateTrajectoriesOutput,
    JudgeTrajectoriesOutput,
)

__all__ = [
    "BaseStepInput",
    "GenerateToolsInput",
    "GenerateToolsOutput",
    "GenerateTrajectoriesInput",
    "GenerateTrajectoriesOutput",
    "JudgeTrajectoriesOutput",
]
