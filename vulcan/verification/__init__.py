"""Verification module for VULCAN pipeline stages.

LLM-based quality verification using per-stage criteria.
"""

from .verifier import StageVerifier
from .criteria import VERIFICATION_CRITERIA

__all__ = ["StageVerifier", "VERIFICATION_CRITERIA"]
