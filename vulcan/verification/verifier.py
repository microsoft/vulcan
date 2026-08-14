"""LLM-based stage verifier for VULCAN pipeline outputs."""

import json
import logging
from typing import Any

from ..core.exceptions import LLMError
from .criteria import VERIFICATION_CRITERIA

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """You are an expert software quality reviewer for a synthetic data generation pipeline.
Evaluate whether the output meets ALL quality criteria.
Respond with JSON: {"passed": true/false, "issues": ["issue 1", ...], "confidence": 0.0-1.0}"""


class StageVerifier:
    """LLM-based verifier that checks pipeline stage outputs against quality criteria.

    Sends each item to the configured model and evaluates it against the
    per-stage check_items.  Only relevant fields from the item are sent to
    keep prompts small.
    """

    def __init__(self, llm: Any, config: dict):
        """Initialize the verifier.

        Args:
            llm: Async HTTP client or query function for LLM calls.
            config: Verification config dict. Supported keys:
                - model (str): Model name. Omit to use the configured provider's
                  own default — never hardcode a vendor model here, or it would
                  be sent verbatim to whichever provider is configured.
                - max_tokens (int): Max response tokens, default 512
                - temperature (float): Sampling temperature, default 0.0
        """
        self.llm = llm
        self.model = config.get("model") or None
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.0)

    async def verify(self, step_name: str, item: dict) -> tuple[bool, str]:
        """Evaluate a pipeline output item against the criteria for its step.

        Args:
            step_name: The pipeline stage name (e.g. "generate_tools", "generate_trajectories").
            item: The output item dict from that stage.

        Returns:
            (passed, reason) — passed=True if all criteria met, reason describes issues.

        Raises:
            LLMError: the provider was unreachable or failed after its retries.
                This is not a verdict about *item*, so it propagates instead of
                being reported as a verification failure — otherwise a provider
                outage would silently reject every item in the run.
        """
        criteria = VERIFICATION_CRITERIA.get(step_name)
        if criteria is None:
            return True, f"no criteria defined for step '{step_name}'"

        relevant = self._extract_relevant_fields(step_name, item)
        prompt = self._build_prompt(step_name, relevant, criteria)

        try:
            # StageRunner passes a LLMClient, whose interface is .call(messages, ...)
            # — the instance is not callable.
            resp = await self.llm.call(
                [{"role": "system", "content": VERIFIER_SYSTEM},
                 {"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return self._parse_verdict(resp["choices"][0]["message"]["content"])
        except LLMError:
            # Transport / provider failure — the model never returned a verdict.
            # Reporting False here would mark a perfectly good item as rejected,
            # so let the caller see the outage.
            logger.error(
                "verifier: LLM provider error for step=%s — re-raising, not failing the item",
                step_name,
            )
            raise
        except Exception as e:
            logger.warning("verifier call failed for step=%s: %s", step_name, e)
            return False, f"verifier error: {e}"

    def _build_prompt(self, step_name: str, item: dict, criteria: dict) -> str:
        """Build a focused verification prompt for the given step and item.

        Args:
            step_name: Pipeline stage name.
            item: Relevant fields extracted from the output item.
            criteria: Dict with "check_items" list.

        Returns:
            Formatted user prompt string.
        """
        check_items = criteria.get("check_items", [])
        checklist = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(check_items))
        item_str = json.dumps(item, indent=2, ensure_ascii=False)
        return (
            f"Stage: {step_name}\n\n"
            f"Quality criteria to verify:\n{checklist}\n\n"
            f"Output to evaluate:\n{item_str}\n\n"
            "Does this output meet ALL criteria? "
            'Respond with JSON: {"passed": true/false, "issues": [...], "confidence": 0.0-1.0}'
        )

    def _extract_relevant_fields(self, step_name: str, item: dict) -> dict:
        """Extract only the fields relevant to verifying a given stage.

        Keeps prompts small by dropping fields that are not useful for verification.

        Args:
            step_name: Pipeline stage name.
            item: Full output item dict.

        Returns:
            Reduced dict with only the relevant fields.
        """
        # Fields relevant per stage — drop large unrelated keys
        stage_fields = {
            "generate_tools": ["tool_code", "decision"],
            "verify_tools": ["tool_code", "decision", "check_name_description"],
            "assemble_environment": ["tool_code", "decision"],
            "debug_environment": ["tool_code", "environment_ready", "test_iterations"],
            "generate_states": ["initial_state"],
            "sample_arguments": ["functions", "initial_state"],
            "build_graph": ["graph"],
            "score_sequences": ["judgment"],
            "generate_trajectories": ["conversation"],
            "judge_trajectories": ["judgment"],
            "generate_queries": ["query"],
        }

        fields = stage_fields.get(step_name)
        if fields is None:
            # No field filter defined — return as-is (truncated)
            return dict(list(item.items())[:10])

        result = {}
        for f in fields:
            if f in item:
                val = item[f]
                # Truncate very long string fields to avoid huge prompts
                if isinstance(val, str) and len(val) > 3000:
                    val = val[:3000] + "\n... [truncated]"
                result[f] = val
        return result

    def _parse_verdict(self, result: str) -> tuple[bool, str]:
        """Parse the LLM verifier response into (passed, reason).

        Expects JSON: {"passed": bool, "issues": [...], "confidence": float}

        Args:
            result: Raw string response from the LLM.

        Returns:
            (passed, reason) tuple.
        """
        if not result:
            return False, "empty verifier response"

        # Try to extract JSON from response
        text = result.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except (json.JSONDecodeError, ValueError):
                    return False, f"could not parse verifier response: {text[:200]}"
            else:
                return False, f"no JSON found in verifier response: {text[:200]}"

        passed = bool(data.get("passed", False))
        issues = data.get("issues", [])
        confidence = data.get("confidence", 1.0)

        if passed:
            return True, f"passed (confidence={confidence:.2f})"
        else:
            issues_str = "; ".join(issues) if issues else "no specific issues listed"
            return False, f"failed (confidence={confidence:.2f}): {issues_str}"
