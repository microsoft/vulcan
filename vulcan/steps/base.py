"""Base step class for all VULCAN pipeline steps."""

from __future__ import annotations

import os
from typing import Any

from ..llm import LLMClient
from ..core.exceptions import StepError, ValidationError
from ..environment.base import EnvironmentConfig


class BaseStep:
    """Base class for all pipeline steps.

    All concrete step classes must:
    1. Be decorated with ``@register_step("<name>")`` from vulcan.steps.
    2. Call ``super().__init__(config, llm, env)`` in their ``__init__``.
    3. Override ``async run(self, item, progress=None) -> dict``.

    Optional class attributes (override in subclass for schema-driven validation):
        input_schema:  a Pydantic model class or None
        output_schema: a Pydantic model class or None

    Step ordering and input resolution live in ``core/stage_runner.py``
    (``PIPELINE_ORDER`` and ``_STANDARD_INPUT``); steps do not describe their own
    predecessors.
    """

    name: str = ""
    requires_llm: bool = True

    # Optional Pydantic schemas — set in subclass for validate_input()
    input_schema: Any = None
    output_schema: Any = None

    def __init__(
        self,
        config: dict,
        llm: LLMClient,
        env: EnvironmentConfig,
    ) -> None:
        self.config = config
        self.llm = llm
        self.env = env

        self.step_config: dict = config.get("steps", {}).get(self.name, {})

        # Resolve model / generation settings: step-specific > defaults
        defaults = config.get("defaults", {})
        # No hardcoded model fallback: an empty value means "use whatever the
        # configured client defaults to", and the client raises a clear error
        # when nothing is configured at all.
        self.model: str = self.step_config.get("model") or defaults.get("model") or ""
        self.temperature: float = self.step_config.get(
            "temperature", defaults.get("temperature", 0)
        )
        self.max_tokens: int = self.step_config.get(
            "max_tokens", defaults.get("max_tokens", 16000)
        )
        self.thinking_model: bool = self.step_config.get(
            "thinking_model", defaults.get("thinking_model", False)
        )
        self.empty_response_retries: int = self.step_config.get(
            "empty_response_retries", defaults.get("empty_response_retries", 10)
        )

        # Set by runner before processing begins
        self.logger = None
        self._category: str | None = None
        self.out_dir: str | None = None

    # ── Config propagation ─────────────────────────────────────────────────

    def _propagate_config_to_agents(self) -> None:
        """Propagate empty_response_retries to all Agent attributes.

        Called automatically after ``__init__`` completes.  Subclasses may
        call this manually if they create agents conditionally.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if hasattr(attr, "empty_response_retries") and hasattr(attr, "_step_name"):
                attr.empty_response_retries = self.empty_response_retries

    def set_logger(self, logger, category: str | None = None) -> None:
        """Propagate logger (and related config) to this step and all its agents."""
        self.logger = logger
        self._category = category
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if hasattr(attr, "logger") and hasattr(attr, "_step_name"):
                self._wire_agent(attr)

    def _wire_agent(self, agent) -> None:
        """Attach this step's logger and retry budget to *agent*."""
        agent.logger = self.logger
        agent._step_name = self.name
        agent._category = self._category
        agent.empty_response_retries = self.empty_response_retries

    def make_agent(
        self,
        name: str,
        system_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking_model: bool | None = None,
        output_parser=None,
    ):
        """Build an Agent that is wired to this step.

        Use this instead of constructing ``Agent`` directly whenever the agent
        cannot be created in ``__init__`` — for example when its system prompt
        depends on the item being processed.

        ``set_logger`` and ``_propagate_config_to_agents`` discover agents by
        walking ``dir(self)``, so an agent held only in a local variable is
        invisible to them: it never receives the logger (its calls go uncounted
        in ``llm_calls=``) and never receives ``empty_response_retries``, so it
        silently falls back to the ``Agent`` default of 0 and stops retrying
        empty responses. Going through this factory avoids both.

        Omitted arguments default to the step's resolved config.
        """
        from ..core.agent import Agent

        agent = Agent(
            name,
            system_prompt,
            self.llm,
            model=self.model if model is None else model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            thinking_model=self.thinking_model if thinking_model is None else thinking_model,
            output_parser=output_parser,
        )
        self._wire_agent(agent)
        return agent

    # ── Input preprocessing ────────────────────────────────────────────────

    @staticmethod
    def preprocess_input(data: list[dict]) -> list[dict]:
        """Transform loaded input data before the main processing loop.

        Default: pass through unchanged.  Override in subclasses that need to
        flatten category-level data into individual items (e.g. generate_tools,
        generate_states) or explode one item into multiple (generate_states).
        """
        return data

    # ── Schema validation (optional) ──────────────────────────────────────

    def validate_input(self, item: dict) -> dict:
        """Validate *item* against ``input_schema`` if defined.

        Returns the (possibly coerced) item dict, or raises ValidationError.
        No-op when ``input_schema`` is None.
        """
        if self.input_schema is None:
            return item
        try:
            validated = self.input_schema.model_validate(item)
            return validated.model_dump(mode="python")
        except Exception as exc:
            raise ValidationError(
                f"Input validation failed for step {self.name!r}: {exc}",
                field="input",
                value=item,
            ) from exc

    # ── Core processing ────────────────────────────────────────────────────

    async def run(self, item: dict, progress=None) -> dict:
        """Process one input item.  Must be overridden in every concrete step."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    # ── Output validation ──────────────────────────────────────────────────

    def validate_output(self, item: dict) -> tuple[bool, str]:
        """Run deterministic validation on an output item.

        Checks the global VALIDATORS registry first (for backward compat),
        then falls back to ``output_schema`` if defined.

        Returns ``(passed: bool, reason: str)``.
        """
        # Try registered validator (deterministic.py style)
        try:
            from ..validators.deterministic import VALIDATORS  # type: ignore[import]
            validator = VALIDATORS.get(self.name)
            if validator:
                return validator(item)
        except ImportError:
            pass

        # Try output_schema validation
        if self.output_schema is not None:
            try:
                self.output_schema.model_validate(item)
                return True, ""
            except Exception as exc:
                return False, str(exc)

        return True, ""

    # ── Path helpers ───────────────────────────────────────────────────────

    def get_output_dir(self) -> str:
        """Return the output directory for this step."""
        base = self.config.get("output_base") or os.path.join(os.getcwd(), "outputs")
        return os.path.join(base, self.name)

    def get_output_path(self) -> str:
        """Return the default output file path for this step."""
        return os.path.join(self.get_output_dir(), f"final_out_{self.name}.jsonl")
