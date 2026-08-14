"""Agents built mid-run must still be wired to their step.

`set_logger` and `_propagate_config_to_agents` find agents by walking
`dir(self)`, so an agent held only in a local variable is invisible to both.
`generate_trajectories` builds all eleven of its agents that way — its system
prompts depend on the item — and the consequences were:

  * every call went uncounted, so `llm_calls=` in that stage's log read 0 even
    on a run that produced thousands of turns; and
  * `empty_response_retries` never arrived, so the agents fell back to the
    `Agent` default of 0 and stopped retrying empty responses, while every other
    stage retried up to 11 times.

The second is a data bug, not an observability one: an empty turn ended the
trajectory instead of being retried.

`BaseStep.make_agent` is the fix. These tests pin both the factory's behaviour
and the rule that the stage never bypasses it.
"""

import asyncio
import inspect
from unittest.mock import Mock

import pytest

from vulcan.steps import STEP_REGISTRY
from vulcan.steps.base import BaseStep


def _step(step_config: dict | None = None, defaults: dict | None = None):
    config = {
        "steps": {"generate_trajectories": step_config or {}},
        "defaults": defaults or {},
    }
    return STEP_REGISTRY["generate_trajectories"](config, Mock(), Mock())


# ── The factory wires what dir(self) cannot reach ─────────────────────────────


class TestMakeAgentWiring:
    def test_agent_receives_the_logger(self):
        step = _step()
        logger = Mock()
        step.set_logger(logger, "ticket_api")
        assert step.make_agent("a", "sys").logger is logger

    def test_agent_receives_step_name_and_category(self):
        step = _step()
        step.set_logger(Mock(), "ticket_api")
        agent = step.make_agent("a", "sys")
        assert agent._step_name == "generate_trajectories"
        assert agent._category == "ticket_api"

    def test_agent_receives_the_retry_budget(self):
        """The data bug: without this the agent silently never retries."""
        step = _step({"empty_response_retries": 7})
        step.set_logger(Mock(), "c")
        assert step.make_agent("a", "sys").empty_response_retries == 7

    def test_retry_budget_arrives_even_without_a_logger(self):
        """A run with logging off must still get the configured retries."""
        step = _step({"empty_response_retries": 4})
        assert step.make_agent("a", "sys").empty_response_retries == 4

    def test_step_config_supplies_the_defaults(self):
        step = _step({"max_tokens": 999, "temperature": 0.7})
        agent = step.make_agent("a", "sys")
        assert agent.max_tokens == 999
        assert agent.temperature == 0.7

    def test_overrides_win(self):
        step = _step({"max_tokens": 999})
        agent = step.make_agent("a", "sys", model="m2", max_tokens=100)
        assert agent.model == "m2"
        assert agent.max_tokens == 100

    def test_zero_temperature_override_is_not_treated_as_missing(self):
        """`0.0` is falsy — the factory must test `is None`, not truthiness."""
        step = _step({"temperature": 0.9})
        assert step.make_agent("a", "sys", temperature=0.0).temperature == 0.0

    def test_zero_retries_can_be_configured_explicitly(self):
        step = _step({"empty_response_retries": 0})
        step.set_logger(Mock(), "c")
        assert step.make_agent("a", "sys").empty_response_retries == 0


# ── End to end: the counter actually increments ───────────────────────────────


class TestLlmCallsIsCounted:
    def _llm_returning(self, content: str):
        llm = Mock()

        async def call(*_a, **_k):
            return {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            }

        llm.call = call
        return llm

    def test_a_factory_built_agent_reports_its_call(self):
        """This is the assertion the bug would fail: llm_calls stayed at 0."""
        step = _step()
        logger = Mock()
        step.set_logger(logger, "ticket_api")
        step.llm = self._llm_returning("hello")

        agent = step.make_agent("traj_agent", "sys")
        asyncio.run(agent.run([{"role": "user", "content": "hi"}]))

        assert logger.llm_call.call_count == 1
        kwargs = logger.llm_call.call_args.kwargs
        assert kwargs["step_name"] == "generate_trajectories"
        assert kwargs["category"] == "ticket_api"
        assert kwargs["total_tokens"] == 12

    def test_an_unwired_agent_reports_nothing(self):
        """Characterises the old behaviour, so the test above cannot pass vacuously."""
        from vulcan.core.agent import Agent

        logger = Mock()
        agent = Agent("traj_agent", "sys", self._llm_returning("hello"))
        asyncio.run(agent.run([{"role": "user", "content": "hi"}]))
        assert logger.llm_call.call_count == 0
        assert agent.empty_response_retries == 0

    def test_empty_responses_are_retried_the_configured_number_of_times(self):
        """The data bug: an unwired agent tried once and gave up."""
        step = _step({"empty_response_retries": 3})
        step.set_logger(Mock(), "c")

        calls = {"n": 0}

        async def call(*_a, **_k):
            calls["n"] += 1
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                    "usage": {}}

        step.llm = Mock()
        step.llm.call = call

        agent = step.make_agent("traj_agent", "sys")
        agent_sleep = asyncio.sleep
        asyncio.sleep = lambda *_a, **_k: agent_sleep(0)  # don't wait out the backoff
        try:
            asyncio.run(agent.run([{"role": "user", "content": "hi"}]))
        finally:
            asyncio.sleep = agent_sleep

        assert calls["n"] == 4, "expected 1 initial attempt + 3 retries"


# ── The stage must not bypass the factory again ───────────────────────────────


class TestGenerateTrajectoriesUsesTheFactory:
    def _source(self) -> str:
        from vulcan.steps.trajectory import generate_trajectories

        return inspect.getsource(generate_trajectories)

    def test_no_direct_agent_construction(self):
        src = self._source()
        assert "Agent(" not in src, (
            "generate_trajectories must build agents via self.make_agent, or they are "
            "invisible to set_logger and lose logging and retries"
        )

    def test_every_agent_goes_through_make_agent(self):
        src = self._source()
        assert src.count("self.make_agent(") == 11, (
            "expected all 11 agent constructions to route through the factory"
        )

    def test_the_factory_exists_on_the_base_class(self):
        assert callable(getattr(BaseStep, "make_agent", None))
        assert callable(getattr(BaseStep, "_wire_agent", None))


class TestNoOtherStepBypassesIt:
    """Steps that build agents in __init__ are fine — dir(self) finds those."""

    @pytest.mark.parametrize("step_name", sorted(STEP_REGISTRY))
    def test_any_direct_agent_construction_is_on_self(self, step_name):
        module = inspect.getmodule(STEP_REGISTRY[step_name])
        try:
            src = inspect.getsource(module)
        except OSError:  # pragma: no cover
            pytest.skip("source unavailable")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("agent") and "= Agent(" in stripped:
                pytest.fail(
                    f"{step_name}: local Agent construction — use self.make_agent"
                )
