"""A `with_user_and_policy` record must carry its policy.

The agent was driven under the policy and `judge_trajectories` scored the
transcript against it. If the exported record omits it, training on that record
shows the model policy-compliant behaviour with no policy in context — which
teaches the domain's rules as unconditional habits instead of as
instruction-following. `reconstruct_policy_fc_json` / `_xml` existed for this and
were imported but never called.
"""

import asyncio

import pytest

from vulcan.steps.output import prepend_policy
from vulcan.steps.output import export_dataset as ed

POLICY = "Rule 1: always confirm the city.\nRule 2: never book without a date."


class TestPrependPolicy:
    """Used by the reasoning and tagged formats, whose templates have no slot."""

    def test_policy_goes_first_in_the_system_message(self):
        out = prepend_policy([{"role": "system", "content": "You are helpful."}], POLICY)
        assert out[0]["content"] == f"{POLICY}\n\nYou are helpful."

    def test_no_policy_is_a_no_op(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        assert prepend_policy(msgs, "") == msgs
        assert prepend_policy(msgs, "   ") == msgs

    def test_it_never_adds_a_second_system_message(self):
        out = prepend_policy([{"role": "system", "content": "x"},
                              {"role": "user", "content": "hi"}], POLICY)
        assert sum(1 for m in out if m["role"] == "system") == 1

    def test_a_conversation_with_no_system_message_gets_one(self):
        out = prepend_policy([{"role": "user", "content": "hi"}], POLICY)
        assert out[0] == {"role": "system", "content": POLICY}
        assert len(out) == 2

    def test_it_is_idempotent(self):
        once = prepend_policy([{"role": "system", "content": "x"}], POLICY)
        assert prepend_policy(once, POLICY) == once

    def test_it_does_not_mutate_the_input(self):
        msgs = [{"role": "system", "content": "x"}]
        prepend_policy(msgs, POLICY)
        assert msgs == [{"role": "system", "content": "x"}]


class _Env:
    def __init__(self, env_type):
        self._t = env_type

    def get_environment_type(self, category):
        return self._t

    def get_domain_rules(self, category):
        return POLICY


def _item(**extra):
    item = {
        "_item_id": "i1",
        "category_name": "hotel_booking",
        "judgment": {"score": 10, "task_completed": True},
    }
    item.update(extra)
    return item


def _run(env_type, monkeypatch, *, thinking=False, tagged=False):
    """Drive export_dataset with the reconstructors stubbed to a known shape."""
    stub = [{"role": "system", "content": "BASE"}, {"role": "user", "content": "hi"}]
    for name in ("reconstruct_reasoning", "reconstruct_reasoning_xml",
                 "reconstruct_prompting", "reconstruct_fc_json", "reconstruct_fc_xml"):
        monkeypatch.setattr(ed, name, lambda inp, _s=stub: [dict(m) for m in _s])
    for name in ("reconstruct_policy_fc_json", "reconstruct_policy_fc_xml"):
        monkeypatch.setattr(
            ed, name,
            lambda inp, rules, _s=stub: [{"role": "system", "content": f"{rules}\n\nBASE"},
                                         *[dict(m) for m in _s[1:]]])
    monkeypatch.setattr(ed, "validate_reconstructed", lambda msgs: (True, []))

    step = ed.ExportDatasetStep({}, None, _Env(env_type))
    extra = {}
    if thinking:
        extra["thinking_mode"] = True
    if tagged:
        extra["tool_call_format"] = "tagged"
    out = asyncio.run(step.run(_item(**extra)))
    return out["_training_json"]["messages"][0]["content"]


class TestPolicyReachesEveryFormat:
    @pytest.mark.parametrize("kwargs", [{}, {"thinking": True}, {"tagged": True}])
    def test_with_user_and_policy_carries_the_policy(self, monkeypatch, kwargs):
        system = _run("with_user_and_policy", monkeypatch, **kwargs)
        assert POLICY in system, f"policy missing from the exported record for {kwargs or 'native FC'}"
        assert system.startswith(POLICY)

    @pytest.mark.parametrize("env_type", ["tools_only", "with_user"])
    @pytest.mark.parametrize("kwargs", [{}, {"thinking": True}, {"tagged": True}])
    def test_other_environment_types_are_untouched(self, monkeypatch, env_type, kwargs):
        """Only with_user_and_policy has a policy; nothing else gains a prefix."""
        assert _run(env_type, monkeypatch, **kwargs) == "BASE"


class TestTheDedicatedTemplatesAreUsed:
    def test_native_fc_goes_through_the_policy_reconstructors(self, monkeypatch):
        called = []
        for name in ("reconstruct_fc_json", "reconstruct_policy_fc_json"):
            monkeypatch.setattr(ed, name,
                                lambda *a, _n=name, **k: called.append(_n) or
                                [{"role": "system", "content": "S"}])
        monkeypatch.setattr(ed, "reconstruct_fc_xml", lambda *a, **k: [])
        monkeypatch.setattr(ed, "reconstruct_policy_fc_xml", lambda *a, **k: [])
        monkeypatch.setattr(ed, "validate_reconstructed", lambda msgs: (True, []))

        step = ed.ExportDatasetStep({}, None, _Env("with_user_and_policy"))
        asyncio.run(step.run(_item()))
        assert called == ["reconstruct_policy_fc_json"], (
            "native FC must use the purpose-built policy template, not a prepend"
        )
