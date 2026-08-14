"""Tests for the EnvironmentType enum."""

import pytest

from vulcan.environment.types import EnvironmentType


# ── TOOLS_ONLY properties ─────────────────────────────────────────────────────────

class TestToolsOnlyType:
    def test_has_policy_false(self):
        assert EnvironmentType.TOOLS_ONLY.has_policy is False

    def test_has_user_agent_false(self):
        assert EnvironmentType.TOOLS_ONLY.has_user_agent is False

    def test_environment_type_is_static(self):
        assert EnvironmentType.TOOLS_ONLY.environment_type == "tools_only"

    def test_value(self):
        assert EnvironmentType.TOOLS_ONLY.value == "tools_only"


# ── WITH_USER_AND_POLICY properties ────────────────────────────────────────────

class TestWithUserAndPolicyType:
    def test_has_policy_true(self):
        assert EnvironmentType.WITH_USER_AND_POLICY.has_policy is True

    def test_has_user_agent_true(self):
        assert EnvironmentType.WITH_USER_AND_POLICY.has_user_agent is True

    def test_environment_type_is_dynamic(self):
        assert EnvironmentType.WITH_USER_AND_POLICY.environment_type == "with_user_and_policy"

    def test_value(self):
        assert EnvironmentType.WITH_USER_AND_POLICY.value == "with_user_and_policy"


# ── WITH_USER properties ──────────────────────────────────────────────

class TestWithUserType:
    def test_has_policy_false(self):
        assert EnvironmentType.WITH_USER.has_policy is False

    def test_has_user_agent_true(self):
        assert EnvironmentType.WITH_USER.has_user_agent is True

    def test_environment_type_is_dynamic_no_policy(self):
        assert EnvironmentType.WITH_USER.environment_type == "with_user"

    def test_value(self):
        assert EnvironmentType.WITH_USER.value == "with_user"


# ── from_config: explicit environment_type key ────────────────────────────────

class TestFromConfigExplicit:
    def test_static_explicit(self):
        result = EnvironmentType.from_config({"environment_type": "tools_only"})
        assert result is EnvironmentType.TOOLS_ONLY

    def test_dynamic_with_policy_explicit(self):
        result = EnvironmentType.from_config({"environment_type": "with_user_and_policy"})
        assert result is EnvironmentType.WITH_USER_AND_POLICY

    def test_dynamic_no_policy_explicit(self):
        result = EnvironmentType.from_config({"environment_type": "with_user"})
        assert result is EnvironmentType.WITH_USER

    def test_invalid_environment_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown environment_type"):
            EnvironmentType.from_config({"environment_type": "bogus_type"})

    def test_explicit_overrides_boolean_flags(self):
        # Explicit key takes priority; boolean flags are ignored
        result = EnvironmentType.from_config({
            "environment_type": "tools_only",
            "has_user_agent": True,
            "has_policy": True,
        })
        assert result is EnvironmentType.TOOLS_ONLY


# ── from_config: boolean flags (backward compat) ──────────────────────────────

class TestFromConfigBooleanFlags:
    def test_both_flags_true_gives_dynamic_with_policy(self):
        result = EnvironmentType.from_config({"has_user_agent": True, "has_policy": True})
        assert result is EnvironmentType.WITH_USER_AND_POLICY

    def test_user_agent_only_gives_dynamic_no_policy(self):
        result = EnvironmentType.from_config({"has_user_agent": True, "has_policy": False})
        assert result is EnvironmentType.WITH_USER

    def test_user_agent_only_no_policy_key(self):
        result = EnvironmentType.from_config({"has_user_agent": True})
        assert result is EnvironmentType.WITH_USER

    def test_no_flags_gives_static(self):
        result = EnvironmentType.from_config({})
        assert result is EnvironmentType.TOOLS_ONLY

    def test_both_false_gives_static(self):
        result = EnvironmentType.from_config({"has_user_agent": False, "has_policy": False})
        assert result is EnvironmentType.TOOLS_ONLY

    def test_policy_only_without_user_agent_gives_static(self):
        # Policy without user agent — no enum value exists for this; defaults to TOOLS_ONLY
        result = EnvironmentType.from_config({"has_user_agent": False, "has_policy": True})
        assert result is EnvironmentType.TOOLS_ONLY


# ── Enum identity ─────────────────────────────────────────────────────────────

class TestEnumIdentity:
    def test_all_three_members_exist(self):
        members = list(EnvironmentType)
        assert len(members) == 3

    def test_members_are_distinct(self):
        assert EnvironmentType.TOOLS_ONLY is not EnvironmentType.WITH_USER_AND_POLICY
        assert EnvironmentType.TOOLS_ONLY is not EnvironmentType.WITH_USER
        assert EnvironmentType.WITH_USER_AND_POLICY is not EnvironmentType.WITH_USER

    def test_is_string_enum(self):
        assert isinstance(EnvironmentType.TOOLS_ONLY, str)
