"""End-to-end checks on the configs that ship with the repo.

These exist because the rest of the suite tests units in isolation and therefore
missed a `NameError` in `EnvironmentConfig._load_data` that made every real run
fail before its first step. Anything a `vulcan --env <name>` invocation touches
before it starts calling a model belongs here.

No network and no API key: config loading, catalog parsing and environment
construction are all local.
"""

import pytest

from vulcan.config import load_config
from vulcan.environment import get_environment
from vulcan.steps import PIPELINE_ORDER

# Every config a user can select out of the box.
SHIPPED = ["quickstart", "hotel_policy", "mock_test"]

# What each shipped config is expected to describe.
EXPECTED = {
    "quickstart":   ("ticket_api",    8,  "tools_only",           False),
    "hotel_policy": ("hotel_booking", 10, "with_user_and_policy", True),
    "mock_test":    ("ticket_api",    8,  "tools_only",           False),
}


@pytest.mark.parametrize("name", SHIPPED)
class TestShippedConfigLoads:
    def test_config_loads(self, name):
        assert load_config(name)

    def test_environment_constructs(self, name):
        """The step that used to raise NameError before any stage ran."""
        assert get_environment(name, load_config(name)) is not None

    def test_catalog_is_found_and_parsed(self, name):
        env = get_environment(name, load_config(name))
        category, n_tools, _, _ = EXPECTED[name]
        assert env.get_categories() == [category]
        assert len(env.get_api_catalog(category)) == n_tools

    def test_environment_type_resolves(self, name):
        env = get_environment(name, load_config(name))
        category, _, expected_type, _ = EXPECTED[name]
        assert env.get_environment_type(category) == expected_type

    def test_policy_presence_matches_the_environment_type(self, name):
        env = get_environment(name, load_config(name))
        category, _, _, has_policy = EXPECTED[name]
        assert bool(env.get_domain_rules(category)) is has_policy

    def test_declared_categories_exist_in_the_catalog(self, name):
        config = load_config(name)
        env = get_environment(name, config)
        for declared in config.get("categories", []):
            assert declared in env.get_categories(), (
                f"{name} declares category {declared!r} that its catalog does not define"
            )

    def test_every_step_key_names_a_real_step(self, name):
        """A typo under `steps:` is silent — the config key is simply never read."""
        for key in (load_config(name).get("steps") or {}):
            assert key in PIPELINE_ORDER, f"{name} configures unknown step {key!r}"


class TestEnvironmentVocabulary:
    """There is one set of environment-type names, not two."""

    RETIRED = {
        "static", "dynamic", "dynamic_no_policy",
        "env_wo_user_wo_policy", "env_w_user_w_policy", "env_w_user_wo_policy",
    }

    @pytest.mark.parametrize("name", SHIPPED)
    def test_no_retired_environment_type_in_shipped_configs(self, name):
        declared = load_config(name).get("environment_type")
        assert declared not in self.RETIRED, (
            f"{name} uses the retired environment vocabulary: {declared!r}"
        )

    @pytest.mark.parametrize("name", SHIPPED)
    def test_resolved_type_is_never_retired(self, name):
        env = get_environment(name, load_config(name))
        for category in env.get_categories():
            assert env.get_environment_type(category) not in self.RETIRED


class TestToolCallFormat:
    """`tool_call_format` values must be ones the CLI and the step both accept."""

    VALID = {"native", "reasoning", "tagged"}

    @pytest.mark.parametrize("name", SHIPPED)
    def test_configured_format_is_valid(self, name):
        step = (load_config(name).get("steps") or {}).get("generate_trajectories") or {}
        configured = step.get("tool_call_format")
        if configured is not None:
            assert configured in self.VALID

    def test_cli_fallback_is_a_valid_format(self):
        """The thinking-model fallback must not return a retired value."""
        from vulcan.cli import _resolve_output_label

        assert _resolve_output_label({"defaults": {"thinking_model": True}}, None) in self.VALID
        assert _resolve_output_label({"defaults": {"thinking_model": False}}, None) in self.VALID
        assert _resolve_output_label({}, "tagged") == "tagged"
