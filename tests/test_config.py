"""Tests for config loading."""

import os

import pytest
import yaml

from vulcan.config import load_config, get_step_config


# `mock_test` is the smoke-test config that ships inside the package, so it is
# the one config every checkout can load without extra setup.
SHIPPED_ENV = "mock_test"


# ── load_config: base structure ───────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_shipped_config(self):
        cfg = load_config(SHIPPED_ENV)
        assert isinstance(cfg, dict)

    def test_has_llm_section(self):
        cfg = load_config(SHIPPED_ENV)
        assert "llm" in cfg

    def test_llm_has_provider(self):
        cfg = load_config(SHIPPED_ENV)
        assert "provider" in cfg["llm"]

    def test_llm_has_model(self):
        cfg = load_config(SHIPPED_ENV)
        assert isinstance(cfg["llm"]["model"], str)
        assert cfg["llm"]["model"]

    def test_llm_inherits_request_settings_from_base(self):
        cfg = load_config(SHIPPED_ENV)
        # base.yaml supplies these; the per-env file only overrides provider/model.
        assert "request_timeout" in cfg["llm"]
        assert "max_retries" in cfg["llm"]

    def test_no_api_key_in_config(self):
        """Credentials come from the environment, never from a config file."""
        cfg = load_config(SHIPPED_ENV)
        assert "api_key" not in cfg["llm"]

    def test_has_defaults_section(self):
        cfg = load_config(SHIPPED_ENV)
        assert "defaults" in cfg

    def test_defaults_does_not_pin_a_model(self):
        """`defaults.model` must stay unset in the shipped configs.

        A step resolves steps.<step>.model -> defaults.model -> "", and an empty
        value means "use the model this stage's client was built with". Setting
        defaults.model would override llm.model for every stage, sending one
        provider's model name to whichever provider is configured.
        """
        cfg = load_config(SHIPPED_ENV)
        assert not cfg["defaults"].get("model")

    def test_llm_block_supplies_the_model(self):
        cfg = load_config(SHIPPED_ENV)
        assert cfg["llm"]["model"]

    def test_defaults_has_temperature(self):
        cfg = load_config(SHIPPED_ENV)
        assert "temperature" in cfg["defaults"]

    def test_defaults_has_max_tokens(self):
        cfg = load_config(SHIPPED_ENV)
        assert "max_tokens" in cfg["defaults"]

    def test_defaults_has_parallel_workers(self):
        cfg = load_config(SHIPPED_ENV)
        assert "parallel_workers" in cfg["defaults"]

    def test_has_steps_section(self):
        cfg = load_config(SHIPPED_ENV)
        assert "steps" in cfg

    def test_steps_has_generate_tools(self):
        cfg = load_config(SHIPPED_ENV)
        assert "generate_tools" in cfg["steps"]

    def test_steps_has_no_python_api_gen(self):
        """The API-generation step is named generate_tools."""
        cfg = load_config(SHIPPED_ENV)
        assert "python_api_gen" not in cfg["steps"]

    def test_steps_has_verify_tools(self):
        cfg = load_config(SHIPPED_ENV)
        assert "verify_tools" in cfg["steps"]

    def test_steps_has_generate_trajectories(self):
        cfg = load_config(SHIPPED_ENV)
        assert "generate_trajectories" in cfg["steps"]

    def test_steps_has_judge_trajectories(self):
        cfg = load_config(SHIPPED_ENV)
        assert "judge_trajectories" in cfg["steps"]

    def test_steps_has_export_dataset(self):
        cfg = load_config(SHIPPED_ENV)
        assert "export_dataset" in cfg["steps"]

    def test_plan_sequences_has_multipath_keys(self):
        cfg = load_config(SHIPPED_ENV)
        plan_sequences = cfg["steps"]["plan_sequences"]
        assert "enable_multipath" in plan_sequences
        assert "max_multipath_tools" in plan_sequences

    def test_unknown_env_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_environment_xyz")

    def test_not_found_message_lists_available_environments(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("nonexistent_environment_xyz")
        assert SHIPPED_ENV in str(exc_info.value)

    def test_returns_dict(self):
        cfg = load_config(SHIPPED_ENV)
        assert isinstance(cfg, dict)


# ── load_config: path resolution ──────────────────────────────────────────────

class TestPathResolution:
    def test_output_base_is_absolute(self):
        cfg = load_config(SHIPPED_ENV)
        assert os.path.isabs(cfg["output_base"])

    def test_input_catalog_is_absolute(self):
        cfg = load_config(SHIPPED_ENV)
        assert os.path.isabs(cfg["input"]["catalog"])

    def test_input_points_at_the_example_catalog(self):
        cfg = load_config(SHIPPED_ENV)
        catalog = cfg["input"]["catalog"]
        assert catalog.endswith(os.path.join("examples", "ticket_api.jsonl"))
        assert os.path.isfile(catalog), (
            "mock_test should point at a catalog that ships with the repo"
        )

    def test_home_relative_paths_are_expanded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "tilde_env.yaml").write_text(
            yaml.safe_dump({"output_base": "~/vulcan_runs"}), encoding="utf-8"
        )
        cfg = load_config("tilde_env")
        assert "~" not in cfg["output_base"]
        assert os.path.isabs(cfg["output_base"])

    def test_relative_output_base_resolves_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "rel_env.yaml").write_text(
            yaml.safe_dump({"output_base": "./outputs/rel_env_fixture"}), encoding="utf-8"
        )
        cfg = load_config("rel_env")
        expected = os.path.join(os.getcwd(), "outputs", "rel_env_fixture")
        assert os.path.realpath(cfg["output_base"]) == os.path.realpath(expected)


# ── load_config: search order ─────────────────────────────────────────────────

class TestConfigSearchOrder:
    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    def test_explicit_path_is_loaded(self, tmp_path):
        path = tmp_path / "somewhere" / "custom.yaml"
        self._write(path, {"environment": "custom_env"})
        cfg = load_config(str(path))
        assert cfg["environment"] == "custom_env"

    def test_explicit_path_still_merges_base(self, tmp_path):
        path = tmp_path / "custom.yaml"
        self._write(path, {"environment": "custom_env"})
        cfg = load_config(str(path))
        assert "llm" in cfg
        assert "steps" in cfg

    def test_local_configs_dir_is_searched(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write(tmp_path / "configs" / "local_env.yaml", {"environment": "local_env"})
        cfg = load_config("local_env")
        assert cfg["environment"] == "local_env"

    def test_local_configs_dir_wins_over_packaged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write(
            tmp_path / "configs" / f"{SHIPPED_ENV}.yaml",
            {"environment": "overridden_locally"},
        )
        cfg = load_config(SHIPPED_ENV)
        assert cfg["environment"] == "overridden_locally"

    def test_packaged_config_found_from_any_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config(SHIPPED_ENV)
        assert cfg["environment"] == SHIPPED_ENV


# ── get_step_config: merges defaults with step overrides ─────────────────────

class TestGetStepConfig:
    def test_returns_dict(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_tools")
        assert isinstance(result, dict)

    def test_step_config_leaves_model_unpinned(self):
        """A step that does not name a model inherits the client's model."""
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_tools")
        assert not result.get("model")

    def test_step_specific_model_is_honoured(self):
        cfg = load_config(SHIPPED_ENV)
        cfg.setdefault("steps", {}).setdefault("generate_tools", {})["model"] = "some-model"
        result = get_step_config(cfg, "generate_tools")
        assert result["model"] == "some-model"

    def test_step_specific_max_tokens_overrides_default(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_tools")
        # generate_tools sets max_tokens: 32000 in base.yaml
        assert result["max_tokens"] == 32000

    def test_default_temperature_inherited_when_not_overridden(self):
        cfg = load_config(SHIPPED_ENV)
        # judge_trajectories does not override parallel_workers, should inherit default
        result = get_step_config(cfg, "judge_trajectories")
        assert result["parallel_workers"] == cfg["defaults"]["parallel_workers"]

    def test_step_temperature_overrides_default(self):
        cfg = load_config(SHIPPED_ENV)
        # generate_trajectories sets temperature: 0.7
        result = get_step_config(cfg, "generate_trajectories")
        assert result["temperature"] == pytest.approx(0.7)

    def test_generate_states_temperature_override(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_states")
        assert result["temperature"] == pytest.approx(0.8)

    def test_unknown_step_returns_defaults(self):
        cfg = load_config(SHIPPED_ENV)
        # A step with no override should still return defaults
        result = get_step_config(cfg, "unknown_step_xyz")
        assert result == cfg["defaults"]

    def test_plan_sequences_has_max_tools_per_sequence(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "plan_sequences")
        assert "max_tools_per_sequence" in result

    def test_does_not_modify_original_config(self):
        cfg = load_config(SHIPPED_ENV)
        original_default_temp = cfg["defaults"]["temperature"]
        get_step_config(cfg, "generate_trajectories")
        # Ensure deep copy did not mutate the original config
        assert cfg["defaults"]["temperature"] == original_default_temp


# ── deep merge behavior ────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_step_key_survives_merge(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_tools")
        # generate_tools has temperature:0 — this is the step override
        assert result["temperature"] == 0

    def test_default_keys_not_in_step_are_inherited(self):
        cfg = load_config(SHIPPED_ENV)
        result = get_step_config(cfg, "generate_tools")
        # max_retries is a defaults key not in generate_tools overrides
        assert "max_retries" in result
        assert result["max_retries"] == cfg["defaults"]["max_retries"]

    def test_nested_llm_block_is_merged_not_replaced(self):
        cfg = load_config(SHIPPED_ENV)
        # The per-env file sets only provider/model; base.yaml keys survive.
        assert cfg["llm"]["model"] == "gpt-4o"
        assert cfg["llm"]["request_timeout"] == 600
