"""Configuration loading and merging for VULCAN.

Public surface
--------------
load_config(name)               -> dict   load base.yaml + per-env yaml, deep merge
get_step_config(config, step)   -> dict   merge defaults + step overrides
"""

import os
from copy import deepcopy

import yaml


_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo root in a source checkout; the install prefix for a pip-installed
# package. Used only as a fallback when a relative path does not resolve
# against the working directory (see :func:`_resolve_paths`).
_PKG_ROOT = os.path.dirname(os.path.dirname(_CONFIG_DIR))

# Where per-environment configs are looked up, in order, before falling back to
# the packaged config directory.
_LOCAL_CONFIG_DIRNAME = "configs"

# Not selectable as an environment: base.yaml is always loaded, and the example
# is a template to copy rather than a runnable config.
_NON_ENVIRONMENT_FILES = ("base.yaml", "example_environment.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*.

    Dicts are merged recursively; all other types are replaced by the override value.
    Neither input dict is modified.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _env_config_candidates(name: str) -> list[str]:
    """Return the paths ``load_config`` will try for *name*, in priority order."""
    expanded = os.path.expanduser(name)
    candidates: list[str] = []

    # (1) An explicit path to a config file, relative or absolute.
    if os.path.isfile(expanded):
        candidates.append(os.path.abspath(expanded))

    # (2) ./configs/<name>.yaml next to wherever the command was run — this is
    #     how a pip-installed VULCAN picks up configs the user wrote.
    local_dir = os.path.join(os.getcwd(), _LOCAL_CONFIG_DIRNAME)
    candidates.append(os.path.join(local_dir, f"{name}.yaml"))
    candidates.append(os.path.join(local_dir, f"{name}.yml"))

    # (3) The configs shipped inside the package.
    candidates.append(os.path.join(_CONFIG_DIR, f"{name}.yaml"))

    return candidates


def _available_environments() -> list[str]:
    """Names that :func:`load_config` can resolve right now, deduplicated."""
    found: list[str] = []

    for directory in (os.path.join(os.getcwd(), _LOCAL_CONFIG_DIRNAME), _CONFIG_DIR):
        if not os.path.isdir(directory):
            continue
        for f in sorted(os.listdir(directory)):
            if not f.endswith((".yaml", ".yml")):
                continue
            if directory == _CONFIG_DIR and f in _NON_ENVIRONMENT_FILES:
                continue
            stem = os.path.splitext(f)[0]
            if stem not in found:
                found.append(stem)

    return found


def load_config(name: str) -> dict:
    """Load base config merged with environment-specific config.

    Reads the packaged ``base.yaml`` first, then overlays the per-environment
    file. *name* is resolved in this order:

    1. an explicit path, if *name* points at an existing file;
    2. ``./configs/{name}.yaml`` relative to the current working directory;
    3. ``{name}.yaml`` inside this package's config directory.

    Args:
        name: Environment identifier (e.g. ``"mock_test"``, ``"my_env"``), or a
            path to a YAML file.

    Returns:
        Merged config dict with keys such as: ``llm``, ``defaults``,
        ``steps``, ``input``, ``output_base``, ``categories``, etc.

    Raises:
        FileNotFoundError: If no config can be resolved for *name*.
    """
    base_path = os.path.join(_CONFIG_DIR, "base.yaml")

    with open(base_path, encoding="utf-8") as fh:
        base: dict = yaml.safe_load(fh) or {}

    candidates = _env_config_candidates(name)
    env_path = next((p for p in candidates if os.path.isfile(p)), None)

    if env_path is None:
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"Environment config not found: {name}\n"
            f"Searched:\n{searched}\n"
            f"Available configs: {_available_environments()}\n"
            f"To add a new environment, copy config/example_environment.yaml "
            f"to ./{_LOCAL_CONFIG_DIRNAME}/{name}.yaml"
        )

    with open(env_path, encoding="utf-8") as fh:
        env_cfg: dict = yaml.safe_load(fh) or {}

    merged = _deep_merge(base, env_cfg)
    return _resolve_paths(merged)


def _resolve_paths(config: dict) -> dict:
    """Expand ``~`` and make relative paths absolute.

    Lets a config carry portable values (``./outputs/run``,
    ``examples/ticket_api.jsonl``) while the pipeline still receives the
    absolute paths it requires — vulcan joins ``output_base`` directly, so an
    unexpanded ``~`` would create a directory literally named ``~``.

    A relative path is resolved against the current working directory when
    something exists there, and otherwise against the package root, which is
    the repo root in a source checkout. Preferring the working directory is
    what keeps these configs usable from a pip-installed package, where no data
    files sit next to the installed package.
    """
    def _abs(p: str) -> str:
        p = os.path.expanduser(p)
        if os.path.isabs(p):
            return p
        cwd_path = os.path.abspath(p)
        if os.path.exists(cwd_path):
            return cwd_path
        pkg_path = os.path.join(_PKG_ROOT, p)
        if os.path.exists(pkg_path):
            return pkg_path
        # Neither exists yet (e.g. an output directory created later) — the
        # working directory is the sane place to put it.
        return cwd_path

    if isinstance(config.get("output_base"), str):
        config["output_base"] = _abs(config["output_base"])

    initial = config.get("input")
    if isinstance(initial, dict):
        for key, value in initial.items():
            if isinstance(value, str):
                initial[key] = _abs(value)

    return config


def get_step_config(config: dict, step_name: str) -> dict:
    """Return the effective config for *step_name* by merging defaults + step overrides.

    Args:
        config:    Merged config dict from :func:`load_config`.
        step_name: Step key as defined in the ``steps:`` section of base.yaml
                   (e.g. ``"generate_tools"``, ``"generate_trajectories"``).

    Returns:
        Dict with all effective values for the step (llm info is NOT included;
        callers should read ``config["llm"]`` separately if needed).
    """
    defaults: dict = config.get("defaults", {})
    step_overrides: dict = config.get("steps", {}).get(step_name, {})
    return _deep_merge(defaults, step_overrides)
