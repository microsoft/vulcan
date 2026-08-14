"""Environment package for VULCAN.

Public surface
--------------
EnvironmentType      -- enum identifying the environment class (static / dynamic / etc.)
EnvironmentConfig    -- config object loaded from YAML + standardized input JSONL
get_environment()    -- factory function building an EnvironmentConfig by name
"""

from .types import EnvironmentType
from .base import EnvironmentConfig

__all__ = ["EnvironmentType", "EnvironmentConfig", "get_environment"]


def get_environment(name: str, config: dict) -> EnvironmentConfig:
    """Factory function: build an EnvironmentConfig for *name* from a merged config dict.

    Args:
        name:   Environment identifier string (e.g. ``"quickstart"``, ``"hotel_policy"``).
        config: Fully merged config dict, as returned by ``load_config(name)``.

    Returns:
        A ready-to-use :class:`EnvironmentConfig` instance.

    Example::

        from vulcan.config import load_config
        from vulcan.environment import get_environment

        cfg = load_config("my_env")
        env = get_environment("my_env", cfg)
        for cat in env.get_categories():
            apis = env.get_api_catalog(cat)
            ...
    """
    # Ensure the config carries the environment name so EnvironmentConfig can read it.
    if "environment" not in config and "domain" not in config:
        config = {**config, "environment": name}
    return EnvironmentConfig(config)
