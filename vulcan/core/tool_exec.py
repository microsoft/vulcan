"""Shared tool execution utilities for trajectory generation steps.

Every trajectory step (static, dynamic, thinking) calls into this one
canonical implementation rather than carrying its own copy.

Public API:
    clean_for_json(obj)               -> serialisation-safe Python object (async)
    execute_tool(app, name, args)     -> JSON string result
    normalize_schema(func)            -> in-place schema fix for OpenAI tools
"""

from __future__ import annotations

import json
from typing import Any

from .exceptions import EnvironmentError


# ── JSON cleaning ────────────────────────────────────────────────────────────


async def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy / complex types to plain Python for JSON.

    Handles: np.bool_, np.integer, np.floating, np.ndarray, complex.
    Safe to call on plain Python objects (no-op for standard types).
    """
    # Import lazily — numpy may not be installed in all environments
    try:
        import numpy as np  # type: ignore
        _np = np
    except ImportError:
        _np = None

    if isinstance(obj, dict):
        return {k: await clean_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        cleaned = [await clean_for_json(v) for v in obj]
        return type(obj)(cleaned)

    if _np is not None:
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    if isinstance(obj, complex):
        return obj.real

    return obj


# ── Tool execution ───────────────────────────────────────────────────────────


async def execute_tool(app: Any, name: str, args: dict) -> str:
    """Execute a named tool on an environment instance.

    Args:
        app:   instantiated environment object with a function_name_mapping()
               method that returns {tool_name: callable}.
        name:  tool name (must be in function_name_mapping).
        args:  keyword arguments to pass to the tool.

    Returns:
        JSON string with the tool result, or a JSON error object.

    Raises:
        EnvironmentError: when the tool is not found in the mapping or raises
                          an unexpected exception.  Callers in trajectory steps
                          may choose to catch and convert to a JSON error message.
    """
    try:
        mapping = app.function_name_mapping()
    except Exception as exc:
        raise EnvironmentError(
            f"function_name_mapping() failed: {exc}",
            tool_name=name,
        ) from exc

    if name not in mapping:
        raise EnvironmentError(
            f"Unknown tool '{name}'; available: {sorted(mapping.keys())}",
            tool_name=name,
        )

    try:
        func = mapping[name]
        raw = func(**args) if args else func()
        result = await clean_for_json(raw)

        if isinstance(result, str):
            return result
        return json.dumps(result)

    except EnvironmentError:
        raise
    except Exception as exc:
        raise EnvironmentError(
            f"Tool '{name}' raised an exception: {exc}",
            tool_name=name,
        ) from exc


# ── Schema normalisation ─────────────────────────────────────────────────────


def normalize_schema(func: dict) -> None:
    """Fix an API function schema in-place for OpenAI function-calling compatibility.

    Transformations applied:
    - Remove top-level "response" and "constraints" keys.
    - Recursively replace type aliases:
        "dict"  → "object"
        "float" → "number"
        "int"   → "integer"
    - Add default "items": {"type": "string"} to array types missing "items".
    """
    func.pop("response", None)
    func.pop("constraints", None)

    def _fix(obj: Any) -> None:
        if isinstance(obj, dict):
            t = obj.get("type")
            if t == "dict":
                obj["type"] = "object"
            elif t == "float":
                obj["type"] = "number"
            elif t == "int":
                obj["type"] = "integer"
            elif t == "array" and "items" not in obj:
                obj["items"] = {"type": "string"}
            for v in obj.values():
                _fix(v)
        elif isinstance(obj, list):
            for v in obj:
                _fix(v)

    _fix(func)
