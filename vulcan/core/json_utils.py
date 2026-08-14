"""Robust JSON parsing for LLM outputs.

4-strategy approach:
1. Direct json.loads
2. Preprocess (strip markdown, fix Python booleans, trailing commas) + json.loads
3. json_repair library (if installed)
4. Bracket extraction + json_repair
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

try:
    from json_repair import repair_json  # type: ignore
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


def preprocess_json_string(s: str) -> str:
    """Clean common LLM artefacts before JSON parsing."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    # Remove single-line // comments
    s = re.sub(r'(?<!:)//.*', '', s)
    # Fix Python-style booleans / None
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def try_parse_json(raw: str) -> Tuple[Any, Optional[str]]:
    """Multi-strategy JSON parser.

    Returns:
        (parsed_obj, None)           on success
        (None, error_message)        on failure
    Works for both objects and arrays.
    """
    if not raw or not raw.strip():
        return None, "Empty input"

    # Strategy 1: direct parse
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass

    # Strategy 2: preprocess + parse
    cleaned = preprocess_json_string(raw)
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    # Strategy 3: json_repair library
    if HAS_JSON_REPAIR:
        try:
            obj = repair_json(cleaned, return_objects=True)
            if obj is not None:
                return obj, None
        except Exception:
            pass

    # Strategy 4: extract outermost brackets + repair
    for open_br, close_br in [('[', ']'), ('{', '}')]:
        start = cleaned.find(open_br)
        end = cleaned.rfind(close_br)
        if start != -1 and end != -1 and end > start:
            substring = cleaned[start:end + 1]
            try:
                return json.loads(substring), None
            except json.JSONDecodeError:
                pass
            if HAS_JSON_REPAIR:
                try:
                    obj = repair_json(substring, return_objects=True)
                    if obj is not None:
                        return obj, None
                except Exception:
                    pass

    return None, f"All 4 strategies failed to parse JSON from: {raw[:200]}"


def try_parse_json_list(raw: str) -> Tuple[Optional[list], Optional[str]]:
    """Parse and validate as a list of dicts. Unwraps singly-nested lists."""
    obj, err = try_parse_json(raw)
    if err:
        return None, err

    # Unwrap nested lists
    while isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], list):
        obj = obj[0]

    if not isinstance(obj, list) or len(obj) == 0:
        return None, "Not a non-empty list"

    if all(isinstance(item, dict) for item in obj):
        return obj, None

    return None, "List contains non-dict items"


def try_parse_json_dict(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    """Parse and validate as a dict."""
    obj, err = try_parse_json(raw)
    if err:
        return None, err

    if isinstance(obj, dict):
        return obj, None

    return None, f"Not a dict, got {type(obj).__name__}"


def safe_json_loads(raw: str) -> Any:
    """Best-effort JSON parse. Returns the parsed object or the raw string if all fail."""
    obj, err = try_parse_json(raw)
    return obj if err is None else raw
