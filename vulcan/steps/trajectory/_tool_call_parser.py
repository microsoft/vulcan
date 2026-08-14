"""Tool call parser for text-based (non-native) tool-calling output.

Handles parsing tool calls from three formats:
1. <tool_call>[{name:..., arguments:...}]</tool_call>  (standard XML)
2. <|tool_calls_section_begin|>...<|tool_calls_section_end|>  (sectioned token format
   emitted by some reasoning models)
3. Raw JSON array starting with [{  (some models output bare JSON)
"""

from __future__ import annotations

import ast
import json
import re

from vulcan.core.parsers import TagParser


def parse_tool_calls(content: str) -> list[dict] | None:
    """Parse tool calls from multiple LLM output formats.

    Args:
        content: raw assistant message content string.

    Returns:
        List of dicts with "name" and "arguments" keys, or None if no tool
        calls could be parsed.
    """
    if not content:
        return None

    # Format 1: <tool_call> XML tags
    raw = TagParser.extract_first("tool_call", content)
    if raw:
        raw = raw.strip()
        if raw.startswith("["):
            try:
                calls = json.loads(raw)
                if isinstance(calls, list):
                    return calls
            except json.JSONDecodeError:
                try:
                    calls = ast.literal_eval(raw)
                    if isinstance(calls, list):
                        return calls
                except (ValueError, SyntaxError):
                    pass

    # Format 2: sectioned <|tool_calls_section_begin|> token format
    if "<|tool_calls_section_begin|>" in content:
        calls = []
        # Pattern: <|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>JSON<|tool_call_end|>
        pattern = (
            r"<\|tool_call_begin\|>(?:functions\.)?([^:<|]+)(?::\d+)?"
            r"<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>"
        )
        matches = re.findall(pattern, content, re.DOTALL)
        for name, args_str in matches:
            name = name.strip()
            args_str = args_str.strip()
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": name, "arguments": args})
        if calls:
            return calls

    # Format 1b: one or more <tool_call>{object}</tool_call> tags, each a single
    # JSON object (prompting style). Handled by parse_prompting_tool_calls below;
    # fall through here so a single-object tag is still parsed by the generic path.
    if raw:
        raw_s = raw.strip()
        if raw_s.startswith("{"):
            try:
                obj = json.loads(raw_s)
                if isinstance(obj, dict) and "name" in obj:
                    return [obj]
            except json.JSONDecodeError:
                pass

    # Format 3: JSON array in content (some models just output JSON)
    content_stripped = content.strip()
    if content_stripped.startswith("[{"):
        try:
            calls = json.loads(content_stripped)
            if isinstance(calls, list) and calls and "name" in calls[0]:
                return calls
        except json.JSONDecodeError:
            pass

    return None


def parse_prompting_tool_calls(content: str) -> list[dict] | None:
    """Parse tool calls for the prompting style.

    Supports one or more ``<tool_call>...</tool_call>`` tags in a single message,
    where each tag holds either a single JSON object
    ``{"name": ..., "arguments": ...}`` or a JSON array of such objects.

    Args:
        content: raw assistant message content string.

    Returns:
        List of dicts with "name" and "arguments" keys, or None if no tool
        calls could be parsed.
    """
    if not content:
        return None

    raws = TagParser.extract_all("tool_call", content)
    if not raws:
        return None

    calls: list[dict] = []
    for raw in raws:
        raw = raw.strip()
        if not raw:
            continue
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None
        if parsed is None:
            continue

        candidates = parsed if isinstance(parsed, list) else [parsed]
        for obj in candidates:
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": obj.get("name", ""), "arguments": args})

    return calls or None
