"""Output parsers for extracting structured data from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


class TagParser:
    """Extract content from XML-style tags in LLM output."""

    @staticmethod
    def extract_all(tag: str, text: str) -> list[str]:
        """Return all text blocks enclosed in <tag>…</tag>."""
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        return re.findall(pattern, text, re.DOTALL)

    @staticmethod
    def extract_first(tag: str, text: str) -> str | None:
        """Return the first text block enclosed in <tag>…</tag>, or None."""
        matches = TagParser.extract_all(tag, text)
        return matches[0] if matches else None

    @staticmethod
    def extract_pairs(tag1: str, tag2: str, text: str) -> list[dict]:
        """Extract adjacent tag pairs into a list of {tag1: …, tag2: …} dicts."""
        pattern = rf"<{tag1}>\s*(.*?)\s*</{tag1}>\s*<{tag2}>\s*(.*?)\s*</{tag2}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{tag1: m[0].strip(), tag2: m[1].strip()} for m in matches]

    @staticmethod
    def extract_json(tag: str, text: str) -> Any | None:
        """Extract first <tag>…</tag> block and parse it as JSON.

        Returns the parsed object, the raw string (if JSON parse fails), or None.
        """
        raw = TagParser.extract_first(tag, text)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    @staticmethod
    def make_parser(tag: str) -> Callable:
        """Create an async parser function that returns all matches for *tag*."""
        async def parser(inputs: dict) -> list[str]:
            text = inputs.get("content", "")
            return TagParser.extract_all(tag, text)
        return parser

    @staticmethod
    def make_pair_parser(tag1: str, tag2: str) -> Callable:
        """Create an async pair-parser for *tag1* / *tag2* adjacent blocks."""
        async def parser(inputs: dict) -> list[dict]:
            text = inputs.get("content", "")
            return TagParser.extract_pairs(tag1, tag2, text)
        return parser

    @staticmethod
    def make_json_parser(tag: str) -> Callable:
        """Create an async parser that extracts *tag* and parses it as JSON."""
        async def parser(inputs: dict) -> Any:
            text = inputs.get("content", "")
            return TagParser.extract_json(tag, text)
        return parser
