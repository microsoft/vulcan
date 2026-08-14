"""Tests for core/parsers.py TagParser."""

import asyncio
import pytest

from vulcan.core.parsers import TagParser


def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ── extract_first ─────────────────────────────────────────────────────────────

class TestExtractFirst:
    def test_simple_tag(self):
        text = "<answer>42</answer>"
        assert TagParser.extract_first("answer", text) == "42"

    def test_first_of_multiple_tags(self):
        text = "<answer>first</answer>  <answer>second</answer>"
        assert TagParser.extract_first("answer", text) == "first"

    def test_tag_not_found_returns_none(self):
        assert TagParser.extract_first("answer", "no tags here") is None

    def test_tag_with_whitespace_stripped(self):
        text = "<answer>  hello  </answer>"
        assert TagParser.extract_first("answer", text) == "hello"

    def test_nested_content_preserved(self):
        text = "<code>def foo():\n    return 1</code>"
        result = TagParser.extract_first("code", text)
        assert "def foo():" in result
        assert "return 1" in result

    def test_multiline_content(self):
        text = "<output>\nline one\nline two\n</output>"
        result = TagParser.extract_first("output", text)
        assert "line one" in result
        assert "line two" in result

    def test_empty_content(self):
        text = "<answer></answer>"
        result = TagParser.extract_first("answer", text)
        assert result == ""


# ── extract_all ───────────────────────────────────────────────────────────────

class TestExtractAll:
    def test_returns_list(self):
        text = "<item>a</item>"
        result = TagParser.extract_all("item", text)
        assert isinstance(result, list)

    def test_single_match(self):
        text = "<item>only one</item>"
        result = TagParser.extract_all("item", text)
        assert result == ["only one"]

    def test_multiple_matches(self):
        text = "<item>a</item> <item>b</item> <item>c</item>"
        result = TagParser.extract_all("item", text)
        assert result == ["a", "b", "c"]

    def test_no_match_returns_empty_list(self):
        result = TagParser.extract_all("item", "no items here")
        assert result == []

    def test_whitespace_stripped_in_each_match(self):
        text = "<item>  x  </item><item>  y  </item>"
        result = TagParser.extract_all("item", text)
        assert result == ["x", "y"]

    def test_mixed_content(self):
        text = "prefix <item>one</item> middle <item>two</item> suffix"
        result = TagParser.extract_all("item", text)
        assert result == ["one", "two"]


# ── extract_pairs ─────────────────────────────────────────────────────────────

class TestExtractPairs:
    def test_single_pair(self):
        text = "<name>foo</name><value>bar</value>"
        result = TagParser.extract_pairs("name", "value", text)
        assert result == [{"name": "foo", "value": "bar"}]

    def test_multiple_pairs(self):
        text = "<k>a</k><v>1</v>  <k>b</k><v>2</v>"
        result = TagParser.extract_pairs("k", "v", text)
        assert len(result) == 2
        assert result[0] == {"k": "a", "v": "1"}
        assert result[1] == {"k": "b", "v": "2"}

    def test_no_pairs_returns_empty(self):
        result = TagParser.extract_pairs("k", "v", "nothing here")
        assert result == []


# ── extract_json ──────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_valid_json_parsed(self):
        text = '<data>{"key": "value", "num": 42}</data>'
        result = TagParser.extract_json("data", text)
        assert result == {"key": "value", "num": 42}

    def test_invalid_json_returns_raw_string(self):
        text = "<data>{broken json}</data>"
        result = TagParser.extract_json("data", text)
        assert isinstance(result, str)
        assert "broken" in result

    def test_missing_tag_returns_none(self):
        result = TagParser.extract_json("data", "no tag here")
        assert result is None

    def test_json_list(self):
        text = "<items>[1, 2, 3]</items>"
        result = TagParser.extract_json("items", text)
        assert result == [1, 2, 3]


# ── make_parser ────────────────────────────────────────────────────────────────

class TestMakeParser:
    def test_returns_callable(self):
        parser = TagParser.make_parser("answer")
        assert callable(parser)

    def test_parser_is_async(self):
        import inspect
        parser = TagParser.make_parser("answer")
        assert inspect.iscoroutinefunction(parser)

    def test_parser_extracts_all_matches(self):
        parser = TagParser.make_parser("answer")
        text = "<answer>one</answer><answer>two</answer>"
        result = run(parser({"content": text}))
        assert result == ["one", "two"]

    def test_parser_uses_content_key(self):
        parser = TagParser.make_parser("val")
        result = run(parser({"content": "<val>42</val>"}))
        assert result == ["42"]

    def test_parser_missing_content_returns_empty(self):
        parser = TagParser.make_parser("val")
        result = run(parser({}))
        assert result == []

    def test_parser_no_matches_returns_empty_list(self):
        parser = TagParser.make_parser("answer")
        result = run(parser({"content": "no answers here"}))
        assert result == []


# ── make_pair_parser ──────────────────────────────────────────────────────────

class TestMakePairParser:
    def test_returns_callable(self):
        parser = TagParser.make_pair_parser("k", "v")
        assert callable(parser)

    def test_parser_extracts_pairs(self):
        parser = TagParser.make_pair_parser("k", "v")
        text = "<k>key1</k><v>val1</v>"
        result = run(parser({"content": text}))
        assert result == [{"k": "key1", "v": "val1"}]


# ── make_json_parser ──────────────────────────────────────────────────────────

class TestMakeJsonParser:
    def test_returns_callable(self):
        parser = TagParser.make_json_parser("data")
        assert callable(parser)

    def test_parses_json_from_content(self):
        parser = TagParser.make_json_parser("data")
        text = '<data>{"score": 9}</data>'
        result = run(parser({"content": text}))
        assert result == {"score": 9}

    def test_missing_tag_returns_none(self):
        parser = TagParser.make_json_parser("data")
        result = run(parser({"content": "nothing here"}))
        assert result is None
