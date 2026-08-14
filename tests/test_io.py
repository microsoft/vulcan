"""Tests for core/io.py JSONL helpers."""

import json
import os
import tempfile
import pytest

from vulcan.core.io import load_jsonl, save_jsonl, append_jsonl, ensure_dir


# ── load_jsonl ────────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def test_loads_single_line(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"key": "value"}\n')
        result = load_jsonl(str(p))
        assert result == [{"key": "value"}]

    def test_loads_multiple_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
        result = load_jsonl(str(p))
        assert len(result) == 3
        assert result[0] == {"a": 1}
        assert result[2] == {"c": 3}

    def test_handles_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = load_jsonl(str(p))
        assert result == []

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n\n')
        result = load_jsonl(str(p))
        assert len(result) == 2

    def test_returns_list_of_dicts(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"x": 1}\n')
        result = load_jsonl(str(p))
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_preserves_nested_structure(self, tmp_path):
        p = tmp_path / "data.jsonl"
        obj = {"nested": {"inner": [1, 2, 3]}, "flag": True}
        p.write_text(json.dumps(obj) + "\n")
        result = load_jsonl(str(p))
        assert result[0] == obj


# ── save_jsonl ────────────────────────────────────────────────────────────────

class TestSaveJsonl:
    def test_writes_single_item(self, tmp_path):
        p = tmp_path / "out.jsonl"
        save_jsonl([{"a": 1}], str(p))
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"a": 1}

    def test_writes_multiple_items(self, tmp_path):
        p = tmp_path / "out.jsonl"
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        save_jsonl(data, str(p))
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[1]) == {"b": 2}

    def test_overwrites_existing_file(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text('{"old": "data"}\n')
        save_jsonl([{"new": "data"}], str(p))
        result = load_jsonl(str(p))
        assert result == [{"new": "data"}]

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "out.jsonl"
        save_jsonl([{"a": 1}], str(p))
        assert p.exists()

    def test_writes_empty_list_creates_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        save_jsonl([], str(p))
        assert p.exists()
        assert p.read_text() == ""

    def test_round_trip_with_load(self, tmp_path):
        p = tmp_path / "rt.jsonl"
        data = [{"id": i, "val": f"item_{i}"} for i in range(5)]
        save_jsonl(data, str(p))
        loaded = load_jsonl(str(p))
        assert loaded == data


# ── append_jsonl ──────────────────────────────────────────────────────────────

class TestAppendJsonl:
    def test_appends_to_existing_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n')
        append_jsonl([{"b": 2}], str(p))
        result = load_jsonl(str(p))
        assert result == [{"a": 1}, {"b": 2}]

    def test_creates_new_file_if_not_exists(self, tmp_path):
        p = tmp_path / "new.jsonl"
        append_jsonl([{"x": 99}], str(p))
        result = load_jsonl(str(p))
        assert result == [{"x": 99}]

    def test_appends_multiple_items(self, tmp_path):
        p = tmp_path / "data.jsonl"
        append_jsonl([{"a": 1}, {"b": 2}], str(p))
        append_jsonl([{"c": 3}, {"d": 4}], str(p))
        result = load_jsonl(str(p))
        assert len(result) == 4
        assert result[2] == {"c": 3}

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "app.jsonl"
        append_jsonl([{"z": 0}], str(p))
        assert p.exists()

    def test_appending_empty_list_no_change(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a": 1}\n')
        append_jsonl([], str(p))
        result = load_jsonl(str(p))
        assert result == [{"a": 1}]


# ── ensure_dir ────────────────────────────────────────────────────────────────

class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_directory")
        result = ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_returns_the_path(self, tmp_path):
        new_dir = str(tmp_path / "mydir")
        result = ensure_dir(new_dir)
        assert result == new_dir

    def test_no_error_if_already_exists(self, tmp_path):
        existing = str(tmp_path)
        result = ensure_dir(existing)  # should not raise
        assert os.path.isdir(existing)

    def test_creates_nested_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        ensure_dir(nested)
        assert os.path.isdir(nested)
