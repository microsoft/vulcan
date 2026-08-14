"""JSONL I/O and filesystem helpers."""

from __future__ import annotations

import json
import os
from typing import Any


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def append_jsonl(data: list[dict], path: str) -> None:
    """Append dicts to an existing (or new) JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not exist. Return the path."""
    os.makedirs(path, exist_ok=True)
    return path
