"""`score_sequences` must read each record type the way it is actually written.

`plan_sequences` emits two shapes. A sequence record carries `number_turns` and
`sequence`; a multipath record carries `nodes`, `edges`, `multipath_pairs` and
no `number_turns` at all. The step branched on `number_turns`, defaulting to 1,
so every multipath record took the single-turn path and read `graph_info` — a
top-level key neither shape carries. Nodes came out empty and the model was paid
to rate `{"nodes": [], "edges": []}`.
"""

import asyncio
import json

import pytest

from vulcan.steps.graph_construction.score_sequences import ScoreSequencesStep


class _Env:
    def get_api_catalog(self, category):
        return []


def _step(capture):
    step = object.__new__(ScoreSequencesStep)
    step.name = "score_sequences"
    step.env = _Env()

    class _Agent:
        async def run(self, messages, progress=None):
            capture.append(messages)
            return {"content": "<judgment>{}</judgment>", "parsed_content": ["{}"]}

    step.eval_agent = _Agent()
    step._parse_judgment = lambda res: {}
    return step


def _catalog(*names):
    return [{"function": {"name": n},
             "normalized_schema": {"name": n, "response": {"ok": "boolean"}}} for n in names]


def _prompt(messages):
    return messages[-1]["content"]


EMPTY_GRAPH = json.dumps({"nodes": [], "edges": []}, indent=2)


class TestMultipathRecords:
    def test_a_multipath_record_is_scored_on_its_real_graph(self):
        seen = []
        item = {
            "category_name": "c", "type": "multipath", "id": "m1",
            "nodes": ["search", "book"],
            "edges": [{"source": "search", "target": "book", "type": "explicit"}],
            "multipath_pairs": [["search", "book"]],
            "api_list": _catalog("search", "book", "unrelated"),
        }
        asyncio.run(_step(seen).run(item))

        prompt = _prompt(seen[0])
        assert EMPTY_GRAPH not in prompt, "multipath record scored against an empty graph"
        assert '"search"' in prompt and '"book"' in prompt
        assert '"type": "explicit"' in prompt, "the graph's edges never reached the judge"

    def test_the_api_specs_are_filtered_to_the_graph_nodes(self):
        seen = []
        item = {
            "category_name": "c", "type": "multipath", "nodes": ["search"],
            "edges": [], "api_list": _catalog("search", "book"),
        }
        asyncio.run(_step(seen).run(item))
        prompt = seen[0][-1]["content"]
        assert '"search"' in prompt
        assert prompt.count('"book"') == 0, "unrelated tool spec sent to the judge"

    def test_a_multipath_record_has_no_number_turns(self):
        """The premise of the bug — guards against a fixture that hides it."""
        item = {"category_name": "c", "type": "multipath", "nodes": ["a"], "edges": []}
        assert "number_turns" not in item


class TestOtherRecordShapesStillWork:
    def test_a_multi_turn_sequence_record_uses_its_sequence(self):
        seen = []
        item = {
            "category_name": "c", "number_turns": 2,
            "sequence": [{"graph_info": {"nodes": ["search"], "edges": []}},
                         {"graph_info": {"nodes": ["book"], "edges": []}}],
            "api_list": _catalog("search", "book"),
        }
        asyncio.run(_step(seen).run(item))
        prompt = seen[0][-1]["content"]
        assert '"search"' in prompt and '"book"' in prompt

    def test_a_single_turn_record_still_reads_graph_info(self):
        seen = []
        item = {
            "category_name": "c", "number_turns": 1,
            "graph_info": [{"nodes": ["search"], "edges": []}],
            "api_list": _catalog("search", "book"),
        }
        asyncio.run(_step(seen).run(item))
        prompt = _prompt(seen[0])
        assert EMPTY_GRAPH not in prompt
        assert '"search"' in prompt
