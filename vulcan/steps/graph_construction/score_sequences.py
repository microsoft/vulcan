"""score_sequences step: Evaluate graph quality using a thinking model."""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError
from ...prompts.score_sequences import SCORE_SEQUENCES_SYSTEM, SCORE_SEQUENCES_USER


@register_step("score_sequences")
class ScoreSequencesStep(BaseStep):
    """Evaluate API dependency subgraphs for training quality.

    Uses a thinking model when configured to reason carefully about
    data-flow quality, task coherence, complexity, and diversity.
    """

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.eval_agent = Agent(
            "score_sequences", "", llm,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_model=self.thinking_model,
            output_parser=TagParser.make_parser("out_judgment"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            cat_name = inp["category_name"]

            # Use preprocessed specs from input if available, else fall back to domain catalog
            api_specs = inp.get("api_list") or self.env.get_api_catalog(cat_name)

            # Build filtered API specs for the nodes in this graph
            if inp.get("type") == "multipath":
                # A multipath record carries `nodes` / `edges` at the top level and
                # has no `number_turns`. Without this branch it defaulted to 1 and
                # read `graph_info`, a key no record type carries, so the model was
                # asked to rate {"nodes": [], "edges": []}.
                nodes = inp.get("nodes", [])
                api_specs_filtered = [
                    self._get_best_spec(a) for a in api_specs
                    if self._get_best_spec(a).get("name") in nodes
                ]
                graph_str = json.dumps(
                    {"nodes": nodes, "edges": inp.get("edges", [])}, indent=2
                )
            elif inp.get("number_turns", 1) == 1:
                graph_info = inp.get("graph_info", [{}])[0]
                nodes = graph_info.get("nodes", [])
                api_specs_filtered = [
                    self._get_best_spec(a) for a in api_specs
                    if self._get_best_spec(a).get("name") in nodes
                ]
                graph_str = json.dumps(
                    {"nodes": graph_info.get("nodes", []), "edges": graph_info.get("edges", [])},
                    indent=2,
                )
            else:
                sequence = inp.get("sequence", [])
                all_nodes: set[str] = set()
                for s in sequence:
                    gi = s.get("graph_info", {})
                    all_nodes.update(gi.get("nodes", []))
                api_specs_filtered = [
                    self._get_best_spec(a) for a in api_specs
                    if self._get_best_spec(a).get("name") in all_nodes
                ]
                graph_str = json.dumps(sequence, indent=2)

            prompt = SCORE_SEQUENCES_USER.format(
                API_SPECS=json.dumps(api_specs_filtered, indent=2),
                GRAPH_INFO=graph_str,
            )
            messages = [
                {"role": "system", "content": SCORE_SEQUENCES_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            res = await self.eval_agent.run(messages, progress)
            inp["judgment"] = self._parse_judgment(res)

        except StepError:
            raise
        except Exception:
            traceback.print_exc()

        return inp

    @staticmethod
    def _get_best_spec(api: dict) -> dict:
        """Return the API spec: normalized_schema, else normalized_function, else function.

        Selection is on **value, not presence**. These records originate from
        ``env_grouped``, which writes every per-API key unconditionally via
        ``.get()``, so a key being present says nothing about it holding a spec.
        A presence test here returned ``None``, and the caller immediately calls
        ``.get("name")`` on the result — an ``AttributeError`` that the broad
        handler in ``run`` would swallow into a failed item.

        Unlike ``build_graph`` this step does not raise on a missing spec: its
        output is analysis only and nothing downstream reads it, so degrading is
        preferable to failing a run over it.
        """
        return (
            api.get("normalized_schema")
            or api.get("normalized_function")
            or api.get("function", {})
        )

    def _parse_judgment(self, res: dict):
        """Parse judgment JSON from agent response."""
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed:
            try:
                return json.loads(parsed[0]) if isinstance(parsed[0], str) else parsed[0]
            except (json.JSONDecodeError, TypeError):
                pass
        raw = TagParser.extract_first("out_judgment", res["content"])
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return None
