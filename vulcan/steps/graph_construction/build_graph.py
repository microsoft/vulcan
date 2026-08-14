"""build_graph step: Generate API dependency graphs.

For each category, generates a directed dependency graph over all API pairs.
Each edge represents either:
  - explicit: output of source feeds into input of target
  - implicit: source should logically precede target in a workflow
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError
from ...prompts.build_graph import BUILD_GRAPH_SYSTEM, BUILD_GRAPH_USER


@register_step("build_graph")
class BuildGraphStep(BaseStep):
    """Generate dependency graph edges between all pairs of APIs in a category."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.graph_agent = Agent(
            "build_graph", "", llm,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.step_config.get("max_tokens", 2048),
            output_parser=TagParser.make_parser("graph"),
        )

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            api_list = inp["api_list"]

            graph = []
            for first_api in api_list:
                all_edges = []
                for second_api in api_list:
                    if first_api["function"]["name"] == second_api["function"]["name"]:
                        continue

                    first_str = json.dumps(self._get_api_spec(first_api, inp), indent=2)
                    second_str = json.dumps(self._get_api_spec(second_api, inp), indent=2)

                    prompt = BUILD_GRAPH_USER.format(
                        FIRST_API=first_str,
                        SECOND_API=second_str,
                    )
                    messages = [
                        {"role": "system", "content": BUILD_GRAPH_SYSTEM},
                        {"role": "user", "content": prompt},
                    ]
                    res = await self.graph_agent.run(messages, progress)

                    edges = None
                    try:
                        edges = json.loads(res["parsed_content"][0])
                    except (json.JSONDecodeError, TypeError, IndexError):
                        raw = TagParser.extract_first("graph", res["content"])
                        if raw:
                            try:
                                edges = json.loads(raw)
                            except json.JSONDecodeError:
                                edges = None

                    all_edges.append({
                        "target": second_api["function"]["name"],
                        "edges": edges,
                    })

                graph.append({
                    "source": first_api["function"]["name"],
                    "all_edges": all_edges,
                })

            inp["graph"] = graph

        except StepError:
            raise
        except Exception:
            traceback.print_exc()

        return inp

    def _get_api_spec(self, api: dict, inp: dict) -> dict:
        """Return the normalized spec for *api*.

        Deliberately reads one field and does not fall back to the raw catalog
        ``function``. Dependency detection needs the ``response`` schema to find
        an `explicit` edge at all, and only ``normalized_schema`` is guaranteed
        to carry one — ``normalize_specs`` writes it for every API. Falling back
        to the catalog spec would silently degrade the graph to `implicit` edges
        for any tool whose catalog entry omitted ``response``.

        Raises:
            StepError: ``normalized_schema`` is missing or null, which means the
                record did not come through ``normalize_specs``. Failing here is
                deliberate: the alternative is serialising ``null`` into the
                prompt and reporting a successful run with an empty graph.
        """
        spec = api.get("normalized_schema")
        if not spec:
            name = (api.get("function") or {}).get("name", "<unknown>")
            raise StepError(
                self.name,
                inp.get("category_name", "<unknown>"),
                f"API {name!r} has no normalized_schema — rerun normalize_specs "
                f"for this category; build_graph does not read the raw catalog spec",
            )
        return spec
