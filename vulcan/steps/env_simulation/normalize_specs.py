"""Step: normalize_specs — Preprocess API specifications before code generation.

Runs BEFORE generate_tools.  Two sub-steps per API item:
1. Check name/description consistency — fix ambiguities in the API spec.
2. Generate response schema — if the spec is missing a 'response' field, add one.

Output fields added to each item:
  - ``normalized_function``: corrected API spec, only when name/description
    disagreed. Diagnostic — it records that a correction happened.
  - ``normalized_schema``:   **always written.** The resolved spec: corrected
    where needed, carrying a ``response`` schema either from the catalog or
    generated here.

``normalized_schema`` is the contract with every later step. Read that field
alone; do not fall back to the raw ``function``, which may lack the ``response``
schema that dependency detection in ``build_graph`` depends on.
"""

from __future__ import annotations

import json
import traceback

from .. import register_step
from ..base import BaseStep
from ...core.agent import Agent
from ...core.parsers import TagParser
from ...core.exceptions import StepError


PREPROCESS_CHECK_USER = """You are given a JSON API definition. Evaluate the `name` and `description` fields for consistency.

**Instructions:**
1. If the `name` and `description` are consistent, return **PASS**.
2. If there are inconsistencies, return **PROBLEM** with:
   - A brief explanation of the issue.
   - A corrected version of the `description` field.
   - The full modified JSON inside `<json_schema>...</json_schema>` tags.
3. If you encounter a JSON formatting error, fix it.

**Rules:**
- Do NOT modify the types or descriptions of input parameters in `properties`.
- Do NOT add new input arguments.
- Only fix alignment between `name` and `description`.
- The modified JSON must be valid (loadable by `json.loads`).

JSON API Definition:
{INP_DICT}"""

SCHEMA_RESPONSE_SYSTEM = """You are given an API specification that is missing a response schema. Generate the complete API spec with a `response` field added based on the API's description and parameters."""

SCHEMA_RESPONSE_USER = """This API specification has no response schema. Add one.

<api_spec>
{JSON_SCHEMA}
</api_spec>

Return the complete updated JSON spec inside <json_schema> tags. The JSON must be valid and loadable.

<json_schema>complete JSON with response field</json_schema>"""


@register_step("normalize_specs")
class NormalizeSpecsStep(BaseStep):
    """Preprocess API specs: fix name/description inconsistencies, add missing response schemas."""

    def __init__(self, config, llm, env):
        super().__init__(config, llm, env)
        self.check_agent = Agent(
            "check_spec", "", llm,
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("json_schema"),
        )
        self.schema_agent = Agent(
            "schema_gen", SCHEMA_RESPONSE_SYSTEM, llm,
            model=self.model, temperature=0, max_tokens=self.max_tokens,
            output_parser=TagParser.make_parser("json_schema"),
        )

    @staticmethod
    def preprocess_input(data: list[dict]) -> list[dict]:
        """Flatten category-level input into individual API items.

        Accepts:
          - ``{"api_list": [...], "category_name": "..."}``  — category-level record
          - ``{"function": {...}, ...}``                     — already-flattened item
        """
        items = []
        for cat_item in data:
            if "api_list" in cat_item:
                for api in cat_item["api_list"]:
                    item = {
                        "category_name": cat_item.get("category_name", ""),
                        "function": api.get("function", api),
                    }
                    if "constraints" in api:
                        item["constraints"] = api["constraints"]
                    items.append(item)
            elif "function" in cat_item:
                items.append(cat_item)
            else:
                items.append(cat_item)
        return items

    async def run(self, inp: dict, progress=None) -> dict:
        try:
            api_function = inp["function"]

            # Sub-step 1: Check name/description consistency
            prompt = PREPROCESS_CHECK_USER.format(INP_DICT=json.dumps(api_function, indent=2))
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

            for attempt in range(5):
                res = await self.check_agent.run(messages, progress)

                if "PASS" in res["content"] and "PROBLEM" not in res["content"]:
                    break

                if "PROBLEM" in res["content"]:
                    modified = self._extract_json_schema(res)
                    if modified:
                        try:
                            if isinstance(modified, str):
                                modified = json.loads(modified)
                            inp["normalized_function"] = modified
                            api_function = modified
                            break
                        except (json.JSONDecodeError, TypeError):
                            # Retry with error feedback
                            messages.extend([
                                {"role": "assistant", "content": res["content"]},
                                {"role": "user", "content": "JSON parse error. Return valid JSON inside <json_schema> tags."},
                            ])
                            continue

                # No clear PASS or PROBLEM — retry
                messages.extend([
                    {"role": "assistant", "content": res["content"]},
                    {"role": "user", "content": prompt},
                ])

            # Sub-step 2: Generate a response schema when the catalog omitted one.
            if not api_function.get("response"):
                schema = await self._generate_response_schema(api_function, progress)
                if schema:
                    api_function = schema

            # `normalized_schema` is ALWAYS written, and is the single spec every
            # downstream step reads. It resolves to, in order: the corrected spec
            # with a generated response, the corrected spec, the spec with a
            # generated response, or the catalog spec unchanged when it was
            # already consistent and already carried a response.
            #
            # Writing it unconditionally is what lets `build_graph` read one field
            # with no fallback. When this was only set on the
            # response-was-missing path, a catalog that already shipped `response`
            # produced no value here, and `build_graph` — which reads a record
            # whose keys are always present — serialised the spec as `null` and
            # found no dependencies at all.
            inp["normalized_schema"] = api_function

        except (StepError, KeyError):
            raise
        except Exception:
            traceback.print_exc()

        return inp

    async def _generate_response_schema(self, api_function: dict, progress) -> dict | None:
        """Generate response schema with a retry loop."""
        prompt = SCHEMA_RESPONSE_USER.format(JSON_SCHEMA=json.dumps(api_function, indent=2))
        messages = [
            {"role": "system", "content": SCHEMA_RESPONSE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        for _ in range(5):
            res = await self.schema_agent.run(messages, progress)
            schema = self._extract_json_schema(res)
            if schema:
                try:
                    if isinstance(schema, str):
                        schema = json.loads(schema)
                    if "response" in schema:
                        return schema
                except (json.JSONDecodeError, TypeError):
                    pass
            messages.extend([
                {"role": "assistant", "content": res["content"]},
                {"role": "user", "content": f"JSON parse error or missing 'response' field. Try again.\n\n{prompt}"},
            ])
        return None

    def _extract_json_schema(self, res: dict) -> dict | str | None:
        """Extract JSON from <json_schema> tags in *res*."""
        parsed = res.get("parsed_content")
        if parsed and isinstance(parsed, list) and parsed[0]:
            try:
                return json.loads(parsed[0]) if isinstance(parsed[0], str) else parsed[0]
            except (json.JSONDecodeError, TypeError):
                pass
        raw = TagParser.extract_first("json_schema", res["content"])
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return None
