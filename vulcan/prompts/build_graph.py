"""Prompts for the build_graph step (API dependency graph generation).

For each ordered pair (A, B) of APIs, determines the dependency:
  - Explicit: output fields of A feed into input params of B (with field mappings)
  - Implicit: logical/temporal ordering (e.g., authenticate before booking)
  - None: no meaningful dependency
"""

BUILD_GRAPH_SYSTEM = """You are given exactly two API specifications in JSON format.

Each specification follows this schema:
```json
{
  "name": "api_name",
  "description": "What the API does",
  "parameters": {
    "type": "dict",
    "properties": { "param1": {"type": "string", "description": "..."}, ... },
    "required": ["param1"]
  },
  "response": {
    "type": "dict",
    "properties": { "field1": {"type": "string", "description": "..."}, ... }
  }
}
```

Your task: identify ONE directed edge from the first API to the second API, if a meaningful relation exists.

## Edge Types

**Explicit** — At least one field in the first API's `response` can feed into a parameter of the second API's `parameters`. The relation is data-driven: the first API produces something the second API consumes.

Examples:
  - `get_user_id(username) → send_message(receiver_id)`: get_user_id outputs user_id which maps to receiver_id
  - `search_flights(origin, dest) → book_flight(flight_id)`: search outputs flight_id that book requires
  - `estimate_distance(cityA, cityB) → estimate_drive_feasibility(distance)`: distance output feeds into distance input

**Implicit** — A logical or practical ordering even though no data is passed directly. One API should typically be called before the other in a real workflow.

Examples:
  - `login() → create_ticket(title)`: must authenticate before creating tickets
  - `book_flight(flight_id) → book_hotel(location)`: logical sequence in trip planning
  - `lock_doors(mode) → start_engine(mode)`: safety protocol requires locking before starting

**None** — No meaningful dependency in either direction.

## Decision Process

1. Parse both API specs — extract all output fields (response.properties) and input fields (parameters.properties)
2. Check for explicit relation: does any output field of API_1 match (by name, type, or semantics) an input field of API_2?
   - Use semantic matching, not just exact name matching (e.g., "user_id" matches "receiver_id")
   - If found → output explicit edge with field mappings
3. If no explicit match: does the description of API_1 suggest it should logically precede API_2 in a real workflow?
   - If yes → output implicit edge
4. If neither → output empty

## Output Format

Return one of the following inside <graph> tags:

**Explicit relation:**
<graph>
{
  "source": "first_api_name",
  "target": "second_api_name",
  "type": "explicit",
  "source_arguments": ["output_field_1"],
  "target_arguments": ["input_param_1"]
}
</graph>

**Implicit relation:**
<graph>
{
  "source": "first_api_name",
  "target": "second_api_name",
  "type": "implicit"
}
</graph>

**No relation:**
<graph>
{
  "decision": "empty"
}
</graph>

IMPORTANT:
- Do NOT output anything between the <graph> tags except the JSON
- source_arguments must be actual field names from the first API's response
- target_arguments must be actual field names from the second API's parameters
- Use lowercase "explicit" or "implicit" for the type field"""

BUILD_GRAPH_USER = """Analyze the dependency between these two APIs. First check for an explicit relation (output→input data flow). If none, check for an implicit relation (logical ordering). If neither, return empty.

First API:
{FIRST_API}

Second API:
{SECOND_API}"""
