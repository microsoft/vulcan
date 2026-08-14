"""Prompts for the score_sequences step.

Evaluates API dependency subgraphs for quality as training scenarios.
Uses a thinking model (reasoning) for careful judgment.
"""

SCORE_SEQUENCES_SYSTEM = """You are evaluating an API dependency subgraph to determine its quality as a training scenario for tool-calling AI agents.

You receive:
- The API specifications (what each tool does, its parameters and responses)
- A subgraph (nodes = APIs, edges = dependencies between them)

## Evaluation Criteria

### 1. Data Flow Quality (0-3 points)
- Are explicit edges valid? Does the source API actually produce data the target API needs?
- Are the argument mappings correct (source output field → target input parameter)?
- Are implicit edges reasonable? Would the source logically precede the target?
- Deduct points for: incorrect mappings, edges that don't make logical sense, missing obvious dependencies

### 2. Task Coherence (0-3 points)
- Could a real user accomplish a meaningful goal using these APIs together?
- Does the graph represent a realistic workflow (e.g., check availability → book → pay → confirm)?
- Is there a clear start and end to the task?
- Deduct points for: disconnected APIs that don't form a coherent task, artificial/forced combinations

### 3. Complexity & Training Value (0-2 points)
- Is the graph complex enough to be a useful training example? (single edge = low value)
- Does it require multi-step reasoning? (multiple dependent calls, branching logic)
- Is it not SO complex that no realistic user query could trigger it?
- Deduct points for: trivially simple graphs (2 nodes, 1 edge), unrealistically complex chains

### 4. Diversity & Coverage (0-2 points)
- Does the graph exercise different types of tool interactions? (read + write, query + action)
- Does it cover interesting parameter passing patterns?
- Deduct points for: all nodes doing the same type of operation, redundant edges

## Classification

- **GoalOriented**: The graph supports a clear, specific user goal that a person might actually request
  - Example: "Lock the car, check fuel, fill up if needed, and start the engine" → lockDoors → displayCarStatus → fillFuelTank → startEngine
- **Procedural**: The graph represents a valid sequence of operations but lacks a natural user-facing goal
  - Example: gallon_to_liter → displayCarStatus (valid conversion + display, but no user would phrase it as a goal)

## Output

Return a JSON object inside <out_judgment> tags:

<out_judgment>
{
  "score": <0-10>,
  "final_decision": "GoalOriented" or "Procedural",
  "data_flow_score": <0-3>,
  "task_coherence_score": <0-3>,
  "complexity_score": <0-2>,
  "diversity_score": <0-2>,
  "rationale": "2-3 sentence explanation of the score and classification"
}
</out_judgment>"""

SCORE_SEQUENCES_USER = """Evaluate this API dependency subgraph for quality as a training scenario.

<api_specs>
{API_SPECS}
</api_specs>

<graph>
{GRAPH_INFO}
</graph>

Score each criterion, sum for total (0-10), and classify as GoalOriented or Procedural."""
