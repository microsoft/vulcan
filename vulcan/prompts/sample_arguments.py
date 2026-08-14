"""Prompts for the sample_arguments step (function sample generation).

Generates input/output sample pairs for each tool in the environment,
based on a specific initial state. Each initial state gets its own samples
because different states produce different outputs.
"""

FUNC_SAMPLE_SYSTEM = """Generate realistic input/output test samples for all public functions of an environment class.

You receive:
- The environment's Python code (shows exact implementation logic)
- A specific initial state (the values tools will read from and write to)
- The list of function names to cover

## Requirements

1. **Cover every function** in the api_list — do not skip any
2. **At least 3 cases per function** covering:
   - **Typical case**: Common, valid input with the expected output based on the given initial state
   - **Edge case**: Boundary values, empty strings, zero values, maximum values, or uncommon but valid inputs
   - **Error case**: Invalid input that triggers the error handling path (wrong type, out-of-range enum, missing required data in state)

3. **Outputs must match the actual code behavior** for the given initial state:
   - Read the code carefully — trace through the logic with the given state values
   - For example, if initial fuel is 8.5 gallons and capacity is 50, then fillFuelTank(fuelAmount=41.5) should produce fuelLevel=50.0
   - Do NOT guess outputs — derive them from the code + state

4. **Input values must be realistic**:
   - Use values that make sense for the domain (real temperatures, valid door names, actual zip codes from the state)
   - For enum parameters, test both valid and invalid enum values
   - For numeric parameters, test within range and outside range

5. **State awareness**: The initial state affects outputs. If the state has engine="stopped", then startEngine should show the transition. If fuel_level is specific, calculations should use that exact value.

## Output Format

Return a JSON array inside <function_samples> tags. Each entry has the function name and its test cases:

<function_samples>
[
  {
    "function_name": "activateParkingBrake",
    "cases": [
      {
        "input": {"mode": "engage"},
        "expected_output": {"parkingBrakeStatus": "engaged", "_parkingBrakeForce": 3500.0, "_slopeAngle": 7.5}
      },
      {
        "input": {"mode": "release"},
        "expected_output": {"parkingBrakeStatus": "released", "_parkingBrakeForce": 0.0, "_slopeAngle": 0.0}
      },
      {
        "input": {"mode": "invalid_value"},
        "expected_output": {"error": "Invalid mode. Must be 'engage' or 'release'."}
      }
    ]
  },
  {
    "function_name": "fillFuelTank",
    "cases": [
      {
        "input": {"fuelAmount": 20.0},
        "expected_output": {"fuelLevel": 28.5}
      }
    ]
  }
]
</function_samples>

IMPORTANT: Do NOT wrap the output in markdown code fences. Return only the JSON array inside the tags."""

FUNC_SAMPLE_USER = """Generate function test samples for this environment based on the specific initial state provided.

<code>
{PY_CODE}
</code>

<initial_state>
{INITIAL_STATE}
</initial_state>

<api_list>
{API_LIST}
</api_list>

Trace through the code with the given initial state to produce accurate expected outputs. Cover every function in the api_list."""
