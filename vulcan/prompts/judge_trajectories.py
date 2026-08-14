"""Prompts for trajectory verification (judge_trajectories step).

Evaluates agent trajectories for correctness, tool usage, and task completion.
The judge model is whatever the `judge_trajectories` step is configured with; these
prompts assume a non-reasoning model unless the variant name says otherwise.
Tool specs are NOT passed redundantly — they're in the trajectory's system prompt.

Variants, by environment type and whether the agent emits reasoning:
- STATIC_MT_JUDGE:           tools_only,           multi-turn, tool calls
- STATIC_MT_THINKING_JUDGE:  tools_only,           multi-turn, thinking model
- DYNAMIC_JUDGE:             with_user_and_policy, tool calls
- DYNAMIC_THINKING_JUDGE:    with_user_and_policy, thinking model
- DYNAMIC_NO_POLICY_JUDGE:   with_user,            tool calls
- SINGLE_TURN_JUDGE:         single-turn interactions
"""

# ── Error Classification (shared across all variants) ────────────────
#
# Every error in a trajectory must be classified into one of these categories:
#   wrong_tool          — Called a tool that wasn't needed for the task
#   wrong_arguments     — Correct tool but incorrect argument values
#   missing_tool_call   — Should have called a tool but gave text instead
#   unnecessary_tool_call — Called extra tools beyond what was needed
#   hallucinated_output — Agent claimed a result that doesn't match the actual tool response
#   incomplete_execution — Started correctly but didn't finish all required steps
#   wrong_order         — Tools called in wrong dependency order
#   ignored_tool_result — Tool returned data but agent didn't use it in response
#   reasoning_error     — (thinking models only) Flawed reasoning led to wrong action
#   policy_violation    — (dynamic only) Agent broke a domain rule
#   auth_failure        — (dynamic only) Didn't verify user identity before acting
#   wrong_denial        — (dynamic only) Denied a valid request
#   no_error            — No errors found
#

# ── Scoring Rubric (shared) ──────────────────────────────────────────
#
#   9-10: Perfect — all tools correct, complete execution, accurate response
#    7-8: Strong — mostly correct, minor issues (e.g., one extra tool call)
#    5-6: Partial — some correct steps but notable errors or missing calls
#    3-4: Weak — significant errors, incomplete execution, wrong tools
#    1-2: Failed — completely wrong approach or no meaningful progress
#

# ── tools_only, multi-turn, tool calls ─────────────────────────────

STATIC_MT_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn conversation.

The full list of available tools is provided in <available_tools> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END of the conversation. Focus ONLY on the agent's behavior — did it use the right tools, handle responses correctly, and achieve what the user asked?

IMPORTANT: Do NOT compare tool outputs against the initial_state to find inconsistencies. The environment may have its own logic (e.g., requiring login before accessing data). If a tool returns empty or error results, that is the environment's behavior — not the agent's fault. Judge the agent based on how it REACTED to the tool outputs it received, not whether the outputs match the initial_state.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Evaluate the final outcome based on the tool results the agent actually received.
2. **Agent's response to tool results**: Did the agent correctly interpret and act on the tool outputs? If a tool returned an error, did the agent handle it appropriately (retry, try alternative, inform user)?
3. **Self-correction**: If the agent made a mistake but recognized and corrected it, this is POSITIVE behavior — not an error.
4. **Final response accuracy**: Does the agent's final text response accurately reflect the actual tool results it received? No fabricated data.
5. **Efficiency**: Did the agent achieve the goal with a reasonable number of tool calls?

## Error Classification

Only classify UNCORRECTED errors — if the agent recovered from a mistake, do NOT list it as an error:
- **wrong_tool**: Called a wrong tool AND never corrected by calling the right one
- **wrong_arguments**: Used incorrect arguments AND the task was not completed because of it
- **missing_tool_call**: Failed to call a required tool, leaving part of the task undone
- **hallucinated_output**: Agent's final response contains fabricated values not from tool results
- **incomplete_execution**: Part of the user's request was never addressed
- **ignored_tool_result**: Tool returned data the user asked about but agent didn't include it in the response
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but successfully corrected it (this is informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal fully achieved, efficient execution, accurate response
- 7-8: Goal achieved with minor inefficiencies (extra calls, slightly verbose) or self-corrections
- 5-6: Goal partially achieved — some parts done correctly, others missing or wrong
- 3-4: Goal mostly not achieved — significant parts of the request unaddressed
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: The user's goal was achieved by the end of the conversation (even if the path had mistakes that were corrected)
- false: The user's goal was NOT achieved — the task remains undone or the result is wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on whether the goal was achieved and why",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

STATIC_MT_JUDGE_USER = """Evaluate this agent trajectory. Judge the agent's behavior based on the tool results it received — did it use the right tools, handle responses correctly, and achieve the user's goal?

<available_tools>
{TOOL_SPECS}
</available_tools>

<conversation>
{CONV}
</conversation>"""

# ── tools_only, multi-turn, thinking model ─────────────────────────

STATIC_MT_THINKING_JUDGE_SYSTEM = """Evaluate an agent trajectory from a reasoning model. The agent may show its reasoning through explanation text before making tool calls, or may reason implicitly through its action choices.

The conversation's system prompt (the first message, role=system) contains the full list of available tools inside <tool> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Tool Call Format

The agent uses text-based tool calls in this format:
<tool_call>
[{"name": "tool_name", "arguments": {...}}]
</tool_call>

Tool responses appear as role=tool messages with results. Evaluate the agent's tool usage regardless of the specific formatting.

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END. Focus ONLY on the agent's behavior — not on environment inconsistencies.

IMPORTANT: Do NOT compare tool outputs against the initial_state. The environment may require specific conditions (login, permissions) before returning data. If tools return empty or error, judge how the agent REACTED — not whether the output matches initial_state. Self-correction is POSITIVE behavior.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Based on the tool results the agent actually received.
2. **Reasoning quality**: Is the thinking logical? Does the agent correctly identify what needs to be done?
3. **Agent's response to tool results**: Did the agent handle errors/empty results appropriately?
4. **Self-correction in reasoning**: If the agent reasoned "that was wrong, let me try X instead" and succeeded — this is good, not an error.
5. **Final response accuracy**: Does the response accurately reflect the tool results the agent received?
6. **Efficiency**: Reasonable number of steps to achieve the goal?

## Error Classification

Only classify UNCORRECTED errors:
- **wrong_tool**: Called wrong tool AND never corrected
- **wrong_arguments**: Incorrect arguments that caused task failure
- **missing_tool_call**: Required tool never called, task incomplete
- **hallucinated_output**: Final response contains fabricated values
- **incomplete_execution**: Part of the request never addressed
- **ignored_tool_result**: Agent didn't use important returned data in final response
- **reasoning_error**: Flawed reasoning that led to an uncorrected wrong action
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but recognized and fixed it (informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal achieved, sound reasoning, efficient execution
- 7-8: Goal achieved with self-corrections or minor inefficiencies
- 5-6: Goal partially achieved — some parts done, others missing
- 3-4: Goal mostly not achieved despite some correct reasoning
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: User's goal achieved by end of conversation (self-corrections are fine)
- false: User's goal NOT achieved — task undone or result wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on goal achievement",
  "reasoning_quality": "strong|adequate|weak",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

STATIC_MT_THINKING_JUDGE_USER = """Evaluate this agent trajectory (includes reasoning blocks). Assess both the quality of reasoning AND the correctness of actions. Judge the agent's behavior based on the tool results it received — did it reason correctly, use the right tools, and achieve the user's goal?

<conversation>
{CONV}
</conversation>"""

# ── with_user_and_policy, tool calls ────────────────────────────────

DYNAMIC_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn customer service conversation.

The conversation starts with a system prompt (role=system). The domain policy rules are under the <policy> tag — use these to evaluate the agent's behavior. Ignore the <instructions> tag — those are formatting directives for the agent, not evaluation criteria.

The full list of available tools is provided in <available_tools> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END of the conversation. Focus ONLY on the agent's behavior — did it use the right tools, handle responses correctly, and achieve what the user asked?

The agent operates under domain policy rules (visible in the conversation's system prompt). The agent must follow the critical policy rules defined there. Do NOT penalize for minor procedural details that did not affect the outcome.

IMPORTANT: Do NOT compare tool outputs against any external state. The environment may have its own logic. If a tool returns empty or error results, that is the environment's behavior — not the agent's fault. Judge the agent based on how it REACTED to the tool outputs it received.

IMPORTANT: Some tools are termination tools (e.g., transfer tools that hand off to another agent). When the agent calls a termination tool, the conversation ends immediately — no further messages are architecturally possible. Do NOT penalize the agent for not sending a message after a termination tool call, even if the policy mentions a required post-transfer message. This is a system limitation, not an agent error.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Evaluate the final outcome based on the tool results the agent actually received.
2. **Policy compliance**: Did the agent follow the critical domain rules defined in the system prompt?
3. **Agent's response to tool results**: Did the agent correctly interpret and act on the tool outputs? If a tool returned an error, did the agent handle it appropriately (retry, try alternative, inform user)?
4. **Self-correction**: If the agent made a mistake but recognized and corrected it, this is POSITIVE behavior — not an error.
5. **Final response accuracy**: Does the agent's final text response accurately reflect the actual tool results it received? No fabricated data.
6. **Efficiency**: Did the agent achieve the goal with a reasonable number of tool calls?

## Error Classification

Only classify UNCORRECTED errors — if the agent recovered from a mistake, do NOT list it as an error:
- **policy_violation**: Agent broke a critical domain rule defined in the system prompt
- **wrong_denial**: Incorrectly denied a valid request that policy allows
- **wrong_tool**: Called a wrong tool AND never corrected by calling the right one
- **wrong_arguments**: Used incorrect arguments AND the task was not completed because of it
- **missing_tool_call**: Failed to call a required tool, leaving part of the task undone
- **hallucinated_output**: Agent's final response contains fabricated values not from tool results
- **incomplete_execution**: Part of the user's request was never addressed
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but successfully corrected it (this is informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal fully achieved, policy respected, efficient execution, accurate response
- 7-8: Goal achieved with minor inefficiencies (extra calls, slightly verbose) or self-corrections
- 5-6: Goal partially achieved — some parts done correctly, others missing or wrong
- 3-4: Goal mostly not achieved — significant parts of the request unaddressed or major policy violations
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: The user's goal was achieved by the end of the conversation (even if the path had mistakes that were corrected)
- false: The user's goal was NOT achieved — the task remains undone or the result is wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on whether the goal was achieved and why",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

DYNAMIC_JUDGE_USER = """Evaluate this {DOMAIN_NAME} customer service trajectory.

<available_tools>
{TOOL_SPECS}
</available_tools>

<conversation>
{CONV}
</conversation>"""

# ── with_user_and_policy, thinking model ───────────────────────────

DYNAMIC_THINKING_JUDGE_SYSTEM = """Evaluate an agent trajectory from a reasoning model in a policy-governed customer service environment.

The conversation starts with a system prompt (role=system). The domain policy rules are under the <policy> tag — use these to evaluate the agent's behavior. Ignore the <instructions> tag — those are formatting directives for the agent, not evaluation criteria.

The conversation's system prompt also contains the full list of available tools inside <tool> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Trajectory Format

This is a reasoning model trajectory. The conversation may contain:
- **reasoning_content**: The agent's internal reasoning (in a separate field on assistant messages)
- **Text-based tool calls**: The agent generates tool calls as text in formats like <tool_call>[...]</tool_call> or native model format
- **role=tool_call**: Parsed tool call with name and arguments
- **role=tool_response**: Raw tool execution result
- **role=tool**: Tool results (may contain multiple results as JSON array)

Evaluate the agent's actions and final outcome — the reasoning shows the agent's thought process but the actions and results are what matter.

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END of the conversation. Focus ONLY on the agent's behavior — did it use the right tools, handle responses correctly, and achieve what the user asked?

The agent operates under domain policy rules (visible in the conversation's system prompt). The agent must follow the critical policy rules defined there. Do NOT penalize for minor procedural details that did not affect the outcome.

IMPORTANT: Do NOT compare tool outputs against any external state. The environment may have its own logic. If a tool returns empty or error results, that is the environment's behavior — not the agent's fault. Judge the agent based on how it REACTED to the tool outputs it received.

IMPORTANT: Some tools are termination tools (e.g., transfer tools that hand off to another agent). When the agent calls a termination tool, the conversation ends immediately — no further messages are architecturally possible. Do NOT penalize the agent for not sending a message after a termination tool call, even if the policy mentions a required post-transfer message. This is a system limitation, not an agent error.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Evaluate the final outcome based on the tool results the agent actually received.
2. **Policy compliance**: Did the agent follow the critical domain rules defined in the system prompt?
3. **Reasoning quality**: Is the agent's reasoning logical? Does it correctly identify what needs to be done?
4. **Agent's response to tool results**: Did the agent correctly interpret and act on the tool outputs? If a tool returned an error, did the agent handle it appropriately (retry, try alternative, inform user)?
5. **Self-correction**: If the agent reasoned "that was wrong, let me try X instead" and succeeded — this is good, not an error.
6. **Final response accuracy**: Does the agent's final text response accurately reflect the actual tool results it received? No fabricated data.
7. **Efficiency**: Did the agent achieve the goal with a reasonable number of tool calls?

## Error Classification

Only classify UNCORRECTED errors — if the agent recovered from a mistake, do NOT list it as an error:
- **policy_violation**: Agent broke a critical domain rule defined in the system prompt
- **wrong_denial**: Incorrectly denied a valid request that policy allows
- **wrong_tool**: Called a wrong tool AND never corrected by calling the right one
- **wrong_arguments**: Used incorrect arguments AND the task was not completed because of it
- **missing_tool_call**: Failed to call a required tool, leaving part of the task undone
- **hallucinated_output**: Agent's final response contains fabricated values not from tool results
- **incomplete_execution**: Part of the user's request was never addressed
- **reasoning_error**: Flawed reasoning that led to an uncorrected wrong action
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but successfully corrected it (this is informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal fully achieved, policy respected, sound reasoning, efficient execution
- 7-8: Goal achieved with self-corrections or minor inefficiencies
- 5-6: Goal partially achieved — some parts done correctly, others missing or wrong
- 3-4: Goal mostly not achieved — significant parts of the request unaddressed or major policy violations
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: The user's goal was achieved by the end of the conversation (even if the path had mistakes that were corrected)
- false: The user's goal was NOT achieved — the task remains undone or the result is wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on whether the goal was achieved and why",
  "reasoning_quality": "strong|adequate|weak",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

DYNAMIC_THINKING_JUDGE_USER = """Evaluate this {DOMAIN_NAME} customer service trajectory from a reasoning model.

<conversation>
{CONV}
</conversation>"""

# ── with_user, tool calls ───────────────────────────────────────────

DYNAMIC_NO_POLICY_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn conversation.

The conversation starts with a system prompt (role=system) that contains conversation guidelines.

The full list of available tools is provided in <available_tools> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END of the conversation. Focus ONLY on the agent's behavior — did it use the right tools, handle responses correctly, and achieve what the user asked?

IMPORTANT: Do NOT compare tool outputs against any external state. The environment may have its own logic. If a tool returns empty or error results, that is the environment's behavior — not the agent's fault. Judge the agent based on how it REACTED to the tool outputs it received.

IMPORTANT: The conversation ends when the user sends '###STOP###'. This means the user is satisfied and the task is complete. Do NOT penalize the agent for the conversation ending this way.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Evaluate the final outcome based on the tool results the agent actually received.
2. **Agent's response to tool results**: Did the agent correctly interpret and act on the tool outputs? If a tool returned an error, did the agent handle it appropriately (retry, try alternative, inform user)?
3. **Self-correction**: If the agent made a mistake but recognized and corrected it, this is POSITIVE behavior — not an error.
4. **Final response accuracy**: Does the agent's final text response accurately reflect the actual tool results it received? No fabricated data.
5. **Efficiency**: Did the agent achieve the goal with a reasonable number of tool calls?

## Error Classification

Only classify UNCORRECTED errors — if the agent recovered from a mistake, do NOT list it as an error:
- **wrong_tool**: Called a wrong tool AND never corrected by calling the right one
- **wrong_arguments**: Used incorrect arguments AND the task was not completed because of it
- **missing_tool_call**: Failed to call a required tool, leaving part of the task undone
- **hallucinated_output**: Agent's final response contains fabricated values not from tool results
- **incomplete_execution**: Part of the user's request was never addressed
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but successfully corrected it (this is informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal fully achieved, efficient execution, accurate response
- 7-8: Goal achieved with minor inefficiencies (extra calls, slightly verbose) or self-corrections
- 5-6: Goal partially achieved — some parts done correctly, others missing or wrong
- 3-4: Goal mostly not achieved — significant parts of the request unaddressed
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: The user's goal was achieved by the end of the conversation (even if the path had mistakes that were corrected)
- false: The user's goal was NOT achieved — the task remains undone or the result is wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on whether the goal was achieved and why",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

DYNAMIC_NO_POLICY_JUDGE_USER = """Evaluate this {DOMAIN_NAME} agent trajectory.

<available_tools>
{TOOL_SPECS}
</available_tools>

<conversation>
{CONV}
</conversation>"""

# ── Single-turn judge ────────────────────────────────────────────────

SINGLE_TURN_JUDGE_SYSTEM = """Evaluate a single-turn agent interaction where the agent must plan and execute all tool calls in one response.

## Evaluation Criteria
1. **Tool selection**: Were the correct tools chosen?
2. **Argument correctness**: Were argument values valid and properly sourced?
3. **Completeness**: Were all required tool calls made?
4. **Response accuracy**: Does the summary match actual tool results?

## Error Classification
- **wrong_tool**: Called a tool not needed for the task
- **wrong_arguments**: Correct tool but wrong argument values
- **missing_tool_call**: Didn't call a required tool
- **hallucinated_output**: Summary contains fabricated values
- **no_error**: Everything correct

## Scoring (1-10)
- 9-10: All tools correct, complete, accurate summary
- 7-8: Mostly correct, minor issues
- 5-6: Some errors in tool selection or arguments
- 3-4: Significant missing or wrong tools
- 1-2: Completely wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "explanation",
  "task_completed": true|false,
  "errors": [
    {"type": "<error_category>", "description": "what went wrong", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

SINGLE_TURN_JUDGE_USER = """Evaluate this single-turn agent interaction:

<conversation>
{CONV}
</conversation>

<ground_truth>
{GT_ANSWER}
</ground_truth>

<api_examples>
{API_IO_EXAMPLES}
</api_examples>"""



# ── Prompting style (text-based tool calls; tools embedded in system prompt) ──
# Prompting trajectories embed the full tool catalog inside the conversation's
# system prompt (<tools> tags) and use text-based tool calls of the form
# <tool_call>{"name": ..., "arguments": ...}</tool_call>. Tool results are fed
# back as user messages wrapped in <tool_response>...</tool_response>, and the
# final user-facing answer is wrapped in <answer>...</answer>. Tool specs are
# therefore already in the conversation — do NOT pass them separately.

STATIC_MT_PROMPTING_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn conversation.

The agent operates in a prompting setup: the conversation's system prompt (the first message, role=system) contains the full list of available tools inside <tools> tags. Use these to:
- Verify the agent only called tools that exist in the spec
- Check if the agent used correct argument names and types as defined in the spec
- Identify if the agent missed calling a required tool that was available
- Evaluate whether the agent chose the most appropriate tool for each step

## Tool Call Format

The agent issues text-based tool calls of the form:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

Tool results are returned in subsequent messages wrapped in <tool_response>...</tool_response>. The agent's final user-facing answer is wrapped in <answer>...</answer>. Evaluate the agent's tool usage and final answer regardless of the exact surrounding formatting.

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END of the conversation. Focus ONLY on the agent's behavior — did it use the right tools, handle responses correctly, and achieve what the user asked?

IMPORTANT: Do NOT compare tool outputs against the initial_state to find inconsistencies. The environment may have its own logic (e.g., requiring login before accessing data). If a tool returns empty or error results, that is the environment's behavior — not the agent's fault. Judge the agent based on how it REACTED to the tool outputs it received, not whether the outputs match the initial_state.

## Evaluation Criteria (in order of importance)

1. **Goal achievement**: Did the user get what they asked for by the end? Evaluate the final outcome based on the tool results the agent actually received.
2. **Agent's response to tool results**: Did the agent correctly interpret and act on the tool outputs? If a tool returned an error, did the agent handle it appropriately (retry, try alternative, inform user)?
3. **Self-correction**: If the agent made a mistake but recognized and corrected it, this is POSITIVE behavior — not an error.
4. **Final answer accuracy**: Does the agent's final <answer> accurately reflect the actual tool results it received? No fabricated data.
5. **Format adherence**: Did the agent use well-formed <tool_call> JSON and wrap its final response in <answer> tags? Minor formatting slips that don't affect outcome are not major errors.
6. **Efficiency**: Did the agent achieve the goal with a reasonable number of tool calls?

## Error Classification

Only classify UNCORRECTED errors — if the agent recovered from a mistake, do NOT list it as an error:
- **wrong_tool**: Called a wrong tool AND never corrected by calling the right one
- **wrong_arguments**: Used incorrect arguments AND the task was not completed because of it
- **missing_tool_call**: Failed to call a required tool, leaving part of the task undone
- **hallucinated_output**: Agent's final answer contains fabricated values not from tool results
- **incomplete_execution**: Part of the user's request was never addressed
- **ignored_tool_result**: Tool returned data the user asked about but agent didn't include it in the answer
- **malformed_tool_call**: Tool call JSON was malformed such that the intended action did not execute and was not corrected
- **repetitive_tool_call**: Agent called the same tool with the same arguments multiple times in a row without changing inputs. This is a severe error — score should be very low (1-3).
- **self_corrected**: Agent made a mistake but successfully corrected it (this is informational, NOT an error)
- **no_error**: Everything is correct

IMPORTANT: If the agent calls the same tool consecutively with identical arguments (a stuck loop), this is a critical failure regardless of whether the overall goal was achieved. Calling the same tool with DIFFERENT arguments is acceptable (e.g., searching with refined parameters).

## Scoring (1-10)
- 9-10: Goal fully achieved, efficient execution, accurate final answer
- 7-8: Goal achieved with minor inefficiencies (extra calls, slightly verbose) or self-corrections
- 5-6: Goal partially achieved — some parts done correctly, others missing or wrong
- 3-4: Goal mostly not achieved — significant parts of the request unaddressed
- 1-3: Repetitive tool calls with identical arguments (stuck loop), or goal not achieved at all

## task_completed Rules
- true: The user's goal was achieved by the end of the conversation (even if the path had mistakes that were corrected)
- false: The user's goal was NOT achieved — the task remains undone or the result is wrong

## Output

<out_judgment>
{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on whether the goal was achieved and why",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}
  ]
}
</out_judgment>"""

STATIC_MT_PROMPTING_JUDGE_USER = """Evaluate this agent trajectory. The available tools are embedded in the conversation's system prompt (<tools> tags). Judge the agent's behavior based on the tool results it received — did it use the right tools, handle responses correctly, and achieve the user's goal?

<conversation>
{CONV}
</conversation>"""

DYNAMIC_PROMPTING_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn customer service conversation for the {DOMAIN_NAME} domain.

The agent operates in a prompting setup: the conversation's system prompt (role=system) contains the full list of available tools inside <tools> tags, along with the domain policy the agent must follow. Use the tool specs to verify the agent only called existing tools with correct argument names/types, chose appropriate tools, and did not miss required tool calls.

## Tool Call Format

The agent issues text-based tool calls of the form:
<tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>

Tool results are returned wrapped in <tool_response>...</tool_response>, and the final user-facing answer is wrapped in <answer>...</answer>.

## Core Principle

The PRIMARY criteria are (1) whether the user's goal was achieved AND (2) whether the agent followed the domain policy. A policy violation is a serious error even if the goal was achieved. Judge the agent on how it REACTED to the tool outputs it received, not on environment inconsistencies. Self-correction is POSITIVE behavior.

## Evaluation Criteria (in order of importance)
1. **Policy compliance**: Did the agent follow the domain policy (authentication, authorization, required confirmations, transfer rules)?
2. **Goal achievement**: Did the user get what they asked for by the end?
3. **Agent's response to tool results**: Correct interpretation and handling of errors/empty results.
4. **Final answer accuracy**: Does the final <answer> reflect the actual tool results? No fabrication.
5. **Efficiency**: Reasonable number of tool calls.

## Error Classification (only UNCORRECTED errors)
- **policy_violation**: Agent violated the domain policy (e.g., acted without required authentication/confirmation)
- **wrong_tool**: Called a wrong tool AND never corrected
- **wrong_arguments**: Incorrect arguments that caused task failure
- **missing_tool_call**: Required tool never called, task incomplete
- **hallucinated_output**: Final answer contains fabricated values
- **incomplete_execution**: Part of the request never addressed
- **ignored_tool_result**: Important returned data not used in the answer
- **malformed_tool_call**: Malformed tool call JSON that prevented the intended action and was not corrected
- **repetitive_tool_call**: Same tool, same arguments, repeated in a row — severe error (score 1-3)
- **self_corrected**: Mistake recognized and fixed (informational, NOT an error)
- **no_error**: Everything is correct

## Scoring (1-10)
- 9-10: Goal achieved AND policy fully followed, efficient
- 7-8: Goal achieved, policy followed, with minor inefficiencies or self-corrections
- 5-6: Goal partially achieved, or minor policy lapse
- 3-4: Goal mostly not achieved, or notable policy violation
- 1-3: Policy violation with harmful effect, stuck loop, or goal not achieved

## task_completed Rules
- true: User's goal achieved by end AND no uncorrected policy violation
- false: Goal not achieved OR an uncorrected policy violation occurred

## Output
<out_judgment>
{{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on goal achievement and policy compliance",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {{"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}}
  ]
}}
</out_judgment>"""

DYNAMIC_PROMPTING_JUDGE_USER = """Evaluate this agent trajectory for the {DOMAIN_NAME} domain. The available tools and the domain policy are embedded in the conversation's system prompt (<tools> tags). Judge both policy compliance and goal achievement based on the tool results the agent received.

<conversation>
{CONV}
</conversation>"""

DYNAMIC_NO_POLICY_PROMPTING_JUDGE_SYSTEM = """Evaluate whether an agent achieved the user's goal in a multi-turn conversation for the {DOMAIN_NAME} domain (no explicit domain policy).

The agent operates in a prompting setup: the conversation's system prompt (role=system) contains the full list of available tools inside <tools> tags. Use the tool specs to verify the agent only called existing tools with correct argument names/types, chose appropriate tools, and did not miss required tool calls.

## Tool Call Format

The agent issues text-based tool calls of the form:
<tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>

Tool results are returned wrapped in <tool_response>...</tool_response>, and the final user-facing answer is wrapped in <answer>...</answer>.

## Core Principle

The PRIMARY criterion is whether the user's goal was achieved at the END. Judge the agent on how it REACTED to the tool outputs it received, not on environment inconsistencies. Self-correction is POSITIVE behavior.

## Evaluation Criteria (in order of importance)
1. **Goal achievement**: Did the user get what they asked for by the end?
2. **Agent's response to tool results**: Correct interpretation and handling of errors/empty results.
3. **Self-correction**: Recognized and fixed mistakes are positive, not errors.
4. **Final answer accuracy**: Does the final <answer> reflect the actual tool results? No fabrication.
5. **Efficiency**: Reasonable number of tool calls.

## Error Classification (only UNCORRECTED errors)
- **wrong_tool**: Called a wrong tool AND never corrected
- **wrong_arguments**: Incorrect arguments that caused task failure
- **missing_tool_call**: Required tool never called, task incomplete
- **hallucinated_output**: Final answer contains fabricated values
- **incomplete_execution**: Part of the request never addressed
- **ignored_tool_result**: Important returned data not used in the answer
- **malformed_tool_call**: Malformed tool call JSON that prevented the intended action and was not corrected
- **repetitive_tool_call**: Same tool, same arguments, repeated in a row — severe error (score 1-3)
- **self_corrected**: Mistake recognized and fixed (informational, NOT an error)
- **no_error**: Everything is correct

## Scoring (1-10)
- 9-10: Goal fully achieved, efficient execution, accurate final answer
- 7-8: Goal achieved with minor inefficiencies or self-corrections
- 5-6: Goal partially achieved
- 3-4: Goal mostly not achieved
- 1-3: Stuck loop (repetitive identical calls) or goal not achieved

## task_completed Rules
- true: User's goal achieved by end of conversation
- false: User's goal NOT achieved

## Output
<out_judgment>
{{
  "score": <1-10>,
  "rationale": "2-3 sentence explanation focusing on goal achievement",
  "task_completed": true|false,
  "tool_call_correctness": "correct|partially_correct|incorrect",
  "errors": [
    {{"type": "<error_category>", "description": "specific description", "tool": "<tool_name or null>"}}
  ]
}}
</out_judgment>"""

DYNAMIC_NO_POLICY_PROMPTING_JUDGE_USER = """Evaluate this agent trajectory for the {DOMAIN_NAME} domain. The available tools are embedded in the conversation's system prompt (<tools> tags). Judge the agent's behavior based on the tool results it received — did it use the right tools, handle responses correctly, and achieve the user's goal?

<conversation>
{CONV}
</conversation>"""
