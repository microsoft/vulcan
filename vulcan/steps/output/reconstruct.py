"""Reconstruction logic for converting trajectories into training format.

Provides:
    Validation:
        check_chat_template(messages)           -> (bool, str)
        check_tool_call_response_pairs(messages) -> (bool, str)
        check_no_empty_assistant(messages)       -> (bool, str)
        validate_reconstructed(messages)         -> (bool, list[str])

    FC reconstruction (native OpenAI format → text-based tool calls):
        reconstruct_fc_json(item)               -> list[dict]
        reconstruct_fc_xml(item)                -> list[dict]

    Policy-governed reconstruction (handles both old role=tool_call and native FC):
        reconstruct_policy_fc_json(item, domain_rules) -> list[dict]
        reconstruct_policy_fc_xml(item, domain_rules)  -> list[dict]

    Thinking-model reconstruction (text tool_call → cleaned training format):
        reconstruct_thinking_json(item)         -> list[dict]
        reconstruct_thinking_xml(item)          -> list[dict]

    Reasoning reconstruction (adds <think> blocks):
        reconstruct_reasoning(item)             -> list[dict]
        reconstruct_reasoning_xml(item)         -> list[dict]

    Tagged reconstruction (tool calls carried as tags in the message text):
        reconstruct_prompting(item)             -> list[dict]

Note: the rebuilt system prompt takes its tool specs from each ``api_list``
entry's raw ``function`` (see ``_get_tool_specs``), not from the
``normalized_schema`` that ``normalize_specs`` writes.
"""

from __future__ import annotations

import json
import re

from vulcan.prompts.training_format import (
    JSON_SYSTEM_TEMPLATE,
    XML_SYSTEM_TEMPLATE,
    POLICY_JSON_SYSTEM_TEMPLATE,
    POLICY_XML_SYSTEM_TEMPLATE,
    REASONING_SYSTEM_TEMPLATE,
    REASONING_XML_SYSTEM_TEMPLATE,
    PROMPTING_SYSTEM_TEMPLATE,
)


# ── Trajectory validation checks ─────────────────────────────────────────────

def check_chat_template(messages: list[dict]) -> tuple[bool, str]:
    """Check messages follow system, (user, assistant)* alternation.

    Valid pattern: system?, (user, assistant)*, user?
    Tool responses appear as user role — that's fine.
    Ending on either user or assistant is acceptable.

    Returns:
        (valid, reason)
    """
    if not messages:
        return False, "empty messages"

    i = 0
    if messages[0].get("role") == "system":
        i = 1

    if i >= len(messages):
        return False, "no messages after system"

    if messages[i].get("role") != "user":
        return False, (
            f"first non-system message is '{messages[i].get('role')}', expected 'user'"
        )

    prev_role = None
    for msg in messages[i:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            return False, f"unexpected role '{role}'"
        if prev_role is not None and role == prev_role:
            return False, f"consecutive '{role}' messages"
        prev_role = role

    return True, ""


def check_tool_call_response_pairs(messages: list[dict]) -> tuple[bool, str]:
    """Check every tool_call has a matching tool_response.

    Each assistant message with <tool_call> tags must be followed by
    a user message with matching <tool_response> tags.

    Returns:
        (valid, reason)
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        call_count = content.count("<tool_call>")
        if call_count == 0:
            continue

        if i + 1 >= len(messages):
            return False, f"tool_call at message {i} has no following tool_response"
        next_msg = messages[i + 1]
        if next_msg.get("role") != "user":
            return False, (
                f"tool_call at message {i} followed by '{next_msg.get('role')}', "
                "expected 'user' with tool_response"
            )
        resp_count = next_msg.get("content", "").count("<tool_response>")
        if resp_count != call_count:
            return False, (
                f"tool_call count ({call_count}) != tool_response count ({resp_count}) "
                f"at messages {i}/{i+1}"
            )

    return True, ""


def check_no_empty_assistant(messages: list[dict]) -> tuple[bool, str]:
    """Check no assistant message has both empty content and no tool_call.

    Returns:
        (valid, reason)
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").strip()
        if not content:
            return False, f"empty assistant message at index {i}"

    return True, ""


def prepend_policy(messages: list[dict], domain_rules: str) -> list[dict]:
    """Put *domain_rules* at the top of the record's system message.

    For a ``with_user_and_policy`` category the agent was driven under the policy
    and judged against it, so the exported record has to carry it — otherwise the
    model is trained on policy-compliant behaviour with no policy in context, which
    teaches the domain's rules as unconditional habits rather than as
    instruction-following.

    ``reconstruct_policy_fc_json`` / ``_xml`` already interpolate the policy through
    their own templates. This is for the reasoning and tagged formats, whose
    templates have no ``{domain_rules}`` slot; the policy goes first, matching where
    ``POLICY_JSON_SYSTEM_TEMPLATE`` puts it.

    Returns *messages* unchanged when there is no policy text, and never adds a
    second system message.
    """
    if not domain_rules or not domain_rules.strip():
        return messages
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": domain_rules.strip()}, *messages]

    head = dict(messages[0])
    existing = head.get("content", "")
    if domain_rules.strip() in existing:
        return messages
    head["content"] = f"{domain_rules.strip()}\n\n{existing}" if existing else domain_rules.strip()
    return [head, *messages[1:]]


def validate_reconstructed(messages: list[dict]) -> tuple[bool, list[str]]:
    """Run all validation checks on reconstructed messages.

    Returns:
        (valid, reasons) — reasons is a list of failure descriptions.
    """
    reasons: list[str] = []

    ok, reason = check_chat_template(messages)
    if not ok:
        reasons.append(f"chat_template: {reason}")

    ok, reason = check_tool_call_response_pairs(messages)
    if not ok:
        reasons.append(f"tool_pairs: {reason}")

    ok, reason = check_no_empty_assistant(messages)
    if not ok:
        reasons.append(f"empty_assistant: {reason}")

    return len(reasons) == 0, reasons


# ── Tool spec helpers ─────────────────────────────────────────────────────────

def _get_tool_specs(item: dict) -> list[dict]:
    """Extract tool specs from api_list field."""
    api_list = item.get("api_list", [])
    specs = []
    for api in api_list:
        func = api.get("function", api)
        specs.append(func)
    return specs


def _format_json_system(tool_specs: list[dict]) -> str:
    specs_str = json.dumps(tool_specs, indent=2)
    return JSON_SYSTEM_TEMPLATE.format(tool_specs=specs_str)


def _format_xml_system(tool_specs: list[dict]) -> str:
    lines = [json.dumps(spec) for spec in tool_specs]
    return XML_SYSTEM_TEMPLATE.format(tool_specs="\n".join(lines))


def _format_policy_json_system(tool_specs: list[dict], domain_rules: str) -> str:
    specs_str = json.dumps(tool_specs, indent=2)
    return POLICY_JSON_SYSTEM_TEMPLATE.format(domain_rules=domain_rules, tool_specs=specs_str)


def _format_policy_xml_system(tool_specs: list[dict], domain_rules: str) -> str:
    lines = [json.dumps(spec) for spec in tool_specs]
    return POLICY_XML_SYSTEM_TEMPLATE.format(domain_rules=domain_rules, tool_specs="\n".join(lines))


def _format_reasoning_system(tool_specs: list[dict]) -> str:
    specs_str = json.dumps(tool_specs)
    return REASONING_SYSTEM_TEMPLATE.format(tool_specs=specs_str)


def _format_reasoning_xml_system(tool_specs: list[dict]) -> str:
    lines = [json.dumps(spec) for spec in tool_specs]
    return REASONING_XML_SYSTEM_TEMPLATE.format(tool_specs="\n".join(lines))


# ── FC call/response builders ─────────────────────────────────────────────────

def _get_tool_name_by_call_id(conv: list[dict], call_id: str) -> str:
    for msg in conv:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                if tc.get("id") == call_id:
                    return tc.get("function", {}).get("name", "unknown")
    return "unknown"


def _fc_json_call_builder(tool_calls: list[dict], text_content: str) -> str:
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parts.append(f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>')
    content = "".join(parts)
    if text_content:
        content = text_content + content
    return content


def _fc_json_resp_builder(tool_name: str, tool_content) -> str:
    return f'<tool_response>{json.dumps({"name": tool_name, "results": tool_content})}</tool_response>'


def _fc_xml_call_builder(tool_calls: list[dict], text_content: str) -> str:
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        param_lines = [f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()]
        params = "\n".join(param_lines)
        parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
    content = "\n".join(parts)
    if text_content:
        content = text_content + "\n" + content
    return content


def _fc_xml_resp_builder(tool_name: str, tool_content) -> str:
    return f'<tool_response>\n{json.dumps({"name": tool_name, "results": tool_content})}\n</tool_response>'


# ── FC message conversion ─────────────────────────────────────────────────────

def _convert_fc_messages(item: dict, system_builder, call_builder, resp_builder) -> list[dict]:
    """Generic FC message converter — shared by JSON and XML."""
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": system_builder(tool_specs)}]

    i = 0
    while i < len(conv):
        msg = conv[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            messages.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                content = call_builder(tool_calls, msg.get("content") or "")
                messages.append({"role": "assistant", "content": content})
                responses = []
                j = i + 1
                while j < len(conv) and conv[j].get("role") == "tool":
                    tool_msg = conv[j]
                    call_id = tool_msg.get("tool_call_id", "")
                    tool_name = _get_tool_name_by_call_id(conv, call_id)
                    tool_content = tool_msg.get("content", "")
                    if isinstance(tool_content, str):
                        try:
                            tool_content = json.loads(tool_content)
                        except json.JSONDecodeError:
                            pass
                    responses.append(resp_builder(tool_name, tool_content))
                    j += 1
                if responses:
                    messages.append({"role": "user", "content": "".join(responses)})
                i = j
            else:
                messages.append({"role": "assistant", "content": msg.get("content", "")})
                i += 1
            continue

        if role == "tool":
            i += 1
            continue
        i += 1

    return messages


def reconstruct_fc_json(item: dict) -> list[dict]:
    """Reconstruct native FC trajectory into JSON tool_call format."""
    return _convert_fc_messages(item, _format_json_system, _fc_json_call_builder, _fc_json_resp_builder)


def reconstruct_fc_xml(item: dict) -> list[dict]:
    """Reconstruct native FC trajectory into XML tool_call format."""
    return _convert_fc_messages(item, _format_xml_system, _fc_xml_call_builder, _fc_xml_resp_builder)


# ── Policy-governed FC call builders ──────────────────────────────────────────

def _policy_fc_json_call_builder(calls: list[dict]) -> str:
    parts = []
    for c in calls:
        name = c.get("name", "")
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parts.append(f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>')
    return "".join(parts)


def _policy_fc_xml_call_builder(calls: list[dict]) -> str:
    parts = []
    for c in calls:
        name = c.get("name", "")
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        param_lines = [f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()]
        params = "\n".join(param_lines)
        parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
    return "\n".join(parts)


# ── Policy-governed FC message conversion ────────────────────────────────────

def _convert_policy_fc_messages(
    item: dict,
    system_builder,
    call_builder,
    resp_builder,
    domain_rules: str = "",
) -> list[dict]:
    """Convert `with_user_and_policy` trajectories to training format.

    Handles two input formats:
    - Old: role=tool_call/tool_response (iter_0, iter_1)
    - Native FC: role=assistant with tool_calls, role=tool (iter_2+)
    """
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": system_builder(tool_specs, domain_rules)}]

    has_native_fc = any(m.get("role") == "tool" and m.get("tool_call_id") for m in conv)

    i = 0
    while i < len(conv):
        msg = conv[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            messages.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                calls = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    calls.append({"name": name, "arguments": args})

                text = msg.get("content", "").strip()
                call_str = call_builder(calls)
                content = (text + " " + call_str) if text else call_str
                messages.append({"role": "assistant", "content": content})

                responses = []
                j = i + 1
                while j < len(conv) and conv[j].get("role") == "tool":
                    tool_msg = conv[j]
                    tool_call_id = tool_msg.get("tool_call_id", "")
                    tool_name = _get_tool_name_by_call_id(conv, tool_call_id)
                    tool_content = tool_msg.get("content", "")
                    if isinstance(tool_content, str):
                        try:
                            tool_content = json.loads(tool_content)
                        except json.JSONDecodeError:
                            pass
                    responses.append(resp_builder(tool_name, tool_content))
                    j += 1
                if responses:
                    messages.append({"role": "user", "content": "".join(responses)})
                i = j
                continue

            else:
                content = msg.get("content", "")
                if not has_native_fc and i + 1 < len(conv) and conv[i + 1].get("role") == "tool_call" and not content.strip():
                    i += 1
                    continue
                messages.append({"role": "assistant", "content": content})
                i += 1
                continue

        if role == "tool_call":
            calls = msg.get("content", [])
            if not isinstance(calls, list):
                calls = []
            content = call_builder(calls)
            messages.append({"role": "assistant", "content": content})
            i += 1
            continue

        if role == "tool_response":
            tool_name = "unknown"
            if i > 0 and conv[i - 1].get("role") == "tool_call":
                prev_calls = conv[i - 1].get("content", [])
                if isinstance(prev_calls, list) and prev_calls:
                    tool_name = prev_calls[0].get("name", "unknown")
            result = msg.get("content", "")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            messages.append({"role": "user", "content": resp_builder(tool_name, result)})
            i += 1
            continue

        if role == "tool":
            i += 1
            continue

        i += 1

    return messages


def reconstruct_policy_fc_json(item: dict, domain_rules: str = "") -> list[dict]:
    """Reconstruct a policy-governed trajectory into JSON tool_call format."""
    return _convert_policy_fc_messages(
        item, _format_policy_json_system, _policy_fc_json_call_builder, _fc_json_resp_builder, domain_rules
    )


def reconstruct_policy_fc_xml(item: dict, domain_rules: str = "") -> list[dict]:
    """Reconstruct a policy-governed trajectory into XML tool_call format."""
    return _convert_policy_fc_messages(
        item, _format_policy_xml_system, _policy_fc_xml_call_builder, _fc_xml_resp_builder, domain_rules
    )


# ── Thinking-model tool call converters ───────────────────────────────────────

def _thinking_json_call_converter(content: str) -> str:
    """Convert a [list] tool_call into individual JSON tool_call tags."""
    tc_match = re.search(r"<tool_call>\s*\[(.+?)\]\s*</tool_call>", content, re.DOTALL)
    if not tc_match:
        return content
    try:
        calls = json.loads("[" + tc_match.group(1) + "]")
        prefix = content[:tc_match.start()].strip()
        call_strs = [f'<tool_call>{json.dumps(c)}</tool_call>' for c in calls]
        new_content = "".join(call_strs)
        return (prefix + new_content) if prefix else new_content
    except json.JSONDecodeError:
        return content


def _thinking_xml_call_converter(content: str) -> str:
    """Convert a [list] tool_call into XML parameter format."""
    tc_match = re.search(r"<tool_call>\s*\[(.+?)\]\s*</tool_call>", content, re.DOTALL)
    if not tc_match:
        return content
    try:
        calls = json.loads("[" + tc_match.group(1) + "]")
        prefix = content[:tc_match.start()].strip()
        call_strs = []
        for c in calls:
            name = c.get("name", "")
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            param_lines = [f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()]
            params = "\n".join(param_lines)
            call_strs.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
        new_content = "\n".join(call_strs)
        return (prefix + "\n" + new_content) if prefix else new_content
    except json.JSONDecodeError:
        return content


# ── Thinking-model message conversion ─────────────────────────────────────────

def _convert_thinking_messages(item: dict, system_builder, call_converter, resp_builder) -> list[dict]:
    """Generic thinking-model message converter — shared by JSON and XML."""
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": system_builder(tool_specs)}]

    i = 0
    while i < len(conv):
        msg = conv[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            messages.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            content = msg.get("content", "")
            new_content = call_converter(content)
            messages.append({"role": "assistant", "content": new_content})

            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool":
                tool_msg = conv[i + 1]
                tool_content = tool_msg.get("content", "")
                if isinstance(tool_content, str):
                    try:
                        tool_content = json.loads(tool_content)
                    except json.JSONDecodeError:
                        pass
                responses = []
                if isinstance(tool_content, list):
                    for tr in tool_content:
                        name = tr.get("tool", "unknown")
                        result = tr.get("result", "")
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                pass
                        responses.append(resp_builder(name, result))
                if responses:
                    messages.append({"role": "user", "content": "".join(responses)})
                i += 2
            else:
                i += 1
            continue

        if role == "tool":
            i += 1
            continue
        i += 1

    return messages


def reconstruct_thinking_json(item: dict) -> list[dict]:
    """Reconstruct a thinking-model trajectory into JSON tool_call training format."""
    return _convert_thinking_messages(item, _format_json_system, _thinking_json_call_converter, _fc_json_resp_builder)


def reconstruct_thinking_xml(item: dict) -> list[dict]:
    """Reconstruct a thinking-model trajectory into XML tool_call training format."""
    return _convert_thinking_messages(item, _format_xml_system, _thinking_xml_call_converter, _fc_xml_resp_builder)


# ── Reasoning (thinking model with <think>) helpers ───────────────────────────

def _parse_native_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Parse the token-delimited native tool call format.

    Returns (prefix_text, calls_list).
    """
    native_match = re.search(r"<\|tool_calls_section_begin\|>", content)
    if not native_match:
        return content, []

    prefix = content[:native_match.start()].strip()
    tool_pattern = r"<\|tool_call_begin\|>functions\.([^:]+):\d+<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>"
    matches = re.findall(tool_pattern, content, re.DOTALL)

    calls = []
    for name, args_str in matches:
        try:
            args = json.loads(args_str) if args_str.strip() else {}
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": name, "arguments": args})

    return prefix, calls


def _parse_xml_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Parse <tool_call>[{...}]</tool_call> format. Returns (prefix_text, calls_list)."""
    tc_match = re.search(r"<tool_call>\s*\[(.+?)\]\s*</tool_call>", content, re.DOTALL)
    if not tc_match:
        return content, []

    prefix = content[:tc_match.start()].strip()
    try:
        calls = json.loads("[" + tc_match.group(1) + "]")
    except json.JSONDecodeError:
        return content, []

    return prefix, calls


def _build_reasoning_assistant_msg(reasoning: str, content: str) -> str:
    """Build assistant message with <think> block and JSON tool calls.

    Format: <think> reasoning </think> text <tool_call>{...}</tool_call>...
    """
    assistant_msg = ""
    if reasoning:
        assistant_msg += "<think> " + reasoning + " </think>"

    text, calls = _parse_native_tool_calls(content)
    if not calls:
        text, calls = _parse_xml_tool_calls(content)

    if calls:
        call_strs = "".join(
            "<tool_call>" + json.dumps({"name": c.get("name", ""), "arguments": c.get("arguments", {})}) + "</tool_call>"
            for c in calls
        )
        if text:
            assistant_msg += " " + text + " " + call_strs
        else:
            assistant_msg += " " + call_strs
    elif text:
        assistant_msg += " " + text

    return assistant_msg


def _build_reasoning_tool_responses(tool_content) -> str:
    """Build tool response string in JSON format from tool message content."""
    if isinstance(tool_content, str):
        try:
            tool_content = json.loads(tool_content)
        except json.JSONDecodeError:
            pass

    if isinstance(tool_content, list):
        resp_parts = []
        for tr in tool_content:
            name = tr.get("tool", "unknown")
            result = tr.get("result", "")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            resp_parts.append(
                "<tool_response>" + json.dumps({"name": name, "results": result}) + "</tool_response>"
            )
        return "".join(resp_parts)
    else:
        return "<tool_response>" + json.dumps(tool_content) + "</tool_response>"


def _build_reasoning_xml_assistant_msg(reasoning: str, content: str) -> str:
    """Build assistant message with <think> block and XML tool calls."""
    assistant_msg = ""
    if reasoning:
        assistant_msg += "<think> " + reasoning + " </think>"

    text, calls = _parse_native_tool_calls(content)
    if not calls:
        text, calls = _parse_xml_tool_calls(content)

    if calls:
        call_parts = []
        for c in calls:
            name = c.get("name", "")
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            param_lines = [f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()]
            params = "\n".join(param_lines)
            call_parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
        call_strs = "\n".join(call_parts)
        if text:
            assistant_msg += " " + text + " " + call_strs
        else:
            assistant_msg += " " + call_strs
    elif text:
        assistant_msg += " " + text

    return assistant_msg


def _build_reasoning_xml_tool_responses(tool_content) -> str:
    """Build XML tool response string from tool message content."""
    if isinstance(tool_content, str):
        try:
            tool_content = json.loads(tool_content)
        except json.JSONDecodeError:
            pass

    if isinstance(tool_content, list):
        resp_parts = []
        for tr in tool_content:
            name = tr.get("tool", "unknown")
            result = tr.get("result", "")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            resp_parts.append(
                "<tool_response>\n" + json.dumps({"name": name, "results": result}) + "\n</tool_response>"
            )
        return "".join(resp_parts)
    else:
        return "<tool_response>\n" + json.dumps(tool_content) + "\n</tool_response>"


# ── Reasoning reconstruction ──────────────────────────────────────────────────

def reconstruct_reasoning(item: dict) -> list[dict]:
    """Reconstruct a thinking-model trajectory into reasoning training format
    with <think> blocks.

    Handles both formats:
    - Old: role=tool_call/tool_response (early runs)
    - New: role=tool (current runs)
    """
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": _format_reasoning_system(tool_specs)}]

    i = 0
    while i < len(conv):
        msg = conv[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            messages.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")

            # Old format: assistant → tool_call → tool_response
            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool_call":
                tc_msg = conv[i + 1]
                calls = tc_msg.get("content", [])
                if not isinstance(calls, list):
                    calls = []

                clean_content = re.sub(
                    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
                    "", content, flags=re.DOTALL
                ).strip()
                clean_content = re.sub(r"<tool_call>.*?</tool_call>", "", clean_content, flags=re.DOTALL).strip()

                assistant_msg = ""
                if reasoning:
                    assistant_msg += "<think> " + reasoning + " </think>"

                call_strs = "".join(
                    "<tool_call>" + json.dumps({"name": c.get("name", ""), "arguments": c.get("arguments", {})}) + "</tool_call>"
                    for c in calls
                )
                if clean_content:
                    assistant_msg += " " + clean_content + " " + call_strs
                else:
                    assistant_msg += " " + call_strs

                messages.append({"role": "assistant", "content": assistant_msg})

                if i + 2 < len(conv) and conv[i + 2].get("role") == "tool_response":
                    tr_content = conv[i + 2].get("content", "")
                    if isinstance(tr_content, str):
                        try:
                            tr_parsed = json.loads(tr_content)
                        except json.JSONDecodeError:
                            tr_parsed = tr_content
                    else:
                        tr_parsed = tr_content

                    tool_name = calls[0].get("name", "unknown") if calls else "unknown"
                    resp_str = (
                        "<tool_response>" +
                        json.dumps({"name": tool_name, "results": tr_parsed}) +
                        "</tool_response>"
                    )
                    messages.append({"role": "user", "content": resp_str})
                    i += 3
                else:
                    i += 2
                continue

            # New format or text-only assistant
            assistant_msg = _build_reasoning_assistant_msg(reasoning, content)
            messages.append({"role": "assistant", "content": assistant_msg})

            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool":
                tool_content = conv[i + 1].get("content", "")
                resp_str = _build_reasoning_tool_responses(tool_content)
                messages.append({"role": "user", "content": resp_str})
                i += 2
            else:
                i += 1
            continue

        if role in ("tool_call", "tool_response", "tool"):
            i += 1
            continue
        i += 1

    return messages


def reconstruct_reasoning_xml(item: dict) -> list[dict]:
    """Reconstruct a thinking-model trajectory into reasoning XML training format.

    Handles both old (tool_call/tool_response) and new (tool) formats.
    """
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": _format_reasoning_xml_system(tool_specs)}]

    i = 0
    while i < len(conv):
        msg = conv[i]
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            messages.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")

            # Old format: assistant → tool_call → tool_response
            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool_call":
                tc_msg = conv[i + 1]
                calls = tc_msg.get("content", [])
                if not isinstance(calls, list):
                    calls = []

                clean_content = re.sub(
                    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
                    "", content, flags=re.DOTALL
                ).strip()
                clean_content = re.sub(r"<tool_call>.*?</tool_call>", "", clean_content, flags=re.DOTALL).strip()

                assistant_msg = ""
                if reasoning:
                    assistant_msg += "<think> " + reasoning + " </think>"

                call_parts = []
                for c in calls:
                    name = c.get("name", "")
                    args = c.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    param_lines = [f"<parameter={k}>\n{v}\n</parameter>" for k, v in args.items()]
                    params = "\n".join(param_lines)
                    call_parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
                call_strs = "\n".join(call_parts)
                if clean_content:
                    assistant_msg += " " + clean_content + " " + call_strs
                else:
                    assistant_msg += " " + call_strs

                messages.append({"role": "assistant", "content": assistant_msg})

                if i + 2 < len(conv) and conv[i + 2].get("role") == "tool_response":
                    tr_content = conv[i + 2].get("content", "")
                    if isinstance(tr_content, str):
                        try:
                            tr_parsed = json.loads(tr_content)
                        except json.JSONDecodeError:
                            tr_parsed = tr_content
                    else:
                        tr_parsed = tr_content
                    tool_name = calls[0].get("name", "unknown") if calls else "unknown"
                    resp_str = (
                        "<tool_response>\n" +
                        json.dumps({"name": tool_name, "results": tr_parsed}) +
                        "\n</tool_response>"
                    )
                    messages.append({"role": "user", "content": resp_str})
                    i += 3
                else:
                    i += 2
                continue

            # New format or text-only
            assistant_msg = _build_reasoning_xml_assistant_msg(reasoning, content)
            messages.append({"role": "assistant", "content": assistant_msg})

            if i + 1 < len(conv) and conv[i + 1].get("role") == "tool":
                tool_content = conv[i + 1].get("content", "")
                resp_str = _build_reasoning_xml_tool_responses(tool_content)
                messages.append({"role": "user", "content": resp_str})
                i += 2
            else:
                i += 1
            continue

        if role in ("tool_call", "tool_response", "tool"):
            i += 1
            continue
        i += 1

    return messages


# ── Prompting (reasoning) reconstruction ──────────────────────────────────────

def _format_prompting_system(tool_specs: list[dict]) -> str:
    """Canonical prompting_synth system prompt with tools as an OpenAI-format
    JSON array embedded inside <tools></tools>."""
    arr = [{"type": "function", "function": fn} for fn in tool_specs]
    return PROMPTING_SYSTEM_TEMPLATE.format(tools_json=json.dumps(arr))


def _split_prompting_assistant(content: str) -> str:
    """Convert a model-generated prompting assistant turn into the reasoning
    training format.

    The model emits free reasoning text followed by <tool_call> blocks or an
    <answer> block, but no <think> wrapper. For tool-call turns we take the text
    before the first <tool_call> and wrap it in a single <think>...</think> block.
    For answer turns we emit the <answer> with NO <think> wrapper.

      "reasoning text <tool_call>{...}</tool_call>"
        -> "<think>reasoning text</think><tool_call>{...}</tool_call>"
      "<answer>final</answer>"                 (no preceding text)
        -> "<answer>final</answer>"
    """
    if content is None:
        content = ""

    tc_idx = content.find("<tool_call>")
    ans_idx = content.find("<answer>")

    candidates = [idx for idx in (tc_idx, ans_idx) if idx != -1]
    if not candidates:
        # No tool_call and no answer: treat the whole text as the answer body
        # (answer turns carry no <think> wrapper).
        body = content.strip()
        return "<answer>" + body + "</answer>"

    # Answer turn (no preceding tool_call): emit the answer with no <think>.
    if ans_idx != -1 and (tc_idx == -1 or ans_idx <= tc_idx):
        return content[ans_idx:].strip()

    split = min(candidates)
    pre = content[:split].strip()
    rest = content[split:].strip()
    return "<think>" + pre + "</think>" + rest


def _normalize_tool_response(content: str) -> str:
    """Trim surrounding whitespace inside each <tool_response>...</tool_response>
    so the result matches the compact form <tool_response>{...}</tool_response>."""
    if not content:
        return content

    def _trim(m):
        return "<tool_response>" + m.group(1).strip() + "</tool_response>"

    return re.sub(r"<tool_response>(.*?)</tool_response>", _trim, content,
                  flags=re.DOTALL).strip()


def reconstruct_prompting(item: dict) -> list[dict]:
    """Reconstruct a prompting-style (model-generated) trajectory into the
    prompting_synth reasoning training format.

    Transform per message:
      - system  -> canonical reasoning system prompt + <tools>[OpenAI-array]</tools>
      - user (task)         -> "<query>"
      - user (tool results) -> kept as <tool_response>...</tool_response> (normalized)
      - assistant           -> pre-block text wrapped in <think>...</think>,
                               followed by the original <tool_call>/<answer> blocks
    """
    tool_specs = _get_tool_specs(item)
    conv = item.get("conversation", [])
    messages = [{"role": "system", "content": _format_prompting_system(tool_specs)}]

    for msg in conv:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, (list, dict)):
            content = json.dumps(content)

        if role == "system":
            continue

        if role == "user":
            if "<tool_response>" in content:
                messages.append({"role": "user",
                                 "content": _normalize_tool_response(content)})
            else:
                # A user instruction / follow-up task.
                messages.append({"role": "user", "content": content.strip()})
            continue

        if role == "assistant":
            messages.append({"role": "assistant",
                             "content": _split_prompting_assistant(content)})
            continue

    return messages
