"""Unified Agent class for all LLM interactions.

    Agent(name, system_prompt, llm, ...)
    await agent.run(messages, progress, ...)       -> dict
    await agent.run_conversation(...)              -> list[dict]

Every step in the pipeline drives its model through this class, which owns
system-prompt injection, empty-response retries, token accounting, and optional
output parsing. Provider differences are handled a layer below, in
:mod:`vulcan.llm` — an Agent never learns which vendor answered it.

``LLMError`` propagates rather than being swallowed, so a dead API key fails the
run loudly instead of producing rows with no model contribution.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

from .exceptions import ParseError

if TYPE_CHECKING:  # avoids a cycle: vulcan.llm.base imports vulcan.core.exceptions
    from ..llm import LLMClient


class Agent:
    """Single agent handling single-turn, multi-turn, and function-calling flows."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        thinking_model: bool = False,
        output_parser: Callable | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_model = thinking_model
        self.output_parser = output_parser

        # Set by the owning step / runner:
        self.empty_response_retries: int = 0
        self.logger = None          # PipelineLogger | None
        self._step_name: str = ""
        self._category: str | None = None

    # ── Main entry points ────────────────────────────────────────────────

    async def run(
        self,
        messages: list[dict],
        progress=None,
        *,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        """Send messages and return the parsed response.

        Args:
            messages:      list of ``{"role": …, "content": …}`` dicts. The
                           system message is prepended automatically when absent.
            progress:      tqdm progress bar (optional).
            system_prompt: override the agent's default system prompt.
            tools:         OpenAI tool definitions for function calling.
            stop:          stop sequences, passed through when the model supports them.

        Returns:
            dict with keys: role, content, reasoning_content, finish_reason,
            usage, parsed_content, messages — plus ``tool_calls`` when the model
            requested one.

        Raises:
            LLMError: when the provider fails after all retries.
        """
        sp = system_prompt or self.system_prompt

        # Prepend system message if not already present
        if messages and messages[0].get("role") == "system":
            final_messages = deepcopy(messages)
        else:
            final_messages = [{"role": "system", "content": sp}] + deepcopy(messages)

        response: dict | None = None
        for retry in range(1 + self.empty_response_retries):
            response = await self.llm.call(
                final_messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_model=self.thinking_model,
                tools=tools,
                stop=stop,
            )
            # A turn that requests a tool call legitimately carries no text, so
            # tool_calls counts as a non-empty response. Treating it as empty
            # burns the whole retry budget and can end a trajectory early.
            message = (response.get("choices") or [{}])[0].get("message", {})
            if (message.get("content") or "").strip() or message.get("tool_calls"):
                break
            if retry < self.empty_response_retries:
                await asyncio.sleep(2)

        if progress is not None:
            progress.update(1)

        # Token logging
        usage = response.get("usage", {}) if response else {}
        if self.logger and usage:
            self.logger.llm_call(
                step_name=self._step_name,
                agent_name=self.name,
                category=self._category,
                model=self.model or self.llm.default_model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        choices = (response or {}).get("choices") or []
        if not choices:
            return self._empty_response(final_messages)

        choice = choices[0]
        msg = choice.get("message", {})
        content: str = msg.get("content", "") or ""
        reasoning_content: str = (
            msg.get("reasoning_content", "") or msg.get("cot", "") or ""
        )
        finish_reason: str = choice.get("finish_reason", "")
        usage = (response or {}).get("usage", {})

        # Native function-call response
        if msg.get("tool_calls"):
            return {
                "role": self.name,
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_calls": msg["tool_calls"],
                "finish_reason": finish_reason,
                "usage": usage,
                "parsed_content": None,
                "messages": final_messages,
            }

        parsed: Any = content
        if self.output_parser:
            try:
                parsed = await self.output_parser(
                    {
                        "content": content,
                        "cot": reasoning_content,
                        "finish_reason": finish_reason,
                    }
                )
            except ParseError:
                parsed = content  # fall back to raw content
            except Exception:
                parsed = content

        return {
            "role": self.name,
            "content": content,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "usage": usage,
            "parsed_content": parsed,
            "messages": final_messages,
        }

    async def run_conversation(
        self,
        initial_messages: list[dict],
        follow_ups: list[str],
        progress=None,
    ) -> list[dict]:
        """Multi-turn conversation with sequential follow-up user messages.

        Returns a list of response dicts, one per turn.
        """
        messages = deepcopy(initial_messages)
        results: list[dict] = []

        result = await self.run(messages, progress)
        results.append(result)
        if not result["content"]:
            return results

        for follow_up in follow_ups:
            messages.append({"role": "assistant", "content": result["content"]})
            messages.append({"role": "user", "content": follow_up})
            result = await self.run(messages, progress)
            results.append(result)
            if not result["content"]:
                break

        return results

    # ── Helpers ──────────────────────────────────────────────────────────

    def _empty_response(self, messages: list[dict]) -> dict:
        """Return a well-formed but empty response dict."""
        return {
            "role": self.name,
            "content": "",
            "reasoning_content": "",
            "finish_reason": "error",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "parsed_content": None,
            "messages": messages,
        }
