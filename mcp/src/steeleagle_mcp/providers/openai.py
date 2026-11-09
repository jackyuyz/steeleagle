# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: 0BSD
"""OpenAI provider implementation for MCP client."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from steeleagle_mcp.config import DEFAULT_OPENAI_MODEL
from steeleagle_mcp.providers import LLMProvider

logger = logging.getLogger("provider.openai")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


def _uses_responses(model: str) -> bool:
    return model == "gpt-6.1-sol" or model.startswith("gpt-6.1-sol-")


class OpenAIProvider(LLMProvider):
    """Provider implementation for OpenAI models."""

    name = "openai"
    default_model = DEFAULT_OPENAI_MODEL

    def __init__(
        self, base_url: str = "", reasoning_effort: str | None = None
    ) -> None:
        self._client_class = OpenAI
        self._base_url = base_url
        self._reasoning_effort = reasoning_effort

    def chat_completion_options(self, model: str) -> dict[str, str]:
        """Shared OpenAI request options for MCP and GCS Chat tool loops."""
        if _uses_responses(model):
            raise ValueError("GPT-6.1 Sol tool calling requires the Responses API.")
        # Legacy chat models do not accept reasoning_effort, even if it remains
        # in the config after switching models (including a CLI --model override).
        if model.startswith(("gpt-4", "gpt-3.5")):
            return {}

        effort = self._reasoning_effort
        if model == "gpt-6-sol" or model.startswith("gpt-6-sol-"):
            # GPT-6 Sol supports Chat Completions tool calling only at "none".
            # Default it here so older config files work in both entry points.
            if effort is not None and effort != "none":
                raise ValueError(
                    'GPT-6 Sol tool calling requires reasoning_effort="none" '
                    "with Chat Completions."
                )
            effort = "none"

        if effort is None:
            return {}
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Invalid OpenAI reasoning_effort: {effort!r}")
        return {"reasoning_effort": effort}

    def responses_reasoning_options(self, model: str) -> dict:
        if model.startswith(("gpt-4", "gpt-3.5")):
            return {}
        effort = self._reasoning_effort
        if _uses_responses(model):
            if effort is None:
                effort = "medium"
            if effort not in ("low", "medium", "high", "xhigh", "max"):
                raise ValueError("GPT-6.1 Sol reasoning_effort must be low, medium, high, xhigh, or max.")
        return {"reasoning": {"effort": effort}} if effort is not None else {}

    def create_client(self, api_key: str) -> OpenAI:
        """Create and return an OpenAI client instance."""
        if self._base_url:
            return self._client_class(api_key=api_key, base_url=self._base_url)
        return self._client_class(api_key=api_key)

    def tools_to_provider_format(self, mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP tools to OpenAI tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in mcp_tools
        ]

    def request_tool_turn(
        self, client: OpenAI, model: str, messages: list[dict],
        instructions: str, tools: list[dict],
    ) -> tuple[dict, list[ToolCall]]:
        """Request one model turn for either MCP or the GCS mission tools."""
        if _uses_responses(model):
            inputs = []
            for message in messages:
                if "response_items" in message:
                    inputs.extend(message["response_items"])
                elif message["role"] == "tool":
                    inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                                   "output": message["content"]})
                else:
                    inputs.append(message)
            response = client.responses.create(
                model=model, instructions=instructions, input=inputs,
                tools=[{"type": "function", **tool["function"], "strict": False} for tool in tools],
                max_output_tokens=4096, store=False, include=["reasoning.encrypted_content"],
                **self.responses_reasoning_options(model),
            )
            if response.status != "completed":
                raise RuntimeError(f"OpenAI Responses turn did not complete ({response.status}).")
            # Preserve native function calls and reasoning items for stateless continuation.
            assistant = {"role": "assistant", "content": response.output_text,
                         "response_items": [item.model_dump(exclude_none=True) for item in response.output]}
            calls = [ToolCall(item.call_id, item.name, item.arguments) for item in response.output
                     if item.type == "function_call"]
            return assistant, calls

        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": instructions}, *messages],
            tools=tools, max_completion_tokens=4096, **self.chat_completion_options(model),
        )
        message = response.choices[0].message
        assistant = {**message.model_dump(exclude_none=True), "role": "assistant"}
        calls = [ToolCall(call.id, call.function.name, call.function.arguments) for call in message.tool_calls or []]
        return assistant, calls

    async def agentic_loop(
        self,
        client: OpenAI,
        session: Any,
        provider_tools: list[dict],
        messages: list[dict],
        system_prompt: str,
        model: str,
    ) -> None:
        """Run one LLM turn: send messages, execute any tool calls."""
        logger.info("LLM request  model=%s messages=%d", model, len(messages))
        assistant, calls = await asyncio.to_thread(
            self.request_tool_turn, client, model, messages, system_prompt, provider_tools,
        )
        messages.append(assistant)
        logger.info("LLM response  tools=%d content=%s", len(calls), (assistant.get("content") or "")[:120])
        for i, call in enumerate(calls, start=1):
            logger.info("tool [%d/%d] %s(%s)", i, len(calls), call.name, call.arguments)
            result = await _execute_tool(session, call)
            logger.info("tool [%d/%d] %s -> %s", i, len(calls), call.name, result["content"][:120])
            messages.append(result)

    def tool_call_to_id(self, tool_call: Any) -> str:
        """Extract tool call ID from an OpenAI tool call."""
        return tool_call.id

    def tool_result_to_message(self, tool_call_id: str, content: str) -> dict[str, Any]:
        """Convert tool result to an OpenAI tool message."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


async def _execute_tool(session: Any, tool_call: ToolCall) -> dict[str, Any]:
    """Execute an MCP tool and return the result as a message."""
    name = tool_call.name
    args = json.loads(tool_call.arguments) if tool_call.arguments else {}

    try:
        result = await session.call_tool(name, args)
    except Exception:
        logger.exception("MCP call failed: %s", name)
        raise

    if result.isError:
        logger.error("MCP error: %s  %s", name, result.content)

    text = result.content[0].text if hasattr(result.content[0], "text") else str(result)
    return {"role": "tool", "tool_call_id": tool_call.id, "content": text}
