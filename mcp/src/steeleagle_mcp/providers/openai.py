# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
# SPDX-License-Identifier: 0BSD
"""OpenAI provider implementation for MCP client."""

import json
import logging
from typing import Any

from openai import OpenAI

from steeleagle_mcp.providers import LLMProvider

logger = logging.getLogger("provider.openai")


class OpenAIProvider(LLMProvider):
    """Provider implementation for OpenAI models."""

    name = "openai"
    default_model = "gpt-4o"

    def __init__(
        self, base_url: str = "", reasoning_effort: str | None = None
    ) -> None:
        self._client_class = OpenAI
        self._base_url = base_url
        self._reasoning_effort = reasoning_effort

    def chat_completion_options(self, model: str) -> dict[str, str]:
        """Shared OpenAI request options for MCP and GCS Chat tool loops."""
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
        messages_with_system = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        logger.info("LLM request  model=%s messages=%d", model, len(messages))
        response = client.chat.completions.create(
            model=model,
            messages=messages_with_system,
            tools=provider_tools,
            max_completion_tokens=4096,
            **self.chat_completion_options(model),
        )

        choice = response.choices[0]
        message = choice.message
        n_tools = len(message.tool_calls) if message.tool_calls else 0
        logger.info("LLM response  finish=%s tools=%d content=%s",
                 choice.finish_reason, n_tools, (message.content or "")[:120])

        assistant_msg = message.model_dump(exclude_none=True)
        # ensure role is present
        assistant_msg["role"] = "assistant"
        messages.append(assistant_msg)
        if message.tool_calls:
            for i, tc in enumerate(message.tool_calls):
                logger.info("tool [%d/%d] %s(%s)", i + 1, n_tools,
                         tc.function.name, tc.function.arguments)
                result = await _execute_tool(session, tc)
                logger.info("tool [%d/%d] %s -> %s", i + 1, n_tools,
                         tc.function.name, result["content"][:120])
                messages.append(result)

    def tool_call_to_id(self, tool_call: Any) -> str:
        """Extract tool call ID from an OpenAI tool call."""
        return tool_call.id

    def tool_result_to_message(self, tool_call_id: str, content: str) -> dict[str, Any]:
        """Convert tool result to an OpenAI tool message."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


async def _execute_tool(session: Any, tool_call: Any) -> dict[str, Any]:
    """Execute an MCP tool and return the result as a message."""
    name = tool_call.function.name
    raw = tool_call.function.arguments
    args = json.loads(raw) if isinstance(raw, str) and raw else raw if isinstance(raw, dict) else {}

    try:
        result = await session.call_tool(name, args)
    except Exception:
        logger.exception("MCP call failed: %s", name)
        raise

    if result.isError:
        logger.error("MCP error: %s  %s", name, result.content)

    text = result.content[0].text if hasattr(result.content[0], "text") else str(result)
    return {"role": "tool", "tool_call_id": tool_call.id, "content": text}
