"""Exercise provider request serialization without an API or a live drone."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from openai import OpenAI

from steeleagle_mcp.client import _create_provider, run_chat
from steeleagle_mcp.config import load_config
from steeleagle_mcp.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-6-sol", "none", {"reasoning_effort": "none"}),
        ("gpt-6-sol", None, {"reasoning_effort": "none"}),
        ("gpt-6-sol-2026-09-22", None, {"reasoning_effort": "none"}),
        ("gpt-4o", None, {}),
        ("gpt-4o", "none", {}),
        ("gpt-5", "low", {"reasoning_effort": "low"}),
    ],
)
def test_openai_reasoning_options(model, effort, expected):
    assert OpenAIProvider(reasoning_effort=effort).chat_completion_options(model) == expected


def test_gpt6_sol_rejects_incompatible_tool_reasoning():
    with pytest.raises(ValueError, match='requires reasoning_effort="none"'):
        OpenAIProvider(reasoning_effort="high").chat_completion_options("gpt-6-sol")


def test_openai_rejects_unknown_reasoning_effort():
    with pytest.raises(ValueError, match="Invalid OpenAI reasoning_effort"):
        OpenAIProvider(reasoning_effort="invalid").chat_completion_options("gpt-5")


@pytest.mark.parametrize(("model", "effort"), [
    ("gpt-6-sol", "none"), ("gpt-6-sol", None), ("gpt-6.1-sol", "medium"), ("gpt-6.1-sol", None),
])
def test_mcp_config_reaches_every_openai_tool_round(tmp_path, model, effort):
    config_path = tmp_path / "client.toml"
    config_path.write_text(
        f'[client]\nprovider = "openai"\nmodel = "{model}"\n'
        'openai_api_key = "test-key"\n'
        + (f'reasoning_effort = "{effort}"\n' if effort else "")
    )
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if model == "gpt-6.1-sol":
            assert request.url.path == "/v1/responses"
            if len(requests) == 1:
                output = [
                    {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "reasoning-fixture"},
                    {"id": "fc_1", "type": "function_call", "call_id": "call-compile",
                     "name": "compile_mission_dsl", "arguments": "{}", "status": "completed"},
                ]
            else:
                output = [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "Mission ready.", "annotations": []}]}]
            return httpx.Response(200, json={"id": "resp-test", "object": "response", "created_at": 0,
                                             "model": model, "status": "completed", "output": output})
        message = {"role": "assistant", "content": "Mission ready."}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-compile",
                    "type": "function",
                    "function": {"name": "compile_mission_dsl", "arguments": "{}"},
                }],
            }
        return httpx.Response(200, json={
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
            }],
        })

    session = AsyncMock()
    session.list_tools.return_value = SimpleNamespace(tools=[SimpleNamespace(
        name="compile_mission_dsl", description="Compile a mission", inputSchema={"type": "object"}
    )])
    session.call_tool.return_value = SimpleNamespace(
        isError=False, content=[SimpleNamespace(text='{"ok": true}')]
    )

    @asynccontextmanager
    async def fake_stdio(_params):
        yield None, None

    @asynccontextmanager
    async def fake_session(_read, _write):
        yield session

    with OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    ) as api:
        with (
            patch("steeleagle_mcp.client.stdio_client", fake_stdio),
            patch("steeleagle_mcp.client.ClientSession", fake_session),
            patch.object(OpenAIProvider, "create_client", return_value=api),
            patch("builtins.input", side_effect=["Draft a mission", "quit"]),
        ):
            asyncio.run(run_chat(load_config(str(config_path)), None, None, None, None))

    assert len(requests) == 2
    assert all(r["model"] == model for r in requests)
    if model == "gpt-6.1-sol":
        assert all(r["reasoning"] == {"effort": "medium"} and r["store"] is False for r in requests)
        assert requests[0]["tools"][0]["name"] == "compile_mission_dsl"
        assert requests[0]["tools"][0]["strict"] is False
        assert requests[1]["input"][1]["encrypted_content"] == "reasoning-fixture"
        assert requests[1]["input"][-1] == {
            "type": "function_call_output", "call_id": "call-compile", "output": '{"ok": true}',
        }
    else:
        assert all(r["reasoning_effort"] == "none" for r in requests)
        assert requests[1]["messages"][-1]["role"] == "tool"
    session.call_tool.assert_awaited_once_with("compile_mission_dsl", {})


def test_mcp_anthropic_does_not_receive_openai_reasoning():
    provider = _create_provider("anthropic", reasoning_effort="none")
    client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Mission ready.")], stop_reason="end_turn"
    ))))
    asyncio.run(provider.agentic_loop(
        client=client,
        session=AsyncMock(),
        provider_tools=[],
        messages=[{"role": "user", "content": "Draft a mission"}],
        system_prompt="Plan missions",
        model=provider.default_model,
    ))
    request = client.messages.create.call_args.kwargs
    assert "reasoning_effort" not in request
    assert "thinking" not in request
    assert "output_config" not in request
