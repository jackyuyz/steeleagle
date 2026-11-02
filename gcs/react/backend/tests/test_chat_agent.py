import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import OpenAI

from app.chat_agent import (
    CHAT_RECOMMENDED_WORKFLOW,
    build_system_prompt,
    dispatch_tool,
    _extract_dsl_fence,
    load_client_settings,
    resolve_api_key,
    run_chat_turn,
    sanitize_translate_payload,
)
from app.dsl_graph import init_dsl_parser, parse_dsl_to_graph
from steeleagle_mcp.providers.anthropic import AnthropicProvider
from steeleagle_mcp.providers.openai import OpenAIProvider


MINIMAL_DSL = """\
Actions:
    TakeOff take_off(take_off_altitude = 10.0)
    Land land()
Mission:
    Start take_off
    During take_off:
        done -> land
"""


@pytest.fixture(scope="module", autouse=True)
def dsl_parser():
    init_dsl_parser()


def test_resolve_api_key_prefers_config_then_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key({"openai_api_key": "sk-from-config"}, "openai") == "sk-from-config"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert resolve_api_key({}, "openai") == "sk-from-env"
    assert resolve_api_key({"openai_api_key": ""}, "openai") == "sk-from-env"


def test_dispatch_unknown_tool():
    result = dispatch_tool("takeoff", {})
    assert result["ok"] is False


def test_sanitize_translate_payload_strips_save_workflow():
    payload = {
        "ok": True,
        "recommended_workflow": [
            "Call compile_mission_dsl with the DSL.",
            "After compile succeeds, call save_mission_files with the compile_id.",
        ],
    }
    cleaned = sanitize_translate_payload(payload)
    assert cleaned["recommended_workflow"] == CHAT_RECOMMENDED_WORKFLOW
    assert all("save_mission_files" not in step for step in cleaned["recommended_workflow"])
    # Original payload left unchanged.
    assert "save_mission_files" in payload["recommended_workflow"][1]


def test_build_system_prompt_includes_catalog():
    prompt = build_system_prompt()
    assert "SteelEagle GCS mission-planning assistant" in prompt
    assert "DSL catalog (reference only" in prompt
    assert "TakeOff" in prompt or "takeoff" in prompt.lower()
    assert "{latitude, longitude, altitude, heading}" in prompt
    assert "{catalog_section}" not in prompt


def test_parse_dsl_to_graph_minimal():
    graph = parse_dsl_to_graph(MINIMAL_DSL)
    assert graph["start_id"] == "take_off"
    assert len(graph["nodes"]) == 2
    assert any(e["event_id"] == "done" for e in graph["edges"])


def test_load_client_settings_reads_repo_config():
    settings = load_client_settings()
    assert settings["provider"] in {"openai", "anthropic"}
    assert settings["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "effort", "expected_effort"),
    [("gpt-6-sol", "none", "none"), ("gpt-6-sol", None, "none"), ("gpt-4o", "none", None)],
)
async def test_openai_config_and_reasoning_reach_every_chat_tool_round(
    tmp_path, model, effort, expected_effort
):
    config_path = tmp_path / "client.toml"
    config_path.write_text(
        f'[client]\nprovider = "openai"\nmodel = "{model}"\n'
        'openai_api_key = "test-key"\n'
        + ('reasoning_effort = "none"\n' if effort else "")
    )
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": f"Mission ready.\n```dsl\n{MINIMAL_DSL}```"}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-compile",
                    "type": "function",
                    "function": {
                        "name": "compile_mission_dsl",
                        "arguments": json.dumps({"dsl": MINIMAL_DSL}),
                    },
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

    with OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    ) as api:
        with patch.object(OpenAIProvider, "create_client", return_value=api):
            result = await run_chat_turn(
                [{"role": "user", "content": "Take off to 10 meters, then land."}],
                config_path=config_path,
            )

    assert len(requests) == 2
    assert all(r["model"] == model for r in requests)
    if expected_effort is None:
        assert all("reasoning_effort" not in r for r in requests)
    else:
        assert all(r["reasoning_effort"] == expected_effort for r in requests)
    tool_reply = requests[1]["messages"][-1]
    assert tool_reply["role"] == "tool"
    assert json.loads(tool_reply["content"])["ok"]
    assert result["draft"]["normalized_dsl"] == MINIMAL_DSL
    assert result["artifacts"][0]["payload"]["start_id"] == "take_off"


@pytest.mark.asyncio
async def test_anthropic_chat_ignores_openai_reasoning_config(tmp_path):
    config_path = tmp_path / "client.toml"
    config_path.write_text(
        '[client]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\n'
        'anthropic_api_key = "test-key"\nreasoning_effort = "none"\n'
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Apply puts the draft into the FSM Builder.")],
        stop_reason="end_turn",
    ))))
    with patch.object(AnthropicProvider, "create_client", return_value=client):
        await run_chat_turn(
            [{"role": "user", "content": "What does Apply do?"}], config_path=config_path
        )
    request = client.messages.create.call_args.kwargs
    assert "reasoning_effort" not in request
    assert "thinking" not in request
    assert "output_config" not in request


@pytest.mark.asyncio
async def test_run_chat_turn_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_settings(_path=None):
        return {
            "config_path": "mcp/config.toml",
            "provider": "openai",
            "model": "gpt-4o",
            "base_url": "",
            "api_key": None,
        }

    with patch("app.chat_agent.load_client_settings", fake_settings):
        with pytest.raises(ValueError, match="No API key"):
            await run_chat_turn([{"role": "user", "content": "patrol area B"}])


@pytest.mark.asyncio
async def test_run_chat_turn_builds_artifact(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def fake_openai(settings, messages, system, tool_payloads, status):
        tool_payloads.append(
            {
                "ok": True,
                "normalized_dsl": MINIMAL_DSL,
                "mission_json": {"start_action_id": "take_off_1"},
                "compile_id": "compile-test",
                "errors": [],
            }
        )
        return "Here is a draft mission.\n```dsl\n...\n```"

    with patch("app.chat_agent._run_openai", fake_openai):
        with patch(
            "app.chat_agent.load_client_settings",
            return_value={
                "config_path": "mcp/config.toml",
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "",
                "api_key": "sk-test",
            },
        ):
            result = await run_chat_turn(
                [{"role": "user", "content": "take off then return home"}]
            )

    assert "draft" in result
    assert result["draft"]["normalized_dsl"]
    assert len(result["artifacts"]) == 1
    art = result["artifacts"][0]
    assert art["target"] == "fsm-builder"
    assert art["payload"]["start_id"] == "take_off"
    assert len(art["payload"]["nodes"]) == 2


def test_extract_dsl_fence():
    text = f"Updated draft:\n```dsl\n{MINIMAL_DSL}\n```\nYou can apply it."
    assert "TakeOff" in (_extract_dsl_fence(text) or "")


@pytest.mark.asyncio
async def test_run_chat_turn_builds_artifact_from_fenced_dsl_without_tools(monkeypatch):
    """When the model skips compile_mission_dsl, still produce Apply from ```dsl."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def fake_openai(settings, messages, system, tool_payloads, status):
        assert tool_payloads == []
        return (
            "I'll update the mission.\n"
            f"```dsl\n{MINIMAL_DSL}\n```\n"
            "You can apply this to the FSM Builder."
        )

    with patch("app.chat_agent._run_openai", fake_openai):
        with patch(
            "app.chat_agent.load_client_settings",
            return_value={
                "config_path": "mcp/config.toml",
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "",
                "api_key": "sk-test",
            },
        ):
            result = await run_chat_turn(
                [{"role": "user", "content": "change battery to 30 percent"}]
            )

    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["label"] == "Apply draft to FSM Builder"
    assert result["draft"]["normalized_dsl"]
