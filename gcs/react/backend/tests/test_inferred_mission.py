import asyncio
import json
import subprocess
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Client, MockTransport, Response
from openai import OpenAI

from app.inferred_mission import routes
from steeleagle_mcp.inferred_mission.config import Settings
from steeleagle_mcp.inferred_mission.media import media_available
from steeleagle_mcp.inferred_mission.service import AnalysisService, Runtime
from steeleagle_mcp.inferred_mission.vision import load_prompt

INSTRUCTION = "Wait 5 seconds, then return home."
DSL = """Actions:
    Wait wait(duration = 5.0)
    ReturnToHome home()
Mission:
    Start wait
    During wait:
        done -> home
"""


def analysis_app(tmp_path, monkeypatch, *, gate=None):
    prompts = []

    async def infer(prompt, frames):
        prompts.append((prompt, frames))
        if gate and len(prompts) == 1:
            await gate.wait()
        return INSTRUCTION

    async def generate(candidate, reference):
        assert "Wait" in reference["reference"]["schema"]["actions"]
        wrong = candidate.dispatch("compile_mission_dsl", {"dsl": DSL.replace("5.0", "10.0")})
        assert wrong["status"] == "task_mismatch"
        assert candidate.dispatch("compile_mission_dsl", {"dsl": DSL})["ok"]

    config = Settings(section_seconds=1.0, max_frame_side=96)
    runtime = Runtime({"vision_model": "mock-vision", "dsl_model": "mock-dsl"}, infer, generate)
    monkeypatch.setattr(routes, "analysis_settings", lambda: config)
    monkeypatch.setattr(routes, "runtime", lambda *_args, **_kwargs: (runtime, True))
    app = FastAPI()
    app.include_router(routes.router)
    app.state.inferred_mission = AnalysisService(tmp_path / "analysis")
    app.state.inferred_mission_local_vehicles = {"test-drone": None, "other-drone": None}
    app.state.inferred_mission_remote_webserver = None
    app.state.inferred_mission_source_tasks = set()
    app.state.inferred_mission_upload_lock = asyncio.Lock()
    return app, prompts


async def wait_for(client, run_id, predicate):
    async with asyncio.timeout(15):
        while True:
            response = await client.get(f"/api/inferred-mission/sessions/{run_id}")
            assert response.status_code == 200
            snapshot = response.json()
            if predicate(snapshot):
                return snapshot
            await asyncio.sleep(0.03)


@pytest.mark.asyncio
@pytest.mark.skipif(not media_available(), reason="FFmpeg and ffprobe are required for MP4 tests")
async def test_uploaded_video_pause_resume_and_download(tmp_path, monkeypatch):
    video = tmp_path / "flight.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i", "color=blue:s=96x64:r=8:d=2.25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True, timeout=15)
    gate = asyncio.Event()
    app, prompts = analysis_app(tmp_path, monkeypatch, gate=gate)
    service = app.state.inferred_mission
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        bad = await client.post("/api/inferred-mission/videos", files={"file": ("broken.mp4", b"broken", "video/mp4")})
        assert bad.status_code == 422
        uploaded = await client.post("/api/inferred-mission/videos", files={"file": ("private-filename.mp4", video.read_bytes(), "video/mp4")})
        assert uploaded.status_code == 201, uploaded.text
        asset = uploaded.json()
        assert asset["duration"] == 2.25
        media = await client.get(f"/api/inferred-mission/videos/{asset['id']}/media", headers={"Range": "bytes=0-15"})
        assert media.status_code == 206 and media.content == video.read_bytes()[:16]
        created = await client.post("/api/inferred-mission/sessions", json={
            "source_type": "uploaded_video", "video_asset_id": asset["id"],
        })
        assert created.status_code == 201, created.text
        run_id = created.json()["id"]
        await wait_for(client, run_id, lambda r: r["sections"] and r["sections"][0]["status"] == "inferring")
        paused = await client.post(f"/api/inferred-mission/sessions/{run_id}/pause")
        assert paused.json()["status"] == "pausing"
        gate.set()
        await wait_for(client, run_id, lambda r: r["status"] == "paused")
        await service.close()
        recovered = AnalysisService(service.root)
        recovered.recover()
        app.state.inferred_mission = recovered
        resumed = await client.post(f"/api/inferred-mission/sessions/{run_id}/resume")
        assert resumed.status_code == 200, resumed.text
        complete = await wait_for(client, run_id, lambda r: r["status"] == "completed")
        assert [r["status"] for r in complete["sections"]] == ["completed"] * 3
        assert complete["sections"][-1]["end"] == 2.25
        assert len(prompts) == 3
        assert all("private-filename.mp4" not in prompt for prompt, _ in prompts)
        assert all(prompt.startswith(load_prompt().rstrip()) for prompt, _ in prompts)
        assert all(0 <= frame.time < 0.25 for frame in prompts[-1][1])
        path = f"/api/inferred-mission/sessions/{run_id}/sections/3"
        detail = (await client.get(path)).json()
        assert [item["section"] for item in detail["history"]] == [1, 2]
        assert detail["history"][-1]["weight"] == 1
        assert [attempt["status"] for attempt in detail["attempts"]] == ["task_mismatch", "sdk_compiled"]
        assert detail["inference_text"] == INSTRUCTION
        assert (await client.get(f"{path}/files/mission.dsl")).text == detail["dsl"]
        assert (await client.get(f"{path}/files/mission.json")).json() == detail["mission_json"]
        assert (await client.get(f"{path}/frames/0")).headers["content-type"] == "image/jpeg"
        arm = await client.post(f"/api/inferred-mission/sessions/{run_id}/backup/arm")
        assert arm.status_code == 404
        await recovered.close()


@pytest.mark.asyncio
async def test_live_sections_use_capture_time_and_isolate_sources(tmp_path, monkeypatch):
    from io import BytesIO

    from PIL import Image

    app, prompts = analysis_app(tmp_path, monkeypatch)
    service = app.state.inferred_mission
    image = BytesIO()
    Image.new("RGB", (32, 32), "blue").save(image, "JPEG")
    jpeg = image.getvalue()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/inferred-mission/sessions", json={"source_type": "live", "vehicle_id": "test-drone"})
        assert created.status_code == 201, created.text
        run_id = created.json()["id"]
        stamp = time.time() - 3
        service.ingest("other-drone", "local", stamp - 10, jpeg)
        service.ingest("test-drone", "local", stamp - 120, jpeg, received_at=stamp)
        service.ingest("test-drone", "local", stamp, jpeg)
        service.ingest("test-drone", "local", stamp, jpeg)
        service.ingest("test-drone", "local", stamp + 1, jpeg)
        first = await wait_for(client, run_id, lambda r: r["sections"] and r["sections"][0]["status"] == "completed")
        assert first["observation_started_at"] == stamp
        assert len(prompts[0][1]) == 1 and prompts[0][1][0].time == 0
        assert first["time_basis"] == "observation"
        await client.post(f"/api/inferred-mission/sessions/{run_id}/stop")
        stopped = await wait_for(client, run_id, lambda r: r["status"] == "stopped")
        assert len(stopped["sections"]) == 2
        assert stopped["sections"][1]["status"] == "completed"
        assert not service.active_live("test-drone", "local")
    await service.close()


@pytest.mark.asyncio
async def test_local_broadcaster_uses_receipt_time_for_untimestamped_simulator_frames(tmp_path, monkeypatch):
    from app import api
    from steeleagle_sdk.protocol.messages.telemetry_pb2 import Frame

    app, prompts = analysis_app(tmp_path, monkeypatch)
    service = app.state.inferred_mission
    monkeypatch.setattr(api.app.state, "inferred_mission", service, raising=False)
    messages = asyncio.Queue()

    class LocalSocket:
        async def recv(self, *, flags):
            return await messages.get()

    frame = Frame(id=42, h_res=32, v_res=32, channels=3, data=bytes(32 * 32 * 3))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/inferred-mission/sessions", json={"source_type": "live", "vehicle_id": "test-drone"})
        assert created.status_code == 201, created.text
        run_id = created.json()["id"]
        broadcaster = asyncio.create_task(api._imagery_broadcaster(LocalSocket(), "test-drone"))
        try:
            await messages.put(frame.SerializeToString())
            collecting = await wait_for(client, run_id, lambda r: r["collecting_frames"] == 1)
            assert collecting["source_notice"] == "Receiving local imagery using GCS receipt time."
            await asyncio.sleep(1.05)
            frame.id += 1
            await messages.put(frame.SerializeToString())
            await wait_for(client, run_id, lambda r: r["sections"] and r["sections"][0]["status"] == "completed")
            row = (await client.get(f"/api/inferred-mission/sessions/{run_id}/sections/1")).json()
            assert row["uses_gcs_receive_time"]
            assert row["capture_times"] == [None]
            assert row["frame_ids"] == [42]
            assert len(prompts[0][1]) == 1
        finally:
            broadcaster.cancel()
            await asyncio.gather(broadcaster, return_exceptions=True)
            await service.close()


@pytest.mark.asyncio
async def test_remote_frames_keep_the_edge_capture_timestamp(tmp_path, monkeypatch):
    from io import BytesIO

    from PIL import Image

    from app.inferred_mission import adapter
    from steeleagle_sdk.protocol.messages.telemetry_pb2 import Frame

    app, prompts = analysis_app(tmp_path, monkeypatch)
    app.state.inferred_mission_remote_webserver = "http://test-edge"
    image = BytesIO()
    Image.new("RGB", (32, 32), "blue").save(image, "JPEG")
    timestamp_ns = time.time_ns() - 2_000_000_000
    calls = 0

    def fetch(request):
        nonlocal calls
        assert str(request.url) == "http://test-edge/raw/test-drone/latest.frame"
        calls += 1
        index = min((calls - 1) // 2, 1)
        frame = Frame(id=index + 8, data=image.getvalue())
        frame.vehicle_info.name = "test-drone"
        frame.timestamp.FromNanoseconds(timestamp_ns + index * 1_250_000_000)
        return Response(200, content=frame.SerializeToString())

    monkeypatch.setattr(adapter.httpx, "AsyncClient", lambda **kwargs: AsyncClient(transport=MockTransport(fetch), **kwargs))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/inferred-mission/sessions", json={
            "source_type": "live", "feed": "remote", "vehicle_id": "test-drone",
        })
        assert created.status_code == 201, created.text
        run_id = created.json()["id"]
        await wait_for(client, run_id, lambda r: r["sections"] and r["sections"][0]["status"] == "completed")
        assert len(prompts[0][1]) == 1
        row = (await client.get(f"/api/inferred-mission/sessions/{run_id}/sections/1")).json()
        assert row["frame_ids"] == [8]
        assert row["capture_times"][0] == pytest.approx(timestamp_ns / 1_000_000_000, abs=1e-6)
        await client.post(f"/api/inferred-mission/sessions/{run_id}/stop")
        await wait_for(client, run_id, lambda r: r["status"] == "stopped")
    await asyncio.gather(*app.state.inferred_mission_source_tasks)
    await app.state.inferred_mission.close()


def test_analysis_rejects_the_sdk_unavailable_placeholder(monkeypatch):
    from steeleagle_mcp.inferred_mission.generation import Generation
    from steeleagle_mcp.nl2dsl import pipeline
    from steeleagle_mcp.nl2dsl.compiler import CompileResult

    monkeypatch.setattr(pipeline, "compile_dsl", lambda _dsl: CompileResult(False, False, None, "SDK unavailable"))
    result = Generation(INSTRUCTION, {"task": "wait", "seconds": 5}, area=None).dispatch("compile_mission_dsl", {"dsl": DSL})
    assert result["status"] == "sdk_unavailable"
    assert "mission_json" not in result


@pytest.mark.asyncio
async def test_analysis_generation_retries_using_gpt61_sol_tool_requests(monkeypatch):
    from app import chat_agent
    from steeleagle_mcp.inferred_mission.generation import Generation

    requests = []

    def complete(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        requests.append(body)
        dsl = DSL.replace("5.0", "10.0") if len(requests) == 1 else DSL
        return Response(200, json={
            "id": "mock", "created_at": 1, "object": "response", "model": "gpt-6.1-sol", "status": "completed",
            "output": [{"id": f"fc_{len(requests)}", "type": "function_call", "status": "completed",
                        "call_id": f"compile-{len(requests)}", "name": "compile_mission_dsl",
                        "arguments": json.dumps({"dsl": dsl})}],
        })

    with OpenAI(api_key="test", http_client=Client(transport=MockTransport(complete))) as client:
        monkeypatch.setattr(chat_agent.OpenAIProvider, "create_client", lambda *_args: client)
        generation = Generation(INSTRUCTION, {"task": "wait", "seconds": 5}, area=None)
        await chat_agent.run_analysis_generation({
            "provider": "openai", "api_key": "test", "model": "gpt-6.1-sol",
            "reasoning_effort": "medium", "request_timeout": 5,
        }, INSTRUCTION, generation.reference(), generation.dispatch)

    assert len(requests) == 2
    assert all(body["reasoning"] == {"effort": "medium"} for body in requests)
    result = requests[1]["input"][-1]
    assert result["type"] == "function_call_output" and result["call_id"] == "compile-1"
    assert json.loads(result["output"])["status"] == "task_mismatch"
    assert [attempt["status"] for attempt in generation.attempts] == ["task_mismatch", "sdk_compiled"]


@pytest.mark.asyncio
async def test_default_vision_sends_gpt61_sol_medium_with_images(monkeypatch):
    from openai import AsyncOpenAI
    from steeleagle_mcp.inferred_mission import vision

    requests = []

    def infer(request):
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return Response(200, json={
            "id": "mock", "created_at": 1, "object": "response", "model": "gpt-6.1-sol", "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": INSTRUCTION, "annotations": []}]}],
        })

    client = AsyncOpenAI(api_key="test", http_client=AsyncClient(transport=MockTransport(infer)))
    monkeypatch.setattr(vision, "AsyncOpenAI", lambda **_kwargs: client)
    result = await vision.infer_openai(
        model=Settings().vision_model, prompt="infer", frames=[vision.FrameSample(0, b"jpeg")],
        api_key="test", base_url="", timeout=5,
    )
    assert result == INSTRUCTION
    assert requests[0]["model"] == "gpt-6.1-sol"
    assert requests[0]["reasoning"] == {"effort": "medium"}
    assert requests[0]["input"][0]["content"][-1]["type"] == "input_image"
