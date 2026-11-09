from __future__ import annotations

import asyncio
import hashlib
import math
import os
from pathlib import Path
from urllib.parse import quote

import httpx
from google.protobuf.message import DecodeError

from app.chat_agent import (
    default_mcp_config_path,
    load_client_settings,
    resolve_api_key,
    run_analysis_generation,
)
from steeleagle_mcp.config import load_config
from steeleagle_mcp.inferred_mission.config import Settings, load_settings
from steeleagle_mcp.inferred_mission.service import AnalysisService, Run, Runtime
from steeleagle_mcp.inferred_mission.vision import infer_openai
from steeleagle_mcp.providers.openai import OpenAIProvider
from steeleagle_sdk.protocol.messages.telemetry_pb2 import Frame

_MCP_ROOT = Path(__file__).resolve().parents[5] / "mcp"
_MAX_REMOTE_FRAME = 8 * 1024**2


def create_service() -> AnalysisService:
    root = Path(os.environ.get("STEELEAGLE_INFERRED_MISSION_DATA", str(_MCP_ROOT / "inferred_mission_data")))
    return AnalysisService(root.expanduser())


def analysis_settings() -> Settings:
    config = Path(os.environ.get("STEELEAGLE_INFERRED_MISSION_CONFIG", str(_MCP_ROOT / "inference.toml")))
    return load_settings(config.expanduser())


def runtime(settings: Settings, *, require_key: bool = True) -> tuple[Runtime, bool]:
    path = default_mcp_config_path()
    client = load_config(str(path) if path.is_file() else None)["client"]
    vision_key = resolve_api_key(client, "openai")
    generation = load_client_settings(path)
    vision_url = os.environ.get("STEELEAGLE_INFERENCE_OPENAI_BASE_URL", client.get("openai_base_url") or "")
    configured = bool(vision_key and generation["api_key"] and generation["provider"] in {"openai", "anthropic"})
    if require_key and not configured:
        raise ValueError("Configure an OpenAI key for vision and a supported GCS Chat provider for DSL generation.")
    generation["request_timeout"] = min(settings.request_timeout, settings.generation_timeout)
    models = {"vision_provider": "openai", "vision_model": settings.vision_model,
              "dsl_provider": generation["provider"], "dsl_model": generation["model"],
              "reasoning_effort": generation.get("reasoning_effort"),
              "vision_endpoint_hash": hashlib.sha256(vision_url.encode()).hexdigest(),
              "dsl_endpoint_hash": hashlib.sha256(generation["base_url"].encode()).hexdigest()}
    vision_reasoning = OpenAIProvider(reasoning_effort=settings.vision_reasoning_effort).responses_reasoning_options(
        settings.vision_model
    ).get("reasoning")
    if vision_reasoning:
        models["vision_reasoning_effort"] = vision_reasoning["effort"]

    async def infer(prompt, frames):
        return await infer_openai(model=settings.vision_model, prompt=prompt, frames=frames,
                                  api_key=vision_key, base_url=vision_url, timeout=settings.request_timeout,
                                  reasoning_effort=settings.vision_reasoning_effort)

    async def generate(candidate, reference):
        await run_analysis_generation(generation, candidate.instruction, reference, candidate.dispatch)

    return Runtime(models, infer, generate), configured


def capture_time(frame: Frame) -> float:
    return frame.timestamp.seconds + frame.timestamp.nanos / 1_000_000_000


def telemetry_from_frame(frame: Frame) -> dict | None:
    if frame.HasField("position_info") and frame.position_info.HasField("relative_position"):
        altitude = frame.position_info.relative_position.z
        if math.isfinite(altitude):
            return {"relative_altitude_m": altitude, "source": "imaging_frame"}
    return None


async def observe_remote(service: AnalysisService, run: Run, webserver: str) -> None:
    vehicle = run.metadata["vehicle_id"]
    url = f"{webserver.rstrip('/')}/raw/{quote(vehicle, safe='')}/latest.frame"
    interval = min(1.0, 0.5 / run.metadata["settings"]["frame_fps"])
    async with httpx.AsyncClient(timeout=5) as client:
        while service.active_live(vehicle, "remote") and service.live.get(("remote", vehicle)) is run:
            try:
                async with client.stream("GET", url, headers={"Cache-Control": "no-cache"}) as response:
                    if response.status_code != 200:
                        run.source_notice = "Waiting for the edge telemetry engine's timestamped latest.frame feed."
                    else:
                        payload = bytearray()
                        async for chunk in response.aiter_bytes():
                            payload.extend(chunk)
                            if len(payload) > _MAX_REMOTE_FRAME:
                                raise ValueError("Frame exceeds size limit")
                        frame = Frame.FromString(bytes(payload))
                        if frame.vehicle_info.name == vehicle and frame.data.startswith(b"\xff\xd8"):
                            service.ingest(vehicle, "remote", capture_time(frame), frame.data, telemetry_from_frame(frame), frame.id)
                        else:
                            run.source_notice = "Remote frame lacks matching vehicle or JPEG data."
            except (httpx.HTTPError, ValueError, DecodeError):
                run.source_notice = "The timestamped remote imagery feed is unavailable."
            await asyncio.sleep(interval)
