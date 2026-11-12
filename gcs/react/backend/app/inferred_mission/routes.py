from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from app.inferred_mission.adapter import analysis_settings, observe_remote, runtime
from steeleagle_mcp.inferred_mission.config import SectionMode, Settings
from steeleagle_mcp.inferred_mission.media import media_available, probe_video
from steeleagle_mcp.inferred_mission.storage import write_json
from steeleagle_mcp.inferred_mission.vision import PROMPT_VERSION

router = APIRouter(prefix="/api/inferred-mission", tags=["inferred mission"])
_CHUNK = 1024**2
_MIN_FREE = 256 * _CHUNK


class StartSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["live", "uploaded_video"]
    mode: Literal["analysis_only"] = "analysis_only"
    vehicle_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
    feed: Literal["local", "remote"] = "local"
    video_asset_id: str | None = None
    history_mode: Literal["recency", "uniform", "no_memory"] | None = None
    flight_started_at: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    video_takeoff_offset: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    section_mode: SectionMode | None = None


def _service(request: Request):
    return request.app.state.inferred_mission


def _run(request: Request, session_id: str):
    run = _service(request).runs.get(session_id)
    if run is None:
        raise HTTPException(404, "Analysis session not found.")
    return run


def _settings() -> Settings:
    try:
        return analysis_settings()
    except (ValueError, TypeError, OSError):
        raise HTTPException(503, "Invalid inferred mission configuration. See inference.template.toml.") from None


@router.get("/status")
async def status(request: Request):
    settings = _settings()
    try:
        configured_runtime, configured = runtime(settings, require_key=False)
    except (ValueError, TypeError, OSError):
        raise HTTPException(503, "Could not read the MCP model configuration.") from None
    return {"settings": asdict(settings), "models": configured_runtime.models, "configured": configured,
            "media_available": media_available(), "prompt_version": PROMPT_VERSION,
            "local_vehicles": sorted(request.app.state.inferred_mission_local_vehicles),
            "remote_available": bool(request.app.state.inferred_mission_remote_webserver)}


def _copy_upload(upload: UploadFile, directory: Path, settings: Settings) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "source.mp4"
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as out:
        while chunk := upload.file.read(_CHUNK):
            size += len(chunk)
            if size > settings.max_video_bytes:
                raise HTTPException(413, "The MP4 exceeds the configured upload limit.")
            if shutil.disk_usage(directory).free < _MIN_FREE + len(chunk):
                raise HTTPException(507, "Insufficient storage for video upload.")
            digest.update(chunk)
            out.write(chunk)
    try:
        media = probe_video(path, settings)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    name = (upload.filename or "video.mp4").replace("\\", "/").rsplit("/", 1)[-1][:255]
    metadata = {"id": directory.name, "name": name, "size_bytes": size, "sha256": digest.hexdigest(), **media}
    write_json(directory / "metadata.json", metadata)
    return metadata


@router.post("/videos", status_code=201)
async def upload_video(request: Request):
    settings = _settings()
    if not media_available():
        raise HTTPException(503, "Install FFmpeg and ffprobe on the GCS backend to analyze MP4 files.")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
        raise HTTPException(415, "Send one MP4 in the multipart file field.")
    limit = settings.max_video_bytes + 64 * 1024
    count = 0

    async def bounded_stream():
        nonlocal count
        async for chunk in request.stream():
            count += len(chunk)
            if count > limit:
                raise MultiPartException("The MP4 exceeds the configured upload limit.")
            yield chunk

    async with request.app.state.inferred_mission_upload_lock:
        parser = MultiPartParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
        try:
            form = await parser.parse()
        except MultiPartException as exc:
            raise HTTPException(413 if count > limit else 400, exc.message) from None
        except ValueError:
            raise HTTPException(400, "Invalid multipart upload.") from None
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or Path(upload.filename or "").suffix.lower() != ".mp4":
                raise HTTPException(422, "Choose a single .mp4 file.")
            directory = _service(request).root / "videos" / uuid.uuid4().hex
            return await asyncio.to_thread(_copy_upload, upload, directory, settings)
        finally:
            await form.close()


@router.get("/videos/{asset_id}")
async def video_metadata(request: Request, asset_id: str):
    try:
        return _service(request).video(asset_id)
    except KeyError:
        raise HTTPException(404, "Uploaded video not found.") from None


@router.get("/videos/{asset_id}/media")
async def video_media(request: Request, asset_id: str):
    await video_metadata(request, asset_id)
    return FileResponse(_service(request).root / "videos" / asset_id / "source.mp4", media_type="video/mp4")


@router.post("/sessions", status_code=201)
async def start_session(request: Request, body: StartSession):
    service = _service(request)
    settings = _settings()
    if body.history_mode:
        settings = replace(settings, history_mode=body.history_mode)
    if body.section_mode:
        settings = replace(settings, section_mode=body.section_mode)
    try:
        configured_runtime, _ = runtime(settings)
    except (ValueError, OSError, TypeError):
        raise HTTPException(503, "Configure OpenAI vision and the GCS Chat generation provider before starting.") from None
    try:
        if body.source_type == "uploaded_video":
            if not body.video_asset_id:
                raise HTTPException(422, "Upload an MP4 first.")
            if not media_available():
                raise HTTPException(503, "Install FFmpeg and ffprobe on the GCS backend.")
            run = service.start_video(body.video_asset_id, settings, configured_runtime, body.video_takeoff_offset)
        else:
            if not body.vehicle_id:
                raise HTTPException(422, "Select a vehicle for live observation.")
            if body.flight_started_at and body.flight_started_at > time.time():
                raise HTTPException(422, "The takeoff time must be in the past.")
            if body.feed == "local" and body.vehicle_id not in request.app.state.inferred_mission_local_vehicles:
                raise HTTPException(404, "The selected vehicle has no configured local imagery stream.")
            remote = request.app.state.inferred_mission_remote_webserver
            if body.feed == "remote" and not remote:
                raise HTTPException(503, "Configure the GCS remote telemetry webserver first.")
            run = service.start_live(body.vehicle_id, body.feed, settings, configured_runtime, body.flight_started_at)
            if body.feed == "remote":
                task = asyncio.create_task(observe_remote(service, run, remote))
                tasks = request.app.state.inferred_mission_source_tasks
                tasks.add(task)
                task.add_done_callback(tasks.discard)
    except KeyError:
        raise HTTPException(404, "Uploaded video not found.") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return run.snapshot()


@router.get("/sessions")
async def sessions(request: Request):
    runs = sorted(_service(request).runs.values(), key=lambda item: item.metadata["created_at"], reverse=True)
    return [run.metadata for run in runs]


@router.get("/sessions/{session_id}")
async def session(request: Request, session_id: str):
    return _run(request, session_id).snapshot()


@router.post("/sessions/{session_id}/pause")
async def pause(request: Request, session_id: str):
    run = _run(request, session_id)
    try:
        _service(request).pause(run)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return run.snapshot()


@router.post("/sessions/{session_id}/resume")
async def resume(request: Request, session_id: str):
    run = _run(request, session_id)
    try:
        configured_runtime, _ = runtime(Settings(**run.metadata["settings"]))
    except (ValueError, OSError, TypeError):
        raise HTTPException(503, "Check the configured vision and generation providers before resuming.") from None
    try:
        _service(request).resume(run, configured_runtime)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return run.snapshot()


@router.post("/sessions/{session_id}/stop")
async def stop(request: Request, session_id: str):
    run = _run(request, session_id)
    _service(request).stop(run)
    return run.snapshot()


@router.get("/sessions/{session_id}/events")
async def events(request: Request, session_id: str):
    run = _run(request, session_id)

    async def stream():
        previous = None
        while not await request.is_disconnected():
            state = (run.metadata["revision"], int(run.last_received or 0), run.source_notice)
            if state != previous:
                previous = state
                yield f"id: {run.metadata['revision']}\nevent: update\ndata: {json.dumps(run.snapshot(), ensure_ascii=False)}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@router.get("/sessions/{session_id}/sections")
async def sections(request: Request, session_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=250)):
    rows = _run(request, session_id).snapshot()["sections"]
    return {"items": rows[offset:offset + limit], "total": len(rows)}


def _section(request: Request, session_id: str, index: int):
    run = _run(request, session_id)
    row = run.records.get(index)
    if row is None:
        raise HTTPException(404, "Section not found.")
    return run, row


@router.get("/sessions/{session_id}/sections/{index}")
async def section(request: Request, session_id: str, index: int):
    _, row = _section(request, session_id, index)
    return {key: value for key, value in row.items() if key != "frame_files"}


@router.get("/sessions/{session_id}/sections/{index}/frames/{frame_index}")
async def frame(request: Request, session_id: str, index: int, frame_index: int):
    run, row = _section(request, session_id, index)
    paths = row.get("frame_files", [])
    if not 0 <= frame_index < len(paths):
        raise HTTPException(404, "Frame not found.")
    path = (run.directory / paths[frame_index]).resolve()
    if not path.is_relative_to(run.section_dir(index)) or not path.is_file():
        raise HTTPException(404, "Frame not found.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/sessions/{session_id}/sections/{index}/files/{filename}")
async def download(request: Request, session_id: str, index: int,
                   filename: Literal["mission.dsl", "mission.json", "inference.json"]):
    _, row = _section(request, session_id, index)
    if filename == "mission.dsl":
        value, media_type = row.get("dsl"), "text/plain"
    elif filename == "mission.json":
        value, media_type = row.get("mission_json"), "application/json"
    else:
        value = {key: item for key, item in row.items() if key != "frame_files"}
        media_type = "application/json"
    if value is None:
        raise HTTPException(404, "This section has no such artifact.")
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return Response(content, media_type=media_type, headers={
        "Content-Disposition": f'attachment; filename="section-{index:06d}-{filename}"',
    })
