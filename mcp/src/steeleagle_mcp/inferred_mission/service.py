from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings
from .generation import UNBOUND_AREA, Generation
from .media import sample_section
from .storage import read_json, valid_id, write_json
from .vision import PROMPT_VERSION, FrameSample, build_prompt, load_prompt, parse_instruction, select_history

logger = logging.getLogger(__name__)
_FINAL = {"completed", "failed", "skipped", "gap"}


@dataclass(frozen=True)
class Runtime:
    models: dict
    infer: Callable[[str, list[FrameSample]], Awaitable[str]]
    generate: Callable[[Generation, dict], Awaitable[None]]


@dataclass
class Run:
    directory: Path
    metadata: dict
    records: dict[int, dict] = field(default_factory=dict)
    task: asyncio.Task | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    pending: tuple[dict, list[FrameSample]] | None = None
    frames: list[FrameSample] = field(default_factory=list)
    telemetry: dict | None = None
    anchor: float | None = None
    last_capture: float = 0.0
    last_received: float | None = None
    index: int = 1
    source_notice: str = "Waiting for timestamped imagery."

    def save(self, **changes) -> None:
        self.metadata.update(changes)
        self.metadata["revision"] += 1
        write_json(self.directory / "session.json", self.metadata)

    def record(self, record: dict) -> None:
        write_json(self.section_dir(record["index"]) / "record.json", record)
        self.records[record["index"]] = record
        self.save()

    def section_dir(self, index: int) -> Path:
        return self.directory / "sections" / f"{index:06d}"

    def snapshot(self) -> dict:
        sections = []
        for row in sorted(self.records.values(), key=lambda item: item["index"]):
            sections.append({key: row.get(key) for key in (
                "index", "start", "end", "status", "inference_status", "compile_status", "inference_text", "error",
            )})
        return {**self.metadata, "sections": sections, "last_frame_received": self.last_received,
                "collecting_section": self.index if self.frames else None,
                "collecting_frames": len(self.frames),
                "source_notice": self.source_notice if self.metadata["source_type"] == "live" else ""}


def _archive_live(directory: Path, frames: list[FrameSample], side: int) -> tuple[list[FrameSample], list[Path]]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=False)
    samples, paths = [], []
    for index, frame in enumerate(frames, start=1):
        with Image.open(io.BytesIO(frame.jpeg)) as source:
            image = source.convert("RGB")
            image.thumbnail((side, side))
            path = directory / f"{index:04d}.jpg"
            image.save(path, format="JPEG", quality=88)
        samples.append(FrameSample(frame.time, path.read_bytes(), frame.capture_time, frame.frame_id))
        paths.append(path)
    return samples, paths


class AnalysisService:
    """Owns the lifetime of GCS analysis runs and their durable records."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.runs: dict[str, Run] = {}
        self.live: dict[tuple[str, str], Run] = {}

    def recover(self) -> None:
        for path in (self.root / "runs").glob("*/session.json"):
            try:
                metadata = read_json(path)
                if not valid_id(path.parent.name) or metadata.get("id") != path.parent.name:
                    continue
                run = Run(path.parent, metadata)
                for record in sorted((run.directory / "sections").glob("*/record.json")):
                    row = read_json(record)
                    run.records[row["index"]] = row
                self.runs[metadata["id"]] = run
                if metadata["status"] in {"observing", "analyzing", "pausing", "stopping"}:
                    live = metadata["source_type"] == "live"
                    for row in list(run.records.values()):
                        if row["status"] not in _FINAL:
                            row.update(status="failed" if live else "interrupted", error="Analysis was interrupted.")
                            run.record(row)
                    run.save(status="stopped" if live else "paused", error="GCS restarted during analysis.")
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Could not recover inferred mission run %s", path.parent.name)

    async def close(self) -> None:
        tasks = [run.task for run in self.runs.values() if run.task and not run.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _new(self, settings: Settings, runtime: Runtime, **source) -> Run:
        active = sum(bool(run.task and not run.task.done()) for run in self.runs.values())
        if active >= settings.max_active_runs:
            raise ValueError("The active analysis run limit has been reached.")
        run_id = uuid.uuid4().hex
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        prompt = load_prompt()
        (directory / "prompt.md").write_text(prompt, encoding="utf-8")
        run = Run(directory, {
            "id": run_id, "schema_version": 1, "mode": "analysis_only", "created_at": time.time(),
            "revision": 0, "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "settings": asdict(settings), "models": runtime.models, "error": None, **source,
        })
        run.save()
        self.runs[run_id] = run
        return run

    def video(self, asset_id: str) -> dict:
        if not valid_id(asset_id):
            raise KeyError(asset_id)
        path = self.root / "videos" / asset_id / "metadata.json"
        if not path.is_file():
            raise KeyError(asset_id)
        return read_json(path)

    def start_video(self, asset_id: str, settings: Settings, runtime: Runtime,
                    takeoff_offset: float | None = None) -> Run:
        video = self.video(asset_id)
        if takeoff_offset is not None and takeoff_offset >= video["duration"]:
            raise ValueError("The takeoff offset must be within the video.")
        run = self._new(settings, runtime, source_type="uploaded_video", video=video, status="analyzing",
                        total_sections=math.ceil(video["duration"] / settings.section_seconds),
                        time_basis="video", video_takeoff_offset=takeoff_offset)
        run.task = asyncio.create_task(self._run_video(run, settings, runtime))
        return run

    def resume(self, run: Run, runtime: Runtime) -> None:
        if run.metadata["source_type"] != "uploaded_video" or run.metadata["status"] != "paused":
            raise ValueError("Only a paused video analysis can be resumed.")
        if run.task and not run.task.done():
            raise ValueError("This analysis is still pausing.")
        settings = Settings(**run.metadata["settings"])
        active = sum(bool(other.task and not other.task.done()) for other in self.runs.values())
        if active >= settings.max_active_runs:
            raise ValueError("The active analysis run limit has been reached.")
        if runtime.models != run.metadata["models"]:
            raise ValueError("The model configuration has changed. Start a new analysis to use it.")
        prompt = (run.directory / "prompt.md").read_text(encoding="utf-8")
        if hashlib.sha256(prompt.encode()).hexdigest() != run.metadata["prompt_sha256"]:
            raise ValueError("The archived prompt has changed. Start a new analysis.")
        run.save(status="analyzing", error=None)
        run.task = asyncio.create_task(self._run_video(run, settings, runtime))

    def pause(self, run: Run) -> None:
        if run.metadata["source_type"] != "uploaded_video" or run.metadata["status"] != "analyzing":
            raise ValueError("Only a running video analysis can be paused.")
        run.save(status="pausing")

    def stop(self, run: Run) -> None:
        if run.metadata["status"] == "paused":
            run.save(status="stopped")
        elif run.metadata["status"] in {"observing", "analyzing", "pausing"}:
            run.save(status="stopping")
            if run.metadata["source_type"] == "live" and run.frames:
                self._queue_live(run, partial=True)
            run.wake.set()

    async def _run_video(self, run: Run, settings: Settings, runtime: Runtime) -> None:
        video = run.metadata["video"]
        path = self.root / "videos" / video["id"] / "source.mp4"
        try:
            def source_digest():
                with path.open("rb") as source:
                    return hashlib.file_digest(source, "sha256").hexdigest()

            if not path.is_file() or await asyncio.to_thread(source_digest) != video["sha256"]:
                run.save(status="error", error="The uploaded video has changed or is missing. Upload the original again.")
                return
            for index in range(1, run.metadata["total_sections"] + 1):
                if run.metadata["status"] != "analyzing":
                    break
                if run.records.get(index, {}).get("status") in _FINAL:
                    continue
                start = (index - 1) * settings.section_seconds
                end = min(index * settings.section_seconds, video["duration"])
                record = {"index": index, "start": start, "end": end,
                          "input_quality": "partial" if end - start < settings.section_seconds else "complete"}
                await self._process(run, record, settings, runtime, video_path=path)
            status = run.metadata["status"]
            run.save(status={"pausing": "paused", "stopping": "stopped"}.get(status, "completed"))
        except asyncio.CancelledError:
            run.save(status="paused", error="GCS stopped during analysis.")
            raise
        except Exception:
            logger.exception("Video analysis worker failed")
            run.save(status="paused", error="Analysis was interrupted. Check GCS logs and resume.")

    def start_live(self, vehicle: str, feed: str, settings: Settings, runtime: Runtime,
                   flight_started_at: float | None = None) -> Run:
        key = (feed, vehicle)
        if key in self.live:
            raise ValueError("This vehicle already has a live analysis session.")
        run = self._new(settings, runtime, source_type="live", vehicle_id=vehicle, feed=feed,
                        flight_id=uuid.uuid4().hex, status="observing", flight_started_at=flight_started_at,
                        time_basis="takeoff" if flight_started_at else "observation")
        self.live[key] = run
        run.task = asyncio.create_task(self._run_live(run, key, settings, runtime))
        return run

    def active_live(self, vehicle: str, feed: str) -> bool:
        run = self.live.get((feed, vehicle))
        return run is not None and run.metadata["status"] == "observing"

    def ingest(self, vehicle: str, feed: str, captured_at: float, jpeg: bytes,
               telemetry: dict | None = None, frame_id: int | None = None) -> None:
        run = self.live.get((feed, vehicle))
        if run is None or run.metadata["status"] != "observing":
            return
        now = time.time()
        earliest = max(now - 30, run.metadata["created_at"] - 5)
        if not math.isfinite(captured_at) or not earliest <= captured_at <= now + 3:
            run.source_notice = "Waiting for fresh imagery with a valid capture timestamp."
            return
        if captured_at <= run.last_capture:
            return
        if run.anchor is None:
            run.anchor = captured_at
            run.save(observation_started_at=captured_at)
        duration = run.metadata["settings"]["section_seconds"]
        elapsed = captured_at - run.anchor
        index = math.floor(elapsed / duration) + 1
        if index > run.index:
            self._queue_live(run)
            if index > run.index + 1:
                start = run.index * duration + self._offset(run)
                end = (index - 1) * duration + self._offset(run)
                run.record({"index": run.index + 1, "start": start, "end": end, "status": "gap",
                            "error": "No fresh frames were received in this interval."})
            run.index = index
        run.last_capture = captured_at
        run.last_received = now
        run.source_notice = "Receiving timestamped imagery."
        relative = elapsed - (index - 1) * duration
        interval = 1 / run.metadata["settings"]["frame_fps"]
        if not run.frames or relative - run.frames[-1].time >= interval - 1e-6:
            run.frames.append(FrameSample(relative, jpeg, captured_at, frame_id))
            run.telemetry = telemetry

    @staticmethod
    def _offset(run: Run) -> float:
        takeoff = run.metadata.get("flight_started_at")
        return max(0.0, run.anchor - takeoff) if takeoff else 0.0

    def _queue_live(self, run: Run, *, partial: bool = False) -> None:
        if not run.frames:
            return
        if run.pending:
            old, _ = run.pending
            run.record({**old, "status": "skipped", "error": "A newer live section replaced this pending window."})
        duration = run.metadata["settings"]["section_seconds"]
        start = (run.index - 1) * duration + self._offset(run)
        observed = max(run.last_capture - run.anchor - (run.index - 1) * duration, 0.001)
        end = start + (min(duration, observed + 0.001) if partial else duration)
        quality = "partial" if partial or len(run.frames) < math.ceil(duration * run.metadata["settings"]["frame_fps"] / 2) else "complete"
        record = {"index": run.index, "start": start, "end": end, "telemetry": run.telemetry,
                  "capture_start": run.anchor + (run.index - 1) * duration, "capture_end": run.last_capture,
                  "input_quality": quality}
        run.pending = (record, run.frames)
        run.frames = []
        run.wake.set()

    async def _run_live(self, run: Run, key: tuple[str, str], settings: Settings, runtime: Runtime) -> None:
        try:
            while run.metadata["status"] == "observing" or run.pending:
                await run.wake.wait()
                run.wake.clear()
                if run.pending:
                    record, frames = run.pending
                    run.pending = None
                    await self._process(run, record, settings, runtime, live_frames=frames)
            run.save(status="stopped")
        except asyncio.CancelledError:
            run.save(status="stopped", error="GCS stopped during observation.")
            raise
        except Exception:
            logger.exception("Live analysis worker failed")
            run.save(status="error", error="Live analysis was interrupted. Check GCS logs.")
        finally:
            self.live.pop(key, None)

    async def _process(self, run: Run, record: dict, settings: Settings, runtime: Runtime,
                       *, video_path: Path | None = None, live_frames: list[FrameSample] | None = None) -> None:
        started = time.monotonic()
        index = record["index"]
        archived = run.records.get(index, {})
        past_attempts = archived.get("attempts", [])
        prior_inferences = list(archived.get("prior_inferences", []))
        if archived.get("inference_text"):
            prior_inferences.append({key: archived.get(key) for key in (
                "inference_text", "inference_status", "history", "vision_prompt", "error",
            )})
        record = {**record, "status": "preparing", "inference_status": "pending",
                  "compile_status": "not_attempted", "attempts": past_attempts, "error": None,
                  "prior_inferences": prior_inferences}
        run.record(record)
        generation = None
        try:
            directory = run.section_dir(index) / f"frames-{uuid.uuid4().hex}"
            if video_path:
                frames, paths = await asyncio.to_thread(sample_section, video_path, record["start"], record["end"], directory, settings)
            else:
                frames, paths = await asyncio.to_thread(_archive_live, directory, live_frames or [], settings.max_frame_side)
            if not frames:
                raise ValueError("No decodable frames in this section.")
            records = sorted(run.records.values(), key=lambda row: row["index"])
            history = select_history(records, record["start"], settings)
            base = (run.directory / "prompt.md").read_text(encoding="utf-8")
            prompt = build_prompt(base, record, history, settings.history_mode)
            record.update(status="inferring", inference_status="running", history=history, vision_prompt=prompt,
                          frame_times=[frame.time for frame in frames],
                          capture_times=[frame.capture_time for frame in frames],
                          frame_ids=[frame.frame_id for frame in frames],
                          frame_files=[str(path.relative_to(run.directory)) for path in paths],
                          frame_count=len(frames), input_quality="sparse" if len(frames) == 1 else record["input_quality"])
            run.record(record)
            async with asyncio.timeout(settings.request_timeout):
                text = await runtime.infer(prompt, frames)
            record["inference_text"] = text
            task = parse_instruction(text)
            instruction = " ".join(text.split())
            if "takeoff" in task and (record.get("telemetry") or {}).get("relative_altitude_m", 0) > 2:
                instruction = instruction.split(", then ", 1)[1]
                instruction = instruction[0].upper() + instruction[1:]
                task.pop("takeoff")
                record["adaptation"] = "Omitted takeoff because the last sampled frame reports relative altitude above 2 m."
            unresolved = []
            if task["task"] in {"track", "precision_land"}:
                unresolved.append("A detector class is known. The specific target and execution detector are unverified.")
            if task["task"] in {"patrol", "survey"}:
                unresolved.append(f"{UNBOUND_AREA} is a placeholder. Select a real KML area before planning execution.")
            if task["task"] in {"climb", "wait"}:
                unresolved.append("Observed waiting time does not establish the remaining task duration.")
            record.update(status="generating", inference_status="valid", task=task, instruction=instruction,
                          unresolved=unresolved, context_status="unresolved" if unresolved else "unverified",
                          inference_elapsed=round(time.monotonic() - started, 3))
            run.record(record)
            generation = Generation(instruction, task, area=None)
            generation_started = time.monotonic()
            reference = await asyncio.to_thread(generation.reference)
            if not reference["ok"]:
                raise ValueError("DSL reference is unavailable. Check the SteelEagle SDK installation.")
            record["dsl_reference"] = reference["reference"]
            run.record(record)
            async with asyncio.timeout(settings.generation_timeout):
                await runtime.generate(generation, reference)
            record["generation_elapsed"] = round(time.monotonic() - generation_started, 3)
            latest = generation.attempts[-1] if generation.attempts else None
            if latest is None:
                raise ValueError("The generation model did not submit a DSL candidate.")
            record["compile_status"] = latest["status"]
            if not latest["ok"]:
                raise ValueError("No DSL candidate passed compilation and task checks.")
            artifact = {"dsl": latest["normalized_dsl"], "mission_json": latest["mission_json"]}
            artifact_id = hashlib.sha256(json.dumps(artifact, sort_keys=True, allow_nan=False).encode()).hexdigest()
            record.update(status="completed", artifact_id=artifact_id, sdk_version=latest["sdk_version"], **artifact)
        except asyncio.CancelledError:
            record.update(status="interrupted", error="Analysis was interrupted.")
            raise
        except TimeoutError:
            record.update(status="failed", error="The section exceeded its model processing time limit.")
        except ValueError as exc:
            record.update(status="failed", error=str(exc))
        except Exception:
            logger.warning("Inference section %s failed", index, exc_info=False)
            record.update(status="failed", error="Section processing failed. Check model access, media, and SDK setup.")
        finally:
            if generation is not None:
                record["attempts"] = [*past_attempts, *generation.attempts]
                if record["compile_status"] == "not_attempted" and generation.attempts:
                    record["compile_status"] = generation.attempts[-1]["status"]
            if record["inference_status"] == "running":
                record["inference_status"] = "invalid" if "inference_text" in record else "failed"
            record["elapsed"] = round(time.monotonic() - started, 3)
            run.record(record)
