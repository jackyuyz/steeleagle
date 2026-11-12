from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .vision import FrameSample


def media_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _run(command: list[str], timeout: int = 45) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError:
        raise ValueError("Install FFmpeg and ffprobe to analyze MP4 files.") from None
    except subprocess.TimeoutExpired:
        raise ValueError("Video decoding timed out.") from None
    except subprocess.CalledProcessError:
        raise ValueError("The video could not be decoded as MP4.") from None


def probe_video(path: Path, settings: Settings) -> dict:
    result = _run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov",
        "-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json", str(path),
    ])
    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        width, height = stream["width"], stream["height"]
        duration = float(stream.get("duration") or info["format"]["duration"])
    except (KeyError, IndexError, ValueError, TypeError):
        raise ValueError("The MP4 must contain a video stream with a known duration.") from None
    if not math.isfinite(duration) or not 0 < duration <= settings.max_video_seconds:
        raise ValueError(f"Video duration must be within {settings.max_video_seconds:g} seconds.")
    if not 0 < width * height <= 36_000_000:
        raise ValueError("Video resolution exceeds the analysis limit.")
    _run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
          "-threads", "1", "-f", "mov", "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"])
    return {"duration": duration, "width": width, "height": height}


def sample_section(path: Path, start: float, end: float, directory: Path,
                   settings: Settings) -> tuple[list[FrameSample], list[Path]]:
    """Decode only in-range frames, including an actual frame in a short final section."""
    directory.mkdir(parents=True, exist_ok=False)
    duration = end - start
    side = settings.max_frame_side
    if settings.section_mode == "sliding":
        # Compare sampling slots so a late frame does not shift later samples.
        select = f"isnan(prev_selected_t)+gt(floor(t*{settings.frame_fps:.9f}+1e-6)\\,floor(prev_selected_t*{settings.frame_fps:.9f}+1e-6))"
    else:
        select = f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{1/settings.frame_fps:.9f})"
    filters = (
        f"trim=duration={duration:.9f},select={select},"
        f"scale={side}:{side}:force_original_aspect_ratio=decrease,showinfo"
    )
    result = _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-n",
        "-protocol_whitelist", "file,pipe", "-threads", "1", "-ss", f"{start:.9f}",
        "-f", "mov", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-filter_threads", "1", "-vf", filters, "-fps_mode", "vfr", "-q:v", "3",
        "-frames:v", str(math.ceil(duration * settings.frame_fps) + 1), str(directory / "%04d.jpg"),
    ])
    times = [float(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*\d+\s+pts_time:([\d.eE+-]+)", result.stderr)]
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) != len(times):
        raise ValueError("Decoded frames could not be matched to media timestamps.")
    pairs = [(path, t) for path, t in zip(paths, times, strict=True) if 0 <= t < duration]
    return [FrameSample(round(t, 6), p.read_bytes()) for p, t in pairs], [p for p, _ in pairs]
