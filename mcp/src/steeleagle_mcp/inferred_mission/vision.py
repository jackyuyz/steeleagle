from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from importlib.resources import files

from openai import AsyncOpenAI, OpenAIError

from steeleagle_mcp.providers.openai import OpenAIProvider

from .config import Settings

PROMPT_VERSION = "v5"
TARGETS = (
    "person|bicycle|car|motorcycle|bus|truck|train|boat|airplane|bird|dog|cat|horse|"
    "sheep|cow|elephant|bear|zebra|giraffe|bench|umbrella|backpack|suitcase|kite|"
    "frisbee|skateboard|surfboard|skis|snowboard"
)
_POSITIVE = r"[1-9][0-9]{0,4}"
_RTH = r", then return home\."
_BATTERY_RTH = r" If the battery drops to 30 percent, return home\."
_TASKS = {
    "track": rf"Track the (?P<target>{TARGETS})" + _RTH + _BATTERY_RTH,
    "patrol": rf"Patrol the edge of the area at (?P<alt>{_POSITIVE}) meters" + _RTH + _BATTERY_RTH,
    "survey": rf"Survey the area in a lawnmower pattern at (?P<alt>{_POSITIVE}) meters" + _RTH + _BATTERY_RTH,
    "climb": rf"Climb to (?P<alt>{_POSITIVE}) meters, then wait (?P<seconds>{_POSITIVE}) seconds" + _RTH,
    "wait": rf"Wait (?P<seconds>{_POSITIVE}) seconds" + _RTH,
    "precision_land": (
        rf"Precision land on the (?P<target>{TARGETS})\."
        r" If the battery drops to 30 percent, descend and land at the current position\."
    ),
    "land": r"Descend and land at the current position\.",
}
_TAKE_OFF = re.compile(rf"Take off to (?P<takeoff>{_POSITIVE}) meters, then (?P<task>.+)")


@dataclass(frozen=True)
class FrameSample:
    time: float
    jpeg: bytes
    capture_time: float | None = None
    frame_id: int | None = None


def load_prompt() -> str:
    return files(__package__).joinpath("prompts", "v5.md").read_text(encoding="utf-8")


def parse_instruction(text: str) -> dict:
    """Recognize the seven v5 templates and preserve the fields for review."""
    instruction = " ".join(text.split())
    takeoff = _TAKE_OFF.fullmatch(instruction)
    if takeoff:
        instruction = takeoff["task"]
        instruction = instruction[0].upper() + instruction[1:]
    for kind, pattern in _TASKS.items():
        match = re.fullmatch(pattern, instruction)
        if not match or (takeoff and kind in {"land", "precision_land"}):
            continue
        fields = match.groupdict()
        if takeoff:
            fields["takeoff"] = takeoff["takeoff"]
        for key in fields.keys() - {"target"}:
            fields[key] = int(fields[key])
            if fields[key] % 5:
                raise ValueError("V5 altitude and duration fields must be multiples of five.")
        return {"task": kind, **fields}
    raise ValueError("The model reply does not match a v5 mission template.")


def select_history(records: list[dict], start: float, settings: Settings) -> list[dict]:
    if settings.history_mode == "no_memory" or not settings.history_sections:
        return []
    previous = [r for r in records if r["end"] <= start and r.get("inference_status") == "valid"]
    history = []
    for row in previous[-settings.history_sections:]:
        age = max(0.0, start - row["end"]) / settings.section_seconds
        weight = 2 ** (-age / settings.half_life_sections) if settings.history_mode == "recency" else 1.0
        history.append({"section": row["index"], "start": row["start"], "end": row["end"],
                        "weight": round(weight, 8), "text": row["inference_text"]})
    return history


def build_prompt(base_prompt: str, record: dict, history: list[dict], mode: str) -> str:
    context = {
        "section": record["index"], "duration_seconds": round(record["end"] - record["start"], 6),
        "history_mode": mode, "history": history, "telemetry": record.get("telemetry"),
    }
    return base_prompt.rstrip() + "\n\n## Runtime context\n" + (
        "Use the actual current-section duration and frame timestamps below instead of assuming 30 seconds. "
        "Only attached frames are current visual evidence. Infer the task at the end of this section. "
        "History contains fallible previous hypotheses, with relative reference priorities rather than probabilities. "
        "Clear current visual evidence takes priority. Repeated hypotheses are correlated. "
        "Battery and return-home phrases are conventions, not observations. Do not repeat an earlier takeoff.\n"
    ) + json.dumps(context, ensure_ascii=False)


async def infer_openai(*, model: str, prompt: str, frames: list[FrameSample],
                       api_key: str, base_url: str, timeout: float,
                       reasoning_effort: str | None = None) -> str:
    content = [{"type": "input_text", "text": prompt}]
    for frame in frames:
        content.append({"type": "input_text", "text": f"Frame at {frame.time:g} seconds into this section"})
        content.append({"type": "input_image", "detail": "high", "image_url": (
            "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")
        )})
    try:
        async with AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=0) as client:
            response = await client.responses.create(
                model=model, input=[{"role": "user", "content": content}],
                max_output_tokens=2048, store=False,
                **OpenAIProvider(reasoning_effort=reasoning_effort).responses_reasoning_options(model),
            )
    except OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        detail = f"HTTP {status}" if status else type(exc).__name__
        raise ValueError(f"Vision model request failed ({detail}). Check model access and connectivity.") from None
    if response.status != "completed" or not response.output_text.strip():
        raise ValueError("Vision model response was empty or incomplete.")
    return response.output_text
