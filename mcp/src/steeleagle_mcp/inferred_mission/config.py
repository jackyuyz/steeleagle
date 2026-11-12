from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from steeleagle_mcp.config import DEFAULT_OPENAI_MODEL

HISTORY_MODES = ("recency", "uniform", "no_memory")
SectionMode = Literal["non_overlapping", "sliding"]


@dataclass(frozen=True)
class Settings:
    vision_model: str = DEFAULT_OPENAI_MODEL
    vision_reasoning_effort: str | None = None
    section_seconds: float = 15.0
    frame_fps: float = 1.0
    max_frame_side: int = 768
    history_mode: str = "recency"
    history_sections: int = 10
    half_life_sections: float = 3.0
    request_timeout: float = 60.0
    generation_timeout: float = 120.0
    max_video_bytes: int = 1024**3
    max_video_seconds: float = 3600.0
    max_active_runs: int = 4
    section_mode: SectionMode = "non_overlapping"

    @property
    def section_stride(self) -> float:
        return 1.0 if self.section_mode == "sliding" else self.section_seconds

    def section_count(self, duration: float) -> int:
        if self.section_mode == "non_overlapping":
            return math.ceil(duration / self.section_seconds)
        return 1 + math.ceil(max(0, duration - self.section_seconds) / self.section_stride)

    def __post_init__(self):
        positive = (
            self.section_seconds, self.frame_fps, self.half_life_sections,
            self.request_timeout, self.generation_timeout, self.max_video_seconds,
        )
        if any(not math.isfinite(n) or n <= 0 for n in positive):
            raise ValueError("Inference durations and sampling rates must be finite and positive.")
        if self.section_seconds < 1 or not 1 <= self.section_seconds * self.frame_fps <= 60:
            raise ValueError("A section must be at least 1 second and sample between 1 and 60 frames.")
        if self.section_mode not in ("non_overlapping", "sliding"):
            raise ValueError("Choose a supported section mode.")
        if not 32 <= self.max_frame_side <= 1536 or not 0 <= self.history_sections <= 30:
            raise ValueError("Inference image size or history window is out of range.")
        if self.max_video_bytes < 1 or not 1 <= self.max_active_runs <= 16:
            raise ValueError("Inference upload size or run limit is out of range.")
        if self.history_mode not in HISTORY_MODES or not self.vision_model.strip():
            raise ValueError("Configure an inference model and a supported history mode.")


def load_settings(path: Path) -> Settings:
    if not path.is_file():
        return Settings()
    with path.open("rb") as config:
        return Settings(**tomllib.load(config).get("inferred_mission", {}))
