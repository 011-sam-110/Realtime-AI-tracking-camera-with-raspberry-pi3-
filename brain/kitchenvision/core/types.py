"""Shared cross-module types (INTERFACES.md §3).

`Detection` and `Event` are the ONLY result types that cross module boundaries — do not invent
parallel dict shapes for them. `Frame` is what a `FeedSource` yields each read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """A captured frame, tagged with the servo state known at capture (ego-motion aware)."""
    bgr: np.ndarray                       # working-resolution BGR (W,H = 640,480)
    ts: float                             # capture time, time.time()
    servo_angles: dict                    # {"pan": deg, ...} at capture
    seq: int                              # monotonic frame counter
    hires: Optional[np.ndarray] = None    # optional full-res frame for crops (12 MP future)


@dataclass
class Detection:
    """One recognised/detected face in full working-frame pixel coords."""
    box: list[int]                        # [x, y, w, h], clamped to the frame
    person_id: int                        # db id, or -1 for detect-only / unenrolled
    label: str                            # name, "Unknown #N", or "face"
    kind: str                             # "known" | "unknown"
    track_id: int = -1                    # stable across frames (identity persistence); -1 = none
    age: Optional[float] = None
    sex: Optional[str] = None             # "M" | "F" | None
    score: float = 0.0                    # match cosine similarity
    quality: float = 0.0                  # best-shot quality (blur/size/det fused)


@dataclass
class Event:
    """A structured activity / object / chore record (VLM now, local-CV later)."""
    type: str                             # "activity" | "object" | "chore" | "presence"
    ts: float
    person_id: Optional[int]              # subject; None = scene-level
    action: Optional[str] = None          # verb: "left", "cleared", "washing"
    object: Optional[str] = None          # noun: "plate", "mug"
    location: Optional[str] = None        # zone/surface: "table", "sink"
    text: str = ""                        # human caption
    confidence: float = 1.0
    source: str = "vlm"                   # producing PerceptionSource ("vlm" | "cv" | ...)
    thumb_ref: Optional[str] = None       # event-thumbnail key (store.save_thumb)
    payload: Optional[dict] = None        # raw extras
