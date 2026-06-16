"""PerceptionSource interface + factory (INTERFACES.md §8).

A PerceptionSource turns a raw frame + the present people into structured `Event`s. v1 impl is
`VlmCaptioner` (frame → VLM → per-person events). A later `LocalCvScene` (YOLO/open-vocab + tracker
+ zones + scene-diff) emits the SAME `Event` shape, so it slots in — or runs alongside — without
downstream change. The `PerceptionWorker` (perception/worker.py) owns the cadence thread.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from kitchenvision.core.types import Event
from kitchenvision.vlm.base import VisionModel


@runtime_checkable
class PerceptionSource(Protocol):
    def observe(self, frame: np.ndarray, people: list[dict]) -> list[Event]: ...


def make_source(config: dict, vision: VisionModel) -> "PerceptionSource":
    """Return the configured PerceptionSource (default VlmCaptioner). Built in Phase C."""
    from kitchenvision.perception.vlm_captioner import VlmCaptioner
    return VlmCaptioner(config, vision)
