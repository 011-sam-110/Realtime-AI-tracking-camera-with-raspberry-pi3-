"""FaceEngine interface + factory (INTERFACES.md §6).

`recognize()` turns a raw BGR frame into `Detection`s (detect + embed + match/cluster, quality-gated,
best-shot fused, with a stable `track_id`). Only the recognition thread calls it. The default impl
is `InsightFaceGpu` (CUDA); `YuNet` is a detect-only fallback.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from kitchenvision.core.types import Detection


@runtime_checkable
class FaceEngine(Protocol):
    def recognize(self, frame: np.ndarray) -> list[Detection]: ...
    def refresh(self) -> None: ...      # reload centroids after rename/merge/delete


def make_engine(config: dict) -> "FaceEngine":
    """Select the face engine by config['recognition']['engine']. Built in Phase B."""
    engine = (config.get("recognition", {}) or {}).get("engine", "insightface")
    if engine == "yunet":
        from kitchenvision.recognition.yunet import YuNet
        return YuNet(config)
    from kitchenvision.recognition.insightface_gpu import InsightFaceGpu
    return InsightFaceGpu(config)
