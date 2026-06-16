"""YuNet detect-only face engine (INTERFACES.md §6).

A weak-machine fallback that satisfies `FaceEngine` without any recognition: it runs OpenCV's
`cv2.FaceDetectorYN` (the bundled `brain/models/yunet.onnx`) and emits one `Detection` per face with
`person_id=-1`, `label="face"`, `kind="unknown"` — enough to drive the servo, no identity. PORT of
`Recognizer._recognize_yunet` / `_init_yunet`, adapted to return `Detection` dataclasses.

`refresh()` is a no-op (there are no centroids to reload). `cv2` is imported at module top (it is a
hard, cheap dependency everywhere in the brain); the ONNX model is only opened in `__init__`, so
importing this module stays cheap and offline-safe.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from kitchenvision.core.types import Detection

# Canonical working resolution (the Pi feed is 640x480; the pipeline resizes to this).
W, H = 640, 480

# Bundled detector model: brain/models/yunet.onnx (copied from the old repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../kitchenvision/recognition
_BRAIN = os.path.dirname(os.path.dirname(_HERE))                   # .../brain
MODEL_PATH = os.path.join(_BRAIN, "models", "yunet.onnx").replace("\\", "/")


def _clamp_box(x1: float, y1: float, x2: float, y2: float) -> list[int]:
    """Clamp a float `[x1,y1,x2,y2]` box to the frame; return `[x, y, w, h]` ints."""
    x1 = int(max(0, min(W, round(float(x1)))))
    y1 = int(max(0, min(H, round(float(y1)))))
    x2 = int(max(0, min(W, round(float(x2)))))
    y2 = int(max(0, min(H, round(float(y2)))))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return [x1, y1, w, h]


class YuNet:
    """Detection-only `FaceEngine`: boxes + scores, no identity (`person_id=-1`)."""

    def __init__(self, config: dict) -> None:
        self.config = config
        rec = (config.get("recognition", {}) or {})
        model_path = str(rec.get("yunet_path", MODEL_PATH))
        det_thresh = float(rec.get("det_thresh", 0.6))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        # Detection-only: box + score, no identity. Input size is also (re)set per frame in detect().
        self.det = cv2.FaceDetectorYN.create(
            model_path, "", (W, H),
            score_threshold=det_thresh, nms_threshold=0.3, top_k=50,
        )
        self.det.setInputSize((W, H))

    def recognize(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame; return one detect-only `Detection` each."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        try:
            self.det.setInputSize((int(w), int(h)))
            _, faces = self.det.detect(frame)
        except cv2.error:
            return []
        if faces is None:
            return []
        out: list[Detection] = []
        for row in faces:
            x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            box = _clamp_box(x, y, x + bw, y + bh)
            score = float(row[14]) if len(row) > 14 else 0.0
            out.append(
                Detection(
                    box=box,
                    person_id=-1,
                    label="face",
                    kind="unknown",
                    track_id=-1,
                    age=None,
                    sex=None,
                    score=score,
                    quality=0.0,
                )
            )
        return out

    def refresh(self) -> None:
        """No-op: a detect-only engine has no identity model to reload."""
        return None
