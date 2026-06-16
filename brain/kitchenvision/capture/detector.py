"""Fast face detector for TRACKING (INTERFACES.md §5 — capture seam).

WHY THIS EXISTS
---------------
The servo must follow a moving person in realtime, but the recognition engine (`InsightFaceGpu`,
detect+embed+genderage in one `app.get()`) only runs a few fps — far too slow to aim a camera.
Coupling the `Tracker` to it (the original design) made the camera re-aim only a few times/sec.

So tracking gets its OWN fast detector here: a lightweight `cv2.FaceDetectorYN` pass that produces
face boxes EVERY frame in the capture loop, independent of recognition. Recognition keeps running on
its own thread purely for IDENTITY (names/age/sex/timeline). This is the same "everything heavy is a
swappable source behind an interface" principle as `FeedSource` / `FaceEngine` (ARCHITECTURE §2, §5).

`FaceDetector.detect(frame) -> list[[x,y,w,h]]` returns plain boxes in full 640x480 coords. The
default `YuNetDetector` DOWNSCALES the frame to a small detect size (~320x240) and scales the boxes
back — the proven `track_client.py` trick — so it costs ~3-8 ms/frame on the CPU and never contends
with InsightFace on the GPU (OpenCV's DNN runs on the CPU). `make_detector` returns `None` when
disabled, so the pipeline transparently falls back to recognition-driven boxes (today's behaviour).

`cv2` and the ONNX model are loaded LAZILY in `YuNetDetector.__init__` (not at import), so importing
this module stays cheap and offline-safe (the `--selfcheck` import check never opens the model).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable

# Canonical working resolution (the Pi feed is 640x480; the FeedSource resizes to this). Boxes are
# returned in THIS coord space so the Tracker / overlay / annotation all share one frame.
W, H = 640, 480

# Bundled detector model: brain/models/yunet.onnx (shared with recognition/yunet.py).
_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../kitchenvision/capture
_BRAIN = os.path.dirname(os.path.dirname(_HERE))                   # .../brain
_DEFAULT_MODEL = os.path.join(_BRAIN, "models", "yunet.onnx").replace("\\", "/")


@runtime_checkable
class FaceDetector(Protocol):
    """A fast detect-only face source for the tracker: boxes per frame, no identity."""

    def detect(self, frame) -> list[list[int]]: ...   # [[x, y, w, h], ...] in 640x480 coords


class YuNetDetector:
    """`FaceDetector` over `cv2.FaceDetectorYN` (yunet.onnx), downscaled for speed.

    Construct with the loaded config (reads the optional `detector` block — §1). `detect()` resizes
    the frame to `det_size`x(det_size*H/W), runs YuNet, and scales each box back to 640x480.
    """

    def __init__(self, config: dict) -> None:
        import cv2  # lazy: keep module import cheap + offline-safe

        det = (config.get("detector") or {})
        rec = (config.get("recognition") or {})
        model_path = str(det.get("yunet_path") or rec.get("yunet_path") or _DEFAULT_MODEL)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet detector model not found: {model_path}")

        # Detect at a downscaled size for speed, keeping the 640x480 aspect ratio.
        self.det_w = max(64, int(det.get("det_size", 320)))
        self.det_h = max(48, int(round(self.det_w * H / W)))
        thresh = float(det.get("det_thresh", rec.get("det_thresh", 0.6)))

        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (self.det_w, self.det_h),
            score_threshold=thresh, nms_threshold=0.3, top_k=50,
        )
        self._det.setInputSize((self.det_w, self.det_h))
        # Scale factors from the detect frame back up to the working frame.
        self._sx = W / float(self.det_w)
        self._sy = H / float(self.det_h)

    def detect(self, frame) -> list[list[int]]:
        """Detect faces in a BGR 640x480 frame; return `[x, y, w, h]` boxes in working coords."""
        import cv2  # lazy: same cached module the tests monkeypatch

        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        small = cv2.resize(frame, (self.det_w, self.det_h))
        try:
            self._det.setInputSize((self.det_w, self.det_h))
            _, faces = self._det.detect(small)
        except cv2.error:
            return []
        if faces is None:
            return []

        out: list[list[int]] = []
        for row in faces:
            x = int(round(float(row[0]) * self._sx))
            y = int(round(float(row[1]) * self._sy))
            w = int(round(float(row[2]) * self._sx))
            h = int(round(float(row[3]) * self._sy))
            # Clamp into the frame and keep w/h >= 1.
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            w = max(1, min(W - x, w))
            h = max(1, min(H - y, h))
            out.append([x, y, w, h])
        return out


def make_detector(config: dict) -> Optional["FaceDetector"]:
    """Return the configured fast tracking detector, or `None` when disabled.

    `detector.enabled` false (or an unknown engine that fails to build) -> `None`, and the pipeline
    falls back to driving the tracker from the recognition engine's boxes (the original behaviour).
    Built lazily (opens the ONNX model), so call this from the capture thread, not at import.
    """
    det = (config or {}).get("detector") or {}
    if not det.get("enabled", True):
        return None
    engine = str(det.get("engine", "yunet"))
    if engine != "yunet":
        # Only YuNet exists today; anything else falls through to it rather than crashing tracking.
        print(f"[detector] unknown detector engine {engine!r}; using yunet", flush=True)
    return YuNetDetector(config)
