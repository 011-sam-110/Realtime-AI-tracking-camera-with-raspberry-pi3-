"""Offline test for `kitchenvision.capture.detector` (INTERFACES.md §5 — fast tracking detector).

NO model load, NO GPU, NO camera: `cv2.FaceDetectorYN` is monkeypatched with a fake factory whose
`create()` returns a fake detector yielding a fixed faces array, so we can assert the box-scaling
math (detect at a downscaled size → boxes scaled back to 640×480) without touching the real ONNX.

Asserts:
  * `make_detector` returns `None` when `detector.enabled` is false (pipeline falls back),
  * `make_detector` returns a `YuNetDetector` by default,
  * `detect()` scales detect-size boxes back to working-frame (640×480) coords and clamps them,
  * `detect()` tolerates no-faces (`None`) and an empty frame → `[]`,
  * `YuNetDetector` satisfies the `FaceDetector` Protocol.

Plain-python runnable (pytest may be absent): `python tests/test_detector.py` from `brain/`.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

import cv2  # real module; we monkeypatch cv2.FaceDetectorYN (settable module attribute)

from kitchenvision.capture.detector import FaceDetector, YuNetDetector, make_detector, W, H

# A model path that EXISTS so YuNetDetector.__init__'s isfile() guard passes (the fake factory below
# ignores the path; we just need a real file).
_REAL_FILE = os.path.abspath(__file__)

# YuNet returns rows of [x, y, w, h, <10 landmark coords>, score]; the detector only reads [0:4].
def _face_row(x, y, w, h, score=0.9):
    return [float(x), float(y), float(w), float(h)] + [0.0] * 10 + [float(score)]


def _install_fake_yn(faces):
    """Replace cv2.FaceDetectorYN with a fake whose create() yields a detector returning `faces`."""
    original = cv2.FaceDetectorYN

    class _FakeYN:
        def __init__(self, faces):
            self._faces = faces

        def setInputSize(self, size):
            return None

        def detect(self, img):
            return 1, self._faces

    class _Factory:
        @staticmethod
        def create(model, cfg, size, score_threshold=0.6, nms_threshold=0.3, top_k=50):
            return _FakeYN(faces)

    cv2.FaceDetectorYN = _Factory
    return lambda: setattr(cv2, "FaceDetectorYN", original)


def _cfg(**det):
    base = {"detector": {"enabled": True, "engine": "yunet", "det_size": 320,
                         "det_thresh": 0.6, "yunet_path": _REAL_FILE}}
    base["detector"].update(det)
    return base


def test_make_detector_disabled_returns_none():
    assert make_detector({"detector": {"enabled": False}}) is None
    print("PASS test_make_detector_disabled_returns_none")


def test_make_detector_builds_yunet_and_scales_boxes():
    # One face at the centre of the 320x240 detect frame -> scaled back x2 to 640x480.
    faces = np.array([_face_row(160, 120, 32, 24)], dtype=np.float32)
    restore = _install_fake_yn(faces)
    try:
        det = make_detector(_cfg(det_size=320))
        assert isinstance(det, YuNetDetector)
        assert (det.det_w, det.det_h) == (320, 240), f"bad detect size {(det.det_w, det.det_h)}"

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        boxes = det.detect(frame)
        assert boxes == [[320, 240, 64, 48]], f"bad scaled box: {boxes!r}"
    finally:
        restore()
    print("PASS test_make_detector_builds_yunet_and_scales_boxes")


def test_detect_clamps_boxes_to_frame():
    # A face spilling past the right/bottom edge at detect-scale must clamp to the frame.
    faces = np.array([_face_row(310, 230, 40, 40)], dtype=np.float32)  # ->(620,460,80,80) then clamp
    restore = _install_fake_yn(faces)
    try:
        det = make_detector(_cfg(det_size=320))
        boxes = det.detect(np.zeros((H, W, 3), dtype=np.uint8))
        x, y, w, h = boxes[0]
        assert x == 620 and y == 460, f"bad origin {(x, y)}"
        assert x + w <= W and y + h <= H, f"box not clamped: {boxes[0]!r}"
        assert w >= 1 and h >= 1
    finally:
        restore()
    print("PASS test_detect_clamps_boxes_to_frame")


def test_detect_tolerates_no_faces_and_empty_frame():
    restore = _install_fake_yn(None)  # detector returns no faces
    try:
        det = make_detector(_cfg())
        assert det.detect(np.zeros((H, W, 3), dtype=np.uint8)) == []
        assert det.detect(None) == []
        assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    finally:
        restore()
    print("PASS test_detect_tolerates_no_faces_and_empty_frame")


def test_satisfies_facedetector_protocol():
    faces = np.array([_face_row(10, 10, 10, 10)], dtype=np.float32)
    restore = _install_fake_yn(faces)
    try:
        det = make_detector(_cfg())
        assert isinstance(det, FaceDetector), "YuNetDetector must satisfy the FaceDetector Protocol"
    finally:
        restore()
    print("PASS test_satisfies_facedetector_protocol")


def _run_all():
    test_make_detector_disabled_returns_none()
    test_make_detector_builds_yunet_and_scales_boxes()
    test_detect_clamps_boxes_to_frame()
    test_detect_tolerates_no_faces_and_empty_frame()
    test_satisfies_facedetector_protocol()
    print("\nALL DETECTOR TESTS PASSED")


if __name__ == "__main__":
    _run_all()
