"""Offline test for `kitchenvision.capture.mjpeg.MjpegFeed` (INTERFACES.md §5).

Runs with NO model load, NO GPU, NO network, NO real camera: `cv2.VideoCapture` is monkeypatched
with a fake that reports `isOpened()=True` and `read() -> (True, <synthetic 480x640x3 uint8>)`.

`MjpegFeed` now runs a background GRABBER thread that drains the buffer and keeps only the newest
frame (the realtime fix), so the assertions are LATEST-WINS rather than read-every-frame:
  * `read()` returns a `Frame` whose `bgr.shape == (480, 640, 3)`,
  * `seq` is monotonically INCREASING across reads (stale frames are dropped, so it may skip),
  * `servo_angles` equals whatever `angle_provider()` returned (snapshotted per frame),
  * a non-640x480 frame is resized to the working resolution,
  * once the capture starts failing, the grabber transparently reconnects after `_RECONNECT_AFTER`
    misses (a second capture is constructed),
  * `MjpegFeed` satisfies the `FeedSource` Protocol.

Plain-python runnable (pytest may be absent): `python tests/test_mjpeg.py` from `brain/`.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Make `import kitchenvision...` resolve when run as `python tests/test_mjpeg.py` from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

import cv2  # real module; we monkeypatch its VideoCapture below (lazy `import cv2` in mjpeg.py
            # returns this same cached module object, so the patch is honoured).

from kitchenvision.capture.base import FeedSource
from kitchenvision.capture.mjpeg import MjpegFeed, W, H, _RECONNECT_AFTER
from kitchenvision.core.types import Frame


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: always 'open', yields a synthetic frame of `shape`.

    `fail_after` (None = never) makes `read()` start returning (False, None) after N good frames,
    to exercise the transient-miss + reconnect path. A tiny sleep paces the background grabber thread
    so it doesn't busy-spin the CPU during the test. `released` records cleanup.
    """

    instances: list["_FakeCapture"] = []

    def __init__(self, url, shape=(H, W, 3), fail_after=None):
        self.url = url
        self._shape = shape
        self._fail_after = fail_after
        self._reads = 0
        self.released = False
        _FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def set(self, *a, **k):
        return True

    def read(self):
        time.sleep(0.001)  # pace the grabber thread (~1 kHz max) so the test stays light + sane
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            return False, None
        # A distinct synthetic BGR frame each call (uint8), so resize/identity is observable.
        frame = np.full(self._shape, (self._reads % 256), dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


def _install_fake(monkey_shape=(H, W, 3), fail_after=None):
    """Patch cv2.VideoCapture to construct _FakeCapture; return a restore() callable."""
    _FakeCapture.instances.clear()
    original = cv2.VideoCapture

    def factory(url, *args, **kwargs):
        return _FakeCapture(url, shape=monkey_shape, fail_after=fail_after)

    cv2.VideoCapture = factory
    return lambda: setattr(cv2, "VideoCapture", original)


def _read_blocking(feed, tries=400):
    """Poll feed.read() until it returns a Frame (the grabber may not have produced one yet)."""
    for _ in range(tries):
        f = feed.read()
        if f is not None:
            return f
        time.sleep(0.002)
    raise AssertionError("read() never returned a Frame")


def test_read_returns_frame_with_servo_angles_and_seq():
    angles = {"pan": 123.4, "tilt": 5.0}
    restore = _install_fake(monkey_shape=(H, W, 3))
    try:
        feed = MjpegFeed(
            {"pi_ip": "10.0.0.9", "stream_port": 8000, "stream_path": "/raw"},
            angle_provider=lambda: angles,
        )
        feed.open()

        f1 = _read_blocking(feed)
        assert isinstance(f1, Frame), f"expected Frame, got {type(f1)!r}"
        assert f1.bgr.shape == (H, W, 3) == (480, 640, 3), f"bad shape {f1.bgr.shape}"
        assert f1.bgr.dtype == np.uint8, f"bad dtype {f1.bgr.dtype}"
        assert f1.hires is None
        assert isinstance(f1.ts, float) and f1.ts > 0
        # servo_angles equals what the provider returned (and is a *snapshot*, not the same obj).
        assert f1.servo_angles == angles, f"angles mismatch: {f1.servo_angles!r}"
        assert f1.servo_angles is not angles, "servo_angles should be a per-frame copy"
        assert f1.seq >= 1, f"seq should start at 1+, got {f1.seq}"

        # Latest-wins: seq is monotonically INCREASING across reads (stale frames dropped, may skip).
        f2 = _read_blocking(feed)
        f3 = _read_blocking(feed)
        assert f2.seq > f1.seq, f"seq did not increase: {f1.seq} -> {f2.seq}"
        assert f3.seq > f2.seq, f"seq did not increase: {f2.seq} -> {f3.seq}"

        feed.close()
        assert _FakeCapture.instances[0].released is True, "close() must release the capture"
    finally:
        restore()
    print("PASS test_read_returns_frame_with_servo_angles_and_seq")


def test_url_built_from_config():
    restore = _install_fake()
    try:
        feed = MjpegFeed(
            {"pi_ip": "192.168.5.5", "stream_port": 9001, "stream_path": "/stream"},
            angle_provider=dict,
        )
        feed.open()
        assert _FakeCapture.instances[0].url == "http://192.168.5.5:9001/stream", (
            f"bad URL {_FakeCapture.instances[0].url!r}"
        )
        feed.close()
    finally:
        restore()
    print("PASS test_url_built_from_config")


def test_oversize_frame_is_resized_to_working_res():
    # Fake yields a 720p frame; read() must resize it to (480, 640, 3).
    restore = _install_fake(monkey_shape=(720, 1280, 3))
    try:
        feed = MjpegFeed({}, angle_provider=lambda: {"pan": 90.0})
        feed.open()
        f = _read_blocking(feed)
        assert f.bgr.shape == (H, W, 3), f"frame not resized: {f.bgr.shape}"
        feed.close()
    finally:
        restore()
    print("PASS test_oversize_frame_is_resized_to_working_res")


def test_failing_capture_triggers_reconnect():
    # Good for 2 frames, then read() fails forever. The background grabber should transparently
    # reopen after _RECONNECT_AFTER misses (a second _FakeCapture is constructed).
    restore = _install_fake(monkey_shape=(H, W, 3), fail_after=2)
    try:
        feed = MjpegFeed({}, angle_provider=lambda: {"pan": 90.0})
        feed.open()
        assert len(_FakeCapture.instances) == 1

        # Poll up to ~3 s for the reconnect (30 misses * ~5 ms + fake pacing ~ a few hundred ms).
        deadline = time.time() + 3.0
        while time.time() < deadline and len(_FakeCapture.instances) < 2:
            time.sleep(0.02)
        assert len(_FakeCapture.instances) >= 2, (
            f"expected a reconnect after {_RECONNECT_AFTER} misses; "
            f"{len(_FakeCapture.instances)} captures built"
        )
        feed.close()
    finally:
        restore()
    print("PASS test_failing_capture_triggers_reconnect")


def test_satisfies_feedsource_protocol():
    feed = MjpegFeed({}, angle_provider=dict)
    assert isinstance(feed, FeedSource), "MjpegFeed must satisfy the FeedSource Protocol"
    print("PASS test_satisfies_feedsource_protocol")


def _run_all():
    test_read_returns_frame_with_servo_angles_and_seq()
    test_url_built_from_config()
    test_oversize_frame_is_resized_to_working_res()
    test_failing_capture_triggers_reconnect()
    test_satisfies_feedsource_protocol()
    print("\nALL MJPEG TESTS PASSED")


if __name__ == "__main__":
    _run_all()
