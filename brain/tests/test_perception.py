"""Offline tests for the perception layer (INTERFACES.md §8).

Covers:
  * `VlmCaptioner.observe()` with a STUB `VisionModel` (available=True, `generate` returns canned
    JSON for the present labels) -> yields one Event per matched person, with light keyword tags.
  * the typed-error pass-through (NoVisionError / RateLimitError re-raised) and the swallow-others
    behaviour.
  * `PerceptionWorker._tick()` applies those Events to a TEMPORARY `store.db`
    (`db.add_event`/`db.get_timeline`) and updates a fake STATE's `people[*].current_activity`.
  * the disabled path: a stub VisionModel with `available=False` -> status "disabled", no calls.

NO real VLM, NO network, NO pipeline.start, NO camera. The DB is redirected to a tempfile dir the
same way `tests/test_db.py` does it.

Plain-python runnable (pytest may be absent):  python tests/test_perception.py
Run from the brain root (C:/Users/sampo/pi/brain) so `import kitchenvision...` resolves.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

# Make `import kitchenvision...` work when run as a bare script from the brain root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.core.types import Event  # noqa: E402
from kitchenvision.perception.vlm_captioner import VlmCaptioner  # noqa: E402
from kitchenvision.perception.worker import PerceptionWorker  # noqa: E402
from kitchenvision.store import db  # noqa: E402
from kitchenvision.vlm.base import (  # noqa: E402
    NoVisionError,
    RateLimitError,
    VlmResult,
)


# --------------------------------------------------------------------------- temp-db harness
def _point_db_at_tmp(tmp: str) -> None:
    """Repoint store.db path constants into `tmp` and (re)create a fresh schema."""
    db.DATA_DIR = tmp.replace("\\", "/")
    db.DB_PATH = os.path.join(tmp, "kitchenvision.db").replace("\\", "/")
    db.FACES_DIR = os.path.join(tmp, "faces").replace("\\", "/")
    db.THUMBS_DIR = os.path.join(tmp, "thumbs").replace("\\", "/")
    db.init_db()


# --------------------------------------------------------------------------- stubs / fakes
class StubVision:
    """A canned VisionModel: returns JSON {label: phrase} for the present labels.

    `mode`:
      * "ok"        -> a JSON object mapping each known label to a phrase (with keywords),
      * "no_vision" -> raises NoVisionError,
      * "rate"      -> raises RateLimitError,
      * "boom"      -> raises a generic exception,
      * "garbage"   -> returns non-JSON text (no mapping).
    """

    def __init__(self, available: bool = True, mode: str = "ok") -> None:
        self._available = available
        self.mode = mode
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, image_bgr, prompt, max_tokens: int = 300) -> VlmResult:
        self.calls += 1
        if self.mode == "no_vision":
            raise NoVisionError("upstream has no vision model")
        if self.mode == "rate":
            raise RateLimitError("429 slow down")
        if self.mode == "boom":
            raise RuntimeError("transient network blip")
        if self.mode == "garbage":
            return VlmResult(text="sorry, I cannot answer that", provider="stub")
        # "ok": build a canned mapping by parsing the labels out of the prompt-ish — simpler to
        # just emit a fixed object keyed by the labels we know the test uses. We wrap it in some
        # chatter + a code fence to exercise the robust extraction path.
        body = (
            '{"Sam": "washing plates at the sink", '
            '"Unknown #2": "looking at phone"}'
        )
        return VlmResult(text=f"Here you go:\n```json\n{body}\n```", provider="stub")


class FakeState:
    """Minimal STATE stand-in: a lock + latest_raw + people + activity_status."""

    def __init__(self, frame, people) -> None:
        self.lock = threading.Lock()
        self.latest_raw = frame
        self.people = people
        self.activity_status = {"state": "ok", "message": ""}


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _people() -> list[dict]:
    return [
        {
            "person_id": 1, "label": "Sam", "kind": "known",
            "box": [100, 80, 120, 200], "track_id": 7, "current_activity": "",
            "last_seen_ts": time.time(),
        },
        {
            "person_id": 2, "label": "Unknown #2", "kind": "unknown",
            "box": [380, 90, 110, 190], "track_id": 8, "current_activity": "",
            "last_seen_ts": time.time(),
        },
        # detect-only / unenrolled: must be ignored by observe + worker.
        {
            "person_id": -1, "label": "face", "kind": "unknown",
            "box": [10, 10, 40, 40], "track_id": -1, "current_activity": "",
            "last_seen_ts": time.time(),
        },
    ]


def _config() -> dict:
    return {
        "retention_days": 30,
        "vlm": {"backend": "local", "max_tokens": 64},
        "activity": {
            "enabled": True, "cadence_seconds": 10,
            "save_thumbnails": True, "min_confidence": 0.0,
        },
    }


# --------------------------------------------------------------------------- tests: captioner
def test_observe_yields_events() -> None:
    cap = VlmCaptioner(_config(), StubVision(mode="ok"))
    events = cap.observe(_frame(), _people())
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    by_pid = {e.person_id: e for e in events}
    assert set(by_pid) == {1, 2}, by_pid.keys()
    for e in events:
        assert isinstance(e, Event)
        assert e.type == "activity"
        assert e.source == "vlm"
        assert e.ts > 0
        assert e.text
    assert by_pid[1].text == "washing plates at the sink"
    assert by_pid[2].text == "looking at phone"
    print("  observe yields one Event per matched recognised person")


def test_observe_light_keyword_extraction() -> None:
    cap = VlmCaptioner(_config(), StubVision(mode="ok"))
    events = cap.observe(_frame(), _people())
    by_pid = {e.person_id: e for e in events}
    # "washing plates at the sink"
    assert by_pid[1].action == "washing", by_pid[1].action
    assert by_pid[1].object == "plates", by_pid[1].object
    assert by_pid[1].location == "sink", by_pid[1].location
    # "looking at phone"
    assert by_pid[2].action in ("looking", "using phone"), by_pid[2].action
    assert by_pid[2].object == "phone", by_pid[2].object
    print("  light keyword extraction tags action/object/location")


def test_observe_empty_when_nobody_present() -> None:
    cap = VlmCaptioner(_config(), StubVision(mode="ok"))
    only_detect_only = [
        {"person_id": -1, "label": "face", "box": [1, 2, 3, 4]},
    ]
    assert cap.observe(_frame(), only_detect_only) == []
    assert cap.observe(None, _people()) == []
    assert cap.observe(_frame(), []) == []
    print("  observe returns [] when no recognised people / no frame")


def test_observe_reraises_typed_errors() -> None:
    cap_nv = VlmCaptioner(_config(), StubVision(mode="no_vision"))
    raised = False
    try:
        cap_nv.observe(_frame(), _people())
    except NoVisionError:
        raised = True
    assert raised, "NoVisionError must propagate out of observe()"

    cap_rl = VlmCaptioner(_config(), StubVision(mode="rate"))
    raised = False
    try:
        cap_rl.observe(_frame(), _people())
    except RateLimitError:
        raised = True
    assert raised, "RateLimitError must propagate out of observe()"
    print("  observe re-raises NoVisionError / RateLimitError")


def test_observe_swallows_other_errors_and_garbage() -> None:
    assert VlmCaptioner(_config(), StubVision(mode="boom")).observe(_frame(), _people()) == []
    assert VlmCaptioner(_config(), StubVision(mode="garbage")).observe(_frame(), _people()) == []
    print("  observe returns [] on generic errors and unparseable text")


# --------------------------------------------------------------------------- tests: worker
def test_worker_tick_persists_and_updates_state() -> None:
    cfg = _config()
    people = _people()
    state = FakeState(_frame(), people)
    source = VlmCaptioner(cfg, StubVision(mode="ok"))
    worker = PerceptionWorker(cfg, state, source)

    worker._tick()

    # Events landed in the temp DB and are retrievable via get_timeline.
    tl1 = db.get_timeline(1)
    tl2 = db.get_timeline(2)
    assert len(tl1) == 1, tl1
    assert len(tl2) == 1, tl2
    assert tl1[0]["text"] == "washing plates at the sink"
    assert tl1[0]["type"] == "activity"
    assert tl1[0]["source"] == "vlm"
    # Thumbnail saved + referenced (save_thumbnails=True).
    assert tl1[0]["thumb_ref"], "expected a thumb_ref to be saved"
    assert db.thumb_path(tl1[0]["thumb_ref"]) is not None

    # STATE.people current_activity updated in-place by person_id (detect-only ignored).
    by_pid = {p["person_id"]: p for p in state.people}
    assert by_pid[1]["current_activity"] == "washing plates at the sink"
    assert by_pid[2]["current_activity"] == "looking at phone"
    assert by_pid[-1]["current_activity"] == ""
    assert state.activity_status["state"] == "ok"
    print("  worker._tick persists events + thumbs and updates STATE.current_activity")


def test_worker_idle_when_nobody_present() -> None:
    cfg = _config()
    state = FakeState(_frame(), [{"person_id": -1, "label": "face", "box": [1, 2, 3, 4]}])
    stub = StubVision(mode="ok")
    worker = PerceptionWorker(cfg, state, VlmCaptioner(cfg, stub))
    worker._tick()
    assert state.activity_status["state"] == "ok"
    assert stub.calls == 0, "no VLM call should happen when nobody is recognised"
    assert db.get_events() == []
    print("  worker reports ok/idle and makes no call when nobody present")


def test_worker_disabled_when_vision_unavailable() -> None:
    cfg = _config()
    state = FakeState(_frame(), _people())
    stub = StubVision(available=False, mode="ok")
    worker = PerceptionWorker(cfg, state, VlmCaptioner(cfg, stub))
    worker._tick()
    assert state.activity_status["state"] == "disabled", state.activity_status
    assert stub.calls == 0, "no VLM call should happen when vision is unavailable"
    assert db.get_events() == []
    print("  worker reports disabled and makes no call when vision.available is False")


def test_worker_disabled_when_config_disabled() -> None:
    cfg = _config()
    cfg["activity"]["enabled"] = False
    state = FakeState(_frame(), _people())
    stub = StubVision(mode="ok")
    worker = PerceptionWorker(cfg, state, VlmCaptioner(cfg, stub))
    worker._tick()
    assert stub.calls == 0
    assert db.get_events() == []
    print("  worker makes no call when activity.enabled is False")


def test_worker_no_vision_error_stops_calling() -> None:
    cfg = _config()
    state = FakeState(_frame(), _people())
    stub = StubVision(mode="no_vision")
    worker = PerceptionWorker(cfg, state, VlmCaptioner(cfg, stub))
    worker._tick()
    assert state.activity_status["state"] == "no_vision_model", state.activity_status
    assert stub.calls == 1
    # A second tick must NOT call again (permanently stopped).
    worker._tick()
    assert stub.calls == 1, "worker must stop calling after NoVisionError"
    print("  worker maps NoVisionError -> no_vision_model and stops calling")


def test_worker_min_confidence_gate() -> None:
    cfg = _config()
    cfg["activity"]["min_confidence"] = 2.0  # above the captioner's 1.0 -> gate everything out
    state = FakeState(_frame(), _people())
    worker = PerceptionWorker(cfg, state, VlmCaptioner(cfg, StubVision(mode="ok")))
    worker._tick()
    assert db.get_events() == [], "events below min_confidence must not be persisted"
    print("  worker gates events below activity.min_confidence")


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        test_observe_yields_events,
        test_observe_light_keyword_extraction,
        test_observe_empty_when_nobody_present,
        test_observe_reraises_typed_errors,
        test_observe_swallows_other_errors_and_garbage,
        test_worker_tick_persists_and_updates_state,
        test_worker_idle_when_nobody_present,
        test_worker_disabled_when_vision_unavailable,
        test_worker_disabled_when_config_disabled,
        test_worker_no_vision_error_stops_calling,
        test_worker_min_confidence_gate,
    ]
    failed = 0
    for t in tests:
        tmp = tempfile.mkdtemp(prefix="kv_perc_test_")
        _point_db_at_tmp(tmp)
        try:
            print(f"{t.__name__} ...")
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} tests FAILED")
    else:
        print(f"RESULT: all {len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
