"""Offline unit tests for `kitchenvision.pipeline` (INTERFACES.md §10).

Plain-python runnable (`python tests/test_pipeline.py`) since pytest may be absent. NO threads, NO
camera, NO GPU, NO network, NO model load — exercises only the pure-logic helpers + the public
contract surface:

  * `union_box` geometry (incl. None / empty).
  * `_merge_people` mapping `Detection` dataclasses -> §2 person dicts, carrying `current_activity`
    forward by `person_id`.
  * `_draw_annotations` runs against a real numpy frame (cv2) without raising.
  * the Phase-C guard (`_build_perception`) returns None + marks perception disabled when the
    perception/vlm impl modules are absent, so `start()` can run Phase B without them.
  * `refresh_identities()` is safe to call pre-start (no engine) and sets the consume Event.
  * `STATE` is the re-exported `core.state.STATE` singleton with the §2 fields.
"""
from __future__ import annotations

import os
import sys

# Make `import kitchenvision...` resolve when run as `python tests/test_pipeline.py` from brain/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kitchenvision.pipeline as p
from kitchenvision.core.state import STATE
from kitchenvision.core.types import Detection


def test_union_box():
    # Task's canonical example: min x=10,min y=10, max(x+w)=60, max(y+h)=50 -> (10,10,50,40).
    assert p.union_box([[10, 10, 20, 20], [50, 40, 10, 10]]) == (10, 10, 50, 40)
    # Single box is its own union.
    assert p.union_box([[5, 6, 7, 8]]) == (5, 6, 7, 8)
    # Empty / None -> None (servo overlay sends "none").
    assert p.union_box([]) is None
    assert p.union_box(None) is None
    # Degenerate zero-size box clamps w/h to >= 1.
    x, y, w, h = p.union_box([[3, 4, 0, 0]])
    assert (x, y) == (3, 4) and w >= 1 and h >= 1
    print("ok union_box")


def test_merge_people_from_detections():
    """The engine returns Detection DATACLASSES (not dicts); _merge_people must read attributes and
    carry current_activity forward by person_id."""
    detected = [
        Detection(box=[100, 50, 40, 40], person_id=7, label="Sam", kind="known",
                  track_id=3, age=31.0, sex="M", score=0.82, quality=0.9),
        Detection(box=[300, 60, 30, 30], person_id=-1, label="face", kind="unknown",
                  track_id=5),
    ]
    prev_people = [
        {"person_id": 7, "current_activity": "making tea"},   # should carry forward
        {"person_id": 9, "current_activity": "left the room"},  # person gone -> dropped
    ]
    people = p._merge_people(detected, prev_people)
    assert len(people) == 2

    sam = people[0]
    assert sam["person_id"] == 7
    assert sam["label"] == "Sam"
    assert sam["kind"] == "known"
    assert sam["box"] == [100, 50, 40, 40]
    assert sam["track_id"] == 3
    assert sam["age"] == 31.0
    assert sam["sex"] == "M"
    assert sam["current_activity"] == "making tea"   # carried forward by person_id
    assert isinstance(sam["last_seen_ts"], float)

    face = people[1]
    assert face["person_id"] == -1
    assert face["current_activity"] == ""            # newly seen -> empty
    assert face["box"] == [300, 60, 30, 30]

    # §2 shape: every required key present.
    required = {"person_id", "label", "kind", "box", "track_id", "age", "sex",
                "current_activity", "last_seen_ts"}
    assert required.issubset(sam.keys())

    # Empty inputs are tolerated.
    assert p._merge_people([], None) == []
    print("ok _merge_people")


def test_draw_annotations_runs():
    """The overlay must render onto a real frame without raising (uses cv2 lazily)."""
    import cv2  # available in this env; the pipeline imports it lazily inside the loop
    import numpy as np

    frame = np.zeros((p.H, p.W, 3), dtype=np.uint8)
    people = [
        {"box": [120, 80, 60, 60], "label": "Sam", "kind": "known",
         "current_activity": "washing up"},
        {"box": [400, 90, 40, 40], "label": "Unknown #1", "kind": "unknown",
         "current_activity": ""},
    ]
    out = p._draw_annotations(cv2, frame, people, angle=96.0, fps=24.3,
                              correcting=True, searching=False)
    assert out is frame                      # drawn in place, returns the same array
    assert frame.shape == (p.H, p.W, 3)
    assert int(frame.sum()) > 0              # something was actually drawn

    # No people + searching state: still renders the guides + HUD without raising.
    blank = np.zeros((p.H, p.W, 3), dtype=np.uint8)
    p._draw_annotations(cv2, blank, [], angle=12.0, fps=0.0, correcting=False, searching=True)
    print("ok _draw_annotations")


def test_phase_c_guard_builds_worker():
    """Phase C is now integrated: the perception/vlm impl modules exist on disk, so the GUARDED
    _build_perception() must construct make_vision -> make_source -> PerceptionWorker and return the
    worker (never raising). It must NOT start the thread (no observe / DB write here) — just build."""
    from kitchenvision.perception.worker import PerceptionWorker

    worker = p._build_perception({"vlm": {"backend": "local"}, "activity": {"enabled": True}})
    assert worker is not None, "Phase-C modules exist; _build_perception should return a worker"
    assert isinstance(worker, PerceptionWorker)
    # The worker owns a source whose VisionModel is the local backend (offline-safe construction).
    assert getattr(worker, "source", None) is not None
    assert getattr(worker.source, "vision", None) is not None
    print("ok _build_perception builds PerceptionWorker")


def test_refresh_identities_safe_pre_start():
    """Pre-start, with no engine, refresh_identities() just sets the consume Event (no crash)."""
    p._refresh_requested.clear()
    assert p._engine is None          # not started in these offline tests
    p.refresh_identities()
    assert p._refresh_requested.is_set()
    p._refresh_requested.clear()
    print("ok refresh_identities (pre-start)")


def test_state_reexport_and_shape():
    """`pipeline.STATE` is the re-exported core.state singleton with the §2 fields."""
    from kitchenvision.core.state import STATE as core_state
    assert p.STATE is core_state
    for field in ("lock", "latest_jpeg", "latest_raw", "people", "activity_status",
                  "servo_angle", "servo_angles", "fps"):
        assert hasattr(p.STATE, field), f"STATE missing {field}"
    assert isinstance(p.STATE.servo_angles, dict)
    print("ok STATE re-export + shape")


def test_public_api_surface():
    """The §10 public symbols exist with the right callability."""
    assert callable(p.start)
    assert callable(p.refresh_identities)
    assert callable(p.union_box)
    print("ok public API surface")


def test_iou_geometry():
    """`_iou` is the box-overlap metric used to ride recognised names onto realtime track boxes."""
    assert p._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0          # identical
    assert p._iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0      # disjoint
    # Half-overlap on x: inter=50, union=150 -> 1/3.
    assert abs(p._iou([0, 0, 10, 10], [5, 0, 10, 10]) - (1.0 / 3.0)) < 1e-6
    print("ok _iou geometry")


def test_match_labels_rides_names_onto_track_boxes():
    """`_match_labels` draws the realtime FAST boxes, borrowing the nearest recognised identity by
    IoU; a fast box with no overlapping recognition is a plain detect-only box."""
    people = [{
        "person_id": 7, "label": "Sam", "kind": "known", "box": [100, 100, 40, 40],
        "track_id": 3, "age": 30.0, "sex": "M", "current_activity": "tea", "last_seen_ts": 1.0,
    }]

    # A track box overlapping Sam borrows his identity but keeps the FAST box coords.
    out = p._match_labels([[102, 101, 38, 40]], people)
    assert len(out) == 1
    assert out[0]["label"] == "Sam" and out[0]["kind"] == "known"
    assert out[0]["current_activity"] == "tea" and out[0]["person_id"] == 7
    assert out[0]["box"] == [102, 101, 38, 40], "must draw the realtime box, not the recognition box"

    # A track box with no overlap -> detect-only (no label, person_id -1), still §2-shaped.
    out2 = p._match_labels([[500, 300, 30, 30]], people)
    assert out2[0]["label"] == "" and out2[0]["person_id"] == -1
    assert out2[0]["box"] == [500, 300, 30, 30]
    required = {"person_id", "label", "kind", "box", "track_id", "age", "sex",
                "current_activity", "last_seen_ts"}
    assert required.issubset(out2[0].keys())

    # No recognised people yet, or no track boxes.
    assert p._match_labels([[10, 10, 20, 20]], [])[0]["label"] == ""
    assert p._match_labels([], people) == []
    print("ok _match_labels")


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} pipeline tests passed.")


if __name__ == "__main__":
    _run_all()
