"""Wires capture -> recognition -> tracking -> perception -> STATE; owns the threads (INTERFACES.md §10).

ROLE (see INTERFACES.md §0, §7, §8, §10)
----------------------------------------
`pipeline.py` is the heart of the brain (laptop) side. It owns the background threads and re-exports
the shared `STATE` singleton (which actually lives in `core/state.py` — imported here for caller
convenience per the contract). `start(config)`:

  1. `store.db.init_db()` — schema + data dirs.
  2. builds `engine = make_engine(cfg)`, `tracker = Tracker(cfg)`, `servo = make_servo(cfg)`.
  3. GUARDS the Phase-C pieces (`make_vision` / `make_source` / `PerceptionWorker`) in a
     `try/except` so Phase B (smooth recognised + tracked feed) runs even before the perception /
     VLM modules exist on disk. On failure it logs, marks perception "disabled" on
     `STATE.activity_status`, and simply skips the perception worker.
  4. spawns the RECOGNITION daemon thread — the ONLY caller of `engine.recognize()` / `engine.refresh()`.
  5. spawns the CAPTURE/TRACK daemon thread — owns the `FeedSource`, runs the `Tracker`, sends servo +
     overlay UDP every frame, annotates a display copy, and publishes `STATE.latest_jpeg/latest_raw/
     people/servo_angle/servo_angles/fps`. It NEVER blocks on ML.
  6. starts the `PerceptionWorker` (if it was constructed in step 3).
  7. spawns a periodic prune thread (`store.db.prune`).

`refresh_identities()` sets an Event the recognition thread consumes between passes so a rename /
merge / delete reloads the engine's centroids ON the recognition thread (never a request thread).
`union_box(boxes)` is the single union bounding box shipped to the Pi overlay.

WHY TWO THREADS (the performance fix — PORTED from station/pipeline.py)
----------------------------------------------------------------------
InsightFace recognition is the slow part (a few fps). If it ran INLINE in the capture loop the whole
loop — video, servo, overlay — would be throttled to that rate, so tracking would be jerky and
`/video` choppy. So recognition is DECOUPLED onto its own worker thread:

  * **capture/track thread** (fast, ~15-30 fps): reads each `Frame` from the `FeedSource`, hands the
    most-recent raw frame to the recognition worker (latest-wins slot), reads back the latest
    available detections, runs the Tracker + sends servo + overlay EVERY frame, and annotates +
    JPEG-encodes + publishes `/video` EVERY frame. It never blocks on recognition.
  * **recognition thread** (slow, its own pace): repeatedly grabs the MOST RECENT raw frame and runs
    `engine.recognize()` as fast as it can, publishing the resulting `Detection`s + a version counter.
    It is the ONLY thread that calls `recognize()` (which persists DB rows + crops + embeddings and is
    not safe to call concurrently) and the only thread that calls `engine.refresh()`.

A version-gated hand-off lets the tracker take a control STEP only on a FRESH detection
(`fresh=True`) — otherwise it would keep gliding on a stale box and overshoot to the travel limit.

IMPORTANT: `engine.recognize()` ALREADY persists sightings / crops / embeddings / people. The
pipeline does NOT duplicate any DB writes for detections; it only carries each person's
`current_activity` forward across cycles (the perception worker owns that string) and renders the
overlay.

THREADING
---------
Only the recognition thread touches `recognize()` / `refresh()`. The capture and recognition threads
exchange the latest raw frame + the latest detections through two tiny dedicated locks
(`_frame_lock`, `_detected_lock`); a `_frame_event` lets the recognition thread block instead of
busy-spinning while waiting for the next frame. Every read/write of a `STATE` field is wrapped in
`STATE.lock`. Heavy / optional libs (cv2, numpy) are imported LAZILY inside the loop bodies so this
module imports cheaply and offline (the smoke test loads no models, no GPU, no camera, no network).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from kitchenvision.core.state import STATE as STATE
from kitchenvision.core.types import Detection
from kitchenvision.store import db
from kitchenvision.capture.base import make_feed
from kitchenvision.recognition.base import make_engine
from kitchenvision.tracking.base import make_servo
from kitchenvision.tracking.tracker import Tracker

# Canonical full-frame geometry. The Pi feed is 640x480; the FeedSource resizes anything else to this
# so all downstream math (engine boxes, tracker hysteresis, overlay) shares one coord space.
W, H = 640, 480

# Guide-line geometry mirrored from track_client.py / pi_agent.py so the dashboard overlay matches
# what the Pi draws on its own feed.
CX = W // 2
OUTER_PX = 90   # match the tracker's OUTER_PX so the guide lines show the real pan-trigger band

# Target cap on the capture/track loop so we don't spin the CPU faster than the Pi feed / display can
# use (the feed is ~15-30 fps). A small cap leaves CPU headroom for the recognition thread. 0 = off.
_CAPTURE_FPS_CAP = 30.0
_MIN_FRAME_DT = 0.0 if _CAPTURE_FPS_CAP <= 0 else 1.0 / _CAPTURE_FPS_CAP

# How long the recognition thread waits for a fresh frame before re-checking (also bounds how promptly
# it notices a refresh_identities() request when the feed is idle).
_RECOG_FRAME_WAIT = 0.5

# Seconds between daily-prune runs (the perception worker may also prune; this is belt-and-braces).
_PRUNE_INTERVAL = 24 * 3600.0


# ---------------------------------------------------------------------------
# Module-level references kept so refresh_identities() (called from request threads) can signal the
# RECOGNITION thread to reload the engine's centroids between passes, plus the double-start guard.
# ---------------------------------------------------------------------------
_engine = None              # FaceEngine (built in start())
_tracker: Optional[Tracker] = None
_servo = None               # ServoTransport
_perception_worker = None   # PerceptionWorker | None (None if Phase-C unavailable)
_refresh_requested = threading.Event()   # set by refresh_identities(), consumed by the recog loop
_started = False
_started_lock = threading.Lock()

# --- capture <-> recognition hand-off (decoupled threads) ------------------
# The capture thread writes the most-recent raw frame here; the recognition thread reads it. Tiny
# dedicated locks (NOT STATE.lock) so neither thread is ever blocked by /video or the perception
# worker. `_frame_event` lets the recognition thread sleep until a frame is ready (no busy-spin).
_frame_lock = threading.Lock()
_latest_frame = None        # np.ndarray (BGR, 640x480) | None — newest captured raw frame
_frame_event = threading.Event()

# The recognition thread writes its latest detections here; the capture thread reads them every frame
# to drive the tracker + overlay. A version counter lets the tracker tell FRESH from stale.
_detected_lock = threading.Lock()
_latest_detected: list[Detection] = []   # newest engine.recognize() result
_detected_version = 0                     # bumped on each publish (FRESH vs stale gate)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def union_box(boxes):
    """Bounding box (x, y, w, h) enclosing every [x, y, w, h] in `boxes`, or None if empty.

    This single union box is exactly what `servo.send_overlay()` ships to the Pi (which only stores
    ONE box). Matches INTERFACES.md §10: (min x, min y, max(x+w)-min x, max(y+h)-min y). Pure Python
    (no numpy) so it is dependency-free for the offline smoke test.
    """
    if not boxes:
        return None
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]
    x = min(xs)
    y = min(ys)
    w = max(x2s) - x
    h = max(y2s) - y
    return (int(x), int(y), int(max(1, w)), int(max(1, h)))


def _det_box(d) -> list[int]:
    """Extract a sane [x, y, w, h] int box from a Detection (defensive against a bad box)."""
    box = getattr(d, "box", None) or [0, 0, 1, 1]
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def _merge_people(detected: list[Detection], prev_people: list[dict]) -> list[dict]:
    """Return the new STATE.people list: each `Detection` mapped to the §2 person dict, enriched with
    the `current_activity` carried forward from the previous cycle (matched by `person_id`), plus a
    normalised `last_seen_ts`.

    The pipeline OWNS every field except `current_activity`, which the perception worker writes — so
    we preserve it for a `person_id` already present and default it to "" for newly-seen people.
    `STATE.people` is replaced wholesale each cycle (the worker re-finds its person by id and
    tolerates it vanishing). PORTED from station/pipeline._merge_people, adapted to read `Detection`
    dataclass attributes (the engine returns dataclasses, not dicts).
    """
    prev_activity: dict[int, str] = {}
    for p in prev_people or []:
        try:
            pid = int(p.get("person_id", -1))
        except (TypeError, ValueError):
            continue
        act = p.get("current_activity")
        if act:
            prev_activity[pid] = act

    now = time.time()
    out: list[dict] = []
    for d in detected:
        try:
            pid = int(getattr(d, "person_id", -1))
        except (TypeError, ValueError):
            pid = -1
        out.append({
            "person_id": pid,
            "label": getattr(d, "label", "") or "",
            "kind": getattr(d, "kind", "unknown") or "unknown",
            "box": _det_box(d),
            "track_id": int(getattr(d, "track_id", -1) or -1),
            "age": getattr(d, "age", None),
            "sex": getattr(d, "sex", None),
            "current_activity": prev_activity.get(pid, ""),
            "last_seen_ts": now,
        })
    return out


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two [x, y, w, h] boxes (0.0 if they don't overlap)."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_labels(track_boxes: list[list[int]], people: list[dict]) -> list[dict]:
    """Build a DISPLAY people list on the realtime `track_boxes`, borrowing identity (label / kind /
    activity / age / sex) from the nearest RECOGNISED person by IoU.

    Tracking runs on the fast detector (frame rate) while recognition (identity) runs slower on its
    own thread, so the two box sets differ slightly. For the dashboard we draw the FAST boxes (so the
    rectangle tracks in realtime) and ride each recognised name onto its overlapping fast box. A fast
    box with no recognised match yet is shown as a plain detect-only box (no label). Returns §2-shaped
    person dicts so `_draw_annotations` consumes them unchanged."""
    now = time.time()
    out: list[dict] = []
    for tb in track_boxes:
        box = [int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])]
        best = None
        best_iou = 0.0
        for pp in people:
            i = _iou(box, pp.get("box", [0, 0, 1, 1]))
            if i > best_iou:
                best_iou, best = i, pp
        if best is not None and best_iou >= 0.2:
            out.append({**best, "box": box})
        else:
            out.append({
                "person_id": -1, "label": "", "kind": "unknown", "box": box,
                "track_id": -1, "age": None, "sex": None,
                "current_activity": "", "last_seen_ts": now,
            })
    return out


def _draw_annotations(cv2, frame, people, angle, fps, correcting, searching=False):
    """Draw the dashboard overlay onto `frame` IN PLACE (PORTED from station/pipeline._draw_annotations).

    Per person: the bounding box (orange while recentring, else green for known / cyan for unknown),
    the label above the box, and the `current_activity` beneath it. Plus the static guides (centre
    line + OUTER_PX pan-trigger lines), the group-centre marker (the tracker's control target), and a
    HUD line (angle / faces / fps / state) — matching track_client.py / pi_agent.py so the dashboard
    view reads the same as the Pi's own feed. `cv2` is passed in (imported lazily by the caller).
    """
    h, w = frame.shape[:2]

    # Static guide lines (same colours as track_client.py): pan-trigger edges + centre.
    cv2.line(frame, (CX - OUTER_PX, 0), (CX - OUTER_PX, h), (60, 60, 200), 1)
    cv2.line(frame, (CX + OUTER_PX, 0), (CX + OUTER_PX, h), (60, 60, 200), 1)
    cv2.line(frame, (CX, 0), (CX, h), (255, 0, 0), 1)

    boxes = [p["box"] for p in people]

    for p in people:
        x, y, bw, bh = (int(v) for v in p["box"][:4])
        x2, y2 = x + bw, y + bh
        # Orange while recentring; green for known; cyan-ish for unknown / detect-only.
        if correcting:
            col = (0, 165, 255)
        elif p.get("kind") == "known":
            col = (0, 255, 0)
        else:
            col = (0, 220, 220)
        cv2.rectangle(frame, (x, y), (x2, y2), col, 2)

        # Label above the box (drop a dark plate behind it for legibility).
        label = str(p.get("label") or "")
        if label:
            ty = y - 6 if y - 6 > 12 else min(h - 4, y2 + 16)
            tx = max(0, min(x, w - 1))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (tx + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # Activity beneath the box, if we have one yet.
        act = str(p.get("current_activity") or "")
        if act:
            ay = min(h - 4, y2 + 18)
            ax = max(0, min(x, w - 1))
            (aw, ah), _ = cv2.getTextSize(act, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (ax, ay - ah - 4), (ax + aw + 4, ay + 2), (0, 0, 0), -1)
            cv2.putText(frame, act, (ax + 2, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    # Group-centre marker = midpoint of leftmost/rightmost face centres (the tracker's control
    # target), drawn as a vertical line so the framing intent is visible.
    if boxes:
        centres = [b[0] + b[2] / 2.0 for b in boxes]
        gcx = int(round((min(centres) + max(centres)) / 2.0))
        gcol = (0, 165, 255) if correcting else (0, 255, 0)
        cv2.line(frame, (gcx, 0), (gcx, h), gcol, 1)

    # HUD line (top-left), same font/colour as the Pi/track_client HUD.
    statetxt = "SEARCH" if searching else ("RECENTER" if correcting else "hold")
    cv2.putText(
        frame,
        f"angle={angle:.0f} faces={len(boxes)} fps={fps:.1f} {statetxt}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    return frame


def _consume_refresh() -> None:
    """If a refresh was requested by the web layer, reload the engine's centroids now (on the
    RECOGNITION thread, the only one that touches the engine). Cheap; safe between every pass."""
    if _refresh_requested.is_set():
        _refresh_requested.clear()
        try:
            if _engine is not None:
                _engine.refresh()
        except Exception as e:
            print(f"[pipeline] refresh_identities failed: {e}", flush=True)


def _publish_frame(frame) -> None:
    """Capture thread: hand the newest raw frame to the recognition worker (latest wins).

    The slot write + the event set happen UNDER the lock so the slot and the event flag are always
    consistent (no lost wakeup, no spurious wakeup with an empty slot)."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame
        _frame_event.set()


def _take_frame(timeout: float):
    """Recognition thread: block (up to `timeout` s) for, then atomically take, the most recent raw
    frame. Returns the frame or None if none arrived in time. Always consumes the latest available
    frame (drops any the engine was too slow to keep up with).

    The wait is OUTSIDE the lock (so the capture thread can publish while we sleep); once woken we
    re-check the slot under the lock, since the event could be observed set just as it is toggled."""
    global _latest_frame
    if not _frame_event.wait(timeout):
        return None
    with _frame_lock:
        frame = _latest_frame
        _latest_frame = None
        _frame_event.clear()
    return frame


def _publish_detected(detected: list[Detection]) -> None:
    """Recognition thread: publish the latest detections for the capture loop to read. Bumps a
    version counter so the tracker only takes a control STEP on a FRESH detection."""
    global _latest_detected, _detected_version
    with _detected_lock:
        _latest_detected = detected
        _detected_version += 1


def _get_detected() -> tuple[list[Detection], int]:
    """Capture thread: read the latest detections + their version (shared-ref read)."""
    with _detected_lock:
        return _latest_detected, _detected_version


# ---------------------------------------------------------------------------
# recognition loop (its own daemon thread — the only one that calls recognize()/refresh())
# ---------------------------------------------------------------------------
def _recognition_loop() -> None:
    """Recognition daemon thread body. Repeatedly takes the MOST RECENT captured frame and runs the
    face engine on it as fast as it can, publishing the result for the capture thread. Decoupling
    this from capture is what lets the servo + video run at full fps while recognition plods along at
    its own (few-fps) rate.

    This is the ONLY thread that touches `_engine.recognize()` (which persists DB rows and is not
    safe to call concurrently) and `_engine.refresh()`. Never returns; a recognize() exception is
    logged and the previous detections are left in place.
    """
    print("[pipeline] recognition loop running (decoupled from capture)", flush=True)
    while True:
        # Pick up any rename/merge/delete-triggered centroid reload between passes (on THIS thread,
        # the only one allowed to touch the engine).
        _consume_refresh()

        frame = _take_frame(_RECOG_FRAME_WAIT)
        if frame is None:
            continue  # no new frame yet (feed idle / capture not started) — re-check refresh

        try:
            detected = _engine.recognize(frame)
        except Exception as e:
            # Recognition must never kill the loop; keep the previous published detections.
            print(f"[pipeline] recognize() error: {e}", flush=True)
            continue

        _publish_detected(detected)


# ---------------------------------------------------------------------------
# capture / track loop (fast daemon thread: full-fps video + servo)
# ---------------------------------------------------------------------------
def _capture_loop(config: dict) -> None:
    """The capture/track daemon thread body. Never returns (the FeedSource reconnects on failure).

    Owns the `FeedSource` (built via `make_feed`, tagged with the live servo angle through
    `angle_provider`). Per `Frame` it hands the raw BGR frame to the recognition worker, reads the
    latest available detections, drives the tracker + sends servo / overlay UDP, and publishes the
    raw frame + annotated JPEG to STATE. It NEVER blocks on recognition, so video + servo stay smooth
    regardless of the face engine. cv2 is imported LAZILY here (kept out of module import).
    """
    import cv2  # lazy: keep module import cheap + offline-safe

    # Webcam image-quality optimisation for the DISPLAY copy (Phase E). The recognition thread
    # enhances independently for its own ingest; here we clean up only what the dashboard shows. The
    # RAW frame published to STATE.latest_raw (perception + crops) is never enhanced.
    try:
        from kitchenvision.capture.enhance import make_enhancer
        enhancer = make_enhancer(config)
    except Exception as e:
        print(f"[pipeline] display enhancer unavailable: {e}", flush=True)
        enhancer = None

    # Fast TRACKING detector (decoupled from recognition): produces face boxes EVERY frame so the
    # servo follows in realtime instead of waiting on the slow recognition pass. Built here on the
    # capture thread (opens the ONNX model). If unavailable, tracking transparently falls back to the
    # recognition engine's boxes (the original, slower lock-step behaviour).
    try:
        from kitchenvision.capture.detector import make_detector
        detector = make_detector(config)
    except Exception as e:
        print(f"[pipeline] fast tracking detector unavailable ({e}); using recognition boxes", flush=True)
        detector = None
    if detector is not None:
        print("[pipeline] fast tracking detector enabled (decoupled from recognition)", flush=True)

    # angle_provider gives each captured Frame the servo angle known at capture (ego-motion aware).
    feed = make_feed(config, lambda: dict(STATE.servo_angles))
    feed.open()

    frames = 0
    fps = 0.0
    t0 = time.time()
    last_loop = 0.0           # for the optional frame-rate cap
    last_servo_version = -1   # detection version the tracker last STEPPED on (lock-step gate)

    print(f"[pipeline] capture/track loop running (cap {_CAPTURE_FPS_CAP:.0f} fps)", flush=True)

    while True:
        frame_obj = feed.read()
        if frame_obj is None:
            # Transient miss (the FeedSource handles its own reconnect after enough misses). Brief
            # nap so we don't busy-spin on a momentarily dead feed.
            time.sleep(0.02)
            continue

        # The RAW (unannotated) frame: the recognition worker + perception worker both use it.
        raw = frame_obj.bgr

        # Hand the newest frame to the recognition worker (latest wins; non-blocking), and read back
        # whatever detections it has published so far. The first few frames before recognition
        # produces anything just have an empty box set (servo holds / sweeps from 90 deg).
        _publish_frame(raw)
        detected, det_version = _get_detected()

        # TRACKING boxes come from the FAST detector (a genuinely fresh detection EVERY frame) when
        # one is configured, so the servo follows in realtime decoupled from the slow recognition
        # pass. Without a detector we fall back to the recognition engine's boxes under the original
        # LOCK-STEP gate (step only on a fresh recognition result, else glide on a stale box and
        # overshoot). Recognition still independently drives identity (STATE.people) below.
        if detector is not None:
            try:
                track_boxes = detector.detect(raw)
                track_fresh = True
            except Exception as e:
                print(f"[pipeline] detector error: {e}", flush=True)
                track_boxes = [_det_box(d) for d in detected]
                track_fresh = det_version != last_servo_version
        else:
            track_boxes = [_det_box(d) for d in detected]
            track_fresh = det_version != last_servo_version
        last_servo_version = det_version

        # Step the servo toward the group on a fresh detection; the search sweep (no faces) runs every
        # frame for smoothness. Sends go every frame (the transport throttles the angle send
        # internally) so the Pi gets a smooth sweep + prompt target updates.
        try:
            angles = _tracker.update(track_boxes, W, track_fresh)
            _servo.send_angles(angles)
            _servo.send_overlay(union_box(track_boxes) if track_boxes else None, _tracker.correcting)
        except Exception as e:
            print(f"[pipeline] tracker error: {e}", flush=True)
            angles = dict(STATE.servo_angles)
        pan = float(angles.get("pan", STATE.servo_angle))
        correcting = bool(getattr(_tracker, "correcting", False))
        searching = bool(getattr(_tracker, "searching", False))

        # FPS over a sliding 10-frame window (like track_client.py).
        frames += 1
        if frames % 10 == 0:
            now = time.time()
            dt = now - t0
            if dt > 0:
                fps = 10.0 / dt
            t0 = now

        # Publish RAW frame + people + angles + fps under the lock (briefly). We read back the
        # PREVIOUS people to carry each person's current_activity forward.
        with STATE.lock:
            prev_people = STATE.people
            people = _merge_people(detected, prev_people)
            STATE.latest_raw = raw.copy()
            STATE.people = people
            STATE.servo_angle = pan
            STATE.servo_angles = dict(angles)
            STATE.fps = float(fps)

        # Build the annotated DISPLAY copy from the SAME people we just published, encode to JPEG, and
        # publish it for GET /video. Enhance the display frame first (full webcam-cleanup chain) when
        # enabled; enhancer.apply() returns a fresh array, so the raw frame above is untouched.
        if enhancer is not None and getattr(enhancer, "for_display", False):
            try:
                display = enhancer.apply(raw, for_display=True)
            except Exception:
                display = raw.copy()
        else:
            display = raw.copy()
        # Draw the realtime FAST boxes (so the on-screen rectangle tracks live) with each recognised
        # name matched onto its overlapping box; fall back to the recognition people set when no fast
        # detector is active.
        display_people = _match_labels(track_boxes, people) if detector is not None else people
        _draw_annotations(cv2, display, display_people, pan, fps, correcting, searching)
        ok_enc, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok_enc:
            with STATE.lock:
                STATE.latest_jpeg = jpg.tobytes()

        # Optional frame-rate cap so we leave CPU headroom for the recognition thread and don't
        # busy-spin faster than the feed / display can use.
        if _MIN_FRAME_DT > 0.0:
            now = time.time()
            sleep_for = _MIN_FRAME_DT - (now - last_loop)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_loop = time.time()


# ---------------------------------------------------------------------------
# prune thread
# ---------------------------------------------------------------------------
def _prune_loop(config: dict) -> None:
    """Daily retention prune (best-effort). The perception worker may also prune ~hourly; this is the
    belt-and-braces daily pass the contract asks the pipeline to own."""
    retention_days = config.get("retention_days", 30)
    while True:
        time.sleep(_PRUNE_INTERVAL)
        try:
            sd, ed = db.prune(retention_days)
            if sd or ed:
                print(f"[pipeline] pruned {sd} sightings + {ed} events", flush=True)
        except Exception as e:
            print(f"[pipeline] prune failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Phase-C construction (guarded so Phase B runs without perception/vlm on disk)
# ---------------------------------------------------------------------------
def _build_perception(config: dict):
    """Construct the VisionModel + PerceptionSource + PerceptionWorker, returning the worker (or None).

    Per the task, the Phase-C pieces are GUARDED: if `make_vision` / `make_source` / the
    `PerceptionWorker` import is unavailable (the modules may not exist yet) or raises, we log, mark
    perception "disabled" on `STATE.activity_status`, and return None so the capture + recognition +
    tracking path (Phase B) still runs. Heavy / optional imports stay lazy + inside the guard.
    """
    try:
        from kitchenvision.vlm.base import make_vision
        from kitchenvision.perception.base import make_source
        from kitchenvision.perception.worker import PerceptionWorker

        vision = make_vision(config)
        source = make_source(config, vision)
        worker = PerceptionWorker(config, STATE, source)
        return worker
    except Exception as e:
        print(f"[pipeline] perception disabled (Phase-C unavailable): {e}", flush=True)
        with STATE.lock:
            STATE.activity_status = {
                "state": "disabled",
                "message": f"perception unavailable: {e}",
            }
        return None


# ---------------------------------------------------------------------------
# public API (INTERFACES.md §10)
# ---------------------------------------------------------------------------
def start(config: dict) -> None:
    """Wire up recognition + tracking + (optional) perception and start the background threads.

    Steps (per the contract §10):
      1. store.db.init_db()
      2. engine = make_engine(cfg); tracker = Tracker(cfg); servo = make_servo(cfg)
      3. GUARDED: vision = make_vision(cfg); source = make_source(cfg, vision);
         worker = PerceptionWorker(cfg, STATE, source)  — on failure: log + perception disabled.
      4. spawn the RECOGNITION daemon thread (latest frame -> recognize() -> shared detections).
      5. spawn the CAPTURE/TRACK daemon thread (FeedSource -> track + annotate at full fps).
      6. start the PerceptionWorker if it was constructed.
      7. spawn the periodic prune thread.

    Non-blocking: returns once the threads are launched (the entrypoint then runs uvicorn). Calling
    start() more than once is a no-op (guards against a double-start).
    """
    global _engine, _tracker, _servo, _perception_worker, _started

    with _started_lock:
        if _started:
            print("[pipeline] start() already called; ignoring.", flush=True)
            return
        _started = True

    # 1) Schema + data dirs.
    db.init_db()

    # 2) Engine + tracker + servo (built on THIS thread; recognize() runs on the recognition thread,
    #    the tracker + servo are driven from the capture/track thread).
    _engine = make_engine(config)
    _tracker = Tracker(config)
    _servo = make_servo(config)

    # 3) Phase-C perception pieces — guarded so Phase B runs without them.
    _perception_worker = _build_perception(config)

    # 4) Recognition daemon thread — the ONLY caller of recognize()/refresh(). Started before
    #    capture so it is ready to consume frames immediately.
    threading.Thread(
        target=_recognition_loop, name="pipeline-recognition", daemon=True
    ).start()

    # 5) Capture/track daemon thread (FeedSource -> track + annotate every frame).
    threading.Thread(
        target=_capture_loop, args=(config,), name="pipeline-capture", daemon=True
    ).start()

    # 6) Perception worker (its own daemon thread; shares STATE) — only if it was constructed.
    if _perception_worker is not None:
        try:
            _perception_worker.start()
        except Exception as e:
            print(f"[pipeline] perception worker start failed: {e}", flush=True)
            with STATE.lock:
                STATE.activity_status = {
                    "state": "disabled",
                    "message": f"perception worker failed to start: {e}",
                }

    # 7) Daily prune daemon thread.
    threading.Thread(
        target=_prune_loop, args=(config,), name="pipeline-prune", daemon=True
    ).start()

    print(
        "[pipeline] started (recognition + capture"
        + (" + perception" if _perception_worker is not None else "")
        + " + prune threads launched).",
        flush=True,
    )


def refresh_identities() -> None:
    """Ask the recognition thread to reload the engine's centroids (after a rename / merge / delete in
    the web layer) so subsequent recognize() calls reflect the change.

    Safe to call from any thread (e.g. a FastAPI request handler): it just sets an Event the
    recognition loop consumes between passes, so the actual `engine.refresh()` runs on the recognition
    thread (the only one allowed to touch the engine). The recognition loop wakes at least every
    `_RECOG_FRAME_WAIT` seconds even on an idle feed, so the reload lands promptly. If the pipeline
    hasn't started yet but an engine exists, we refresh inline (best-effort) so a very early rename
    still takes effect.
    """
    _refresh_requested.set()
    if not _started and _engine is not None:
        try:
            _engine.refresh()
            _refresh_requested.clear()
        except Exception:
            pass
