"""Perception daemon worker — owns the cadence thread (INTERFACES.md §0, §8).

`PerceptionWorker` runs on its own daemon thread. On a timer (`config["activity"]["cadence_seconds"]`)
it:

  * if the source's VisionModel is NOT available -> set `STATE.activity_status` to "disabled" and
    sleep (no observe call, no DB writes),
  * snapshot the most recent RAW frame (copied) + `STATE.people` under `STATE.lock`, then RELEASE
    the lock before any heavy work,
  * if no recognised people are present -> status "ok" (idle, healthy) and skip,
  * else call `source.observe(frame, present)` to get a list of `Event`s, and for each Event:
      - if `activity.save_thumbnails`, save a keyframe via `store.db.save_thumb` and set
        `ev.thumb_ref`,
      - `store.db.add_event(ev)`,
      - update `STATE.people[*]["current_activity"]` for that person_id in-place under `STATE.lock`.

It maps the typed VLM errors raised by `observe()` to `STATE.activity_status`:
  * `NoVisionError`        -> {"state":"no_vision_model", ...} and STOP calling (process life).
  * `RateLimitError`       -> {"state":"error", ...} + exponential backoff.
  * any other exception    -> {"state":"error", ...} and it keeps going.

The worker NEVER crashes the process — every failure is reflected only via `activity_status`.
It also calls `store.db.prune(retention_days)` roughly hourly so retention is maintained even when
the pipeline's daily prune thread is not the one running.

Ported in structure from the old `station/activity.py` `ActivityWorker`, but the network/VLM call
and prompt/parse now live behind the `PerceptionSource` (so this file is offline-safe and the
backend is swappable).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from kitchenvision.core.types import Event
from kitchenvision.perception.base import PerceptionSource
from kitchenvision.store import db
from kitchenvision.vlm.base import NoVisionError, RateLimitError

# Backoff bounds for transient errors (rate-limit / network). Exponential, capped.
_BACKOFF_BASE = 5.0
_BACKOFF_MAX = 120.0

# How often to run db.prune() from this worker (seconds) — "roughly hourly".
_PRUNE_INTERVAL = 3600.0

# JPEG quality for saved event thumbnails (keep them small).
_THUMB_JPEG_QUALITY = 80


class PerceptionWorker:
    """Periodic perception captioner (one daemon thread)."""

    def __init__(self, config: dict, state, source: PerceptionSource) -> None:
        self.config = config
        self.state = state
        self.source = source

        act = config.get("activity", {}) or {}
        self.enabled = bool(act.get("enabled", True))
        self.cadence = float(act.get("cadence_seconds", 10) or 10)
        self.save_thumbnails = bool(act.get("save_thumbnails", True))
        self.min_confidence = float(act.get("min_confidence", 0.0) or 0.0)
        self.retention_days = config.get("retention_days", 30)

        self._thread: "threading.Thread | None" = None
        # Set once a no-vision condition is seen: stop observing for the process' life.
        self._stopped_no_vision = False
        self._consec_errors = 0
        self._last_prune = 0.0

    # ---------------------------------------------------------------- public API
    def start(self) -> None:
        """Spawn the worker loop as a daemon thread; return immediately (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.run, name="perception-worker", daemon=True
        )
        self._thread.start()

    def run(self) -> None:
        """The worker loop. Sleeps `cadence` between iterations; never raises."""
        if not self.enabled:
            self._set_status(
                "disabled",
                "Perception is off (set activity.enabled = true to enable).",
            )
        while True:
            try:
                self._tick()
            except Exception as e:  # belt-and-braces: the loop must never die
                self._set_status("error", f"perception worker error: {e}")
            time.sleep(max(1.0, self.cadence))

    # ---------------------------------------------------------------- one cycle
    def _tick(self) -> None:
        """One scheduler iteration: maybe prune, then maybe observe one frame."""
        self._maybe_prune()

        if not self.enabled:
            return
        if self._stopped_no_vision:
            return

        # No vision backend available (no creds / model not loaded) -> disabled, no call.
        if not self._vision_available():
            self._set_status(
                "disabled",
                "No vision model available (check vlm backend / creds / model).",
            )
            return

        # Snapshot under the lock, then release it BEFORE any heavy work.
        frame, people = self._snapshot()
        present = self._present_people(people)
        if frame is None or not present:
            self._set_status("ok", "idle (no recognised people in frame)")
            return

        try:
            events = self.source.observe(frame, people)
        except NoVisionError as e:
            self._stopped_no_vision = True
            self._set_status(
                "no_vision_model",
                f"Vision model unavailable; perception paused. ({e})".strip(),
            )
            return
        except RateLimitError as e:
            self._set_status("error", f"Rate limited; backing off. ({e})".strip())
            self._backoff()
            return
        except Exception as e:
            self._set_status("error", f"observe failed: {e}")
            return

        if not events:
            self._set_status("ok", "idle (no activity captioned)")
            return

        applied = self._apply_events(events, frame)
        self._set_status("ok", f"captioned {applied} of {len(present)} people.")

    # ---------------------------------------------------------------- vision availability
    def _vision_available(self) -> bool:
        """Best-effort read of the source's VisionModel.available (defaults to True)."""
        try:
            vision = getattr(self.source, "vision", None)
            if vision is None:
                return True
            return bool(vision.available)
        except Exception:
            return False

    # ---------------------------------------------------------------- snapshot
    def _snapshot(self) -> "tuple[np.ndarray | None, list[dict]]":
        """Return (frame_copy_or_None, people_list_copy) taken under STATE.lock.

        Copies the raw frame and shallow-copies the people list (per-person dicts treated
        read-only here) so the lock is held only briefly.
        """
        with self.state.lock:
            raw = getattr(self.state, "latest_raw", None)
            frame = None if raw is None else np.array(raw, copy=True)
            people = list(getattr(self.state, "people", []) or [])
        return frame, people

    @staticmethod
    def _present_people(people: list[dict]) -> list[dict]:
        """Recognised, matchable people only (person_id >= 0 with a label + box)."""
        out: list[dict] = []
        for p in people or []:
            try:
                pid = int(p.get("person_id", -1))
            except (TypeError, ValueError):
                continue
            if pid < 0:
                continue
            if not p.get("label") or not p.get("box"):
                continue
            out.append(p)
        return out

    # ---------------------------------------------------------------- apply events
    def _apply_events(self, events: list[Event], frame: np.ndarray) -> int:
        """Persist each Event (optionally with a thumbnail) and update STATE.current_activity.

        Returns the number of distinct people whose `current_activity` we updated. Gates on
        `activity.min_confidence`. Each DB hiccup is swallowed so the others still land.
        """
        # The thumbnail (one keyframe for this cycle) is shared across the cycle's events.
        thumb_ref: "str | None" = None
        if self.save_thumbnails:
            thumb_ref = self._save_keyframe(frame)

        updates: "dict[int, str]" = {}  # person_id -> phrase, for the STATE update pass
        for ev in events:
            if not isinstance(ev, Event):
                continue
            if ev.confidence < self.min_confidence:
                continue
            if thumb_ref and not ev.thumb_ref:
                ev.thumb_ref = thumb_ref
            try:
                db.add_event(ev)
            except Exception:
                # A DB hiccup must not break captioning of the others.
                continue
            if ev.person_id is not None and ev.text:
                updates[int(ev.person_id)] = ev.text

        if updates:
            self._update_current_activity(updates)
        return len(updates)

    def _update_current_activity(self, updates: "dict[int, str]") -> None:
        """Re-find each person by id under STATE.lock and set `current_activity` in-place.

        The pipeline replaces `STATE.people` wholesale each cycle, so a person may have vanished
        or been replaced since the snapshot; we simply skip anyone no longer present.
        """
        with self.state.lock:
            for person in getattr(self.state, "people", []) or []:
                try:
                    pid = int(person.get("person_id", -1))
                except (TypeError, ValueError):
                    continue
                if pid in updates:
                    person["current_activity"] = updates[pid]

    @staticmethod
    def _save_keyframe(frame: np.ndarray) -> "str | None":
        """JPEG-encode the raw frame and store it as an event thumbnail; return its ref or None."""
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMB_JPEG_QUALITY]
            )
            if not ok:
                return None
            ref = db.save_thumb(buf.tobytes())
            return ref or None
        except Exception:
            return None

    # ---------------------------------------------------------------- backoff
    def _backoff(self) -> None:
        """Exponential backoff sleep for rate-limit / transient errors (capped)."""
        self._consec_errors += 1
        delay = min(_BACKOFF_MAX, _BACKOFF_BASE * (2 ** (self._consec_errors - 1)))
        time.sleep(delay)

    # ---------------------------------------------------------------- status
    def _set_status(self, state: str, message: str) -> None:
        """Write STATE.activity_status under the lock; reset the error streak on 'ok'."""
        if state == "ok":
            self._consec_errors = 0
        try:
            with self.state.lock:
                self.state.activity_status = {"state": state, "message": message}
        except Exception:
            try:
                self.state.activity_status = {"state": state, "message": message}
            except Exception:
                pass

    # ---------------------------------------------------------------- prune
    def _maybe_prune(self) -> None:
        """Call db.prune(retention_days) roughly hourly (best-effort)."""
        now = time.time()
        if now - self._last_prune < _PRUNE_INTERVAL:
            return
        self._last_prune = now
        try:
            db.prune(self.retention_days)
        except Exception:
            pass
