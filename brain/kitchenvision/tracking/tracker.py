"""Group-framing pan controller — `Tracker` (INTERFACES.md §7).

ROLE
----
`Tracker` turns the recognizer's face boxes into a servo target ANGLE vector. It GENERALISES
"follow the largest face" to "keep the whole GROUP in view": the control target is the MIDPOINT
of the face spread `group_x = (min cx + max cx) / 2`, so panning keeps the leftmost/rightmost
faces balanced about centre. The angle is returned as a dict `{"pan": angle}` (N-axis: pan now,
tilt-ready — tilt becomes an additive key later) and shipped to the Pi by a `ServoTransport`
(`UdpServo`); this class is pure control law and owns no socket.

LOCK-STEP CONTROL (the over-correction fix)  [ported from station/tracker.py]
-----------------------------------------------------------------------------
Recognition runs on its own thread at only a few fps, but the capture/track loop calls `update()`
~25x/sec. If the servo glided toward the error EVERY call, it would keep stepping the same
direction for the whole ~1 s a detection is stale and overshoot all the way to a travel limit. So
a tracking step is taken ONLY on a FRESH detection (`fresh=True`): we set a new TARGET angle (one
proportional step toward centre) and then HOLD it until the next fresh measurement. The Pi's own
100 Hz slew thread glides the servo smoothly to that target, so motion stays smooth while the
control loop stays in lock-step with what it can actually see — no latency-driven runaway. The
loop gain (`kp` x pixels-per-degree) is kept below 1 so each fresh step removes only a fraction of
the error and it converges without overshoot.

SEARCH SWEEP  [ported from station/tracker.py]
----------------------------------------------
When no face has been seen for `search_after` seconds the tracker enters SEARCH mode and pans
slowly back and forth across the full travel range (`sweep_speed` deg/s, time-based so it is
fps-independent), bouncing off the limits, until any face is detected — then it stops and tracks.
Even a blurry mid-pan face (returned as a detect-only box) halts the sweep so the camera settles
on the person and the next still frame can recognise them.

THREADING
---------
Constructed and driven from the single capture/track thread. `update()` returns the angle dict;
sending is the `ServoTransport`'s job. No cross-thread state — the pipeline reads the result and
publishes it to `STATE` under `STATE.lock`.
"""
from __future__ import annotations

import time

# Servo travel limits (degrees). The Pi independently clamps to the same range.
ANGLE_MIN, ANGLE_MAX = 10, 170

# ---- Module defaults (used only when a `track` config key is absent). These mirror the
# documented tunables; the authoritative defaults live in core/config.py `DEFAULTS["track"]`. ----

# Deadband / hysteresis (pixels from frame centre): the servo does NOTHING until the group centre
# drifts past OUTER_PX, then glides only until it is back within INNER_PX and HOLDS there. A WIDE
# hold band is what stops the hunting — the face can roam the middle of the frame before the camera
# moves at all, and once recentred it sits still until it genuinely drifts out again.
OUTER_PX = 90
INNER_PX = 50

# Proportional control applied ONCE PER FRESH DETECTION (lock-step). Loop gain = KP*(px/deg) is
# kept WELL under 1 so each step removes only PART of the error and the camera eases to centre
# instead of overshooting and hunting; MAX_STEP caps a single fresh step so a far face glides in
# over a few steps rather than one lurch.
KP = 0.04
MAX_STEP = 8.0
DIRECTION = -1          # flip to +1 if the camera pans the WRONG way
EMA = 0.3               # LIGHT smoothing on the fresh group centre (the deadband, not the EMA,
                        # rejects box jitter; heavy smoothing here just adds destabilising lag).

# Search sweep defaults.
SEARCH_AFTER = 2.5      # seconds with no face before the camera starts sweeping to find people
SWEEP_SPEED = 22.0      # sweep rate in deg/s (slow enough for recognition to catch faces)

# Anti-jitter (Phase F, refined for the realtime decoupled pipeline). With fresh fast-detector boxes
# and ~0 ms capture latency the servo can correct smoothly EVERY frame (the Pi's 100 Hz slew glides
# it) without the old latency-driven hunting, so:
#  * on ACQUIRING a face the camera settles at its current angle for HOLD_SECONDS — long enough for
#    the search sweep to stop and the frame to de-blur — then centres CONTINUOUSLY (not in per-step
#    freezes, which made acquisition lurch);
#  * a correction smaller than MIN_STEP degrees is ignored, so a centred face sits perfectly still.
MIN_STEP = 4.0          # ignore corrective steps smaller than this many degrees (anti micro-jitter)
HOLD_SECONDS = 2.0      # settle this long after ACQUIRING a face (not after every move)


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp scalar `v` into the inclusive range [lo, hi]."""
    return lo if v < lo else hi if v > hi else v


class Tracker:
    """Group-framing pan controller with lock-step tracking + a time-based search sweep."""

    def __init__(self, config: dict) -> None:
        self.config = config
        # Tunables come from the nested `track` block (INTERFACES §1), falling back to the module
        # defaults above so a partial / missing block still works. These "calm" knobs are tuned
        # against the real rig via config.json's `track` section without editing code.
        track: dict = config.get("track") or {}
        self.kp = float(track.get("kp", KP))
        self.max_step = float(track.get("max_step", MAX_STEP))
        self.direction = int(track.get("direction", DIRECTION))
        self.outer_px = float(track.get("outer_px", OUTER_PX))
        self.inner_px = float(track.get("inner_px", INNER_PX))
        self.ema = float(track.get("ema", EMA))
        self.search_after = float(track.get("search_after", SEARCH_AFTER))
        self.sweep_speed = float(track.get("sweep_speed", SWEEP_SPEED))
        self.min_step = float(track.get("min_step", MIN_STEP))
        self.hold_seconds = float(track.get("hold_seconds", HOLD_SECONDS))

        # Control state.
        self.angle: float = 90.0       # last commanded servo TARGET angle (degrees)
        self.correcting: bool = False  # True while actively recentring (hysteresis)
        self.searching: bool = False   # True while sweeping to find a face
        self.ema_x: float | None = None  # smoothed group centre x (None = no smoothing yet)
        self.sweep_dir: int = 1        # +1 / -1 current sweep direction
        now = time.time()
        self.last_face_t: float = now    # last time a face was in view (grace before sweeping)
        self.last_update_t: float = now  # for the time-based sweep step
        self.hold_until: float = 0.0     # dwell: freeze the angle until this time (anti-jitter)
        self.had_face: bool = False      # was a face in view last update (detect the acquire edge)

    # ------------------------------------------------------------------ control
    def update(self, boxes: list[list[int]], frame_w: int, fresh: bool = True) -> dict:
        """Compute the new servo TARGET angle and return it as `{"pan": angle}`.

        boxes    -- list of [x, y, w, h] face boxes in full-frame coords (may be stale).
        frame_w  -- full frame width in pixels (640).
        fresh    -- True only when `boxes` is a NEW detection (not a repeat of last frame's). A
                    tracking step is taken ONLY when fresh, so the servo never glides on a stale
                    box (that over-glide is what drove it to the travel limit). The sweep, having
                    no visual feedback, runs every call (time-based) for smooth motion.

        Side effects: updates `self.angle`, `self.correcting`, `self.searching`. Returns a dict so
        a tilt axis can be added later without changing the call site.
        """
        now = time.time()
        dt = now - self.last_update_t
        if dt < 0 or dt > 1.0:
            dt = 0.0   # ignore the first call / long stalls so the sweep can't jump
        self.last_update_t = now

        if boxes:
            # A face is in view: leave search mode and track the group.
            self.searching = False
            acquiring = not self.had_face   # rising edge: we just FOUND a face this update
            self.had_face = True
            self.last_face_t = now

            # COMMIT & HOLD (anti-jitter): the instant a face is acquired, freeze at the current
            # rotation for hold_seconds. This stops the search sweep dead and lets the now-still
            # camera deliver clean, sharp detections instead of chasing a blur.
            if acquiring:
                self.hold_until = now + self.hold_seconds
                self.ema_x = None
                self.correcting = False
                return {"pan": self.angle}

            # While dwelling (just acquired, or just moved) HOLD — ignore detections entirely so the
            # camera sits still instead of micro-correcting every frame.
            if now < self.hold_until:
                return {"pan": self.angle}

            if fresh:
                centres = [b[0] + b[2] / 2.0 for b in boxes]
                group_x = (min(centres) + max(centres)) / 2.0
                if self.ema_x is None:
                    self.ema_x = group_x
                else:
                    self.ema_x = self.ema * self.ema_x + (1.0 - self.ema) * group_x

                err = self.ema_x - frame_w / 2.0
                if not self.correcting and abs(err) > self.outer_px:
                    self.correcting = True
                elif self.correcting and abs(err) <= self.inner_px:
                    self.correcting = False

                if self.correcting:
                    step = clamp(self.direction * self.kp * err, -self.max_step, self.max_step)
                    # SMOOTH CONTINUOUS CENTRING: with fresh fast-detector boxes + ~0 ms latency the
                    # controller corrects a little EVERY frame and the Pi's 100 Hz slew glides the
                    # servo in — proportional `kp*err` eases off as the face nears centre, so it
                    # decelerates into the deadband instead of the old lurch-pause-lurch (which came
                    # from re-arming a per-step freeze after every move). The outer/inner deadband +
                    # EMA reject jitter; `min_step` ignores sub-threshold nudges so a centred face
                    # sits still. (No per-step `hold_until` here — the settle is acquire-only.)
                    if abs(step) >= self.min_step:
                        self.angle = clamp(self.angle + step, ANGLE_MIN, ANGLE_MAX)
            # On a stale (non-fresh) frame we HOLD the existing target — the Pi keeps slewing to
            # it — instead of gliding further on out-of-date pixels.
            return {"pan": self.angle}

        # ---- no faces ----
        self.had_face = False
        self.ema_x = None
        self.correcting = False
        if now - self.last_face_t < self.search_after:
            # Brief grace: the face may just have flickered out — hold position.
            self.searching = False
            return {"pan": self.angle}

        # Search: sweep slowly across the travel range to find people, bouncing off limits.
        self.searching = True
        self.angle += self.sweep_speed * dt * self.sweep_dir
        if self.angle >= ANGLE_MAX:
            self.angle = ANGLE_MAX
            self.sweep_dir = -1
        elif self.angle <= ANGLE_MIN:
            self.angle = ANGLE_MIN
            self.sweep_dir = 1
        return {"pan": self.angle}
