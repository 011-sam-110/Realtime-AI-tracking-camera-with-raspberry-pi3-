"""Offline tests for the pan `Tracker` control law + `UdpServo` transport (INTERFACES.md §7).

NO HARDWARE: the Tracker is pure math and the UdpServo socket is never actually sent through a
real Pi (we swap its socket for a fake that captures the bytes), so this runs with no camera, no Pi, no
GPU, no network. Plain-python runnable: `python tests/test_tracker.py` (pytest optional).

What is asserted (the proven behaviours ported from station/tracker.py):
  1. HOLD: a face parked near frame centre (inside the deadband) never moves the servo and never
     enters `correcting`.
  2. CORRECT: a face past OUTER_PX flips `correcting` True.
  3. CONVERGE: repeated FRESH calls on an off-centre face march the angle monotonically in the
     recentring direction (bounded by max_step) toward the travel limit; a STALE (fresh=False)
     call HOLDS the angle (lock-step).
  4. SEARCH: `update([], 640)` starts sweeping once `search_after` has elapsed (driven by mutating
     `last_face_t`, no real sleeping needed) and bounces off the [10, 170] limits.
  5. UdpServo wire format: pan float to servo_udp; "x,y,w,h,correcting" / "none" to overlay_udp.
"""
from __future__ import annotations

import os
import sys
import time

# Make `import kitchenvision...` resolve when run as `python tests/test_tracker.py` from brain/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchenvision.tracking.tracker import (  # noqa: E402
    ANGLE_MAX,
    ANGLE_MIN,
    Tracker,
)
from kitchenvision.tracking.servo_transport import UdpServo  # noqa: E402

W = 640  # canonical working width


def _cfg(**track):
    """A minimal config dict with an overridable `track` block (matches core/config layout)."""
    base = {
        "pi_ip": "127.0.0.1",
        "servo_udp": 9999,
        "overlay_udp": 9998,
        "track": {
            "outer_px": 90,
            "inner_px": 50,
            "kp": 0.04,
            "max_step": 8.0,
            "direction": -1,
            "ema": 0.3,
            "search_after": 2.5,
            "sweep_speed": 22.0,
            # Disable the Phase-F anti-jitter gates by default so the control-law tests exercise the
            # PURE ported law (the original station tracker had neither). The gates — a short settle
            # after a move (hold_seconds) and a minimum corrective step (min_step) — get their own
            # dedicated test below (test_acquire_hold_and_min_step).
            "min_step": 0.0,
            "hold_seconds": 0.0,
            "axes": ["pan"],
        },
    }
    base["track"].update(track)
    return base


def _box(cx: int, w: int = 80, h: int = 80, y: int = 200):
    """A face box whose horizontal centre is `cx` (so group_x is controllable)."""
    return [int(cx - w / 2), y, w, h]


# --------------------------------------------------------------------------- 1
def test_returns_pan_dict():
    """update() returns {"pan": float} so a tilt axis is additive later."""
    t = Tracker(_cfg())
    out = t.update([_box(W // 2)], W, fresh=True)
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    assert set(out) == {"pan"}, f"expected only 'pan', got {sorted(out)}"
    assert isinstance(out["pan"], float), f"pan must be float, got {type(out['pan'])}"
    print("ok  returns {'pan': float}")


# --------------------------------------------------------------------------- 2
def test_holds_within_deadband():
    """A face parked at frame centre (err=0, well inside OUTER_PX) never moves / never corrects."""
    t = Tracker(_cfg())
    start = t.angle
    for _ in range(40):
        out = t.update([_box(W // 2)], W, fresh=True)
    assert out["pan"] == start, f"angle drifted inside deadband: {start} -> {out['pan']}"
    assert t.correcting is False, "should not be correcting at dead centre"
    assert t.searching is False, "should not be searching while a face is visible"

    # And a face just INSIDE the outer band (err < outer_px) must also hold and not start correcting.
    t2 = Tracker(_cfg())
    inside_cx = W // 2 + 80          # err = +80 px < outer_px(90)
    for _ in range(20):
        out2 = t2.update([_box(inside_cx)], W, fresh=True)
    assert t2.correcting is False, "err inside OUTER_PX must not trip correcting"
    assert out2["pan"] == 90.0, f"held angle expected 90.0, got {out2['pan']}"
    print("ok  holds within deadband (centre and just-inside OUTER_PX)")


# --------------------------------------------------------------------------- 3
def test_enters_correcting_past_outer():
    """A face beyond OUTER_PX flips `correcting` True. The first detection ACQUIRES + settles (it
    never lurches on acquisition — see test_acquire_hold_and_min_step), so correction lands on the
    next fresh call (with the anti-jitter gates off in _cfg, that is immediate)."""
    t = Tracker(_cfg())
    far_cx = W // 2 + 120            # err = +120 px > outer_px(90)
    t.update([_box(far_cx)], W, fresh=True)         # acquire + settle (no lurch)
    out = t.update([_box(far_cx)], W, fresh=True)   # now corrects
    assert t.correcting is True, "should be correcting once past OUTER_PX"
    assert out["pan"] != 90.0, "a correcting step should have moved the angle off centre"
    print(f"ok  enters correcting past OUTER_PX (angle {out['pan']:.2f})")


def test_acquire_hold_and_min_step():
    """Phase-F anti-jitter (now config-tunable via `hold_seconds` / `min_step`):
      * ACQUIRING a face freezes at the current angle for `hold_seconds` — the first detection never
        lurches, and subsequent calls keep holding during the dwell;
      * a corrective step smaller than `min_step` degrees is ignored (no micro-jitter).
    """
    # Acquire + dwell: a far face must NOT move the servo while holding.
    t = Tracker(_cfg(hold_seconds=2.0, min_step=4.0))
    far_cx = W // 2 + 150
    out = t.update([_box(far_cx)], W, fresh=True)         # acquire
    assert t.correcting is False and out["pan"] == 90.0, "acquire must freeze, not lurch"
    out = t.update([_box(far_cx)], W, fresh=True)         # still within the 2 s dwell
    assert out["pan"] == 90.0, "must hold during the dwell after acquiring"

    # min_step gate: with the dwell off, a sub-min_step correction is still ignored. A face only just
    # past OUTER_PX gives kp*err ≈ 0.04*95 ≈ 3.8 deg < min_step(4.0) ⇒ no move.
    t2 = Tracker(_cfg(hold_seconds=0.0, min_step=4.0))
    near_cx = W // 2 + 95
    t2.update([_box(near_cx)], W, fresh=True)             # acquire
    out = t2.update([_box(near_cx)], W, fresh=True)       # would-be tiny step, gated out
    assert out["pan"] == 90.0, f"sub-min_step nudge must be ignored, got {out['pan']}"
    print("ok  acquire-hold + min_step anti-jitter gates")


# --------------------------------------------------------------------------- 4
def test_converges_toward_centre():
    """Repeated FRESH calls march the angle monotonically the recentring way, bounded by max_step.

    Geometry: a right-biased face (group_x > centre ⇒ err > 0) with direction=-1 gives a NEGATIVE
    step, so the target angle steps DOWN toward ANGLE_MIN each fresh call. (This is the open-loop
    behaviour — the boxes don't move in the sim — and is exactly the lock-step glide the Pi slews
    to.) Steps must never exceed max_step and must head toward the limit, not oscillate.
    """
    cfg = _cfg()
    t = Tracker(cfg)
    far_cx = W // 2 + 150            # strongly right of centre ⇒ err>0 ⇒ angle should DECREASE
    prev = t.update([_box(far_cx)], W, fresh=True)["pan"]
    angles = [prev]
    for _ in range(60):
        cur = t.update([_box(far_cx)], W, fresh=True)["pan"]
        # monotonic non-increasing (recentring direction) ...
        assert cur <= prev + 1e-9, f"angle moved the WRONG way: {prev} -> {cur}"
        # ... and each single step is bounded by max_step.
        assert abs(cur - prev) <= cfg["track"]["max_step"] + 1e-9, "step exceeded max_step"
        angles.append(cur)
        prev = cur
    assert angles[-1] < angles[0], "angle did not converge toward the recentring limit"
    assert abs(angles[-1] - ANGLE_MIN) < 1e-6, f"should glide to ANGLE_MIN, got {angles[-1]}"

    # Lock-step: a STALE (fresh=False) call must HOLD the angle (no glide on stale pixels).
    held = t.update([_box(W // 2)], W, fresh=False)["pan"]
    assert held == angles[-1], f"stale call changed the angle: {angles[-1]} -> {held}"
    print(f"ok  converges toward centre ({angles[0]:.2f} -> {angles[-1]:.2f}) + holds on stale")

    # Symmetry: a LEFT-biased face should drive the angle UP toward ANGLE_MAX.
    t2 = Tracker(_cfg())
    left_cx = W // 2 - 150           # err<0 ⇒ angle should INCREASE
    p = t2.update([_box(left_cx)], W, fresh=True)["pan"]
    for _ in range(60):
        p = t2.update([_box(left_cx)], W, fresh=True)["pan"]
    assert abs(p - ANGLE_MAX) < 1e-6, f"left face should glide to ANGLE_MAX, got {p}"
    print(f"ok  symmetric: left-biased face glides up to ANGLE_MAX ({p:.2f})")


# --------------------------------------------------------------------------- 5
def test_group_centre_is_spread_midpoint():
    """Target is midpoint(min cx, max cx), NOT the mean — a lopsided crowd still frames the spread."""
    t = Tracker(_cfg())
    # Three faces; mean is pulled right, but the spread-midpoint is exactly frame centre ⇒ HOLD.
    boxes = [_box(W // 2 - 100), _box(W // 2 + 100), _box(W // 2 + 90)]
    for _ in range(10):
        out = t.update(boxes, W, fresh=True)
    assert t.correcting is False, "spread midpoint is centred ⇒ must not correct"
    assert out["pan"] == 90.0, f"spread-midpoint centring expected hold at 90.0, got {out['pan']}"
    print("ok  group target uses spread midpoint(min cx, max cx)")


# --------------------------------------------------------------------------- 6
def test_search_sweep_after_timeout():
    """update([], 640) sweeps once search_after has elapsed; before that it HOLDS (grace)."""
    cfg = _cfg(sweep_speed=22.0, search_after=2.5)
    t = Tracker(cfg)
    base_angle = t.angle

    # Within the grace window: no movement, not searching.
    out = t.update([], W)
    assert t.searching is False, "must not search inside the grace window"
    assert out["pan"] == base_angle, "must hold position inside the grace window"

    # Drive time forward WITHOUT real sleeping: backdate last_face_t past search_after, and set
    # last_update_t so the loop sees a real (clamped) dt for the time-based sweep step.
    now = time.time()
    t.last_face_t = now - (cfg["track"]["search_after"] + 1.0)
    t.last_update_t = now - 0.1          # dt ~ 0.1 s this call
    out = t.update([], W)
    assert t.searching is True, "should be sweeping after search_after elapsed"
    expected = base_angle + cfg["track"]["sweep_speed"] * 0.1 * 1  # sweep_dir starts +1
    assert abs(out["pan"] - expected) < 1.0, f"sweep step off: {out['pan']} vs ~{expected}"
    print(f"ok  search sweep starts after search_after (angle {out['pan']:.2f})")


# --------------------------------------------------------------------------- 7
def test_sweep_bounces_off_limits():
    """The sweep saturates at ANGLE_MAX/MIN and reverses direction (bounces), never escaping."""
    cfg = _cfg(sweep_speed=300.0, search_after=0.0)   # fast sweep so a few steps hit the limit
    t = Tracker(cfg)
    t.last_face_t = time.time() - 10.0                # already past search_after
    hit_max = False
    for _ in range(30):
        t.last_update_t = time.time() - 0.2           # ~0.2 s dt per call
        out = t.update([], W)
        assert ANGLE_MIN <= out["pan"] <= ANGLE_MAX, f"angle escaped limits: {out['pan']}"
        if out["pan"] == ANGLE_MAX:
            hit_max = True
        if hit_max and t.sweep_dir == -1:
            break
    assert hit_max, "fast sweep should have reached ANGLE_MAX"
    assert t.sweep_dir == -1, "sweep must reverse direction after hitting ANGLE_MAX"

    # Keep going and confirm it comes back down to ANGLE_MIN and reverses again.
    hit_min = False
    for _ in range(60):
        t.last_update_t = time.time() - 0.2
        out = t.update([], W)
        assert ANGLE_MIN <= out["pan"] <= ANGLE_MAX, f"angle escaped limits: {out['pan']}"
        if out["pan"] == ANGLE_MIN:
            hit_min = True
            break
    assert hit_min, "sweep should travel back down to ANGLE_MIN"
    assert t.sweep_dir == 1, "sweep must reverse again after hitting ANGLE_MIN"
    print("ok  sweep bounces off both [10,170] limits and reverses")


# --------------------------------------------------------------------------- 8
def test_face_halts_sweep():
    """A detected box (even mid-sweep) leaves search mode and stops sweeping that same call."""
    cfg = _cfg(search_after=0.0)
    t = Tracker(cfg)
    t.last_face_t = time.time() - 10.0
    t.last_update_t = time.time() - 0.2
    t.update([], W)
    assert t.searching is True, "precondition: should be sweeping"
    t.update([_box(W // 2)], W, fresh=True)
    assert t.searching is False, "a visible face must halt the sweep"
    print("ok  a detected face halts the search sweep")


# --------------------------------------------------------------------------- 9
def test_udpservo_wire_format():
    """UdpServo emits the exact Pi wire bytes (captured via a fake socket — no real network)."""

    class _FakeSock:
        def __init__(self):
            self.sent: list[tuple[bytes, tuple]] = []

        def sendto(self, data, addr):
            self.sent.append((data, addr))

    servo = UdpServo(_cfg())
    fake = _FakeSock()
    servo.sock = fake

    # send_angles: pan as ASCII float "%.1f" to (pi_ip, servo_udp).
    servo.last_sent = 0.0
    servo.send_angles({"pan": 96.54, "tilt": 12.0})
    assert fake.sent, "send_angles produced no packet"
    data, addr = fake.sent[-1]
    assert data == b"96.5", f"pan wire bytes wrong: {data!r}"
    assert addr == ("127.0.0.1", 9999), f"servo addr wrong: {addr}"

    # Throttle: an immediate second call within SEND_INTERVAL is suppressed.
    n = len(fake.sent)
    servo.send_angles({"pan": 50.0})
    assert len(fake.sent) == n, "send_angles was not throttled within SEND_INTERVAL"

    # Missing pan key ⇒ no send.
    servo.last_sent = 0.0
    n = len(fake.sent)
    servo.send_angles({"tilt": 5.0})
    assert len(fake.sent) == n, "send_angles should no-op without a 'pan' key"

    # send_overlay box ⇒ exactly five comma fields "x,y,w,h,correcting" to overlay_udp.
    servo.send_overlay((10, 20, 30, 40), correcting=True)
    data, addr = fake.sent[-1]
    assert data == b"10,20,30,40,1", f"overlay box bytes wrong: {data!r}"
    assert addr == ("127.0.0.1", 9998), f"overlay addr wrong: {addr}"
    assert data.count(b",") == 4, "overlay must have exactly 5 fields"

    servo.send_overlay((1, 2, 3, 4), correcting=False)
    assert fake.sent[-1][0] == b"1,2,3,4,0", f"correcting=False should send ...,0: {fake.sent[-1][0]!r}"

    # send_overlay None ⇒ literal b"none".
    servo.send_overlay(None, correcting=False)
    assert fake.sent[-1][0] == b"none", f"None overlay should send b'none': {fake.sent[-1][0]!r}"
    print("ok  UdpServo wire format (angle float, 5-field overlay, none, throttle)")


# --------------------------------------------------------------------------- 10
def test_udpservo_swallows_oserror():
    """A socket OSError on either send is swallowed — a dropped packet never breaks the pipeline."""

    class _BoomSock:
        def sendto(self, data, addr):
            raise OSError("simulated network down")

    servo = UdpServo(_cfg())
    servo.sock = _BoomSock()
    servo.last_sent = 0.0
    servo.send_angles({"pan": 90.0})     # must not raise
    servo.send_overlay((1, 2, 3, 4), True)
    servo.send_overlay(None, False)
    print("ok  UdpServo swallows OSError on send")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
