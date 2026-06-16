"""ServoTransport default impl — `UdpServo` (INTERFACES.md §7).

The `Tracker` (tracking/tracker.py) computes a target angle vector + decides whether it is
`correcting`; this module SHIPS that to the Pi head over the two fixed UDP channels (INTERFACES
§13 / ARCHITECTURE §3 wire protocol):

  * UDP servo_udp (:9999)  ← target PAN angle as an ASCII float, e.g. b"96.5". The Pi re-clamps
    to [10, 170] and slews its servo smoothly toward this target at 100 Hz. Only `pan` is wired
    now (back-compat scalar protocol); the angle dict is N-axis so tilt is a second channel later.
  * UDP overlay_udp (:9998) ← the box the Pi should draw on its own `/stream` feed. Either the
    literal b"none", or EXACTLY five comma fields "x,y,w,h,correcting" (correcting = 1 while the
    tracker is actively recentring, else 0); the Pi splits on 5 fields and draws ONE box.

Both sends are PORTED verbatim from the old `station/tracker.py` (`send_angle` / `send_overlay`),
with the socket + throttle moved here out of the Tracker per the new contract. One UDP socket is
reused for both channels. The angle send is throttled to `SEND_INTERVAL` so it is cheap to call
every frame (smooth sweep + responsive target updates); the overlay send is not throttled (the Pi
box has a 0.4 s TTL so it must be refreshed promptly). A dropped UDP packet (`OSError`) is
swallowed — it must never break the capture/track pipeline.

THREADING
---------
Constructed and driven from the single capture/track thread (the only caller of `send_*`). The
socket is not shared across threads.
"""
from __future__ import annotations

import socket
import time
from typing import Optional

# Min seconds between angle UDP sends (~33/s): smooth sweep + responsive target updates to the Pi.
SEND_INTERVAL = 0.03


class NullServo:
    """Monitor-only `ServoTransport`: accepts the same calls but transmits NOTHING.

    Selected when `config["servo_enabled"]` is false. The Tracker still runs and the dashboard still
    shows the framing intent (HUD/markers), but no UDP angle/overlay packet is ever sent, so the
    physical servo is provably inert — for an unsupervised dashboard, or any rig whose servo must not
    move. Satisfies the `ServoTransport` Protocol in `tracking/base.py`.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    def send_angles(self, angles: dict) -> None:
        return None

    def send_overlay(self, box: Optional[tuple], correcting: bool) -> None:
        return None


class UdpServo:
    """Default `ServoTransport`: UDP angle + overlay to the Pi head (INTERFACES §7).

    Satisfies the `ServoTransport` Protocol in `tracking/base.py`
    (`send_angles(dict)` / `send_overlay(box, correcting)`).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.pi_ip: str = config.get("pi_ip", "192.168.68.127")
        self.servo_udp: int = int(config.get("servo_udp", 9999))
        self.overlay_udp: int = int(config.get("overlay_udp", 9998))

        # One reused UDP socket for both the angle and the overlay sends.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Angle-send throttle clock.
        self.last_sent: float = 0.0

    # --------------------------------------------------------------------- angles
    def send_angles(self, angles: dict) -> None:
        """UDP-send the target PAN angle as ASCII text to (pi_ip, servo_udp), e.g. b"96.5".

        `angles` is the Tracker's N-axis vector, e.g. {"pan": 96.5, ...}; only `pan` is wired to
        the head now (back-compat scalar protocol). Throttled to `SEND_INTERVAL` so it is cheap to
        call every frame; the Pi re-clamps to [10, 170] and slews smoothly toward this target. A
        missing `pan` key or a dropped packet (`OSError`) is swallowed — it must never break the
        pipeline.
        """
        pan = angles.get("pan")
        if pan is None:
            return
        now = time.time()
        if now - self.last_sent < SEND_INTERVAL:
            return
        self.last_sent = now
        try:
            self.sock.sendto(f"{float(pan):.1f}".encode(), (self.pi_ip, self.servo_udp))
        except OSError:
            pass  # a dropped packet must never break the pipeline

    # -------------------------------------------------------------------- overlay
    def send_overlay(self, box: Optional[tuple], correcting: bool) -> None:
        """Tell the Pi which box to draw on its own feed (pi_agent.py overlay protocol).

        box       -- the UNION (x, y, w, h) bounding box of all faces, or None.
        correcting -- True while the tracker is actively recentring (drawn as the 5th field).

          * a box  -> send EXACTLY five comma fields "x,y,w,h,correcting" (correcting = 1 while
            recentring else 0); the Pi splits on 5 fields and draws ONE box.
          * None   -> send the literal "none".

        A dropped overlay packet (`OSError`) is swallowed — it must never break the pipeline.
        """
        try:
            if box is None:
                self.sock.sendto(b"none", (self.pi_ip, self.overlay_udp))
            else:
                x, y, w, h = box
                c = 1 if correcting else 0
                msg = f"{int(x)},{int(y)},{int(w)},{int(h)},{c}"
                self.sock.sendto(msg.encode(), (self.pi_ip, self.overlay_udp))
        except OSError:
            pass  # a dropped overlay packet must never break the pipeline
