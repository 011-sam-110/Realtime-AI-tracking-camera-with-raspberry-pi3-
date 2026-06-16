"""ServoTransport interface + factory (INTERFACES.md §7).

The Tracker computes a target angle vector; the ServoTransport ships it to the head. Default impl
`UdpServo` speaks the fixed Pi protocol (UDP angle + overlay box). `Tracker` itself lives in
`tracking/tracker.py`.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ServoTransport(Protocol):
    def send_angles(self, angles: dict) -> None: ...                          # {"pan": deg, ...}; throttled
    def send_overlay(self, box: Optional[tuple], correcting: bool) -> None: ...  # union box or None


def make_servo(config: dict) -> "ServoTransport":
    """Return the configured ServoTransport.

    `config["servo_enabled"]` false → `NullServo` (monitor-only: tracker runs, nothing is sent, the
    servo provably cannot move). Otherwise the default `UdpServo` (Phase B).
    """
    if not (config or {}).get("servo_enabled", True):
        from kitchenvision.tracking.servo_transport import NullServo
        return NullServo(config)
    from kitchenvision.tracking.servo_transport import UdpServo
    return UdpServo(config)
