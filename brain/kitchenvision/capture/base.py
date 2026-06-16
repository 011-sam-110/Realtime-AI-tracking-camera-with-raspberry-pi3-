"""FeedSource interface + factory (INTERFACES.md §5).

A FeedSource yields `Frame`s tagged with the servo angle known at capture, so all downstream logic
is ego-motion aware. The default impl is `MjpegFeed` (Pi `/raw`); a future 12 MP `CsiFeed`/`Rtsp`
is a drop-in selected here by config — no downstream change.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from kitchenvision.core.types import Frame


@runtime_checkable
class FeedSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[Frame]: ...      # next Frame (resized to W,H) or None on a transient miss
    def close(self) -> None: ...


def make_feed(config: dict, angle_provider: Callable[[], dict]) -> "FeedSource":
    """Return the configured FeedSource (default MjpegFeed). `angle_provider()` returns the current
    {"pan": deg, ...} so each Frame carries the servo state. Concrete impl built in Phase B."""
    from kitchenvision.capture.mjpeg import MjpegFeed
    return MjpegFeed(config, angle_provider)
