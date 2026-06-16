"""MJPEG FeedSource over the Pi's clean `/raw` stream (INTERFACES.md §5).

`MjpegFeed` wraps a `cv2.VideoCapture` opened on `http://{pi_ip}:{stream_port}{stream_path}`
(from config — §1). It is the v1 default `FeedSource`; a future 12 MP `CsiFeed`/`Rtsp` is a
drop-in selected by `capture/base.make_feed`, with no downstream change.

LATEST-FRAME GRABBER (the realtime fix)
---------------------------------------
`cv2.VideoCapture` on a network MJPEG stream BUFFERS frames in the FFmpeg backend: if the consumer
loop ever falls behind the source even briefly, `read()` keeps handing back progressively STALER
frames (measured ~13 frames / ~540 ms of backlog on this rig), so the tracker aims at where the face
*was* half a second ago. `CAP_PROP_BUFFERSIZE=1` is ignored by this backend.

So a dedicated background thread (`_reader_loop`) reads the capture as fast as the Pi produces and
keeps ONLY the newest decoded frame in a single-slot, lock-guarded variable — it continuously drains
the buffer so it can never back up. `read()` just returns that newest frame (or `None` if no new one
arrived since the last call), so every consumer always works on a near-current frame. The reader
thread snapshots the servo angle at GRAB time (ego-motion aware) and owns the reconnect.

The reconnect-until-ready `open()` loop and the consecutive-miss reconnect threshold are PORTED from
the proven old code (`station/pipeline.open_stream` / `track_client.open_stream` and their
`fails >= 30` reconnect). `read()` returns a `Frame` resized to the canonical W,H = 640,480.

`cv2` is imported lazily inside the methods so importing this module stays cheap and offline-safe
(no OpenCV / camera / network needed at import time); the camera is only touched in `open()`/`read()`.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from kitchenvision.capture.base import FeedSource
from kitchenvision.core.types import Frame

# Canonical working resolution. The Pi feed is already 640x480; anything else is resized to this
# so every downstream consumer (recognizer boxes, tracker hysteresis, overlay) shares one coord
# space. Matches INTERFACES.md (W, H = 640, 480).
W, H = 640, 480

# After this many CONSECUTIVE failed reads, drop the capture and transparently reopen (Pi restart,
# transient network drop, end-of-stream). Ported from station/pipeline._RECONNECT_AFTER and
# track_client.py's `fails >= 30` reconnect threshold.
_RECONNECT_AFTER = 30

# Seconds between connection attempts in the open()/reopen retry loop (ported: old code sleeps 1.0 s
# between cv2.VideoCapture attempts until the Pi agent is up).
_RETRY_INTERVAL = 1.0

# Max seconds read() blocks waiting for a fresh frame before returning None. It returns the instant a
# frame arrives, so in steady state this is just the inter-frame gap (~42 ms at 24 fps); the timeout
# only bites when the feed has gone quiet (then the caller loops and tries again).
_READ_TIMEOUT = 1.0


class MjpegFeed(FeedSource):
    """`FeedSource` backed by a threaded `cv2.VideoCapture` grabber on the Pi MJPEG `/raw` endpoint.

    Construct with the loaded `config` dict (§1) and an `angle_provider()` returning the current
    `{"pan": deg, ...}` servo angles. Call `open()` once (blocks, retrying, until the stream connects
    and the grabber thread starts), then `read()` per frame (returns the freshest `Frame` or `None`),
    and `close()` to release.
    """

    def __init__(self, config: dict, angle_provider: Callable[[], dict]) -> None:
        self._config = config
        self._angle_provider = angle_provider

        pi_ip = config.get("pi_ip", "192.168.68.127")
        stream_port = int(config.get("stream_port", 8000))
        stream_path = config.get("stream_path", "/raw")
        self._url = f"http://{pi_ip}:{stream_port}{stream_path}"

        self._cap = None              # cv2.VideoCapture | None (opened in open()/_reopen())
        self._misses = 0              # consecutive failed reads since the last good frame (reader thread)

        # --- grabber thread + latest-frame slot ---------------------------------------------
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._slot_lock = threading.Lock()
        self._latest = None           # (bgr, seq, ts, angles) newest decoded frame, or None
        self._seq = 0                 # monotonic frame counter stamped onto each Frame
        self._last_returned_seq = 0   # so read() never returns the same frame twice
        self._frame_event = threading.Event()

    # -- lifecycle ---------------------------------------------------------------------------
    def _open_capture(self):
        """Open the MJPEG stream, retrying every `_RETRY_INTERVAL` s until it connects.

        Ported from `station/pipeline.open_stream` / `track_client.open_stream`: blocks (with a
        1 s retry) until the Pi agent is up, so the capture loop survives the Pi being rebooted or
        not yet running. Returns an opened `cv2.VideoCapture`.
        """
        import cv2  # lazy: keep module import cheap + offline-safe

        while True:
            cap = cv2.VideoCapture(self._url)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; ignored by FFmpeg/MJPEG
                except Exception:
                    pass
                print(f"[mjpeg] connected to {self._url}", flush=True)
                return cap
            print(f"[mjpeg] waiting for stream {self._url} ... (is the Pi agent up?)", flush=True)
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            time.sleep(_RETRY_INTERVAL)

    def open(self) -> None:
        """Connect to the Pi MJPEG stream (retrying until ready) and start the grabber thread.

        Re-opens a fresh capture each call, stopping any previous grabber/capture first."""
        self.close()  # stop a prior thread + release a prior capture (idempotent-ish)
        self._cap = self._open_capture()
        self._misses = 0
        with self._slot_lock:
            self._latest = None
            self._seq = 0
            self._last_returned_seq = 0
            self._frame_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name="mjpeg-grabber", daemon=True)
        self._thread.start()

    def _reopen(self) -> None:
        """Transparently tear down and re-establish the capture after too many consecutive misses
        (Pi restart / stream drop). Runs ON the reader thread. Mirrors the old reconnect path."""
        print(f"[mjpeg] stream dropped after {self._misses} misses; reconnecting ...", flush=True)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = self._open_capture()
        self._misses = 0

    # -- grabber thread ----------------------------------------------------------------------
    def _reader_loop(self) -> None:
        """Continuously read the capture, keeping ONLY the newest frame (drain-to-latest).

        This is the whole point: by reading as fast as the Pi produces and discarding everything but
        the most recent decoded frame, the FFmpeg buffer never backs up, so `read()` always hands the
        consumer a near-current frame. Owns the consecutive-miss reconnect (the camera is only touched
        on this thread once `open()` has started it). `cv2` is the same cached module the tests patch.
        """
        import cv2  # lazy

        while self._running:
            cap = self._cap
            if cap is None:
                time.sleep(0.01)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self._misses += 1
                if self._misses >= _RECONNECT_AFTER:
                    self._reopen()
                else:
                    time.sleep(0.005)
                continue
            self._misses = 0

            # Normalise to the canonical working resolution (the Pi feed is already 640x480; resize
            # anything else so all downstream math shares one coord space).
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            # Snapshot the servo angle known at GRAB time (copy so a later mutation of the live
            # STATE.servo_angles dict can't retroactively change this frame's tag).
            try:
                angles = dict(self._angle_provider())
            except Exception:
                angles = {}

            with self._slot_lock:
                self._seq += 1
                self._latest = (frame, self._seq, time.time(), angles)
                self._frame_event.set()

    # -- read --------------------------------------------------------------------------------
    def read(self) -> Optional[Frame]:
        """Return the freshest captured `Frame`, or `None` if no NEW frame is available.

        Blocks up to `_READ_TIMEOUT` for the next fresh frame (returning the instant one arrives), so
        the caller's loop paces naturally to the source rate while always working on a near-current
        frame. `None` (no new frame / quiet feed) just means the caller tries again next tick — it is
        a transient miss, exactly as the contract (§5) allows. Auto-starts the grabber if `read()` is
        somehow called before `open()` (defensive; the pipeline calls `open()` first).
        """
        if self._thread is None or not self._thread.is_alive():
            self.open()

        if not self._frame_event.wait(_READ_TIMEOUT):
            return None
        with self._slot_lock:
            latest = self._latest
            self._frame_event.clear()
        if latest is None:
            return None
        frame, seq, ts, angles = latest
        if seq == self._last_returned_seq:
            return None  # woke without a genuinely new frame
        self._last_returned_seq = seq
        return Frame(bgr=frame, ts=ts, servo_angles=angles, seq=seq, hires=None)

    # -- teardown ----------------------------------------------------------------------------
    def close(self) -> None:
        """Stop the grabber thread and release the underlying capture (best-effort)."""
        self._running = False
        self._frame_event.set()  # wake any reader blocked in wait()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._thread = None
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        finally:
            self._cap = None
