"""Best-shot fusion + identity-persistence tracker (INTERFACES.md §6).

`Fusion` gives each face a STABLE `track_id` that survives across frames, and keeps the single
best-quality observation (embedding + crop) seen so far for each live track (a best-shot buffer of
up to `fusion_window` candidates). This is what makes the displayed identity steady on a low-res
panning feed: the recogniser can match the *best* frame of a track instead of whatever noisy frame
happens to be current, and the dashboard's box keeps the same `track_id` while a person moves.

Tracking is pure geometry — each new frame's boxes are greedily matched to the previous frame's
tracks by IoU (highest overlap first, above `iou_thresh`); an unmatched box starts a new track with
a fresh monotonically-increasing id. Tracks not seen for `max_age` frames are dropped. NO model, NO
GPU, NO I/O here — just numpy + geometry, so it is unit-testable offline.

Threading: a `Fusion` instance is owned by the single recognition thread (the only caller of
`FaceEngine.recognize`), so it needs no internal lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two `[x, y, w, h]` boxes (0.0 if they don't overlap)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    area_a = float(max(0, aw) * max(0, ah))
    area_b = float(max(0, bw) * max(0, bh))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


@dataclass
class BestShot:
    """The best-quality observation retained for one track."""
    quality: float
    box: list[int]
    embedding: Optional[np.ndarray] = None       # L2-normalised ArcFace vector, or None (detect-only)
    crop: Optional[np.ndarray] = None            # BGR face crop of the best frame, or None
    person_id: int = -1                          # last known identity for the track (-1 = unknown)


@dataclass
class _Track:
    """Live per-track state carried across frames."""
    track_id: int
    box: list[int]                               # last box seen (for next-frame IoU matching)
    age: int = 0                                  # frames since last matched (0 = matched this frame)
    best: Optional[BestShot] = None              # best-shot buffer head (highest quality so far)
    window: list[BestShot] = field(default_factory=list)   # recent candidates, newest last


class Fusion:
    """IoU identity tracker with a per-track best-shot buffer.

    Typical per-frame use by the recognition engine:

        fusion.begin_frame()
        for each detected face:
            tid = fusion.assign(box)                 # stable track id (IoU vs previous frame)
            fusion.observe(tid, quality, box, emb, crop, person_id)   # feed the best-shot buffer
        fusion.end_frame()                           # ages out tracks not seen this frame
    """

    def __init__(self, config: dict | None = None) -> None:
        rec = ((config or {}).get("recognition", {}) or {})
        # Best-shot buffer depth (how many recent candidate shots to keep per track).
        self.fusion_window = int(rec.get("fusion_window", 5))
        # IoU floor for "same face as last frame". A panning low-res feed moves boxes a lot between
        # the few-fps recognition frames, so this is deliberately loose.
        self.iou_thresh = float(rec.get("fusion_iou", 0.30))
        # Drop a track this many UNMATCHED frames after it was last seen.
        self.max_age = int(rec.get("fusion_max_age", 8))

        self._tracks: list[_Track] = []
        self._next_id = 1
        self._claimed: set[int] = set()           # track ids already assigned in the current frame

    # ------------------------------------------------------------------ frame lifecycle
    def begin_frame(self) -> None:
        """Start a new frame: nothing may be claimed yet."""
        self._claimed = set()

    def assign(self, box: list[int]) -> int:
        """Return a stable `track_id` for `box`, IoU-matching it to the previous frame's tracks.

        Greedy: the un-claimed track with the highest IoU above `iou_thresh` wins; if none qualify a
        brand-new track id is created. Each track can be claimed by at most one box per frame.
        """
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        best_tid = -1
        best_iou = self.iou_thresh
        for t in self._tracks:
            if t.track_id in self._claimed:
                continue
            ov = iou(box, t.box)
            if ov >= best_iou:
                best_iou = ov
                best_tid = t.track_id

        if best_tid == -1:
            tid = self._next_id
            self._next_id += 1
            self._tracks.append(_Track(track_id=tid, box=box, age=0))
            self._claimed.add(tid)
            return tid

        # Matched an existing track: refresh its box + mark it seen this frame.
        for t in self._tracks:
            if t.track_id == best_tid:
                t.box = box
                t.age = 0
                break
        self._claimed.add(best_tid)
        return best_tid

    def observe(
        self,
        track_id: int,
        quality: float,
        box: list[int],
        embedding: Optional[np.ndarray] = None,
        crop: Optional[np.ndarray] = None,
        person_id: int = -1,
    ) -> None:
        """Feed one observation of `track_id` into its best-shot buffer.

        Keeps the single highest-quality shot as `track.best` and a rolling window of the most recent
        `fusion_window` candidates. `person_id` (when >= 0) is remembered as the track's identity so a
        momentarily-unmatched frame can still display the right person.
        """
        t = self._get(track_id)
        if t is None:
            return
        shot = BestShot(
            quality=float(quality),
            box=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32).ravel(),
            crop=crop,
            person_id=int(person_id),
        )
        t.window.append(shot)
        if len(t.window) > self.fusion_window:
            del t.window[0]
        if t.best is None or shot.quality >= t.best.quality:
            # Preserve a previously-resolved person_id if this (possibly detect-only) shot lacks one.
            if shot.person_id < 0 and t.best is not None and t.best.person_id >= 0:
                shot.person_id = t.best.person_id
            t.best = shot
        elif person_id >= 0:
            t.best.person_id = int(person_id)

    def end_frame(self) -> None:
        """Age every track; drop those unseen for more than `max_age` frames."""
        survivors: list[_Track] = []
        for t in self._tracks:
            if t.track_id in self._claimed:
                t.age = 0
                survivors.append(t)
            else:
                t.age += 1
                if t.age <= self.max_age:
                    survivors.append(t)
        self._tracks = survivors

    # ------------------------------------------------------------------ queries
    def best_shot(self, track_id: int) -> Optional[BestShot]:
        """The highest-quality observation retained for `track_id`, or None."""
        t = self._get(track_id)
        return None if t is None else t.best

    def quality_of(self, track_id: int) -> float:
        """Best quality score seen for `track_id` so far (0.0 if unknown)."""
        t = self._get(track_id)
        if t is None or t.best is None:
            return 0.0
        return float(t.best.quality)

    def person_of(self, track_id: int) -> int:
        """Last resolved person id for `track_id` (-1 if never matched)."""
        t = self._get(track_id)
        if t is None or t.best is None:
            return -1
        return int(t.best.person_id)

    def set_person(self, track_id: int, person_id: int) -> None:
        """Stamp `track_id`'s resolved identity (Phase E).

        The recogniser observes a face (person_id unknown) to seed the best-shot window, runs the
        match, then calls this to record the resolved id on the track's best shot — so a later
        momentarily-unmatched frame still displays the right person via `person_of`.
        """
        t = self._get(track_id)
        if t is None or t.best is None:
            return
        t.best.person_id = int(person_id)

    def fused_embedding(self, track_id: int) -> "Optional[np.ndarray]":
        """Quality-weighted mean of the track's recent best-shot embeddings, L2-normalised (Phase E).

        Matching the FUSED embedding of the last few frames instead of whatever noisy single frame is
        current is the multi-frame fusion win: it averages out webcam noise / motion blur, so the
        identity is far steadier on a low-res panning feed. Returns ``None`` if the track has no
        embeddings buffered yet (e.g. detect-only frames). Window embeddings are already normalised
        (the engine normalises before `observe`); higher-quality shots get more weight.
        """
        t = self._get(track_id)
        if t is None or not t.window:
            return None
        vecs: "list[np.ndarray]" = []
        weights: "list[float]" = []
        for s in t.window:
            if s.embedding is not None and s.embedding.size:
                vecs.append(np.asarray(s.embedding, dtype=np.float32).ravel())
                weights.append(max(1e-3, float(s.quality)))
        if not vecs:
            return None
        arr = np.stack(vecs, axis=0).astype(np.float32)
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        mean = (arr * w).sum(axis=0) / float(w.sum())
        n = float(np.linalg.norm(mean))
        if n > 0.0:
            mean = mean / n
        return mean.astype(np.float32)

    @property
    def live_tracks(self) -> list[int]:
        """Currently-live track ids."""
        return [t.track_id for t in self._tracks]

    def _get(self, track_id: int) -> Optional[_Track]:
        for t in self._tracks:
            if t.track_id == int(track_id):
                return t
        return None
