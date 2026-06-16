"""Recognition quality gate — pure functions (INTERFACES.md §6).

A blurry, partial, or distant face yields a noisy ArcFace embedding; enrolling those is what
fragments ONE person into a swarm of "Unknown #N" clusters on a low-res panning feed. So before a
face may ENROLL (create a new person, append an embedding, nudge a centroid, save a crop) it must
clear ALL of `min_det_score` / `min_face_px` / `min_blur_var`. A face that fails the gate is still
TRACKED (returned as a box so the servo keeps framing it) and, if it matches an EXISTING person,
still records a sighting — it just never mutates the identity model or spawns a new cluster.

These are PORTED verbatim from `station/recognizer.py` (`_blur_var` + the inline gate) and kept as
free functions (no model, no GPU, no I/O) so they are unit-testable offline and reusable by both
`InsightFaceGpu` and the best-shot scorer in `fusion.py`.
"""
from __future__ import annotations

import cv2
import numpy as np


def blur_var(frame_bgr: np.ndarray, box: list[int]) -> float:
    """Variance of the Laplacian over the face crop = a cheap sharpness score.

    Low values mean a blurry / out-of-focus crop (a poor embedding); returns 0.0 on an empty or
    out-of-bounds crop. PORT of `Recognizer._blur_var`.

    `box` is `[x, y, w, h]` ints in full working-frame pixel coords.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return 0.0
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # Clamp negative origins so slicing can't wrap (negative index) into the far edge.
    x0 = max(0, x)
    y0 = max(0, y)
    crop = frame_bgr[y0:y + h, x0:x + w]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_gate(
    det_score: float,
    box: list[int],
    frame_bgr: np.ndarray,
    cfg: dict,
) -> bool:
    """May this face ENROLL (mutate the identity model)?

    Returns True only when the face clears ALL three gates read from `cfg["recognition"]`:
      * detector confidence    `det_score >= min_det_score`
      * on-screen face size    `min(w, h) >= min_face_px`
      * crop sharpness         `blur_var(frame, box) >= min_blur_var`

    `cfg` is the full config dict; the recognition sub-block supplies the thresholds (falling back
    to the documented defaults if a key is absent). PORT of the inline `quality_ok` test in
    `Recognizer._process_face`.
    """
    rec = cfg.get("recognition", {}) or {}
    min_det_score = float(rec.get("min_det_score", 0.65))
    min_face_px = int(rec.get("min_face_px", 70))
    min_blur_var = float(rec.get("min_blur_var", 40.0))

    if float(det_score) < min_det_score:
        return False
    if min(int(box[2]), int(box[3])) < min_face_px:
        return False
    if blur_var(frame_bgr, box) < min_blur_var:
        return False
    return True


def face_size_px(box: list[int]) -> int:
    """Shorter side of the face box in pixels (the size term used by the gate + best-shot score)."""
    return int(min(int(box[2]), int(box[3])))


def quality_score(det_score: float, box: list[int], frame_bgr: np.ndarray) -> float:
    """Fused best-shot quality score (blur + size + detector confidence) used by `fusion.py`.

    Higher is better. Combines three normalised cues so the best frame of a track is the one kept
    in the best-shot buffer:
      * detector confidence (0..1, capped),
      * face size (shorter side, normalised by a 200 px "big face" reference),
      * sharpness (`blur_var`, normalised by a 200.0 "crisp" reference).

    The exact constant weighting is not part of the frozen contract (only `quality` being populated
    is); it is a monotone, deterministic fusion so a sharper / bigger / more-confident crop always
    scores at least as high. Pure: no model, no I/O.
    """
    det = max(0.0, min(1.0, float(det_score)))
    size = max(0.0, min(1.0, face_size_px(box) / 200.0))
    sharp = max(0.0, min(1.0, blur_var(frame_bgr, box) / 200.0))
    return float(0.4 * det + 0.3 * size + 0.3 * sharp)
