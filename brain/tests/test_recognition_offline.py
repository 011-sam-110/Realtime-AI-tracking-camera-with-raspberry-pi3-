"""Offline unit tests for the recognition quality gate + fusion tracker (INTERFACES.md §6).

Imports ONLY the pure modules — `kitchenvision.recognition.quality` and
`kitchenvision.recognition.fusion` — NOT `insightface_gpu` (which needs the model + DB) nor
`yunet` (which opens the ONNX file). So this runs with no model load, no GPU, no network, no DB.

Plain-python runnable (pytest may be absent):  python tests/test_recognition_offline.py
Run from the brain root (C:/Users/sampo/pi/brain) so `import kitchenvision...` resolves.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Make `import kitchenvision...` work when run as a bare script from the brain root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.recognition import quality as q  # noqa: E402
from kitchenvision.recognition.fusion import Fusion, iou  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _sharp_patch(size: int = 120, seed: int = 0) -> np.ndarray:
    """A high-frequency BGR patch (random noise + a hard checkerboard) → high Laplacian variance."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    # Hard 8x8 checkerboard to guarantee strong edges (sharpness) regardless of the RNG.
    block = size // 8
    for by in range(8):
        for bx in range(8):
            if (bx + by) % 2 == 0:
                img[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = 255
            else:
                img[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = 0
    return img


def _blurred_patch(size: int = 120, seed: int = 0) -> np.ndarray:
    """The same content as `_sharp_patch`, heavily Gaussian-blurred → low Laplacian variance."""
    import cv2
    return cv2.GaussianBlur(_sharp_patch(size, seed), (0, 0), sigmaX=9.0)


# --------------------------------------------------------------------------- tests
def test_blur_var_sharp_beats_blurred() -> None:
    """A sharp crop must score far higher than the same crop blurred; empty crop = 0.0."""
    frame_sharp = _sharp_patch()
    frame_blur = _blurred_patch()
    box = [0, 0, 120, 120]

    v_sharp = q.blur_var(frame_sharp, box)
    v_blur = q.blur_var(frame_blur, box)

    assert v_sharp > 0.0, f"sharp blur_var should be positive, got {v_sharp}"
    assert v_blur >= 0.0
    assert v_sharp > v_blur * 5.0, (
        f"sharp ({v_sharp:.1f}) should dominate blurred ({v_blur:.1f})"
    )
    print(f"  blur_var: sharp={v_sharp:.1f}  blurred={v_blur:.1f}  ratio={v_sharp / max(v_blur,1e-6):.1f}x")


def test_blur_var_edge_cases() -> None:
    """Out-of-frame / empty / None inputs return 0.0 and never raise."""
    frame = _sharp_patch()
    assert q.blur_var(frame, [200, 200, 50, 50]) == 0.0   # fully outside → empty crop
    assert q.blur_var(frame, [0, 0, 0, 0]) == 0.0          # zero-area box
    assert q.blur_var(None, [0, 0, 10, 10]) == 0.0          # no frame
    assert q.blur_var(np.zeros((0, 0, 3), np.uint8), [0, 0, 10, 10]) == 0.0
    print("  blur_var edge cases: all 0.0, no raise")


def test_passes_gate() -> None:
    """The gate accepts a big/sharp/confident face and rejects each failing dimension."""
    cfg = {"recognition": {"min_det_score": 0.65, "min_face_px": 70, "min_blur_var": 40.0}}
    sharp = _sharp_patch(size=120)

    good_box = [0, 0, 120, 120]
    assert q.passes_gate(0.9, good_box, sharp, cfg) is True, "big/sharp/confident must pass"

    # Fails det score.
    assert q.passes_gate(0.50, good_box, sharp, cfg) is False
    # Fails size (shorter side < min_face_px).
    assert q.passes_gate(0.9, [0, 0, 50, 50], sharp, cfg) is False
    # Fails blur (blurred crop below min_blur_var).
    assert q.passes_gate(0.9, good_box, _blurred_patch(size=120), cfg) is False
    print("  passes_gate: accepts good face; rejects low-score / small / blurred")


def test_quality_score_monotone() -> None:
    """A sharper + bigger + more-confident shot scores at least as high; result in [0,1]."""
    sharp = _sharp_patch(size=160)
    blur = _blurred_patch(size=160)
    box = [0, 0, 160, 160]
    s_hi = q.quality_score(0.95, box, sharp)
    s_lo = q.quality_score(0.50, box, blur)
    assert 0.0 <= s_lo <= 1.0 and 0.0 <= s_hi <= 1.0
    assert s_hi > s_lo, f"good shot ({s_hi:.3f}) should beat poor shot ({s_lo:.3f})"
    print(f"  quality_score: good={s_hi:.3f} > poor={s_lo:.3f}")


def test_iou_basic() -> None:
    """IoU is 1.0 for identical boxes, 0.0 for disjoint, ~0.14 for the classic half-overlap."""
    assert iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert iou([0, 0, 50, 50], [100, 100, 50, 50]) == 0.0
    half = iou([0, 0, 100, 100], [50, 50, 100, 100])   # inter=2500, union=17500 → 1/7
    assert abs(half - (1.0 / 7.0)) < 1e-6, half
    print(f"  iou: identical=1.0  disjoint=0.0  half-overlap={half:.3f}")


def test_fusion_track_continuity() -> None:
    """A box that drifts a little keeps its track_id; a far-away box gets a new one."""
    fusion = Fusion({"recognition": {"fusion_iou": 0.30, "fusion_window": 5}})

    # Frame 1: two faces.
    fusion.begin_frame()
    a0 = [100, 100, 80, 80]
    b0 = [400, 100, 80, 80]
    tid_a = fusion.assign(a0)
    tid_b = fusion.assign(b0)
    fusion.observe(tid_a, 0.50, a0, person_id=7)
    fusion.observe(tid_b, 0.40, b0)
    fusion.end_frame()
    assert tid_a != tid_b, "distinct faces must get distinct track ids"

    # Frame 2: face A drifts 15px (high IoU → same id); face B drifts 12px (same id);
    # a brand-new face appears far away (new id).
    fusion.begin_frame()
    a1 = [115, 108, 80, 80]
    b1 = [388, 112, 80, 80]
    c1 = [100, 350, 60, 60]
    tid_a2 = fusion.assign(a1)
    tid_b2 = fusion.assign(b1)
    tid_c2 = fusion.assign(c1)
    fusion.observe(tid_a2, 0.30, a1)            # lower quality than frame-1 best
    fusion.observe(tid_b2, 0.60, b1)            # higher quality than frame-1 best
    fusion.observe(tid_c2, 0.20, c1)
    fusion.end_frame()

    assert tid_a2 == tid_a, f"drifted face A should keep track {tid_a}, got {tid_a2}"
    assert tid_b2 == tid_b, f"drifted face B should keep track {tid_b}, got {tid_b2}"
    assert tid_c2 not in (tid_a, tid_b), "the new far-away face must get a fresh track id"
    print(f"  track continuity: A {tid_a}->{tid_a2}, B {tid_b}->{tid_b2}, new={tid_c2}")

    # Best-shot buffer: A keeps its frame-1 high-quality shot; B's best updates to frame-2.
    assert abs(fusion.quality_of(tid_a) - 0.50) < 1e-6, "A's best-shot quality should stay 0.50"
    assert abs(fusion.quality_of(tid_b) - 0.60) < 1e-6, "B's best-shot quality should rise to 0.60"
    # person_id learned in frame 1 persists even though frame-2 observe passed no person_id.
    assert fusion.person_of(tid_a) == 7, "A's resolved person_id should persist across frames"
    print("  best-shot: A.q=0.50 (kept)  B.q=0.60 (raised)  A.person=7 (persisted)")


def test_fused_embedding() -> None:
    """fused_embedding = quality-weighted, L2-normalised mean of a track's window embeddings."""
    fusion = Fusion({"recognition": {"fusion_window": 5, "fusion_iou": 0.30}})
    box = [100, 100, 80, 80]

    # No track / no embeddings yet → None.
    assert fusion.fused_embedding(999) is None, "unknown track → None"

    fusion.begin_frame()
    tid = fusion.assign(box)
    # A detect-only observation (no embedding) must NOT produce a fused vector.
    fusion.observe(tid, 0.5, box, embedding=None)
    assert fusion.fused_embedding(tid) is None, "no embeddings buffered → None"

    # Two orthogonal embeddings with equal weight → mean points between them, renormalised to unit.
    e1 = np.zeros(8, np.float32); e1[0] = 1.0
    e2 = np.zeros(8, np.float32); e2[1] = 1.0
    fusion.observe(tid, 0.5, box, embedding=e1)
    fusion.observe(tid, 0.5, box, embedding=e2)
    fused = fusion.fused_embedding(tid)
    assert fused is not None
    assert abs(float(np.linalg.norm(fused)) - 1.0) < 1e-5, "fused vector must be unit-norm"
    assert abs(fused[0] - fused[1]) < 1e-5, "equal weights → symmetric mean of e1,e2"
    assert abs(fused[0] - (1.0 / np.sqrt(2))) < 1e-4

    # Quality weighting: a much higher-quality e1 pulls the fused vector toward e1.
    fusion2 = Fusion({"recognition": {"fusion_window": 5}})
    fusion2.begin_frame()
    t2 = fusion2.assign(box)
    fusion2.observe(t2, 0.95, box, embedding=e1)   # high quality
    fusion2.observe(t2, 0.05, box, embedding=e2)   # low quality
    fw = fusion2.fused_embedding(t2)
    assert fw[0] > fw[1], "higher-quality shot should dominate the fused embedding"
    print(f"  fused_embedding: unit-norm symmetric mean; quality-weighted leans to e1 ({fw[0]:.2f}>{fw[1]:.2f})")


def test_fusion_track_ageout() -> None:
    """A track disappears after `fusion_max_age` unseen frames; one frame absence keeps it."""
    fusion = Fusion({"recognition": {"fusion_max_age": 2, "fusion_iou": 0.30}})
    fusion.begin_frame()
    box = [200, 200, 60, 60]
    tid = fusion.assign(box)
    fusion.observe(tid, 0.5, box)
    fusion.end_frame()
    assert tid in fusion.live_tracks

    # Two empty frames == max_age → still alive after frame 1, still alive after frame 2 (age==2 <= 2).
    for _ in range(2):
        fusion.begin_frame()
        fusion.end_frame()
    assert tid in fusion.live_tracks, "track should survive exactly max_age unseen frames"

    # The 3rd empty frame pushes age to 3 > 2 → dropped.
    fusion.begin_frame()
    fusion.end_frame()
    assert tid not in fusion.live_tracks, "track should be dropped after max_age unseen frames"
    print("  age-out: track survives max_age unseen frames, dropped on the next")


def _run_all() -> int:
    tests = [
        test_blur_var_sharp_beats_blurred,
        test_blur_var_edge_cases,
        test_passes_gate,
        test_quality_score_monotone,
        test_iou_basic,
        test_fusion_track_continuity,
        test_fused_embedding,
        test_fusion_track_ageout,
    ]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__} ...")
            t()
            print(f"  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} tests FAILED")
    else:
        print(f"RESULT: all {len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
