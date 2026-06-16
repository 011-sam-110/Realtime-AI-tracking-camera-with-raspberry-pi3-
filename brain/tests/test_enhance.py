"""Offline unit tests for the webcam enhancement chain (INTERFACES.md §1 `enhance`, Phase E).

Imports ONLY `kitchenvision.capture.enhance` (pure cv2 + numpy — no model, no GPU, no camera, no
network). Verifies the chain preserves shape/dtype, that auto-gamma brightens a dark frame, that
CLAHE lifts local contrast, that the display path differs from the recognition path, and that bad
input is tolerated.

Plain-python runnable (pytest may be absent):  python tests/test_enhance.py
Run from the brain root (C:/Users/sampo/pi/brain) so `import kitchenvision...` resolves.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.capture import enhance as E  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _dark_lowcontrast(h: int = 120, w: int = 160, base: int = 40, spread: int = 18,
                      seed: int = 0) -> np.ndarray:
    """A dim, low-contrast BGR frame: values clustered in a narrow dark band (the cheap-webcam case)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(base, base + spread, size=(h, w, 3), dtype=np.uint8)
    return img


def _params(**over) -> dict:
    base = {
        "enabled": True, "for_recognition": True, "for_display": True,
        "white_balance": True, "auto_gamma": True, "gamma": 1.0, "target_luma": 0.52,
        "clahe": True, "clahe_clip": 2.0, "clahe_grid": 8,
        "unsharp": True, "unsharp_amount": 0.6, "unsharp_sigma": 1.0,
        "denoise": False, "denoise_strength": 5,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- tests
def test_shape_dtype_preserved() -> None:
    """Both paths keep HxWx3 uint8 and never mutate the input in place."""
    enh = E.Enhancer(_params())
    src = _dark_lowcontrast()
    before = src.copy()
    for disp in (False, True):
        out = enh.apply(src, for_display=disp)
        assert out.shape == src.shape, f"shape changed ({disp})"
        assert out.dtype == np.uint8, f"dtype changed ({disp})"
    assert np.array_equal(src, before), "input frame must not be mutated in place"
    print("  shape/dtype preserved on both paths; input untouched")


def test_auto_gamma_brightens_dark_frame() -> None:
    """The recognition path lifts a dark frame's mean luma toward the target."""
    enh = E.Enhancer(_params())
    src = _dark_lowcontrast(base=30, spread=15)
    m0 = E.luma_mean(src)
    out = enh.apply(src, for_display=False)        # recognition path = auto-gamma + CLAHE
    m1 = E.luma_mean(out)
    assert m0 < 0.2, f"fixture should start dark, got {m0:.3f}"
    assert m1 > m0 + 0.1, f"auto-gamma should brighten ({m0:.3f} -> {m1:.3f})"
    print(f"  auto-gamma: mean luma {m0:.3f} -> {m1:.3f} (target 0.52)")


def test_clahe_lifts_contrast() -> None:
    """CLAHE (+gamma) raises the global-contrast metric on a low-contrast frame."""
    enh = E.Enhancer(_params(unsharp=False, white_balance=False))  # isolate gamma+clahe
    src = _dark_lowcontrast(base=90, spread=12)   # mid-grey, very low contrast
    s0 = E.luma_std(src)
    out = enh.apply(src, for_display=False)
    s1 = E.luma_std(out)
    assert s1 > s0, f"contrast (luma std) should rise ({s0:.4f} -> {s1:.4f})"
    print(f"  CLAHE: luma std {s0:.4f} -> {s1:.4f}")


def test_display_differs_from_recognition() -> None:
    """The display path (white-balance + unsharp on top) must differ from the recognition path."""
    enh = E.Enhancer(_params())
    src = _dark_lowcontrast(seed=3)
    rec = enh.apply(src, for_display=False)
    disp = enh.apply(src, for_display=True)
    assert not np.array_equal(rec, disp), "display path should add WB + unsharp vs recognition path"
    print("  display path != recognition path (extra WB + unsharp applied)")


def test_disabled_and_factory() -> None:
    """`enabled:false` → make_enhancer returns None; an Enhancer with enabled False is a passthrough."""
    assert E.make_enhancer({"enhance": {"enabled": False}}) is None
    assert isinstance(E.make_enhancer({"enhance": {"enabled": True}}), E.Enhancer)
    assert isinstance(E.make_enhancer({}), E.Enhancer), "absent block → defaults enabled"

    off = E.Enhancer(_params(enabled=False))
    src = _dark_lowcontrast()
    assert np.array_equal(off.apply(src), src), "disabled enhancer must pass the frame through"
    print("  factory: enabled flag honoured; disabled = exact passthrough")


def test_robust_to_bad_input() -> None:
    """None / empty / non-3-channel frames are returned unchanged and never raise."""
    enh = E.Enhancer(_params())
    assert enh.apply(None) is None
    empty = np.zeros((0, 0, 3), np.uint8)
    assert enh.apply(empty).size == 0
    gray = np.full((40, 40), 80, np.uint8)         # 2-D (not BGR)
    assert np.array_equal(enh.apply(gray), gray), "2-D frame returned unchanged"
    assert E.luma_mean(None) == 0.0 and E.luma_std(None) == 0.0
    print("  robust: None/empty/2-D handled, no raise")


def test_gamma_lut_monotone() -> None:
    """apply_gamma<1 brightens, >1 darkens, ==1 is a no-op; gray-world WB keeps shape/dtype."""
    mid = np.full((20, 20, 3), 100, np.uint8)
    assert E.apply_gamma(mid, 1.0) is mid, "gamma 1.0 is a no-op (same object)"
    assert int(E.apply_gamma(mid, 0.5).mean()) > 100, "gamma<1 brightens"
    assert int(E.apply_gamma(mid, 2.0).mean()) < 100, "gamma>1 darkens"
    wb = E.gray_world_white_balance(_dark_lowcontrast())
    assert wb.shape == (120, 160, 3) and wb.dtype == np.uint8
    print("  gamma LUT monotone; WB preserves shape/dtype")


def _run_all() -> int:
    tests = [
        test_shape_dtype_preserved,
        test_auto_gamma_brightens_dark_frame,
        test_clahe_lifts_contrast,
        test_display_differs_from_recognition,
        test_disabled_and_factory,
        test_robust_to_bad_input,
        test_gamma_lut_monotone,
    ]
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__} ...")
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print()
    print(f"RESULT: {'all ' + str(len(tests)) + ' tests passed' if not failed else str(failed) + '/' + str(len(tests)) + ' FAILED'}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
