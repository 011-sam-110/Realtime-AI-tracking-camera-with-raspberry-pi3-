"""Webcam image-quality optimisation (INTERFACES.md §1 `enhance`, Phase E).

The cheap USB webcam produces dim, low-contrast, slightly soft frames in typical kitchen lighting.
A small, FAST, deterministic enhancement chain cleans them up — both for what the user SEES on the
dashboard and (a gentler subset) for what the recogniser ingests, so detection + ArcFace embeddings
are computed on a cleaner image. Everything here is pure ``cv2`` + ``numpy`` (no model, no GPU, no
I/O), so it is unit-testable offline and cheap enough to run every frame at 640x480.

THE CHAIN (in order)
--------------------
  1. **gray-world white balance** *(display only)* — neutralise a colour cast.
  2. **gamma / auto-exposure** — lift a dark frame toward a target mean luminance (``auto_gamma``),
     or apply a fixed ``gamma``. A single LUT, so it is ~free.
  3. **CLAHE** on the L (luma) channel of LAB — local contrast that pulls a face out of flat,
     uneven lighting. This is the biggest single win for recognition on a low-end sensor.
  4. **unsharp mask** *(display only)* — counteract sensor softness for a crisper picture.
  5. **denoise** *(display only, OFF by default)* — fast bilateral; opt-in (it is the costly step).

WHY A DISPLAY vs RECOGNITION SPLIT
----------------------------------
InsightFace's detector/embedder are trained on natural images; aggressive sharpening or a colour
shift can perturb the embedding. So the recognition path applies only the *safe* subset
(auto-gamma + CLAHE) while the display path runs the full chain. ``apply(frame, for_display=...)``
selects which. Both are gated by config so the whole stage can be turned off in one place.

Used by: ``recognition.insightface_gpu`` (before ``app.get``) and ``pipeline`` (the display copy).
``make_enhancer(config)`` returns an ``Enhancer`` or ``None`` when disabled.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# pure helpers (each takes/returns a BGR uint8 frame; all no-ops on bad input)
# ---------------------------------------------------------------------------
def luma_mean(frame_bgr: np.ndarray) -> float:
    """Mean luma in [0,1] (0.0 on an empty frame). Used by auto-gamma + tests."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return float(gray.mean()) / 255.0


def luma_std(frame_bgr: np.ndarray) -> float:
    """Std-dev of luma in [0,1] — a cheap global-contrast metric (0.0 on an empty frame)."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return float(gray.std()) / 255.0


def gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so its mean matches the overall grey mean.

    Cheap, assumption-light cast removal. Per-channel gains are clamped so a near-monochrome frame
    can't blow up. Returns a new uint8 array.
    """
    f = frame_bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)            # B, G, R means
    grey = float(means.mean())
    if grey <= 1e-3:
        return frame_bgr
    gains = grey / np.maximum(means, 1e-3)
    gains = np.clip(gains, 0.5, 2.0)                  # never more than 2x / less than 0.5x a channel
    out = f * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _gamma_lut(gamma: float) -> np.ndarray:
    """256-entry uint8 LUT for ``out = 255*(in/255)**gamma``."""
    g = max(1e-3, float(gamma))
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip((x ** g) * 255.0, 0, 255).astype(np.uint8)


def apply_gamma(frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply a fixed gamma via a LUT (gamma < 1 brightens, > 1 darkens). 1.0 is a no-op."""
    if abs(float(gamma) - 1.0) < 1e-3:
        return frame_bgr
    return cv2.LUT(frame_bgr, _gamma_lut(gamma))


def auto_gamma(frame_bgr: np.ndarray, target_luma: float = 0.52,
               lo: float = 0.40, hi: float = 2.5) -> np.ndarray:
    """Pick a gamma that moves the frame's mean luma toward ``target_luma`` and apply it.

    Using ``mean_out ≈ mean_in**gamma`` (means in [0,1]): ``gamma = log(target)/log(mean_in)``.
    The gamma is clamped to ``[lo, hi]`` so a nearly-black or nearly-white frame can't be pushed to
    an extreme. No-op when the frame is already near target or is degenerate.
    """
    m = luma_mean(frame_bgr)
    if m <= 1e-3 or m >= 0.999:
        return frame_bgr
    t = float(min(0.99, max(0.01, target_luma)))
    if abs(m - t) < 0.02:
        return frame_bgr
    import math
    gamma = math.log(t) / math.log(m)
    gamma = float(min(hi, max(lo, gamma)))
    return apply_gamma(frame_bgr, gamma)


def clahe_luma(frame_bgr: np.ndarray, clahe: "cv2.CLAHE") -> np.ndarray:
    """Apply CLAHE to the L channel of LAB and convert back to BGR (local contrast)."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def unsharp_mask(frame_bgr: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask: ``out = frame + amount*(frame - blur(frame))`` — crisper edges. amount<=0 no-op."""
    if amount <= 0.0:
        return frame_bgr
    blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=max(0.1, float(sigma)))
    return cv2.addWeighted(frame_bgr, 1.0 + amount, blur, -amount, 0)


# ---------------------------------------------------------------------------
# Enhancer
# ---------------------------------------------------------------------------
class Enhancer:
    """Config-driven webcam enhancement with a reused CLAHE object.

    ``apply(frame, for_display)`` runs the chain; the display path adds white-balance, unsharp and
    (optional) denoise on top of the recognition-safe auto-gamma + CLAHE. Robust to ``None`` / empty
    / non-3-channel frames (returned unchanged). ``for_recognition`` / ``for_display`` flags let the
    caller skip the stage entirely.
    """

    def __init__(self, params: dict) -> None:
        p = params or {}
        self.enabled = bool(p.get("enabled", True))
        self.for_recognition = bool(p.get("for_recognition", True))
        self.for_display = bool(p.get("for_display", True))

        self.use_white_balance = bool(p.get("white_balance", True))      # display-only
        self.use_auto_gamma = bool(p.get("auto_gamma", True))
        self.gamma = float(p.get("gamma", 1.0))                          # manual, if not auto
        self.target_luma = float(p.get("target_luma", 0.52))

        self.use_clahe = bool(p.get("clahe", True))
        self.clahe_clip = float(p.get("clahe_clip", 2.0))
        self.clahe_grid = int(p.get("clahe_grid", 8))

        self.use_unsharp = bool(p.get("unsharp", True))                  # display-only
        self.unsharp_amount = float(p.get("unsharp_amount", 0.6))
        self.unsharp_sigma = float(p.get("unsharp_sigma", 1.0))

        self.use_denoise = bool(p.get("denoise", False))                 # display-only, costly
        self.denoise_strength = int(p.get("denoise_strength", 5))

        grid = max(1, self.clahe_grid)
        self._clahe = cv2.createCLAHE(clipLimit=max(0.1, self.clahe_clip),
                                      tileGridSize=(grid, grid))

    # ------------------------------------------------------------------ apply
    def apply(self, frame_bgr: np.ndarray, for_display: bool = False) -> np.ndarray:
        """Return an enhanced copy of ``frame_bgr``.

        Recognition path (``for_display=False``): auto-gamma/gamma + CLAHE only (embedding-safe).
        Display path (``for_display=True``): white-balance + gamma + CLAHE + unsharp + optional
        denoise. Never raises and never mutates the input in place; bad input is returned unchanged.
        """
        if not self.enabled:
            return frame_bgr
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0 or frame_bgr.ndim != 3:
            return frame_bgr
        try:
            out = frame_bgr
            if for_display and self.use_white_balance:
                out = gray_world_white_balance(out)
            if self.use_auto_gamma:
                out = auto_gamma(out, self.target_luma)
            elif abs(self.gamma - 1.0) >= 1e-3:
                out = apply_gamma(out, self.gamma)
            if self.use_clahe:
                out = clahe_luma(out, self._clahe)
            if for_display and self.use_unsharp:
                out = unsharp_mask(out, self.unsharp_amount, self.unsharp_sigma)
            if for_display and self.use_denoise:
                h = max(1, self.denoise_strength)
                out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)
            # Guarantee a fresh array even if every step happened to be a no-op (callers may keep
            # the original raw frame separately).
            return out if out is not frame_bgr else frame_bgr.copy()
        except cv2.error:
            return frame_bgr


def make_enhancer(config: dict) -> Optional[Enhancer]:
    """Return an :class:`Enhancer` from ``config['enhance']`` (or ``None`` when disabled/absent)."""
    params = (config or {}).get("enhance", {}) or {}
    if not params.get("enabled", True):
        return None
    return Enhancer(params)
