"""Configuration loading (INTERFACES.md §1).

`load_config()` = `DEFAULTS` deep-merged with an optional `config.json` at the brain root. The
nested blocks (`recognition`, `track`, `vlm`, `activity`) merge key-by-key; `data_dir` is forced
absolute. A missing / partial / malformed config.json is tolerated (→ defaults). The returned dict
is a fresh copy callers may mutate.
"""
from __future__ import annotations

import copy
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../brain/kitchenvision/core
_BRAIN = os.path.dirname(os.path.dirname(_HERE))            # .../brain
CONFIG_PATH = os.path.join(_BRAIN, "config.json").replace("\\", "/")
_DEFAULT_DATA_DIR = os.path.join(_BRAIN, "data").replace("\\", "/")

DEFAULTS: dict = {
    "pi_ip": "192.168.68.127",
    "stream_port": 8000,
    "stream_path": "/raw",
    "servo_udp": 9999,
    "overlay_udp": 9998,
    "servo_enabled": True,        # false → NullServo (tracker runs, servo provably never moves)
    "dashboard_port": 8090,
    "retention_days": 30,
    "data_dir": _DEFAULT_DATA_DIR,
    "recognition": {
        "engine": "insightface",
        "model": "buffalo_l",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "det_size": 640,
        "det_thresh": 0.6,
        "recog_threshold": 0.45,
        "min_det_score": 0.65,
        "min_face_px": 70,
        "min_blur_var": 40.0,
        "centroid_alpha": 0.9,
        "fusion_window": 5,
        # --- Phase E recognition hardening ---
        "templates_per_person": 4,   # exemplars kept per identity for matching (1 = centroid only)
        "match_fused": True,         # match the per-track fused (best-shot mean) embedding
        "recog_margin": 0.04,        # min gap top1 vs best OTHER person to enrol (ambiguity guard)
        "unknown_min_margin": 0.10,  # best sim must be < threshold-this to spawn a NEW unknown
    },
    "enhance": {                     # Phase E webcam image-quality optimisation (capture/enhance.py)
        "enabled": True,
        "for_recognition": True,     # auto-gamma + CLAHE before app.get() (embedding-safe subset)
        "for_display": True,         # full chain on the dashboard video copy
        "white_balance": True,       # display-only (gray-world cast removal)
        "auto_gamma": True,          # lift a dark frame toward target_luma
        "gamma": 1.0,                # fixed gamma if auto_gamma is False
        "target_luma": 0.52,
        "clahe": True,
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "unsharp": True,             # display-only
        "unsharp_amount": 0.6,
        "unsharp_sigma": 1.0,
        "denoise": False,            # display-only, costly (fast bilateral) — opt in
        "denoise_strength": 5,
    },
    "detector": {                    # fast face detector that drives TRACKING at frame rate,
        "enabled": True,             # decoupled from the slow recognition engine (capture/detector.py)
        "engine": "yunet",           # cv2.FaceDetectorYN over brain/models/yunet.onnx
        "det_size": 320,             # downscaled detect width (320x240) -> ~3-8 ms/frame on CPU
        "det_thresh": 0.6,           # YuNet score threshold
    },
    "track": {                       # "responsive but smooth" tuning (decoupled fast-box tracking).
        "outer_px": 55,              # start correcting once the group drifts this far off centre
        "inner_px": 30,              # ...and recentre until it is back within this band
        "kp": 0.06,                  # proportional gain (loop gain still < 1 -> no overshoot)
        "max_step": 10.0,            # cap a single fresh step (deg); the Pi 100 Hz slew smooths it
        "direction": -1,
        "ema": 0.25,                 # light group-centre smoothing (fresh 24 fps tolerates it)
        "search_after": 2.5,         # grace before sweeping (a brief blur-loss must not trigger it)
        "sweep_speed": 22.0,
        "min_step": 2.0,             # ignore corrective steps smaller than this (anti micro-jitter)
        "hold_seconds": 0.2,         # short settle after a move so the servo de-blurs (not a 2 s freeze)
        "axes": ["pan"],
    },
    "vlm": {
        "backend": "local",
        "local_model": "qwen2-vl-2b",
        "device": "cuda",
        "max_tokens": 300,
        "base_url": "",
        "api_key": "",
        "model": "auto",
    },
    "activity": {
        "enabled": True,
        "cadence_seconds": 10,
        "save_thumbnails": True,
        "min_confidence": 0.0,
    },
}

_NESTED = ("recognition", "track", "vlm", "activity", "enhance", "detector")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k in _NESTED and isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg = _deep_merge(cfg, user)
    except (FileNotFoundError, ValueError, OSError):
        pass  # missing / malformed config.json → defaults
    cfg["data_dir"] = os.path.abspath(cfg.get("data_dir") or _DEFAULT_DATA_DIR).replace("\\", "/")
    return cfg
