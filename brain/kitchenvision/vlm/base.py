"""VisionModel interface, result type, errors + factory (INTERFACES.md §9).

A VisionModel is a thin "image + text prompt → text" generator, so the local and cloud backends are
interchangeable behind one contract. The perception layer owns the prompt + parsing; the model just
generates. Errors are typed so the perception worker can map them to `STATE.activity_status`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class VlmResult:
    text: str
    provider: str          # local model id, or the cloud X-Routed-Via backend


class VlmError(Exception):
    """Base class for VisionModel failures."""


class NoVisionError(VlmError):
    """Upstream has no vision-capable model (cloud 422) — worker → 'no_vision_model', stop calling."""


class RateLimitError(VlmError):
    """Rate limited (cloud 429) — worker → 'error' + exponential backoff."""


@runtime_checkable
class VisionModel(Protocol):
    def generate(self, image_bgr: np.ndarray, prompt: str, max_tokens: int = 300) -> VlmResult: ...
    @property
    def available(self) -> bool: ...      # False ⇒ perception worker reports 'disabled'


def make_vision(config: dict) -> "VisionModel":
    """Select the vision backend by config['vlm']['backend'] (default 'local'). Built in Phase C."""
    backend = (config.get("vlm", {}) or {}).get("backend", "local")
    if backend == "cloud":
        from kitchenvision.vlm.cloud_vlm import CloudVlm
        return CloudVlm(config)
    from kitchenvision.vlm.local_vlm import LocalVlm
    return LocalVlm(config)
