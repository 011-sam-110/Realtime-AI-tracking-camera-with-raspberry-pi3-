"""Local (on-device CUDA) VisionModel backend (INTERFACES.md §9).

`LocalVlm` runs a small vision-language model locally via 🤗 transformers on CUDA, giving a private,
quota-free "image + text prompt -> text" generator that is interchangeable with `CloudVlm` behind
the `vlm.base.VisionModel` contract.

Config: `config["vlm"]["local_model"]` selects the model (one of):
    "qwen2-vl-2b" -> "Qwen/Qwen2-VL-2B-Instruct"
    "moondream2"  -> "vikhyatk/moondream2"
    "florence2"   -> "microsoft/Florence-2-base"
`config["vlm"]["device"]` (default "cuda") picks the device.

OFFLINE-SAFE CONSTRUCTION
-------------------------
`torch` + `transformers` are imported LAZILY inside `_load()`, which runs only on the FIRST
`generate()` call — NEVER in `__init__`. So importing this module and constructing `LocalVlm`
requires no torch, no CUDA, and no network. `available` checks merely whether the deps are
importable (via `importlib.util.find_spec`, which does NOT import them).

RIG SETUP (separate step)
-------------------------
The actual model download + the torch-CUDA install are a SEPARATE one-off step performed on the rig
(like the onnxruntime-CUDA fix), not by this code. This file only needs to exist and import cleanly
without those deps present; the model weights are fetched lazily by transformers on first use.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from kitchenvision.vlm.base import VlmError, VlmResult

# config alias -> Hugging Face model id.
_MODEL_MAP = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "moondream2": "vikhyatk/moondream2",
    "florence2": "microsoft/Florence-2-base",
}
_DEFAULT_ALIAS = "qwen2-vl-2b"


class LocalVlm:
    """Small CUDA VLM via transformers. Implements `vlm.base.VisionModel`. Lazy model load."""

    def __init__(self, config: dict) -> None:
        self.config = config
        vlm = (config.get("vlm", {}) or {})
        alias = (vlm.get("local_model") or _DEFAULT_ALIAS)
        self.model_id: str = _MODEL_MAP.get(alias, _MODEL_MAP[_DEFAULT_ALIAS])
        self.device: str = (vlm.get("device") or "cuda")
        # Lazily-loaded handles — populated by _load() on the first generate().
        self._model = None
        self._processor = None
        self._loaded = False

    # ------------------------------------------------------------------ availability
    @property
    def available(self) -> bool:
        """True iff torch AND transformers are importable — checked WITHOUT importing them.

        `find_spec` only inspects the import machinery (cheap, offline); it does not execute the
        package, so this never loads CUDA or pulls a model.
        """
        try:
            return (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None
            )
        except (ImportError, ValueError):
            return False

    # ------------------------------------------------------------------ model load (lazy)
    def _load(self) -> None:
        """Load torch + transformers and the chosen model/processor ONCE (first generate()).

        Raises `VlmError` on any import / load failure (caller surfaces it via the worker status).
        """
        if self._loaded:
            return
        try:
            import torch  # noqa: F401  (lazy heavy import)
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = self._load_model().to(self.device)
            self._model.eval()
            self._loaded = True
            print(f"[vlm] loaded {self.model_id} on {self.device} "
                  f"({type(self._model).__name__})", flush=True)
        except Exception as e:
            self._model = None
            self._processor = None
            self._loaded = False
            print(f"[vlm] FAILED to load {self.model_id}: {e}", flush=True)
            raise VlmError(f"could not load local VLM '{self.model_id}': {e}") from e

    def _load_model(self):
        """Load the weights with the CORRECT class for the model.

        Vision-language models like Qwen2-VL are image-text-to-text / conditional-generation models,
        NOT plain causal LMs — `AutoModelForCausalLM` raises on them. Qwen2-VL gets its dedicated
        class; everything else tries the modern VLM auto-classes, falling back to causal-LM last.
        """
        mid = self.model_id.lower()
        if "qwen2-vl" in mid or "qwen2vl" in mid:
            from transformers import Qwen2VLForConditionalGeneration
            return Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", trust_remote_code=True
            )
        import transformers
        last_err = None
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(
                    self.model_id, torch_dtype="auto", trust_remote_code=True
                )
            except Exception as e:  # try the next class
                last_err = e
        raise VlmError(f"no compatible model class for '{self.model_id}': {last_err}")

    # ------------------------------------------------------------------ generate
    def generate(
        self, image_bgr: np.ndarray, prompt: str, max_tokens: int = 300
    ) -> VlmResult:
        """Run image+prompt -> text on the local model; provider = the HF model id.

        Loads the model on the first call (lazy). Raises `VlmError` on load/inference failure.
        """
        self._load()
        try:
            image = self._to_pil(image_bgr)
            text = self._run_inference(image, prompt, max_tokens)
        except VlmError:
            raise
        except Exception as e:
            raise VlmError(f"local VLM inference failed: {e}") from e
        return VlmResult(text=text, provider=self.model_id)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_pil(image_bgr: np.ndarray):
        """Convert a BGR numpy frame to an RGB PIL image (lazy cv2 + PIL imports)."""
        if image_bgr is None:
            raise VlmError("no frame to run the local VLM on")
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _run_inference(self, image, prompt: str, max_tokens: int) -> str:
        """Generic transformers VLM inference: build chat -> processor -> generate -> decode.

        Uses the chat-template path common to Qwen2-VL-style processors, with a flat fallback for
        processors (Florence-2 / Moondream) that don't expose `apply_chat_template`.
        """
        import torch

        processor = self._processor
        model = self._model

        # Build the model input. Prefer the chat-template path (Qwen2-VL etc.).
        text_prompt = prompt
        apply_tmpl = getattr(processor, "apply_chat_template", None)
        if callable(apply_tmpl):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                text_prompt = apply_tmpl(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text_prompt = prompt

        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)

        # Strip the prompt tokens so we decode only the freshly generated continuation.
        try:
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if input_ids is not None:
                trimmed = [
                    out[len(inp):] for inp, out in zip(input_ids, generated)
                ]
            else:
                trimmed = generated
        except Exception:
            trimmed = generated

        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = decoded[0] if decoded else ""
        return text.strip()
