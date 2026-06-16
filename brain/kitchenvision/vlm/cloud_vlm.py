"""Cloud (OpenAI-compatible) VisionModel backend (INTERFACES.md §9).

`CloudVlm` is a thin "image + text prompt -> text" generator over any OpenAI-compatible chat
endpoint (Sampo's FreeLLMAPI gateway, which routes to Gemini / Groq / ... behind one base_url).
It is the direct port of the proven OpenAI call from the old `station/activity.py`: JPEG + base64
the BGR frame into a `data:` URL, make ONE `chat.completions` call (via `with_raw_response` so we
can read the upstream routing header), and wrap the assistant text in a `VlmResult`.

The perception layer owns the prompt + JSON parsing; this class just generates text and maps errors
to the typed `vlm.base` exceptions the worker turns into `STATE.activity_status`:
  * `openai.UnprocessableEntityError` / a "no vision" message -> `NoVisionError`  (-> no_vision_model)
  * `openai.RateLimitError`                                   -> `RateLimitError`  (-> error + backoff)
  * anything else                                            -> `VlmError`        (-> error)

`openai` is imported LAZILY inside methods, so importing this module + constructing `CloudVlm` is
offline-safe (no network, no SDK import) — a missing `openai` only matters at the first `generate()`.
"""
from __future__ import annotations

import base64

import numpy as np

from kitchenvision.vlm.base import (
    NoVisionError,
    RateLimitError,
    VlmError,
    VlmResult,
)

# JPEG quality for the image we ship upstream (keep the payload small).
_JPEG_QUALITY = 80

# Header FreeLLMAPI sets to tell us which upstream provider actually served the request.
_PROVIDER_HEADER = "X-Routed-Via"


class CloudVlm:
    """OpenAI-compatible cloud VisionModel (FreeLLMAPI gateway). Implements `vlm.base.VisionModel`."""

    def __init__(self, config: dict) -> None:
        self.config = config
        vlm = (config.get("vlm", {}) or {})
        self.base_url: str = (vlm.get("base_url") or "").strip()
        self.api_key: str = (vlm.get("api_key") or "").strip()
        self.model: str = vlm.get("model", "auto") or "auto"
        # Built lazily on first use so construction stays offline-safe (no openai import here).
        self._client = None

    # ------------------------------------------------------------------ availability
    @property
    def available(self) -> bool:
        """True only when we have both a base_url and an api_key (else the worker -> 'disabled')."""
        return bool(self.base_url and self.api_key)

    # ------------------------------------------------------------------ client (lazy)
    def _get_client(self):
        """Build the OpenAI client once (lazy import of `openai`)."""
        if self._client is None:
            from openai import OpenAI  # lazy: keep module import offline-safe

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------ generate
    def generate(
        self, image_bgr: np.ndarray, prompt: str, max_tokens: int = 300
    ) -> VlmResult:
        """JPEG+base64 the BGR frame, make ONE chat call, return the text + routed provider.

        Raises `NoVisionError` (422 / no-vision), `RateLimitError` (429), or `VlmError` (else).
        """
        import openai  # lazy: error-type handles + the client constructor

        data_url = self._encode_data_url(image_bgr)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        try:
            client = self._get_client()
            raw = client.chat.completions.with_raw_response.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except openai.UnprocessableEntityError as e:
            # 422 => upstream has no vision-capable model to route to.
            raise NoVisionError(
                self._err_message(e, "Upstream has no vision model (422).")
            ) from e
        except openai.RateLimitError as e:
            raise RateLimitError(
                self._err_message(e, "Rate limited (429).")
            ) from e
        except openai.APIStatusError as e:
            # Some gateways report the no-vision condition with a non-422 status + a
            # recognisable message; treat that as terminal too.
            if self._looks_like_no_vision(e):
                raise NoVisionError(
                    self._err_message(e, "Upstream reports no vision model.")
                ) from e
            raise VlmError(self._err_message(e, "Cloud VLM API error.")) from e
        except (NoVisionError, RateLimitError, VlmError):
            raise
        except Exception as e:  # network / timeout / unexpected -> transient VlmError
            raise VlmError(f"Cloud VLM call failed: {e}") from e

        # Provider: prefer the upstream routing header, fall back to the model name.
        provider = self._provider_from_headers(raw)

        try:
            completion = raw.parse()
            text = self._extract_text(completion)
        except Exception as e:
            raise VlmError(f"Could not read cloud VLM response: {e}") from e

        return VlmResult(text=text, provider=provider)

    # ------------------------------------------------------------------ encoding
    @staticmethod
    def _encode_data_url(image_bgr: np.ndarray) -> str:
        """JPEG-encode a BGR frame and wrap it as a `data:image/jpeg;base64,...` URL."""
        import cv2  # lazy (cv2 is a heavy-ish optional dep; keep import out of module top)

        if image_bgr is None:
            raise VlmError("no frame to encode for the cloud VLM")
        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
        )
        if not ok:
            raise VlmError("could not JPEG-encode the frame for the cloud VLM")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _extract_text(completion) -> str:
        """Pull the assistant message text out of a parsed chat completion (str or parts list)."""
        try:
            choice = completion.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                    else:
                        t = getattr(part, "text", None)
                    if t:
                        parts.append(t)
                return "".join(parts)
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------ headers
    def _provider_from_headers(self, raw) -> str:
        """Read the X-Routed-Via header from a raw response, falling back to the model name."""
        try:
            headers = getattr(raw, "headers", None)
            if headers is not None:
                via = headers.get(_PROVIDER_HEADER)
                if via:
                    return str(via).strip()
        except Exception:
            pass
        return str(self.model)

    # ------------------------------------------------------------------ errors
    @staticmethod
    def _err_message(exc, default: str) -> str:
        """Best-effort human-readable detail from an OpenAI error."""
        try:
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                err = body.get("error")
                msg = body.get("message")
                if not msg and isinstance(err, dict):
                    msg = err.get("message")
                if msg:
                    return f"{default} ({msg})"
            msg = getattr(exc, "message", None)
            if msg:
                return f"{default} ({msg})"
        except Exception:
            pass
        return default

    @staticmethod
    def _looks_like_no_vision(exc) -> bool:
        """Heuristic: does a non-422 API error actually mean 'no vision model'?"""
        try:
            text = (
                str(getattr(exc, "message", "") or "")
                + " "
                + str(getattr(exc, "body", "") or "")
            ).lower()
            return (
                ("no vision" in text)
                or ("vision model" in text)
                or ("does not support" in text and "image" in text)
            )
        except Exception:
            return False
