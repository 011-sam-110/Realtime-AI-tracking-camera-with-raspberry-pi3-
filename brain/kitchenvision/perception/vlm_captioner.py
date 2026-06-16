"""VLM activity captioner — the v1 `PerceptionSource` (INTERFACES.md §8).

`VlmCaptioner` turns a raw BGR frame + the present recognised people into structured `Event`s.
It ports the proven prompt + robust JSON-extraction from the old `station/activity.py`, but is
now decoupled from the network/threading: the actual "image + prompt -> text" call is delegated
to a `VisionModel` (`vlm/`), and the cadence/threading/DB writes live in `perception/worker.py`.

Flow of `observe(frame_bgr, people)`:

  1. filter `people` down to RECOGNISED, matchable entries (person_id >= 0, has label + box),
  2. draw each one's label+box on a *copy* of the frame (so the VLM can tie a phrase to a face),
  3. build the labelled prompt and call `self.vision.generate(annotated, prompt, max_tokens)`,
  4. parse the model's text into a `{label: phrase}` JSON mapping (fence-strip + brace-scan),
  5. match each label back to a person (tolerant of case/whitespace) and emit ONE `Event` per
     matched person: `type="activity"`, `text=phrase`, `person_id=...`, `source="vlm"`,
     `ts=now`, `confidence=...`, plus a LIGHT keyword guess of `action`/`object`/`location`
     (small maps only — full structured extraction is a later `LocalCvScene` job).

Errors: the typed VLM errors (`NoVisionError`, `RateLimitError`) are RE-RAISED so the worker can
map them to `STATE.activity_status`. Any other failure (encode/parse/transient) yields `[]` —
`observe` never raises fatally for those. Returns `[]` when nobody is present.

Module import + construction are offline-safe: no model load, no network here.
"""
from __future__ import annotations

import json
import re
import time

import cv2
import numpy as np

from kitchenvision.core.types import Event
from kitchenvision.vlm.base import NoVisionError, RateLimitError, VisionModel

# JPEG quality is irrelevant here (we hand the VLM a numpy frame, not bytes), but the annotation
# style is ported verbatim from the old worker so the model sees the same legible labels.
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prompt: list the present labels and demand a strict JSON object {label -> activity phrase}.
# Ported verbatim from station/activity.py (proven against the FreeLLMAPI gateway).
_PROMPT_TEMPLATE = (
    "You are watching a single still frame from a kitchen webcam. The following people are "
    "visible, each drawn with a labelled bounding box: {names}.\n"
    "For EACH listed label, describe in 3 to 6 words what that person appears to be doing "
    "right now (e.g. \"making a cup of tea\", \"looking at phone\", \"standing talking\"). "
    "If you cannot tell, say \"present\".\n"
    "Respond with ONLY a single minified JSON object mapping each EXACT label string to its "
    "activity phrase string. No markdown, no code fences, no extra text. "
    "Example: {{\"Alice\": \"chopping vegetables\", \"Unknown #2\": \"looking at phone\"}}"
)

# --------------------------------------------------------------------------------------
# LIGHT keyword maps for action/object/location. Deliberately small — this is a best-effort
# tag, NOT real scene understanding (that is a later LocalCvScene job). First match wins.
# --------------------------------------------------------------------------------------
_ACTION_KEYWORDS: "tuple[tuple[str, str], ...]" = (
    ("washing", "washing"),
    ("wash", "washing"),
    ("cleaning", "cleaning"),
    ("clean", "cleaning"),
    ("wiping", "cleaning"),
    ("drying", "drying"),
    ("cooking", "cooking"),
    ("cook", "cooking"),
    ("frying", "cooking"),
    ("boiling", "cooking"),
    ("stirring", "cooking"),
    ("chopping", "chopping"),
    ("cutting", "chopping"),
    ("slicing", "chopping"),
    ("making", "making"),
    ("preparing", "making"),
    ("pouring", "pouring"),
    ("drinking", "drinking"),
    ("eating", "eating"),
    ("opening", "opening"),
    ("closing", "closing"),
    ("reading", "reading"),
    ("looking", "looking"),
    ("texting", "using phone"),
    ("phone", "using phone"),
    ("talking", "talking"),
    ("standing", "standing"),
    ("sitting", "sitting"),
    ("walking", "walking"),
    ("leaving", "leaving"),
    ("left", "left"),
    ("entering", "entering"),
    ("carrying", "carrying"),
    ("holding", "holding"),
    ("loading", "loading"),
    ("unloading", "unloading"),
)

_OBJECT_KEYWORDS: "tuple[str, ...]" = (
    "tea", "coffee", "cup", "mug", "kettle", "plate", "plates", "bowl", "bowls",
    "pan", "pot", "knife", "fork", "spoon", "glass", "bottle", "phone", "laptop",
    "dish", "dishes", "cutlery", "food", "vegetables", "fruit", "bread", "sandwich",
    "fridge", "oven", "microwave", "toaster", "dishwasher", "sink", "tap", "towel",
    "cupboard", "drawer",
)

_LOCATION_KEYWORDS: "tuple[str, ...]" = (
    "sink", "table", "counter", "countertop", "worktop", "stove", "hob", "oven",
    "fridge", "cupboard", "dishwasher", "microwave", "window", "door", "floor",
)


class VlmCaptioner:
    """v1 `PerceptionSource`: one VLM call per cycle -> per-person activity `Event`s."""

    def __init__(self, config: dict, vision: VisionModel) -> None:
        self.config = config
        self.vision = vision

        vcfg = config.get("vlm", {}) or {}
        act = config.get("activity", {}) or {}
        # max_tokens for the caption call (vlm block owns the real default of 300).
        self.max_tokens = int(vcfg.get("max_tokens", 300) or 300)
        # Floor on per-event confidence the worker may later gate on (kept on the Event).
        self.min_confidence = float(act.get("min_confidence", 0.0) or 0.0)
        # Webcam enhancement applied to the frame the VLM sees: a backlit, dim webcam image makes a
        # small VLM default to "present"; brightening + local contrast lets it actually read the
        # scene (the cup, the phone, etc.). Uses the full display chain.
        try:
            from kitchenvision.capture.enhance import make_enhancer
            self.enhancer = make_enhancer(config)
        except Exception:
            self.enhancer = None

    # ------------------------------------------------------------------ public API
    def observe(self, frame: np.ndarray, people: list[dict]) -> list[Event]:
        """Caption the present recognised people in `frame`; return one Event per matched person.

        Returns `[]` when `frame` is None or nobody recognised is present. Re-raises the typed
        VLM errors (`NoVisionError`/`RateLimitError`) so the worker can set status; swallows any
        other failure (encode/parse/transient) and returns `[]`.
        """
        if frame is None:
            return []
        present = self._present_people(people)
        if not present:
            return []

        src = frame
        if self.enhancer is not None:
            try:
                src = self.enhancer.apply(frame, for_display=True)
            except Exception:
                src = frame
        annotated = self._annotate(src, present)
        if annotated is None:
            return []

        labels = [p["label"] for p in present]
        prompt = _PROMPT_TEMPLATE.format(names=", ".join(labels))

        try:
            result = self.vision.generate(annotated, prompt, self.max_tokens)
        except (NoVisionError, RateLimitError):
            # Health-significant: let the worker map these to activity_status.
            raise
        except Exception:
            # Transient / unexpected — never raise fatally from a PerceptionSource.
            return []

        text = getattr(result, "text", "") or ""
        mapping = self._extract_json_mapping(text)
        if not mapping:
            return []

        return self._mapping_to_events(mapping, present)

    # ------------------------------------------------------------------ people filter
    @staticmethod
    def _present_people(people: list[dict]) -> list[dict]:
        """Recognised, matchable people only: person_id >= 0 with a usable label + box.

        Returns light dicts `{person_id, label, box}` (box as a list of ints, len 4).
        """
        out: list[dict] = []
        for p in people or []:
            try:
                pid = int(p.get("person_id", -1))
            except (TypeError, ValueError):
                continue
            if pid < 0:
                continue
            label = p.get("label")
            box = p.get("box")
            if not label or not box:
                continue
            try:
                b = [int(v) for v in list(box)[:4]]
            except (TypeError, ValueError):
                continue
            if len(b) != 4:
                continue
            out.append({"person_id": pid, "label": str(label), "box": b})
        return out

    # ------------------------------------------------------------------ annotation
    @staticmethod
    def _annotate(frame: np.ndarray, present: list[dict]) -> "np.ndarray | None":
        """Draw each present person's label + box on a COPY of `frame`. None on failure.

        Ported from the old worker's `_encode_annotated` drawing (sans the JPEG/base64 step —
        the VisionModel takes the BGR frame directly).
        """
        try:
            img = np.array(frame, copy=True)
            h, w = img.shape[:2]
            for p in present:
                x, y, bw, bh = (int(v) for v in p["box"][:4])
                x2, y2 = x + bw, y + bh
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                label = p["label"]
                ty = y - 6 if y - 6 > 10 else min(h - 4, y + bh + 16)
                tx = max(0, min(x, w - 1))
                (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
                cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
                cv2.putText(
                    img, label, (tx + 2, ty), _FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA
                )
            return img
        except Exception:
            return None

    # ------------------------------------------------------------------ mapping -> Events
    def _mapping_to_events(self, mapping: dict, present: list[dict]) -> list[Event]:
        """Match each JSON label back to a person_id and build one Event per matched person.

        Matching tolerates exact label, case-insensitive, and whitespace differences. Last
        write wins if the model repeats a label. `confidence` defaults to 1.0 but is floored
        at `min_confidence` (so a gate downstream sees a consistent value).
        """
        by_label = {p["label"]: p["person_id"] for p in present}
        by_norm = {self._norm(p["label"]): p["person_id"] for p in present}

        applied: "dict[int, str]" = {}  # person_id -> phrase
        for raw_label, phrase in mapping.items():
            if not isinstance(raw_label, str):
                continue
            text = self._clean_phrase(phrase)
            if not text:
                continue
            pid = by_label.get(raw_label)
            if pid is None:
                pid = by_norm.get(self._norm(raw_label))
            if pid is None:
                continue
            applied[pid] = text

        now = time.time()
        # The VLM gives us no per-caption confidence, so report a full 1.0 (a successful caption
        # the worker may still gate against activity.min_confidence before persisting).
        confidence = 1.0
        events: list[Event] = []
        for pid, phrase in applied.items():
            action, obj, location = self._extract_keywords(phrase)
            events.append(
                Event(
                    type="activity",
                    ts=now,
                    person_id=pid,
                    action=action,
                    object=obj,
                    location=location,
                    text=phrase,
                    confidence=confidence,
                    source="vlm",
                )
            )
        return events

    # ------------------------------------------------------------------ light keyword extraction
    @staticmethod
    def _extract_keywords(phrase: str) -> "tuple[str | None, str | None, str | None]":
        """Best-effort (action, object, location) guess from a caption phrase.

        Small keyword maps only — first match wins. Returns (None, None, None) when nothing
        matches. This is intentionally shallow; real structured extraction is a later job.
        """
        words = re.findall(r"[a-z']+", phrase.lower())
        wordset = set(words)

        action: "str | None" = None
        for kw, canon in _ACTION_KEYWORDS:
            if kw in wordset:
                action = canon
                break

        obj: "str | None" = None
        for kw in _OBJECT_KEYWORDS:
            if kw in wordset:
                obj = kw
                break

        location: "str | None" = None
        for kw in _LOCATION_KEYWORDS:
            if kw in wordset:
                location = kw
                break

        return action, obj, location

    # ------------------------------------------------------------------ JSON extraction (ported)
    @staticmethod
    def _extract_json_mapping(text: str) -> dict:
        """Robustly extract a {label: phrase} mapping from the model's text.

        Strip code fences, try a direct json.loads, then fall back to grabbing the outermost
        balanced {...} via a brace scan and json.loads that. Returns {} on failure. Ported from
        the old worker.
        """
        if not text:
            return {}
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s).strip()

        for candidate in (s, VlmCaptioner._first_brace_block(s)):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _first_brace_block(s: str) -> "str | None":
        """Return the first balanced {...} substring of `s`, or None.

        Brace- and string-aware (a `}` inside a JSON string value does not end the block early).
        """
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    # ------------------------------------------------------------------ phrase / label helpers
    @staticmethod
    def _clean_phrase(phrase) -> str:
        """Coerce a JSON value to a short clean activity string ('' to skip)."""
        if phrase is None:
            return ""
        if not isinstance(phrase, str):
            phrase = str(phrase)
        phrase = " ".join(phrase.split()).strip().strip('"').strip()
        if len(phrase) > 120:
            phrase = phrase[:117].rstrip() + "..."
        return phrase

    @staticmethod
    def _norm(label) -> str:
        """Normalise a label for tolerant matching (case/space-insensitive)."""
        return " ".join(str(label).split()).strip().lower()
