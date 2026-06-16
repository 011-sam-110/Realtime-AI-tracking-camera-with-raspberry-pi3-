"""InsightFace recognition on the GPU (INTERFACES.md §6).

`InsightFaceGpu` turns a raw BGR frame into a list of `Detection`s in ONE `app.get(frame)` pass
(RetinaFace detection + 512-d ArcFace embeddings + gender/age), then matches each face by cosine
similarity against per-person **templates** held in memory.

PHASE E HARDENING (improve recognition as much as possible on the cheap webcam)
------------------------------------------------------------------------------
  * **Image enhancement** — a gentle, embedding-safe `Enhancer` (auto-gamma + CLAHE) cleans the
    frame before `app.get()`, so detection + embeddings see a brighter, higher-contrast image.
  * **Multi-template identity** — each person is represented by SEVERAL exemplar vectors (its
    centroid + its best stored embeddings), not one mean. A probe matches the BEST exemplar over all
    of a person's templates → far fewer fragmented "Unknown #N" clusters under pose/lighting change.
  * **Multi-frame fusion** — the probe is the quality-weighted mean of the track's last few
    best-shot embeddings (`fusion.fused_embedding`), averaging out webcam noise / motion blur.
  * **Margin / ambiguity guards** — a match enrols (mutates the identity model) only when it also
    beats the best OTHER person by `recog_margin`; a brand-new "Unknown" is created only when the
    best similarity is clearly below threshold (`unknown_min_margin`). This stops two people merging
    and stops near-duplicate unknowns spawning.

Match outcomes per face:
  * confident match (>= `recog_threshold` AND margin >= `recog_margin`) → existing person: log a
    sighting + bump last_seen; on a good-quality frame also append the embedding, nudge the centroid,
    add an in-memory template, save a crop.
  * ambiguous match (>= threshold but margin too small) → attribute to top-1 for display + sighting,
    but DO NOT mutate the identity model (avoid drift / merges).
  * clear non-match on a GOOD-quality frame → a brand-new "Unknown #N".
  * otherwise (gray-zone / low quality) → tracked-only (a box so the servo keeps framing it).

THREADING
---------
`recognize()` / `refresh()` are called only from the single recognition thread. A small lock guards
ONLY the in-memory template matrix so `refresh()` (after a rename / merge / delete) can never tear a
match mid-read. All DB writes go through `store.db` (each opens its own short-lived connection —
connections never cross threads). `insightface` is imported LAZILY in `__init__` so this module
imports cheaply and offline.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2  # noqa: F401  (used by _maybe_save_crop; kept module-level like the rest of the package)
import numpy as np

from kitchenvision.core.types import Detection
from kitchenvision.recognition import quality as q
from kitchenvision.recognition.fusion import Fusion
from kitchenvision.store import db

# Canonical working resolution (the Pi feed is 640x480; the pipeline resizes to this).
W, H = 640, 480

# ArcFace embedding dimensionality (buffalo_l / w600k_r50).
EMB_DIM = 512


def _normalize(vec) -> np.ndarray:
    """L2-normalise a 1-D vector to float32 (zero vectors returned unchanged). PORT."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n > 0.0:
        v = v / n
    return v.astype(np.float32)


def _clamp_box(x1: float, y1: float, x2: float, y2: float) -> list[int]:
    """Clamp a float `[x1,y1,x2,y2]` box to the frame; return `[x, y, w, h]` ints. PORT."""
    x1 = int(max(0, min(W, round(float(x1)))))
    y1 = int(max(0, min(H, round(float(y1)))))
    x2 = int(max(0, min(W, round(float(x2)))))
    y2 = int(max(0, min(H, round(float(y2)))))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return [x1, y1, w, h]


def _preload_cuda_dlls(providers) -> None:
    """Make the onnxruntime CUDA EP loadable from the `nvidia-*-cu12` wheels.

    onnxruntime-gpu ships the CUDA provider DLL but NOT its CUDA 12 / cuDNN 9 dependencies (cuBLAS,
    cuDNN, cuFFT, cudart). Those come from the `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` /
    `nvidia-cudnn-cu12` / `nvidia-cufft-cu12` wheels, which install under `site-packages/nvidia/*/bin`
    — a path Windows does not search by default, so without this the CUDA EP fails to load
    (`cublasLt64_12.dll ... missing`) and onnxruntime silently falls back to CPU. `preload_dlls()`
    (onnxruntime 1.20+) loads them from those wheels so `CUDAExecutionProvider` can initialise.
    Best-effort: a no-op if CUDA isn't requested or the helper/wheels are absent.
    """
    if not any("CUDA" in str(p) for p in (providers or [])):
        return
    import os
    # cuDNN 9 is split into many DLLs and loads its engine sub-libraries (cudnn_engines_*,
    # cudnn_cnn_*, ...) BY NAME at inference time; cuBLAS / cuFFT / nvrtc are dependencies too. The
    # nvidia-*-cu12 wheels put these under site-packages/nvidia/<comp>/bin, which Windows does not
    # search by default — so add every such dir to BOTH the DLL search path and PATH. Without this
    # the top-level CUDA EP loads but the first Conv fails (cudnn_engines_*_9.dll not found /
    # CUDNN_FE_API_FAILED) and onnxruntime silently falls back to CPU.
    try:
        import nvidia
        roots = list(getattr(nvidia, "__path__", []))
    except Exception:
        roots = []
    for root in roots:
        try:
            comps = os.listdir(root)
        except OSError:
            continue
        for comp in comps:
            bindir = os.path.join(root, comp, "bin")
            if os.path.isdir(bindir):
                try:
                    os.add_dll_directory(bindir)
                except (OSError, AttributeError):
                    pass
                if bindir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass  # CPU fallback still works; never block construction on this


class InsightFaceGpu:
    """GPU InsightFace recognition with quality-gated multi-template clustering + best-shot fusion."""

    def __init__(self, config: dict) -> None:
        self.config = config
        rec = (config.get("recognition", {}) or {})
        self.threshold = float(rec.get("recog_threshold", 0.45))
        self.centroid_alpha = float(rec.get("centroid_alpha", 0.9))
        # Phase E hardening knobs.
        self.templates_per_person = max(1, int(rec.get("templates_per_person", 4)))
        self.match_fused = bool(rec.get("match_fused", True))
        self.recog_margin = float(rec.get("recog_margin", 0.04))
        self.unknown_min_margin = float(rec.get("unknown_min_margin", 0.10))

        # In-memory template store: person_id -> list[normalised f32 vec] (centroid + exemplars),
        # plus the derived stacked matrix used for vectorised cosine matching.
        #   _matrix  : (N_templates, EMB_DIM) L2-normalised rows
        #   _row_pid : (N_templates,) person id for each row
        self._lock = threading.Lock()
        self._templates: "dict[int, list[np.ndarray]]" = {}
        self._matrix: np.ndarray = np.zeros((0, EMB_DIM), dtype=np.float32)
        self._row_pid: np.ndarray = np.zeros((0,), dtype=np.int64)

        # Identity persistence: IoU tracker + best-shot buffer (pure geometry, no GPU).
        self.fusion = Fusion(config)

        # Webcam image-quality optimisation (embedding-safe subset before app.get()).
        try:
            from kitchenvision.capture.enhance import make_enhancer  # lazy: cv2-only, offline-safe
            self.enhancer = make_enhancer(config)
        except Exception:
            self.enhancer = None

        # --- build the InsightFace app (lazy import so module import stays cheap/offline) ---
        model = str(rec.get("model", "buffalo_l"))
        ds = int(rec.get("det_size", 640))
        det_thresh = float(rec.get("det_thresh", 0.6))
        providers = list(rec.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
        # Load CUDA 12 / cuDNN 9 from the nvidia wheels BEFORE any onnxruntime session is created,
        # so the CUDA EP initialises instead of silently falling back to CPU (see _preload_cuda_dlls).
        _preload_cuda_dlls(providers)
        from insightface.app import FaceAnalysis  # noqa: PLC0415  (intentional lazy import)
        # buffalo_l ships five ONNX models; we only read detection bbox, ArcFace embedding, and
        # age/sex, so the two landmark models are excluded — less GPU per pass, no quality loss.
        allowed = list(rec.get("insightface_modules", ["detection", "recognition", "genderage"]))
        # HEURISTIC cuDNN conv-algo search (vs onnxruntime's EXHAUSTIVE default) cuts first-run
        # session init/autotune from ~60 s to a few seconds with negligible inference cost for these
        # small 640x480 models. provider_options is aligned element-wise to `providers`.
        provider_options = [
            {"cudnn_conv_algo_search": "HEURISTIC"} if "CUDA" in str(p) else {}
            for p in providers
        ]
        kwargs: dict = {"name": model, "providers": providers, "provider_options": provider_options}
        if allowed:
            kwargs["allowed_modules"] = allowed
        try:
            self.app = FaceAnalysis(**kwargs)
        except TypeError:
            # Older InsightFace may not forward provider_options — fall back without it.
            kwargs.pop("provider_options", None)
            self.app = FaceAnalysis(**kwargs)
        # ctx_id=0 selects GPU 0 for the CUDA provider (CPU provider ignores it). Raising det_thresh
        # above InsightFace's 0.5 default keeps the worst motion-blur smears from detecting at all.
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(ds, ds))

        # Warm the template cache so the first frame can already recognise enrolled people.
        self._load_templates()

    # -------------------------------------------------------------- templates
    def _rebuild_locked(self) -> None:
        """Rebuild `_matrix` / `_row_pid` from `_templates`. CALLER MUST HOLD `self._lock`."""
        rows: "list[np.ndarray]" = []
        ids: "list[int]" = []
        for pid, vecs in self._templates.items():
            for v in vecs:
                rows.append(v)
                ids.append(int(pid))
        if rows:
            self._matrix = np.stack(rows, axis=0).astype(np.float32)
            self._row_pid = np.asarray(ids, dtype=np.int64)
        else:
            self._matrix = np.zeros((0, EMB_DIM), dtype=np.float32)
            self._row_pid = np.zeros((0,), dtype=np.int64)

    def _load_templates(self) -> None:
        """(Re)build the in-memory templates from the DB (under the lock)."""
        if self.templates_per_person <= 1:
            rows = db.load_centroids()  # [{id, name, kind, centroid}]
            templates = {int(r["id"]): [_normalize(r["centroid"])] for r in rows}
        else:
            rows = db.load_templates(self.templates_per_person)  # [{id, name, kind, vec}]
            templates = {}
            for r in rows:
                templates.setdefault(int(r["id"]), []).append(_normalize(r["vec"]))
        with self._lock:
            self._templates = templates
            self._rebuild_locked()

    def refresh(self) -> None:
        """Reload templates after a rename / merge / delete so `recognize()` reflects it."""
        self._load_templates()

    def _add_template(self, person_id: int, emb_normed: np.ndarray) -> None:
        """Add a fresh exemplar for a person in memory, capping at `templates_per_person` (FIFO).

        Keeps a new person matchable on the very next frame (no DB reload needed) and lets an
        existing identity adapt within a session, bounded so the matrix can't grow without limit.
        """
        v = _normalize(emb_normed)
        with self._lock:
            lst = self._templates.setdefault(int(person_id), [])
            lst.append(v)
            if len(lst) > self.templates_per_person:
                del lst[0]  # FIFO: keep the most recent exemplars
            self._rebuild_locked()

    def _match(self, probe: np.ndarray) -> "tuple[Optional[int], float, float]":
        """Best (person_id, cosine_score, margin) for a normalised probe.

        `score` is the best cosine similarity over ALL templates; `margin` is the gap between that
        and the best similarity to any DIFFERENT person (0.0 if only one person is enrolled). The
        margin is what tells a confident identification from an ambiguous one. Returns
        `(None, 0.0, 0.0)` when nothing is enrolled.
        """
        with self._lock:
            if self._matrix.shape[0] == 0:
                return None, 0.0, 0.0
            sims = self._matrix @ probe  # all unit-norm → cosine similarity per template row
            j = int(np.argmax(sims))
            best_pid = int(self._row_pid[j])
            best = float(sims[j])
            other = sims[self._row_pid != best_pid]
            second = float(other.max()) if other.size else 0.0
            return best_pid, best, best - second

    # ------------------------------------------------------------------ crops
    def _maybe_save_crop(self, person_id: int, frame_bgr: np.ndarray, box: list[int]) -> None:
        """Best-effort: save a face crop if this person has fewer than MAX_CROPS already. PORT."""
        try:
            if len(db.list_crops(person_id)) >= db.MAX_CROPS:
                return
            x, y, w, h = box
            crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                return
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                db.save_crop(person_id, jpg.tobytes())
        except Exception:
            pass  # a crop must never break the pipeline

    # -------------------------------------------------------------- recognise
    def recognize(self, frame: np.ndarray) -> list[Detection]:
        """Detect + identify faces in a BGR frame → one `Detection` per face.

        The frame is enhanced (embedding-safe subset) before detection; the ORIGINAL frame is used
        for the quality gate + saved crops so thresholds tuned on raw pixels still hold and the
        gallery stays truthful.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        proc = frame
        if self.enhancer is not None and getattr(self.enhancer, "for_recognition", False):
            try:
                proc = self.enhancer.apply(frame, for_display=False)
            except Exception:
                proc = frame

        try:
            faces = self.app.get(proc)
        except Exception:
            return []
        if not faces:
            self.fusion.begin_frame()
            self.fusion.end_frame()  # age out tracks even on an empty frame
            return []

        now = time.time()
        self.fusion.begin_frame()
        results: list[Detection] = []
        for f in faces:
            try:
                det = self._process_face(f, frame, now)
            except Exception:
                # One malformed face must not abort the whole frame.
                continue
            if det is not None:
                results.append(det)
        self.fusion.end_frame()
        return results

    def _detect_only_result(
        self,
        tid: int,
        box: list[int],
        age: Optional[float],
        sex: Optional[str],
        score: float,
    ) -> Detection:
        """A tracked-but-not-enrolled face: keeps the servo framing it, spawns no cluster.

        If the track was previously recognised, keep showing that identity rather than "face".
        Assumes the caller has already `assign`/`observe`d the track this frame.
        """
        pid = self.fusion.person_of(tid)
        if pid >= 0:
            person = db.get_person(pid)
            if person is not None:
                return Detection(
                    box=box, person_id=int(pid), label=person["label"], kind=person["kind"],
                    track_id=tid, age=age, sex=sex, score=float(score),
                    quality=self.fusion.quality_of(tid),
                )
        return Detection(
            box=box, person_id=-1, label="face", kind="unknown",
            track_id=tid, age=age, sex=sex, score=float(score),
            quality=self.fusion.quality_of(tid),
        )

    def _result_for(
        self, pid: int, box: list[int], tid: int,
        age: Optional[float], sex: Optional[str], score: float,
    ) -> Detection:
        """Build the Detection for a resolved person id (label/kind from the DB)."""
        person = db.get_person(pid)
        if person is not None:
            label, kind = person["label"], person["kind"]
        else:
            label, kind = f"Unknown #{pid}", "unknown"
        return Detection(
            box=box, person_id=int(pid), label=label, kind=kind, track_id=tid,
            age=age, sex=sex, score=float(score), quality=self.fusion.quality_of(tid),
        )

    def _process_face(self, f, frame_bgr: np.ndarray, now: float) -> Optional[Detection]:
        bbox = getattr(f, "bbox", None)
        if bbox is None:
            return None
        box = _clamp_box(bbox[0], bbox[1], bbox[2], bbox[3])

        # age/sex from the genderage model; guarded so a trimmed module set degrades to None.
        try:
            age = None if f.age is None else float(f.age)
        except (KeyError, AttributeError, TypeError):
            age = None
        try:
            sex = f.sex  # InsightFace property: 'M' / 'F' / None
        except (KeyError, AttributeError):
            sex = None

        det_score = float(getattr(f, "det_score", 1.0) or 0.0)
        quality_val = q.quality_score(det_score, box, frame_bgr)

        # Stable track id every face (IoU vs previous frame), regardless of recognition outcome.
        tid = self.fusion.assign(box)

        emb_raw = getattr(f, "normed_embedding", None)  # already L2-normalised, or None (no recog)
        if emb_raw is None:
            self.fusion.observe(tid, quality_val, box, embedding=None, person_id=-1)
            return self._detect_only_result(tid, box, age, sex, 0.0)
        emb = _normalize(np.asarray(emb_raw, dtype=np.float32).ravel())

        # Seed the best-shot window with THIS embedding so the fused probe includes it, then match
        # the multi-frame fused embedding (steadier than any single noisy frame).
        self.fusion.observe(tid, quality_val, box, embedding=emb, person_id=-1)
        probe = self.fusion.fused_embedding(tid) if self.match_fused else None
        if probe is None:
            probe = emb

        quality_ok = q.passes_gate(det_score, box, frame_bgr, self.config)
        pid, score, margin = self._match(probe)

        confident = pid is not None and score >= self.threshold and margin >= self.recog_margin
        ambiguous = pid is not None and score >= self.threshold and margin < self.recog_margin

        if confident:
            assert pid is not None
            # Always log the sighting + bump last_seen so a known person stays "live" even on a
            # blurry frame; only mutate the identity model when the face is good quality.
            db.set_last_seen(pid, now)
            db.add_sighting(pid, now, tuple(box))
            if quality_ok:
                db.add_embedding(pid, emb, quality_val)
                old = db.get_centroid(pid)
                if old is not None:
                    new_centroid = _normalize(
                        self.centroid_alpha * old + (1.0 - self.centroid_alpha) * emb
                    )
                    db.update_centroid(pid, new_centroid)
                self._add_template(pid, emb)
                self._maybe_save_crop(pid, frame_bgr, box)
            self.fusion.set_person(tid, pid)
            return self._result_for(pid, box, tid, age, sex, score)

        if ambiguous:
            assert pid is not None
            # Too close to a second person to safely enrol: show + log top-1, but DON'T mutate the
            # identity model (prevents two people slowly merging into one cluster).
            db.set_last_seen(pid, now)
            db.add_sighting(pid, now, tuple(box))
            self.fusion.set_person(tid, pid)
            return self._result_for(pid, box, tid, age, sex, score)

        if quality_ok and score < (self.threshold - self.unknown_min_margin):
            # Clear non-match on a good frame → a brand-new person.
            pid = db.create_person(centroid=emb, kind="unknown", age=age, sex=sex)
            db.add_embedding(pid, emb, quality_val)
            db.set_last_seen(pid, now)
            db.add_sighting(pid, now, tuple(box))
            self._add_template(pid, emb)
            self._maybe_save_crop(pid, frame_bgr, box)
            self.fusion.set_person(tid, pid)
            return self._result_for(pid, box, tid, age, sex, 1.0)

        # Gray-zone (near but below threshold) OR low quality with no confident match: track only,
        # so the servo keeps framing them but we don't spawn a near-duplicate unknown.
        return self._detect_only_result(tid, box, age, sex, score)
