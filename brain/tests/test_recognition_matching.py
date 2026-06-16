"""Offline unit tests for the Phase-E recognition matching core (INTERFACES.md §6).

`insightface_gpu` imports cheaply (cv2 + numpy; the `insightface` model is imported LAZILY in
`__init__`), so we can test the pure matching logic — `_match` (best person + margin),
`_add_template` (FIFO cap), `_rebuild_locked` — by constructing the object via `object.__new__`
WITHOUT loading the model, GPU, or DB. This proves the multi-template + margin behaviour that
hardens recognition on the low-res webcam.

Plain-python runnable (pytest may be absent):  python tests/test_recognition_matching.py
Run from the brain root (C:/Users/sampo/pi/brain) so `import kitchenvision...` resolves.
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.recognition.insightface_gpu import (  # noqa: E402
    EMB_DIM, InsightFaceGpu, _normalize,
)


# --------------------------------------------------------------------------- helpers
def _unit(seed: int, dim: int = EMB_DIM) -> np.ndarray:
    """A deterministic random unit vector (stand-in for an ArcFace embedding)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _near(v: np.ndarray, seed: int, cos: float = 0.98) -> np.ndarray:
    """A unit vector with EXACT cosine similarity `cos` to `v` — two easily-confused faces.

    Builds an orthonormal companion direction so the result is `cos*v + sqrt(1-cos^2)*d`, giving a
    precise similarity independent of dimensionality (a naive `v + eps*randn` perturbation has norm
    that grows with sqrt(dim), so it would NOT stay close in 512-D).
    """
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(v.shape).astype(np.float32)
    d = d - float(d @ v) * v                 # orthogonalise against v
    d = d / np.linalg.norm(d)
    out = cos * v + np.sqrt(max(0.0, 1.0 - cos * cos)) * d
    return _normalize(out)


def _make(templates: dict, **kw) -> InsightFaceGpu:
    """Build an InsightFaceGpu with ONLY the matching state populated (no model/DB/GPU)."""
    eng = object.__new__(InsightFaceGpu)
    eng._lock = threading.Lock()
    eng.templates_per_person = int(kw.get("templates_per_person", 4))
    eng.threshold = float(kw.get("threshold", 0.45))
    eng.recog_margin = float(kw.get("recog_margin", 0.04))
    eng.unknown_min_margin = float(kw.get("unknown_min_margin", 0.10))
    eng._templates = {int(p): [_normalize(v) for v in vs] for p, vs in templates.items()}
    eng._matrix = np.zeros((0, EMB_DIM), dtype=np.float32)
    eng._row_pid = np.zeros((0,), dtype=np.int64)
    with eng._lock:
        eng._rebuild_locked()
    return eng


# --------------------------------------------------------------------------- tests
def test_match_empty() -> None:
    """No enrolled templates → (None, 0.0, 0.0); never raises."""
    eng = _make({})
    pid, score, margin = eng._match(_unit(1))
    assert pid is None and score == 0.0 and margin == 0.0
    print("  empty store: (None, 0.0, 0.0)")


def test_match_basic_and_margin() -> None:
    """A probe equal to A's template returns A with score ~1.0 and a large margin over B."""
    a, b = _unit(1), _unit(2)
    eng = _make({10: [a], 20: [b]})
    pid, score, margin = eng._match(a)
    assert pid == 10, f"probe==A should match person 10, got {pid}"
    assert score > 0.99, f"self-match score should be ~1.0, got {score:.3f}"
    # Two independent random 512-d unit vectors are near-orthogonal → margin close to score.
    assert margin > 0.5, f"distinct people should give a large margin, got {margin:.3f}"
    print(f"  basic: pid=10 score={score:.3f} margin={margin:.3f}")


def test_match_ambiguous_small_margin() -> None:
    """Two near-identical people produce a small margin (the ambiguity the engine refuses to enrol)."""
    a = _unit(1)
    b = _near(a, seed=99, cos=0.98)   # B is 0.98-similar to A (confusable)
    eng = _make({10: [a], 20: [b]})
    pid, score, margin = eng._match(a)
    assert pid == 10 and score > 0.99
    assert margin < 0.05, f"confusable people should give a small margin, got {margin:.3f}"
    # The engine's policy: score over threshold but margin under recog_margin ⇒ 'ambiguous'.
    assert score >= eng.threshold and margin < eng.recog_margin
    print(f"  ambiguous: score={score:.3f} margin={margin:.3f} (< recog_margin {eng.recog_margin})")


def test_match_picks_best_exemplar() -> None:
    """Multi-template: a probe matches via a person's CLOSEST exemplar, not their mean.

    Person 10 has two very different exemplars; the probe equals the second. A single-centroid
    (mean) representation would score poorly, but per-exemplar matching nails it — the core
    anti-fragmentation win.
    """
    e1, e2 = _unit(1), _unit(7)            # two dissimilar shots of the same person
    other = _unit(42)
    multi = _make({10: [e1, e2], 20: [other]})
    pid, score, margin = multi._match(e2)
    assert pid == 10 and score > 0.99, f"should match person 10 via exemplar e2, got {pid}/{score:.3f}"

    # Contrast: with only the *mean* of e1,e2 as a single template, the same probe scores far lower.
    mean_only = _make({10: [_normalize(e1 + e2)], 20: [other]})
    _, score_mean, _ = mean_only._match(e2)
    assert score_mean < score - 0.1, (
        f"centroid-only ({score_mean:.3f}) should be clearly worse than multi-template ({score:.3f})"
    )
    print(f"  multi-template: exemplar match={score:.3f} vs centroid-only={score_mean:.3f}")


def test_add_template_fifo_cap() -> None:
    """_add_template caps a person at templates_per_person, keeping the most recent (FIFO)."""
    eng = _make({10: [_unit(1)]}, templates_per_person=3)
    vecs = [_unit(s) for s in (2, 3, 4, 5)]
    for v in vecs:
        eng._add_template(10, v)
    assert len(eng._templates[10]) == 3, f"should cap at 3, got {len(eng._templates[10])}"
    # The matrix has exactly the capped rows for this single person.
    assert eng._matrix.shape[0] == 3 and int((eng._row_pid == 10).sum()) == 3
    # The newest vector is retained and still matches with score ~1.0.
    pid, score, _ = eng._match(vecs[-1])
    assert pid == 10 and score > 0.99, "the most-recent exemplar must be kept + matchable"
    # The very first (oldest) vector was evicted: it should no longer self-match at ~1.0.
    _, old_score, _ = eng._match(_unit(1))
    assert old_score < 0.99, f"evicted oldest template should not self-match, got {old_score:.3f}"
    print(f"  FIFO cap: kept newest 3/5; newest score={score:.3f}, evicted oldest={old_score:.3f}")


def test_new_person_added_matchable() -> None:
    """_add_template for a fresh id makes that person matchable immediately (next-frame dedupe)."""
    eng = _make({10: [_unit(1)]}, templates_per_person=4)
    newv = _unit(500)
    eng._add_template(77, newv)            # brand-new person id, as the engine does on a new unknown
    pid, score, _ = eng._match(newv)
    assert pid == 77 and score > 0.99, "a newly-added person must match on the next probe"
    assert 77 in eng._templates and 10 in eng._templates
    print("  new person matchable immediately after _add_template")


def test_rebuild_row_pid_consistency() -> None:
    """_rebuild_locked keeps _matrix rows and _row_pid aligned and counts right."""
    eng = _make({1: [_unit(1), _unit(2)], 2: [_unit(3)], 3: [_unit(4), _unit(5), _unit(6)]})
    assert eng._matrix.shape == (6, EMB_DIM)
    assert eng._row_pid.shape == (6,)
    counts = {pid: int((eng._row_pid == pid).sum()) for pid in (1, 2, 3)}
    assert counts == {1: 2, 2: 1, 3: 3}, counts
    # Every row is a unit vector.
    norms = np.linalg.norm(eng._matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "all template rows must be L2-normalised"
    print(f"  rebuild: 6 rows, per-person counts {counts}, all unit-norm")


def _run_all() -> int:
    tests = [
        test_match_empty,
        test_match_basic_and_margin,
        test_match_ambiguous_small_margin,
        test_match_picks_best_exemplar,
        test_add_template_fifo_cap,
        test_new_person_added_matchable,
        test_rebuild_row_pid_consistency,
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
