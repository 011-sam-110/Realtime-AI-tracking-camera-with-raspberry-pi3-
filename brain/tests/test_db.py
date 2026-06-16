"""Offline unit tests for the thread-safe SQLite store (INTERFACES.md §4).

Exercises the real ``kitchenvision.store.db`` module against a TEMPORARY data dir so it never
touches the user's live ``data/`` and leaves nothing behind. No model load, no GPU, no network —
just sqlite3 + numpy + the stdlib.

How the temp dir is wired
-------------------------
``store.db`` derives ``DATA_DIR``/``DB_PATH``/``FACES_DIR``/``THUMBS_DIR`` from
``config.load_config()["data_dir"]`` **once at import time**. So to redirect them we monkeypatch
those module-level constants to point inside a ``tempfile.mkdtemp()`` BEFORE calling ``init_db()``.
Every helper reads the constants live (via ``crops_dir`` etc.), so patching them is sufficient.

Plain-python runnable (pytest may be absent):  python tests/test_db.py
Run from the brain root (C:/Users/sampo/pi/brain) so ``import kitchenvision...`` resolves.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import numpy as np

# Make `import kitchenvision...` work when run as a bare script from the brain root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRAIN = os.path.dirname(_HERE)
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from kitchenvision.core.types import Event  # noqa: E402
from kitchenvision.store import db  # noqa: E402


# --------------------------------------------------------------------------- temp-dir harness
def _point_db_at_tmp(tmp: str) -> None:
    """Repoint every path constant in ``store.db`` into ``tmp`` and (re)create a fresh schema."""
    db.DATA_DIR = tmp.replace("\\", "/")
    db.DB_PATH = os.path.join(tmp, "kitchenvision.db").replace("\\", "/")
    db.FACES_DIR = os.path.join(tmp, "faces").replace("\\", "/")
    db.THUMBS_DIR = os.path.join(tmp, "thumbs").replace("\\", "/")
    db.init_db()


# --------------------------------------------------------------------------- helpers
def _vec(seed: int, dim: int = 512) -> np.ndarray:
    """A deterministic pseudo-embedding (unnormalised f32)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _jpg(tag: int) -> bytes:
    """A few unique bytes standing in for JPEG content (the store treats it as opaque blob)."""
    return bytes([0xFF, 0xD8, 0xFF, tag & 0xFF]) + f"crop{tag}".encode()


# --------------------------------------------------------------------------- tests
def test_constants_and_paths() -> None:
    """The required constants exist with the documented values/derivation."""
    assert db.MAX_CROPS == 5
    assert db.DB_PATH.endswith("kitchenvision.db")
    assert db.FACES_DIR.endswith("/faces")
    assert db.THUMBS_DIR.endswith("/thumbs")
    # init_db() must have created the data dirs.
    assert os.path.isdir(db.DATA_DIR)
    assert os.path.isdir(db.FACES_DIR)
    assert os.path.isdir(db.THUMBS_DIR)
    print(f"  constants ok: MAX_CROPS={db.MAX_CROPS} db={os.path.basename(db.DB_PATH)}")


def test_connect_pragmas() -> None:
    """connect() returns a Row-factory WAL connection with a 30 s busy timeout."""
    conn = db.connect()
    try:
        assert conn.row_factory is __import__("sqlite3").Row
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal", f"expected WAL, got {mode}"
        bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert int(bt) == 30000, f"expected 30000ms busy_timeout, got {bt}"
        # Row access by column name works.
        row = conn.execute("SELECT 1 AS one").fetchone()
        assert row["one"] == 1
    finally:
        conn.close()
    print("  connect: WAL + busy_timeout=30000 + Row factory")


def test_person_roundtrip_and_label() -> None:
    """create_person → list_people/get_person; labels rank unknowns by id; set_name flips kind."""
    p1 = db.create_person(_vec(1), kind="unknown")
    p2 = db.create_person(_vec(2), kind="unknown")
    p3 = db.create_person(_vec(3), kind="unknown")
    assert p1 < p2 < p3, "ids should be ascending"

    people = db.list_people()
    assert [p["id"] for p in people] == [p1, p2, p3], "list_people ordered by id asc"
    # All three unnamed → Unknown #1..#3 by id rank.
    labels = {p["id"]: p["label"] for p in people}
    assert labels[p1] == "Unknown #1"
    assert labels[p2] == "Unknown #2"
    assert labels[p3] == "Unknown #3"
    assert all(p["kind"] == "unknown" for p in people)

    # get_person matches the list shape and label.
    got = db.get_person(p2)
    assert got is not None
    for k in ("id", "name", "kind", "age", "sex", "created_ts", "last_seen_ts", "label"):
        assert k in got, f"get_person missing key {k}"
    assert got["label"] == "Unknown #2"
    assert db.get_person(999999) is None, "missing id → None"

    # Name the middle person → it leaves the unknown ranking; the others re-rank.
    db.set_name(p2, "  Sam  ")  # surrounding whitespace must be stripped
    assert db.get_person(p2)["name"] == "Sam"
    assert db.get_person(p2)["kind"] == "known", "set_name flips kind to 'known'"
    assert db.get_person(p2)["label"] == "Sam"
    relabels = {p["id"]: p["label"] for p in db.list_people()}
    assert relabels[p1] == "Unknown #1"
    assert relabels[p3] == "Unknown #2", "p3 should move up to #2 once p2 is named"

    # Blank set_name is a no-op (does not blank an existing name / change kind).
    db.set_name(p1, "   ")
    assert db.get_person(p1)["name"] is None
    assert db.get_person(p1)["kind"] == "unknown"
    print("  people: ids asc, Unknown #N rank, set_name strips+flips+re-ranks, blank=no-op")


def test_embeddings_and_centroids_normalised() -> None:
    """add_embedding stores quality; load_centroids returns only L2-normalised centroids."""
    pid = db.create_person(_vec(10), kind="unknown")
    db.add_embedding(pid, _vec(11), quality=0.7)
    db.add_embedding(pid, _vec(12))  # default quality=0.0
    # quality column round-trips.
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT quality FROM embeddings WHERE person_id = ? ORDER BY id", (pid,)
        ).fetchall()
        qs = sorted(round(r["quality"], 3) for r in rows)
        assert qs == [0.0, 0.7], f"stored qualities should be [0.0, 0.7], got {qs}"
    finally:
        conn.close()

    # A person with a NULL centroid is skipped by load_centroids.
    skip = db.create_person(_vec(13), kind="unknown")
    conn = db.connect()
    try:
        with db._write_lock:
            conn.execute("UPDATE people SET centroid = NULL WHERE id = ?", (skip,))
            conn.commit()
    finally:
        conn.close()

    cents = db.load_centroids()
    by_id = {c["id"]: c for c in cents}
    assert pid in by_id, "person with a centroid must be returned"
    assert skip not in by_id, "NULL-centroid person must be skipped"
    c = by_id[pid]
    assert c["centroid"].dtype == np.float32
    nrm = float(np.linalg.norm(c["centroid"]))
    assert abs(nrm - 1.0) < 1e-5, f"centroid must be L2-normalised, |c|={nrm}"
    assert set(c.keys()) == {"id", "name", "kind", "centroid"}
    print(f"  embeddings+centroids: quality stored, |centroid|={nrm:.6f}, NULL skipped")


def test_get_centroid() -> None:
    """get_centroid returns the stored centroid L2-normalised; None for missing / NULL."""
    raw = _vec(15)
    pid = db.create_person(raw, kind="unknown")
    got = db.get_centroid(pid)
    assert got is not None and got.dtype == np.float32
    assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-5, "centroid must be unit-norm"
    assert np.allclose(got, (raw / np.linalg.norm(raw)).astype(np.float32), atol=1e-6)
    assert db.get_centroid(999999) is None, "missing id → None"

    # NULL centroid → None.
    conn = db.connect()
    try:
        with db._write_lock:
            conn.execute("UPDATE people SET centroid = NULL WHERE id = ?", (pid,))
            conn.commit()
    finally:
        conn.close()
    assert db.get_centroid(pid) is None, "NULL centroid → None"
    print("  get_centroid: normalised vector; None for missing/NULL")


def test_load_templates() -> None:
    """load_templates(k) returns centroid + top-(k-1) quality embeddings per person, all normalised."""
    # Person A: centroid + 3 embeddings of distinct quality.
    a = db.create_person(_vec(1), kind="unknown")
    db.add_embedding(a, _vec(2), quality=0.9)   # best
    db.add_embedding(a, _vec(3), quality=0.5)   # middle
    db.add_embedding(a, _vec(4), quality=0.1)   # worst (should be dropped at k=3)
    # Person B: centroid only (no embeddings).
    b = db.create_person(_vec(20), kind="unknown")

    rows = db.load_templates(k=3)
    by_pid: "dict[int, list]" = {}
    for r in rows:
        assert set(r.keys()) == {"id", "name", "kind", "vec"}, r.keys()
        assert r["vec"].dtype == np.float32
        assert abs(float(np.linalg.norm(r["vec"])) - 1.0) < 1e-5, "every template unit-norm"
        by_pid.setdefault(r["id"], []).append(r["vec"])

    assert len(by_pid[a]) == 3, f"A: centroid + top-2 embeddings = 3, got {len(by_pid[a])}"
    assert len(by_pid[b]) == 1, f"B: centroid only = 1, got {len(by_pid[b])}"

    # k=1 degenerates to one template (the centroid) per person — like load_centroids.
    one = db.load_templates(k=1)
    counts = {}
    for r in one:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    assert counts == {a: 1, b: 1}, counts

    # A person with embeddings but a NULL centroid still yields up to k embedding-templates.
    c = db.create_person(_vec(30), kind="unknown")
    db.add_embedding(c, _vec(31), quality=0.8)
    db.add_embedding(c, _vec(32), quality=0.7)
    conn = db.connect()
    try:
        with db._write_lock:
            conn.execute("UPDATE people SET centroid = NULL WHERE id = ?", (c,))
            conn.commit()
    finally:
        conn.close()
    crows = [r for r in db.load_templates(k=3) if r["id"] == c]
    assert len(crows) == 2, f"NULL-centroid person → its embeddings as templates, got {len(crows)}"
    print("  load_templates: centroid+top-quality embeddings, k cap, k=1 centroid-only, NULL-centroid ok")


def test_blob_roundtrip_exact() -> None:
    """The float32 blob round-trip is exact (centroid stored is the normalised input)."""
    raw = _vec(20)
    pid = db.create_person(raw, kind="unknown")
    expected = raw / np.linalg.norm(raw)
    got = db.load_centroids()
    stored = next(c["centroid"] for c in got if c["id"] == pid)
    assert np.allclose(stored, expected.astype(np.float32), atol=1e-6), "centroid blob mismatch"
    print("  blob: float32 centroid round-trips bit-for-bit (normalised)")


def test_sightings_and_last_seen() -> None:
    """add_sighting writes a row; set_last_seen/update_centroid mutate the person."""
    pid = db.create_person(_vec(30), kind="unknown")
    db.add_sighting(pid, 1000.0, (10, 20, 30, 40))
    db.add_sighting(pid, 1001.0, [11, 21, 31, 41])  # list box also accepted
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT x, y, w, h FROM sightings WHERE person_id = ? ORDER BY ts", (pid,)
        ).fetchall()
        assert len(rows) == 2
        assert tuple(rows[0]) == (10, 20, 30, 40)
        assert tuple(rows[1]) == (11, 21, 31, 41)
    finally:
        conn.close()

    db.set_last_seen(pid, 5555.5)
    assert abs(db.get_person(pid)["last_seen_ts"] - 5555.5) < 1e-6

    new_c = _vec(31)
    db.update_centroid(pid, new_c)
    stored = next(c["centroid"] for c in db.load_centroids() if c["id"] == pid)
    assert np.allclose(stored, (new_c / np.linalg.norm(new_c)).astype(np.float32), atol=1e-6)
    print("  sightings: rows written; set_last_seen + update_centroid applied")


def test_events_and_timeline() -> None:
    """add_event maps the Event dataclass → row (payload JSON); get_events/get_timeline newest-first."""
    pid = db.create_person(_vec(40), kind="unknown")
    other = db.create_person(_vec(41), kind="unknown")

    e1 = Event(
        type="activity", ts=100.0, person_id=pid, action="left", object="plate",
        location="table", text="Sam left a plate on the table", confidence=0.8,
        source="vlm", thumb_ref=None, payload={"raw": "x", "n": 3},
    )
    e2 = Event(
        type="chore", ts=200.0, person_id=pid, action="cleared", object="plate",
        location="table", text="Sam cleared the plate", confidence=0.9, source="vlm",
    )
    e3 = Event(  # scene-level (no subject) + different person filter check
        type="presence", ts=150.0, person_id=None, text="someone entered", source="cv",
    )
    e4 = Event(type="activity", ts=120.0, person_id=other, text="Alex washing up")

    id1 = db.add_event(e1)
    id2 = db.add_event(e2)
    id3 = db.add_event(e3)
    id4 = db.add_event(e4)
    assert id1 and id2 and id3 and id4 and len({id1, id2, id3, id4}) == 4, "distinct row ids"

    # get_events (no filter) → ALL events, newest first by ts.
    allev = db.get_events()
    assert [e["ts"] for e in allev] == [200.0, 150.0, 120.0, 100.0], "newest-first by ts"

    # payload round-trips back to a dict; non-payload event decodes to None.
    first = next(e for e in allev if e["id"] == id1)
    assert first["payload"] == {"raw": "x", "n": 3}, first["payload"]
    assert first["action"] == "left" and first["object"] == "plate" and first["location"] == "table"
    assert first["type"] == "activity" and first["source"] == "vlm"
    e2row = next(e for e in allev if e["id"] == id2)
    assert e2row["payload"] is None, "missing payload decodes to None"

    # person filter.
    samev = db.get_events(person_id=pid)
    assert {e["id"] for e in samev} == {id1, id2}, "person filter keeps only that subject"
    assert [e["ts"] for e in samev] == [200.0, 100.0]

    # since filter (inclusive ts >= since).
    recent = db.get_events(since=150.0)
    assert {e["ts"] for e in recent} == {200.0, 150.0}, "since is inclusive lower bound"

    # limit cap.
    assert len(db.get_events(limit=2)) == 2

    # get_timeline = one person's events, newest first.
    tl = db.get_timeline(pid)
    assert [e["id"] for e in tl] == [id2, id1], "timeline newest-first for the subject"
    assert db.get_timeline(other)[0]["text"] == "Alex washing up"
    print("  events: dataclass->row, payload JSON<->dict, person/since/limit filters, timeline order")


def test_activity_log_backcompat() -> None:
    """add_activity writes a back-compat freeform caption row (separate from events)."""
    pid = db.create_person(_vec(50), kind="unknown")
    db.add_activity(pid, 10.0, "freeform caption", "groq")
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT person_id, text, provider FROM activity_log WHERE person_id = ?", (pid,)
        ).fetchone()
        assert row is not None
        assert row["text"] == "freeform caption" and row["provider"] == "groq"
    finally:
        conn.close()
    print("  activity_log: back-compat caption row written")


def test_crops_rotation() -> None:
    """save_crop writes 0.jpg newest, rotates older up, keeps at most MAX_CROPS; list_crops newest-first."""
    pid = db.create_person(_vec(60), kind="unknown")
    # Write MAX_CROPS+2 crops; newest is the last written.
    n = db.MAX_CROPS + 2
    for i in range(n):
        db.save_crop(pid, _jpg(i))
    crops = db.list_crops(pid)
    assert len(crops) == db.MAX_CROPS, f"keep at most MAX_CROPS, got {len(crops)}"
    # 0.jpg holds the newest content (last tag written = n-1).
    with open(crops[0], "rb") as f:
        newest = f.read()
    assert newest == _jpg(n - 1), "0.jpg must hold the newest crop"
    # All paths are real files, ordered 0..MAX_CROPS-1.
    assert crops == [
        os.path.join(db.crops_dir(pid, create=False), f"{i}.jpg").replace("\\", "/")
        for i in range(db.MAX_CROPS)
    ]
    assert all(os.path.isfile(p) for p in crops)
    # Empty bytes is a no-op (does not rotate/destroy existing crops).
    db.save_crop(pid, b"")
    assert len(db.list_crops(pid)) == db.MAX_CROPS
    print(f"  crops: rotate, cap at {db.MAX_CROPS}, 0.jpg=newest, empty=no-op")


def test_thumbnails() -> None:
    """save_thumb writes a uniquely-keyed jpg under THUMBS_DIR; thumb_path resolves it."""
    ref_a = db.save_thumb(_jpg(100))
    ref_b = db.save_thumb(_jpg(101))
    assert ref_a and ref_b and ref_a != ref_b, "distinct non-empty refs"
    pa = db.thumb_path(ref_a)
    assert pa is not None and os.path.isfile(pa), "thumb_path resolves to a real file"
    assert pa.startswith(db.THUMBS_DIR), "thumb lives under THUMBS_DIR"
    with open(pa, "rb") as f:
        assert f.read() == _jpg(100), "thumb content round-trips"
    # Robust to a stored ".jpg" suffix and to unknown refs.
    assert db.thumb_path(ref_a + ".jpg") == pa, "trailing .jpg tolerated"
    assert db.thumb_path("does-not-exist") is None
    assert db.thumb_path("") is None
    assert db.thumb_path(None) is None
    # Empty input → empty ref, writes nothing.
    assert db.save_thumb(b"") == ""
    print(f"  thumbnails: keyed write under thumbs/, thumb_path resolves, robust to .jpg/None")


def test_merge_people() -> None:
    """merge_people reassigns rows + crops to dst, recomputes centroid, deletes src; src==dst no-op."""
    src = db.create_person(_vec(70), kind="unknown")
    dst = db.create_person(_vec(71), kind="unknown")
    db.set_name(dst, "Keeper")

    # Give each some data.
    db.add_embedding(src, _vec(72), quality=0.5)
    db.add_embedding(dst, _vec(73), quality=0.5)
    db.add_sighting(src, 10.0, (1, 2, 3, 4))
    db.add_sighting(dst, 20.0, (5, 6, 7, 8))
    db.add_event(Event(type="activity", ts=11.0, person_id=src, text="src event"))
    db.add_event(Event(type="activity", ts=21.0, person_id=dst, text="dst event"))
    db.add_activity(src, 12.0, "src caption", "p")
    db.save_crop(src, _jpg(200))
    db.save_crop(dst, _jpg(201))
    db.set_last_seen(src, 9999.0)  # src is later → dst.last_seen should advance to this
    db.set_last_seen(dst, 50.0)

    # No-op when src == dst.
    db.merge_people(dst, dst)
    assert db.get_person(dst) is not None

    db.merge_people(src, dst)

    # src gone; dst remains with the merged rows.
    assert db.get_person(src) is None, "src person deleted after merge"
    assert db.get_person(dst)["name"] == "Keeper"

    conn = db.connect()
    try:
        n_emb = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (dst,)
        ).fetchone()[0]
        n_sig = conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE person_id = ?", (dst,)
        ).fetchone()[0]
        n_evt = conn.execute(
            "SELECT COUNT(*) FROM events WHERE person_id = ?", (dst,)
        ).fetchone()[0]
        n_act = conn.execute(
            "SELECT COUNT(*) FROM activity_log WHERE person_id = ?", (dst,)
        ).fetchone()[0]
        # No rows left pointing at the deleted src.
        orphan = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM embeddings WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM sightings WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM events WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM activity_log WHERE person_id=?)",
            (src, src, src, src),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_emb == 2, f"both embeddings now on dst, got {n_emb}"
    assert n_sig == 2 and n_evt == 2 and n_act == 1
    assert orphan == 0, "no rows should reference the deleted src id"

    # dst.last_seen advanced to the later of the two (src's 9999.0).
    assert abs(db.get_person(dst)["last_seen_ts"] - 9999.0) < 1e-6

    # dst centroid recomputed = normalised mean of the two combined embeddings, still unit-norm.
    cent = next(c["centroid"] for c in db.load_centroids() if c["id"] == dst)
    assert abs(float(np.linalg.norm(cent)) - 1.0) < 1e-5

    # src crop got merged into dst (dst now has both crops) and src crop dir removed.
    assert len(db.list_crops(dst)) == 2
    assert not os.path.isdir(db.crops_dir(src, create=False)), "src crop dir removed"
    print("  merge: rows+crops reassigned, src deleted, last_seen=max, centroid re-normalised")


def test_delete_person() -> None:
    """delete_person erases the person + all their rows + crops/thumbs dir (privacy)."""
    pid = db.create_person(_vec(80), kind="unknown")
    db.add_embedding(pid, _vec(81))
    db.add_sighting(pid, 1.0, (1, 1, 1, 1))
    db.add_event(Event(type="activity", ts=2.0, person_id=pid, text="x"))
    db.add_activity(pid, 3.0, "y", "p")
    db.save_crop(pid, _jpg(300))
    crop_dir = db.crops_dir(pid, create=False)
    assert os.path.isdir(crop_dir)

    db.delete_person(pid)

    assert db.get_person(pid) is None
    conn = db.connect()
    try:
        total = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM embeddings WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM sightings WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM events WHERE person_id=?) + "
            "(SELECT COUNT(*) FROM activity_log WHERE person_id=?)",
            (pid, pid, pid, pid),
        ).fetchone()[0]
    finally:
        conn.close()
    assert total == 0, "all of the person's rows must be gone"
    assert not os.path.isdir(crop_dir), "crops dir removed"
    print("  delete: person + embeddings/sightings/events/activity + crops removed")


def test_prune_keeps_people_and_embeddings() -> None:
    """prune deletes OLD sightings + events (and activity_log) but KEEPS people + embeddings."""
    pid = db.create_person(_vec(90), kind="unknown")
    db.add_embedding(pid, _vec(91))

    now = time.time()
    old = now - 40 * 86400.0   # 40 days ago (older than a 30-day window)
    recent = now - 1 * 86400.0  # 1 day ago

    db.add_sighting(pid, old, (1, 1, 1, 1))
    db.add_sighting(pid, recent, (2, 2, 2, 2))
    db.add_event(Event(type="activity", ts=old, person_id=pid, text="old event"))
    db.add_event(Event(type="activity", ts=recent, person_id=pid, text="recent event"))
    db.add_activity(pid, old, "old caption", "p")
    db.add_activity(pid, recent, "recent caption", "p")

    s_del, e_del = db.prune(30)
    assert s_del == 1, f"exactly the one old sighting pruned, got {s_del}"
    assert e_del == 1, f"exactly the one old event pruned, got {e_del}"

    # People + embeddings survive.
    assert db.get_person(pid) is not None, "prune must KEEP people"
    conn = db.connect()
    try:
        n_emb = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (pid,)
        ).fetchone()[0]
        n_sig = conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE person_id = ?", (pid,)
        ).fetchone()[0]
        n_evt = conn.execute(
            "SELECT COUNT(*) FROM events WHERE person_id = ?", (pid,)
        ).fetchone()[0]
        n_act = conn.execute(
            "SELECT COUNT(*) FROM activity_log WHERE person_id = ?", (pid,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_emb == 1, "embeddings must be KEPT by prune"
    assert n_sig == 1 and n_evt == 1, "only the recent sighting/event remain"
    assert n_act == 1, "only the recent activity_log row remains"
    print("  prune: old sightings+events+activity pruned; people+embeddings kept")


def _run_all() -> int:
    tests = [
        test_constants_and_paths,
        test_connect_pragmas,
        test_person_roundtrip_and_label,
        test_embeddings_and_centroids_normalised,
        test_get_centroid,
        test_load_templates,
        test_blob_roundtrip_exact,
        test_sightings_and_last_seen,
        test_events_and_timeline,
        test_activity_log_backcompat,
        test_crops_rotation,
        test_thumbnails,
        test_merge_people,
        test_delete_person,
        test_prune_keeps_people_and_embeddings,
    ]
    failed = 0
    for t in tests:
        # Each test gets a FRESH temp DB so counts (Unknown #N rank, prune totals, merge/delete
        # row counts) are deterministic and tests can't contaminate each other.
        tmp = tempfile.mkdtemp(prefix="kv_db_test_")
        _point_db_at_tmp(tmp)
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
        finally:
            # Every helper opens/closes its own conn, so the temp tree is unlocked for removal.
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} tests FAILED")
    else:
        print(f"RESULT: all {len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
