"""Thread-safe SQLite persistence for the Kitchen Vision brain (INTERFACES.md §4).

Ported from the proven ``station/db.py`` (threading model, float32 blob round-trip,
normalised-centroid storage, the "Unknown #N" label ranking, and crop rotation), then
EXTENDED with:

  * an ``events`` table — the unified activity/object/chore record. ``add_event`` maps the
    :class:`kitchenvision.core.types.Event` dataclass onto a row (``payload`` stored as JSON
    text); ``get_events`` / ``get_timeline`` return rows newest-first.
  * event **thumbnails** — ``save_thumb`` writes a uniquely-keyed JPEG under ``THUMBS_DIR`` and
    returns the key; ``thumb_path`` resolves a key back to a filesystem path.

THREADING MODEL (critical — see INTERFACES.md §0)
-------------------------------------------------
The capture/track thread, the recognition thread, the perception worker, the prune thread, and
the FastAPI request threads all touch this database. A single ``sqlite3.Connection`` object
CANNOT be shared safely across threads. Therefore:

  * :func:`connect` returns a BRAND-NEW connection every call. Every helper in this module opens
    its own short-lived connection (open -> work -> close). Connections never cross threads, so
    the default ``check_same_thread=True`` is correct.
  * All WRITES go through the module-level :data:`_write_lock` so concurrent writers from different
    threads serialise (SQLite is single-writer). WAL mode + a 30 s busy timeout give readers
    concurrency and absorb brief contention.

Vectors (embeddings + centroids) are stored as raw float32 bytes::

    to blob:   np.asarray(arr, np.float32).tobytes()
    from blob: np.frombuffer(blob, np.float32)

Centroids are stored **L2-normalised** and :func:`load_centroids` returns them normalised.

Retention: :func:`prune` deletes old sightings + events (and the back-compat activity_log);
people and embeddings are KEPT permanently (labelled identities + face vectors are durable).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid

import numpy as np

from kitchenvision.core import config
from kitchenvision.core.types import Event

# --- paths derived from the (possibly overridden) data_dir, resolved once at import ----------
_CFG = config.load_config()
DATA_DIR: str = _CFG["data_dir"]
DB_PATH: str = os.path.join(DATA_DIR, "kitchenvision.db").replace("\\", "/")
FACES_DIR: str = os.path.join(DATA_DIR, "faces").replace("\\", "/")
THUMBS_DIR: str = os.path.join(DATA_DIR, "thumbs").replace("\\", "/")

# How many representative face crops to retain per person (0.jpg .. (MAX_CROPS-1).jpg).
MAX_CROPS: int = 5

# Serialises all writers across threads (SQLite allows a single writer at a time).
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# blob <-> numpy helpers
# ---------------------------------------------------------------------------
def _to_blob(arr) -> bytes:
    """Pack a 1-D vector to float32 bytes for storage."""
    return np.asarray(arr, dtype=np.float32).tobytes()


def _from_blob(blob) -> "np.ndarray | None":
    """Unpack float32 bytes to a 1-D numpy array (copy, writable). None -> None."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def _normalize(vec) -> "np.ndarray":
    """L2-normalise a vector to float32. Zero vectors are returned unchanged."""
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32)


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    """Return a NEW sqlite3 connection (the caller owns + closes it).

    ``row_factory`` is :class:`sqlite3.Row` so callers can use column names. WAL +
    ``busy_timeout=30 s`` give safe multi-thread reader/writer behaviour. A fresh connection
    per call is REQUIRED — connections must never cross threads (INTERFACES.md §0).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.DatabaseError:
        pass
    return conn


def init_db() -> None:
    """Create the data dirs, the §4 schema (IF NOT EXISTS), and helpful indexes (call once)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)
    with _write_lock:
        conn = connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id            INTEGER PRIMARY KEY,
                    name          TEXT,
                    kind          TEXT,
                    centroid      BLOB,
                    age           REAL,
                    sex           TEXT,
                    created_ts    REAL,
                    last_seen_ts  REAL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id         INTEGER PRIMARY KEY,
                    person_id  INTEGER,
                    vec        BLOB,
                    quality    REAL,
                    ts         REAL
                );

                CREATE TABLE IF NOT EXISTS sightings (
                    id         INTEGER PRIMARY KEY,
                    person_id  INTEGER,
                    ts         REAL,
                    x          INTEGER,
                    y          INTEGER,
                    w          INTEGER,
                    h          INTEGER
                );

                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY,
                    ts          REAL,
                    person_id   INTEGER,
                    type        TEXT,
                    action      TEXT,
                    object      TEXT,
                    location    TEXT,
                    text        TEXT,
                    confidence  REAL,
                    source      TEXT,
                    thumb_ref   TEXT,
                    payload     TEXT
                );

                CREATE TABLE IF NOT EXISTS activity_log (
                    id         INTEGER PRIMARY KEY,
                    person_id  INTEGER,
                    ts         REAL,
                    text       TEXT,
                    provider   TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_emb_person   ON embeddings(person_id);
                CREATE INDEX IF NOT EXISTS idx_sight_person ON sightings(person_id, ts);
                CREATE INDEX IF NOT EXISTS idx_sight_ts     ON sightings(ts);
                CREATE INDEX IF NOT EXISTS idx_evt_person   ON events(person_id, ts);
                CREATE INDEX IF NOT EXISTS idx_evt_ts       ON events(ts);
                CREATE INDEX IF NOT EXISTS idx_act_person   ON activity_log(person_id, ts);
                CREATE INDEX IF NOT EXISTS idx_act_ts       ON activity_log(ts);
                """
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# label helper (Unknown #N ordinal among unknowns by id)
# ---------------------------------------------------------------------------
def _compute_unknown_ordinals(conn: sqlite3.Connection) -> "dict[int, int]":
    """Return ``{person_id: ordinal}`` for every unnamed person, 1-based, ranked by id asc.

    An "unknown" for labelling is any person whose ``name`` IS NULL (or empty). The ordinal is
    its 1-based position among those people ordered by id ascending, so the lowest-id unknown is
    "Unknown #1". Stable for a given DB state.
    """
    rows = conn.execute(
        "SELECT id FROM people WHERE name IS NULL OR name = '' ORDER BY id ASC"
    ).fetchall()
    return {r["id"]: i + 1 for i, r in enumerate(rows)}


def _label_for(row: sqlite3.Row, ordinals: "dict[int, int]") -> str:
    """Compute the display label for a people row given the unknown-ordinal map."""
    name = row["name"]
    if name:
        return name
    ordinal = ordinals.get(row["id"], 0)
    return f"Unknown #{ordinal}"


def _row_to_person(row: sqlite3.Row, ordinals: "dict[int, int]") -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "age": row["age"],
        "sex": row["sex"],
        "created_ts": row["created_ts"],
        "last_seen_ts": row["last_seen_ts"],
        "label": _label_for(row, ordinals),
    }


# ---------------------------------------------------------------------------
# people CRUD
# ---------------------------------------------------------------------------
def create_person(centroid, kind: str, age=None, sex=None) -> int:
    """Insert a new person with an initial centroid; return its new id.

    ``centroid`` is stored as L2-normalised float32 bytes. ``created_ts`` and ``last_seen_ts``
    are both set to ``time.time()``.
    """
    ts = time.time()
    blob = _to_blob(_normalize(centroid))
    age = None if age is None else float(age)
    with _write_lock:
        conn = connect()
        try:
            cur = conn.execute(
                "INSERT INTO people (name, kind, centroid, age, sex, created_ts, last_seen_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (None, kind, blob, age, sex, ts, ts),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def add_embedding(person_id: int, vec, quality: float = 0.0) -> None:
    """Append a face embedding (with its best-shot quality) for a person."""
    ts = time.time()
    blob = _to_blob(vec)
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO embeddings (person_id, vec, quality, ts) VALUES (?, ?, ?, ?)",
                (int(person_id), blob, float(quality), ts),
            )
            conn.commit()
        finally:
            conn.close()


def update_centroid(person_id: int, centroid) -> None:
    """Replace a person's centroid (stored L2-normalised as float32 bytes)."""
    blob = _to_blob(_normalize(centroid))
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "UPDATE people SET centroid = ? WHERE id = ?", (blob, int(person_id))
            )
            conn.commit()
        finally:
            conn.close()


def set_last_seen(person_id: int, ts: float) -> None:
    """Update a person's ``last_seen_ts``."""
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "UPDATE people SET last_seen_ts = ? WHERE id = ?",
                (float(ts), int(person_id)),
            )
            conn.commit()
        finally:
            conn.close()


def list_people() -> "list[dict]":
    """Return all people as dicts ``{id,name,kind,age,sex,created_ts,last_seen_ts,label}``.

    Ordered by id ascending. ``label`` = name, or "Unknown #N" where N is the person's 1-based
    rank among unnamed people ordered by id.
    """
    conn = connect()
    try:
        ordinals = _compute_unknown_ordinals(conn)
        rows = conn.execute("SELECT * FROM people ORDER BY id ASC").fetchall()
        return [_row_to_person(r, ordinals) for r in rows]
    finally:
        conn.close()


def get_person(person_id: int) -> "dict | None":
    """Return one person dict (same shape as :func:`list_people` entries) or ``None``."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM people WHERE id = ?", (int(person_id),)
        ).fetchone()
        if row is None:
            return None
        ordinals = _compute_unknown_ordinals(conn)
        return _row_to_person(row, ordinals)
    finally:
        conn.close()


def set_name(person_id: int, name: str) -> None:
    """Name a person (and flip ``kind`` to 'known'). A blank/whitespace name is a no-op."""
    name = (name or "").strip()
    if not name:
        return
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "UPDATE people SET name = ?, kind = 'known' WHERE id = ?",
                (name, int(person_id)),
            )
            conn.commit()
        finally:
            conn.close()


def merge_people(src_id: int, dst_id: int) -> None:
    """Merge person ``src_id`` into ``dst_id``.

    Reassigns all embeddings, sightings, events, and activity_log rows from src to dst,
    recomputes dst's centroid as the normalised mean of dst's (now combined) embeddings,
    advances ``dst.last_seen_ts`` to the later of the two, moves any src face crops into dst
    (keeping the newest MAX_CROPS), then deletes the src person row and its crops dir. No-op
    if ``src_id == dst_id``.
    """
    src_id, dst_id = int(src_id), int(dst_id)
    if src_id == dst_id:
        return
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "UPDATE embeddings SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
            )
            conn.execute(
                "UPDATE sightings SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
            )
            conn.execute(
                "UPDATE events SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
            )
            conn.execute(
                "UPDATE activity_log SET person_id = ? WHERE person_id = ?", (dst_id, src_id)
            )

            # Recompute dst centroid from the combined embeddings.
            rows = conn.execute(
                "SELECT vec FROM embeddings WHERE person_id = ?", (dst_id,)
            ).fetchall()
            vecs = [_from_blob(r["vec"]) for r in rows if r["vec"] is not None]
            if vecs:
                mean = np.mean(np.stack(vecs, axis=0), axis=0)
                conn.execute(
                    "UPDATE people SET centroid = ? WHERE id = ?",
                    (_to_blob(_normalize(mean)), dst_id),
                )

            # last_seen = max of the two.
            src_row = conn.execute(
                "SELECT last_seen_ts FROM people WHERE id = ?", (src_id,)
            ).fetchone()
            dst_row = conn.execute(
                "SELECT last_seen_ts FROM people WHERE id = ?", (dst_id,)
            ).fetchone()
            if src_row is not None and dst_row is not None:
                latest = max(src_row["last_seen_ts"] or 0.0, dst_row["last_seen_ts"] or 0.0)
                conn.execute(
                    "UPDATE people SET last_seen_ts = ? WHERE id = ?", (latest, dst_id)
                )

            conn.execute("DELETE FROM people WHERE id = ?", (src_id,))
            conn.commit()
        finally:
            conn.close()

    # Best-effort crop merge (outside the DB transaction): copy src crops into dst, then drop
    # src's crop dir. save_crop keeps only the newest MAX_CROPS afterwards.
    try:
        src_dir = crops_dir(src_id, create=False)
        for p in _ordered_crop_files(src_id):
            try:
                with open(p, "rb") as f:
                    save_crop(dst_id, f.read())
            except OSError:
                pass
        if os.path.isdir(src_dir):
            shutil.rmtree(src_dir, ignore_errors=True)
    except OSError:
        pass


def delete_person(person_id: int) -> None:
    """Delete a person and ALL their rows (embeddings, sightings, events, activity_log) and
    their face-crops directory. (Privacy: "delete a person" erases everything about them.)"""
    person_id = int(person_id)
    with _write_lock:
        conn = connect()
        try:
            conn.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM sightings WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM events WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM activity_log WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
            conn.commit()
        finally:
            conn.close()
    try:
        d = crops_dir(person_id, create=False)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# sightings / events / timeline
# ---------------------------------------------------------------------------
def add_sighting(person_id: int, ts: float, box) -> None:
    """Record a sighting. ``box`` is ``(x, y, w, h)``."""
    x, y, w, h = box
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO sightings (person_id, ts, x, y, w, h) VALUES (?, ?, ?, ?, ?, ?)",
                (int(person_id), float(ts), int(x), int(y), int(w), int(h)),
            )
            conn.commit()
        finally:
            conn.close()


def _payload_to_text(payload) -> "str | None":
    """Serialise an Event.payload dict to JSON text, tolerating non-JSON values."""
    if payload is None:
        return None
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(payload))


def _payload_from_text(text) -> "dict | None":
    """Parse stored payload JSON text back to a dict (None / bad text -> None)."""
    if not text:
        return None
    try:
        val = json.loads(text)
    except (TypeError, ValueError):
        return None
    return val if isinstance(val, dict) else None


def _row_to_event(row: sqlite3.Row) -> dict:
    """Convert an events row to a dict, decoding ``payload`` JSON back to a dict."""
    d = dict(row)
    d["payload"] = _payload_from_text(d.get("payload"))
    return d


def add_event(event: Event) -> int:
    """Insert an :class:`Event` dataclass as a row; return the new event id.

    ``payload`` is stored as JSON text. Maps every Event field onto its column
    (``type/action/object/location/text/confidence/source/thumb_ref``).
    """
    with _write_lock:
        conn = connect()
        try:
            cur = conn.execute(
                "INSERT INTO events "
                "(ts, person_id, type, action, object, location, text, confidence, source, "
                " thumb_ref, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    float(event.ts),
                    None if event.person_id is None else int(event.person_id),
                    event.type,
                    event.action,
                    event.object,
                    event.location,
                    event.text or "",
                    float(event.confidence),
                    event.source,
                    event.thumb_ref,
                    _payload_to_text(event.payload),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def get_events(person_id=None, limit: int = 200, since=None) -> "list[dict]":
    """Return events newest-first as dicts, with optional filters.

    ``person_id`` restricts to one subject (``None`` = all, including scene-level events).
    ``since`` (a unix ts) restricts to ``ts >= since``. ``limit`` caps the row count.
    ``payload`` is decoded back to a dict.
    """
    clauses: "list[str]" = []
    params: "list" = []
    if person_id is not None:
        clauses.append("person_id = ?")
        params.append(int(person_id))
    if since is not None:
        clauses.append("ts >= ?")
        params.append(float(since))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, ts, person_id, type, action, object, location, text, confidence, "
            "source, thumb_ref, payload FROM events" + where +
            " ORDER BY ts DESC, id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [_row_to_event(r) for r in rows]
    finally:
        conn.close()


def get_timeline(person_id: int, limit: int = 200) -> "list[dict]":
    """Return up to ``limit`` of a person's events, NEWEST FIRST (an events view of one subject).

    Each row is an event dict (see :func:`get_events`); ``payload`` decoded to a dict.
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, ts, person_id, type, action, object, location, text, confidence, "
            "source, thumb_ref, payload FROM events "
            "WHERE person_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
            (int(person_id), int(limit)),
        ).fetchall()
        return [_row_to_event(r) for r in rows]
    finally:
        conn.close()


def add_activity(person_id: int, ts: float, text: str, provider: str) -> None:
    """Append a back-compat freeform-caption row to ``activity_log``."""
    with _write_lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO activity_log (person_id, ts, text, provider) VALUES (?, ?, ?, ?)",
                (int(person_id), float(ts), text, provider),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# centroids (for the recognizer to load into memory)
# ---------------------------------------------------------------------------
def load_centroids() -> "list[dict]":
    """Return ``[{id, name, kind, centroid}]`` for every person with a usable centroid.

    ``centroid`` is a float32 numpy array, L2-normalised. People whose centroid is NULL or empty
    are skipped.
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, name, kind, centroid FROM people ORDER BY id ASC"
        ).fetchall()
        out = []
        for r in rows:
            vec = _from_blob(r["centroid"])
            if vec is None or vec.size == 0:
                continue
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "kind": r["kind"],
                    "centroid": _normalize(vec),
                }
            )
        return out
    finally:
        conn.close()


def get_centroid(person_id: int) -> "np.ndarray | None":
    """Return one person's current centroid (L2-normalised float32), or ``None`` if missing/empty.

    Used by the recogniser's running-mean centroid update (Phase E) so it can nudge the *stored*
    centroid without keeping a parallel in-memory copy.
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT centroid FROM people WHERE id = ?", (int(person_id),)
        ).fetchone()
        if row is None:
            return None
        vec = _from_blob(row["centroid"])
        if vec is None or vec.size == 0:
            return None
        return _normalize(vec)
    finally:
        conn.close()


def load_templates(k: int = 4) -> "list[dict]":
    """Return up to ``k`` matching templates PER PERSON as ``[{id, name, kind, vec}]`` (Phase E).

    Single-centroid matching fragments one identity into several "Unknown #N" clusters on a low-res
    panning feed, because one mean vector can't cover the pose/lighting spread of a real face. So the
    recogniser matches against several exemplars per person and takes the best similarity. Each
    person contributes:

      * its **centroid** (the smooth running mean), then
      * its top ``k-1`` highest-``quality`` stored **embeddings** (distinct shots),

    every vector L2-normalised. People with no usable centroid are skipped. ``k<=1`` degenerates to
    centroid-only (equivalent to :func:`load_centroids`). One row per (person, template); the
    recogniser maps each row back to its ``id``.
    """
    k = max(1, int(k))
    conn = connect()
    try:
        prows = conn.execute(
            "SELECT id, name, kind, centroid FROM people ORDER BY id ASC"
        ).fetchall()
        out: "list[dict]" = []
        for pr in prows:
            cvec = _from_blob(pr["centroid"])
            templates: "list[np.ndarray]" = []
            if cvec is not None and cvec.size:
                templates.append(_normalize(cvec))
            need = k - len(templates)
            if need > 0:
                erows = conn.execute(
                    "SELECT vec FROM embeddings WHERE person_id = ? "
                    "ORDER BY quality DESC, id DESC LIMIT ?",
                    (pr["id"], need),
                ).fetchall()
                for er in erows:
                    ev = _from_blob(er["vec"])
                    if ev is not None and ev.size:
                        templates.append(_normalize(ev))
            for t in templates:
                out.append(
                    {"id": pr["id"], "name": pr["name"], "kind": pr["kind"], "vec": t}
                )
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def prune(retention_days) -> "tuple[int, int]":
    """Delete sightings + events older than ``now - retention_days*86400`` (and old activity_log).

    People and embeddings are KEPT (labelled identities + face vectors are permanent). Returns
    ``(sightings_deleted, events_deleted)``.
    """
    cutoff = time.time() - float(retention_days) * 86400.0
    with _write_lock:
        conn = connect()
        try:
            c_sight = conn.execute(
                "DELETE FROM sightings WHERE ts < ?", (cutoff,)
            ).rowcount
            c_evt = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
            conn.execute("DELETE FROM activity_log WHERE ts < ?", (cutoff,))
            conn.commit()
            return (int(c_sight or 0), int(c_evt or 0))
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# face-crop helpers  (<data_dir>/faces/<id>/0.jpg .. (MAX_CROPS-1).jpg)
# ---------------------------------------------------------------------------
def crops_dir(person_id: int, create: bool = True) -> str:
    """Return the crops directory path for a person (creating it if ``create``)."""
    d = os.path.join(FACES_DIR, str(int(person_id))).replace("\\", "/")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _ordered_crop_files(person_id: int) -> "list[str]":
    """Existing crop file paths for a person, newest first (0.jpg is newest)."""
    d = crops_dir(person_id, create=False)
    out: "list[str]" = []
    if os.path.isdir(d):
        for i in range(MAX_CROPS):
            p = os.path.join(d, f"{i}.jpg").replace("\\", "/")
            if os.path.isfile(p):
                out.append(p)
    return out


def save_crop(person_id: int, jpg_bytes: bytes) -> None:
    """Save a new JPEG crop as the newest (0.jpg), rotating older ones up to MAX_CROPS.

    Rotation: drop ``(MAX_CROPS-1).jpg``, then shift ``i.jpg -> (i+1).jpg`` for each slot, then
    write the new bytes to ``0.jpg``. Best-effort; any OS error is swallowed so a crop failure
    never breaks the pipeline.
    """
    if not jpg_bytes:
        return
    with _write_lock:
        try:
            d = crops_dir(person_id, create=True)
            oldest = os.path.join(d, f"{MAX_CROPS - 1}.jpg").replace("\\", "/")
            if os.path.isfile(oldest):
                try:
                    os.remove(oldest)
                except OSError:
                    pass
            for i in range(MAX_CROPS - 2, -1, -1):
                src = os.path.join(d, f"{i}.jpg").replace("\\", "/")
                dst = os.path.join(d, f"{i + 1}.jpg").replace("\\", "/")
                if os.path.isfile(src):
                    try:
                        os.replace(src, dst)
                    except OSError:
                        pass
            with open(os.path.join(d, "0.jpg").replace("\\", "/"), "wb") as f:
                f.write(jpg_bytes)
        except OSError:
            pass


def list_crops(person_id: int) -> "list[str]":
    """Return existing crop file paths for a person, newest first (0.jpg .. (N-1).jpg)."""
    return _ordered_crop_files(person_id)


# ---------------------------------------------------------------------------
# event thumbnails  (<data_dir>/thumbs/<ref>.jpg)
# ---------------------------------------------------------------------------
def save_thumb(jpg_bytes: bytes) -> str:
    """Write a uniquely-keyed JPEG thumbnail under THUMBS_DIR; return its ``thumb_ref`` key.

    The key is a short uuid (no extension); :func:`thumb_path` resolves it. Best-effort: the file
    is still keyed even if the write fails (so the caller never raises), but in practice the write
    succeeds. Empty input returns an empty string.
    """
    if not jpg_bytes:
        return ""
    ref = uuid.uuid4().hex
    try:
        os.makedirs(THUMBS_DIR, exist_ok=True)
        path = os.path.join(THUMBS_DIR, f"{ref}.jpg").replace("\\", "/")
        with open(path, "wb") as f:
            f.write(jpg_bytes)
    except OSError:
        pass
    return ref


def thumb_path(thumb_ref: str) -> "str | None":
    """Resolve a ``thumb_ref`` key to its on-disk JPEG path, or ``None`` if missing/unset."""
    if not thumb_ref:
        return None
    # Defend against a stray ".jpg" or path separators in the stored ref.
    ref = os.path.basename(str(thumb_ref))
    if ref.endswith(".jpg"):
        ref = ref[:-4]
    if not ref:
        return None
    path = os.path.join(THUMBS_DIR, f"{ref}.jpg").replace("\\", "/")
    return path if os.path.isfile(path) else None


if __name__ == "__main__":
    init_db()
    print(f"initialised {DB_PATH}")
    print(f"people: {len(list_people())}")
