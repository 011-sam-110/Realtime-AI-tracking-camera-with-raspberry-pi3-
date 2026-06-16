"""Offline tests for the FastAPI web layer (INTERFACES.md §11).

Exercises the REAL ``kitchenvision.api.app`` against a TEMPORARY data dir and a
``fastapi.testclient.TestClient`` — NO pipeline, NO camera, NO servo, NO models, NO network.

How the temp DB is wired
------------------------
``store.db`` resolves its path constants from ``config.load_config()["data_dir"]`` ONCE at import
time. The app reads/writes through the ``store.db`` helpers (which read those constants live), so we
repoint the constants into a fresh ``tempfile.mkdtemp()`` and re-init the schema BEFORE driving any
route. This must happen before ``app`` issues any DB call, but since the routes only touch the DB at
request time, importing ``app`` first is fine — we patch the constants, then make requests.

The pipeline is never started; ``app._refresh_identities`` lazily imports + best-effort-calls
``pipeline.refresh_identities`` which, with no pipeline running, is a harmless no-op.

Plain-python runnable (pytest may be absent):  python tests/test_api.py
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

from fastapi.testclient import TestClient  # noqa: E402

from kitchenvision.core.types import Event  # noqa: E402
from kitchenvision.store import db  # noqa: E402
from kitchenvision.api import app as app_mod  # noqa: E402
from kitchenvision.core.state import STATE  # noqa: E402


# --------------------------------------------------------------------------- temp-dir harness
def _point_db_at_tmp(tmp: str) -> None:
    """Repoint every path constant in ``store.db`` into ``tmp`` and (re)create a fresh schema."""
    db.DATA_DIR = tmp.replace("\\", "/")
    db.DB_PATH = os.path.join(tmp, "kitchenvision.db").replace("\\", "/")
    db.FACES_DIR = os.path.join(tmp, "faces").replace("\\", "/")
    db.THUMBS_DIR = os.path.join(tmp, "thumbs").replace("\\", "/")
    db.init_db()


def _reset_state() -> None:
    """Clear STATE so each test starts from a clean live snapshot (no camera involved)."""
    with STATE.lock:
        STATE.latest_jpeg = None
        STATE.latest_raw = None
        STATE.people = []
        STATE.activity_status = {"state": "ok", "message": ""}
        STATE.servo_angle = 90.0
        STATE.servo_angles = {"pan": 90.0}
        STATE.fps = 0.0


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _make_request(path: str):
    """A minimal Starlette ``Request`` for driving a route's StreamingResponse directly.

    ``receive`` never reports an http.disconnect, so ``request._is_disconnected`` stays False and
    ``await request.is_disconnected()`` returns False — the streaming generators will yield at least
    one part/tick (which is all the tests pull before closing the iterator by hand).
    """
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
    }

    async def _receive():
        # Block forever if the generator ever polls again — but the tests only read one item then
        # close the iterator, so this is never awaited to completion in practice.
        import anyio
        await anyio.sleep_forever()

    return Request(scope, receive=_receive)


# --------------------------------------------------------------------------- assertions
def _eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def _true(cond, msg=""):
    if not cond:
        raise AssertionError(f"not true: {msg}")


# --------------------------------------------------------------------------- tests
def test_people_crud_and_timeline() -> None:
    """People list/rename/merge/delete + timeline through the HTTP layer."""
    client = TestClient(app_mod.app)

    p1 = db.create_person(_vec(1), kind="unknown")
    p2 = db.create_person(_vec(2), kind="unknown")
    db.add_embedding(p1, _vec(11))
    db.add_embedding(p2, _vec(22))

    # GET /api/people — both present, with crop-url fields.
    r = client.get("/api/people")
    _eq(r.status_code, 200, "list status")
    people = r.json()["people"]
    _eq(len(people), 2, "people count")
    ids = {p["id"] for p in people}
    _eq(ids, {p1, p2}, "people ids")
    for p in people:
        _true("crop_url" in p and "crop_urls" in p and "crop_count" in p, "crop fields present")
        _true(p["label"].startswith("Unknown #"), "unknown label")

    # POST name -> known + label updates.
    r = client.post(f"/api/people/{p1}/name", json={"name": "Sam"})
    _eq(r.status_code, 200, "name status")
    _eq(r.json()["person"]["label"], "Sam", "renamed label")
    _eq(db.get_person(p1)["kind"], "known", "kind flipped to known")

    # blank name -> 400; missing person -> 404.
    _eq(client.post(f"/api/people/{p1}/name", json={"name": "  "}).status_code, 400, "blank 400")
    _eq(client.post("/api/people/999999/name", json={"name": "X"}).status_code, 404, "missing 404")

    # timeline (empty initially) for a real person, 404 for a missing one.
    r = client.get(f"/api/people/{p1}/timeline")
    _eq(r.status_code, 200, "timeline status")
    _eq(r.json()["timeline"], [], "timeline empty")
    _eq(client.get("/api/people/999999/timeline").status_code, 404, "timeline 404")

    # add an event to p2 then merge p2 INTO p1; the event must follow.
    db.add_event(Event(type="chore", ts=time.time(), person_id=p2, action="left",
                       object="plate", location="table", text="left a plate"))
    r = client.post(f"/api/people/{p2}/merge", json={"into": p1})
    _eq(r.status_code, 200, "merge status")
    _true(db.get_person(p2) is None, "src deleted after merge")
    tl = client.get(f"/api/people/{p1}/timeline").json()["timeline"]
    _eq(len(tl), 1, "merged event in dst timeline")
    _eq(tl[0]["object"], "plate", "merged event object")

    # merge into self -> 400; merge missing -> 404.
    _eq(client.post(f"/api/people/{p1}/merge", json={"into": p1}).status_code, 400, "self merge 400")
    _eq(client.post(f"/api/people/{p1}/merge", json={"into": "x"}).status_code, 400, "bad into 400")

    # DELETE p1.
    _eq(client.delete(f"/api/people/{p1}").status_code, 200, "delete status")
    _true(db.get_person(p1) is None, "deleted")
    _eq(client.delete(f"/api/people/{p1}").status_code, 404, "delete-again 404")
    _eq(len(client.get("/api/people").json()["people"]), 0, "empty roster")


def test_events_endpoint() -> None:
    """GET /api/events newest-first with since / person_id / limit filters."""
    client = TestClient(app_mod.app)
    pid = db.create_person(_vec(3), kind="known")

    t0 = time.time()
    db.add_event(Event(type="activity", ts=t0 - 100, person_id=pid, text="old"))
    db.add_event(Event(type="chore", ts=t0 - 10, person_id=pid, text="recent"))
    db.add_event(Event(type="presence", ts=t0 - 5, person_id=None, text="scene"))

    all_evts = client.get("/api/events").json()["events"]
    _eq(len(all_evts), 3, "all events")
    _eq(all_evts[0]["text"], "scene", "newest first")  # ts=-5 is newest

    # since filter drops the old one.
    since_evts = client.get("/api/events", params={"since": t0 - 50}).json()["events"]
    _eq(len(since_evts), 2, "since filter")
    _true(all(e["text"] != "old" for e in since_evts), "old dropped by since")

    # person_id filter excludes the scene-level event.
    pe = client.get("/api/events", params={"person_id": pid}).json()["events"]
    _eq(len(pe), 2, "person filter")
    _true(all(e["person_id"] == pid for e in pe), "only this person")

    # limit caps.
    lim = client.get("/api/events", params={"limit": 1}).json()["events"]
    _eq(len(lim), 1, "limit 1")


def test_stats_endpoint() -> None:
    """GET /api/stats per-person aggregates computed from sightings + events."""
    client = TestClient(app_mod.app)
    pid = db.create_person(_vec(4), kind="known")
    db.set_name(pid, "Alex")

    base = time.time() - 1000.0
    # two close sightings (one continuous visit) then a far one (a second visit).
    db.add_sighting(pid, base, (0, 0, 50, 50))
    db.add_sighting(pid, base + 5, (1, 1, 50, 50))
    db.add_sighting(pid, base + 500, (2, 2, 50, 50))
    db.add_event(Event(type="chore", ts=base + 6, person_id=pid, text="washed up"))
    db.add_event(Event(type="activity", ts=base + 7, person_id=pid, text="standing"))

    r = client.get("/api/stats")
    _eq(r.status_code, 200, "stats status")
    body = r.json()
    # New §11 shape (matches web/src/api/types.ts StatsResponse): {people:[...], window_days}.
    _true("people" in body, "stats has people[]")
    _true("window_days" in body, "stats has window_days")
    stats = body["people"]
    me = next(s for s in stats if s["person_id"] == pid)
    _eq(me["label"], "Alex", "stats label")
    _eq(me["sighting_count"], 3, "sighting count")
    _eq(me["visits"], 2, "two visits")
    # totals consumed by the SPA charts:
    _true(me["total_presence_seconds"] >= 5.0, "presence accrued from the close pair")
    _eq(me["total_event_count"], 2, "event count")
    _eq(me["total_chore_count"], 1, "chore count")
    _eq(me["activity_count"], 1, "activity count")
    _true(me["first_seen"] is not None and me["last_seen"] is not None, "seen bounds set")
    # daily time series for the "presence over time" line chart.
    _true(isinstance(me["buckets"], list) and len(me["buckets"]) >= 1, "buckets present")
    b0 = me["buckets"][0]
    _true({"bucket", "presence_seconds", "event_count", "chore_count"} <= set(b0), "bucket shape")


def test_thumbs_endpoint() -> None:
    """GET /api/thumbs/{ref}.jpg serves a saved thumbnail; unknown ref -> 404."""
    client = TestClient(app_mod.app)
    jpg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"thumbnail-bytes"
    ref = db.save_thumb(jpg)
    _true(bool(ref), "thumb saved with a ref")

    r = client.get(f"/api/thumbs/{ref}.jpg")
    _eq(r.status_code, 200, "thumb status")
    _eq(r.headers["content-type"], "image/jpeg", "thumb content-type")
    _eq(r.content, jpg, "thumb bytes round-trip")

    _eq(client.get("/api/thumbs/deadbeef.jpg").status_code, 404, "unknown thumb 404")


def test_faces_endpoint() -> None:
    """GET /api/faces/{id}/{n}.jpg serves a saved crop; out-of-range -> 404."""
    client = TestClient(app_mod.app)
    pid = db.create_person(_vec(5), kind="unknown")
    crop = bytes([0xFF, 0xD8, 0xFF, 0xDB]) + b"face-crop"
    db.save_crop(pid, crop)

    r = client.get(f"/api/faces/{pid}/0.jpg")
    _eq(r.status_code, 200, "face status")
    _eq(r.content, crop, "face bytes round-trip")
    _eq(client.get(f"/api/faces/{pid}/9.jpg").status_code, 404, "out-of-range 404")


def test_video_yields_a_frame() -> None:
    """GET /video yields a multipart JPEG part once STATE.latest_jpeg is set (no camera).

    The route's MJPEG generator is an INFINITE loop (it keeps resending the latest frame), so we
    drive its ``StreamingResponse.body_iterator`` directly and pull the FIRST part, rather than via
    ``TestClient.stream`` whose context-exit would block trying to drain an unending stream. This is
    a faithful test of the exact bytes the route produces — just without the never-ending teardown.
    """
    import anyio

    fake_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"fake-annotated-frame" + bytes([0xFF, 0xD9])
    with STATE.lock:
        STATE.latest_jpeg = fake_jpeg

    # A minimal Request whose disconnect probe stays False (so the generator yields a frame).
    req = _make_request("/video")
    resp = app_mod.video(req)
    _true(resp.media_type.startswith("multipart/x-mixed-replace"), "video mime")

    async def _first_part() -> bytes:
        it = resp.body_iterator
        try:
            async for part in it:
                return part if isinstance(part, bytes) else part.encode()
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()
        return b""

    chunk = anyio.run(_first_part)
    _true(b"--FRAME" in chunk, "boundary present")
    _true(b"Content-Type: image/jpeg" in chunk, "part header present")
    _true(fake_jpeg in chunk, "the JPEG payload is in the part")


def test_events_one_tick_ws_and_sse() -> None:
    """One live tick over both the WebSocket and the SSE fallback at /events."""
    client = TestClient(app_mod.app)
    with STATE.lock:
        STATE.people = [{
            "person_id": 7, "label": "Sam", "kind": "known", "box": [10, 20, 30, 40],
            "track_id": 1, "age": 30.0, "sex": "M", "current_activity": "washing up",
            "last_seen_ts": time.time(),
        }]
        STATE.activity_status = {"state": "ok", "message": ""}
        STATE.servo_angles = {"pan": 95.0}
        STATE.servo_angle = 95.0
        STATE.fps = 12.5

    # WebSocket: receive one frame.
    with client.websocket_connect("/events") as ws:
        import json as _json
        payload = _json.loads(ws.receive_text())
    _eq(set(["people", "activity_status", "servo_angles", "fps"]) - set(payload), set(),
        "ws payload has the required keys")
    _eq(payload["fps"], 12.5, "ws fps")
    _eq(payload["servo_angles"]["pan"], 95.0, "ws servo pan")
    _eq(payload["people"][0]["label"], "Sam", "ws person label")
    _eq(payload["people"][0]["current_activity"], "washing up", "ws activity")

    # SSE fallback: drive the route's StreamingResponse body iterator directly and read the first
    # `data:` frame (the SSE generator is infinite, like /video, so we close the iterator by hand).
    import anyio
    import json as _json

    sse_resp = anyio.run(app_mod.events_sse, _make_request("/events"))
    _true(sse_resp.media_type.startswith("text/event-stream"), "sse mime")

    async def _first_sse() -> str:
        it = sse_resp.body_iterator
        try:
            async for item in it:
                return item if isinstance(item, str) else item.decode()
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()
        return ""

    frame = anyio.run(_first_sse)
    _true(frame.startswith("data:"), "sse data frame")
    sse_payload = _json.loads(frame[len("data:"):].strip())
    _eq(sse_payload["people"][0]["label"], "Sam", "sse person label")
    _eq(sse_payload["fps"], 12.5, "sse fps")


def test_config_endpoint() -> None:
    """GET /api/config returns a read-only, secret-free config subset for the Settings view."""
    client = TestClient(app_mod.app)
    r = client.get("/api/config")
    _eq(r.status_code, 200, "config status")
    cfg = r.json()
    _true("retention_days" in cfg, "config has retention_days")
    _true("recog_threshold" in cfg.get("recognition", {}), "config has recog_threshold")
    _true("backend" in cfg.get("vlm", {}), "config has vlm.backend")
    # The cloud api_key must NEVER be exposed — only a presence flag.
    _true("api_key" not in cfg.get("vlm", {}), "api_key not leaked")
    _true("cloud_configured" in cfg.get("vlm", {}), "config has cloud_configured flag")


def test_root_serves_something() -> None:
    """GET / serves the SPA index or the placeholder (never crashes when web/dist is absent)."""
    client = TestClient(app_mod.app)
    r = client.get("/")
    _eq(r.status_code, 200, "root status")
    _true("text/html" in r.headers["content-type"], "root html")
    _true(len(r.content) > 0, "root has a body")


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        test_people_crud_and_timeline,
        test_events_endpoint,
        test_stats_endpoint,
        test_config_endpoint,
        test_thumbs_endpoint,
        test_faces_endpoint,
        test_video_yields_a_frame,
        test_events_one_tick_ws_and_sse,
        test_root_serves_something,
    ]
    tmp = tempfile.mkdtemp(prefix="kv_api_test_")
    failed = 0
    try:
        for t in tests:
            _point_db_at_tmp(tmp)          # fresh schema per test (drop + recreate dir)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
            _point_db_at_tmp(tmp)
            _reset_state()
            try:
                t()
                print(f"PASS {t.__name__}")
            except Exception as e:
                failed += 1
                print(f"FAIL {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


# pytest fixtures: ensure each pytest-collected test also gets a temp DB + clean STATE.
try:
    import pytest

    @pytest.fixture(autouse=True)
    def _tmp_db():
        # Save the original path constants so teardown restores the shared db module state — leaving
        # it pointing at a since-deleted temp dir would break later test files (this module mutates
        # GLOBAL constants on the `db` module).
        saved = (db.DATA_DIR, db.DB_PATH, db.FACES_DIR, db.THUMBS_DIR)
        tmp = tempfile.mkdtemp(prefix="kv_api_test_")
        _point_db_at_tmp(tmp)
        _reset_state()
        try:
            yield
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            db.DATA_DIR, db.DB_PATH, db.FACES_DIR, db.THUMBS_DIR = saved
except ImportError:
    pass


if __name__ == "__main__":
    raise SystemExit(_run_all())
