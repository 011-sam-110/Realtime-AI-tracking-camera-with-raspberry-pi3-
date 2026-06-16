"""FastAPI web layer for the Kitchen Vision brain (INTERFACES.md §11).

`uvicorn` target: ``kitchenvision.api.app:app`` (started by ``__main__.py`` after
``pipeline.start()``). This module is intentionally THIN: every route either reads the shared
in-memory :data:`STATE` singleton (always under ``STATE.lock``) or opens its own short-lived
``store.db`` connection via the thread-safe helpers, and any identity mutation (rename / merge /
delete) is followed by ``pipeline.refresh_identities()`` so the running recogniser reloads its
centroids on the recognition thread.

ROUTES (INTERFACES.md §11)
--------------------------
* ``GET  /video``                       — annotated MJPEG (``multipart/x-mixed-replace``) of
  ``STATE.latest_jpeg``.
* ``GET  /events``                      — live ``{people, activity_status, servo_angles, fps}``;
  **WebSocket preferred** (``/events`` upgrades when the client requests it) with a **SSE
  fallback** (``/events`` GET → ``text/event-stream``) and an explicit ``/events/sse`` alias.
* ``GET  /api/people``                  — ``{people:[... +crop urls]}``.
* ``POST /api/people/{id}/name``        — body ``{name}`` → ``set_name`` + ``refresh_identities``.
* ``POST /api/people/{id}/merge``       — body ``{into}`` → merge {id} INTO {into} + refresh.
* ``DELETE /api/people/{id}``           — ``delete_person`` + refresh.
* ``GET  /api/people/{id}/timeline``    — ``{timeline: store.get_timeline(id)}``.
* ``GET  /api/events?since=&limit=&person_id=`` — ``{events:[...]}``.
* ``GET  /api/stats``                   — per-person aggregates (presence time, event/chore
  counts), computed from the DB.
* ``GET  /api/faces/{id}/{n}.jpg``      — the n-th face crop bytes.
* ``GET  /api/thumbs/{ref}.jpg``        — an event thumbnail by key.
* ``GET  /`` + static mount             — serves the built SPA from ``brain/web/dist`` with a
  SPA fallback to ``index.html``. Tolerates ``web/dist`` being absent (serves a placeholder).

THREADING
---------
uvicorn serves this app on worker threads. We never cache/share a sqlite connection across
requests — every DB-touching route calls the helpers in ``store.db`` (each opens + closes a fresh
connection). The MJPEG / SSE / WS streamers only ever read ``STATE`` under ``STATE.lock`` and copy
out before yielding, so they never hold the lock across a network write.

``pipeline.refresh_identities`` is imported LAZILY inside the mutation handlers so importing this
module (the ``--selfcheck`` gate) does NOT drag in the heavy pipeline import graph.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from kitchenvision.core import config as config_mod
from kitchenvision.store import db

# --- paths -------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../kitchenvision/api
_BRAIN = os.path.dirname(os.path.dirname(_HERE))                   # .../brain
WEB_DIST = os.path.join(_BRAIN, "web", "dist").replace("\\", "/")  # built SPA (may be absent)

# --- MJPEG framing (matches pi_agent.py /stream so any MJPEG client works) ----
_BOUNDARY = "FRAME"
_VIDEO_INTERVAL = 1.0 / 15.0      # resend the latest annotated JPEG at ~15 fps
_EVENTS_INTERVAL = 1.0 / 3.0      # push live STATE at ~3 Hz (within the §11 1-4 Hz band)


# ---------------------------------------------------------------------------
# app + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Kitchen Vision")

# Dev: the Vite dev server (localhost:5173) proxies most routes but CORS keeps direct calls /
# WS handshakes working from any localhost origin. In prod the SPA is same-origin (served below),
# so this is a no-op there.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _refresh_identities() -> None:
    """Lazily import + call ``pipeline.refresh_identities()`` after a people mutation.

    Imported lazily so this module imports cheaply + offline (the ``--selfcheck`` gate and the
    test harness must not pull in the heavy pipeline graph). Best-effort: if the pipeline never
    started (e.g. under TestClient) the call is a harmless no-op / swallowed error.
    """
    try:
        from kitchenvision import pipeline
        pipeline.refresh_identities()
    except Exception:
        pass


async def _json_body(request: Request) -> dict:
    """Parse a JSON request body, tolerating an empty/malformed body (→ ``{}``)."""
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _people_with_crops() -> list[dict]:
    """``db.list_people()`` enriched with crop URLs (newest crop + a count) for the gallery."""
    people = db.list_people()
    for p in people:
        crops = db.list_crops(p["id"])
        n = len(crops)
        p["crop_count"] = n
        p["has_crop"] = n > 0
        p["crop_url"] = f"/api/faces/{p['id']}/0.jpg" if n > 0 else None
        # All crop URLs newest-first, so the People view can show a strip.
        p["crop_urls"] = [f"/api/faces/{p['id']}/{i}.jpg" for i in range(n)]
    return people


def _snapshot_state() -> dict:
    """Build the live ``/events`` payload from ``STATE`` under the lock.

    Imports ``STATE`` lazily (kept out of module import for offline-safety) and copies out only the
    JSON-serialisable fields each per-person dict needs, INSIDE the lock — ``STATE.people`` dicts
    are shared with (and mutated in place by) the perception worker, so reading them outside the
    lock would be a data race. Never holds the lock across the network write.
    """
    from kitchenvision.core.state import STATE

    people: list[dict] = []
    with STATE.lock:
        for p in STATE.people:
            box = p.get("box")
            people.append(
                {
                    "person_id": p.get("person_id"),
                    "label": p.get("label", "") or "",
                    "kind": p.get("kind", "unknown") or "unknown",
                    "current_activity": p.get("current_activity", "") or "",
                    "box": [int(v) for v in box] if box is not None else None,
                    "track_id": p.get("track_id"),
                    "age": p.get("age"),
                    "sex": p.get("sex"),
                    "last_seen_ts": p.get("last_seen_ts"),
                }
            )
        status = dict(STATE.activity_status) if STATE.activity_status else {
            "state": "ok",
            "message": "",
        }
        servo_angles = dict(STATE.servo_angles) if STATE.servo_angles else {}
        servo_angle = float(STATE.servo_angle)
        fps = float(STATE.fps)

    return {
        "people": people,
        "activity_status": status,
        "servo_angles": servo_angles,
        "servo_angle": servo_angle,   # back-compat scalar (some clients still read it)
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# GET /video  — annotated MJPEG
# ---------------------------------------------------------------------------
@app.get("/video")
def video(request: Request) -> StreamingResponse:
    """Stream the latest annotated JPEG as ``multipart/x-mixed-replace`` (~15 fps).

    Reads ``STATE.latest_jpeg`` under the lock, copies the bytes out, and yields a fresh multipart
    part each tick. Stops promptly when the client disconnects. If no frame exists yet (pipeline
    not started / camera idle) it simply waits.
    """
    from kitchenvision.core.state import STATE

    def gen():
        last_send = 0.0
        while True:
            try:
                if request._is_disconnected:    # set by Starlette on socket drop
                    break
            except Exception:
                pass

            with STATE.lock:
                buf = STATE.latest_jpeg
            if buf is None:
                time.sleep(0.05)
                continue

            now = time.time()
            wait = _VIDEO_INTERVAL - (now - last_send)
            if wait > 0:
                time.sleep(wait)

            try:
                yield (
                    b"--" + _BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + ("Content-Length: %d\r\n\r\n" % len(buf)).encode()
                    + buf
                    + b"\r\n"
                )
            except (BrokenPipeError, ConnectionResetError, GeneratorExit):
                break
            last_send = time.time()

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=%s" % _BOUNDARY,
        headers={"Cache-Control": "no-cache, private", "Pragma": "no-cache"},
    )


# ---------------------------------------------------------------------------
# GET /events  — WebSocket (preferred) + SSE (fallback)
# ---------------------------------------------------------------------------
@app.websocket("/events")
async def events_ws(ws: WebSocket) -> None:
    """Preferred live channel: push ``_snapshot_state()`` over a WebSocket at ~3 Hz.

    Mounted at the SAME path as the SSE route below. Starlette dispatches by scope type, so a WS
    upgrade lands here and a plain GET lands on the SSE handler — clients pick their transport.
    The snapshot (which takes ``STATE.lock``) runs in a thread so the event loop never blocks.
    """
    await ws.accept()
    try:
        while True:
            payload = await asyncio.to_thread(_snapshot_state)
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(_EVENTS_INTERVAL)
    except (WebSocketDisconnect, RuntimeError):
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


def _sse_response(request: Request) -> StreamingResponse:
    """Build the SSE ``StreamingResponse`` that streams ``_snapshot_state()`` at ~3 Hz."""
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            payload = await asyncio.to_thread(_snapshot_state)
            yield "data: " + json.dumps(payload) + "\n\n"
            await asyncio.sleep(_EVENTS_INTERVAL)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events")
async def events_sse(request: Request) -> StreamingResponse:
    """SSE fallback for ``/events`` (a plain GET; the WS route handles upgrades on the same path)."""
    return _sse_response(request)


@app.get("/events/sse")
async def events_sse_alias(request: Request) -> StreamingResponse:
    """Explicit SSE alias for clients that cannot share the WS path."""
    return _sse_response(request)


# ---------------------------------------------------------------------------
# REST: people CRUD + timeline
# ---------------------------------------------------------------------------
@app.get("/api/people")
def api_people() -> dict:
    return {"people": _people_with_crops()}


@app.get("/api/people/{person_id}/timeline")
def api_timeline(person_id: int) -> Any:
    if db.get_person(person_id) is None:
        return JSONResponse({"error": "no such person"}, status_code=404)
    return {"timeline": db.get_timeline(person_id)}


@app.post("/api/people/{person_id}/name")
async def api_set_name(person_id: int, request: Request) -> Any:
    body = await _json_body(request)
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    if db.get_person(person_id) is None:
        return JSONResponse({"error": "no such person"}, status_code=404)
    db.set_name(person_id, name)
    _refresh_identities()
    return {"ok": True, "person": db.get_person(person_id)}


@app.post("/api/people/{person_id}/merge")
async def api_merge(person_id: int, request: Request) -> Any:
    body = await _json_body(request)
    into = body.get("into")
    try:
        into = int(into)
    except (TypeError, ValueError):
        return JSONResponse({"error": "'into' must be an integer person id"}, status_code=400)
    if into == person_id:
        return JSONResponse({"error": "cannot merge a person into itself"}, status_code=400)
    if db.get_person(person_id) is None or db.get_person(into) is None:
        return JSONResponse({"error": "no such person"}, status_code=404)
    # Merge {person_id} (src) INTO {into} (dst) — matches db.merge_people(src, dst).
    db.merge_people(person_id, into)
    _refresh_identities()
    return {"ok": True, "into": into}


@app.delete("/api/people/{person_id}")
def api_delete(person_id: int) -> Any:
    if db.get_person(person_id) is None:
        return JSONResponse({"error": "no such person"}, status_code=404)
    db.delete_person(person_id)
    _refresh_identities()
    return {"ok": True}


# ---------------------------------------------------------------------------
# REST: events feed
# ---------------------------------------------------------------------------
@app.get("/api/events")
def api_events(
    since: Optional[float] = None,
    limit: int = 200,
    person_id: Optional[int] = None,
) -> dict:
    """The structured event feed, newest-first, with optional ``since`` / ``person_id`` filters."""
    try:
        limit = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        limit = 200
    return {"events": db.get_events(person_id=person_id, limit=limit, since=since)}


# ---------------------------------------------------------------------------
# REST: per-person analytics aggregates
# ---------------------------------------------------------------------------
def _compute_stats() -> dict:
    """Per-person aggregates + a daily time series for the Analytics view, computed from the DB.

    Returns the shape the SPA consumes (``web/src/api/types.ts`` ``StatsResponse``):
    ``{"people": [PersonStats...], "window_days": int}`` where each ``PersonStats`` is::

        { person_id, label, total_presence_seconds, total_chore_count, total_event_count,
          buckets: [ { bucket: "YYYY-MM-DD", presence_seconds, chore_count, event_count } ... ] }

    INTERFACES §11 deliberately does not pin the exact JSON; the SPA's richer shape (with a
    per-day time series for the "Presence over time" line chart) is the authoritative one, so the
    API emits it directly — no client adapter needed. Extra per-person fields (visits, kind, ...)
    are included too (harmless to the typed client).

    Presence is estimated by chaining consecutive sightings: each gap up to ``_PRESENCE_GAP`` s
    counts as continuous presence; a longer gap (or a lone sighting) credits a small floor and opens
    a new visit. Buckets group that presence + events by UTC day. One connection, grouped queries.
    """
    import datetime as _dt

    _PRESENCE_GAP = 30.0       # s: sightings closer than this are one continuous visit
    _SINGLE_FLOOR = 2.0        # s credited to an isolated sighting

    def _day(ts: float) -> str:
        return _dt.datetime.fromtimestamp(
            float(ts), _dt.timezone.utc
        ).strftime("%Y-%m-%d")

    window_days = int(config_mod.load_config().get("retention_days", 30) or 30)

    people = db.list_people()
    if not people:
        return {"people": [], "window_days": window_days}

    conn = db.connect()
    try:
        # presence + sighting bounds + per-day presence, in one pass over sightings (person, ts).
        runs: dict[int, dict] = {}
        prev_ts: dict[int, float] = {}
        # per-person per-day: {pid: {day: {"presence": s, "events": n, "chores": n}}}
        buckets: dict[int, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"presence": 0.0, "events": 0, "chores": 0})
        )
        for r in conn.execute(
            "SELECT person_id, ts FROM sightings ORDER BY person_id ASC, ts ASC"
        ):
            pid = r["person_id"]
            ts = float(r["ts"])
            acc = runs.setdefault(
                pid,
                {"presence": 0.0, "visits": 0, "count": 0, "first": ts, "last": ts},
            )
            acc["count"] += 1
            acc["first"] = min(acc["first"], ts)
            acc["last"] = max(acc["last"], ts)
            if pid in prev_ts:
                gap = ts - prev_ts[pid]
                add = gap if 0 <= gap <= _PRESENCE_GAP else _SINGLE_FLOOR
                if not (0 <= gap <= _PRESENCE_GAP):
                    acc["visits"] += 1             # a new visit started
            else:
                add = _SINGLE_FLOOR
                acc["visits"] += 1                 # first sighting opens the first visit
            acc["presence"] += add
            buckets[pid][_day(ts)]["presence"] += add
            prev_ts[pid] = ts

        # event counts grouped by (person, type) for the totals.
        evt_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        evt_total: dict[int, int] = defaultdict(int)
        for r in conn.execute(
            "SELECT person_id, type, COUNT(*) AS c FROM events "
            "WHERE person_id IS NOT NULL GROUP BY person_id, type"
        ):
            pid = r["person_id"]
            evt_counts[pid][r["type"]] = int(r["c"])
            evt_total[pid] += int(r["c"])

        # per-day event + chore counts for the time series.
        for r in conn.execute(
            "SELECT person_id, ts, type FROM events WHERE person_id IS NOT NULL"
        ):
            pid = r["person_id"]
            day = buckets[pid][_day(r["ts"])]
            day["events"] += 1
            if r["type"] == "chore":
                day["chores"] += 1
    finally:
        conn.close()

    out: list[dict] = []
    for p in people:
        pid = p["id"]
        run = runs.get(pid)
        types = evt_counts.get(pid, {})
        day_map = buckets.get(pid, {})
        series = [
            {
                "bucket": day,
                "presence_seconds": round(v["presence"], 1),
                "event_count": int(v["events"]),
                "chore_count": int(v["chores"]),
            }
            for day, v in sorted(day_map.items())
        ]
        out.append(
            {
                "person_id": pid,
                "id": pid,
                "label": p["label"],
                "name": p.get("name"),
                "kind": p.get("kind"),
                # totals consumed by the SPA charts:
                "total_presence_seconds": round(run["presence"], 1) if run else 0.0,
                "total_event_count": evt_total.get(pid, 0),
                "total_chore_count": int(types.get("chore", 0)),
                # extra per-person detail (harmless to the typed client):
                "visits": run["visits"] if run else 0,
                "sighting_count": run["count"] if run else 0,
                "first_seen": run["first"] if run else None,
                "last_seen": run["last"] if run else p.get("last_seen_ts"),
                "activity_count": int(types.get("activity", 0)),
                "object_count": int(types.get("object", 0)),
                "buckets": series,
            }
        )
    return {"people": out, "window_days": window_days}


@app.get("/api/stats")
def api_stats() -> dict:
    """Per-person analytics aggregates + daily time series for the Analytics view.

    Emits ``{people, window_days}`` (the SPA's ``StatsResponse`` shape) directly so the React
    Analytics view needs no adapter — see ``_compute_stats`` for the field contract.
    """
    return _compute_stats()


# ---------------------------------------------------------------------------
# REST: read-only config (Settings view)
# ---------------------------------------------------------------------------
@app.get("/api/config")
def api_config() -> dict:
    """A read-only, SECRET-FREE subset of the live config for the Settings view.

    The SPA's ``getConfig()`` tolerates a 404 (showing documented defaults), but serving this makes
    the Settings view reflect the ACTUAL running config. The cloud ``api_key`` is deliberately
    NEVER returned (only whether creds are present), so the dashboard never leaks the gateway key.
    """
    cfg = config_mod.load_config()
    vlm = cfg.get("vlm", {}) or {}
    recog = cfg.get("recognition", {}) or {}
    activity = cfg.get("activity", {}) or {}
    return {
        "retention_days": cfg.get("retention_days", 30),
        "recognition": {
            "recog_threshold": recog.get("recog_threshold", 0.45),
            "engine": recog.get("engine", "insightface"),
            "model": recog.get("model", "buffalo_l"),
        },
        "vlm": {
            "backend": vlm.get("backend", "local"),
            "local_model": vlm.get("local_model", "qwen2-vl-2b"),
            "model": vlm.get("model", "auto"),
            # creds presence only — never the api_key itself.
            "cloud_configured": bool((vlm.get("base_url") or "").strip()
                                     and (vlm.get("api_key") or "").strip()),
        },
        "activity": {
            "enabled": activity.get("enabled", True),
            "cadence_seconds": activity.get("cadence_seconds", 10),
            "min_confidence": activity.get("min_confidence", 0.0),
        },
    }


# ---------------------------------------------------------------------------
# REST: image bytes (face crops + event thumbnails)
# ---------------------------------------------------------------------------
@app.get("/api/faces/{person_id}/{n}.jpg")
def api_face(person_id: int, n: int) -> Response:
    """The n-th face crop for a person (0 = newest). 404 if out of range / missing."""
    crops = db.list_crops(person_id)   # newest first (0.jpg first)
    if n < 0 or n >= len(crops):
        return Response(status_code=404)
    path = crops[n]
    if not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/api/thumbs/{ref}.jpg")
def api_thumb(ref: str) -> Response:
    """An event thumbnail by its ``thumb_ref`` key (404 if unknown/missing)."""
    path = db.thumb_path(ref)
    if not path or not os.path.isfile(path):
        return Response(status_code=404)
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
    )


# ---------------------------------------------------------------------------
# SPA: serve the built React app from web/dist (tolerate it being absent)
# ---------------------------------------------------------------------------
_PLACEHOLDER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kitchen Vision</title>
<style>
  body{margin:0;background:#0f1115;color:#e8eaf0;
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       display:flex;min-height:100vh;align-items:center;justify-content:center}
  .card{max-width:620px;padding:32px 36px;background:#171a21;border:1px solid #2a2f3a;
        border-radius:16px}
  h1{font-size:20px;margin:0 0 12px}
  code{background:#1e222b;padding:2px 7px;border-radius:6px;color:#9cd0ff}
  a{color:#4f9dff}
  ul{color:#9aa3b2;padding-left:20px}
  img{max-width:100%;border:1px solid #2a2f3a;border-radius:10px;margin-top:16px}
</style></head>
<body><div class="card">
  <h1>Kitchen Vision — brain is running</h1>
  <p>The React dashboard hasn't been built yet. Build it with
     <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>
     (output → <code>web/dist</code>) and this page will be replaced by the app.</p>
  <p>The API is live in the meantime:</p>
  <ul>
    <li><a href="/video">/video</a> — annotated MJPEG feed</li>
    <li><a href="/api/people">/api/people</a> — people roster</li>
    <li><a href="/api/events">/api/events</a> — event feed</li>
    <li><a href="/api/stats">/api/stats</a> — per-person analytics</li>
    <li><code>/events</code> — live WebSocket (SSE fallback)</li>
  </ul>
  <img src="/video" alt="live feed">
</div></body></html>
"""


class _SpaStaticFiles(StaticFiles):
    """StaticFiles that falls back to ``index.html`` for any unmatched path (client-side routes).

    A React Router SPA owns paths like ``/people`` / ``/analytics`` that have no file on disk; the
    server must return ``index.html`` for those so the client router can take over. Genuine missing
    assets under known prefixes still 404 (handled by the API routes registered above this mount).
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except Exception:
            # Anything not found → serve the SPA shell so client-side routing works.
            index = os.path.join(self.directory, "index.html")
            if os.path.isfile(index):
                return FileResponse(index, media_type="text/html")
            return HTMLResponse(_PLACEHOLDER)


if os.path.isdir(WEB_DIST) and os.path.isfile(os.path.join(WEB_DIST, "index.html")):
    # Built SPA present: mount it at root with SPA fallback. Mounted LAST so all /api, /video,
    # /events routes above win; only otherwise-unmatched paths reach the SPA.
    app.mount("/", _SpaStaticFiles(directory=WEB_DIST, html=True), name="spa")
else:
    # No build yet — serve a tiny placeholder at / (do NOT crash). The API still works.
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_PLACEHOLDER)


# Re-export config for convenience / introspection (cheap, offline).
CONFIG = config_mod.load_config()
