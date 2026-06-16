# Kitchen Vision — INTERFACES (authoritative contract)

**This is the single source of truth** for how the `kitchenvision` brain package's modules talk to
each other. Every module MUST implement the public APIs below **exactly** — same import paths,
class names, method names, argument order/names, and return shapes — so independently-built pieces
compose without further coordination. If something here is ambiguous or wrong, fix it **here first**
and flag it; do not diverge silently.

- **Package root:** `C:/Users/sampo/pi/brain/kitchenvision/` — run as `python -m kitchenvision`
  from `C:/Users/sampo/pi/brain` (so `import kitchenvision...` resolves).
- Use **forward slashes** in code (fine on Windows). Type-hint everything. Python 3.13.
- The Pi head (`head/agent.py`) and its UDP/MJPEG protocol (§13) are a fixed external contract.
- Canonical working resolution **W, H = 640, 480**. All boxes are `[x, y, w, h]` ints in
  full working-frame pixel coords, clamped to the frame. (A future 12 MP `FeedSource` may downscale
  to the working size for ML while exposing a high-res frame for crops — see §5.)

---

## 0. System shape & threading

One process hosts these daemon threads + uvicorn, sharing one guarded `STATE` (`core/state.py`):

```
FeedSource ─► capture/track thread ─► STATE.latest_jpeg ─► GET /video
               │  │  └─► Tracker ─► ServoTransport: UDP angle + overlay ─► Pi
               │  └─► (publish latest raw frame) ─► recognition thread ─► FaceEngine ─► detections
               │                                      └─► store (sightings/embeddings) + STATE.people
               └─► STATE.latest_raw ─► perception thread ─► PerceptionSource ─► VisionModel
                                          └─► store.add_event + STATE.people[*].current_activity
GET /api/* , GET /events (WS/SSE) read STATE + store.
```

Threads (all share `STATE`; guard every cross-thread STATE field access with `STATE.lock`):
- **capture/track** — `pipeline._capture_loop`; owns the feed, runs the Tracker, writes
  `latest_jpeg/latest_raw/people/servo_angle/fps`. Never blocks on ML.
- **recognition** — `pipeline._recognition_loop`; the **only** caller of `FaceEngine.recognize()`
  / `.refresh()`. Takes the latest raw frame, publishes detections.
- **perception** — `PerceptionWorker`; reads `latest_raw` + `people`, calls `PerceptionSource`,
  writes events + `current_activity`.
- **prune** — periodic `store.prune()`.
- **uvicorn request threads** — serve `api.app`.

### DB threading rule (CRITICAL)
`sqlite3` connections **cannot cross threads**. Never cache/share a connection across threads. Call
the `store.db` helper functions (each opens its own short-lived connection) or `store.db.connect()`
for a fresh connection used+closed in the same thread. Writes are serialised by a module-level
`threading.Lock`; DB is WAL with a 30 s busy timeout.

---

## 1. `core/config.py`

```python
DEFAULTS: dict
CONFIG_PATH: str                       # "C:/Users/sampo/pi/brain/config.json"
load_config(path=CONFIG_PATH) -> dict  # DEFAULTS deep-merged with config.json (missing/partial ok)
```

`load_config()` returns a fresh dict (callers may mutate). Nested blocks (`activity`, `vlm`,
`recognition`, `track`) merge key-by-key. `data_dir` is forced absolute.

### Keys & defaults
| key | default | meaning |
|---|---|---|
| `pi_ip` | `"192.168.68.127"` | Pi host |
| `stream_port` | `8000` | Pi MJPEG port |
| `stream_path` | `"/raw"` | clean feed path |
| `servo_udp` | `9999` | UDP servo-angle port on the Pi |
| `overlay_udp` | `9998` | UDP overlay-box port on the Pi |
| `dashboard_port` | `8090` | FastAPI port |
| `data_dir` | `<brain>/data` (abs) | SQLite + crops + thumbnails |
| `retention_days` | `30` | rolling window for sightings + events |
| **`recognition`** | see below | face-engine block |
| **`track`** | see below | tracker block |
| **`vlm`** | see below | vision-LLM block |
| **`activity`** | see below | perception/event-worker block |
| **`enhance`** | see below | webcam image-quality block (Phase E) |
| **`detector`** | see below | fast tracking-detector block (decoupled realtime tracking) |

`recognition`: `engine`("insightface"|"yunet", def "insightface"), `model`("buffalo_l"; antelopev2
optional), `providers`(["CUDAExecutionProvider","CPUExecutionProvider"]), `det_size`(640),
`det_thresh`(0.6), `recog_threshold`(0.45), `min_det_score`(0.65), `min_face_px`(70),
`min_blur_var`(40.0), `centroid_alpha`(0.9), `fusion_window`(5) — best-shot buffer size.
*Phase E hardening:* `templates_per_person`(4) — exemplars kept per identity for matching (1 =
centroid-only), `match_fused`(true) — match the per-track fused best-shot embedding, `recog_margin`
(0.04) — min gap top-1 vs best OTHER person required to enrol (ambiguity guard), `unknown_min_margin`
(0.10) — best sim must be `< recog_threshold - this` to spawn a NEW unknown.

`enhance` (Phase E, `capture/enhance.py`): `enabled`(true), `for_recognition`(true) — auto-gamma +
CLAHE before `app.get()` (embedding-safe subset), `for_display`(true) — full chain on the dashboard
video copy, `white_balance`(true, display-only), `auto_gamma`(true) + `target_luma`(0.52),
`gamma`(1.0, used when `auto_gamma` false), `clahe`(true) + `clahe_clip`(2.0) + `clahe_grid`(8),
`unsharp`(true, display-only) + `unsharp_amount`(0.6) + `unsharp_sigma`(1.0), `denoise`(false,
display-only, costly) + `denoise_strength`(5). The recognition path applies only the embedding-safe
subset (auto-gamma + CLAHE); the display path adds white-balance + unsharp (+ optional denoise).

`track` (responsive-but-smooth, decoupled fast-box tracking): `outer_px`(55), `inner_px`(30),
`kp`(0.06), `max_step`(10.0), `direction`(-1), `ema`(0.25), `search_after`(2.5), `sweep_speed`(22.0),
`min_step`(2.0) — ignore corrective steps smaller than this (anti micro-jitter), `hold_seconds`(0.2)
— short settle after a move so the servo de-blurs (not a multi-second freeze), `axes`(["pan"]).

`detector` (fast tracking detector, `capture/detector.py`): `enabled`(true) — false ⇒ tracking falls
back to the recognition engine's boxes (slower lock-step), `engine`("yunet"), `det_size`(320) —
downscaled detect width, the box is scaled back to 640×480, `det_thresh`(0.6). Run EVERY frame in the
capture loop so the servo follows at frame rate, decoupled from the few-fps recognition pass.

`vlm`: `backend`("local"|"cloud", def "local"), `local_model`("qwen2-vl-2b" | "moondream2" |
"florence2"), `device`("cuda"), `max_tokens`(300); cloud sub-keys `base_url`(""), `api_key`(""),
`model`("auto"). Empty cloud creds ⇒ cloud backend is "disabled".

`activity` (perception worker): `enabled`(true), `cadence_seconds`(10), `save_thumbnails`(true),
`min_confidence`(0.0).

---

## 2. `core/state.py`

```python
class State:
    lock: threading.Lock
    latest_jpeg: bytes | None          # most recent ANNOTATED frame (GET /video)
    latest_raw:  "np.ndarray | None"   # most recent RAW BGR frame (for perception)
    people:      list[dict]            # see shape below
    activity_status: dict              # {"state": "ok"|"disabled"|"no_vision_model"|"error", "message": str}
    servo_angle: float                 # last pan angle sent (degrees) — back-compat scalar
    servo_angles: dict                 # {"pan": float, "tilt": float?} — N-axis
    fps:         float

STATE = State()      # module-level singleton; import as `from kitchenvision.core.state import STATE`
```

### `STATE.people` item (written by the pipeline each cycle)
```python
{ "person_id": int, "label": str, "kind": str, "box": [x,y,w,h],
  "track_id": int, "age": float|None, "sex": str|None,
  "current_activity": str, "last_seen_ts": float }
```
The pipeline owns every field except `current_activity`, which the perception worker writes
in-place (re-found by `person_id` under `STATE.lock`). `STATE.people` is replaced wholesale each
cycle; the worker tolerates a person vanishing.

`activity_status.state`: `ok` (success/idle-healthy) · `disabled` (perception/cloud off or no
creds) · `no_vision_model` (cloud 422 / no-vision → stop calling) · `error` (transient + backoff).

---

## 3. Shared types — `core/types.py`

Concrete dataclasses every module imports (`from kitchenvision.core.types import Detection, Event, Frame`):

```python
@dataclass
class Frame:
    bgr: "np.ndarray"          # working-resolution BGR (640x480)
    ts: float                  # capture time (time.time())
    servo_angles: dict         # {"pan": float, ...} known at capture (ego-motion aware)
    seq: int                   # monotonic frame counter
    hires: "np.ndarray | None" = None   # optional full-res frame for crops (12MP future); else None

@dataclass
class Detection:
    box: list[int]             # [x, y, w, h], full working-frame coords
    person_id: int             # db id, or -1 for detect-only / unenrolled
    label: str                 # name or "Unknown #N" or "face"
    kind: str                  # "known" | "unknown"
    track_id: int = -1         # stable across frames (identity persistence); -1 if none
    age: float | None = None
    sex: str | None = None     # "M" | "F" | None
    score: float = 0.0         # match cosine sim
    quality: float = 0.0       # best-shot quality score (blur/size/det fused)

@dataclass
class Event:
    type: str                  # "activity" | "object" | "chore" | "presence"
    ts: float
    person_id: int | None      # subject (None = scene-level)
    action: str | None = None  # verb, e.g. "left", "cleared", "washing"
    object: str | None = None  # noun, e.g. "plate", "mug"
    location: str | None = None# zone/surface, e.g. "table", "sink"
    text: str = ""             # human caption ("Sam left a plate on the table")
    confidence: float = 1.0
    source: str = "vlm"        # which PerceptionSource produced it ("vlm" | "cv" | ...)
    thumb_ref: str | None = None  # event thumbnail key (store.thumbnails), if saved
    payload: dict | None = None   # raw extras
```

`Detection`/`Event` are the **only** types crossing module boundaries for results — do not invent
parallel dict shapes.

---

## 4. `store/db.py`

Constants: `DATA_DIR`, `DB_PATH`, `FACES_DIR`, `THUMBS_DIR`, `MAX_CROPS = 5`.
Vectors stored as float32 bytes (`np.asarray(a, np.float32).tobytes()` ⇄ `np.frombuffer`); centroids
stored **L2-normalised**; `load_centroids()` returns them normalised.

### Schema (CREATE TABLE IF NOT EXISTS)
```
people(id PK, name TEXT NULL, kind TEXT, centroid BLOB, age REAL NULL, sex TEXT NULL,
       created_ts REAL, last_seen_ts REAL)
embeddings(id PK, person_id INT, vec BLOB, quality REAL, ts REAL)
sightings(id PK, person_id INT, ts REAL, x INT, y INT, w INT, h INT)
events(id PK, ts REAL, person_id INT NULL, type TEXT, action TEXT NULL, object TEXT NULL,
       location TEXT NULL, text TEXT, confidence REAL, source TEXT, thumb_ref TEXT NULL,
       payload TEXT NULL)
activity_log(id PK, person_id INT, ts REAL, text TEXT, provider TEXT)   # back-compat
```

### Functions (each opens its own connection; thread-safe; never share conns)
```python
connect() -> sqlite3.Connection            # WAL, row_factory=Row, busy_timeout=30s
init_db() -> None                          # dirs + schema + indexes (call once at startup)

# people
create_person(centroid, kind, age=None, sex=None) -> int
add_embedding(person_id, vec, quality=0.0) -> None
update_centroid(person_id, centroid) -> None
set_last_seen(person_id, ts) -> None
list_people() -> list[dict]                # {id,name,kind,age,sex,created_ts,last_seen_ts,label}
get_person(person_id) -> dict | None       # same shape (+label = name else "Unknown #N" by id rank)
set_name(person_id, name) -> None          # sets name AND kind="known"; blank name = no-op
merge_people(src_id, dst_id) -> None       # reassign embeddings/sightings/events; recompute dst
                                           # centroid; move crops; delete src. no-op if src==dst
delete_person(person_id) -> None           # delete person + all their rows + crops/thumbs

# sightings / events / timeline
add_sighting(person_id, ts, box) -> None                       # box=(x,y,w,h)
add_event(event) -> int                                        # Event dataclass → row; returns id
get_events(person_id=None, limit=200, since=None) -> list[dict]# newest first; optional filters
get_timeline(person_id, limit=200) -> list[dict]               # that person's events, newest first
add_activity(person_id, ts, text, provider) -> None            # back-compat freeform caption

# centroids / templates / retention
load_centroids() -> list[dict]             # [{id,name,kind,centroid(np.float32 normalised)}]
get_centroid(person_id) -> np.ndarray|None # one person's centroid, L2-normalised (Phase E)
load_templates(k=4) -> list[dict]          # [{id,name,kind,vec}] centroid+top-quality exemplars/person (Phase E)
prune(retention_days) -> tuple[int,int]    # delete old sightings+events; KEEP people+embeddings

# crops + thumbnails (best-effort; never raise into the pipeline)
crops_dir(person_id, create=True) -> str
save_crop(person_id, jpg_bytes) -> None    # newest = 0.jpg, rotate, keep MAX_CROPS
list_crops(person_id) -> list[str]         # newest first
save_thumb(jpg_bytes) -> str               # write a thumbnail, return its thumb_ref key
thumb_path(thumb_ref) -> str | None        # path for a key, or None
```

---

## 5. `capture/` — `FeedSource`

```python
# capture/base.py
class FeedSource(Protocol):
    def open(self) -> None: ...            # connect/retry until ready (blocking ok)
    def read(self) -> Frame | None: ...    # next Frame (resized to W,H), or None on a transient miss
    def close(self) -> None: ...

def make_feed(config: dict, angle_provider: "Callable[[], dict]") -> FeedSource
    # factory: returns MjpegFeed(config, angle_provider) by default (config decides later impls).
    # angle_provider() returns the current {"pan":deg,...} so each Frame is tagged with servo state.
```

```python
# capture/mjpeg.py
class MjpegFeed(FeedSource):   # wraps cv2.VideoCapture on http://{pi_ip}:{stream_port}{stream_path}
    # Port open_stream() reconnect loop from the old track_client/pipeline. read() resizes to W,H,
    # stamps ts/seq and servo_angles via angle_provider(). Reconnects after _RECONNECT_AFTER misses.
    # A background grabber thread continuously DRAINS the FFmpeg buffer (keeps only the newest decoded
    # frame), so read() returns a near-current frame, not a ~0.5 s-stale buffered one (the realtime
    # fix); read() returns None when no NEW frame has arrived since the last call.

# capture/detector.py — FAST tracking detector, decoupled from the recognition engine (§7 driver).
class FaceDetector(Protocol):
    def detect(self, frame: "np.ndarray") -> list[list[int]]: ...   # [[x,y,w,h], ...] in 640x480

def make_detector(config: dict) -> "FaceDetector | None"
    # YuNetDetector (cv2.FaceDetectorYN over brain/models/yunet.onnx, downscaled to detector.det_size
    # for speed, boxes scaled back to 640x480); None when detector.enabled is false. The pipeline
    # runs detect() EVERY frame to drive the Tracker + overlay at frame rate, so the servo follows in
    # realtime instead of waiting on the few-fps recognition pass. Recognition still independently
    # produces identity (STATE.people); None ⇒ the tracker falls back to recognition boxes.
```

## 6. `recognition/` — `FaceEngine`

```python
# recognition/base.py
class FaceEngine(Protocol):
    def recognize(self, frame: "np.ndarray") -> list[Detection]: ...
    def refresh(self) -> None: ...         # reload centroids after rename/merge/delete

def make_engine(config: dict) -> FaceEngine    # InsightFaceGpu (default) | YuNet (config.engine)
```

```python
# recognition/insightface_gpu.py
class InsightFaceGpu(FaceEngine):
    # ONE app.get(frame) pass → bbox, embedding, age, sex per face. providers from config
    # (CUDA first, CPU fallback). allowed_modules trimmed to detection+recognition+genderage.
    # Pipeline: detect → quality-gate (min_det_score/min_face_px/min_blur_var) → match cosine vs
    #   in-memory templates (>= recog_threshold ⇒ existing person) → else create unknown (good
    #   quality only). Append embedding + nudge centroid (centroid_alpha running mean) on good
    #   frames; always log sighting + last_seen on a match. Assign/maintain track_id via IoU+emb
    #   so identity persists across frames (fusion.py). Save up to MAX_CROPS best crops.
    #   Returns Detection per face. Guards its own template matrix with a lock; refresh() reloads.
    # Phase E: (1) the frame is enhanced (auto-gamma+CLAHE, capture/enhance.py) before app.get();
    #   (2) each person is matched by its BEST of several templates (centroid + top-quality
    #   embeddings, db.load_templates) — not one mean — via _match() returning (pid,score,margin);
    #   (3) the probe is the per-track fused best-shot embedding (fusion.fused_embedding) when
    #   match_fused; (4) margin guards: enrol only if margin>=recog_margin (else 'ambiguous' →
    #   attribute+sighting but no model mutation), create a new unknown only if
    #   score < recog_threshold-unknown_min_margin. _add_template adapts in-memory (FIFO cap).
```
```python
# recognition/yunet.py  — detect-only fallback (cv2.FaceDetectorYN, yunet.onnx). person_id=-1.
# recognition/fusion.py — best-shot buffer per track_id + IoU tracker + quality scoring.
#                     Phase E: fused_embedding(track_id) = quality-weighted L2-normalised mean of
#                     the window's embeddings (multi-frame fusion); set_person(track_id, pid) stamps
#                     the resolved identity onto the track's best shot.
# capture/enhance.py  — Enhancer(params).apply(frame, for_display) = webcam image-quality chain
#                     (gray-world WB · auto-gamma · CLAHE · unsharp · denoise). make_enhancer(config)
#                     → Enhancer | None. Pure cv2/numpy; recognition uses the embedding-safe subset.
# recognition/quality.py— blur_var / size / det-score gate (pure functions).
```
YuNet model path: ship `yunet.onnx` into `brain/models/yunet.onnx` (copy from the old repo root).

## 7. `tracking/` — `Tracker` + `ServoTransport`

```python
# tracking/servo_transport.py
class ServoTransport(Protocol):
    def send_angles(self, angles: dict) -> None: ...     # {"pan":deg,...}; throttled internally
    def send_overlay(self, box: tuple | None, correcting: bool) -> None: ...

class UdpServo(ServoTransport):    # one reused UDP socket. send_angles sends pan as ASCII float to
    # (pi_ip, servo_udp) [back-compat]; send_overlay sends "x,y,w,h,correcting" / "none" to overlay_udp.

# tracking/tracker.py
class Tracker:
    def __init__(self, config: dict): ...
    def update(self, boxes: list[list[int]], frame_w: int, fresh: bool = True) -> dict: ...
        # PORT the proven control law (group-centre target = midpoint(min cx,max cx); EMA; OUTER/
        # INNER hysteresis; KP*err step clamped to MAX_STEP; lock-step on `fresh`; time-based search
        # sweep after search_after). Returns {"pan": angle} (+tilt later). Exposes .correcting/.searching.
        # `boxes` are the FAST detector's (every frame, fresh=True) for realtime tracking, or the
        # recognition engine's (lock-step) as fallback. Anti-jitter knobs: `min_step` (ignore tiny
        # corrections), `hold_seconds` (short settle after a move) — both from the §1 `track` block.
```
Pipeline each cycle: `angles = tracker.update(boxes, 640, fresh)`; `servo.send_angles(angles)`;
`servo.send_overlay(union_box(boxes) if boxes else None, tracker.correcting)`; store into
`STATE.servo_angles` (+ `STATE.servo_angle = angles["pan"]`).

## 8. `perception/` — `PerceptionSource` + `PerceptionWorker`

```python
# perception/base.py
class PerceptionSource(Protocol):
    def observe(self, frame: "np.ndarray", people: list[dict]) -> list[Event]: ...
        # Given a raw BGR frame + the present STATE.people, return structured Events. Never raises
        # fatally; signals health via the worker's status. v1 impl = VlmCaptioner.

def make_source(config: dict, vision: "VisionModel") -> PerceptionSource   # VlmCaptioner (v1)

# perception/worker.py
class PerceptionWorker:                      # owns the perception daemon thread
    def __init__(self, config: dict, state, source: PerceptionSource): ...
    def start(self) -> None: ...             # every cadence_seconds: snapshot raw+people under lock,
        # skip if nobody present (status "ok"/idle), else source.observe(...) → for each Event:
        # optionally save_thumb, store.add_event, update STATE.people[*].current_activity by person_id.
        # Reflect failures only via STATE.activity_status (never crash the process).
```
```python
# perception/vlm_captioner.py
class VlmCaptioner(PerceptionSource):
    # Build the labelled prompt (port the proven prompt + JSON-extraction from old activity.py),
    # call self.vision.generate(...), parse {label: phrase} → Events (type="activity", text=phrase,
    # person_id matched by label, tolerant case/space). Parses object/action/location out of the
    # phrase where possible (light keyword map; full extraction is a later LocalCvScene job).
```

## 9. `vlm/` — `VisionModel`

```python
# vlm/base.py
@dataclass
class VlmResult: text: str; provider: str

class VisionModel(Protocol):
    def generate(self, image_bgr: "np.ndarray", prompt: str, max_tokens: int = 300) -> VlmResult: ...
    @property
    def available(self) -> bool: ...        # False ⇒ worker reports "disabled"

def make_vision(config: dict) -> VisionModel    # LocalVlm (default) | CloudVlm (config.vlm.backend)
```
```python
# vlm/local_vlm.py  — load the chosen small VLM on CUDA once; generate() runs image+prompt → text.
#                     provider = local model id. available = model loaded ok.
# vlm/cloud_vlm.py   — port old activity.py OpenAI-compatible call (FreeLLMAPI): base_url/api_key/
#                     model, image as data: URL, X-Routed-Via → provider. Maps 422→ a NoVisionError
#                     the worker turns into status "no_vision_model". available = creds present.
```
`generate()` raises typed errors the worker maps to status: `NoVisionError`→no_vision_model,
`RateLimitError`→error+backoff, anything else→error. Defined in `vlm/base.py`.

## 10. `pipeline.py`

```python
STATE: State                               # re-exported from core.state for convenience
def start(config: dict) -> None
    # 1) store.db.init_db()
    # 2) engine=make_engine(cfg); tracker=Tracker(cfg); servo=UdpServo(cfg);
    #    vision=make_vision(cfg); source=make_source(cfg, vision)
    # 3) spawn recognition thread (sole caller of engine.recognize/refresh)
    # 4) spawn capture/track thread: feed=make_feed(cfg, angle_provider); detector=make_detector(cfg);
    #    per Frame → publish raw to recognition; tracker.update runs on the FAST detector's boxes
    #    (every frame, decoupled, fresh=True) when enabled, else on the latest recognition boxes
    #    (lock-step); servo sends; annotate (realtime fast boxes + recognised names via _match_labels)
    #    → STATE. Recognition independently produces identity (STATE.people).
    # 5) start PerceptionWorker(cfg, STATE, source)
    # 6) spawn periodic prune thread
    # Non-blocking; idempotent (guards double-start).
def refresh_identities() -> None           # set an event the recognition thread consumes → engine.refresh()
def union_box(boxes) -> tuple | None       # (min x,min y,max(x+w)-min x,max(y+h)-min y) or None
```
Decoupling (capture↔recognition via latest-wins frame slot + version-gated detections) and the
annotate/draw overlay are **ported from the old `station/pipeline.py`** (proven). `angle_provider`
returns `STATE.servo_angles`.

## 11. `api/app.py` (FastAPI)

```python
app: fastapi.FastAPI         # uvicorn target = kitchenvision.api.app:app
```
Routes (read STATE + store; call `pipeline.refresh_identities()` after people mutations):
- `GET /video` — annotated MJPEG (`multipart/x-mixed-replace; boundary=FRAME`), yields `STATE.latest_jpeg`.
- `GET /events` (WS preferred; SSE fallback) — ~1–4 Hz `{people, activity_status, servo_angles, fps}`.
- `GET /api/people` → `{people:[... +crop_urls]}` · `POST /api/people/{id}/name {name}` ·
  `POST /api/people/{id}/merge {into}` · `DELETE /api/people/{id}`
- `GET /api/people/{id}/timeline` → `{timeline: store.get_timeline(id)}`
- `GET /api/events?since=&limit=&person_id=` → `{events:[...]}`
- `GET /api/stats` → per-person aggregates (presence time, event/chore counts) for analytics
- `GET /api/faces/{id}/{n}.jpg` · `GET /api/thumbs/{ref}.jpg`
- `GET /` and static — serve the built SPA from `web/dist` (mount; SPA fallback to index.html).
CORS open to localhost in dev; same-origin in prod (SPA served by this app).

## 12. `web/` — React SPA (consumes §11 only)

Vite + React + TS + Tailwind + shadcn/ui + Recharts. Dev proxies `/api`,`/video`,`/events` →
`http://localhost:8090`. Build → `web/dist`, served by FastAPI. Views: **Live** (`<img src=/video>`
+ live people/status over WS), **People** (gallery, rename/merge/delete, crops), **Person** (timeline
+ events + thumbnails), **Events** (filterable feed), **Analytics** (Recharts: presence + chore/mess
per person over time), **Settings** (thresholds, vlm backend, retention). Dark theme.

## 13. Head protocol — `head/agent.py` (do NOT break)

Refactor of `pi_agent.py`; the wire contract is **fixed**:
- MJPEG `http://{pi}:8000/raw` (clean) + `/stream` (Pi-annotated), 640×480, `--FRAME` multipart.
- UDP `:9999` ← ASCII float pan angle (Pi clamps [10,170], 100 Hz slew).
- UDP `:9998` ← `"x,y,w,h,correcting"` (5 ints) or `"none"`; one box, 0.4 s TTL.
Keep HW-PWM (GPIO18/pwmchip0) + slew + camera reconnect verbatim. New work: factor capture behind a
`Camera` so a 12 MP CSI module (picamera2/libcamera) is a drop-in; optional tilt on a 2nd channel.

---

## 14. Build rules
- Implement **only** against this contract; if you must change a signature, edit this file + note it.
- **Port, don't reinvent** the ported pieces (ARCHITECTURE §12): control law, MJPEG framing, quality
  gate/clustering, DB threading, VLM call/parse, decoupled-thread pattern.
- Every module: docstring citing its INTERFACES section; type hints; no cross-thread STATE access
  without `STATE.lock`; no sqlite connection sharing across threads.
- Tests live in `brain/tests/`; a `python -m kitchenvision --selfcheck` must construct everything
  without opening the camera or binding the port (offline wiring check).
