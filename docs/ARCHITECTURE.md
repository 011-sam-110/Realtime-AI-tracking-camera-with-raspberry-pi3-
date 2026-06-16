# Kitchen Vision — Architecture

**Status:** Design v1 (redesign) · **Owner:** Sam · **Supersedes:** `PRD.md` + the `station/` package

A pan camera that **follows people**, **recognises** who they are, and **understands what they
do** (activity + object/chore events), with a polished local dashboard. This document is the
design; **`INTERFACES.md` is the authoritative cross-module contract** the build implements.

---

## 1. Goals (co-equal)

1. **Smooth tracking** — keep the group framed; servo motion is fluid; video + UI never stutter.
2. **Recognition** — reliably identify household members; auto-cluster + name unknowns; *improve
   as far as the hardware allows*.
3. **Activity & object intelligence** — log, per person, what they do and the chore/mess events
   they cause (e.g. "Sam left a plate on the table", "Alex cleared it") on a timeline + analytics.

"As far as the hardware allows" is doing work: a cheap webcam now → a 12 MP camera later, both on
a pan servo. The design squeezes the cheap sensor (best-frame selection, multi-frame fusion, image
clean-up) and is **built to get strictly better when the 12 MP arrives**, without a rewrite.

## 2. Principles

- **Edge/brain split.** The **Pi (head)** does only what must be physically next to the camera:
  capture + servo. The **laptop (brain)** does all ML, storage, API, UI. The control loop stays
  local-ish so tracking survives a flaky network; the brain can be offline and the head still pans.
- **Everything heavy is a swappable source behind an interface.** Capture, the face engine, the
  perception/event source, the vision-LLM, and the servo transport are each an interface with a
  default impl and room for more (12 MP capture, local-CV scene-diff, a different VLM, WebRTC).
  *This is the "expandable" requirement made concrete — see the seams in §5.*
- **Local-first / privacy.** Biometric data stays on the machine by default. The cloud VLM is
  opt-in. "Delete a person" erases everything about them. No continuous video is recorded.
- **On-demand.** Start fast, stop clean, persist between sessions. No always-on service machinery
  now, but nothing precludes moving the brain to a 24/7 box later.
- **Reuse what's proven.** The old servo control law, MJPEG framing, quality-gated clustering, and
  the robust VLM call/parse code work well — they are **ported**, not reinvented (see §12).

## 3. Topology

```
   Raspberry Pi 3 — "HEAD"                     Windows laptop — "BRAIN"  (RTX 4050, 6 GB)
   ┌───────────────────────────┐               ┌──────────────────────────────────────────────┐
   │ capture  USB webcam→12MP   │   WiFi LAN    │ ingest      pull feed, tag frames w/ servo ang │
   │ servo    pan, HW-PWM+slew  │ ───────────►  │ recognition InsightFace (CUDA) + fusion        │
   │ MJPEG    /raw  /stream     │   MJPEG       │ tracking    group-frame → servo angle (N-axis) │
   │ UDP in   :9999 ang :9998 ov│ ◄───────────  │ perception  Event engine (VLM src │ local-CV)  │
   └───────────────────────────┘   UDP angle    │ vlm         LocalVlm (default) │ CloudVlm       │
                                   + overlay     │ store       SQLite + thumbnails + retention     │
                                                 │ api         FastAPI: REST + WS + MJPEG proxy    │
                                                 │ web         React SPA (served static by api)    │
                                                 └──────────────────────────────────────────────┘
```

**Wire protocol (unchanged from today, so the head & brain decouple cleanly):**
- Brain ← Head: MJPEG `http://<pi>:8000/raw` (clean) and `/stream` (Pi-annotated), 640×480.
- Brain → Head: UDP `:9999` target servo angle (ASCII float); UDP `:9998` overlay box
  `"x,y,w,h,correcting"` / `"none"`.

These are abstracted behind `FeedSource` and `ServoTransport` so a future 12 MP/CSI camera or a
WebRTC transport is a new impl, not a protocol change for the rest of the brain.

## 4. Runtime model

One **brain process** (`python -m kitchenvision`) hosts several daemon threads sharing one guarded
`STATE` singleton, plus the FastAPI/uvicorn server. On-demand: it comes up, connects to the head,
and runs until stopped; SQLite + face crops + thumbnails persist between sessions.

GPU: InsightFace runs on the **CUDA** execution provider (onnxruntime already exposes it). The
local VLM also uses the GPU. v1 does **not** load YOLO, so 6 GB comfortably holds face + VLM.

## 5. Module map (brain) and the seams

```
kitchenvision/
  core/        config, STATE singleton, logging, lifecycle (start/stop)
  capture/     FeedSource  ── MjpegFeed (now) │ CsiFeed/UsbDirect/Rtsp (later, 12MP)
  recognition/ FaceEngine  ── InsightFaceGpu (default) │ YuNet (detect-only fallback)
               + best-shot fusion, identity persistence (track↔person), quality gate
  tracking/    Tracker (ported control law) + ServoTransport ── UdpServo (now) │ … (later)
               angle is an N-axis vector: pan now, tilt-ready
  perception/  PerceptionSource → Event  ── VlmCaptioner (v1) │ LocalCvScene (later)
               event structuring (person/action/object/location) + dedup
  vlm/         VisionModel ── LocalVlm (default) │ CloudVlm (FreeLLMAPI fallback/deep-pass)
  store/       SQLite (WAL, per-thread conns): people, embeddings, sightings, events,
               activity, thumbnails; retention prune
  api/         FastAPI: people CRUD, timelines, events, stats, live WS/SSE, MJPEG proxy,
               serves the SPA build
  pipeline.py  wires capture→recognition→tracking→perception→STATE; owns the threads
  __main__.py  entrypoint: load config → start pipeline → run uvicorn
```

**The seams that make it expandable (each is an interface in `INTERFACES.md`):**

| Seam | v1 default | Drop-in later (no downstream change) |
|---|---|---|
| `FeedSource` | `MjpegFeed` (Pi `/raw`) | `CsiFeed`/`Rtsp` for the 12 MP camera |
| `FaceEngine` | `InsightFaceGpu` | bigger model / different backbone |
| `PerceptionSource` | `VlmCaptioner` | `LocalCvScene` (YOLO/open-vocab + tracker + zones + scene-diff) — or run **both** and merge |
| `VisionModel` | `LocalVlm` | `CloudVlm`, or a better local model |
| `ServoTransport` | `UdpServo` | WebRTC/serial/other |
| servo axes | pan | + tilt (the angle is already a vector) |

A `PerceptionSource` emits the **same `Event` shape** however it's powered, so upgrading the object
intelligence from "VLM describes the scene" to "local CV tracks every plate" is additive: a second
source feeding the same store and timeline.

## 6. Data flow

**Fast path (every frame, ~real-time):** `FeedSource` yields a frame (+ current servo angle) →
publish to the recognition worker (latest-wins) → read back the latest detections → `Tracker`
computes the target angle → `ServoTransport` sends angle + overlay → annotate a display copy →
publish `STATE.latest_jpeg` for `/video`. **Never blocks on ML.**

**Recognition path (own thread, GPU pace):** take the most-recent frame → `FaceEngine.recognize()`
(detect + embed + match/cluster, quality-gated, best-shot fused) → publish detections + persist
sightings/embeddings.

**Perception path (own thread, cadence/holds):** snapshot the raw frame + present people →
`PerceptionSource.observe()` → `VisionModel` (local) → parse into `Event`s → persist + update each
person's `current_activity` + timeline.

## 7. Threading

Mirrors the proven decoupling in today's `station`: **capture/track** (fast), **recognition**
(slow, sole caller of the face engine), **perception** (cadence), **prune** (periodic), **uvicorn**
request threads. Every cross-thread field on `STATE` is guarded by `STATE.lock`. SQLite
connections never cross threads (each helper opens its own; WAL + busy timeout). Full rules in
`INTERFACES.md §0`.

## 8. Data model (sketch — authoritative schema in `INTERFACES.md`)

- **person** — `id, name?, kind(known|unknown), centroid, age?, sex?, created_ts, last_seen_ts`
- **embedding** — `person_id, vec(512 f32), quality, ts`
- **sighting** — `person_id, ts, box`
- **event** — `id, ts, person_id?, type, action?, object?, location?, confidence, source,
  thumb_ref?, payload` — the unified activity/object/chore record (VLM now, local-CV later)
- **activity_log** — kept for back-compat / freeform captions (subsumed by `event`)
- **thumbnail** — small JPEG keyframe per event (retention-bound), for the dashboard
- **config/state** — thresholds, servo calibration, zones (later)

## 9. Hardware roadmap

| Axis | Now | Later (design already supports) |
|---|---|---|
| Camera | cheap USB webcam, 640×480 | single **12 MP** (CSI module or USB) via a new `FeedSource` |
| Servo | 1× pan (SG90, HW-PWM) | + tilt (angle vector already N-axis) |
| Compute | laptop RTX 4050 (6 GB) | dedicated 24/7 box (brain is portable) |
| Object intel | VLM captions → events | add `LocalCvScene` (YOLO/open-vocab + tracker + zones) |

## 10. Tech stack

| Layer | Choice |
|---|---|
| Head | Python 3.13, OpenCV/V4L2 (→ picamera2/libcamera for CSI), sysfs HW-PWM servo, stdlib HTTP/UDP |
| Brain | Python 3.13, onnxruntime-**CUDA**, InsightFace, OpenCV, NumPy, FastAPI, uvicorn |
| VLM | local (Qwen2-VL-2B / moondream2 / Florence-2 — chosen by benchmark) via transformers/llama.cpp; cloud = OpenAI-compatible FreeLLMAPI |
| Store | SQLite (WAL) |
| Web | Vite + React + TypeScript + Tailwind + shadcn/ui + Recharts; built static, served by FastAPI (single origin, no runtime Node) |

## 11. Phasing (see `ROADMAP.md`)

A scaffold+contract → B core backend (smooth recognised+tracked feed) → C perception + local VLM
→ D API + React SPA → E webcam optimisation + recognition hardening → F verify on rig + retire old.

## 12. Reuse map (port, don't reinvent)

| From old | Reused as |
|---|---|
| `pi_agent.py` | `head/agent.py` — refactored for `FeedSource`-readiness + 12 MP; HW-PWM + slew + MJPEG framing kept verbatim |
| `track_client.py` / `station/tracker.py` | `tracking/tracker.py` — control law (hysteresis, KP/MAX_STEP glide, lock-step, search sweep) ported |
| `station/recognizer.py` | `recognition/` — quality gate + cluster/centroid logic ported, moved to GPU + best-shot fusion |
| `station/db.py` | `store/db.py` — threading rules + schema extended with `event`/`thumbnail` |
| `station/activity.py` | `vlm/` + `perception/` — robust call/parse/JSON-extraction split into VisionModel + event structuring |
| `station/pipeline.py` | `pipeline.py` — decoupled-thread design + STATE pattern carried over |
