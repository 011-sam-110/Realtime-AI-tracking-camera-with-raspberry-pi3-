# Kitchen Vision — Roadmap

Each phase is independently testable. The build is orchestrated as an **agent team against the
frozen `INTERFACES.md` contract**: a phase fans out parallel implementer agents (one per module),
then a review/verify pass, then integration. Phases gate on the prior one's done-criteria.

| Phase | Scope | Done when |
|---|---|---|
| **A — Contract & scaffold** *(done)* | `ARCHITECTURE.md`, `INTERFACES.md`, scaffold, base types/config/state, deps | Contract is coherent + self-reviewed; package imports; `--selfcheck` skeleton exists |
| **B — Core backend** | `core/*`, `store/db`, `capture/MjpegFeed`, `recognition/InsightFaceGpu`(+YuNet, fusion, quality), `tracking/Tracker`+`UdpServo`, `pipeline` | Live feed shows smooth **recognised + tracked** people at real-time fps on the **GPU**; `--selfcheck` passes |
| **C — Perception** | `vlm/*` (LocalVlm default + CloudVlm), `perception/*` (VlmCaptioner + worker), structured `Event`s | Per-person structured events land in the timeline from the **local** VLM; cloud fallback works |
| **D — API + SPA** | `api/app` (REST + WS + MJPEG proxy + stats), `web/` React SPA + analytics | Dashboard runs against the live backend: live view, people CRUD, timelines, events, charts |
| **E — Optimise & harden** | webcam image clean-up + best-frame; multi-frame fusion; model eval (antelopev2/glintr100); optional guided enrol; threshold tuning | Recognition measurably more reliable on the live webcam (fewer fragmented clusters, higher true-match) |
| **F — Verify & retire** | end-to-end on the real rig; refactor `head/agent.py` (12 MP-ready) without breaking the live service; polish; **delete old files** | Redesign fully replaces the old system; old files archived/removed |

## Phase E status (2026-06-16) — core built & verified offline
**Done (offline, 37 tests + selfcheck green on the brain venv):**
- `capture/enhance.py` — webcam clean-up chain (gray-world WB · auto-gamma · CLAHE · unsharp ·
  optional denoise). Wired into recognition (embedding-safe subset) + the dashboard display copy.
- Multi-template identity matching (`db.load_templates`, in-memory exemplars per person) — kills the
  single-centroid fragmentation that splits one person into many "Unknown #N".
- Multi-frame fusion (`fusion.fused_embedding`) — match the quality-weighted mean of a track's
  recent best shots, not one noisy frame.
- Margin / ambiguity guards (`recog_margin`, `unknown_min_margin`) — stop two people merging and stop
  near-duplicate unknowns spawning.
- venv frozen → `brain/requirements-venv.txt` (torch cu126 + onnxruntime-gpu both GPU-verified).

**Remaining for the live session (needs the camera — folds into Phase F):**
- Threshold tuning on real faces (recog_threshold / margins / blur gate) under live lighting.
- Model eval (buffalo_l vs antelopev2 / glintr100) measured on the actual webcam.
- Guided multi-shot enrolment UX (capture N pose-diverse quality-gated shots when naming a person).

## Orchestration notes
- The contract (`INTERFACES.md`) is frozen before each fan-out; implementers may only change it by
  editing it + flagging. This is what lets modules be built in parallel and still compose.
- **Reuse** the proven old code (ARCHITECTURE §12) — port the control law, MJPEG framing, quality
  gate/clustering, DB threading, VLM call/parse. Don't reinvent.
- Each phase ends with an **adversarial review** (a reviewer agent tries to break the contract
  compliance + threading rules) and a `--selfcheck` + targeted tests before integration.

## Hardware milestones (design already supports; no rework)
- **12 MP camera** → new `FeedSource` (`CsiFeed`/`UsbDirect`) + head `Camera` swap; unlocks
  `perception/LocalCvScene` (YOLO/open-vocab + tracker + zones) for higher-fidelity object/chore events.
- **Tilt servo** → second axis in `Tracker`/`ServoTransport` (angle is already a vector).
- **Dedicated 24/7 box** → the brain is portable; move it, keep everything else.
