<p align="center">
  <img src="docs/images/dashboard-live.png" width="640" alt="Kitchen Vision live dashboard">
</p>

<h1 align="center">Kitchen Vision</h1>
<p align="center">A real-time pan-tracking camera that follows people, recognises who they are, and logs what they do.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%2F%20onnxruntime-76B900?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Recognition-InsightFace-FF6F00">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/UI-React%20%2B%20TypeScript-61DAFB?logo=react&logoColor=black">
  <img src="https://img.shields.io/badge/Edge-Raspberry%20Pi%203-A22846?logo=raspberrypi&logoColor=white">
</p>

A Raspberry Pi pans a camera to keep people framed while a GPU "brain" on a laptop runs face
recognition, multi-person tracking, and a vision-LLM activity log, all streamed to a local
dashboard. It is split across two machines on an edge/brain architecture so each does only what
it is best at, and it is local-first: biometric data never leaves the machine, the cloud model
is opt-in, and "delete a person" erases everything about them.

## ✨ Features
- **Real-time tracking** — detects people every frame and pans a servo to keep the group framed; tracking runs on a fast loop decoupled from the slow ML, so the video never stutters (~24–30 fps).
- **GPU face recognition** — identifies household members with InsightFace (`buffalo_l`) on CUDA, auto-clusters strangers, and lets you name / merge / delete identities; multi-template matching + best-shot fusion stop one person fragmenting into many "Unknown #N".
- **Vision-LLM activity log** — captions per-person activity and chore/object events ("standing talking", "looking at phone", "left a plate on the table") onto a searchable timeline + analytics, using a local VLM with an optional cloud fallback.
- **Edge/brain split** — the Pi does only capture + servo (hardware-PWM, jitter-free); the laptop does all ML, storage, API and UI, so tracking survives a flaky network.
- **Local-first & private** — biometric data stays on the machine, the cloud model is opt-in, no continuous video is recorded, and deleting a person erases their data.

## 📸 Screenshots
| Live tracking + recognition | People (auto-clustered identities) |
|---|---|
| ![](docs/images/recognition-tracking.jpg) | ![](docs/images/dashboard-people.png) |
| *Face recognised as "Sam", box locked, servo re-centring at ~29 fps.* | *Identities the brain clustered itself — rename, merge, or delete.* |

| Activity timeline | Analytics |
|---|---|
| ![](docs/images/dashboard-events.png) | ![](docs/images/dashboard-analytics.png) |
| *Per-person events logged by the vision-LLM.* | *Presence time + event/chore breakdowns (Recharts).* |

## 🛠 Stack
| Layer | Tech |
|---|---|
| **Head (edge)** | Python 3.13, OpenCV / V4L2, sysfs hardware-PWM servo, stdlib HTTP + UDP |
| **Brain (ML)** | onnxruntime-CUDA, InsightFace, OpenCV, NumPy |
| **Perception** | local vision-LLM (Qwen2-VL) via transformers; OpenAI-compatible cloud fallback |
| **API / store** | FastAPI · uvicorn · WebSocket/SSE · SQLite (WAL) |
| **Web** | Vite · React · TypeScript · Tailwind · Recharts (built static, served by the API) |

## 🚀 Run
You need two machines on the same Wi-Fi LAN: a **Raspberry Pi** with a camera + pan servo (the
*head*), and a **laptop/PC with an NVIDIA GPU** (the *brain*).

**1. Head — on the Raspberry Pi** (serves MJPEG on `:8000`, pans the servo from UDP):
```bash
# one-time: enable hardware PWM on GPIO18, then reboot
echo 'dtoverlay=pwm,pin=18,func=2' | sudo tee -a /boot/firmware/config.txt
sudo cp head/kitchen-vision.service /etc/systemd/system/
sudo systemctl enable --now kitchen-vision
```

**2. Brain — on the laptop (NVIDIA GPU):**
```bash
cd brain
cp config.example.json config.json     # set "pi_ip" to your Pi's LAN IP
python -m venv .venv && .venv/Scripts/activate     # Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m kitchenvision --selfcheck     # offline wiring check
python -m kitchenvision                 # open the dashboard at http://localhost:8090
```

First run downloads InsightFace `buffalo_l` (~300 MB) once. Set `servo_enabled: false` to run the
full pipeline without moving the camera. Docker alternative: `cd brain && docker compose up -d --build`
(see [brain/DOCKER.md](brain/DOCKER.md)).

## 🧠 How it works
Two machines, one decoupled wire protocol:

```
   Raspberry Pi 3 — "HEAD"                         Windows laptop — "BRAIN"  (RTX 4050)
   ┌────────────────────────────┐                  ┌────────────────────────────────────────────┐
   │ USB webcam capture          │   MJPEG  :8000   │ ingest      pull feed, tag w/ servo angle    │
   │ pan servo (HW-PWM + slew)   │ ───────────────► │ recognition InsightFace on CUDA + fusion     │
   │ MJPEG server  /raw  /stream │                  │ tracking    group-frame → servo angle        │
   │ UDP in :9999 angle :9998 ov │ ◄─────────────── │ perception  vision-LLM → structured events   │
   └────────────────────────────┘   UDP angle       │ store       SQLite (WAL) + crops + thumbs     │
                                     + overlay        │ api/web     FastAPI + React SPA dashboard     │
                                                      └────────────────────────────────────────────┘
```

The key engineering problem was that the servo loop stalled ~2 s every time GPU recognition ran.
Splitting capture/track (a fast YuNet loop driving the servo every frame) from recognition
(GPU-paced, latest-wins) plus a zero-staleness frame grabber took tracking from a 2 s freeze to a
steady ~24–30 fps. Capture, face engine, perception source, vision-LLM and servo transport are
each an interface with a default impl, so a 12 MP CSI camera, a tilt axis, or a local-CV object
tracker drop in without touching the rest. Full design in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the cross-module contract is
[docs/INTERFACES.md](docs/INTERFACES.md).

## 🗺 Roadmap
Built phase-by-phase against a frozen interface contract, with a 96-test pytest suite and an
offline `--selfcheck` wiring gate.

- **A–E ✅** scaffold & contract → core backend → perception + local VLM → API + React SPA → webcam optimisation & recognition hardening.
- [ ] **F** end-to-end verification on the live rig + 12 MP-camera readiness
- Designed to scale without a rewrite: a 12 MP camera (new `FeedSource`), a tilt servo (second axis), and a local-CV object tracker are drop-in upgrades.
