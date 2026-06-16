# Running the brain in Docker

Containerises the **laptop "brain"** (recognition + realtime tracking + dashboard). It runs on the
laptop GPU and talks to the Pi "head" over the LAN — the Pi itself is **not** containerised (it keeps
running `pi_agent.py`).

## Prerequisites (already satisfied on this machine)
- Docker Desktop (Docker 29 + Compose v2) with the **NVIDIA container runtime** registered.
- An NVIDIA GPU + driver (RTX 4050, driver 591.x). Check: `docker info | grep -i nvidia`.
- The Pi reachable on the LAN (`ping 192.168.68.127`). **Turn off any VPN** — it blackholes the Pi's
  `192.168.68.x` subnet.

## Build + run
```bash
cd C:\Users\sampo\pi\brain
docker compose up -d --build      # first build ~10 GB / ~15-20 min (torch+cuda+insightface)
docker compose logs -f            # watch startup (InsightFace loads ~25-30 s)
```
Open **http://localhost:8090**. Stop with `docker compose down` (add `-v` to also wipe the cached
model volumes).

First run downloads the models into named volumes (one-time): InsightFace `buffalo_l` (~300 MB) and,
because `activity.enabled` is on, the Qwen2-VL caption model (~a few GB) into the HF cache. Both
persist across restarts.

## Smoke test (offline, no GPU/Pi/port needed)
```bash
docker compose build
docker run --rm kitchen-vision-brain python -m kitchenvision --selfcheck   # prints "selfcheck ok"
```

## Retuning
`config.json` is **bind-mounted read-only**, so edit it on the host and:
```bash
docker compose restart brain
```
No rebuild needed for config/tuning changes. Rebuild (`up -d --build`) only after changing Python
code under `kitchenvision/`.

## What's mounted / persisted
| Mount | Purpose |
|---|---|
| `./config.json` → `/app/config.json` (ro) | live tuning surface (track block, detector, vlm) |
| `./data` → `/app/data` | SQLite DB + face crops + thumbnails (survives `down`) |
| `insightface-cache` volume | `buffalo_l` recognition models |
| `hf-cache` volume | local VLM weights (qwen2-vl) |

## Notes / gotchas
- **GPU:** confirm CUDA is actually used (not CPU fallback) with
  `docker compose logs | grep -i "Applied providers"` → should list `CUDAExecutionProvider`. The
  Dockerfile registers the pip CUDA wheel libs with `ldconfig` so onnxruntime finds cuDNN 9.
- **Pi networking:** Docker Desktop NATs container traffic to the LAN, so the MJPEG pull and UDP
  servo/overlay sends reach `192.168.68.127` on the default bridge. If the container can't reach the
  Pi, it's almost always a host VPN (see prerequisites).
- **Reuse host-downloaded models** instead of re-downloading: replace the named volumes with binds,
  e.g. `- C:/Users/sampo/.insightface:/root/.insightface`.
- The brain expects the Pi feed at `pi_ip` in `config.json` (`192.168.68.127`). It does not move the
  servo unless `servo_enabled: true`.
