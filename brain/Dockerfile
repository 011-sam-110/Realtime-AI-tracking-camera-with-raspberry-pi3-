# syntax=docker/dockerfile:1
#
# Kitchen Vision — "brain" (laptop side): recognition (InsightFace/CUDA) + realtime tracking +
# FastAPI dashboard. Runs on the laptop GPU and talks to the Pi "head" over the LAN
# (MJPEG pull :8000, UDP servo :9999 / overlay :9998).
#
# CUDA strategy: we DON'T use an nvidia/cuda base image. The CUDA 12.6 / cuDNN 9 libraries come from
# the `nvidia-*-cu12` pip wheels already pinned in requirements-venv.txt (+ torch's +cu126 wheels);
# the host only supplies the NVIDIA driver, injected by the nvidia container runtime (--gpus all).
FROM python:3.13-slim

# OS libs: OpenCV needs libGL + glib at import time; build-essential covers any source-built wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (the big layer: torch+cu126, onnxruntime-gpu, insightface, transformers, ...) ---
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements-docker.txt \
         --extra-index-url https://download.pytorch.org/whl/cu126

# onnxruntime-gpu loads cuDNN 9 / cuBLAS BY NAME at inference. On Windows the app's
# _preload_cuda_dlls() adds the wheel bin dirs; on Linux we register the wheel `lib` dirs (+ torch's)
# with ldconfig so the CUDA EP resolves them (else it silently falls back to CPU). ort.preload_dlls()
# in the app is the secondary safety net.
RUN set -eux; \
    SP="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    : > /etc/ld.so.conf.d/nvidia-pip.conf; \
    for d in "$SP"/nvidia/*/lib; do if [ -d "$d" ]; then echo "$d" >> /etc/ld.so.conf.d/nvidia-pip.conf; fi; done; \
    python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))" >> /etc/ld.so.conf.d/nvidia-pip.conf; \
    ldconfig

# --- App code. data/ + the model caches (~/.insightface, HF) are volumes; config.json is bind- ----
# mounted by compose so host edits + `docker compose restart` retune without a rebuild.
COPY kitchenvision ./kitchenvision
COPY models ./models
COPY web/dist ./web/dist
COPY config.json ./config.json

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8090
# --selfcheck is an offline import/wiring gate (no GPU, camera or port needed) — handy as a smoke test:
#   docker run --rm kitchen-vision-brain python -m kitchenvision --selfcheck
CMD ["python", "-m", "kitchenvision"]
