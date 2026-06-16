"""Entrypoint — `python -m kitchenvision` (INTERFACES.md §0, §10).

Loads config, starts the pipeline (capture + recognition + tracking + perception + prune threads),
then runs uvicorn serving the API + built SPA. `--selfcheck` performs an offline wiring check:
it constructs the lightweight pieces and import-checks the heavy ones WITHOUT opening the camera,
loading big models, or binding the port — a fast "does it all compose?" gate.
"""
from __future__ import annotations

import importlib
import sys

from kitchenvision.core import config as config_mod


def selfcheck() -> int:
    cfg = config_mod.load_config()

    from kitchenvision.store import db
    db.init_db()

    # Lightweight constructions (no model load, no socket bind beyond a UDP handle).
    from kitchenvision.tracking.tracker import Tracker
    from kitchenvision.tracking.base import make_servo
    Tracker(cfg)
    make_servo(cfg)

    # Import-check the heavy modules so a typo/contract break is caught, without loading models.
    for mod in (
        "kitchenvision.capture.mjpeg",
        "kitchenvision.capture.detector",
        "kitchenvision.recognition.insightface_gpu",
        "kitchenvision.recognition.yunet",
        "kitchenvision.vlm.local_vlm",
        "kitchenvision.vlm.cloud_vlm",
        "kitchenvision.perception.vlm_captioner",
        "kitchenvision.perception.worker",
        "kitchenvision.pipeline",
        "kitchenvision.api.app",
    ):
        importlib.import_module(mod)

    print("selfcheck ok")
    return 0


def main() -> int:
    if "--selfcheck" in sys.argv:
        return selfcheck()

    cfg = config_mod.load_config()
    from kitchenvision import pipeline
    import uvicorn

    pipeline.start(cfg)
    uvicorn.run(
        "kitchenvision.api.app:app",
        host="0.0.0.0",
        port=int(cfg["dashboard_port"]),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
