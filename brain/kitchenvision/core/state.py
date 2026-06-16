"""The shared, mutable, cross-thread STATE singleton (INTERFACES.md §2).

Guard EVERY cross-thread access to the fields below with `STATE.lock`. The capture/track thread
writes most fields; the perception worker writes only `people[*].current_activity` and
`activity_status`; the FastAPI request threads read. `STATE.people` is replaced wholesale each
capture cycle — readers/writers re-find a person by `person_id`.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None        # most recent ANNOTATED frame (GET /video)
        self.latest_raw: Optional[np.ndarray] = None    # most recent RAW BGR frame (perception)
        self.people: list[dict] = []                    # see INTERFACES §2 item shape
        self.activity_status: dict = {"state": "ok", "message": ""}
        self.servo_angle: float = 90.0                  # last pan angle (back-compat scalar)
        self.servo_angles: dict = {"pan": 90.0}         # N-axis: pan now, tilt-ready
        self.fps: float = 0.0


# The one true instance every module imports: `from kitchenvision.core.state import STATE`.
STATE = State()
