"""The cell board's side of the depth service (``service.py``).

One thread per camera sends the camera's newest frame, waits for the completed depth and
sends the next: each camera completes as fast as the service answers, and a slow or
absent service only leaves the raw depth in place. What the board keeps is ``fuse``d: the
sensor's depth wherever it measured, the model's only in nearby holes, both from the
same frame.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any, Callable

import numpy as np

from .service import decode, encode

TIMEOUT_S = 10.0
RETRY_S = 2.0
# Completed depth fills only holes this close to a pixel the sensor measured: further out
# (beyond the sensor's range, a wall nobody measured) the model would be inventing it.
FILL_RADIUS_PX = 60


def fuse(raw: np.ndarray, completed: np.ndarray, radius_px: int = FILL_RADIUS_PX) -> np.ndarray:
    """The sensor's depth wherever it measured; the model's only in holes near a measurement."""

    import cv2

    holes = raw <= 0
    if not holes.any() or not (~holes).any():
        return raw if not holes.any() else np.zeros_like(raw)
    distance = cv2.distanceTransform(holes.astype(np.uint8), cv2.DIST_L2, 3)
    return np.where(~holes, raw, np.where(distance <= radius_px, completed, 0.0)).astype(np.float32)


def service_info(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/info", timeout=3.0) as reply:
        return json.loads(reply.read())


def complete(url: str, rgb: np.ndarray, depth: np.ndarray, intrinsic: Any) -> tuple[np.ndarray, float]:
    request = urllib.request.Request(url.rstrip("/") + "/v1/complete", method="POST",
                                     data=encode(rgb=rgb, depth=depth.astype(np.float32),
                                                 intrinsic=np.asarray(intrinsic, float)),
                                     headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            result = decode(reply.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(json.loads(exc.read() or b"{}").get("error", str(exc))) from None
    return result["depth"], float(result["ms"])


class Completer(threading.Thread):
    def __init__(self, url: str, latest: Callable[[], tuple[int, dict[str, Any] | None]], stop: threading.Event) -> None:
        super().__init__(daemon=True)
        self.url, self.latest, self.stop = url, latest, stop
        self.result: tuple[int, np.ndarray, float, float] | None = None     # sequence, depth, ms, at
        self.error: str | None = None
        self.lock = threading.Lock()

    def run(self) -> None:
        done = -1
        while not self.stop.is_set():
            sequence, frame = self.latest()
            if frame is None or sequence == done:
                time.sleep(0.01)
                continue
            try:
                depth, ms = complete(self.url, frame["rgb"], frame["depth"], frame["intrinsic_cv"])
                depth = fuse(frame["depth"], depth)
                with self.lock:
                    self.result, self.error = (sequence, depth, ms, time.time()), None
                done = sequence
            except Exception as exc:  # noqa: BLE001 - shown on the board; raw depth stays in use
                self.error = f"{type(exc).__name__}: {exc}"
                self.stop.wait(RETRY_S)

    def completed(self, max_age_s: float = 2.0) -> tuple[int, np.ndarray] | None:
        with self.lock:
            if self.result is None or time.time() - self.result[3] > max_age_s:
                return None
            return self.result[0], self.result[1]

    def status(self) -> dict[str, Any]:
        with self.lock:
            result = self.result
        return {"ms": None if result is None else result[2], "age_s": None if result is None else time.time() - result[3],
                "error": self.error}
