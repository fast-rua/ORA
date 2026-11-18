"""The YAM station's original camera pixels for the world view (``native/execution-capture``), Control-only.

The same recorder as every simulator's (``ora/environment/execution_capture.py``):
its frames are what ``ora web``'s camera panel and the ``/live`` world view show,
never what the policy reads. Frames name the round, the action and the control
tick since the round's reset (``yam_control_tick_since_round_reset``, at the
station's control period), so both rigs keep one clock.

- ``sim`` renders the station model's four cameras at every boundary (reset, the
  start and end of an action, the end of a round, close) and during a motion at
  the configured rate, as a simulator does.
- ``read`` and ``on`` read nothing extra from the real cameras while the arms can
  move: a frame is the pixels an observation already read (``observation``).
  A boundary that needs a frame (a round's reset, an action's end) is written
  with the next observation's pixels, which the Supervisor takes right after
  it. Only the end of a round reads the cameras once more, between motions.

A frame that cannot be taken is recorded in ``execution-capture-failure.json``
and never stops the episode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ora.environment.controllers import yam_station as Y
from ora.environment.execution_capture import MAX_FRAME_BYTES, ExecutionCapture, resolve_execution_capture

from .common import write_json

#: The station's public cameras, in the policy's order (benchmarks/yam.py PUBLIC_CAMERAS).
CAMERAS = ("overhead", "front", "wrist_left", "wrist_right")
STEP_UNIT = "yam_control_tick_since_round_reset"
FAILURE_SCHEMA = "ora.auxiliary-media-failure.v1"

Images = Mapping[str, Any]


def rgb_images(cameras: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The colour image of each public camera of ``rig.cameras()``, as uint8 RGB."""

    return {name: np.asarray(cameras[name]["rgb"]).astype(np.uint8) for name in CAMERAS}


class YamCapture:
    """One recorder for a YAM session; ``renders`` is True for the simulated station."""

    def __init__(self, output_root: Path, *, task: str, rig_mode: str, config: Mapping[str, Any] | None,
                 read_cameras: Callable[[], Images], clock: Callable[[], int]) -> None:
        self.output_root = Path(output_root)
        self.recorder = ExecutionCapture(
            self.output_root, backend="yam", cameras=list(CAMERAS), config=config,
            identity={"run_id": self.output_root.parent.name, "task": task, "rig_mode": rig_mode},
        )
        self.renders = rig_mode == "sim"
        self.read_cameras = read_cameras
        self.clock = clock
        self.round_index = 0
        self.round_start = 0
        #: The round's end was recorded (``terminal``): a close after the verdict adds none.
        self.ended = False
        self.action: dict[str, Any] = {}
        #: A boundary the next observation's pixels complete (real rigs): (phase, action fields).
        self.deferred: tuple[str, dict[str, Any]] | None = None
        self.error: str | None = None

    @classmethod
    def enabled(cls, config: Mapping[str, Any] | None) -> bool:
        return bool(resolve_execution_capture(config)["enabled"])

    # ----------------------------------------------------------- boundaries
    def reset(self, round_index: int) -> None:
        self.round_index, self.round_start, self.action, self.ended = round_index, self.clock(), {}, False
        self._boundary("reset", {})

    def action_start(self, action: Mapping[str, Any], sequence: int) -> None:
        self.action = {"action_id": action.get("action_id"), "action": action.get("action"),
                       "action_sequence": sequence}
        if self.renders:
            self._fresh("action_start", self.action)
        elif self.deferred is not None and self.deferred[0] == "reset":
            self.deferred = None     # no observation came between the reset and this action

    def control_tick(self) -> None:
        """During a motion: a frame when one is due (the simulated station only)."""

        if self.renders and self.action:
            self._fresh("control", self.action, force=False)

    def action_end(self) -> None:
        action, self.action = self.action, {}
        if action:
            self._boundary("action_end", action)

    def action_failed(self) -> None:
        """The action raised before a receipt: no end is awaited."""

        self.action = {}
        self.deferred = None

    def observation(self, images: Images) -> None:
        """An observation's pixels: a deferred boundary's frame, else (real rigs) the world as it is now."""

        if self.deferred is not None:
            phase, action = self.deferred
            self.deferred = None
            self._write(phase, action, images, force=True)
        elif not self.renders:
            self._write("observation", {}, images, force=True)

    def terminal(self) -> None:
        """The end of a round (``restart_round``, the verdict, or a close without one: an operator's
        quit persists no verdict): one more read of the cameras."""

        self.deferred = None
        self.ended = True
        self._fresh("terminal", {})

    def close(self) -> dict[str, Any] | None:
        if not self.ended:
            self.terminal()
        if self.renders:
            self._fresh("close", {})
        self.deferred = None
        try:
            return self.recorder.close()
        except Exception as exc:  # noqa: BLE001 - the episode's evidence comes first
            self._failed(exc)
            return None

    # ----------------------------------------------------------- frames
    def _boundary(self, phase: str, action: dict[str, Any]) -> None:
        if self.renders:
            self._fresh(phase, action)
        else:
            self.deferred = (phase, dict(action))

    def _fresh(self, phase: str, action: Mapping[str, Any], *, force: bool = True) -> None:
        step = self.clock() - self.round_start
        if not force and not self.recorder.due(round_index=self.round_index, sim_time_s=step * Y.CONTROL_PERIOD_S):
            return
        try:
            images = self.read_cameras()
        except Exception as exc:  # noqa: BLE001 - a lost frame never stops the episode
            self._failed(exc)
            return
        self._write(phase, action, images, force=force)

    def _write(self, phase: str, action: Mapping[str, Any], pixels: Images, *, force: bool) -> None:
        step = max(0, self.clock() - self.round_start)
        missing = [name for name in CAMERAS if name not in pixels]
        size = sum(np.asarray(pixels[name]).nbytes for name in CAMERAS if name in pixels)
        if missing or size > MAX_FRAME_BYTES:
            # Refused here, as a lost frame: the recorder would stop recording for the rest of the run.
            self._failed(ValueError(f"{phase} frame not recorded: "
                                    + (f"no {', '.join(missing)} image" if missing else f"{size} bytes of pixels")))
            return
        try:
            self.recorder.capture(
                lambda: pixels, round_index=self.round_index, action_id=action.get("action_id"),
                action=action.get("action"), action_sequence=action.get("action_sequence"),
                sim_step=step, sim_time_s=step * Y.CONTROL_PERIOD_S, step_unit=STEP_UNIT, phase=phase, force=force,
            )
        except Exception as exc:  # noqa: BLE001 - as above
            self._failed(exc)

    def _failed(self, error: Exception) -> None:
        self.error = f"{type(error).__name__}: {error}"
        try:
            write_json(self.output_root / "execution-capture-failure.json", {
                "schema": FAILURE_SCHEMA, "component": "original_execution_capture", "error": self.error,
                "recorded_unix_s": time.time()})
        except Exception:  # noqa: BLE001 - nothing more can be done
            pass


__all__ = ["CAMERAS", "STEP_UNIT", "YamCapture", "rgb_images"]
