"""What reached the simulator, per native action: recorded while a run executes and fed back by its replay.

Isaac Lab's differential-IK term turns a pose into joint targets on every physics step from the measured state, so
a plan or a TCP target does not name the motion. What does is the joint-target buffer the articulation hands PhysX, in
``Articulation.write_data_to_sim``: ``JointTargetTap`` sits there. One record per action, in the ``controller-replay``
layout RoboTwin and LIBERO-Pro use (``environment/controller_replay.py``), holds for an action of T native actions:

    native_actions   (T, 8) float32: the tensor ``env.step`` received, flange pose (xyz, wxyz) and finger action
    joint_targets    (T, D, J) float32: every joint's position target at each of the D physics steps

Recording is neutral: it copies a buffer between ``apply_action`` and the write, draws no random number, writes nothing
to the simulator and makes no simulator call that changes state, and a failed recording stops the record, not the run
(the run's program is then unavailable). The reset's two held native actions are not recorded: a replay runs the same
reset. The replay (``environment/replay_program_worker.py``) writes each recorded row over the buffer before it reaches
PhysX, so neither the IK term nor any executor decides anything again, and a later code change cannot alter it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ora.environment.benchmarks.robolab import DECIMATION, PUBLIC_CAMERAS
from ora.environment.controller_replay import ControllerReplayRecorder

from .common import array, write_json

BACKEND = "robolab"
CONTROLLER = "robolab_joint_targets"
#: Beside the record when a write failed: the record ends there and the run goes on.
FAILURE_FILE = "recording-failed.json"


class JointTargetTap:
    """Stands in for one articulation's ``write_data_to_sim``; ``take`` hands over the rows since the last one."""

    def __init__(self, robot: Any) -> None:
        self.robot, self.write = robot, robot.write_data_to_sim
        self.rows: list[Any] = []
        self.feed: Iterator[Any] | None = None  # a replay's rows, one per physics step
        robot.write_data_to_sim = self

    def __call__(self) -> None:
        if self.feed is not None:
            self.robot.set_joint_position_target(next(self.feed))
        self.rows.append(array(self.robot.data.joint_pos_target[0]).copy())
        self.write()

    def take(self) -> Any:
        rows, self.rows = self.rows, []
        return np.stack(rows)


class ControlRecord:
    """The run's ``controller-replay`` record; no method raises, so nothing here can stop a run."""

    def __init__(self, output_root: Path, *, identity: Mapping[str, Any]) -> None:
        self.output_root = Path(output_root)
        self.recorder: ControllerReplayRecorder | None = None
        try:
            self.recorder = ControllerReplayRecorder(output_root=self.output_root, backend=BACKEND,
                                                     environment_identity=identity, camera_panels=list(PUBLIC_CAMERAS))
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._stop(exc)

    def _stop(self, error: Exception) -> None:
        self.recorder = None
        try:
            write_json(self.output_root / "controller-replay" / FAILURE_FILE,
                       {"reason": f"{type(error).__name__}: {error}"})
        except Exception:  # noqa: BLE001 - the missing entries say the same
            pass

    def begin_round(self, round_index: int) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.begin_round(round_index)
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._stop(exc)

    def action(self, sequence: int, action: Mapping[str, Any], ticks: Sequence[tuple[Any, Any]],
               receipt: Mapping[str, Any]) -> None:
        """One executed action's ticks (``RobolabBridge.take_ticks``); none for an action that moved nothing."""
        if self.recorder is None:
            return
        try:
            arrays = ({"native_actions": np.stack([native for native, _ in ticks]),
                       "joint_targets": np.stack([rows for _, rows in ticks])} if ticks else {})
            self.recorder.record(action_sequence=sequence, action=action, controller=CONTROLLER, arrays=arrays,
                                 physics_steps=len(ticks) * DECIMATION, replayable=bool(ticks))
            self.recorder.update(sequence, execution_status=str(receipt.get("execution_status")))
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._stop(exc)


__all__ = ["BACKEND", "CONTROLLER", "ControlRecord", "FAILURE_FILE", "JointTargetTap"]
