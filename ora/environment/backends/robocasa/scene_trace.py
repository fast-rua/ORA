"""A RoboCasa365 scene for the scene trace: the session's simulator and PandaOmron, for ``mujoco_scene_trace.py``.

Runs in the replay worker around an episode's re-execution. The robot's one tree holds the Omron base, the torso, the
Panda and its gripper, so the base's pose is drawn like any link; the arm's TCP is the grip site the receipts read, under
the name ``right`` as in the receipts. Each round builds a new kitchen and ``MjSim``, so each reads its own.

Three things differ from LIBERO-Pro:

* robosuite's lite physics steps through ``MjSim.step1`` and ``step2``, so ``PhysicsStepTap``'s ``step`` never runs.
  The replay drives the control steps itself and calls ``after_tick``, and a row falls on every control step (the
  ``stride`` is its 25 MuJoCo steps, 20 Hz). A row reads the state the run read at that step: nothing runs
  ``mj_forward`` after the last ``step2`` of a control step.
* A kitchen's long counter and island pass ``BACKDROP_EXTENT_M`` in one direction, and they are what the arm works over.
  Only a body past it in two directions (the walls, the floor and the shells behind them) is the backdrop.
* A kitchen is mostly fixtures, 265 bodies in the kitchens checked: ``kind`` stays the model's, so a cabinet door is an
  ``object``, moving with a pose per row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ora.environment.controllers import robocasa_omron as C
from ora.environment.mujoco_scene_trace import MuJoCoSceneTrace

#: A static body past ``scene_trace.BACKDROP_EXTENT_M`` in this many directions is a wall or a floor.
BACKDROP_DIRECTIONS = 2


class RoboCasaSceneTrace(MuJoCoSceneTrace):
    def __init__(self, directory: Path, *, run_id: str) -> None:
        super().__init__(directory, run_id=run_id, backend="robocasa")
        self.writer.backdrop_directions = BACKDROP_DIRECTIONS
        self.steps_per_tick = 0

    def watch(self, scene: Any) -> None:
        """No tap: ``after_tick`` counts the steps (see the module docstring)."""

    def begin_round(self, session: Any, round_index: int) -> None:  # type: ignore[override]
        environment = session.environment
        self.steps_per_tick = int(session.physics_steps_per_tick)
        super().begin_round(environment.sim, round_index,
                            robot_roots=[str(robot.robot_model.root_body) for robot in environment.robots],
                            tcp_sites={"right": C.GRIP_SITE})
        self.writer.stride = self.steps_per_tick

    def after_tick(self) -> None:
        """One control step of the round ran."""
        for _ in range(self.steps_per_tick):
            self.writer.after_step()


__all__ = ["BACKDROP_DIRECTIONS", "RoboCasaSceneTrace"]
