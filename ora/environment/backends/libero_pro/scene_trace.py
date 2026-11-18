"""A LIBERO-Pro scene for the scene trace: the session's simulator and Panda, for ``mujoco_scene_trace.py``.

Runs in the replay worker around an episode's re-execution. The simulator lives in this process
(``direct.py``), so the session's attribute port hands over the live ``MjSim`` and robot. The arm's TCP is the
``grip_site`` the OSC controller and the receipts use (``harness_grip_site_pose``), under the name ``right`` as in
the receipts. A LIBERO hard reset builds a new ``MjSim``, so each round reads its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ora.environment.mujoco_scene_trace import MuJoCoSceneTrace


class LiberoSceneTrace(MuJoCoSceneTrace):
    def __init__(self, directory: Path, *, run_id: str) -> None:
        super().__init__(directory, run_id=run_id, backend="libero_pro")

    def begin_round(self, session: Any, round_index: int) -> None:  # type: ignore[override]
        robots = session._worker_attr("robots")
        super().begin_round(session._worker_attr("sim"), round_index,
                            robot_roots=[str(robot.robot_model.root_body) for robot in robots],
                            tcp_sites={"right": str(robots[0].gripper.important_sites["grip_site"])})


__all__ = ["LiberoSceneTrace"]
