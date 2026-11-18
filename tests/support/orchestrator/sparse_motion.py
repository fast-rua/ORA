"""A worker session that answers the scripted sparse-motion qualifications without a simulator.

``backends/robocasa/qualification.py`` and ``backends/robolab/qualification.py``
drive one real worker session through a fixed script and record each check.
``ScriptedSession`` stands in for that session: every motion reaches its
target within a millimetre, the gripper reaches its target, a wait uses its
ticks, the torso lifts, and the first ``move_pose`` can be refused as the
RoboCasa script expects of its into-the-cabinet probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: A receipt that every check of both scripts accepts.
REACHED = {
    "execution_status": "reached", "steps_used": 4, "native_actions": 4, "motion_strategy": "joint_path",
    "arms": {"right": {"pose_error": {"translation_m": 0.001, "orientation_rad": 0.001},
                       "gripper_outcome": {"state": "reached_target"}}},
    "wait": {"ticks_used": 5}, "torso": {"stop_reason": "reached", "from_m": 0.1, "to_m": 0.3},
    "base": {"base_error": {"translation_m": 0.01}},
}
REFUSED = {"execution_status": "plan_rejected", "failure_code": "target_in_collision_at_base_pose",
           "native_actions": 0, "arms": {"right": {"pose_error": None}}}
CAMERA_FILES = {"rgb_lossless_png": {}, "depth_numpy_m_f32": {}, "calibration": {}}
OBSERVATION = {
    "cameras": {"agentview": {"files": dict(CAMERA_FILES)}, "wrist": {"files": dict(CAMERA_FILES)}},
    "proprioception": {"base": {"pose_xy_yaw": [1.0, 2.0, 0.5]},
                       "end_effectors": {"right": {"pose_xyz_xyzw": [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                                   "jaw_separation_m": 0.085}}},
    "robot_geometry_verification": {"passed": True},
}


class ScriptedSession:
    """``start``, ``observe``, ``execute``, ``probe_motion``, ``status``, the verdict and ``close``."""

    def __init__(self, *, reset: Mapping[str, Any], verdict: Mapping[str, Any],
                 refuse_first_move_pose: bool = False, close_error: Exception | None = None,
                 fail_at: str | None = None) -> None:
        self.reset, self.verdict = dict(reset), dict(verdict)
        self.refuse_first_move_pose, self.close_error, self.fail_at = refuse_first_move_pose, close_error, fail_at
        self.actions: list[dict[str, Any]] = []
        self.observed: list[Path] = []
        self.probes: list[dict[str, Any]] = []
        self.closed = False

    def start(self) -> dict[str, Any]:
        return dict(self.reset)

    def observe(self, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        self.observed.append(root)
        return {**OBSERVATION, "observation_id": f"obs-{len(self.observed)}"}, {}

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        self.actions.append(dict(action))
        if action["action"] == self.fail_at:
            raise RuntimeError(f"the worker lost {self.fail_at}")
        moves = [item for item in self.actions if item["action"] == "move_pose"]
        if self.refuse_first_move_pose and action["action"] == "move_pose" and len(moves) == 1:
            return dict(REFUSED)
        return dict(REACHED)

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.probes.append(dict(request))
        return {"schema": "ora.motion-probe-result.v1",
                "results": [{"reachable": True, "variants": [{"reachable": True}, {"reachable": False}]}]}

    def status(self) -> dict[str, Any]:
        return {"terminal": False, "reason": "running"}

    def persist_official_verdict(self) -> dict[str, Any]:
        return dict(self.verdict)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


__all__ = ["OBSERVATION", "REACHED", "REFUSED", "ScriptedSession"]
