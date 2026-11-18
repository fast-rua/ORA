"""The public snapshot a world-mode CuRobo planner builds its collision world from.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. When the planner client is bound to a world model other than ``none``,
every plan, probe and reach check first writes the current public state to
``curobo/world-inputs/<kind>-NNNN.npz`` (``depth_world.write_world_inputs``):
both cameras' metric depth exactly as ``observe`` would publish it, their
calibration, the measured base pose and joints, the finger positions and the
held gripper directive. The worker reads that file by path and sha256.

The depth is current. ``current_raw_obs`` is the observation of the last
physics step that ran: the driver's ``step`` replaces it, a joint trajectory
hands back its last tick's observation (``planned_motion._refresh_joint_observation``)
and a reset sets it, so between two steps it shows the scene the planner is
asked about, and no new rendering is needed.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.controllers.depth_world import write_world_inputs

#: Public camera name -> (LIBERO camera, raw depth key), as ``observe`` reads them.
WORLD_CAMERAS = {
    "agentview": ("agentview", "agentview_depth"),
    "wrist": ("robot0_eye_in_hand", "robot0_eye_in_hand_depth"),
}


class LiberoWorldInputMixin:
    """Write the world-input snapshot of one planner call, in world mode only."""

    def _world_input(self, motion_state: Mapping[str, Any], kind: str) -> dict[str, Any]:
        """``{"world_input": {path, sha256}}`` for the planner request, or ``{}`` robot-only."""

        import numpy as np

        if getattr(self.motion_planner, "world_model", "none") == "none":
            return {}
        if kind == "action":
            name = f"action-{self.action_sequence:04d}"
        else:
            name = f"{kind}-{self.world_input_sequence:06d}"
            self.world_input_sequence += 1
        raw = self._raw_observation()
        cameras = {}
        for public_name, (source_name, depth_key) in WORLD_CAMERAS.items():
            height, width = np.asarray(raw[depth_key]).shape[:2]
            depth, meta = self._public_depth(raw, source_name, depth_key, int(height), int(width))
            cameras[public_name] = {
                "depth_m": depth, "intrinsic_cv": meta["intrinsic_K"],
                "cam2world_cv": meta["extrinsic_cam2world"],
                "depth_range_m": (float(meta["depth_near"]), float(meta["depth_far"])),
            }
        record = write_world_inputs(
            (self.output_root / "curobo/world-inputs" / f"{name}.npz").resolve(),
            cameras=cameras,
            base_pose_world_xyzw=motion_state["base_pose_world_xyzw"],
            joint_positions=motion_state["joint_positions"],
            gripper_finger_qpos=[float(value) for value in raw["robot0_gripper_qpos"]],
            gripper_directive=self._gripper_hold_directive(),
        )
        return {"world_input": record}
