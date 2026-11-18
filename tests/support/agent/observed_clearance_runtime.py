"""Test helpers for observed clearance runtime, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/test_observed_clearance_runtime.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from ora.agent.toolkit.host.motion_rules import clearance_runtime as rt


#: link6 approach (+x) straight down; jaw line along world y.
DOWN = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
TABLE_Z, RIM_TOP = 0.75, 0.842
#: A basket wall 5 mm thick beside the jaws' x span.
WALL = ((0.046, -0.20, TABLE_Z), (0.051, 0.20, RIM_TOP))


def _render(camera_to_world, intrinsic, size, boxes):
    from tests.support.agent.observed_clearance import _render as render

    return render(camera_to_world, intrinsic, size, boxes)


def _write_scene(root: Path, *, left_pose, right_pose, left_close=False, left_opening=1.0) -> dict:
    """A Dojo-shaped observation of a table and one basket wall, three depth cameras."""
    from tests.support.agent.observed_clearance import _camera

    root.mkdir(parents=True)
    cameras = {}
    specs = {"head": ((0.0, -0.35, 1.6), (0.0, 0.25, -1.0)),
             "left_wrist": ((-0.6, 0.0, 1.5), (0.0, 0.0, -1.0)),
             "right_wrist": ((0.6, 0.0, 1.5), (0.0, 0.0, -1.0))}
    for name, (position, look) in specs.items():
        size = (400, 300)
        camera_to_world, intrinsic = _camera(position, look=look, size=size, focal=300.0)
        folder = root / name
        folder.mkdir()
        np.save(folder / "depth-m.npy", _render(camera_to_world, intrinsic, size, [WALL]), allow_pickle=False)
        (folder / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": np.linalg.inv(camera_to_world).tolist(),
            "cam2world_cv": camera_to_world.tolist()}))
        (folder / "rgb.png").write_bytes(b"")
        cameras[name] = {"files": {
            role: {"path": f"{name}/{file}", "sha256": hashlib.sha256((folder / file).read_bytes()).hexdigest()}
            for role, file in (("depth_numpy_m_f32", "depth-m.npy"), ("calibration", "calibration.json"),
                               ("rgb_lossless_png", "rgb.png"))}}
    def effector(pose, close, opening):
        finger = 0.044 * opening
        return {"pose_xyz_xyzw": list(pose), "gripper_finger_qpos": [finger, finger],
                "gripper_command_target": "close" if close else "open", "gripper": opening,
                "arm_joint_qpos": [0.0] * 6,
                "gripper_feedback": {"arrival_check": {"tolerance_normalized": 0.03}}}
    observation = {
        "observation_id": "robodojo:scene:round-0:observation-0007", "cameras": cameras,
        "proprioception": {"end_effectors": {
            "left": effector(left_pose, left_close, left_opening),
            "right": effector(right_pose, False, 1.0)}}}
    (root / "observation.json").write_text(json.dumps(observation))
    return observation


HOME_RIGHT = [0.6, -0.3, 1.1, *DOWN]


def _dojo_runtime(tmp_path, *, declared=True, left_close=False, left_opening=1.0):
    from tests.support.agent.move_pose_reach import _dojo_runtime as runtime_for

    runtime = runtime_for(tmp_path)
    if not declared:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            item for item in runtime.surface.capabilities if item != rt.V2_CAPABILITY))
    source = tmp_path / "scene"
    observation = _write_scene(source, left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT,
                               left_close=left_close, left_opening=left_opening)
    pointer, _source, released = runtime._current()
    observation = {**copy.deepcopy(released), **observation,
                   "proprioception": {**released["proprioception"], **observation["proprioception"]}}
    pointer = {**pointer, "observation_sha256": hashlib.sha256(
        json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    runtime.align_plan_callback = None
    return runtime


def _consume(runtime):
    """The environment takes the submitted action; the same observation stays current here."""
    (runtime.control_root / "submitted-action.json").unlink()
