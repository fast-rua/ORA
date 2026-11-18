"""Test helpers for observed clearance v2, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/test_observed_clearance_v2.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


TABLE_Z, RIM_TOP = 0.75, 0.842
WALL = ((0.046, -0.20, TABLE_Z), (0.051, 0.20, RIM_TOP))
#: ALOHA TCP approach (+x) straight down, jaw line (+y) along world y.
ALOHA_DOWN = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
#: Panda grip-site approach (+z) straight down, jaw line (+x) along world x.
PANDA_DOWN = [1.0, 0.0, 0.0, 0.0]


def _write_scene(root: Path, cameras: dict, effectors: dict, *, boxes=(WALL,)) -> dict:
    from tests.support.agent.observed_clearance import _camera, _render

    root.mkdir(parents=True)
    records = {}
    for name, (position, look) in cameras.items():
        size = (400, 300)
        camera_to_world, intrinsic = _camera(position, look=look, size=size, focal=300.0)
        folder = root / name
        folder.mkdir()
        np.save(folder / "depth-m.npy", _render(camera_to_world, intrinsic, size, list(boxes)), allow_pickle=False)
        (folder / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": np.linalg.inv(camera_to_world).tolist()}))
        records[name] = {"files": {
            role: {"path": f"{name}/{file}", "sha256": hashlib.sha256((folder / file).read_bytes()).hexdigest()}
            for role, file in (("depth_numpy_m_f32", "depth-m.npy"), ("calibration", "calibration.json"))}}
    observation = {"observation_id": "port:scene:observation-0003", "cameras": records,
                   "proprioception": {"end_effectors": effectors}}
    (root / "observation.json").write_text(json.dumps(observation))
    return observation


def _aloha(pose, *, close=False, opening_m=0.04):
    return {"pose_xyz_xyzw": list(pose), "gripper_command_target": "close" if close else "open",
            "gripper": 0.5, "gripper_opening_m": opening_m}


def _panda(pose, *, close=False, finger=0.02):
    return {"pose_xyz_xyzw": list(pose), "gripper_command_target": "close" if close else "open",
            "gripper": finger / 0.04, "gripper_finger_qpos": [finger, -finger]}


ROBOTWIN_CAMERAS = {"head_camera": ((0.0, -0.35, 1.6), (0.0, 0.25, -1.0)),
                    "left_camera": ((-0.6, 0.0, 1.5), (0.0, 0.0, -1.0)),
                    "right_camera": ((0.6, 0.0, 1.5), (0.0, 0.0, -1.0))}
LIBERO_CAMERAS = {"agentview": ((0.0, -0.35, 1.6), (0.0, 0.25, -1.0)),
                  "wrist": ((0.3, 0.3, 1.5), (0.0, 0.0, -1.0))}
