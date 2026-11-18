"""Test helpers for libero reach check, shared by several test modules.

They were defined in ``tests/ora/environment/backends/libero_pro/test_libero_reach_check.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ora.environment.backends.libero_pro import reach_check as libero_reach
from ora.environment.backends.libero_pro.planned_motion import LiberoPlannedMotionMixin
from ora.environment.backends.libero_pro.planner_port import IK_RECORD_SCHEMA
from ora.environment.backends.libero_pro.planner_request import validate_libero_ik_request
from ora.environment.backends.libero_pro.world_inputs import LiberoWorldInputMixin


NAMES = [f"panda_joint{index}" for index in range(1, 8)]
MEASURED = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]
#: The robosuite Panda limits.
LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
          [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]
DOWN = [1.0, 0.0, 0.0, 0.0]  # grip site +Z pointing down: 180 degrees about world x


def _matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _pose(transform):
    return [*map(float, transform[:3, 3]), *map(float, Rotation.from_matrix(transform[:3, :3]).as_quat())]


#: A yawed, offset robot base and a grip site that is not a pure translation of
#: the hand, so a frame error in either direction changes the numbers.
WORLD_FROM_BASE = _pose(_matrix([-0.66, 0.05, 0.912, *Rotation.from_euler("z", 0.3).as_quat()]))
HAND_FROM_GRIP = _pose(_matrix([0.0, 0.0, 0.1034, *Rotation.from_euler("z", math.pi / 2).as_quat()]))
WORLD_FROM_HAND = _pose(_matrix([-0.2, 0.0, 1.25, *DOWN]))
WORLD_FROM_GRIP = _pose(_matrix(WORLD_FROM_HAND) @ _matrix(HAND_FROM_GRIP))
BASE_FROM_HAND = _pose(np.linalg.inv(_matrix(WORLD_FROM_BASE)) @ _matrix(WORLD_FROM_HAND))


def _state():
    return {"joint_names": list(NAMES), "joint_positions": list(MEASURED),
            "base_pose_world_xyzw": list(WORLD_FROM_BASE), "hand_pose_world_xyzw": list(WORLD_FROM_HAND),
            "grip_site_pose_world_xyzw": list(WORLD_FROM_GRIP)}


class FakeIk:
    """The planner's IK stand-in in the base frame, answering in the session's record.

    Reaches a hand position within ``reach_m`` of the base origin. A solution
    moves joint 1 by 0.1 rad from its seed, and its forward kinematics is the
    requested hand pose, optionally turned by ``turn_rad`` about hand x.
    """

    def __init__(self, *, reach_m=0.9, turn_rad=0.0, start=None, step=None):
        self.reach_m, self.turn_rad, self.step = reach_m, turn_rad, step
        self.start = list(BASE_FROM_HAND if start is None else start)
        self.requests: list[dict[str, Any]] = []

    def solve_ik(self, request):
        validated = validate_libero_ik_request(request)
        self.requests.append(copy.deepcopy(validated))
        if self.step is not None:
            self.step()
        results, ends = [], []
        for query in validated["queries"]:
            start = query["from_index"]
            seed = validated["joint_positions"] if start is None else ends[start]
            if seed is None:
                results.append({"solved": False, "skipped": True, "joint_positions": None,
                                "ee_pose_base_xyzw": None, "position_error_m": None, "rotation_error": None})
                ends.append(None)
                continue
            target = query["target_pose_base_xyzw"]
            if float(np.linalg.norm(target[:3])) > self.reach_m:
                results.append({"solved": False, "skipped": False, "joint_positions": None,
                                "ee_pose_base_xyzw": None, "position_error_m": 0.2, "rotation_error": 0.3})
                ends.append(None)
                continue
            joints = [seed[0] + 0.1, *seed[1:]]
            reached = _matrix(target) @ _matrix([0, 0, 0, *Rotation.from_euler("x", self.turn_rad).as_quat()])
            results.append({"solved": True, "skipped": False, "joint_positions": joints,
                            "ee_pose_base_xyzw": _pose(reached), "position_error_m": 0.0, "rotation_error": 0.0})
            ends.append(joints)
        return {"schema": IK_RECORD_SCHEMA, "status": "solved", "world_model": "none",
                "joint_names": list(NAMES), "joint_limits": copy.deepcopy(LIMITS),
                "start_ee_pose_base_xyzw": list(self.start), "results": results, "solve_time_s": 0.01}


class _Session(LiberoPlannedMotionMixin, LiberoWorldInputMixin, libero_reach.LiberoReachCheckMixin):
    """The two real mixins over a stand-in plant; nothing here can step physics."""

    def __init__(self, planner):
        self.motion_planner = planner
        self.world_mutation_sequence = 0
        self.environment = SimpleNamespace(_elapsed_steps=7)

    def _require_environment(self):
        return self.environment

    def _motion_state(self):
        return _state()
