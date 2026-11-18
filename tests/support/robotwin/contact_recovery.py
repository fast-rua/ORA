"""Test helpers for robotwin contact recovery, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robotwin/test_robotwin_contact_recovery.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ora.environment.backends.robotwin.planning import ValidatedRoboTwinPlan
from ora.environment.controllers.embodiment_control import load_embodiment_control_profile
from ora.environment.controllers.motion_transaction import MotionBackendCapabilities
from ora.environment.controllers.profile_capabilities import FROZEN_CONTROL_PROFILE_SHA256


ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = ROOT / "ora/policy/embodiments/aloha-agilex"


def binding():
    return load_embodiment_control_profile(PROFILE_ROOT / "control-profile-v10.json", FROZEN_CONTROL_PROFILE_SHA256[10])


def path(distances):
    points = np.asarray([[distance, 0, 1, 0, 0, 0, 1] for distance in distances], dtype=float)
    joints = points[:, :3]
    return ValidatedRoboTwinPlan(joints, np.zeros_like(joints), points, points, {"status": "accepted"})


class Plant:
    capabilities = MotionBackendCapabilities(
        backend_id="robotwin", embodiment_id="aloha_agilex", arms=("left", "right"),
        controlled_point="parallel_jaw_grip_center_tcp",
        planner_frames={"left": "left_link6", "right": "right_link6"},
        synchronized_groups=(("left", "right"),), global_joint_planning=True, local_cartesian=True,
    )

    def __init__(self, *, separation=.4, gain=1.):
        self.measured = {"left": np.array([-separation / 2, 0., 1.]), "right": np.array([separation / 2, 0., 1.])}
        self.targets = copy.deepcopy(self.measured)
        self.gain = gain
        self.steps = 0
        self.writes = []

    def read_joint_positions(self, arm):
        return self.measured[arm].tolist()

    def read_public_tcp_pose(self, arm):
        return [*self.read_joint_positions(arm), 0., 0., 0., 1.]

    def write_joint_target(self, arm, position, velocity):
        self.targets[arm] = np.asarray(position, dtype=float)
        self.writes.append((arm, list(position)))

    def step_physics(self):
        self.steps += 1
        for arm in self.measured:
            self.measured[arm] += self.gain * (self.targets[arm] - self.measured[arm])

    def capture_action_start(self):
        return {"measured_joint_positions": copy.deepcopy(self.measured)}

    def compute_local_joint_target(self, arm, joints, current, target, _profile):
        return {"position": list(target[:3]), "velocity": [0.] * 3,
                "predicted_score_before": float(np.linalg.norm(np.asarray(target[:3]) - current[:3])),
                "predicted_score_after": 0., "line_search_scale": 1., "solver": {"fixture": True}}
