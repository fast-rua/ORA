"""Test helpers for robotwin ik feedback, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robotwin/test_robotwin_ik_feedback.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ora.environment.backends.robotwin.planning import ValidatedRoboTwinPlan, select_topk_validated_plan
from ora.environment.backends.robotwin.session import RoboTwinLiveSession
from ora.environment.controllers.profile_capabilities import PROFILE_VERSION_IDS
from tests.support.robotwin.correctness_repair import vendored_planner_with_fake_solver


def _plan(solutions, successes, *, expose_limits=True):
    # Execute the actual vendored class with the existing CPU-only solver port.
    planner = vendored_planner_with_fake_solver(ValueError)
    namespace = planner._plan_pose_via_nearest_ik.__globals__
    tensor = namespace["torch"].tensor

    class IK:
        def __init__(self, _config):
            self.kinematics = SimpleNamespace()
            if expose_limits:
                self.kinematics.get_joint_limits = lambda: SimpleNamespace(
                    position=[[-1] * 6, [1] * 6],
                )

        def solve_single(self, *args, **kwargs):
            return SimpleNamespace(
                success=successes, solution=tensor(solutions),
                position_error=None, rotation_error=None,
            )

    namespace["IKSolver"] = IK
    return planner._plan_pose_via_nearest_ik(object(), [0.0] * 6, object())


def _no_valid_endpoint():
    """The actual vendored solve: two seeds, both refused by the solver."""
    return _plan([[.1] * 6, [.2] * 6], [False, False])


def _valid_and_planned():
    """Two valid endpoints; the first planned joint goal produced a trajectory."""
    planner = vendored_planner_with_fake_solver(ValueError)
    return planner._plan_pose_via_nearest_ik(object(), [0.0] * 6, object())


def _validated():
    return ValidatedRoboTwinPlan(
        position=np.zeros((2, 6)), velocity=np.zeros((2, 6)),
        planned_link6_pose_xyzw=np.zeros((2, 7)), planned_tcp_pose_xyzw=np.zeros((2, 7)),
        summary={"planning_endpoint_error": {"translation_m": 0.001, "orientation_rad": 0.01}},
    )


def _session(tmp_path, raw_factory, *, version=10, admit=False):
    def plan(targets):
        return {"schema": "robotwin.curobo-topk-batch-result.v1", "results": [raw_factory() for _ in targets]}

    environment = SimpleNamespace(robot=SimpleNamespace(left_plan_multi_path=plan, right_plan_multi_path=plan))
    session = object.__new__(RoboTwinLiveSession)
    session.control_profile = {
        "profile": {"profile_id": PROFILE_VERSION_IDS[version], "plan_validation": {}},
        "sha256": "c" * 64,
    }
    session.output_root = tmp_path
    session.source_manifest = {"robotwin_revision": "test-source"}
    session.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
    session.world_mutation_sequence = 0
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session._require_environment = lambda: environment
    session._uses_motion_transaction_profile = lambda: False
    session._measured_arm_joints = lambda _env, _arm: [0.0] * 6
    session._public_pose_wxyz = lambda _env, _arm: [0, 0, 1, 1, 0, 0, 0]
    session._public_to_planner_wxyz = lambda _arm, pose: pose
    session._arm_joint_limits = lambda _env, _arm: [[-1, 1]] * 6
    session._arm_profile = lambda _arm: {"planner_joint_names": [f"j{i}" for i in range(6)]}
    if admit:
        # The real top-K selection over the vendored candidates; only the
        # FK/workspace validator and the plan-reference writer are faked.
        session._select_motion_probe_candidate = lambda probe, arm, variant, raw: (
            *select_topk_validated_plan(raw, validator=lambda _candidate: _validated())[:2], {})
        session._register_motion_probe_plan = lambda *args: "plan-" + "b" * 32
    return session
