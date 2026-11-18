"""RoboLab on the shared motion base: the planned route, its refusals, the servo fallback and the reach check.

The planner, timing and Isaac are fakes; the frames, the route rule, the receipts and the reach rows are the
backend's own. The live path is ``ora qualify robolab motion`` and the smokes.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ora.contracts.policy_receipt import policy_receipt
from ora.environment.backends.robolab import action_plan, motion, reach_check, shared_motion
from ora.environment.backends.robolab.common import RobolabError
from ora.environment.controllers import robolab_droid as D
from ora.environment.controllers.depth_world import ROBOT_WORLDS, tool_from_hand
from ora.environment.controllers.jaw_flip_retry import jaw_twin
from ora.environment.controllers.motion_pace import PacedPath, PaceRefusal
from ora.environment.controllers.motion_planner import IMPLEMENTATIONS, ROBOTS
from ora.environment.planners.curobo_v2 import robots

IDENTITY = [0.0, 0.0, 0.0, 1.0]
START = [0.3, 0.0, 0.4, *IDENTITY]
FAR = [0.6, 0.2, 0.3, *IDENTITY]
JOINTS = [0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0]


class Bridge:
    def __init__(self, *, stuck: bool = False) -> None:
        self.pose, self.joints, self.stuck = list(START), list(JOINTS), stuck
        self.targets: list[list[float]] = []
        self.rows: list[list[float]] = []

    def tcp_pose(self) -> list[float]:
        return list(self.pose)

    def arm_qpos(self) -> list[float]:
        return list(self.joints)

    def clock(self) -> int:
        return 7

    def step(self, target: list[float], _finger: float) -> dict[str, Any]:
        self.targets.append(list(target))
        self.pose = list(target)
        return {}

    def step_joints(self, joints: list[float], _finger: float) -> dict[str, Any]:
        self.rows.append(list(joints))
        if not self.stuck:
            self.joints = list(joints)
            self.pose = list(FAR)
        return {}

    def gripper_contact_force(self) -> float:
        return 0.0


class Session:
    finger_action, native_actions, physics_steps_per_tick, round_index, is_terminal = 0.0, 0, 8, 0, False
    native_action_limit = 900
    home_pose, last_safe_pose = list(START), None
    motion_planner = object()

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge

    def arm_state(self) -> dict[str, Any]:
        return {"right": {"pose_xyz_xyzw": self.bridge.tcp_pose()}}

    def after_native_action(self, _result: Any) -> None:
        self.native_actions += 1

    def must_stop(self) -> bool:
        return False

    def workspace_refusal(self, _pose: Any) -> None:
        return None


class Paced:
    def rows(self) -> list[list[float]]:
        return [[0.1 * index, *JOINTS[1:]] for index in range(1, 4)]

    def report(self) -> dict[str, Any]:
        return {"speed": {"speed": "fast"}, "wrist": {"start_deg": 0.0}, "planned_ticks": 3,
                "timing": {"implementation": "toppra_ruckig"}}


def _action(pose: list[float], interaction: str = "free_space") -> dict[str, Any]:
    return {"action_id": "action-1", "action": "move_pose", "arguments": {
        "targets": {"right": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}},
        "interactions": {"right": interaction}}}


def _run(monkeypatch, answer: dict[str, Any], *, pose: list[float] = FAR, paced: Any = Paced,
         stuck: bool = False, interaction: str = "free_space") -> tuple[dict[str, Any], Bridge]:
    monkeypatch.setattr(shared_motion, "plan", lambda _session, _pose: answer)
    monkeypatch.setattr(shared_motion, "pace", lambda *_args: paced())
    bridge = Bridge(stuck=stuck)
    return motion.execute_motion_v3(Session(bridge), _action(pose, interaction)), bridge


PLANNED = {"status": "planned", "reason": None, "start_joints": JOINTS, "path_joints": [JOINTS], "world_sha256": "0"}


def test_the_robot_joins_the_shared_lists_without_editing_the_kernel_files():
    assert ROBOTS["franka_robotiq"] == {"joint_names": D.ARM_JOINT_NAMES, "tool_frame": "panda_hand"}
    assert "franka_robotiq" in IMPLEMENTATIONS["curobo_v2"].robots
    assert ROBOT_WORLDS["franka_robotiq"][0].fingertip_m == pytest.approx(0.1685)
    # The hand convention is panda_link8's axes: approach +z, jaw line y.
    assert (tool_from_hand("franka_robotiq")[:3, :3] @ [0.0, 1.0, 0.0]) == pytest.approx(
        [-np.sqrt(0.5), np.sqrt(0.5), 0.0])


def test_the_robotiq_replaces_the_hands_spheres_and_keeps_the_arms():
    bundled = {"robot_cfg": {"kinematics": {"collision_spheres": {
        "panda_link7": [{"center": [0, 0, 0], "radius": 0.05}], "panda_hand": [], "panda_leftfinger": [],
        "panda_rightfinger": []}}}}
    kinematics = robots.robot_config("franka_robotiq", bundled, held_link="attached_object", held_slots=32)[
        "robot_cfg"]["kinematics"]
    spheres = kinematics["collision_spheres"]
    assert [len(spheres[link]) for link in ("panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger")] == [
        1, 6, 10, 10]
    assert kinematics["collision_sphere_buffer"] == robots.SPHERE_BUFFER_M
    # The fingertips reach 168.5 mm along the approach; the spheres cover them.
    left = max(item["center"][2] + 0.0584 + item["radius"] for item in spheres["panda_leftfinger"])
    assert 0.1685 <= left <= 0.18


def test_the_tcp_and_the_planners_hand_are_one_rigid_transform():
    assert shared_motion.HAND_FROM_TCP[:3, 3] == pytest.approx([0.0, 0.0, 0.14917])
    # The TCP's approach is the hand's +z.
    assert shared_motion.HAND_FROM_TCP[:3, 2] == pytest.approx([0.0, 0.0, 1.0])
    pose = [0.36, 0.1, 0.34, 0.0, 1.0, 0.0, 0.0]
    assert shared_motion.tcp_from_hand_pose(shared_motion.hand_from_tcp_pose(pose)) == pytest.approx(pose)


@pytest.mark.parametrize("interaction,target,planned", [
    ("free_space", FAR, True), ("precision", FAR, True), ("contact_approach", FAR, False),
    ("free_space", [0.33, 0.0, 0.4, *IDENTITY], False), ("precision", [0.3, 0.0, 0.36, *IDENTITY], False),
    ("free_space", [0.3, 0.0, 0.4, 0.0, 0.0, 0.2588, 0.9659], True),
])
def test_a_free_move_plans_and_a_contact_or_short_move_stays_on_the_servo(interaction, target, planned):
    assert shared_motion.global_route(interaction, START, target) is planned


def test_a_planned_move_streams_its_rows_then_the_servo_settles_it(monkeypatch):
    receipt, bridge = _run(monkeypatch, PLANNED)
    assert receipt["execution_status"] == "reached" and receipt["motion_strategy"] == "planned_joint_path"
    assert receipt["executor"] == [motion.PATH_EXECUTOR, motion.ARM_EXECUTOR]
    assert bridge.rows == Paced().rows() and receipt["endpoint"]["path"]["rows_sent"] == 3
    assert "timing" not in json.dumps(policy_receipt(receipt)) and receipt["motion"]["planned_ticks"] == 3


def test_a_planner_refusal_moves_nothing(monkeypatch):
    diagnostics = {"phase": "ik", "outcome": "goal_unreachable", "recovery": ["The fingers would be 1.0 cm inside."]}
    receipt, bridge = _run(monkeypatch, {"status": "refused", "reason": "path_blocked",
                                         "planning_diagnostics": diagnostics})
    assert (receipt["execution_status"], receipt["failure_stage"], receipt["failure_code"]) == (
        "plan_rejected", "planner", "path_blocked")
    assert receipt["planning_diagnostics"] == diagnostics
    assert bridge.rows == [] and bridge.targets == [] and receipt["native_actions"] == 0


def test_a_paced_refusal_moves_nothing_and_names_the_numbers(monkeypatch):
    def refused() -> None:
        raise PaceRefusal("the wrist unwinds", code="paced_path_wrist_unwind", constraints={"excess_deg": 80.0})

    receipt, bridge = _run(monkeypatch, PLANNED, paced=refused)
    assert (receipt["failure_stage"], receipt["failure_code"]) == ("pacing", "paced_path_wrist_unwind")
    assert receipt["motion"]["refused"]["excess_deg"] == 80.0 and bridge.rows == []


def test_a_planner_that_cannot_answer_leaves_the_move_to_the_servo(monkeypatch):
    receipt, bridge = _run(monkeypatch, {"status": "unavailable", "reason": "stale_observation"})
    assert receipt["execution_status"] == "reached" and bridge.rows == [] and bridge.targets
    assert receipt["motion_strategy"] == "cartesian_servo_lift_traverse"
    assert receipt["motion"]["planner"] == {"status": "unavailable", "reason": "stale_observation"}


def test_an_arm_held_back_ends_the_path(monkeypatch):
    far_rows = type("FarPaced", (Paced,), {"rows": lambda self: [[1.0, *JOINTS[1:]]] * 40})
    receipt, bridge = _run(monkeypatch, PLANNED, paced=far_rows, stuck=True)
    assert receipt["execution_status"] == "not_reached" and receipt["failure_code"] == "controller_stalled"
    assert len(bridge.rows) == D.PATH_BLOCKED_TICKS


class Planner:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def solve_ik(self, request: dict[str, Any]) -> dict[str, Any]:
        assert request["robot"] == shared_motion.ROBOT and request["world"]["model"] == "public_depth_voxel_v1"
        self.request = request
        return {"status": "solved", "reason": None, "results": self.results,
                "joint_limits": [[-3.0] * 7, [3.0] * 7]}


def _reach(monkeypatch, results: list[dict[str, Any]], starts: list[int | None]) -> dict[str, Any]:
    session = Session(Bridge())
    session.motion_planner = Planner(results)
    monkeypatch.setattr(shared_motion, "_world_input", lambda _session: {"path": "/world.npz", "sha256": "0" * 64})
    queries = [{"pose_xyzw": START, "interaction": "free_space", "from_index": start} for start in starts]
    return reach_check.reach_check_batch(session, "right", queries, position_tolerance_m=D.POSITION_TOLERANCE_M,
                                         orientation_tolerance_rad=D.ORIENTATION_TOLERANCE_RAD)


def _solved(pose: list[float], joints: list[float] = JOINTS) -> dict[str, Any]:
    return {"solved": True, "skipped": False, "refusal": None, "joint_positions": joints,
            "tool_pose_base_xyzw": shared_motion.hand_from_tcp_pose(pose), "position_error_m": 0.0,
            "rotation_error": 0.0}


FAILED = {"solved": False, "skipped": False, "refusal": None, "joint_positions": None, "tool_pose_base_xyzw": None,
          "position_error_m": None, "rotation_error": None}
SKIPPED = {**FAILED, "skipped": True}


def test_the_reach_check_answers_at_the_tcp_and_never_claims_the_admission(monkeypatch):
    answer = _reach(monkeypatch, [_solved(START), _solved([0.31, 0.0, 0.4, *IDENTITY]), FAILED, SKIPPED],
                    [None, None, None, 2])
    assert answer["ik_model"] == reach_check.IK_MODEL and answer["matches_move_pose_admission"] is False
    solved, residual, failed, orphan = answer["results"]
    assert solved["reachable"] is True and solved["position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert solved["joint_limit_margin_rad"] == pytest.approx(0.5)
    assert (residual["reachable"], residual["reason"]) == (False, "fk_residual_exceeded")
    assert (failed["reason"], orphan["reason"]) == ("ik_failed", "start_unreachable")


def test_the_reach_check_refuses_without_a_planner_and_serves_only_its_schema():
    session = Session(Bridge())
    session.motion_planner = None
    with pytest.raises(RobolabError, match="cannot answer"):
        reach_check.reach_check_batch(session, "right", [{"pose_xyzw": START, "from_index": None}],
                                      position_tolerance_m=0.005, orientation_tolerance_rad=0.05)
    with pytest.raises(RobolabError, match="only the internal reach check"):
        reach_check.RobolabReachCheckMixin().align_plan({"schema": "other"})


def test_the_planner_binding_comes_from_the_runs_preflight(tmp_path):
    planner = {"implementation": "curobo_v2", "python_env": "CUROBO_V2_PYTHON", "root_env": "CUROBO_V2_ROOT"}
    assert shared_motion.binding({}, tmp_path / "native") is None
    named = {**planner, "python_executable": "/env/python", "root": "/curobo"}
    assert shared_motion.binding({"motion_planner": named}, tmp_path / "native") == named
    with pytest.raises(RobolabError, match="preflight binding is absent"):
        shared_motion.binding({"motion_planner": planner}, tmp_path / "native")
    (tmp_path / "preflight.json").write_text(json.dumps({"checks": [
        {"check": shared_motion.PREFLIGHT_CHECK, "passed": True,
         "detail": {"python_executable": "/env/python", "root": "/curobo"}}]}), encoding="utf-8")
    assert shared_motion.binding({"motion_planner": planner}, tmp_path / "native") == named


TWIN = jaw_twin(FAR)
TWIN_ROWS = [[0.1 * index, *JOINTS[1:]] for index in range(1, 4)]


class TwinBridge(Bridge):
    """A path that runs lands the TCP on the jaw twin of ``FAR``."""

    def step_joints(self, joints: list[float], finger: float) -> dict[str, Any]:
        super().step_joints(joints, finger)
        self.pose = list(TWIN)
        return {}


def _twin_stubs(monkeypatch) -> list[list[float]]:
    """The wrist check refuses the path to any pose but the jaw twin of ``FAR``; returns the poses planned."""
    asked: list[list[float]] = []

    def plan(_session: Any, pose: list[float]) -> dict[str, Any]:
        asked.append(list(pose))
        return {**PLANNED, "path_joints": TWIN_ROWS if list(pose) == TWIN else [JOINTS]}

    def pace(_session: Any, planned: dict[str, Any], _speed: Any) -> PacedPath:
        if planned["path_joints"] != TWIN_ROWS:
            raise PaceRefusal("the wrist unwinds", code="paced_path_wrist_unwind",
                              constraints={"unrequested_turn_deg": 160.0})
        return PacedPath(grid=None, knots={"joints": {"arm": TWIN_ROWS}, "ticks": 3, "duration_s": 0.2}, fraction=1.0,
                         pace=None, slow_pace=None, speed={"speed": "fast"}, wrist={"path_turn_deg": 20.0})

    monkeypatch.setattr(shared_motion, "plan", plan)
    monkeypatch.setattr(shared_motion, "pace", pace)
    monkeypatch.setattr(shared_motion, "forward_tcp", lambda _session, rows: [list(TWIN) for _ in rows])
    return asked


def test_a_wrist_refusal_with_open_jaws_runs_the_jaw_twin_and_the_servo_settles_on_it(monkeypatch):
    asked = _twin_stubs(monkeypatch)
    bridge = TwinBridge()
    receipt = motion.execute_motion_v3(Session(bridge), _action(FAR))
    assert asked == [FAR, TWIN] and bridge.rows == TWIN_ROWS
    assert receipt["execution_status"] == "reached" and receipt["motion_strategy"] == "planned_joint_path"
    retry = receipt["motion"]["wrist"]["jaw_flip_retry"]
    assert retry["applied"] is True and retry["executed_orientation_xyzw"] == TWIN[3:]
    assert retry["requested_orientation_xyzw"] == FAR[3:] and retry["refused_unrequested_turn_deg"] == 160.0
    assert receipt["arms"]["right"]["pose_error"]["orientation_rad"] == pytest.approx(0.0, abs=1e-9)
    assert "jaw_flip_retry" in json.dumps(policy_receipt(receipt))  # the policy reads it with the wrist report


def test_held_jaws_and_a_recovery_keep_the_wrist_refusal(monkeypatch):
    asked = _twin_stubs(monkeypatch)
    session = Session(TwinBridge())
    session.finger_action = D.GRIPPER_CLOSE_ACTION
    refused = motion.execute_motion_v3(session, _action(FAR))
    assert (refused["failure_stage"], refused["failure_code"]) == ("pacing", "paced_path_wrist_unwind")
    assert "jaw_flip_retry" not in refused["motion"]["refused"] and asked == [FAR]
    session.finger_action, session.home_pose = D.GRIPPER_OPEN_ACTION, list(FAR)
    recovered = motion.execute_motion_v3(session, {"action_id": "action-2", "action": "recover_motion",
                                                   "arguments": {"targets": {"right": "home"}}})
    assert recovered["failure_code"] == "paced_path_wrist_unwind" and asked == [FAR, FAR]
    assert session.bridge.rows == []


@pytest.mark.parametrize("jaws", [D.GRIPPER_OPEN_ACTION, D.GRIPPER_CLOSE_ACTION])
def test_a_dry_run_answers_what_the_real_call_does(monkeypatch, jaws):
    asked = _twin_stubs(monkeypatch)
    session = Session(TwinBridge())
    session.finger_action = jaws
    plan = action_plan.plan_action(session, _action(FAR))
    receipt = motion.execute_motion_v3(session, _action(FAR))
    if jaws == D.GRIPPER_OPEN_ACTION:
        path, settle = plan["segments"]
        assert plan["status"] == "planned" and settle["poses"][-1] == TWIN and asked == [FAR, TWIN] * 2
        assert path["jaw_flip_retry"] == receipt["motion"]["wrist"]["jaw_flip_retry"]
        assert receipt["execution_status"] == "reached"
    else:
        assert (plan["status"], plan["failure_code"]) == ("refused", "paced_path_wrist_unwind") and asked == [FAR] * 2
        assert (receipt["execution_status"], receipt["failure_code"]) == ("plan_rejected", "paced_path_wrist_unwind")
