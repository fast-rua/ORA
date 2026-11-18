"""Host-safe contracts of the RoboCasa365 executor (no simulator import).

The RoboLab executor's twin contracts are in
``tests/ora/environment/backends/robolab/test_robolab_host_safe.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ora.agent.session_core.receipt_projection import public_motion_failure_code
from ora.environment.benchmarks import robocasa as robocasa_benchmark
from ora.environment.backends.robocasa import motion as robocasa_motion
from ora.environment.controllers import robocasa_omron as C


@pytest.mark.parametrize("stop_reason", ["episode_ended", "no_progress", "budget_exhausted"])
def test_arm_stops_use_public_codes_the_abort_gate_does_not_count(stop_reason):
    status, stage, code = robocasa_motion._arm_status(stop_reason)
    assert (status, stage) == ("not_reached", "execution")
    assert public_motion_failure_code("execution", code) != "execution_fault", code


def test_arrival_and_contact_stops_are_reached():
    assert robocasa_motion._arm_status("settled")[0] == "reached"
    assert robocasa_motion._arm_status("contact_detected")[0] == "reached"


def test_robocasa_joint_path_stops_use_public_codes():
    # Settled joints with the grip site still off target is an endpoint failure,
    # and an arm pressed against something is blocked; neither is an execution fault.
    status, stage, code = robocasa_motion._arm_status("settled", endpoint_ok=False)
    assert (status, public_motion_failure_code(stage, code)) == ("not_reached", "endpoint_residual_exceeded")
    status, stage, code = robocasa_motion._arm_status("joint_lag")
    assert (status, public_motion_failure_code(stage, code)) == ("not_reached", "blocked")
    # Joints held short of the last target, at rest or past the wait: arrived when the grip
    # site is on target, else blocked.
    for held in ("settle_stalled", "settle_timeout"):
        assert robocasa_motion._arm_status(held, endpoint_ok=True)[0] == "reached"
        status, stage, code = robocasa_motion._arm_status(held, endpoint_ok=False)
        assert (status, public_motion_failure_code(stage, code)) == ("not_reached", "blocked")


def test_robocasa_refusal_detail_crosses_only_public_fields():
    from ora.agent.session_core.receipt_projection import ROBOCASA_SPARSE_MOTION_RECEIPT_SCHEMA as SCHEMA
    from ora.agent.session_core.refusal_detail import public_refusal_detail

    record = {"schema": SCHEMA, "reach_check": {
        "path_kind": "joint", "path_fraction": 0.3127, "penetration_m": 0.034347, "robot_part": "arm",
        "scope": "worker-only", "ik_residual": {"translation_m": 0.016, "orientation_rad": 0.1}}}
    detail = public_refusal_detail(record, SCHEMA)["admission_refusal"]
    assert detail == {"path_kind": "joint", "robot_part": "arm", "path_fraction": 0.313, "penetration_m": 0.0343,
                      "ik_residual": {"translation_m": 0.016, "orientation_deg": 5.73}}
    assert public_refusal_detail({**record, "schema": "ora.libero-sparse-motion-receipt.v1"}, SCHEMA) == {}
    assert public_refusal_detail({"schema": SCHEMA}, SCHEMA) == {}


def test_robocasa_refusal_detail_says_why_a_line_failed_where_to_stand_and_what_the_lift_did():
    from ora.agent.session_core.receipt_projection import ROBOCASA_SPARSE_MOTION_RECEIPT_SCHEMA as SCHEMA
    from ora.agent.session_core.refusal_detail import public_refusal_detail

    suggestion = {"x_m": 2.30512, "y_m": -1.1, "yaw_deg": 12.04, "torso_height_m": 0.12, "distance_m": 0.21,
                  "turn_deg": 12.0, "arm_path": "planned", "cost": 0.3}
    record = {"schema": SCHEMA, "torso": {"from_m": 0.31261, "to_m": 0.28771, "target_m": 0.28,
                                          "stop_reason": "reached", "arm_path_replanned": False, "private": 1},
              "reach_check": {
        "path_kind": "line", "path_fraction": 0.424, "cause": "joint_branch_jump", "joint": 4, "jump_rad": 0.41234,
        "bound_rad": 0.25, "at_pose": [2.412345, -1.1, 1.05, 0.0, 0.0, 0.0, 1.0], "endpoint": "reachable",
        "planner_reason": "worker-only", "base_suggestions": [suggestion],
        "base_search": {"stands": 240, "asked": 39, "ik_reached": 18, "checked_in_world": 2, "max_travel_m": 1.5,
                        "private": "x"}}}
    projected = public_refusal_detail(record, SCHEMA)
    detail = projected["admission_refusal"]
    assert {key: detail[key] for key in ("cause", "joint", "jump_rad", "bound_rad", "endpoint")} == {
        "cause": "joint_branch_jump", "joint": 4, "jump_rad": 0.4123, "bound_rad": 0.25, "endpoint": "reachable"}
    assert detail["at_pose"] == [2.4123, -1.1, 1.05, 0.0, 0.0, 0.0, 1.0] and "planner_reason" not in detail
    assert detail["base_suggestions"] == [{"x_m": 2.305, "y_m": -1.1, "yaw_deg": 12.0, "torso_height_m": 0.12,
                                           "distance_m": 0.21, "turn_deg": 12.0, "arm_path": "planned"}]
    assert detail["base_search"] == {"stands": 240, "asked": 39, "ik_reached": 18, "checked_in_world": 2,
                                     "max_travel_m": 1.5}
    assert projected["torso"] == {"from_m": 0.3126, "to_m": 0.2877, "target_m": 0.28, "stop_reason": "reached",
                                  "arm_path_replanned": False}
    carried = public_refusal_detail({"schema": SCHEMA, "reach_check": {
        "path_kind": "base", "robot_part": "arm", "path_fraction": 0.375, "penetration_m": 0.0121,
        "obstacle_height_m": 1.309, "obstacle_xy_m": [0.68, -1.543],
        "base_search": {"asked": 20, "ik_reached": 9, "arm_blocked": 4, "floor_unseen": 3, "arrival_unreached": 1}}},
        SCHEMA)["admission_refusal"]
    assert carried["robot_part"] == "arm" and carried["obstacle_xy_m"] == [0.68, -1.543]
    assert carried["base_search"] == {"asked": 20, "ik_reached": 9, "arm_blocked": 4}, (
        "the carried arm's count crosses; the other screens stay in the receipt")
    reach = public_refusal_detail({"schema": SCHEMA, "reach_check": {
        "path_kind": "reach", "distance_from_mount_m": 1.18912, "reach_m": 1.05, "nearest_mount_height_m": 1.04}},
        SCHEMA)["admission_refusal"]
    assert reach == {"path_kind": "reach", "distance_from_mount_m": 1.1891, "reach_m": 1.05,
                     "nearest_mount_height_m": 1.04}
    assert public_refusal_detail({"schema": SCHEMA, "torso": record["torso"]}, SCHEMA) == {"torso": projected["torso"]}


def test_robocasa_action_layout_matches_the_joint_controller():
    assert C.ARM_SLICE[1] - C.ARM_SLICE[0] == len(C.ARM_JOINTS)
    indices = [*range(*C.ARM_SLICE), C.GRIPPER_INDEX, *range(*C.BASE_SLICE), C.TORSO_INDEX, C.MODE_INDEX]
    assert sorted(indices) == list(range(C.ACTION_DIM))
    assert C.ARM_CONTROLLER["type"] == "JOINT_POSITION" and C.ARM_CONTROLLER["input_type"] == "delta"
    assert C.ARM_CONTROLLER["output_max"] == C.ARM_DELTA_LIMIT_RAD


def test_jaw_outcomes_map_to_the_shared_codes():
    motion = robocasa_motion
    assert motion.gripper_failure_code("reached_target", terminal=False) is None
    assert motion.gripper_failure_code("stalled_before_target", terminal=False) == "gripper_stalled_before_target"
    assert motion.gripper_failure_code("moving_at_budget", terminal=False) == "gripper_not_reached"
    assert motion.gripper_failure_code("interrupted", terminal=True) == "episode_terminal_during_execution"


def test_move_base_result_has_no_contact_force():
    import inspect

    source = inspect.getsource(robocasa_motion.MotionRun.move_base)
    assert "contact_force_n" not in source


def test_only_the_admitted_roster_validates():
    task = sorted(robocasa_benchmark.TASKS)[0]
    assert robocasa_benchmark.validate_runtime_extension({"task": task, "seed": 0})["task"] == task
    with pytest.raises(ValueError):
        robocasa_benchmark.validate_runtime_extension({"task": "NotATask", "seed": 0})
    with pytest.raises(ValueError):
        robocasa_benchmark.validate_runtime_extension({"task": task, "seed": -1})
    assert robocasa_benchmark.public_context()["environment_id"] == "robocasa"


# The NavigateKitchen evaluation horizon (450 control steps): the worker builds
# robosuite with ignore_done=True, so the session itself owns the boundary.
HORIZON_TASK = "NavigateKitchen"


class _KinematicOmronIO:
    """OmronIO stand-in for the base: it moves at the commanded world velocity."""

    def __init__(self, control_steps: int) -> None:
        self.control_steps = control_steps
        self.pose = [0.0, 0.0, 0.0]

    def base_pose(self) -> list[float]:
        return list(self.pose)

    def base_anchor(self) -> list[float]:
        return self.pose[:2]

    def anchor_offset(self) -> list[float]:
        return [0.0, 0.0]

    def slide_axes(self) -> list[list[float]]:
        yaw = self.pose[2]
        return [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]

    def contact_forces(self) -> dict[str, float]:
        return {"gripper": 0.0, "base": 0.0, "arm": 0.0}

    def grip_pose(self) -> list[float]:
        return [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def gripper_opening(self) -> float:
        return 1.0

    def arm_qpos(self) -> list[float]:
        return [0.0] * len(C.ARM_JOINTS)

    def torso_height(self) -> float:
        return 0.0

    def arm_base_pose(self) -> list[float]:
        return [*self.pose[:2], 0.7, 0.0, 0.0, 0.0, 1.0]

    gripper_directive = C.GRIPPER_OPEN_COMMAND

    def finger_qpos(self) -> list[float]:
        return [C.PANDA_FINGER_OPEN_M, -C.PANDA_FINGER_OPEN_M]

    def state_sha256(self) -> str:
        return "0" * 64

    def take_commands(self) -> list:
        return []

    def step(self, *, base_velocity_world=None, **_command):
        from ora.environment.backends.robocasa.robot import StepResult

        vx, vy, yaw_rate = base_velocity_world or (0.0, 0.0, 0.0)
        x, y, yaw = self.pose
        self.pose = [x + vx * C.CONTROL_PERIOD_S, y + vy * C.CONTROL_PERIOD_S, yaw + yaw_rate * C.CONTROL_PERIOD_S]
        self.control_steps += 1
        return StepResult(reward=0.0, official_success=False)


def _session_at(tmp_path, control_steps: int):
    from ora.environment.backends.robocasa.session import RoboCasaSession

    session = RoboCasaSession(config={"task": HORIZON_TASK, "seed": 0}, output_root=tmp_path)
    session.io = _KinematicOmronIO(control_steps)
    session.horizon = robocasa_benchmark.TASKS[HORIZON_TASK]["horizon_control_steps"]
    # The cameras see nothing within range, so the footprint check passes.
    session._camera = lambda _source: (None, np.zeros((4, 4), dtype=np.float32), {
        "intrinsic": np.eye(3), "cam2world": np.eye(4), "near": 0.01, "far": 10.0})
    return session


def test_move_base_takes_no_control_step_past_the_horizon(tmp_path):
    session = _session_at(tmp_path, control_steps=424)
    assert session.horizon == 450
    receipt = session.execute({
        "schema": "ora-motion-action.v3", "observation_id": "observation-0000", "action_id": "action-0000",
        "bound_observation_sha256": "0" * 64, "action": "move_base",
        # 3 m at the 0.6 m/s cap is far more than the 26 remaining ticks cover.
        "arguments": {"target": {"x_m": 3.0, "y_m": 0.0, "yaw_rad": 0.0}},
    })
    assert receipt["base"]["stop_reason"] == "episode_ended"
    assert receipt["failure_code"] == "episode_terminal_during_execution"
    assert receipt["native_actions"] <= 26
    assert receipt["native_actions"] == receipt["steps_used"] == 26
    assert receipt["control_steps_total"] == session.io.control_steps == 450
    assert receipt["terminal_after_action"] is True
    assert session.status()["reason"] == "robocasa_horizon_exhausted"


def test_a_base_move_into_what_the_depth_shows_moves_nothing_and_its_dry_run_says_the_same(tmp_path):
    from ora.environment.backends.robocasa.action_plan import plan_action

    session = _session_at(tmp_path, control_steps=10)
    # A camera 0.2 m up looking along +x at a wall 0.5 m ahead, in front of the base's footprint.
    looking = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]])
    session._camera = lambda _source: (None, np.full((4, 4), 0.5, dtype=np.float32), {
        "intrinsic": np.array([[10.0, 0.0, 1.5], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]), "cam2world": looking,
        "near": 0.01, "far": 10.0})
    action = {"schema": "ora-motion-action.v3", "observation_id": "observation-0000", "action_id": "action-0000",
              "bound_observation_sha256": "0" * 64, "action": "move_base",
              "arguments": {"target": {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0}}}
    preview = plan_action(session, action)
    receipt = session.execute(action)
    assert (receipt["execution_status"], receipt["failure_stage"], receipt["failure_code"]) == (
        "plan_rejected", "admission", "base_path_in_collision")
    assert receipt["native_actions"] == 0 and session.io.pose == [0.0, 0.0, 0.0]
    assert receipt["reach_check"]["robot_part"] == "base" and receipt["reach_check"]["obstacle_height_m"] > 0.1
    assert (preview["status"], preview["failure_code"]) == ("refused", "base_path_in_collision")
    assert preview["constraints"] == receipt["reach_check"]
    back = {**action, "action_id": "action-0001", "arguments": {"target": {"x_m": -0.3, "y_m": 0.0, "yaw_rad": 0.0}}}
    assert plan_action(session, back)["status"] == "planned"


@pytest.mark.parametrize("control_step, latched", [(450, True), (451, False)])
def test_success_is_latched_only_within_the_horizon(tmp_path, control_step, latched):
    from ora.environment.backends.robocasa.robot import StepResult

    session = _session_at(tmp_path, control_steps=control_step)
    session.after_control_step(StepResult(reward=1.0, official_success=True))
    assert session.truncated is True
    assert session.terminated is latched
    assert (session.first_success is not None) is latched
    assert session.persist_official_verdict()["official_success"] is latched


def _extended_session(tmp_path, *, control_steps: int):
    from ora.environment.backends.robocasa.session import RoboCasaSession

    session = RoboCasaSession(config={"task": HORIZON_TASK, "seed": 0, "official_limit_multiple": 3},
                              output_root=tmp_path)
    session.io = _KinematicOmronIO(control_steps)
    session.horizon = robocasa_benchmark.TASKS[HORIZON_TASK]["horizon_control_steps"]
    return session


def test_an_extended_episode_snapshots_the_horizon_and_runs_on(tmp_path):
    from ora.environment.backends.robocasa.robot import StepResult

    session = _extended_session(tmp_path, control_steps=450)
    session.after_control_step(StepResult(reward=0.0, official_success=False))
    assert session.official_limit_snapshot == {"official_success": False, "native_actions": 450}
    assert session.must_stop() is False
    assert session.status()["terminal"] is False
    session.io.control_steps = 1350
    session.after_control_step(StepResult(reward=0.0, official_success=False))
    assert session.status()["reason"] == "robocasa_horizon_exhausted"
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False
    assert verdict["horizon_control_steps"] == verdict["official_native_action_limit"] == 450
    assert verdict["extended_native_action_limit"] == 1350
    assert verdict["success_after_official_limit"] is False


def test_a_success_past_the_horizon_is_recorded_apart_from_the_official_one(tmp_path):
    from ora.environment.backends.robocasa.robot import StepResult

    session = _extended_session(tmp_path, control_steps=450)
    session.after_control_step(StepResult(reward=0.0, official_success=False))
    session.io.control_steps = 700
    session.after_control_step(StepResult(reward=1.0, official_success=True))
    assert session.must_stop() is True
    assert session.status()["reason"] == "robocasa_horizon_exhausted"
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminated"] is False
    assert verdict["first_success"] is None
    assert verdict["success_after_official_limit"] is True
    assert verdict["success_after_official_limit_native_actions"] == 700


def test_a_success_on_the_horizon_step_stays_official_under_an_extended_plan(tmp_path):
    from ora.environment.backends.robocasa.robot import StepResult

    session = _extended_session(tmp_path, control_steps=450)
    session.after_control_step(StepResult(reward=1.0, official_success=True))
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True
    assert verdict["official_limit_snapshot"] is None
    assert verdict["success_after_official_limit"] is False
