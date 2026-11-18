"""RoboCasa's read-only answer to a dry run (``backends/robocasa/action_plan.py``), on a fake of its session."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ora.environment.backends.robocasa import action_plan as robocasa
from ora.environment.controllers import robocasa_omron

IDENTITY = [0.0, 0.0, 0.0, 1.0]
START = [0.0, 0.0, 0.5, *IDENTITY]
GOAL = [0.3, 0.0, 0.5, *IDENTITY]


def _move(pose=GOAL, interaction="free_space", arm="right"):
    return {"action": "move_pose", "action_id": "action-1", "arguments": {
        "targets": {arm: {"position_m": pose[:3], "orientation_xyzw": pose[3:]}}, "interactions": {arm: interaction}}}


class Paced:
    def __init__(self, rows):
        self.rows_ = rows

    def rows(self):
        return self.rows_

    def report(self):
        return {"planned_ticks": len(self.rows_), "duration_s": 0.1 * len(self.rows_)}


class Io:
    model, data = None, None
    gripper_directive = -1.0

    def grip_pose(self):
        return list(START)

    def arm_qpos(self):
        return [0.0] * 7

    def base_pose(self):
        return (0.0, 0.0, 0.0)


def _robocasa(monkeypatch, plan=None, *, bounds=None, blocked=None):
    asked = []
    session = SimpleNamespace(io=Io(), motion_planner=None, workspace_refusal=lambda pose: bounds,
                              reach_detail=lambda pose, code, interaction, speed=None: {
                                  "path_kind": "reach", "distance_from_mount_m": 1.2},
                              arm_path_plan=lambda pose, interaction, speed=None: asked.append(speed) or plan,
                              base_path_refusal=lambda target: blocked, home_arm_qpos=None, last_safe_arm_qpos=None,
                              asked=asked)
    monkeypatch.setattr(robocasa, "_paced_plan", lambda *args: Paced([[0.1] * 7]))
    monkeypatch.setattr(robocasa.kinematics, "grip_poses", lambda model, data, rows: [list(GOAL) for _ in rows])
    return session


def test_robocasa_answers_with_the_paced_path(monkeypatch):
    session = _robocasa(monkeypatch, {"refusal": None, "code": None, "kind": "joint", "torso_height_m": None})
    answer = robocasa.plan_action(session, _move())
    assert answer["route"] == "global" and answer["segments"][0]["native_actions"] == 1


def test_robocasa_shows_the_path_its_admission_planned_from_the_lifted_height(monkeypatch):
    session = _robocasa(monkeypatch, {"refusal": None, "code": None, "kind": "joint", "torso_height_m": 0.4,
                                      "lift_m": 0.1, "lift_paced": (Paced([[0.1] * 7, [0.2] * 7]), None)})
    answer = robocasa.plan_action(session, _move())
    torso, arm = answer["segments"]
    assert (answer["route"], torso["kind"], torso["planned"], torso["height_m"]) == ("lift_then_arm", "torso", False, 0.4)
    assert arm["planned"] is True and arm["native_actions"] == 2
    assert arm["poses"][0][2] == pytest.approx(START[2] + 0.1) and arm["poses"][-1][2] == pytest.approx(GOAL[2] + 0.1)


def test_robocasa_passes_the_call_s_speed_to_the_admission(monkeypatch):
    session = _robocasa(monkeypatch, {"refusal": None, "code": None, "kind": "joint", "torso_height_m": None})
    action = _move()
    action["arguments"]["speed"] = {"preset": "careful"}
    robocasa.plan_action(session, action)
    assert session.asked == [{"preset": "careful"}]


def test_robocasa_s_refusal_carries_what_the_receipt_does(monkeypatch):
    refusal = {"path_kind": "line", "cause": "ik_unsolved", "base_suggestions": [{"x_m": 1.0}]}
    session = _robocasa(monkeypatch, {"refusal": refusal, "code": "target_unreachable_at_base_pose", "kind": "line",
                                      "torso_height_m": None})
    answer = robocasa.plan_action(session, _move())
    assert (answer["status"], answer["failure_stage"]) == ("refused", "plan_validation")
    assert answer["constraints"]["base_suggestions"] == [{"x_m": 1.0}] and answer["constraints"]["cause"] == "ik_unsolved"
    far = robocasa.plan_action(_robocasa(monkeypatch, bounds="target_out_of_reach_from_base"), _move())
    assert (far["failure_code"], far["failure_stage"]) == ("target_out_of_reach_from_base", "admission")
    assert far["constraints"] == {"path_kind": "reach", "distance_from_mount_m": 1.2}


def test_robocasa_s_base_is_a_servo_with_no_path_whose_footprint_and_bound_are_checked(monkeypatch):
    session = _robocasa(monkeypatch)
    near = {"action": "move_base", "arguments": {"target": {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.5}}}
    answer = robocasa.plan_action(session, near)
    (line,) = answer["segments"]
    assert line["kind"] == "base" and line["planned"] is False and line["native_actions"] is None and "no path" in line["label"]
    assert line["footprint"] == robocasa.FOOTPRINT_CLEAR
    far = {"action": "move_base", "arguments": {"target": {"x_m": robocasa_omron.BASE_MAX_TRAVEL_M + 1, "y_m": 0.0,
                                                           "yaw_rad": 0.0}}}
    assert robocasa.plan_action(session, far)["failure_code"] == "target_too_far"
    detail = {"path_kind": "base", "robot_part": "base", "path_fraction": 0.4, "obstacle_height_m": 0.2}
    blocked = robocasa.plan_action(_robocasa(monkeypatch, blocked=detail), near)
    assert (blocked["status"], blocked["failure_code"], blocked["failure_stage"]) == (
        "refused", "base_path_in_collision", "admission")
    assert blocked["constraints"] == detail


def test_nothing_that_moves_nothing_is_answered_for_as_a_plan(monkeypatch):
    assert robocasa.plan_action(_robocasa(monkeypatch), {"action": "wait", "arguments": {}})["status"] == "unavailable"


def test_a_refused_lift_s_record_stays_in_the_receipt(monkeypatch):
    """The lift's admission record (``torso_admission``) is evidence: a dry run's refusal leaves it out, as the real
    call's public projection does."""
    refusal = {"path_kind": "line", "cause": "joint_branch_jump", "torso_admission": {"tried": []}}
    session = _robocasa(monkeypatch, {"refusal": refusal, "code": "target_unreachable_at_base_pose", "kind": "line",
                                      "torso_height_m": None})
    answer = robocasa.plan_action(session, _move(interaction="precision"))
    assert answer["failure_code"] == "target_unreachable_at_base_pose"
    assert "torso_admission" not in json.dumps(answer) and "joint_branch_jump" in json.dumps(answer)
