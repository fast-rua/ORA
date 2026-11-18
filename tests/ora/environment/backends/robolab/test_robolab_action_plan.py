"""RoboLab's read-only answer to a dry run (``backends/robolab/action_plan.py``), on a fake of its session."""
from __future__ import annotations

from types import SimpleNamespace

from ora.environment.backends.robolab import action_plan as robolab, shared_motion
from ora.environment.controllers.kept_plans import KeptPlans
from ora.environment.controllers.motion_pace import PaceRefusal

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


def _robolab(monkeypatch, plan=None):
    bridge = SimpleNamespace(tcp_pose=lambda: list(START), arm_qpos=lambda: [0.0] * 7)
    session = SimpleNamespace(bridge=bridge, motion_planner=None, home_pose=START, last_safe_pose=None,
                              workspace_refusal=lambda pose: None)
    monkeypatch.setattr(shared_motion, "plan", lambda session, pose: plan)
    monkeypatch.setattr(shared_motion, "pace", lambda session, planned, speed: Paced([[0.1] * 7, [0.2] * 7]))
    monkeypatch.setattr(shared_motion, "forward_tcp", lambda session, rows: [list(GOAL) for _ in rows])
    return session


def test_robolab_answers_a_free_move_with_the_paced_path_and_its_servo_end(monkeypatch):
    session = _robolab(monkeypatch, {"status": "planned", "reason": None, "start_joints": [0.0] * 7, "path_joints": []})
    answer = robolab.plan_action(session, _move())
    assert answer["status"] == "planned" and answer["route"] == "global"
    (path, settle) = answer["segments"]
    assert path["planned"] is True and path["native_actions"] == 2 and path["poses"][0] == START
    assert settle["planned"] is False and settle["native_actions"] is None and "servo" in settle["label"]


def test_robolab_answers_a_contact_with_the_servo_s_line_and_no_cost(monkeypatch):
    answer = robolab.plan_action(_robolab(monkeypatch), _move(interaction="contact_approach"))
    (line,) = answer["segments"]
    assert answer["route"] == "servo" and line["planned"] is False and line["native_actions"] is None
    assert line["poses"] == [START, GOAL]


def test_robolab_refuses_as_its_real_call_refuses(monkeypatch):
    session = _robolab(monkeypatch, {"status": "refused", "reason": "goal_unreachable", "planning_diagnostics": {}})
    assert robolab.plan_action(session, _move())["failure_code"] == "goal_unreachable"

    def paced(*_args):
        raise PaceRefusal("wrist", code="wrist_unwind")

    session = _robolab(monkeypatch, {"status": "planned", "reason": None, "start_joints": [0.0] * 7, "path_joints": []})
    monkeypatch.setattr(shared_motion, "pace", paced)
    answer = robolab.plan_action(session, _move())
    assert (answer["failure_code"], answer["failure_stage"]) == ("wrist_unwind", "pacing")
    assert robolab.plan_action(session, {"action": "recover_motion", "arguments": {"targets": {"right": "previous"}}})[
        "failure_code"] == "recovery_state_unavailable"


def test_robolab_s_session_keeps_what_a_dry_run_planned(monkeypatch):
    """The kept planner stands where the real call asks: a plan made inside ``previewing`` is the one it gets."""
    asked = []

    class Client:
        def plan(self, request):
            asked.append(request)
            return {"answer": len(asked)}

    kept = KeptPlans(Client())
    with kept.previewing():
        first = kept.plan({"goal": 1})
    assert kept.plan({"goal": 1}) == first and len(asked) == 1
