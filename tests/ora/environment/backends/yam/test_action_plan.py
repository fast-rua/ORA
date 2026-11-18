"""The YAM station's read-only answer to a dry run (``backends/yam/action_plan.py``), on a fake of its session."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.yam import action_plan as yam, motion as yam_motion

IDENTITY = [0.0, 0.0, 0.0, 1.0]
START = [0.0, 0.0, 0.5, *IDENTITY]
GOAL = [0.3, 0.0, 0.5, *IDENTITY]


def _move(pose=GOAL, interaction="free_space", arm="right"):
    return {"action": "move_pose", "action_id": "action-1", "arguments": {
        "targets": {arm: {"position_m": pose[:3], "orientation_xyzw": pose[3:]}}, "interactions": {arm: interaction}}}


class Kinematics:
    def fk(self, arm, q):
        pose = np.eye(4)
        pose[0, 3] = float(np.asarray(q)[0])
        return pose


class Rig:
    def __init__(self):
        self.q = np.zeros(6)

    def arm_q(self, arm):
        return self.q.copy()


def _yam():
    return SimpleNamespace(kinematics=Kinematics(), rig=Rig(), kept_schedules={}, ran_previewed_plan=False, home_q={},
                           last_safe_q=None)


def _schedule(value=0.1):
    return yam_motion.Schedule([{"left": np.full(6, value)}, {"left": np.full(6, value * 2)}], {"left": np.full(6, value * 2)},
                               {"duration_s": 0.04, "min_inter_arm_distance_m": 0.3})


def test_yam_answers_with_its_schedule_and_the_real_call_follows_those_ticks(monkeypatch):
    session = _yam()
    monkeypatch.setattr(yam, "move_plans", lambda session, arguments: ({}, {}))
    monkeypatch.setattr(yam, "plan_motion", lambda session, plans: _schedule())
    action = _move(arm="left")
    answer = yam.plan_action(session, action)
    (path,) = answer["segments"]
    assert answer["route"] == "scheduled" and path["planned"] is True and path["native_actions"] == 2
    assert path["poses"][-1][0] == pytest.approx(0.2) and session.rig.q.tolist() == [0.0] * 6, "nothing moved"
    assert yam.scheduled(session, action, lambda: _schedule(0.5)).finals["left"][0] == pytest.approx(0.2)
    assert session.ran_previewed_plan is True


def test_yam_s_kept_schedule_does_not_outlive_a_motion(monkeypatch):
    session = _yam()
    monkeypatch.setattr(yam, "move_plans", lambda session, arguments: ({}, {}))
    monkeypatch.setattr(yam, "plan_motion", lambda session, plans: _schedule())
    action = _move(arm="left")
    yam.plan_action(session, action)
    session.rig.q[0] = 0.3
    assert yam.scheduled(session, action, lambda: _schedule(0.5)).finals["left"][0] == pytest.approx(1.0)
    assert session.ran_previewed_plan is False


def test_yam_refuses_as_its_real_call_refuses_and_has_no_base(monkeypatch):
    def refused(session, arguments):
        raise yam_motion.Refusal("admission", "target_out_of_reach", arm="left")

    monkeypatch.setattr(yam, "move_plans", refused)
    answer = yam.plan_action(_yam(), _move(arm="left"))
    assert (answer["failure_code"], answer["failure_stage"]) == ("target_out_of_reach", "admission")
    assert yam.plan_action(_yam(), {"action": "move_base", "arguments": {}})["failure_code"] == "mobile_base_unavailable"
