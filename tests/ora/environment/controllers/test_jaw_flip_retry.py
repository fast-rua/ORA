"""The jaw twin a wrist refusal is retried to, and when the retry runs (``controllers/jaw_flip_retry.py``)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.environment.backends.robocasa import kinematics
from ora.environment.controllers import jaw_flip_retry
from ora.environment.controllers.motion_pace import PaceRefusal

POSE = [0.4, -0.1, 0.3, *Rotation.from_euler("xyz", [170.0, 20.0, -35.0], degrees=True).as_quat().tolist()]


def _same_orientation(first, second) -> bool:
    return abs(abs(float(np.dot(first, second))) - 1.0) < 1e-9


def _wrist_refusal() -> PaceRefusal:
    return PaceRefusal("the wrist unwinds", code="paced_path_wrist_unwind",
                       constraints={"path_turn_deg": 300.0, "requested_turn_deg": 100.0, "unrequested_turn_deg": 200.0})


def test_the_twin_is_the_pose_turned_a_half_turn_about_its_approach_axis():
    twin = jaw_flip_retry.jaw_twin(POSE)
    assert twin[:3] == POSE[:3] and twin[6] >= 0.0
    turn = Rotation.from_quat(POSE[3:]).inv() * Rotation.from_quat(twin[3:])
    # A half-turn about the hand's own approach axis (local +z): the approach stays, the jaw line reverses.
    assert math.degrees(turn.magnitude()) == pytest.approx(180.0)
    assert abs(turn.as_rotvec()[2]) / np.linalg.norm(turn.as_rotvec()) == pytest.approx(1.0)
    # LIBERO's probe variant and RoboCasa's jaw equivalent name the same orientation.
    x, y, z, w = POSE[3:]
    assert _same_orientation(twin[3:], [y, -x, w, -z])
    assert _same_orientation(twin[3:], kinematics.jaw_equivalent(POSE)[3:])
    assert _same_orientation(jaw_flip_retry.jaw_twin(twin)[3:], POSE[3:])
    assert jaw_flip_retry.jaw_twin([0, 0, 0, 0, 0, 0, 1]) == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_only_a_wrist_refusal_with_open_jaws_is_retried():
    asked = []

    def admit(goal):
        asked.append(goal)
        return "planned"

    budget = PaceRefusal("too long", code="paced_motion_budget")
    assert jaw_flip_retry.retry(POSE, budget, admit, jaws_open=lambda: True) is None
    # A jaw that may hold something keeps the request: the load would turn with the hand.
    refusal = _wrist_refusal()
    assert jaw_flip_retry.retry(POSE, refusal, admit, jaws_open=lambda: False) is None
    assert asked == [] and "jaw_flip_retry" not in refusal.constraints


def test_an_admitted_twin_runs_and_says_so():
    twin, admitted, record = jaw_flip_retry.retry(POSE, _wrist_refusal(), lambda goal: {"goal": goal},
                                                  jaws_open=lambda: True)
    assert twin == jaw_flip_retry.jaw_twin(POSE) and admitted == {"goal": twin}
    assert record == {"applied": True, "reason": "paced_path_wrist_unwind", "turn_deg": 180.0,
                      "refused_unrequested_turn_deg": 200.0, "requested_orientation_xyzw": POSE[3:],
                      "executed_orientation_xyzw": twin[3:]}


@pytest.mark.parametrize("answer", ["path_blocked", PaceRefusal("again", code="paced_path_wrist_unwind")])
def test_a_refused_twin_leaves_the_original_refusal_saying_it_was_tried(answer):
    def admit(_goal):
        if isinstance(answer, Exception):
            raise answer
        return answer

    refusal = _wrist_refusal()
    assert jaw_flip_retry.retry(POSE, refusal, admit, jaws_open=lambda: True) is None
    assert refusal.code == "paced_path_wrist_unwind" and refusal.constraints["unrequested_turn_deg"] == 200.0
    assert refusal.constraints["jaw_flip_retry"] == {
        "applied": False, "reason": "paced_path_wrist_unwind", "turn_deg": 180.0, "refused_unrequested_turn_deg": 200.0,
        "twin_refused": "path_blocked" if isinstance(answer, str) else "paced_path_wrist_unwind"}
