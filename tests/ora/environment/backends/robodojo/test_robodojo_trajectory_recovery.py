"""The previous recovery anchor around joint trajectories, on the fake plant.

A trajectory moves the anchor to its start and records the joints measured at
the end of every native action it ran (``last_safe_paths``); a trajectory
marked ``recovery`` keeps both; an anchor with no path of its own (as the
SDK executor's move_pose left one) drops the path; the contract and the host
check the restated anchors.
"""
from __future__ import annotations

import pytest

from ora.contracts.action import ActionValidationError, validate_runtime_action
from ora.contracts.execution_state import FIELDS, pack, recovery_goals, validate
from tests.support.robodojo.backend import v3

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")

HOME = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]


def _knots(start, step=0.04, count=5, joint=0):
    rows, current = [], list(start)
    for _ in range(count):
        current = list(current)
        current[joint] += step
        rows.append(current)
    return rows


def _trajectory(session, tmp_path, knots, **arguments):
    action = v3(session, tmp_path, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": knots}}, **arguments}
    return action


def _recovery(session, strategy="backtrack_last_safe"):
    joints, poses = ((session.last_safe_joints, session.last_safe_poses) if strategy == "backtrack_last_safe"
                     else (session.home_joints, session.home_poses))
    return {"targets": {"left": strategy},
            "recovery_targets": {"left": {"strategy": strategy, "arm_joint_qpos": list(joints["left"]),
                                          "pose_xyz_xyzw": list(poses["left"])}}}


def test_a_trajectory_records_its_measured_path_with_the_anchor(sparse, tmp_path):
    sparse.environment.track_gain = 0.5
    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(start, count=6), guards={"progress": None}))
    path = sparse.last_safe_paths["left"]
    assert sparse.last_safe_joints["left"] == pytest.approx(start) and path[0] == pytest.approx(start)
    # One row per native action the trajectory ran, each the joints measured at its end (lagging the knots).
    assert len(path) == result["receipt"]["native_actions"] + 1
    measured_end = sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"]
    assert path[-1] == pytest.approx(measured_end) and path[-1][0] < start[0] + 6 * 0.04
    assert "recovery" not in result["receipt"]["trajectory"]
    packed = pack({name: getattr(sparse, name) for name in FIELDS})
    assert packed["exec.recovery.left.previous"]["measured_path_rad"] == path
    validate({"schema": "ora.execution-state.v1", "values": packed})


def test_a_recovery_trajectory_keeps_the_anchor_and_its_path(sparse, tmp_path):
    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    sparse.execute(_trajectory(sparse, tmp_path, _knots(start)))
    anchor, path = list(sparse.last_safe_joints["left"]), [list(row) for row in sparse.last_safe_paths["left"]]
    here = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    back = [list(row) for row in reversed(path[:-1])]
    result = sparse.execute(_trajectory(sparse, tmp_path, back, recovery=_recovery(sparse)))
    receipt = result["receipt"]
    assert receipt["trajectory"]["recovery"] == {"targets": {"left": "backtrack_last_safe"},
                                                 "previous_anchor": "kept"}
    assert sparse.last_safe_joints["left"] == anchor and sparse.last_safe_paths["left"] == path
    assert here != anchor  # the recovery moved the arm, yet the anchor is the original move's start


def test_a_later_move_replaces_the_anchor_after_a_recovery(sparse, tmp_path):
    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    sparse.execute(_trajectory(sparse, tmp_path, _knots(start)))
    sparse.execute(_trajectory(sparse, tmp_path, [start], recovery=_recovery(sparse)))
    here = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    sparse.execute(_trajectory(sparse, tmp_path, _knots(here, joint=1)))
    assert sparse.last_safe_joints["left"] == pytest.approx(here)
    assert sparse.last_safe_paths["left"][0] == pytest.approx(here)


def test_an_anchor_with_no_path_of_its_own_drops_the_old_path(sparse, tmp_path):
    from types import SimpleNamespace

    from ora.environment.backends.robodojo.execution_request import ExecutionRequest

    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    sparse.execute(_trajectory(sparse, tmp_path, _knots(start)))
    # A new anchor with no path recorded (the SDK executor's move_pose set one so); the request's exit drops the old.
    sparse.last_safe_joints["left"] = [value + 0.02 for value in sparse.last_safe_joints["left"]]
    assert sparse.last_safe_paths["left"][0] != sparse.last_safe_joints["left"]
    values = pack({name: getattr(sparse, name) for name in FIELDS})
    right = values["exec.recovery.right.previous"]
    right["measured_path_rad"] = [list(right["arm_joint_qpos"]), list(right["arm_joint_qpos"])]
    request = ExecutionRequest(SimpleNamespace(started=False), {"schema": "ora.execution-state.v1",
                                                                "values": values})
    with request:
        pass
    updated = request.updates["values"]
    assert "measured_path_rad" not in updated["exec.recovery.left.previous"]
    assert updated["exec.recovery.right.previous"]["measured_path_rad"] == right["measured_path_rad"]


def test_the_contract_validates_the_restated_anchors(sparse, tmp_path):
    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    action = _trajectory(sparse, tmp_path, [start], recovery=_recovery(sparse))
    assert validate_runtime_action(action)["arguments"]["recovery"]["targets"] == {"left": "backtrack_last_safe"}
    plain = validate_runtime_action(_trajectory(sparse, tmp_path, [start]))
    assert "recovery" not in plain["arguments"]  # unchanged for every other trajectory
    for broken in ({"targets": {"left": "previous"}, "recovery_targets": _recovery(sparse)["recovery_targets"]},
                   {"targets": {"right": "home"}, "recovery_targets": {}},
                   {"targets": {"left": "home"}, "recovery_targets": _recovery(sparse)["recovery_targets"]}):
        with pytest.raises(ActionValidationError):
            validate_runtime_action(_trajectory(sparse, tmp_path, [start], recovery=broken))


def test_the_host_admits_only_the_current_anchor_and_never_restates_the_path(sparse, tmp_path):
    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    sparse.execute(_trajectory(sparse, tmp_path, _knots(start)))
    values = pack({name: getattr(sparse, name) for name in FIELDS})
    action = _trajectory(sparse, tmp_path, [start], recovery=_recovery(sparse))
    goals = recovery_goals(action, values)
    assert set(goals["left"]) == {"strategy", "arm_joint_qpos", "pose_xyz_xyzw"}
    stale = _recovery(sparse)
    stale["recovery_targets"]["left"]["arm_joint_qpos"][0] += 0.1
    with pytest.raises(ValueError, match="stale_or_mismatched_recovery_target"):
        recovery_goals(_trajectory(sparse, tmp_path, [start], recovery=stale), values)
    # The released recover_motion restatement still matches although the anchor now carries a path.
    released = v3(sparse, tmp_path, kind="recover_motion", targets={"left": "backtrack_last_safe"})
    released["arguments"]["recovery_targets"] = _recovery(sparse)["recovery_targets"]
    assert recovery_goals(released, values) == goals
    assert recovery_goals(_trajectory(sparse, tmp_path, [start]), values) == {}


def test_a_malformed_recorded_path_is_refused():
    values = {"exec.recovery.left.previous": {"arm_joint_qpos": HOME, "pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
                                              "measured_path_rad": [HOME, [0.0] * 5]}}
    with pytest.raises(ValueError, match="recorded recovery path"):
        validate({"schema": "ora.execution-state.v1", "values": values})
