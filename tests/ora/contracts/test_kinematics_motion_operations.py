"""The kinematics request's shared-motion operations: closed fields, public refusals."""
import pytest

from ora.contracts.kinematics import REQUEST_SCHEMA, KinematicsRequestDrift, MAXIMUM_PATH_ROWS, validate_request
from ora.contracts.refusal import EnvironmentRefusal

POSE = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
ROW = [0.0] * 6
SPEED = {"speed": "normal", "slow_approaching": False}


def _plan(**fields):
    return {"schema": REQUEST_SCHEMA, "operation": "plan_path", "arm": "left", "goal_pose_xyz_xyzw": POSE,
            "goal_joints": None, **fields}


def _pace(**fields):
    return {"schema": REQUEST_SCHEMA, "operation": "pace_path", "arm": "right", "joints_rad": [ROW, ROW],
            "speed": SPEED, "slow_approach": False, "budget_ticks": 40, "cut_to_budget": True,
            "unwind_check": True, **fields}


def test_a_plan_names_a_pose_or_joints_but_not_both():
    assert validate_request(_plan())["goal_joints"] is None
    assert validate_request(_plan(goal_pose_xyz_xyzw=None, goal_joints=ROW))["goal_joints"] == ROW
    for bad in (_plan(goal_joints=ROW), _plan(goal_pose_xyz_xyzw=None)):
        with pytest.raises(EnvironmentRefusal):
            validate_request(bad)
    with pytest.raises(KinematicsRequestDrift):
        validate_request({**_plan(), "extra": 1})


def test_a_pace_request_validates_its_speed_budget_and_rows():
    assert validate_request(_pace())["speed"] == SPEED
    with pytest.raises(EnvironmentRefusal):
        validate_request(_pace(speed={"speed": "warp", "slow_approaching": False}))
    with pytest.raises(EnvironmentRefusal):
        validate_request(_pace(budget_ticks=0))
    with pytest.raises(KinematicsRequestDrift):
        validate_request(_pace(unwind_check="yes"))
    with pytest.raises(EnvironmentRefusal):
        validate_request(_pace(joints_rad=[ROW] * (MAXIMUM_PATH_ROWS + 1)))


def test_a_contact_path_is_joint_rows():
    request = {"schema": REQUEST_SCHEMA, "operation": "contact_path", "arm": "left", "joints_rad": [ROW]}
    assert validate_request(request)["joints_rad"] == [ROW]
