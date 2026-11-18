"""The end-effector speed request of a move_pose: policy arguments, wire form and the action envelope."""

from __future__ import annotations

import pytest

from ora.contracts import validate_runtime_action
from ora.contracts.action import ActionValidationError
from ora.contracts.motion_speed import (
    PRESET_CAPS, SpeedArgumentError, custom_caps, speed_caps, speed_request, validate_speed,
)


def _action(arguments, action="move_pose"):
    return {"schema": "ora-motion-action.v3", "observation_id": "observation-1", "action_id": "action-1",
            "action": action, "arguments": arguments, "bound_observation_sha256": "0" * 64}


def _move(**extra):
    return {"targets": {"right": {"position_m": [0.1, 0.0, 1.0], "orientation_xyzw": [0, 0, 0, 1]}},
            "interactions": {"right": "free_space"}, **extra}


def test_a_call_that_names_no_speed_carries_none():
    assert speed_request({"position_mode": "absolute"}) == (None, {"position_mode": "absolute"})
    assert speed_request({"speed": None, "speed_m_s": None, "slow_approaching": None, "x": 1}) == (None, {"x": 1})


@pytest.mark.parametrize("arguments,expected", [
    ({"speed": "normal"}, {"speed": "normal", "slow_approaching": False}),
    ({"speed": "fast", "slow_approaching": True}, {"speed": "fast", "slow_approaching": True}),
    ({"slow_approaching": True}, {"speed": "fast", "slow_approaching": True}),
    ({"speed_m_s": 0.2, "slow_approaching": True}, {"speed": "custom", "speed_m_s": 0.2, "slow_approaching": True}),
    ({"speed_m_s": 1, "speed": None}, {"speed": "custom", "speed_m_s": 1.0, "slow_approaching": False}),
])
def test_speed_options_compile_to_the_wire_form(arguments, expected):
    request, rest = speed_request({**arguments, "position_mode": "hold_current"})
    assert request == expected and rest == {"position_mode": "hold_current"}


@pytest.mark.parametrize("arguments,field,message", [
    ({"speed": "normal", "speed_m_s": 0.2}, "speed_m_s", "give speed or speed_m_s, not both"),
    ({"speed_m_s": 0}, "speed_m_s", "speed_m_s must be a number above 0 (m/s)"),
    ({"speed_m_s": True}, "speed_m_s", "speed_m_s must be a number above 0 (m/s)"),
    ({"speed_m_s": float("nan")}, "speed_m_s", "speed_m_s must be a number above 0 (m/s)"),
    ({"speed": "slow"}, "speed", "speed must be fast, normal or careful"),
    ({"slow_approaching": "yes"}, "slow_approaching", "slow_approaching must be true or false"),
    ({"speed": "careful", "slow_approaching": True}, "slow_approaching",
     "slow_approaching has no effect with speed=careful"),
    ({"speed_m_s": 0.1, "slow_approaching": True}, "slow_approaching",
     "slow_approaching has no effect with speed_m_s at or below 0.1 m/s"),
])
def test_refusals_name_the_field_in_the_toolkit_wording(arguments, field, message):
    with pytest.raises(SpeedArgumentError) as error:
        speed_request(arguments)
    assert error.value.field == field and str(error.value) == message


def test_custom_caps_follow_the_turn_rule_and_keep_fasts_joint_caps():
    assert custom_caps(0.05) == {"linear_m_s": 0.05, "angular_deg_s": 20.0, "linear_accel_m_s2": 0.2,
                                 "angular_accel_deg_s2": 80.0, "joint_speed_rad_s": 4.0, "joint_accel_rad_s2": 40.0}
    assert custom_caps(0.5)["angular_deg_s"] == 100.0
    caps, slow = speed_caps({"speed": "normal", "slow_approaching": True})
    assert caps == PRESET_CAPS["normal"] and slow == PRESET_CAPS["careful"]
    assert speed_caps(None) == (PRESET_CAPS["fast"], None)


@pytest.mark.parametrize("value", [
    {"speed": "custom", "slow_approaching": False},
    {"speed": "normal", "speed_m_s": 0.2, "slow_approaching": False},
    {"speed": "careful", "slow_approaching": True},
    {"speed": "custom", "speed_m_s": 0.05, "slow_approaching": True},
    {"speed": "fast"},
    "fast",
])
def test_the_wire_form_is_closed(value):
    with pytest.raises(ValueError):
        validate_speed(value)


def test_the_motion_action_carries_an_optional_speed_on_move_pose_only():
    plain = validate_runtime_action(_action(_move()))
    assert "speed" not in plain["arguments"]
    paced = validate_runtime_action(_action(_move(speed={"speed": "custom", "speed_m_s": 1, "slow_approaching": True})))
    assert paced["arguments"]["speed"] == {"speed": "custom", "speed_m_s": 1.0, "slow_approaching": True}
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action(_move(speed={"speed": "slow", "slow_approaching": False})))
    free = _move(speed={"speed": "normal", "slow_approaching": False})
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action(free, action="move_free_space"))


def test_explicit_caps_may_name_a_jerk_and_nothing_else():
    from ora.contracts.motion_speed import CAP_FIELDS, validate_speed

    caps = {key: None for key in CAP_FIELDS} | {"joint_speed_rad_s": 4.5, "joint_accel_rad_s2": 60.0}
    request = {"speed": "explicit", "caps": caps, "slow_caps": None, "slow_approaching": False}
    assert "joint_jerk_rad_s3" not in validate_speed(request)["caps"]
    jerked = validate_speed({**request, "caps": {**caps, "joint_jerk_rad_s3": 5000}})
    assert jerked["caps"]["joint_jerk_rad_s3"] == 5000.0
    for bad in ({**caps, "joint_jerk_rad_s3": 0}, {**caps, "joint_jerk_rad_s3": None}, {**caps, "jerk": 1.0}):
        with pytest.raises(ValueError):
            validate_speed({**request, "caps": bad})
