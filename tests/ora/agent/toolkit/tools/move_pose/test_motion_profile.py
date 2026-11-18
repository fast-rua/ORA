"""The ARX X5 motion profile a tool context binds: fast and the jaw forms, and nothing else.

Profile v1 is the toolkit's built-in values, so a context without a profile (every run sealed before v2) builds the
requests it recorded; profile v2 is RoboDojo's default: a faster fast with its own certification jerk and shorter jaw
commands. normal, careful, speed_m_s and the recovery retrace never follow the profile.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from ora.agent.toolkit.host.motion import profile_entry
from ora.agent.toolkit.lib.motion_plan import kept, recovery, timing
from ora.agent.toolkit.tools.move_pose import _profile, _smooth
from ora.agent.toolkit.tools.recover_pose import _trajectory
from ora.agent.toolkit.tools.set_gripper.tool import CLOSE, OPEN, RELEASE, rate_limited_targets
from ora.contracts.joint_trajectory import validate_gripper_target
from ora.contracts.kinematics import validate_request
from ora.orchestrator.paths import PROJECT_ROOT

V1 = "ora/policy/embodiments/arx-x5/motion-profile-v1.json"
V2 = "ora/policy/embodiments/arx-x5/motion-profile-v2.json"


def _record(path):
    import hashlib

    return {"path": path, "sha256": hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()}


def _config(path):
    return {"motion": {"profile": profile_entry(_record(path), root=PROJECT_ROOT)}}


def test_profile_v1_states_the_built_in_values():
    entry = _profile.bound(_config(V1))
    pace, settle = _profile.fast({})
    assert entry["fast"]["pace"] == pace == _smooth.FULL and entry["fast"]["settle"] == settle == _smooth.FULL_SETTLE
    for form, legacy in (("close", CLOSE), ("release", RELEASE), ("open", OPEN)):
        assert entry["gripper"][form] == legacy == _profile.gripper_form({}, form)


def test_profile_v2_fast_is_faster_within_the_x5_joint_limit():
    pace, settle = _profile.fast(_config(V2))
    assert (pace.joint_speed_rad_s, pace.joint_accel_rad_s2, pace.joint_jerk_rad_s3) == (4.5, 60.0, 5000.0)
    assert pace.joint_speed_rad_s < timing.JOINT_SPEED_LIMIT_RAD_S and settle == _smooth.FULL_SETTLE
    assert pace.linear_m_s is None and pace.angular_deg_s is None


def test_only_fast_follows_the_profile():
    speeds = _smooth.speeds(_config(V2))
    assert speeds["fast"][0].joint_speed_rad_s == 4.5
    assert speeds["normal"] == _smooth.SPEEDS["normal"] and speeds["careful"] == _smooth.SPEEDS["careful"]
    assert _smooth.speeds({}) == _smooth.SPEEDS
    # speed_m_s keeps profile v1's joint caps (its own definition), whatever the context binds.
    assert _smooth.custom_pace(0.2).joint_speed_rad_s == 4.0 and _smooth.custom_pace(0.2).joint_jerk_rad_s3 is None
    paces = _trajectory.paces(_config(V2))
    assert paces[recovery.JOINT].joint_speed_rad_s == 4.5 and paces[recovery.RETRACE] == _smooth.NORMAL
    assert _trajectory.paces({}) == _trajectory.PACES


def test_a_pace_without_a_jerk_requests_and_keys_what_it_always_did():
    caps = _smooth.FULL.caps()
    assert list(caps) == list(timing.CAPS) and "joint_jerk_rad_s3" not in _smooth.FULL.record()
    legacy = {field.name: getattr(_smooth.FULL, field.name) for field in dataclasses.fields(_smooth.FULL)
              if field.name != "joint_jerk_rad_s3"}
    assert kept._plain(_smooth.FULL) == legacy  # the kept-plan key of every earlier plan
    fast = _profile.fast(_config(V2))[0]
    assert fast.caps()["joint_jerk_rad_s3"] == 5000.0 and timing.Pace(**fast.record()) == fast


def test_the_v2_pace_is_a_valid_timing_request():
    fast = _profile.fast(_config(V2))[0]
    request = {"schema": "ora.kinematics-request.v1", "operation": "pace_path", "arm": "left",
               "joints_rad": [[0.0] * 6, [0.1] * 6],
               "speed": {"speed": "explicit", "caps": fast.caps(), "slow_caps": None, "slow_approaching": False},
               "slow_approach": False, "budget_ticks": 50, "cut_to_budget": True, "unwind_check": True}
    assert validate_request(request)["speed"]["caps"]["joint_jerk_rad_s3"] == 5000.0


def test_the_jaw_forms_follow_the_profile_and_validate():
    effectors = {"left": {"gripper_command_target": "close"}, "right": {"gripper_command_target": "open"}}
    v1 = rate_limited_targets({"left": "open", "right": "close"}, effectors)
    assert v1 == {"left": {"opening": "open", **RELEASE}, "right": {"opening": "close", **CLOSE}}
    v2 = rate_limited_targets({"left": "open", "right": "close"}, effectors, _config(V2))
    close, release = v2["right"], v2["left"]
    assert close["rate_per_s"] is None and close["max_lead_m"] is None and close["hold_lead_m"] == 0.005
    assert close["arm_settle"]["first_reading"] == "joint_velocity"
    assert close["end"]["stall_window_native_actions"] == 2 and release["rate_per_s"] is None
    for arm, target in v2.items():
        normalized = validate_gripper_target(target, arm)
        assert normalized["end"]["stall_window_native_actions"] == 2
    assert validate_gripper_target(close, "right")["hold_lead_m"] == 0.005
    # A v1 form normalizes exactly as before: no hold lead, no first reading.
    normalized = validate_gripper_target(v1["right"], "right")
    assert "hold_lead_m" not in normalized and "first_reading" not in normalized["arm_settle"]


def test_a_profile_must_be_the_bytes_the_plan_hashed():
    record = {**_record(V2), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="changed after the plan hashed it"):
        profile_entry(record, root=PROJECT_ROOT)
    assert profile_entry(None, root=PROJECT_ROOT) is None
    broken = json.loads((PROJECT_ROOT / V2).read_text())
    broken["fast"]["pace"]["joint_speed_rad_s"] = 5.5  # beyond the X5's 5 rad/s
    with pytest.raises(_profile.ProfileError):
        _profile.context_entry(broken, path=V2, sha256="0" * 64)


def test_the_robodojo_baselines_bind_profile_v2():
    from ora.agent.composition.baseline import BASELINE_CATALOGS, TOOLKIT_BASELINE_IDS, load_native_baseline

    for baseline in sorted(TOOLKIT_BASELINE_IDS):
        assert baseline in BASELINE_CATALOGS
        runtime = load_native_baseline(PROJECT_ROOT, baseline)["instances"]["robodojo"]["runtime"]
        assert runtime["motion_profile"] == _record(V2)
    assert Path(PROJECT_ROOT / V1).is_file()
