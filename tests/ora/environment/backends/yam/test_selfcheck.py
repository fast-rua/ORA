"""The YAM cell's start-up self-check lists every failed check with its fix (selfcheck.py)."""

from __future__ import annotations

from ora.environment.backends.yam import selfcheck

NOW = 1_800_000_000.0
LIMITS = {"left": [6.413, 1.111], "right": [5.886, 0.495]}


def _reading(*, raw: float, temperature: float = 25.0, errors=None):
    return {"gripper_raw_rad": raw, "max_temperature_c": temperature, "errors": errors or {}}


def _checks(**changes):
    args = dict(
        readings={"left": _reading(raw=1.119), "right": _reading(raw=0.501)},
        joints={"outside_limits": {}, "passed": True},
        wrist_table={"left": {"passed": True, "problems": []}, "right": {"passed": True, "problems": []}},
        calibration={"gripper_limits": LIMITS, "verified": {"at": NOW - 86400 * 2}},
        cameras={role: {"depth_valid_fraction": 0.8, "depth_below_floor": False}
                 for role in ("overhead", "front", "wrist_left", "wrist_right")},
        now=NOW)
    args.update(changes)
    return {check["id"]: check for check in selfcheck.startup_checks(**args)}


def test_a_healthy_cell_passes_every_check():
    checks = _checks()
    assert all(check["passed"] is True for check in checks.values()), checks
    assert selfcheck.refusal(list(checks.values())) is None


def test_the_evenings_problems_are_all_named_at_once_with_their_fixes():
    """2026-11-14: a moved gripper zero, a table 10 cm higher, a camera with no frames, a hot motor."""
    checks = _checks(
        readings={"left": _reading(raw=1.119), "right": _reading(raw=-5.10, temperature=55.0)},
        wrist_table={"left": {"passed": False, "problems": ["table at 0.009 m in the wrist view, -0.093 m calibrated"]},
                     "right": {"passed": True, "problems": []}},
        cameras={"wrist_right": {"error": "CellError: streams no frames"},
                 "front": {"depth_valid_fraction": 0.98, "depth_below_floor": False}})
    failed = {key for key, check in checks.items() if check["passed"] is False}
    assert failed == {"grippers_calibrated", "table_seen", "cameras_depth", "motors_cool"}
    message = selfcheck.refusal(list(checks.values()))
    assert "ora cell grippers" in message and "--redo table" in message and "wrist_right" in message
    assert message.count("\n- ") == 4


def test_unknowns_and_an_old_calibration_are_named_but_do_not_refuse():
    checks = _checks(wrist_table={"left": {"passed": None}, "right": {"passed": True}},
                     calibration={"gripper_limits": LIMITS, "verified": {"at": NOW - 86400 * 30}})
    assert checks["table_seen"]["passed"] is None and checks["calibration_recent"]["passed"] is None
    assert "30 天" in checks["calibration_recent"]["detail"]
    assert selfcheck.refusal(list(checks.values())) is None


def test_motor_faults_joints_past_limits_and_unmeasured_grippers_refuse():
    checks = _checks(readings={"left": _reading(raw=1.119, errors={"0x01": "overload"}), "right": _reading(raw=0.5)},
                     joints={"outside_limits": {"left": {"joint2": -0.2}}, "passed": False},
                     calibration={"gripper_limits": {"left": LIMITS["left"]}})
    assert {key for key, check in checks.items() if check["passed"] is False} == {
        "motors_ok", "joints_in_limits", "grippers_calibrated"}
