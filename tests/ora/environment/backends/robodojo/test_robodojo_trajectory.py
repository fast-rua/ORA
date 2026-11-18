"""follow_joint_trajectory and the rate-limited set_gripper on the fake plant.

The fake plant's ``native_tick_sequence`` (tests/support/robodojo/backend.py)
moves each joint ``track_gain`` of the way to its per-step target, freezes an
arm named in ``blocked`` and stops closing fingers at ``jaw_object``; its toy
kinematics read joints 0-2 as the link6 position.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ora.contracts.action import ActionValidationError, validate_runtime_action
from ora.contracts.joint_trajectory import native_action_cap
from ora.environment.backends.robodojo import native_sequence
from tests.support.robodojo.backend import v3

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")

HOME = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
KEY = "left_arm_joint_state"


def _knots(step=0.04, count=10, joint=0):
    rows, current = [], list(HOME)
    for _ in range(count):
        current = list(current)
        current[joint] += step
        rows.append(current)
    return rows


def _trajectory(session, tmp_path, knots, **arguments):
    action = v3(session, tmp_path, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": knots}}, **arguments}
    return action


def _journal(session, receipt):
    trace = json.loads((session.output_root / receipt["receipt"]["control_trace"]["path"]).read_text())
    lines = [json.loads(line) for line in (session.output_root / trace["physics_journal"]["path"]).read_text().splitlines()]
    return trace, lines


def test_each_physics_step_moves_evenly_with_a_matching_velocity(sparse, tmp_path):
    knots = _knots()
    result = sparse.execute(_trajectory(sparse, tmp_path, knots))
    receipt = result["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["native_actions"] == 10
    trace, lines = _journal(sparse, result)
    assert trace["schema"] == native_sequence.CONTROL_TRACE_V2_SCHEMA and len(lines) == 100
    targets = [line["requested"][KEY]["position"][0] for line in lines]
    # One target every 4 ms, 0.004 rad apart: no restart from the measurement, no hold inside a tick.
    assert np.allclose(np.diff([0.0, *targets]), 0.004)
    velocities = [line["requested"][KEY]["velocity"][0] for line in lines]
    assert np.allclose(velocities[:-1], 0.04 / 0.04) and velocities[-1] == 0.0
    # The SDK buffers equal the resolved command, positions and velocities.
    assert all(line["applied_velocity_targets"][KEY][0] == line["command"][KEY]["velocity"][0] for line in lines)
    assert all("control_steps" in sample and len(sample["control_steps"]) == 10 for sample in trace["samples"])
    assert receipt["trajectory"]["status"] == "completed" and receipt["command_invariant"]["schema"] == "ora.command-invariant.v3"
    assert sparse.environment.robot_manager.control_manager.prev_control[0][KEY]["velocity"] == [0.0] * 6


def test_a_blocked_arm_trips_the_guard_and_holds_without_a_snap(sparse, tmp_path):
    sparse.environment.blocked = {"left_arm": True}
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.1, count=20)))
    receipt = result["receipt"]
    stop = receipt["trajectory"]["stop"]
    assert receipt["execution_status"] == "not_reached" and stop["reason"] in {"tracking", "progress"}
    assert receipt["native_actions"] < 20
    _, lines = _journal(sparse, result)
    tripped = next(index for index, line in enumerate(lines) if "stop" in line)
    held = [line["requested"][KEY] for line in lines[tripped + 1:]]
    # After the trip every step holds the joints measured at the trip, with velocity 0.
    assert held and all(step["position"] == pytest.approx(HOME) and step["velocity"] == [0.0] * 6 for step in held)
    assert sparse.environment.robot_manager.control_manager.prev_control[0][KEY]["position"] == pytest.approx(HOME)


def test_a_lagging_arm_is_stopped_at_a_quarter_knot_step(sparse, tmp_path):
    sparse.environment.track_gain = 0.02
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.15, count=20),
                                        guards={"progress": None, "tracking_growth": None}))
    stop = result["receipt"]["trajectory"]["stop"]
    assert stop["reason"] == "tracking" and stop["rule"] == "bound" and stop["bound"] == pytest.approx(0.0375)
    slow = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.01, count=20),
                                      guards={"progress": None, "tracking_growth": None}))
    assert slow["receipt"]["trajectory"]["guards"]["tracking_bound_rad"]["left"]["min"] == pytest.approx(0.012)


def test_a_growing_lag_trips_long_before_the_bound(sparse, tmp_path):
    # An arm pushed back: it follows less and less of each step, so its lag keeps growing.
    sparse.environment.track_gain = 0.05
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.03, count=20),
                                        guards={"progress": None, "tracking_rad": 1.0}))
    stop = result["receipt"]["trajectory"]["stop"]
    assert stop["reason"] == "tracking" and stop["rule"] == "growing_error"
    assert 0.004 < stop["error"] < 0.05 and stop["error_rise_rad"] > 0.0015
    assert result["receipt"]["native_actions"] <= 4


def test_expect_contact_turns_off_only_the_growing_error_rule(sparse, tmp_path):
    sparse.environment.track_gain = 0.05
    pushed = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.03, count=20),
                                        guards={"progress": None, "tracking_rad": 1.0}, expect_contact=True))
    receipt = pushed["receipt"]
    # The same growing lag that trips growing_error without it runs on (the loose bound never trips here).
    assert receipt["trajectory"]["stop"] is None and receipt["trajectory"]["expect_contact"] is True
    assert receipt["trajectory"]["guards"]["tracking_growth"] is None
    assert receipt["arms"]["left"]["motion_progress"]["expect_contact"] is True


def test_with_expect_contact_the_bound_still_stops_a_real_block_and_says_so(sparse, tmp_path):
    sparse.environment.track_gain = 0.02
    blocked = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.15, count=20),
                                         guards={"progress": None}, expect_contact=True))["receipt"]
    stop = blocked["trajectory"]["stop"]
    # The hard bound still stops a real block, and says so.
    assert stop["reason"] == "tracking" and stop["rule"] == "bound"
    assert stop["note"] == "stopped by the tracking bound despite expect_contact"
    progress = blocked["arms"]["left"]["motion_progress"]
    assert progress["stop_note"] == stop["note"] and progress["expect_contact"] is True


def test_the_receipt_echoes_how_each_target_orientation_was_chosen(sparse, tmp_path):
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(count=3), jaw_symmetry={"left": "twin_nearer_current"}))
    receipt = result["receipt"]
    assert receipt["trajectory"]["jaw_symmetry"] == {"left": "twin_nearer_current"}
    assert receipt["arms"]["left"]["motion_progress"]["jaw_symmetry"] == "twin_nearer_current"
    with pytest.raises(ActionValidationError):
        validate_runtime_action({"schema": "ora-motion-action.v3", "observation_id": "o", "action_id": "a",
                                 "bound_observation_sha256": "s", "action": "follow_joint_trajectory",
                                 "arguments": {"arms": {"left": {"knots_rad": [HOME]}}, "jaw_symmetry": {"left": "flip"}}})


def test_a_steady_small_lag_does_not_trip(sparse, tmp_path):
    # A plant that tracks closely (clean motion) keeps its small lag constant: neither rule trips.
    sparse.environment.track_gain = 0.6
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(step=0.02, count=20), guards={"progress": None}))
    assert result["receipt"]["trajectory"]["stop"] is None and result["receipt"]["execution_status"] == "reached"


def test_an_unsettled_arm_inside_the_endpoint_acceptance_is_reached_and_named(sparse, tmp_path):
    sparse.environment.track_gain = 0.3
    knots = _knots(step=0.002, count=5)
    result = sparse.execute(_trajectory(sparse, tmp_path, knots, guards={"progress": None},
                                        arms={"left": {"knots_rad": knots, "expected_link6_xyz_xyzw": [
                                            [k[0], k[1], k[2], 0.0, 0.0, 0.0, 1.0] for k in knots]}},
                                        settle={"native_actions": 1, "joint_tolerance_rad": 1e-9,
                                                "joint_speed_rad_s": 1e-9}))
    receipt = result["receipt"]
    assert receipt["trajectory"]["settle"]["settled"] is False
    assert receipt["execution_status"] == "reached" and receipt["failure_code"] == "arm_not_settled"
    assert receipt["failure_stage"] == "settle"
    progress = receipt["arms"]["left"]["motion_progress"]
    assert progress["outcome"] == "arrived_not_settled" and progress["settle_joint_error_rad"] is not None


def test_admission_refuses_an_unfollowable_step_before_any_write(sparse, tmp_path):
    knots = _knots(step=0.04)
    knots[3] = [value + 0.5 for value in knots[3]]
    result = sparse.execute(_trajectory(sparse, tmp_path, knots))
    receipt = result["receipt"]
    assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
    assert receipt["failure_code"] == "trajectory_step_exceeds_limit"
    first = sparse.execute(_trajectory(sparse, tmp_path, [[0.5, 0, 0.8, 0, 0, 0]]))["receipt"]
    assert first["failure_code"] == "trajectory_start_discontinuous"


def test_a_budget_below_the_knots_runs_the_part_it_covers_and_ends_at_rest(sparse, tmp_path):
    sparse.request_action_budget = 4
    try:
        result = sparse.execute(_trajectory(sparse, tmp_path, _knots(count=10)))
    finally:
        del sparse.request_action_budget
    receipt = result["receipt"]
    assert receipt["native_actions"] == 4 and receipt["failure_code"] == "native_action_budget_exhausted"
    _, lines = _journal(sparse, result)
    assert lines[-1]["requested"][KEY]["velocity"] == [0.0] * 6


def test_an_explicit_settle_waits_for_the_arm_and_reports_its_residual(sparse, tmp_path):
    sparse.environment.track_gain = 0.3
    result = sparse.execute(_trajectory(sparse, tmp_path, _knots(count=5), guards={"progress": None},
                                        settle={"native_actions": 5, "joint_tolerance_rad": 0.001,
                                                "joint_speed_rad_s": 0.05}))
    settle = result["receipt"]["trajectory"]["settle"]
    assert settle["settled"] and 0 < settle["native_actions"] <= 5
    assert result["receipt"]["execution_status"] == "reached"


def test_restore_sends_the_recorded_steps_again(sparse, tmp_path):
    from ora.environment.backends.robodojo.control_replay import recorded_actions

    sparse.execute(_trajectory(sparse, tmp_path, _knots(count=3)))
    actions = recorded_actions(sparse.output_root)
    steps = [step for sample in actions[-1]["samples"] for step in sample["control_steps"]]
    sparse.environment.sequence_steps = []
    for sample in actions[-1]["samples"]:
        native_sequence.replay_recorded_tick(sparse, sample["control_steps"])
    assert sparse.environment.sequence_steps == steps


def test_the_rate_limited_close_ramps_bounds_its_lead_and_holds_a_constant_squeeze(sparse, tmp_path):
    sparse.environment.jaw_object = {"left_arm": 0.02}
    action = v3(sparse, tmp_path, kind="set_gripper", targets={"left": {
        "opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005,
        "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 3}}})
    result = sparse.execute(action)
    receipt = result["receipt"]
    assert receipt["failure_code"] == "gripper_stalled_before_target"
    assert receipt["gripper_ramp"]["arm_settle"]["settled"] is True
    _, lines = _journal(sparse, result)
    ramp = [line for line in lines if line["segment"] == "gripper"]
    targets = [line["requested"]["left_ee_joint_state"]["position"][0] for line in ramp]
    steps = -np.diff([0.044, *targets])
    assert np.all(steps <= 2.0 * 0.054 * 0.004 + 1e-9)  # the rate, per physics step
    fingers = [line["measured"]["left"]["gripper_finger_qpos"][0] for line in ramp]
    assert np.all(np.asarray(fingers) - np.asarray(targets) <= 0.005 + 1e-9)  # never more than the lead
    tail = targets[-20:]
    assert max(tail) - min(tail) < 1e-9  # a constant squeeze, no 40 ms pulse


def test_a_v2_close_approaches_at_the_sdk_lead_and_squeezes_at_its_hold_lead(sparse, tmp_path):
    sparse.environment.jaw_object = {"left_arm": 0.02}
    action = v3(sparse, tmp_path, kind="set_gripper", targets={"left": {
        "opening": "close", "rate_per_s": None, "max_lead_m": None, "hold_lead_m": 0.005,
        "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 5, "first_reading": "joint_velocity"},
        "end": {"tolerance": 0.03, "stall_window_native_actions": 2, "stall_still_step": 0.005}}})
    result = sparse.execute(action)
    receipt = result["receipt"]
    # Stopped on the object: honestly stalled, not reached.
    assert receipt["failure_code"] == "gripper_stalled_before_target"
    assert receipt["arms"]["left"]["gripper_outcome"]["state"] == "stalled_before_target"
    # The arm was at rest by its measured joint velocities: no settle look was spent.
    settle = receipt["gripper_ramp"]["arm_settle"]
    assert settle == {"joint_speed_limit_rad_s": 0.05, "native_actions": 0, "settled": True,
                      "first_reading": "joint_velocity", "arm_joint_speed_rad_s": 0.0}
    # One moving native action, then two still ones end it (window 2).
    assert receipt["native_actions"] == 3 and receipt["gripper_ramp"]["ramp_native_actions"] == 3
    arm = receipt["gripper_ramp"]["arms"]["left"]
    assert arm["hold_lead_m"] == 0.005 and arm["held_at_end"] is True and arm["max_lead_m"] == 0.0108
    _, lines = _journal(sparse, result)
    ramp = [line for line in lines if line["segment"] == "gripper"]
    targets = [line["requested"][EE]["position"][0] for line in ramp]
    fingers = [line["measured"]["left"]["gripper_finger_qpos"][0] for line in ramp]
    leads = np.asarray(fingers) - np.asarray(targets)
    assert leads.max() <= 0.0108 + 1e-9 and leads[:20].max() > 0.005  # the approach and the first still tick
    assert np.allclose(leads[-10:], 0.005)  # the squeeze it keeps: the hold lead
    assert ramp[-1]["gripper_ramp"]["holding"] == {"left": True}
    held = sparse.environment.robot_manager.control_manager.prev_control[0][EE]["position"][0]
    assert held == pytest.approx(0.02 - 0.005)


def test_a_moving_arm_still_holds_the_close_until_it_settles(sparse, tmp_path):
    data = sparse.environment.robot_manager.robot_key[0].data
    data.joint_vel[0][:6] = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]  # still moving: the first reading says so
    action = v3(sparse, tmp_path, kind="set_gripper", targets={"left": {
        "opening": "close", "rate_per_s": None, "max_lead_m": None, "hold_lead_m": 0.005,
        "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 5, "first_reading": "joint_velocity"}}})
    receipt = sparse.execute(action)["receipt"]
    settle = receipt["gripper_ramp"]["arm_settle"]
    assert settle["first_reading"] == "joint_velocity" and settle["native_actions"] == 1 and settle["settled"] is True
    # Closing on nothing reaches the floor: reached, not stalled.
    assert receipt["arms"]["left"]["gripper_outcome"]["state"] == "reached_target"


def test_a_close_without_the_v2_fields_keeps_its_receipt(sparse, tmp_path):
    sparse.environment.jaw_object = {"left_arm": 0.02}
    action = v3(sparse, tmp_path, kind="set_gripper", targets={"left": {
        "opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005,
        "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 3}}})
    result = sparse.execute(action)
    ramp = result["receipt"]["gripper_ramp"]
    assert set(ramp["arm_settle"]) == {"joint_speed_limit_rad_s", "native_actions", "settled", "arm_joint_speed_rad_s"}
    assert ramp["arm_settle"]["native_actions"] == 1  # the look a v1 close always spends
    assert "hold_lead_m" not in ramp["arms"]["left"] and "held_at_end" not in ramp["arms"]["left"]
    _, lines = _journal(sparse, result)
    assert all("holding" not in line["gripper_ramp"] for line in lines)


def test_a_trajectory_that_is_part_of_a_move_is_never_reported_as_arrival(sparse, tmp_path):
    knots = _knots(count=5)
    goal = [0.6, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    result = sparse.execute(_trajectory(sparse, tmp_path, knots,
                                        part_of_move={"fraction": 0.4, "goals_xyz_xyzw": {"left": goal}}))
    receipt = result["receipt"]
    assert receipt["trajectory"]["status"] == "completed"  # every knot ran ...
    assert receipt["execution_status"] == "not_reached"  # ... but the move did not reach its goal
    assert receipt["failure_code"] == "native_action_budget_exhausted"
    part = receipt["trajectory"]["part_of_move"]
    assert part["residuals_against"] == "cut_point" and part["remaining_to_goal_m"]["left"] == pytest.approx(0.4)
    progress = receipt["arms"]["left"]["motion_progress"]
    assert progress["outcome"] == "cut_short" and progress["pose_error_against"] == "cut_point"
    assert progress["remaining_to_goal_m"] == pytest.approx(0.4)


RATE_STEP = 2.0 * 0.054 * 0.004  # 2 openings/s on the 0.054 m range, per physics step
EE = "left_ee_joint_state"


def _close_on_object(session, tmp_path):
    session.environment.jaw_object = {"left_arm": 0.02}
    action = v3(session, tmp_path, kind="set_gripper",
                targets={"left": {"opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005}})
    session.execute(action)
    held = session.environment.robot_manager.control_manager.prev_control[0][EE]["position"][0]
    assert held == pytest.approx(0.02 - 0.005)  # the object holds the finger; the target leads it by 5 mm


def _ramp_targets(session, result):
    _, lines = _journal(session, result)
    return [line["requested"][EE]["position"][0] for line in lines if line["segment"] == "gripper"]


def test_a_repeated_close_on_a_held_object_keeps_its_squeeze(sparse, tmp_path):
    _close_on_object(sparse, tmp_path)
    again = sparse.execute(v3(sparse, tmp_path, kind="set_gripper",
                              targets={"left": {"opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005}}))
    targets = _ramp_targets(sparse, again)
    # The ramp starts from the held target, not the measured finger: no step back toward the finger.
    assert targets and max(targets) <= 0.015 + 1e-9 and min(targets) == pytest.approx(0.015)


def test_a_slow_release_ramps_the_squeeze_down(sparse, tmp_path):
    _close_on_object(sparse, tmp_path)
    release = sparse.execute(v3(sparse, tmp_path, kind="set_gripper",
                                targets={"left": {"opening": "open", "rate_per_s": 2.0, "max_lead_m": None}}))
    targets = _ramp_targets(sparse, release)
    steps = np.diff([0.015, *targets])
    # From the held 0.015 m, never more than the rate per physics step: no one-step drop of the squeeze.
    assert targets[0] <= 0.015 + RATE_STEP + 1e-9
    assert np.all(steps >= -1e-12) and np.all(steps <= RATE_STEP + 1e-9)


def test_trajectory_gripper_knots_start_from_the_held_target(sparse, tmp_path):
    _close_on_object(sparse, tmp_path)
    knots = _knots(count=3)
    result = sparse.execute(_trajectory(sparse, tmp_path, knots,
                                        arms={"left": {"knots_rad": knots, "gripper_knots": [0.0, 0.0, 0.0]}}))
    _, lines = _journal(sparse, result)
    first = lines[0]["requested"][EE]["position"][0]
    assert first <= 0.015 + 1e-9  # not a dip to the measured finger at 0.02


def test_the_fake_plant_applies_the_sdk_gripper_limiter(sparse):
    from tests.support.robodojo.backend import sdk_gripper_limit

    robot = sparse.environment.robot_manager.robot_list[0]
    assert sdk_gripper_limit(robot, 0.0, 0.044)[0] == pytest.approx(0.044 - 0.0108)
    assert sdk_gripper_limit(robot, 0.04, 0.044)[0] == 0.04


def test_contract_admits_the_new_actions_and_caps_their_native_actions():
    base = {"schema": "ora-motion-action.v3", "observation_id": "o", "action_id": "a",
            "bound_observation_sha256": "s"}
    trajectory = validate_runtime_action({**base, "action": "follow_joint_trajectory",
                                          "arguments": {"arms": {"left": {"knots_rad": [HOME]}}}})
    assert trajectory["arguments"]["settle"]["native_actions"] == 0 and native_action_cap(trajectory) == 250
    gripper = validate_runtime_action({**base, "action": "set_gripper",
                                       "arguments": {"targets": {"left": {"opening": "close", "rate_per_s": 1.0}}}})
    assert native_action_cap(gripper) == 250
    assert trajectory["arguments"]["expect_contact"] is False
    with pytest.raises(ActionValidationError):
        validate_runtime_action({**base, "action": "follow_joint_trajectory",
                                 "arguments": {"arms": {"left": {"knots_rad": [HOME]}}, "expect_contact": "yes"}})
    assert native_action_cap({**base, "action": "set_gripper", "arguments": {"targets": {"left": "close"}}}) == 20
    with pytest.raises(ActionValidationError):
        validate_runtime_action({**base, "action": "follow_joint_trajectory",
                                 "arguments": {"arms": {"left": {"knots_rad": [HOME[:5]]}}}})
    with pytest.raises(ActionValidationError):
        validate_runtime_action({**base, "action": "set_gripper",
                                 "arguments": {"targets": {"left": {"opening": "close", "max_lead_m": 0.05}}}})


def test_a_fresh_plant_restores_a_trajectory_and_a_ramp_exactly_from_the_recorded_steps(tmp_path):
    from ora.environment.backends.robodojo.control_replay import ReplayJournal, recorded_actions, replay_control
    from tests.support.robodojo.backend import sparse_session

    original = sparse_session(tmp_path / "original")
    original.environment.track_gain = 0.5
    original.execute(_trajectory(original, tmp_path, _knots(count=6), guards={"progress": None}))
    original.environment.jaw_object = {"left_arm": 0.02}
    original.execute(v3(original, tmp_path, kind="set_gripper",
                        targets={"left": {"opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005}}))
    restored = sparse_session(tmp_path / "restored")
    restored.environment.track_gain = 0.5
    restored.environment.jaw_object = {"left_arm": 0.02}
    report = replay_control(restored, recorded_actions(original.output_root),
                            journal=ReplayJournal(tmp_path / "replay"))
    assert report["ticks_replayed"] == report["ticks_recorded"] > 6
    assert report["applied_target_mismatches"] == 0 and not any(report["max_divergence"].values())
    lines = [json.loads(line) for line in (tmp_path / "replay/controller/action-0000.jsonl").read_text().splitlines()]
    assert all("applied_velocity_targets" in line and "requested" in line for line in lines)
    assert restored.environment.sequence_steps == original.environment.sequence_steps
