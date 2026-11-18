"""The station's arm paths timed by the shared TOPP-RA + Ruckig worker (``runtime.motion_timing``, off by default).

The switch-only cases run anywhere. The planner cases build the station model (MuJoCo and i2rt); the timed ones
also need the timing interpreter (``ORA_TIMING_PYTHON``, which the test conftest hands over as
``ORA_TEST_TIMING_PYTHON``) and skip without it, as in CI.
"""

from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest

from ora.contracts.motion_speed import speed_caps
from ora.environment.backends.yam import motion, shared_motion
from ora.environment.controllers import motion_timing
from ora.environment.controllers import yam_station as Y

TIMING_PYTHON = os.environ.get("ORA_TEST_TIMING_PYTHON")
ON = {"motion_timing": {"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}}
DOWN = [1.0, 0.0, 0.0, 0.0]
TICK = Y.CONTROL_PERIOD_S
JERK = motion_timing.jerk_limits(shared_motion.ROBOT, Y.ARM_JOINT_COUNT)["rad_s3"]


def test_each_interaction_keeps_the_station_speeds_and_reaches_them_in_the_declared_ramp():
    for interaction in ("free_space", "precision", "contact_approach"):
        caps, slow = speed_caps(shared_motion.speed_request(interaction))
        assert slow is None
        assert caps["linear_m_s"] == Y.LINEAR_SPEED_M_S[interaction]
        assert math.isclose(math.radians(caps["angular_deg_s"]), Y.ANGULAR_SPEED_RAD_S[interaction])
        assert caps["joint_speed_rad_s"] == Y.MAX_JOINT_SPEED_RAD_S
        assert math.isclose(caps["linear_accel_m_s2"] * Y.TIMING_RAMP_S, caps["linear_m_s"])
        assert math.isclose(caps["joint_accel_rad_s2"] * Y.TIMING_RAMP_S, caps["joint_speed_rad_s"])
    assert motion_timing.jerk_limits(shared_motion.ROBOT, 6)["source"] == motion_timing.UNVERIFIED


def test_the_switch_is_off_by_default_and_without_a_worker_the_station_times_and_says_so(monkeypatch):
    still = [np.zeros(6), np.full(6, 0.1)]
    off = types.SimpleNamespace(config={})
    assert not shared_motion.enabled(off) and shared_motion.retime(off, "left", still, {}) is None
    on = types.SimpleNamespace(config=ON)
    motion_timing.configure(None)
    assert shared_motion.retime(on, "left", still, shared_motion.speed_request("free_space")) == {
        "implementation": shared_motion.STATION_IMPLEMENTATION, "reason": "no timing interpreter (ORA_TIMING_PYTHON)"}
    assert shared_motion.retime(on, "left", [np.zeros(6), np.zeros(6)], {})["reason"] == "the path does not move"


def _station(config):
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    from ora.environment.backends.yam.kinematics import StationKinematics, pose_matrix
    from ora.environment.backends.yam.scene import DEFAULT_TABLE, build_cell_model

    kin = StationKinematics(build_cell_model(table=DEFAULT_TABLE))
    q = {arm: np.zeros(6) for arm in Y.ARMS}
    rig = types.SimpleNamespace(arm_q=lambda arm: np.asarray(q[arm], float),
                                table_top_z_m=float(DEFAULT_TABLE["top_z_m"]))
    session = types.SimpleNamespace(kinematics=kin, rig=rig, home_q={}, last_safe_q=None, config=dict(config),
                                    finger_state=lambda: {arm: Y.FINGER_OPEN_M for arm in Y.ARMS})

    def place(arm, xyz):
        result = kin.ik(arm, pose_matrix(xyz, DOWN), q[arm], max_iterations=400)
        assert result.ok, xyz
        q[arm] = result.q

    return session, place, lambda xyz: pose_matrix(xyz, DOWN)


@pytest.fixture
def timing():
    if not TIMING_PYTHON:
        pytest.skip("no timing interpreter (ORA_TIMING_PYTHON)")
    motion_timing.configure(TIMING_PYTHON)
    yield
    motion_timing.reset()


def _profile(session, arm, schedule):
    """Per tick, from the start at rest to the end at rest: joint speed, acceleration and jerk, TCP speed and ramp."""

    rows = np.vstack([session.rig.arm_q(arm), [tick[arm] for tick in schedule.ticks]])
    rest = np.zeros((1, rows.shape[1]))
    velocity = np.vstack([rest, np.diff(rows, axis=0) / TICK, rest])
    acceleration = np.diff(velocity, axis=0) / TICK
    jerk = np.diff(acceleration, axis=0) / TICK
    tcp = np.asarray([session.kinematics.fk(arm, row)[:3, 3] for row in rows])
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(tcp, axis=0), axis=1) / TICK, [0.0]])
    return rows, velocity, acceleration, jerk, speed, np.diff(speed) / TICK


def _moves(session, place, at):
    """A free-space lift, traverse and descent of the left arm, and a precision descent of the right."""

    place("left", [0.28, 0.22, 0.05])
    place("right", [0.42, -0.14, 0.112])
    current = session.kinematics.fk("left", session.rig.arm_q("left"))
    waypoints = motion.free_space_waypoints(session, "left", current, at([0.40, 0.15, 0.05]))
    assert len(waypoints) == 3
    return {"free_space": {"left": (waypoints, "free_space")},
            "precision": {"right": ([at([0.42, -0.14, 0.03])], "precision")},
            "both": {"left": ([at([0.30, 0.20, 0.10])], "precision"),
                     "right": ([at([0.40, -0.20, 0.06])], "contact_approach")}}


def test_a_retimed_motion_keeps_every_cap_starts_and_ends_at_rest_and_reaches_the_same_joints(timing):
    session, place, at = _station(ON)
    for name, plans in _moves(session, place, at).items():
        session.config = {}
        fixed = motion.plan_motion(session, plans)
        assert fixed.motion is None
        session.config = dict(ON)
        timed = motion.plan_motion(session, plans)
        assert timed.summary["ticks"] == len(timed.ticks) == timed.motion["planned_ticks"]
        for arm, (_waypoints, interaction) in plans.items():
            caps = speed_caps(shared_motion.speed_request(interaction))[0]
            assert np.allclose(timed.finals[arm], fixed.finals[arm], atol=1e-9), (name, arm)
            assert np.allclose(timed.ticks[-1][arm], fixed.finals[arm], atol=1e-9)
            _rows, velocity, acceleration, jerk, speed, ramp = _profile(session, arm, timed)
            assert np.abs(velocity).max() <= caps["joint_speed_rad_s"], (name, arm)
            assert np.abs(acceleration).max() <= caps["joint_accel_rad_s2"], (name, arm)
            assert np.all(np.abs(jerk) <= np.asarray(JERK)), (name, arm)
            assert speed.max() <= caps["linear_m_s"], (name, arm)
            assert np.abs(ramp[1:-1]).max() <= caps["linear_accel_m_s2"], (name, arm)
            # From rest and to rest: the first and last tick move a few millimetres per second, not the full speed.
            assert speed[1] < 0.25 * caps["linear_m_s"] and speed[-2] < 0.25 * caps["linear_m_s"], (name, arm)
            # The station's own timing starts and stops at full speed: the jerk the switch removes.
            _, fixed_velocity, fixed_acceleration, *_ = _profile(session, arm, fixed)
            assert np.abs(fixed_acceleration).max() > 4 * caps["joint_accel_rad_s2"]
            entry = timed.motion["timing"][arm]
            assert entry["implementation"] == "toppra_ruckig", (name, entry)
            assert all(item["jerk_source"] == motion_timing.UNVERIFIED for item in entry["segments"])
            assert timed.motion["speed"][arm]["linear_m_s"] == caps["linear_m_s"]
        # A lift, traverse and descent stops at its corners: each straight piece is timed rest to rest.
        if name == "free_space":
            assert len(timed.motion["timing"]["left"]["segments"]) == 3


def test_the_per_tick_checks_run_on_the_retimed_ticks(timing):
    """The from-rest widening of 11-14 (bbbd52ee9) holds on the timed rows; table clearance still refuses."""

    session, place, at = _station(ON)
    kin = session.kinematics
    rest = {"left": np.array([-0.018, 0.002, 0.001, -0.053, -0.016, -0.071]),
            "right": np.array([-0.014, 0.001, 0.001, -0.052, -0.008, 0.001])}
    kin.joint_offsets = {"left": np.array([0.0, -0.0154, 0.0001, -0.0547, 0.0354, 0.0]),
                         "right": np.array([0.0, -0.0094, -0.0177, -0.0311, 0.0181, 0.0])}
    session.rig.arm_q = lambda arm: rest[arm].copy()
    kin.set_state(rest, session.finger_state())
    lifted = kin.fk("right", rest["right"])
    lifted[2, 3] += 0.03
    seen = []
    within = kin.within_limits
    kin.within_limits = lambda arm, q, start=None: seen.append(np.asarray(q).copy()) or within(arm, q, start=start)
    schedule = motion.plan_motion(session, {"right": ([lifted], "free_space")})
    assert schedule.motion["timing"]["right"]["implementation"] == "toppra_ruckig"
    assert len(seen) == len(schedule.ticks) and all(np.array_equal(a, b["right"]) for a, b in zip(seen, schedule.ticks))
    limits = kin.widened_limits("right", rest["right"])
    assert all(np.all(tick["right"] >= limits[:, 0] - 1e-6) for tick in schedule.ticks), "no joint went further out"

    session2, place2, at2 = _station(ON)
    place2("right", [0.42, -0.14, 0.112])
    with pytest.raises(motion.Refusal) as sinking:
        motion.plan_motion(session2, {"right": ([at2([0.42, -0.14, 0.009])], "free_space")})
    assert sinking.value.code == "table_clearance"


def test_a_recovery_joint_line_is_retimed_with_every_arm_on_one_progress(timing):
    session, place, _at = _station(ON)
    place("left", [0.28, 0.22, 0.10])
    place("right", [0.42, -0.14, 0.112])
    start = {arm: session.rig.arm_q(arm) for arm in Y.ARMS}
    goals = {"left": start["left"] + np.array([0.0, 0.25, 0.25, 0.0, 0.0, 0.0]),
             "right": start["right"] + np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0])}
    schedule = motion.plan_joint_motion(session, goals)
    assert list(schedule.motion["timing"]) == ["left"]
    for arm in Y.ARMS:
        rows, velocity, acceleration, *_ = _profile(session, arm, schedule)
        assert np.array_equal(rows[-1], goals[arm])
        assert np.abs(velocity).max() <= Y.MAX_JOINT_SPEED_RAD_S * 0.5
        assert np.abs(acceleration).max() <= Y.MAX_JOINT_SPEED_RAD_S * 0.5 / Y.TIMING_RAMP_S
        # Every row lies on the arm's joint line.
        travel = goals[arm] - start[arm]
        share = (rows - start[arm]) @ travel / (travel @ travel)
        assert np.allclose(rows, start[arm] + share[:, None] * travel, atol=1e-9)
        assert np.all(np.diff(share) >= -1e-12)


def test_a_session_binds_the_worker_its_preflight_started_or_says_the_station_times(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("ORA_TIMING_PYTHON", raising=False)
    native = tmp_path / "run" / "native"
    native.mkdir(parents=True)
    assert shared_motion.start(types.SimpleNamespace(config={}, output_root=native)) is None
    assert not (native / "timing").exists()
    session = types.SimpleNamespace(config=ON, output_root=native)
    # A direct session (no preflight record) and no ORA_TIMING_PYTHON: the station times, and says so.
    assert shared_motion.start(session)["implementation"] == shared_motion.STATION_IMPLEMENTATION
    assert json.loads((native / "timing/binding.json").read_text())["implementation"] == "yam_fixed_speed"
    # A run whose preflight did not start the worker takes the station's timing explicitly.
    (native.parent / "preflight.json").write_text(json.dumps({"checks": []}))
    assert shared_motion.start(session)["implementation"] == shared_motion.STATION_IMPLEMENTATION
    assert motion_timing.client() is None
    if not TIMING_PYTHON:
        return
    from ora.orchestrator.preflight.checks import motion_timing_check

    monkeypatch.setenv("ORA_TIMING_PYTHON", TIMING_PYTHON)
    check = motion_timing_check(ON["motion_timing"])
    assert check["passed"], check
    (native.parent / "preflight.json").write_text(json.dumps({"checks": [check]}))
    monkeypatch.delenv("ORA_TIMING_PYTHON")
    bound = shared_motion.start(session)
    assert bound["implementation"] == "toppra_ruckig" and bound["python_sha256"] == check["detail"]["python_sha256"]
    assert json.loads((native / "timing/binding.json").read_text())["python_sha256"] == bound["python_sha256"]
    shared_motion.close(session)
