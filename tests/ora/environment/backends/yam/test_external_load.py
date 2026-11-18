"""A YAM arm pushed on from outside stops where it is (external_load.py, RealRig's guards)."""

from __future__ import annotations

import types

import numpy as np

from ora.environment.backends.yam import external_load
from ora.environment.controllers import yam_station as Y


class _Kdl:
    def compute_inverse_dynamics(self, q, qd, qdd):
        return np.array([0.0, 3.0, 1.5, 0.4, 0.1, 0.0])            # gravity at this pose


def _robot(**fields):
    return types.SimpleNamespace(kdl=_Kdl(), gravity_comp_factor=np.ones(7), use_coulomb_friction=True,
                                 _coulomb_friction=np.full(7, 0.2), **fields)


def test_the_residual_is_measured_minus_i2rts_own_gravity_and_friction():
    q, vel = np.zeros(6), np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0])
    expected = np.array([0.0, 3.2, 1.3, 0.4, 0.1, 0.0])
    assert np.allclose(external_load.model_torque(_robot(), q, vel), expected)
    observation = {"joint_pos": q, "joint_vel": vel, "joint_eff": expected + np.array([0, 7.0, 0, 0, 0, 0])}
    assert np.allclose(external_load.residual(_robot(), observation), [0, 7.0, 0, 0, 0, 0])
    assert external_load.residual(types.SimpleNamespace(), observation) is None   # no model, no estimate


def test_a_load_over_the_limit_for_the_ticks_stops_and_names_the_joint():
    monitor = external_load.LoadMonitor(["left", "right"])
    load = np.zeros(6)
    load[1] = Y.EXTERNAL_LOAD_STOP_NM[1] + 1.0
    for _ in range(Y.EXTERNAL_LOAD_STOP_TICKS - 1):
        assert monitor.update("left", load) is None
    event = monitor.update("left", load)
    assert event["code"] == "external_load_stop" and event["arm"] == "left" and event["joint"] == "joint2"
    assert event["residual_torque_nm"] == round(load[1], 2) and "contact" not in event
    peaks = monitor.take_peaks()
    assert peaks["left"][1] == round(load[1], 3) and peaks["right"] == [0.0] * 6
    assert monitor.take_peaks()["left"] == [0.0] * 6, "taken peaks start again"


def test_a_load_that_dips_below_the_limit_starts_counting_again():
    monitor = external_load.LoadMonitor(["left"])
    high, low = np.full(6, 0.0), np.zeros(6)
    high[3] = Y.EXTERNAL_LOAD_STOP_NM[3] + 0.5
    for _ in range(Y.EXTERNAL_LOAD_STOP_TICKS - 1):
        assert monitor.update("left", high) is None
    assert monitor.update("left", low) is None
    assert all(monitor.update("left", high) is None for _ in range(Y.EXTERNAL_LOAD_STOP_TICKS - 1))
    monitor.reset("left")
    assert monitor.update("left", high) is None


def test_the_limits_are_far_below_the_motor_guard():
    for limit, (_, kind) in zip(Y.EXTERNAL_LOAD_STOP_NM, Y.ARM_MOTORS):
        assert limit <= Y.JOINT_EFFORT_STOP_FRACTION * Y.MOTOR_TORQUE_MAX_NM[kind] / 3   # at most a third
