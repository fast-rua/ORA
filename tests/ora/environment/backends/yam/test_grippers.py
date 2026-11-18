"""A YAM gripper gives torque only while it works: held arms relax it at once, ``grip`` engages
it for the move and relaxes it after, and relaxing twice or engaging a relaxed one is exact."""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from ora.environment.backends.yam import grippers
from ora.environment.controllers import yam_station as Y
from ora.environment.backends.yam.cell_adapter import YamHeld


class Robot:
    def __init__(self) -> None:
        self._gripper_index = 6
        self._kp, self._kd = np.full(7, 20.0), np.full(7, 1.0)
        self.gravity_comp_factor = np.ones(7)
        self._coulomb_friction = np.full(7, 0.1)
        self._command_lock = threading.Lock()
        self._commands = types.SimpleNamespace(kp=self._kp, kd=self._kd)
        self.motor_chain = types.SimpleNamespace(running=True)
        self.jaws, self.q = 0.0, np.zeros(6)
        self._gripper_limits = [6.4, 1.1]                  # motor angles at [closed, open]
        self.block = 0.0                                   # an object stops closing jaws at this opening
        self.commanded_jaws: list[float] = []
        self.torques: list[float] = []

    def update_kp_kd(self, kp, kd) -> None:
        self._kp, self._kd = kp, kd

    def command_joint_pos(self, target) -> None:
        self._commands.torques = np.zeros(7)              # every i2rt command zeroes the feed-forward torques
        self.commanded_jaws.append(float(target[6]))
        if self._commands.kp[6] > 0:
            self.jaws = float(target[6])

    def get_observations(self):
        # In force control the jaws move with the host's torque (motor angle up = closing), up to an object.
        torque = float(np.asarray(getattr(self._commands, "torques", np.zeros(7)))[6])
        if torque:
            self.torques.append(torque)
            closed, opened = self._gripper_limits
            motor = closed + self.jaws * (opened - closed) + 0.5 * torque
            self.jaws = float(np.clip((motor - closed) / (opened - closed), self.block, 1.0))
        return {"joint_pos": self.q.copy(), "gripper_pos": np.array([self.jaws]), "gripper_eff": np.zeros(1),
                "temp_mos": np.full(7, 30.0), "temp_rotor": np.full(7, 30.0)}

    def close(self) -> None:
        pass


def test_relax_takes_every_gripper_torque_and_engage_gives_it_back() -> None:
    robot = Robot()
    grippers.relax(robot)
    grippers.relax(robot)
    assert robot._commands.kp[6] == 0.0 and robot._commands.kd[6] == 0.0
    assert robot.gravity_comp_factor[6] == 0.0 and robot._coulomb_friction[6] == 0.0
    assert np.all(robot._kp[:6] == 20.0)                 # the arm keeps its gains
    grippers.engage(robot)
    grippers.engage(robot)
    assert robot._commands.kp[6] == 20.0 and robot._commands.kd[6] == 1.0
    assert robot.gravity_comp_factor[6] == 1.0 and robot._coulomb_friction[6] == 0.1


def test_held_arms_relax_their_grippers_and_grip_works_only_while_it_moves(monkeypatch) -> None:
    from ora.environment.backends.yam import cell_adapter

    monkeypatch.setattr(cell_adapter.time, "sleep", lambda _s: None)
    robot = Robot()
    held = YamHeld({"left": robot}, {"left": "can_left"})
    assert grippers.is_relaxed(robot)
    held.command(held.joints())
    assert robot.jaws == 0.0                             # a hold sends no gripper torque

    measured = held.grip({"left": 0.6})
    assert measured["left"] == pytest.approx(0.6, abs=Y.GRIPPER_ARRIVAL_TOLERANCE)
    assert grippers.is_relaxed(robot) and not grippers.in_force_control(robot)   # relaxed again once there
    assert held.alerts() == []


def test_the_operators_grip_closes_like_a_motion_and_stops_on_what_it_holds(monkeypatch) -> None:
    """``ora cell``'s grip (the board calibration closes and opens the jaws) once stepped the command to closed
    behind i2rt's 50 N limiter; now it ramps in force control and stops where the jaws stall."""
    from ora.environment.backends.yam import cell_adapter

    monkeypatch.setattr(cell_adapter.time, "sleep", lambda _s: None)
    robot = Robot()
    robot.jaws, robot.block = 0.9, 0.4
    held = YamHeld({"left": robot}, {"left": "can_left"})
    measured = held.grip({"left": 0.0})
    assert measured["left"] == pytest.approx(0.4, abs=0.01)          # stopped on the object
    cap = Y.GRIPPER_FORCE_LIMIT_N * Y.GRIPPER_FORCE_TO_TORQUE
    assert robot.torques and max(abs(t) for t in robot.torques) <= cap + 1e-9
    # every command a few millimetres from the jaw at most, never the end of the travel at once
    lead = Y.GRIPPER_COMMAND_LEAD_M / Y.FINGER_OPEN_M
    assert min(robot.commanded_jaws) >= 0.4 - lead - 1e-9
    assert len(robot.commanded_jaws) < Y.GRIPPER_TICKS                 # it stopped on the stall, not the budget
    assert grippers.is_relaxed(robot) and not grippers.in_force_control(robot)


def test_limit_force_lowers_the_grip_force_i2rts_limiter_holds() -> None:
    """``get_yam_robot`` builds the limiter with 50 N; the station lowers it to GRIPPER_FORCE_LIMIT_N."""
    from functools import partial

    def force_map(motor_stroke, gripper_stroke, gripper_force, current_angle):
        return gripper_force * gripper_stroke / motor_stroke

    robot = Robot()
    robot._gripper_force_limiter = types.SimpleNamespace(
        max_force=50.0, gripper_force_torque_map=partial(force_map, motor_stroke=6.57, gripper_stroke=0.096,
                                                         gripper_force=50.0))
    assert grippers.limit_force(robot, 15.0) == 50.0
    limiter = robot._gripper_force_limiter
    assert limiter.max_force == 15.0 and robot._limit_gripper_force == 15.0
    assert abs(limiter.gripper_force_torque_map(current_angle=0.0) - 15.0 * 0.096 / 6.57) < 1e-12
    assert grippers.limit_force(types.SimpleNamespace(_gripper_index=None), 15.0) is None


def test_force_control_caps_the_torque_from_the_first_contact_on() -> None:
    """The jaws can never push harder than GRIPPER_FORCE_LIMIT_N: the cap holds friction compensation included."""
    cap = Y.GRIPPER_FORCE_LIMIT_N * Y.GRIPPER_FORCE_TO_TORQUE
    for error in (-6.0, -0.41, -0.05, 0.05, 0.41, 6.0):            # 0.41 rad of motor angle: 6 mm of aperture
        torque = grippers.force_torque(error, 0.0)
        assert abs(torque) <= cap + 1e-12 and np.sign(torque) == np.sign(error)
    assert abs(grippers.force_torque(6.0, 0.0)) == pytest.approx(cap)
    assert grippers.force_torque(0.01, 0.0) == pytest.approx(Y.GRIPPER_SOFT_KP_NM_PER_RAD * 0.01)   # no friction
    assert grippers.force_torque(6.0, 0.0, force_n=8.0) == pytest.approx(8.0 * Y.GRIPPER_FORCE_TO_TORQUE)
    # Position control with the arm's 20 N·m/rad on the same 0.41 rad: what force control replaces.
    assert 20.0 * 0.41 > 30 * cap


def test_take_force_control_turns_the_position_gain_off_and_relax_takes_the_rest() -> None:
    robot = Robot()
    robot._gripper_limits = [6.0, 0.5]                                  # [closed, open] motor angles
    grippers.take_force_control(robot)
    assert grippers.in_force_control(robot)
    assert robot._commands.kp[6] == 0.0 and robot._commands.kd[6] == 1.0   # its own damping stays
    assert robot.gravity_comp_factor[6] == 0.0 and robot._coulomb_friction[6] == 0.0
    torque = grippers.apply_force(robot, target_opening=0.0, measured_opening=1.0)   # close from open
    assert torque > 0.0 and robot._commands.torques[6] == torque                 # closed is the larger angle here
    assert np.all(robot._commands.torques[:6] == 0.0)
    grippers.relax(robot)
    assert not grippers.in_force_control(robot)
    assert robot._commands.kd[6] == 0.0 and robot._commands.torques[6] == 0.0
    assert grippers.apply_force(robot, 0.0, 1.0) is None                         # relaxed: nothing sent
    grippers.engage(robot)
    assert robot._commands.kp[6] == 20.0 and not grippers.in_force_control(robot)


def test_every_program_that_takes_an_arm_over_gets_the_same_gripper_protection(monkeypatch) -> None:
    """``arms.connect`` is how both an episode's worker and ``ora cell``'s operator tools take an arm over: the
    force limiter lowered from i2rt's 50 N, the gripper relaxed, the motor temperatures in every observation."""
    import sys
    from functools import partial

    from ora.environment.backends.yam import arms, cell_adapter

    built = []

    def get_yam_robot(**kwargs):
        robot = Robot()
        robot._gripper_force_limiter = types.SimpleNamespace(
            max_force=50.0, gripper_force_torque_map=partial(lambda **k: 0.0, gripper_force=50.0))
        built.append((kwargs, robot))
        return robot

    modules = {"i2rt": types.ModuleType("i2rt"), "i2rt.robots": types.ModuleType("i2rt.robots"),
               "i2rt.robots.get_robot": types.ModuleType("i2rt.robots.get_robot"),
               "i2rt.robots.utils": types.ModuleType("i2rt.robots.utils")}
    modules["i2rt.robots.get_robot"].get_yam_robot = get_yam_robot
    modules["i2rt.robots.utils"].GripperType = type("GripperType", (), {"NO_GRIPPER": "none",
                                                                        "__new__": lambda cls, value: value})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(cell_adapter.YamCellAdapter, "_quiet", lambda self, arms: None)

    held = cell_adapter.YamCellAdapter().take_over(["left"], mode="hold", gripper_limits={"left": [6.4, 1.1]})
    robot = held.robots["left"]
    assert robot._gripper_force_limiter.max_force == Y.GRIPPER_FORCE_LIMIT_N
    assert robot.temp_record_flag is True and grippers.is_relaxed(robot)
    assert built[-1][0]["gripper_limits_override"].tolist() == [6.4, 1.1]
    robot = arms.connect("can_right", gripper=False)
    assert built[-1][0]["gripper_type"] == "none" and robot.temp_record_flag is True
