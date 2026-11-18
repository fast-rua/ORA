"""The I2RT YAM station's ``CellAdapter``: what the real-cell kit (``ora/cell``) needs from it.

Passive reads go through DM DISABLE frames (``devices.py``); taking the arms over goes
through i2rt. With a gripper's measured limits the full arm-and-gripper model is used,
so gravity compensation carries the gripper; without them only the arm motors are
enabled. Either way a gripper gives no torque unless ``grip`` moves it (``grippers.py``):
held arms never keep a gripper pushing, and ``grip`` moves the jaws in force control at an episode's
speed, never as one step to the end of their travel. No i2rt gripper calibration ever runs except in
``calibrate_grippers``.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.cell.adapter import ArmReading
from ora.cell.errors import CellError
from ora.cell.occupancy import can_receivers, can_traffic
from ora.environment.benchmarks.yam import I2RT_REVISION, STATION_MODEL_SHA256
from ora.environment.controllers import yam_station as Y
from ora.environment.controllers.gripper_stall import gripper_stalled

from . import arms as yam_arms, devices, grippers, health
from .arms import release as _release_arms

CAMERA_MOUNTS = {"overhead": "fixed_world", "front": "fixed_world", "wrist_left": "end_effector",
                 "wrist_right": "end_effector"}
WRIST_CAMERAS = {"wrist_left": "left", "wrist_right": "right"}
JOINT_NAMES = ("底座转向", "肩", "肘", "手腕 1", "手腕 2", "手腕 3")
# Operator motions trip earlier than episode motions (Y.TRACKING_ERROR_STOP_RAD): they are
# slow, near rest, and a person may be guiding the arm.
OPERATOR_TRACKING_STOP_RAD = 0.15
# Shoulder (J2) and elbow (J3) offsets from rest for calibration poses, the same on both arms:
# the arms rise in front of their bases, never turn, and stay 0.61 m apart.
CALIBRATION_OFFSETS = [np.array((0.0, j2, j3, 0.0, 0.0, 0.0)) for j2, j3 in
                       ((0.45, 0.2), (0.3, 0.45), (0.45, 0.45), (0.3, 0.3), (0.2, 0.45), (0.45, 0.3))]
# Hand-eye planning (ora/cell/operator/handeye.py): from rest, shoulder and elbow raised
# bring the wrist out over the table, the inverse kinematics' seed; no joint is planned
# further than these from rest.
REACH_SEED_OFFSET = np.array((0.0, 0.7, 0.9, 0.0, 0.0, 0.0))
REACH_OFFSET_LIMIT_RAD = np.array((1.0, 1.5, 1.7, 1.3, 1.3, 1.6))
# What the open linear_4310 jaws hide of the wrist D405's 848x480 image: x0, y0, x1, y1.
WRIST_IMAGE_OCCLUDED = ((0, 330, 260, 480), (590, 330, 848, 480))


def _reading(states: Sequence[Any], limits: Sequence[float] | None) -> ArmReading:
    raw = devices.arm_reading(states, limits)
    return ArmReading(q=raw["q"], gripper_raw=raw["gripper_raw_rad"], gripper_opening=raw["gripper_opening"],
                      faults=raw["errors"], status=raw["status"], max_temperature_c=raw["max_temperature_c"],
                      torque_nm=raw["torque_nm"],
                      motors=[{"id": f"0x{s.motor_id:02X}", "status": s.error, "position_rad": s.position_rad,
                               "torque_nm": s.torque_nm, "temperature_mos_c": s.temperature_mos_c,
                               "temperature_rotor_c": s.temperature_rotor_c} for s in states])


class YamReader:
    def __init__(self, buses: Mapping[str, str], gripper_limits: Mapping[str, Sequence[float]]) -> None:
        self.limits = dict(gripper_limits)
        self.readers = {}
        try:
            for arm, channel in buses.items():
                self.readers[arm] = devices.PassiveArmReader(channel)
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, ArmReading]:
        return {arm: _reading(reader.read(), self.limits.get(arm)) for arm, reader in self.readers.items()}

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers = {}


class YamHeld:
    """Arms under i2rt. ``release`` closes i2rt and then disables every motor of those arms."""

    def __init__(self, robots: Mapping[str, Any], buses: Mapping[str, str]) -> None:
        self.robots = dict(robots)
        self.buses = {arm: buses[arm] for arm in robots}
        self.arms = tuple(self.robots)
        self._pushing_since: dict[str, float] = {}
        for robot in self.robots.values():
            grippers.relax(robot)

    def joints(self) -> dict[str, np.ndarray]:
        return {arm: np.asarray(robot.get_observations()["joint_pos"], float) for arm, robot in self.robots.items()}

    def command(self, targets: Mapping[str, np.ndarray]) -> None:
        for arm, q in targets.items():
            robot = self.robots[arm]
            q = np.asarray(q, float)
            if getattr(robot, "_gripper_index", None) is not None:
                # the jaws' target where they are; a relaxed gripper gives no torque anyway
                jaws = float(np.clip(robot.get_observations()["gripper_pos"][0], *Y.GRIPPER_HOLD_RANGE))
                q = np.append(q, jaws)
            robot.command_joint_pos(q)

    def grip(self, openings: Mapping[str, float], ticks: int = Y.GRIPPER_TICKS) -> dict[str, float]:
        """Grippers toward ``openings`` (0 closed .. 1 open), arms held where they are; returns the measured
        openings. The jaws move as an episode's do: in force control (never above ``GRIPPER_FORCE_LIMIT_N``),
        the command ramped (``grippers.ramp``), each gripper relaxed again as soon as it arrives or stalls, every
        one after ``ticks`` control ticks at the latest. Arms without a gripper model are skipped."""

        moving = {arm: self.robots[arm] for arm in openings if grippers.has_gripper(self.robots[arm])}
        hold = {arm: np.asarray(robot.get_observations()["joint_pos"], float) for arm, robot in moving.items()}
        opening = lambda arm: float(np.clip(moving[arm].get_observations()["gripper_pos"][0], 0.0, 1.0))
        goal = {arm: float(np.clip(openings[arm], *Y.GRIPPER_HOLD_RANGE)) for arm in moving}
        traces = {arm: [opening(arm)] for arm in moving}
        closing = {arm: goal[arm] < traces[arm][0] for arm in moving}
        commanded = {arm: traces[arm][0] * Y.FINGER_OPEN_M for arm in moving}
        working = set(moving)
        try:
            for robot in moving.values():
                grippers.take_force_control(robot)
            for _ in range(ticks):
                if not working:
                    break
                for arm in working:
                    measured = traces[arm][-1]
                    commanded[arm] = grippers.ramp(commanded[arm], measured * Y.FINGER_OPEN_M,
                                                   goal[arm] * Y.FINGER_OPEN_M, closing=closing[arm])
                    target = commanded[arm] / Y.FINGER_OPEN_M
                    moving[arm].command_joint_pos(np.append(hold[arm], target))
                    grippers.apply_force(moving[arm], target, measured)
                time.sleep(Y.CONTROL_PERIOD_S)
                for arm in list(working):
                    traces[arm].append(opening(arm))
                    if (grippers.GRIPPER_STALL.arrived(traces[arm][-1], goal[arm])
                            or gripper_stalled(traces[arm], target=goal[arm], rule=grippers.GRIPPER_STALL)):
                        grippers.relax(moving[arm])
                        working.discard(arm)
        finally:
            for robot in moving.values():
                grippers.relax(robot)
        return {arm: trace[-1] for arm, trace in traces.items()}

    def alerts(self) -> list[dict[str, Any]]:
        """What is wrong with the held arms now (``ora.cell.alerts`` entries)."""

        now, stamp, out = time.monotonic(), time.time(), []
        for arm, robot in self.robots.items():
            out += health.arm_alerts(arm, robot, now=now, stamp=stamp, pushing_since=self._pushing_since,
                                     stopped_text="请在看板上点「回到起始位」或「全部断电」，查明原因再继续。")
        return out

    def alive(self) -> dict[str, bool]:
        return {arm: bool(getattr(robot.motor_chain, "running", True)) for arm, robot in self.robots.items()}

    def set_mode(self, mode: str) -> None:
        if mode == "float":
            for robot in self.robots.values():
                robot.enter_gravity_comp_idle()
        elif mode == "hold":
            self.command(self.joints())
        else:
            raise CellError(f"unknown hold mode {mode!r}")

    def release(self) -> None:
        _release_arms(self.robots, self.buses, say=lambda text: print(text, file=sys.stderr))
        self.robots = {}


class YamCellAdapter:
    environment = "yam"
    model = Y.STATION_MODEL
    arms = Y.ARMS
    camera_mounts = CAMERA_MOUNTS
    wrist_cameras = WRIST_CAMERAS
    joint_names = JOINT_NAMES
    control_hz = float(Y.CONTROL_FREQ_HZ)
    tracking_stop_rad = OPERATOR_TRACKING_STOP_RAD
    finger_open_m = Y.FINGER_OPEN_M
    foreign_controller_markers = devices.MOTION_PROCESS_MARKERS
    reach_seed_offset = REACH_SEED_OFFSET
    reach_offset_limit_rad = REACH_OFFSET_LIMIT_RAD
    wrist_image_occluded = WRIST_IMAGE_OCCLUDED

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.buses = dict(config.buses) if config is not None else dict(Y.CAN_CHANNELS)

    def open_reader(self, gripper_limits: Mapping[str, Sequence[float]]) -> YamReader:
        return YamReader(self.buses, gripper_limits)

    def _quiet(self, arms: Sequence[str]) -> None:
        for arm in arms:
            channel = self.buses[arm]
            if can_receivers(channel) or can_traffic(channel, 0.3):
                raise CellError(f"{channel} is in use by another program: refusing to take over")

    def take_over(self, arms: Sequence[str], *, mode: str,
                  gripper_limits: Mapping[str, Sequence[float]]) -> YamHeld:
        self._quiet(arms)
        robots = {}
        try:
            for arm in arms:
                # Without measured limits the arm motors only: i2rt would otherwise run its gripper calibration.
                robots[arm] = yam_arms.connect(self.buses[arm], gripper=arm in gripper_limits,
                                               gripper_limits=gripper_limits.get(arm), float_arm=mode == "float")
        except BaseException:
            YamHeld(robots, self.buses).release()
            raise
        held = YamHeld(robots, self.buses)   # every gripper relaxed
        if mode == "hold":
            held.command(held.joints())
        return held

    def disable_all(self) -> None:
        devices.disable_all(self.buses)

    def calibrate_grippers(self, arms: Sequence[str], should_stop: Callable[[], bool]) -> dict[str, list[float]]:
        """i2rt finds each gripper's [closed, open] motor limits by opening and closing it once;
        the arm holds its pose meanwhile and is released after."""

        found = {}
        for arm in arms:
            if should_stop():
                break
            self._quiet([arm])
            print(f"calibrating the {arm} gripper (it opens and closes once)", flush=True)
            robot = yam_arms.connect(self.buses[arm])     # no limits given: i2rt finds them
            try:
                found[arm] = [float(v) for v in robot._gripper_limits]
            finally:
                YamHeld({arm: robot}, self.buses).release()
        return found

    def kinematics(self, table: Mapping[str, Any] | None = None):
        from .kinematics import StationKinematics
        from .scene import build_cell_model

        return StationKinematics(build_cell_model(table=table))

    def calibration_offsets(self) -> list[np.ndarray]:
        return [offset.copy() for offset in CALIBRATION_OFFSETS]

    def facts(self) -> list[tuple[str, str]]:
        buses = " · ".join(f"{arm} {channel}" for arm, channel in self.buses.items())
        return [
            ("机型", f"{Y.STATION_MODEL}（2× YAM v1，linear_4310 夹爪，腕部 D405）"),
            ("i2rt 版本锁", f"v1.4.0 {I2RT_REVISION[:12]}；站点 MJCF sha256 {STATION_MODEL_SHA256[:12]}…"),
            ("坐标系", "yam_station_world：两基座中点、基座平面为原点，+x 向前，+y 向左；"
                     "左基座 (0, +0.305, 0)，右基座 (0, -0.305, 0)"),
            ("CAN", f"{buses}，1 Mbit/s；电机 0x01–0x03 DM4340，0x04–0x06 DM4310，夹爪 0x07 DM4310"),
            ("禁发帧", "0xFE（写编码器零点）在本后端无法构造；被动读只发 0xFD（失能并回传状态）"),
            ("控制", f"{Y.CONTROL_FREQ_HZ} Hz 关节位置；关节限速 {Y.MAX_JOINT_SPEED_RAD_S} rad/s；"
                   f"跟踪误差 > {Y.TRACKING_ERROR_STOP_RAD} rad 或力矩 > {int(Y.JOINT_EFFORT_STOP_FRACTION * 100)}% "
                   f"持续 {Y.TRACKING_ERROR_GRACE_TICKS} 拍即停；电机 ≥ {Y.MOTOR_TEMPERATURE_STOP_C:.0f} °C 终止"),
            ("rig 模式", "sim 仿真；read 只读（电机从不使能，每个动作只返回计划）；on 真动（每个动作操作员批准，"
                       "RunSpec 须 operator_present: true）"),
        ]


def make_adapter(config: Any = None) -> YamCellAdapter:
    return YamCellAdapter(config)
