"""The YAM arms' DM motors read without the motion layer.

``PassiveArmReader`` reads both arms' joints and grippers with the DM DISABLE frame (0xFD),
the only frame that returns a motor's state without enabling it; it is safe with the arms
in any state. Frames listed in ``DM_FRAMES_FORBIDDEN`` (0xFE writes the encoder zero)
cannot be built here. Whether the buses are free to use at all is ``ora.cell.occupancy``'s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers import yam_station as Y

from .common import YamError

# Command lines of programs that drive YAM arms: i2rt's own tools, GELLO-style
# teleoperation and the ASPIRE station session started with motion allowed.
MOTION_PROCESS_MARKERS = ("--allow-motion", "i2rt.robots", "minimum_gello", "gello", "yam_control",
                          "i2rt/scripts", "teleop")
# DM MIT feedback ranges (i2rt motor_drivers/utils.py, MotorType.get_motor_constants).
DM_RANGES = {"DM4340": {"position": 12.5, "velocity": 10.0, "torque": 28.0},
             "DM4310": {"position": 12.5, "velocity": 30.0, "torque": 10.0}}
DM_ERRORS = {0x0: "disabled", 0x1: "normal", 0x8: "over_voltage", 0x9: "under_voltage", 0xA: "over_current",
             0xB: "mosfet_over_temperature", 0xC: "motor_over_temperature", 0xD: "loss_communication",
             0xE: "overload"}
DM_FAULTS = frozenset(name for code, name in DM_ERRORS.items() if code >= 0x8)
REPLY_ID_OFFSET = 0x10          # i2rt ReceiveMode.p16: a motor answers on its id + 16


# ---------------------------------------------------------------- passive read
@dataclass(frozen=True)
class MotorState:
    motor_id: int
    error: str
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temperature_mos_c: float
    temperature_rotor_c: float


def special_frame(code: int) -> list[int]:
    """The 8-byte DM special frame ``FF..FF code``; forbidden codes cannot be built."""

    if code in Y.DM_FRAMES_FORBIDDEN:
        raise YamError(f"DM frame 0x{code:02X} is forbidden in this backend")
    if code != Y.DM_FRAME_DISABLE:
        raise YamError(f"the passive reader sends DISABLE (0x{Y.DM_FRAME_DISABLE:02X}) only, not 0x{code:02X}")
    return [0xFF] * 7 + [code]


def _from_uint(value: int, limit: float, bits: int) -> float:
    return value * (2 * limit) / ((1 << bits) - 1) - limit


def parse_feedback(data: Sequence[int], motor_id: int, motor_type: str) -> MotorState:
    """A DM MIT feedback frame (i2rt ``parse_recv_message``) as physical values."""

    ranges = DM_RANGES[motor_type]
    position = (data[1] << 8) | data[2]
    velocity = (data[3] << 4) | (data[4] >> 4)
    torque = ((data[4] & 0xF) << 8) | data[5]
    return MotorState(motor_id, DM_ERRORS.get((data[0] & 0xF0) >> 4, f"unknown_{data[0] >> 4:x}"),
                      _from_uint(position, ranges["position"], 16), _from_uint(velocity, ranges["velocity"], 12),
                      _from_uint(torque, ranges["torque"], 12), float(data[6]), float(data[7]))


def unwrap_joint(position: float) -> float:
    """i2rt's start-up wrap-around offset: a joint is reported inside (-pi, pi]."""

    return float(position - round(position / (2 * np.pi)) * 2 * np.pi)


def gripper_opening(raw: float, limits: Sequence[float] | None) -> float | None:
    """0 closed .. 1 open from the gripper motor angle and its [closed, open] limits."""

    if not limits:
        return None
    closed, opened = float(limits[0]), float(limits[1])
    return float(np.clip((raw - closed) / (opened - closed), 0.0, 1.0))


class PassiveArmReader:
    """One arm's motors read with DISABLE frames: never enabled, never written."""

    TIMEOUT_S = 0.05
    RETRIES = 3

    def __init__(self, channel: str) -> None:
        import can

        self.channel = channel
        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self.motors = (*Y.ARM_MOTORS, Y.GRIPPER_MOTOR)
        self.replies = {motor_id + REPLY_ID_OFFSET for motor_id, _ in self.motors}

    def _query(self, motor_id: int, motor_type: str, retries: int = RETRIES) -> MotorState:
        import can

        frame = can.Message(arbitration_id=motor_id, data=special_frame(Y.DM_FRAME_DISABLE), is_extended_id=False)
        for _ in range(retries):
            self.bus.send(frame)
            end = time.monotonic() + self.TIMEOUT_S
            while time.monotonic() < end:
                reply = self.bus.recv(timeout=self.TIMEOUT_S)
                if reply is not None and reply.arbitration_id not in self.replies:
                    # Only motor replies belong on a bus this reader holds: anything else is
                    # another controller, and its arms must not be sent another DISABLE.
                    raise YamError(f"{self.channel} carries frame 0x{reply.arbitration_id:03X} from another "
                                   "program: stopped reading")
                if reply is not None and reply.arbitration_id == motor_id + REPLY_ID_OFFSET and len(reply.data) == 8:
                    return parse_feedback(reply.data, motor_id, motor_type)
        raise YamError(f"motor 0x{motor_id:02X} on {self.channel} does not answer: is the arm powered?")

    def read(self) -> list[MotorState]:
        # One frame to the first motor before any other: an unpowered arm acknowledges
        # nothing, and every unanswered frame waits in the adapter's queue until power-on.
        (first_id, first_type), *rest = self.motors
        return [self._query(first_id, first_type, retries=1), *(self._query(i, t) for i, t in rest)]

    def close(self) -> None:
        self.bus.shutdown()


def arm_reading(states: Sequence[MotorState], gripper_limits: Sequence[float] | None) -> dict[str, Any]:
    """Joint angles (unwrapped like i2rt does), gripper opening and the motors' health."""

    arm, gripper = states[:Y.ARM_JOINT_COUNT], states[Y.ARM_JOINT_COUNT]
    return {"q": np.array([unwrap_joint(s.position_rad) for s in arm]),
            "gripper_raw_rad": gripper.position_rad,
            "gripper_opening": gripper_opening(gripper.position_rad, gripper_limits),
            "errors": {f"0x{s.motor_id:02X}": s.error for s in states if s.error in DM_FAULTS},
            # Codes outside the DM table. The cell's DM4310s answer a disabled DISABLE with
            # 3 (2026-10-28, all eight, cold and torque-free), which no document defines.
            "status": {f"0x{s.motor_id:02X}": s.error for s in states
                       if s.error not in DM_FAULTS and s.error not in {"disabled", "normal"}},
            "max_temperature_c": max(max(s.temperature_mos_c, s.temperature_rotor_c) for s in states),
            "torque_nm": [s.torque_nm for s in states]}


def disable_all(channels: Mapping[str, str]) -> None:
    """Send DISABLE to every motor of each arm: the arms go limp. Used after release only."""

    for channel in channels.values():
        reader = PassiveArmReader(channel)
        try:
            for motor_id, motor_type in reader.motors:
                try:
                    reader._query(motor_id, motor_type)
                except YamError:
                    pass
        finally:
            reader.close()
