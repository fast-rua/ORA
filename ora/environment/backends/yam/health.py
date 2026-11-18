"""What is wrong with a held YAM arm right now, as ``ora.cell.alerts`` entries.

Every program that holds the arms (the play worker, the cell board's holder, a calibration
motion) watches them with ``arm_alerts`` and publishes the result, so the board, Pi's play
terminal and the agent see the same warnings. Warnings come well below the motors' own
protection, which stops the arm's control loop and leaves it uncontrolled.
"""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from ora.environment.controllers import yam_station as Y

from . import grippers

TEMPERATURE_WARN_C = 50.0
TEMPERATURE_CRITICAL_C = 60.0
GRIPPER_EFFORT_WARN_FRACTION = 0.3
GRIPPER_EFFORT_WARN_S = 5.0
ARM_CN = {"left": "左臂", "right": "右臂"}
MOTOR_CN = ("底座转向", "肩", "肘", "手腕 1", "手腕 2", "手腕 3", "夹爪")


def arm_alerts(arm: str, robot: Any, *, now: float, stamp: float, pushing_since: MutableMapping[str, float],
               stopped_text: str, may_relax: bool = True) -> list[dict[str, Any]]:
    """One arm's alerts: a stopped control loop (``stopped_text`` says what to do), a motor
    warming up, a gripper that keeps pushing. A gripper that is hot and pushing is relaxed
    here, unless ``may_relax`` is False (a motion is commanding it this tick)."""

    arm_cn = ARM_CN.get(arm, arm)
    if not getattr(robot.motor_chain, "running", True):
        return [{"code": "arms_not_controlled", "level": "critical", "arm": arm, "since_unix_s": stamp,
                 "text": f"{arm_cn}的控制循环停了：这只手臂现在不受程序控制，电机超时后断电"
                         f"（在起始位不会掉，别的位置会落下）。{stopped_text}"}]
    alerts: list[dict[str, Any]] = []
    obs = robot.get_observations()
    temperatures = np.maximum(np.asarray(obs.get("temp_mos", []), float), np.asarray(obs.get("temp_rotor", []), float))
    for index, value in enumerate(temperatures):
        name = MOTOR_CN[index] if index < len(MOTOR_CN) else f"电机 {index + 1}"
        if value >= TEMPERATURE_CRITICAL_C:
            alerts.append({"code": "motor_hot", "level": "critical", "arm": arm, "motor": name,
                           "value": float(value), "since_unix_s": stamp,
                           "text": f"{arm_cn}{name}电机 {value:.0f} °C，接近过温保护：请让手臂回到起始位并结束运行，等它冷却。"})
        elif value >= TEMPERATURE_WARN_C:
            alerts.append({"code": "motor_warm", "level": "warning", "arm": arm, "motor": name,
                           "value": float(value), "since_unix_s": stamp,
                           "text": f"{arm_cn}{name}电机 {value:.0f} °C，在升温。"})
    effort = abs(float(np.asarray(obs.get("gripper_eff", [0.0]), float)[0])) / Y.MOTOR_TORQUE_MAX_NM["DM4310"]
    if effort > GRIPPER_EFFORT_WARN_FRACTION:
        since = pushing_since.setdefault(arm, now)
        if now - since >= GRIPPER_EFFORT_WARN_S:
            alerts.append({"code": "gripper_pushing", "level": "warning", "arm": arm, "motor": "夹爪",
                           "value": effort, "since_unix_s": stamp,
                           "text": f"{arm_cn}夹爪持续出力 {now - since:.0f} 秒（{effort:.0%} 力矩）："
                                   "可能顶在止挡上或夹着东西，长时间会过热。"})
        if may_relax and len(temperatures) > 6 and temperatures[6] >= TEMPERATURE_CRITICAL_C:
            grippers.relax(robot)
    else:
        pushing_since.pop(arm, None)
    return alerts
