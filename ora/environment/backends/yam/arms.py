"""The real YAM arms under i2rt: one way to take an arm over and one way to let the arms go.

Every program that takes the real arms over goes through ``connect``: an episode's worker
(``real_rig.py``), and the operator tools behind ``ora cell`` (``cell_adapter.py``), including the
gripper calibration. So each gets the same protection from the first tick:

- i2rt's gripper force limiter lowered from the 50 N ``get_yam_robot`` builds it with to
  ``GRIPPER_FORCE_LIMIT_N`` (``grippers.limit_force``);
- the gripper relaxed until a gripper command works it (``grippers.relax``);
- the motor temperatures in every observation (i2rt's ``temp_record_flag``), which the
  temperature guards and alerts read (``health.py``).

``release`` closes i2rt for those arms and then disables every motor on their buses.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.environment.controllers import yam_station as Y

from . import devices, grippers


def connect(channel: str, *, gripper: bool = True, gripper_limits: Sequence[float] | None = None,
            float_arm: bool = False) -> Any:
    """One arm under i2rt, its gripper protected (above). ``gripper`` False drives the arm motors only; with
    ``gripper`` and no ``gripper_limits`` i2rt finds the gripper's limits by opening and closing it once."""

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    if gripper:
        robot = get_yam_robot(channel=channel, gripper_type=GripperType(Y.GRIPPER_TYPE), zero_gravity_mode=float_arm,
                              gripper_limits_override=None if gripper_limits is None
                              else np.asarray(gripper_limits, float))
    else:
        robot = get_yam_robot(channel=channel, gripper_type=GripperType.NO_GRIPPER, zero_gravity_mode=float_arm)
    robot.temp_record_flag = True
    grippers.limit_force(robot, Y.GRIPPER_FORCE_LIMIT_N)
    grippers.relax(robot)
    return robot


def release(robots: Mapping[str, Any], buses: Mapping[str, str], say: Callable[[str], None] = print) -> None:
    """Close i2rt for every arm in ``robots``, then disable every motor on ``buses``; one failing close stops
    no other arm from being released."""

    for robot in robots.values():
        try:
            robot.close()
        except Exception as exc:  # noqa: BLE001 - every arm must still be released
            say(f"i2rt close failed: {exc}")
    devices.disable_all(buses)
