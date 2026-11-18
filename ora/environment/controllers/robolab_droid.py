"""Single-owner control and geometry constants for RoboLab's Droid Franka + Robotiq 2F-85.

Source: RoboLab ad45d4f974725d020f82c2b0d77d78533aeba2b3 (robolab/robots/droid.py,
DroidIKActionCfg) and the loaded Isaac Sim 5.1 stage, measured 2026-10-18: the
absolute differential-IK action targets the Robotiq ``base_link`` flange in the
robot-root frame; the gripper approaches along base_link +x and closes along
base_link y.

The public controlled point is a TCP 0.131 m along base_link +x: the centre of
the two fingertip pads (pad collision extents x in [0.1119, 0.1503] m at the
open reference). Its axes follow the parallel-jaw convention the tools use:
local z = approach (base_link +x), local x = closing line (base_link +y),
local y = base_link +z. The backend converts every TCP target to the flange.
"""

from __future__ import annotations

from math import pi, radians, sqrt

POSITION_FRAME = "robolab_robot_root"
FLANGE = "base_link"
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_OPEN_QPOS_RAD = 0.0
GRIPPER_CLOSED_QPOS_RAD = pi / 4
# Native finger action: 0 opens, 1 closes (the upstream binary joint target).
GRIPPER_OPEN_ACTION = 0.0
GRIPPER_CLOSE_ACTION = 1.0
TCP_OFFSET_FROM_FLANGE_M = (0.131, 0.0, 0.0)
# Rows are the TCP axes expressed in base_link: x = closing line (base_link +y),
# y = base_link +z, z = approach (base_link +x); right-handed (x cross y = z).
TCP_AXES_IN_FLANGE = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
FINGERTIP_PAD_PRIMS = ("left_inner_finger", "right_inner_finger")
MAXIMUM_APERTURE_M = 0.085  # inner fingertip-pad faces measured open (84.995 mm, 2026-10-18), the 2F-85 nominal

CONTROL_FREQ_HZ = 15
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
STEP_TRANSLATION_M = {"free_space": 0.03, "precision": 0.012, "contact_approach": 0.006}
STEP_ROTATION_RAD = {"free_space": 0.15, "precision": 0.08, "contact_approach": 0.05}
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.05
MAX_STEPS = 150
SETTLE_STEPS = 3
STALL_WINDOW_STEPS = 15
STALL_PROGRESS_M = 0.002
CONTACT_FORCE_STOP_N = 5.0
FREE_SPACE_LIFT_M = 0.05
# A free-space target within this horizontal distance is approached directly.
FREE_SPACE_DIRECT_RADIUS_M = 0.08
GRIPPER_STEPS = 20
# Jaw stall on the normalized opening 1 - finger_joint / (pi/4) (0 closed,
# 1 open): three ticks each moving less than 0.005 short of the target.
GRIPPER_STALL_WINDOW = 3
GRIPPER_STALL_STILL_STEP = 0.005
GRIPPER_ARRIVAL_TOLERANCE = 0.03
# Panda reach from the robot root (panda_link0) with margin: the bound while the shared IK cannot answer.
MAX_REACH_FROM_ROOT_M = 0.95
# A move this short stays on the servo, as a LIBERO move does on its OSC route.
LOCAL_TRANSLATION_M = 0.05
LOCAL_ROTATION_RAD = radians(15.0)
# The planned route's joint ceilings, at most what the drives follow (RoboLab's Droid limits 2.175 and 2.61 rad/s).
JOINT_SPEED_CEILING_RAD_S = 1.9
JOINT_ACCEL_CEILING_RAD_S2 = 2.5
# 20 s of native actions for one planned path.
PACE_BUDGET_TICKS = 300
# The drives (stiffness 400, damping 80) lag a moving target by about 0.2 s, so 0.6 rad of lag
# for a second of native actions is an arm held back by something, not a fast move.
PATH_BLOCKED_LAG_RAD = 0.6
PATH_BLOCKED_TICKS = 15
# The Robotiq flange in the planner's ``panda_hand`` frame, from RoboLab's USD: ``base_link`` sits 18.17 mm
# along ``panda_link8`` with its +x on the approach, and ``panda_hand`` is ``panda_link8`` turned -45 degrees.
HAND_FROM_FLANGE_TRANSLATION_M = (0.0, 0.0, 0.01817)
HAND_FROM_FLANGE_ROTATION = (
    (0.0, sqrt(0.5), sqrt(0.5)),
    (0.0, -sqrt(0.5), sqrt(0.5)),
    (1.0, 0.0, 0.0),
)


def published_limits() -> dict[str, float | int]:
    return {
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "max_steps": MAX_STEPS,
        "control_frequency_hz": CONTROL_FREQ_HZ,
        "maximum_aperture_m": MAXIMUM_APERTURE_M,
        "tcp_offset_from_flange_m": TCP_OFFSET_FROM_FLANGE_M[0],
        "gripper_steps": GRIPPER_STEPS,
    }
