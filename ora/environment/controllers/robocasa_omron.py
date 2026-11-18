"""Single-owner control constants for RoboCasa's PandaOmron mobile manipulator.

Values were read from the pinned robosuite ``default_pandaomron.json``
composite controller and the loaded MuJoCo model (2026-10-17); the
arm part of that controller is replaced by :data:`ARM_CONTROLLER`. The
RoboCasa worker, the motion loop and the published limits all import them
from here; nothing else restates them.
"""

from __future__ import annotations

import math

# HYBRID_MOBILE_BASE action vector with the arm on ARM_CONTROLLER: arm joint
# deltas [0:7], gripper [7], base velocity [8:11] as (forward, side, yaw) in
# the current base frame, torso delta [11], and the mode bit [12] (> 0 drives
# the base; the arm holds its joint targets either way).
ACTION_DIM = 13
ARM_SLICE = (0, 7)
GRIPPER_INDEX = 7
BASE_SLICE = (8, 11)
TORSO_INDEX = 11
MODE_INDEX = 12
MODE_ARM = -1.0
MODE_BASE = 1.0
# The arm part of the composite controller: robosuite JOINT_POSITION in place
# of the default OSC_POSE delta (2026-10-18: the OSC servo stalled for good once
# panda joint 4 reached its straight-arm limit). Its input is a delta: robosuite
# sets the goal to the measured joints plus input * ARM_DELTA_LIMIT_RAD, so the
# IO layer sends (target - measured) / ARM_DELTA_LIMIT_RAD and the goal is the
# target itself, while RoboCasa's all-zero settle action at reset still holds
# the arm (an absolute input would read it as "drive every joint to 0"). Measured
# with kp 400: a stream advancing 0.05 rad per tick trails by at most
# 0.09 rad and settles within 0.0001 rad; a base drive leaves the arm joints
# within 0.0002 rad.
ARM_DELTA_LIMIT_RAD = 0.5
ARM_CONTROLLER = {
    "type": "JOINT_POSITION", "input_type": "delta", "input_max": 1.0, "input_min": -1.0,
    "output_max": ARM_DELTA_LIMIT_RAD, "output_min": -ARM_DELTA_LIMIT_RAD, "kp": 400, "damping_ratio": 1,
    "impedance_mode": "fixed", "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10], "qpos_limits": None,
    "interpolation": None,
}

# The arm's mount on the torso: reach is measured from here.
ARM_BASE_SITE = "robot0_right_center"
# The arm's base link and hand: the shared planner's base and tool frame.
ARM_BASE_BODY = "robot0_link0"
HAND_BODY = "robot0_right_hand"
GRIP_SITE = "gripper0_right_grip_site"
FINGER_JOINTS = ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
ARM_JOINTS = tuple(f"robot0_joint{index}" for index in range(1, 8))
BASE_BODY = "mobilebase0_base"
BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)
TORSO_JOINT = "mobilebase0_joint_torso_height"
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")

# The preflight check that binds a RoboCasa worker's planner: its environment is
# allowlisted, so it reads the interpreter and source root from the run's record.
MOTION_PLANNER_PREFLIGHT_CHECK = "shared_motion_planner"
# The planner worker's scene: a counter is no floor for a hand in front of the drawer below it.
MOTION_PLANNER_ENVIRONMENT = {"ORA_PLANNER_SCENE": "kitchen"}

CONTROL_FREQ_HZ = 20
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
# Gripper: -1 opens, +1 closes (robosuite GRIP convention). Panda finger
# travel 0.04 m each, so the jaw separation spans 0 to 80 mm.
GRIPPER_OPEN_COMMAND = -1.0
GRIPPER_CLOSE_COMMAND = 1.0
PANDA_FINGER_OPEN_M = 0.04
MAXIMUM_APERTURE_M = 2 * PANDA_FINGER_OPEN_M
TORSO_RANGE_M = (0.0, 0.34)
# Torso lift: a delta joint-position command of at most 0.05 m per control tick
# (robosuite JOINT_POSITION defaults). Admission frees it only when the arm alone
# cannot reach a target; it then moves first and holds its new height. Under
# gravity it settles about 3 mm short (probe, 2026-10-18); the arm path
# was solved for the planned height and the arrival check reads the measured
# grip site, so the lift only has to come to rest near its target.
TORSO_MAX_DELTA_PER_STEP_M = 0.05
TORSO_TOLERANCE_M = 0.01
TORSO_STILL_STEP_M = 0.001
TORSO_MIN_CHANGE_M = 0.01
TORSO_MAX_STEPS = 30
TORSO_SETTLE_STEPS = 3
# Candidate heights the shared IK is asked about when the arm alone cannot reach.
TORSO_SEARCH_STEP_M = 0.04

# Arm motion: admission computes a joint path and every posture along it is
# checked; the executor streams that path as joint targets. Per control tick
# the target advances at most ARM_JOINT_STEP_RAD per joint and moves the grip
# site at most ARM_STEP_TRANSLATION_M and ARM_STEP_ROTATION_RAD, by interaction.
ARM_JOINT_STEP_RAD = {
    "free_space": 0.05,
    "precision": 0.025,
    "contact_approach": 0.012,
}
ARM_STEP_TRANSLATION_M = {
    "free_space": 0.03,
    "precision": 0.012,
    "contact_approach": 0.008,
}
ARM_STEP_ROTATION_RAD = {
    "free_space": 0.15,
    "precision": 0.08,
    "contact_approach": 0.06,
}
ARM_POSITION_TOLERANCE_M = 0.005
ARM_ORIENTATION_TOLERANCE_RAD = 0.05
ARM_MAX_STEPS = 150
# A pose target farther than this from the arm's mount (ARM_BASE_SITE) at the
# current base pose is refused before any motion: the base has to move first.
ARM_MAX_REACH_FROM_SHOULDER_M = 1.05
# Arrival: the measured joints within ARM_JOINT_SETTLE_RAD of the path's last
# posture for ARM_SETTLE_STEPS consecutive ticks, then the grip-site check.
ARM_JOINT_SETTLE_RAD = 0.005
ARM_SETTLE_STEPS = 3
# Measured joints trailing the streamed target by more than this for
# ARM_BLOCKED_STEPS consecutive ticks mean the arm is pressed against something:
# the action ends as blocked and the arm holds where it stopped.
ARM_BLOCKED_LAG_RAD = 0.2
ARM_BLOCKED_STEPS = 3
# Joints that have not settled this many ticks after the last target was sent
# are held back too (2026-10-18: a hand holding a drawer handle trailed by less
# than ARM_BLOCKED_LAG_RAD and spent the whole 150-tick budget waiting).
ARM_SETTLE_MAX_STEPS = 20
# Joints at rest short of the path's end end that wait sooner: ARM_STILL_STEPS
# consecutive ticks against the last target with every joint moving less than
# ARM_STILL_STEP_RAD per tick while one is still more than ARM_JOINT_SETTLE_RAD
# from it (the rule of motion_progress.JointBlockedMonitor). 2026-11-06, RoboCasa365 run audit
# (111 runs): 581 of 669 settle waits that ran out had been at rest this way,
# then waited a median 12 ticks more.
ARM_STILL_STEP_RAD = 0.001
ARM_STILL_STEPS = 3
CONTACT_FORCE_STOP_N = 15.0
# free_space follows the shared planner's joint path; precision moves along the
# straight line unless it turns the hand more than this; contact_approach always
# moves along the straight line.
JOINT_PATH_MIN_ROTATION_DEG = 15.0
JOINT_PATH_MIN_ROTATION_RAD = math.radians(JOINT_PATH_MIN_ROTATION_DEG)
# Joint-space paths are sampled (checked and streamed) at this joint stride.
JOINT_PATH_CHECK_STRIDE_RAD = 0.02
# Straight-line paths are solved by IK at this spacing, each solution seeded
# from the previous one, and the joints interpolated between them; a jump larger
# than LINE_MAX_JOINT_JUMP_RAD between neighbours (a change of IK branch) refuses
# the line.
LINE_STEP_M = 0.02
LINE_STEP_RAD = 0.2
LINE_MAX_JOINT_JUMP_RAD = 0.25
# How far a checked link may lie inside the public depth world along a line: the
# world's own 1 cm voxel.
LINE_WORLD_TOLERANCE_M = 0.01
GRIPPER_STEPS = 25
# Jaw stall on the normalized opening (0 closed, 1 open: mean finger travel
# over 0.04 m): three ticks each moving less than 0.005 short of the target.
GRIPPER_STALL_WINDOW_STEPS = 3
GRIPPER_STALL_STILL_STEP = 0.005
GRIPPER_ARRIVAL_TOLERANCE = 0.03

# Mobile base: velocity limits in the base frame, arrival tolerances and budget.
# 2026-10-18: raised from 0.30 m/s and 0.60 rad/s (a slide command of 1.0 moves
# the base about 0.75 m/s, so 0.60 m/s plus the deadband stays inside the clip).
BASE_MAX_LINEAR_SPEED_M_S = 0.60
BASE_MAX_YAW_RATE_RAD_S = 1.00
BASE_MAX_YAW_RATE_DEG_S = math.degrees(BASE_MAX_YAW_RATE_RAD_S)  # as the help states it
BASE_LINEAR_GAIN = 3.0
BASE_YAW_GAIN = 3.0
BASE_POSITION_TOLERANCE_M = 0.03
BASE_YAW_TOLERANCE_RAD = 0.04
BASE_MAX_STEPS = 400
BASE_SETTLE_STEPS = 4
# Stall: every tick, outside the arrival tolerance, the servo's progress measure
# (anchor distance plus 0.3 m per radian of heading error) is compared with its
# value BASE_STALL_WINDOW_STEPS ticks earlier; less than BASE_STALL_PROGRESS_M
# of progress ends the move. 2026-11-06, RoboCasa365 run audit: the earlier check, 0.01 m once
# per 30-tick window, let 80 stalls run 33 to 153 ticks (typically 63).
BASE_STALL_WINDOW_STEPS = 10
BASE_STALL_PROGRESS_M = 0.005
BASE_MAX_TRAVEL_M = 6.0
# A brushing contact (a few newtons) is tolerated; a sustained push stops the base.
BASE_CONTACT_STOP_N = 15.0
# Measured 2026-10-18: a slide-joint velocity command c moves the
# base at about c - 0.25 m/s (joint friction against kp 50), so the servo adds
# this offset along each slide axis it drives.
BASE_SLIDE_DEADBAND_COMMAND = 0.25
# The yaw hinge sits 0.21 m behind the base body origin (MJCF jnt_pos); the
# servo drives that anchor so translation and rotation decouple.
BASE_YAW_JOINT = "mobilebase0_joint_mobile_yaw"

# The base's footprint, for move_base's footprint check and the base-placement
# search (controllers/base_placement.py), read from robosuite's
# omron_mobile_base.xml (2026-11-06): the collision box pedestal_feet_col
# (centre (-0.20, 0, 0.192) in the base body frame, half-sizes 0.35, 0.25 and
# 0.19 m), and the column under the arm, the fixed_support and lifted support
# cylinders (radius 0.12 m about x = -0.05 m) from 0.30 m up to the arm mount.
# The mount (robot0_link0, the planner's base) stands on the base origin's
# vertical, 0.70 m above the floor plus the torso height. Each box is its
# (x, y, z) ranges in the base frame; the column's top rises with the lift.
BASE_FOOTPRINT_BOXES_M = (
    ((-0.55, 0.15), (-0.25, 0.25), (0.0, 0.382)),
    ((-0.17, 0.07), (-0.12, 0.12), (0.30, 0.70)),
)
BASE_FOOTPRINT_LIFTING_BOX = 1
# The robot's own depth, taken out of what the footprint and a world seen from
# another stand may meet, is the planner's own model of the arm at its measured
# joints (backends/robocasa/shared_motion.own_body: its collision spheres,
# controllers/robot_spheres.py, grown by the world's 3 cm robot margin, the
# hand's and the base link's boxes, a held load's spheres). Until 2026-11-06 it
# was capsules along the link origins, 0.15 m for the arm and 0.12 m for the
# hand, which took out the scene next to the hand as well (RoboCasa smoke
# rc-ab-952eeed23: all 2,970 points within 5 cm of one refused target).
# Depth points at or below this height are the kitchen floor.
FLOOR_HEIGHT_M = 0.03
# The Panda's shoulder (joint 2) above the arm mount (panda robot.xml link1).
ARM_SHOULDER_ABOVE_MOUNT_M = 0.333
# How many of the torso heights the arm reaches the target from (nearest first)
# the lift's admission tries the whole move from before it refuses.
TORSO_ADMIT_HEIGHTS = 3


def published_limits() -> dict[str, float | int | str]:
    """Numbers the RoboCasa tool help may cite; each has its owner here."""

    return {
        "arm_position_tolerance_m": ARM_POSITION_TOLERANCE_M,
        "arm_orientation_tolerance_rad": ARM_ORIENTATION_TOLERANCE_RAD,
        "arm_max_steps": ARM_MAX_STEPS,
        "arm_controller_type": ARM_CONTROLLER["type"],
        "arm_controller_kp": ARM_CONTROLLER["kp"],
        "control_frequency_hz": CONTROL_FREQ_HZ,
        "base_position_tolerance_m": BASE_POSITION_TOLERANCE_M,
        "base_yaw_tolerance_rad": BASE_YAW_TOLERANCE_RAD,
        "base_max_linear_speed_m_s": BASE_MAX_LINEAR_SPEED_M_S,
        "base_max_yaw_rate_rad_s": BASE_MAX_YAW_RATE_RAD_S,
        "base_max_travel_m": BASE_MAX_TRAVEL_M,
        "gripper_steps": GRIPPER_STEPS,
    }
