"""Single-owner control and safety constants for the bimanual I2RT YAM station.

Source: i2rt fe78bae (v1.4.0), ``i2rt/robot_models/station/yam_station_linear_4310_d405``:
two YAM v1 arms with linear_4310 grippers and wrist RealSense D405 cameras, bases
0.61 m apart along -y with a shared orientation. ``robots/config/yam_v1.yml``:
motors 0x01-0x03 DM4340, 0x04-0x06 DM4310, all directions +1 and zero offsets, so a
joint angle is the motor position; the gripper motor is 0x07.

The public frame ``yam_station_world`` sits on the base plane midway between the two
arm bases: +x forward, +y toward the left arm, +z up. Its origin is the station
model's ``left_base`` shifted by -0.305 m in y, so the two bases sit at y = +/-0.305.

The controlled point of each arm is the station model's ``tcp_{arm}`` site: the centre
of the two fingertip facets, 144.65 mm along the flange +z. The site's z is the
approach and its y the fingers' slide axis (measured on the model, 2026-10-28), so the
public TCP turns it +90 deg about z: public x = site y = closing line, public z = site
z = approach, the parallel-jaw convention the tools use.

The finger joints ``{arm}_joint7``/``8`` are 0 when the fingertips meet at the TCP and
0.04695 m fully open; each finger slides that far, so the open aperture is twice it.

Speeds and envelopes are deliberately conservative: the same numbers drive the real
arms, where a person stands beside the cell.
"""

from __future__ import annotations

POSITION_FRAME = "yam_station_world"
ARMS = ("left", "right")
ARM_JOINT_COUNT = 6
STATION_MODEL = "yam_station_linear_4310_d405"
GRIPPER_TYPE = "linear_4310"
# Translation from the station model frame (left_base) to the public world frame.
STATION_TO_WORLD_XYZ_M = (0.0, 0.305, 0.0)
BASE_IN_WORLD_XYZ_M = {"left": (0.0, 0.305, 0.0), "right": (0.0, -0.305, 0.0)}
CAN_CHANNELS = {"left": "can_left", "right": "can_right"}
TCP_SITES = {"left": "tcp_left", "right": "tcp_right"}
# Public TCP = site * Rz(+90 deg): rows are the public axes expressed in the site frame.
TCP_AXES_IN_SITE = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
FINGER_JOINTS = {"left": ("left_joint7", "left_joint8"), "right": ("right_joint7", "right_joint8")}
FINGER_CLOSED_M = 0.0
FINGER_OPEN_M = 0.04695
MAXIMUM_APERTURE_M = 2 * FINGER_OPEN_M
WRIST_CAMERA_BODIES = {"left": "left_camera", "right": "right_camera"}

# Motor map, read-only: a joint angle is the motor position (directions +1, offsets 0).
ARM_MOTORS = ((0x01, "DM4340"), (0x02, "DM4340"), (0x03, "DM4340"),
              (0x04, "DM4310"), (0x05, "DM4310"), (0x06, "DM4310"))
GRIPPER_MOTOR = (0x07, "DM4310")
# DM MIT special frames. Only DISABLE may be sent by a read-only reader: it returns the
# motor state and leaves an already disabled motor disabled. SAVE_ZERO rewrites the
# encoder zero and must never be sent by this backend.
DM_FRAME_DISABLE = 0xFD
DM_FRAMES_FORBIDDEN = (0xFE,)

CONTROL_FREQ_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
# Smooth arm commands on a real cell (``runtime.smooth_arm_commands``, real_rig.py): between two control ticks the
# arm's position target is streamed along the line between them at this rate, with that line's velocity as the
# feed-forward, instead of one position step per tick with a zero velocity target.
ARM_COMMAND_STREAM_HZ = 250.0
LINEAR_SPEED_M_S = {"free_space": 0.12, "precision": 0.05, "contact_approach": 0.025}
ANGULAR_SPEED_RAD_S = {"free_space": 0.6, "precision": 0.3, "contact_approach": 0.2}
MAX_JOINT_SPEED_RAD_S = 0.6
# The shared timing (runtime.motion_timing, off by default) keeps the three caps above and adds what the station's
# own timing lacks: every cap is reached from rest in TIMING_RAMP_S, at the controlled point and at each joint.
# Declared here, not measured on the arms: 0.25 s is the ramp of a numeric speed request (motion_speed.RAMP_S).
TIMING_RAMP_S = 0.25
MAX_JOINT_ACCEL_RAD_S2 = MAX_JOINT_SPEED_RAD_S / TIMING_RAMP_S
# One motion's longest timed path, two minutes of ticks: no admitted target needs it at the slowest cap.
TIMING_BUDGET_TICKS = CONTROL_FREQ_HZ * 120
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.05
IK_POSITION_TOLERANCE_M = 0.001
IK_ORIENTATION_TOLERANCE_RAD = 0.01
SETTLE_TICKS = 5
MAX_SETTLE_TICKS = 100
FREE_SPACE_LIFT_M = 0.06
# A free-space target within this horizontal distance is approached directly.
FREE_SPACE_DIRECT_RADIUS_M = 0.06

# Envelopes, in yam_station_world. The table height is a per-cell calibration value.
WORKSPACE_X_M = (0.0, 0.65)
WORKSPACE_Y_M = (-0.65, 0.65)
WORKSPACE_Z_MAX_M = 0.60
MAX_REACH_FROM_BASE_M = 0.62
TABLE_CLEARANCE_M = {"free_space": 0.02, "precision": 0.008, "contact_approach": 0.0}
INTER_ARM_MIN_DISTANCE_M = 0.06

# Real-arm guards.
TRACKING_ERROR_STOP_RAD = 0.2
TRACKING_ERROR_GRACE_TICKS = 10
MOTOR_TEMPERATURE_STOP_C = 70.0
# A joint whose measured torque stays above this share of its motor's MIT torque range
# for the grace ticks is pushing against something: the motion stops and holds.
JOINT_EFFORT_STOP_FRACTION = 0.8
MOTOR_TORQUE_MAX_NM = {"DM4340": 28.0, "DM4310": 10.0}
# A joint whose measured torque exceeds i2rt's own model torque (gravity and friction) by more than this, for the
# ticks, is pushed on from outside: the motion stops where the arm is (backends/yam/external_load.py). A first,
# conservative setting, about a fifth of the motor guard above, to be tuned on the cell; per joint, shoulder first.
EXTERNAL_LOAD_STOP_NM = (6.0, 6.0, 6.0, 2.5, 2.5, 2.5)
EXTERNAL_LOAD_STOP_TICKS = 5

# Gripper: i2rt's normalized command, 1 open and 0 closed.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
# A real gripper is never held at either end: 0 and 1 are the mechanical stops i2rt found
# when it calibrated the jaws, and a position held there pushes into the stop until the
# motor overheats (it did, in about two minutes). Held commands stay inside this range.
GRIPPER_HOLD_RANGE = (0.03, 0.97)
# The jaws are ramped, never stepped. A close commanded as one step crossed the whole 94 mm in 0.26 s (about
# 0.3 m/s) and hit the object at that speed, with i2rt's 50 N limit engaging only after the impact: on 2026-11-14 it
# snapped a printed fingertip at its mounting foot. Speeds are per finger (the aperture moves at twice that):
# closing 40 mm/s of aperture, opening 80 mm/s.
GRIPPER_CLOSE_SPEED_M_S = 0.02
GRIPPER_OPEN_SPEED_M_S = 0.04
# The most the jaws may push, from the first contact on: a gripper command runs in force control (grippers.py), the
# motor's own position loop off and the host's torque capped at this force's torque, friction compensation included.
GRIPPER_FORCE_LIMIT_N = 15.0
# The linear 4310's force-to-motor-torque ratio (i2rt's linear map: gripper stroke over motor stroke, m per rad).
GRIPPER_FORCE_TO_TORQUE = 0.096 / 6.57
# Force control's soft position gain (N·m per rad of motor angle), friction compensation (N·m) and the error below
# which no friction compensation is added (rad), as the acef teleoperation follower sets them for this gripper.
GRIPPER_SOFT_KP_NM_PER_RAD = 1.0
GRIPPER_FRICTION_NM = 0.12
GRIPPER_DEADBAND_RAD = 0.02
# Jaws that stalled on an object are held this far (per finger) past where they stopped, not driven on to closed.
GRIPPER_HOLD_SQUEEZE_M = 0.001
# The ramped command never leads the measured jaw by more than this (per finger): on contact the jaw stops and
# the command waits for it, instead of running on through the stall window.
GRIPPER_COMMAND_LEAD_M = 0.003
# The whole travel at the closing speed (118 ticks), the stall window and a margin.
GRIPPER_TICKS = 160
GRIPPER_STALL_WINDOW = 10
GRIPPER_STALL_STILL_STEP = 0.003
GRIPPER_ARRIVAL_TOLERANCE = 0.05


def published_limits() -> dict[str, float | int]:
    return {
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "control_frequency_hz": CONTROL_FREQ_HZ,
        "free_space_speed_m_s": LINEAR_SPEED_M_S["free_space"],
        "max_reach_from_base_m": MAX_REACH_FROM_BASE_M,
        "inter_arm_min_distance_m": INTER_ARM_MIN_DISTANCE_M,
        "gripper_ticks": GRIPPER_TICKS,
    }
