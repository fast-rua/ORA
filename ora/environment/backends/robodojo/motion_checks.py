"""Pure RoboDojo motion/aperture qualification cases and evidence judgements.

Every case runs on the route every episode runs: one ``follow_joint_trajectory``
per move, planned on the host exactly as the toolkit's move_pose, return_home and
recover_pose plan theirs (``orchestrator/backends/robodojo/qualification.py``),
or, for the executor's own admission, knots built here from the measured joints.
Nothing here imports the agent: the host hands the runner each bound action, or
the toolkit's refusal before any action.
"""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
from ora.contracts.joint_trajectory import MAX_KNOT_STEP_RAD, TRAJECTORY_ACTION
from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT
from .gripper_ramp import GRIPPER_RAMP_EXECUTOR
from .sparse_motion import JAW_SYMMETRY_AXIS, SPARSE_RECEIPT_SCHEMA, rotate_about_local_axis
from .trajectory import TRAJECTORY_EXECUTOR, TRAJECTORY_STRATEGY

#: v2: the case table moved to follow_joint_trajectory (v1 drove the SDK executor's move_pose and recover_motion).
SCHEMA = "ora.robodojo-sparse-motion-qualification.v2"
MOTION_ACTION_V3_SCHEMA = "ora-motion-action.v3"
ROTATION_ONLY_DRIFT_BOUND_M = 0.03
REACH_POSITION_TOLERANCE_M = float(ARX_X5_CONTRACT["position_tolerance_m"])
REACH_ORIENTATION_TOLERANCE_RAD = float(ARX_X5_CONTRACT["orientation_tolerance_rad"])
INACTIVE_ARM_DRIFT_BOUND_M = 0.005
GRIPPER_TOLERANCE = float(ARX_X5_CONTRACT["gripper_tolerance_normalized"])
ARM = "left"
OTHER_ARM = "right"
#: A move the toolkit refused before any action: no action was bound, so nothing can have been written.
REFUSED = "refused"
#: The free route of a move (``lib.motion_plan.route``): the shared planner's path where the worker binds one
#: (every episode on the default baseline does), else a joint line to one IK solution.
FREE_ROUTES = ("global", "joint")

# --- robodojo_x5_jaw_aperture_probe_v1 ---------------------------------------
#
# The jaw geometry the probe compares its measurement against, read from the
# same pinned Assets checkout the grasp compiler reads
# (a14409d7fae673c00499e01fd88b4457df6351b1, Assets/Robots/x5): joint7/joint8
# are prismatic on link6 at ``xyz = (0.08657, +/-0.0249, -0.00024)`` with axes
# ``+/-y``, and the link7/link8 inner pad faces sit at ``y = -/+0.02449`` in
# their own joint frames. The values are duplicated here rather than imported
# because ``ora/environment`` never imports ``ora/agent``;
# ``test_robodojo_backend`` asserts the two copies agree, so a drift fails the
# suite instead of producing two different "nominal" apertures.
JAW_APERTURE_PROBE_ID = "robodojo_x5_jaw_aperture_probe_v1"
JAW_CALIBRATION_SCHEMA = (
    "ora.robodojo-x5-jaw-aperture-calibration.v1"
)
JAW_COMMAND_JOINT_RANGE_M = tuple(
    float(value) for value in ARX_X5_CONTRACT["gripper_command"]["primary_joint_range"]
)
JAW_MESH_PAD_OFFSET_M = 0.02449
JAW_CLOSED_PAD_GAP_M = 0.0008
JAW_APERTURE_PER_JOINT_M = 2.0
JAW_NOMINAL_MAXIMUM_APERTURE_M = 0.0888
JAW_FINGER_JOINT_LIMIT_M = (0.0, 0.044)
#: Free-air pass bounds. The design spec's intermediate 0.75/0.5/0.25/0.178
#: commands are not expressible: the public ``set_gripper`` contract accepts
#: only ``open`` and ``close``, so a sweep through them would measure commands
#: no policy can ever issue. The probe measures the two endpoints the policy
#: actually commands, twice, and says so in its own record.
JAW_OPEN_APERTURE_TOLERANCE_M = 0.002
JAW_CLOSED_APERTURE_BOUND_M = 0.001
JAW_MIMIC_BOUND_M = 0.0005
JAW_SWEEP_AGREEMENT_M = 0.001
JAW_FINGER_UNDERSHOOT_BOUND_M = 0.0005
JAW_SWEEPS = 2
JAW_APERTURE_CASE_ID = "jaw_aperture_free_air_sweep"
#: What a free-air sweep cannot reach, and therefore what the calibration
#: record must keep declaring unmeasured: pad compliance under load, the
#: contact-limited stall, and the reading the policy would see with an object
#: between the jaws. Those are the only aperture facts a retention judgement
#: could rest on, so the record's ``contact_leg`` stays explicit rather than
#: absent, and ``contact_leg_passed`` gates the calibrated flag on the grasp
#: side.
JAW_CONTACT_LEG_STATUS = "not_measured"
JAW_CONTACT_LEG_REASON = (
    "a contact leg needs a scripted grasp of an object of source-known extent "
    "in a pinned qualification scene; it is specified but not implemented, so "
    "pad compliance and contact-limited stall stay unmeasured"
)
# X5A.urdf's base_link->link6 chain sums to 0.822 m of link translation
# (0.0605 + 0.0447 + 0.264 + 0.2513 + 0.1099 + 0.0912), an upper bound on how
# far link6 can stand from its own base mount at *any* joint vector, and
# env_cfg/robot/dual_x5.yml bolts the two fixed bases to the table's -y edge.
X5_LINK6_REACH_BOUND_M = 0.822
X5_BASE_MOUNTS_M = ((-0.3, -0.45, 0.765), (0.3, -0.45, 0.765))
# The refusal cases have to reach the *planner*, so the target stays inside the
# live robodojo_arx_x5_tabletop_v1 action envelope (x [-0.75, 0.75],
# y [-0.65, 0.55], z [0.72, 1.40]) with a centimetre of margin on every axis,
# and is still kinematically hopeless: the far top corner diagonally opposite
# the left mount is 1.566 m from it and 1.251 m from the right one, over 1.5x
# the reach bound above and beyond it even if the whole 0.14 m jaw chain
# counted as reach. A motion probe refuses a query outside that envelope, so a
# target outside it would prove a different thing.
UNREACHABLE_POSITION = [0.74, 0.54, 1.39]
#: Why the toolkit refuses a move to it, by free route: the shared planner finds no IK in its world
#: (``goal_unreachable``); the joint route finds no canonical-first IK solution (``paced_path_unreachable``).
UNREACHABLE_REFUSAL_CODES = {"global": ("goal_unreachable",), "joint": ("paced_path_unreachable",)}
#: Past every ARX X5 joint limit (X5A.urdf: +/-10 rad on joints 1-5, +/-3.14 on joint 6): 4 pi on joint 1.
OUTSIDE_JOINT_LIMIT_RAD = 4.0 * math.pi


def _rotate_approach(orientation: Sequence[float], degrees: float) -> list[float]:
    return rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.radians(degrees))


def _pose(position: Sequence[float], orientation: Sequence[float]) -> dict[str, list[float]]:
    """A motion probe query's pose."""
    return {"position_m": [round(float(v), 6) for v in position],
            "orientation_xyzw": [round(float(v), 8) for v in orientation]}


def _move(poses: Mapping[str, Sequence[float]], interaction: str) -> dict[str, Any]:
    return {"kind": "move", "targets": {arm: [float(v) for v in pose] for arm, pose in poses.items()},
            "interactions": {arm: interaction for arm in poses}}


def build_case_request(case_id: str, poses: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """One planned case's move from the measured start poses: link6 targets and their interaction label.

    The host plans it as move_pose plans a compiled request (the label picks the route); orientations are used
    exactly, without move_pose's jaw-twin choice."""
    left = [float(v) for v in poses[ARM]]
    x, y, z = left[:3]
    quat = left[3:]
    if case_id == "free_space_translation_8cm":
        return _move({ARM: [x - 0.08, y + 0.06, z + 0.04, *quat]}, "free_space")
    if case_id == "combined_translation_rotation_45deg":
        return _move({ARM: [x, y - 0.06, z + 0.05, *_rotate_approach(quat, 45.0)]}, "free_space")
    if case_id == "orientation_only_small_12deg":
        # Comfortably inside the 15-degree local envelope so passive drift
        # between observation and planning cannot flip the route.
        return _move({ARM: [x, y, z, *_rotate_approach(quat, 12.0)]}, "precision")
    if case_id == "orientation_only_90deg":
        return _move({ARM: [x, y, z, *_rotate_approach(quat, 90.0)]}, "free_space")
    if case_id == "orientation_only_near_180deg":
        return _move({ARM: [x, y, z, *_rotate_approach(quat, 179.0)]}, "free_space")
    if case_id == "precision_small_local_6mm":
        return _move({ARM: [x, y, z + 0.006, *quat]}, "precision")
    if case_id == "contact_approach_small_8mm":
        return _move({ARM: [x, y, z - 0.008, *quat]}, "contact_approach")
    if case_id == "contact_approach_staged_80mm":
        # Past the 50 mm local envelope: the default surfaces stage it (a free path to 40 mm short, then the
        # straight line), the route move_pose takes for a long contact approach.
        return _move({ARM: [x, y, z - 0.08, *quat]}, "contact_approach")
    if case_id == "planner_refusal_unreachable":
        return _move({ARM: [*UNREACHABLE_POSITION, *quat]}, "free_space")
    if case_id == "below_tolerance_noop":
        return _move({ARM: [x, y, z, *quat]}, "precision")
    if case_id == "dual_arm_synchronized_lift":
        right = [float(v) for v in poses[OTHER_ARM]]
        return _move({ARM: [x, y, z + 0.04, *quat], OTHER_ARM: [right[0], right[1], right[2] + 0.04, *right[3:]]},
                     "free_space")
    raise ValueError(f"unknown qualification case: {case_id}")


def build_case_knots(case_id: str, joints: Sequence[float]) -> list[list[float]]:
    """One admission case's knots for ``ARM`` from its measured joints; the executor must refuse them unwritten."""
    measured = [float(v) for v in joints]
    if case_id == "trajectory_step_exceeds_limit":
        # The first knot holds; the second steps joint 1 by 1.5x the step one native action can follow.
        return [measured, [measured[0] + 1.5 * MAX_KNOT_STEP_RAD, *measured[1:]]]
    if case_id == "trajectory_joint_limit":
        return [[OUTSIDE_JOINT_LIMIT_RAD, *measured[1:]]]
    raise ValueError(f"unknown qualification case: {case_id}")


def knot_arguments(knots: Sequence[Sequence[float]]) -> dict[str, Any]:
    """``follow_joint_trajectory`` arguments moving only ``ARM`` through ``knots``, with the default guards."""
    return {"arms": {ARM: {"knots_rad": [[float(v) for v in row] for row in knots]}},
            "interpolation": "knots_linear_ff"}


def measured_joints(observation: Mapping[str, Any]) -> dict[str, list[float]]:
    """Both arms' measured joints from a public observation."""
    effectors = observation["proprioception"]["end_effectors"]
    return {arm: [float(v) for v in effectors[arm]["arm_joint_qpos"]] for arm in (ARM, OTHER_ARM)}


# ``expected_route`` ``free`` is the run's free route (``FREE_ROUTES``); the toolkit's route rule
# (``lib.motion_plan.route.choose``) gives ``line`` for precision and contact moves inside the local envelope and
# ``staged`` for a longer contact approach. ``knots`` cases bypass the planner to reach the executor's admission.
MOTION_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "free_space_translation_8cm", "expected_status": {"reached"}, "expected_route": "free"},
    {"case_id": "combined_translation_rotation_45deg", "expected_status": {"reached"}, "expected_route": "free"},
    {"case_id": "orientation_only_small_12deg", "expected_status": {"reached"}, "expected_route": "line",
     "rotation_only": True},
    # Large turns may end not_reached (a guard stop) or be refused before any action (the planner, the wrist
    # check, a branch change): either way a pure rotation must not carry link6 away.
    {"case_id": "orientation_only_90deg", "expected_status": {"reached", "not_reached", REFUSED},
     "expected_route": "free", "rotation_only": True},
    {"case_id": "orientation_only_near_180deg", "expected_status": {"reached", "not_reached", REFUSED},
     "expected_route": "free", "rotation_only": True},
    {"case_id": "precision_small_local_6mm", "expected_status": {"reached"}, "expected_route": "line"},
    {"case_id": "contact_approach_small_8mm", "expected_status": {"reached", "not_reached"},
     "expected_route": "line"},
    {"case_id": "contact_approach_staged_80mm", "expected_status": {"reached", "not_reached"},
     "expected_route": "staged"},
    {"case_id": "planner_refusal_unreachable", "expected_status": {REFUSED},
     "expected_refusal_codes": UNREACHABLE_REFUSAL_CODES},
    {"case_id": "below_tolerance_noop", "expected_status": {"reached"}, "expected_route": "line"},
    {"case_id": "dual_arm_synchronized_lift", "expected_status": {"reached"}, "expected_route": "free",
     "dual_arm": True},
    {"case_id": "trajectory_step_exceeds_limit", "knots": True, "expected_status": {"plan_rejected"},
     "expected_failure_code": "trajectory_step_exceeds_limit"},
    {"case_id": "trajectory_joint_limit", "knots": True, "expected_status": {"plan_rejected"},
     "expected_failure_code": "trajectory_joint_limit"},
)

# ``JAW_APERTURE_CASE_ID`` is deliberately absent. Its probe, its judge and
# their unit tests are implemented below, but the fake plant has no finger
# bodies, so the case has never been driven end to end and cannot be allowed
# to fail a live gate for a reason no test has exercised. The measured reach
# evidence also puts the aperture off the causal path for the failures that
# motivated it: every refused grasp candidate sat past the arm's dexterous
# workspace boundary, and no aperture check rejected a single candidate.
# Admit the case only together with a fixture that drives it.
SEQUENCE_CASES: tuple[str, ...] = (
    "recover_home_after_matrix",
    "gripper_close_holds_pose",
    "gripper_open_holds_pose",
    "recover_retrace_after_move",
    "jaw_flip_variants_probed",
    "probe_consistency_unreachable",
    "official_preterminal_boundary",
    "official_terminal_boundary_intact",
    "round_reset_isolation",
)

ALL_CASE_IDS: tuple[str, ...] = tuple([case["case_id"] for case in MOTION_CASES] + list(SEQUENCE_CASES))


def _pose_error(target: Sequence[float], measured: Sequence[float]) -> tuple[float, float]:
    distance = math.sqrt(sum((float(target[i]) - float(measured[i])) ** 2 for i in range(3)))
    dot = abs(sum(float(target[3 + i]) * float(measured[3 + i]) for i in range(4)))
    return distance, 2.0 * math.acos(min(1.0, dot))


def _translation_drift(receipt: Mapping[str, Any], arm: str) -> float | None:
    before = receipt.get("before", {}).get(arm, {}).get("pose_xyz_xyzw")
    after = receipt.get("after", {}).get(arm, {}).get("pose_xyz_xyzw")
    if not (isinstance(before, list) and isinstance(after, list) and len(before) == 7 and len(after) == 7):
        return None
    return math.sqrt(sum((float(after[i]) - float(before[i])) ** 2 for i in range(3)))


def expected_route(case: Mapping[str, Any], free_route: str) -> str | None:
    """The route a planned case must take in a run whose free route is ``free_route``."""
    route = case.get("expected_route")
    return free_route if route == "free" else route


def _within(error: tuple[float, float]) -> bool:
    return error[0] <= REACH_POSITION_TOLERANCE_M and error[1] <= REACH_ORIENTATION_TOLERANCE_RAD


def judge_motion_case(case: Mapping[str, Any], receipt: Mapping[str, Any], *,
                      targets: Mapping[str, Sequence[float]] | None = None, route: str | None = None,
                      free_route: str = "global") -> dict[str, Any]:
    """Pass/fail one executed motion case using only public receipt fields.

    ``targets`` are the requested link6 poses and ``route`` the planned route (None for knots cases). A reached
    arm must end within the arrival tolerance both of its plan's last knot (the receipt's ``pose_error``, against
    the knots' forward kinematics) and of the requested target."""
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    if receipt.get("action") != TRAJECTORY_ACTION:
        notes.append(f"action {receipt.get('action')} is not {TRAJECTORY_ACTION}")
    status = str(receipt.get("execution_status"))
    if status not in case["expected_status"]:
        notes.append(f"status {status} not in {sorted(case['expected_status'])}")
    if status == "reached":
        if receipt.get("motion_strategy") != TRAJECTORY_STRATEGY:
            notes.append(f"strategy {receipt.get('motion_strategy')} is not {TRAJECTORY_STRATEGY}")
        executor = receipt.get("executor")
        if not isinstance(executor, list) or TRAJECTORY_EXECUTOR not in executor:
            notes.append(f"executor {executor} lacks {TRAJECTORY_EXECUTOR}")
    wanted = expected_route(case, free_route)
    if not case.get("knots") and wanted is not None and route != wanted:
        notes.append(f"route {route} is not {wanted}")
    if case.get("expected_failure_code") is not None and receipt.get("failure_code") != case["expected_failure_code"]:
        notes.append(f"failure_code {receipt.get('failure_code')}")
    if status == "plan_rejected" and (receipt.get("state_changed") or receipt.get("native_actions")):
        notes.append("refusal wrote the world")
    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    after = receipt.get("after") if isinstance(receipt.get("after"), Mapping) else {}
    targeted = [ARM, OTHER_ARM] if case.get("dual_arm") else [ARM]
    for arm in targeted:
        effect = arms.get(arm) if isinstance(arms.get(arm), Mapping) else {}
        if effect.get("targeted") is not True:
            notes.append(f"{arm} is not marked targeted")
        if status != "reached":
            continue
        pose_error = effect.get("pose_error")
        if not isinstance(pose_error, Mapping):
            notes.append(f"{arm} reached without a pose error")
        elif not _within((float(pose_error.get("translation_m", math.inf)),
                          float(pose_error.get("orientation_rad", math.inf)))):
            notes.append(f"{arm} reached with residual {pose_error} from its last knot")
        measured = (after.get(arm) or {}).get("pose_xyz_xyzw")
        if targets is not None and arm in targets:
            if not isinstance(measured, list) or len(measured) != 7:
                notes.append(f"{arm} reached without an after pose")
            elif not _within(error := _pose_error(targets[arm], measured)):
                notes.append(f"{arm} reached {error[0]:.4f} m / {error[1]:.4f} rad from its requested target")
    for arm in (OTHER_ARM,) if not case.get("dual_arm") else ():
        drift = (arms.get(arm) or {}).get("inactive_arm_drift") if isinstance(arms.get(arm), Mapping) else None
        if not isinstance(drift, Mapping):
            notes.append(f"{arm} inactive drift is missing")
        elif float(drift.get("translation_m", math.inf)) > INACTIVE_ARM_DRIFT_BOUND_M:
            notes.append(f"{arm} moved {drift['translation_m']:.4f} m while not targeted")
    invariant = receipt.get("command_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("verified") is not True:
        notes.append("command invariant not verified")
    if case.get("rotation_only"):
        drift = _translation_drift(receipt, ARM)
        if drift is None:
            notes.append("rotation-only case lacks before/after poses")
        elif drift > ROTATION_ONLY_DRIFT_BOUND_M:
            notes.append(f"pure rotation realized {drift:.3f} m translation drift")
    return {"passed": not notes, "notes": notes}


def judge_refusal(case: Mapping[str, Any], refusal: Mapping[str, Any], *, free_route: str = "global") -> dict[str, Any]:
    """Pass/fail one planned case the toolkit refused before binding any action."""
    notes: list[str] = []
    if REFUSED not in case["expected_status"]:
        notes.append(f"refused before any action ({refusal.get('code')}): {refusal.get('message')}")
    codes = (case.get("expected_refusal_codes") or {}).get(free_route)
    if codes is not None and refusal.get("code") not in codes:
        notes.append(f"refusal code {refusal.get('code')!r} is not one of {list(codes)}")
    return {"passed": not notes, "notes": notes}


def judge_gripper_case(directive: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Pass/fail one rate-limited ``set_gripper`` (the object form every surface sends) in free air."""
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    if receipt.get("execution_status") != "reached":
        notes.append(f"gripper {directive} ended {receipt.get('execution_status')}")
    if receipt.get("executor") != [GRIPPER_RAMP_EXECUTOR]:
        notes.append(f"executor {receipt.get('executor')}")
    effect = receipt.get("arms", {}).get(ARM, {}) if isinstance(receipt.get("arms"), Mapping) else {}
    error = effect.get("gripper_error")
    if not isinstance(error, (int, float)) or float(error) > GRIPPER_TOLERANCE:
        notes.append(f"gripper residual {error}")
    invariant = receipt.get("command_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("verified") is not True:
        notes.append("command invariant not verified")
    if receipt.get("pose_targets_before") != receipt.get("pose_targets_after"):
        notes.append("gripper action changed retained pose targets")
    return {"passed": not notes, "notes": notes}


def judge_recovery_case(receipt: Mapping[str, Any], anchor: Sequence[float], *, strategy: str,
                        arm: str = ARM) -> dict[str, Any]:
    """Pass/fail one arm of a recovery trajectory (``strategy`` ``home`` or ``retrace``) against its anchor pose."""
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    if receipt.get("execution_status") != "reached":
        notes.append(f"recovery ended {receipt.get('execution_status')}")
    executor = receipt.get("executor")
    if not isinstance(executor, list) or TRAJECTORY_EXECUTOR not in executor:
        notes.append(f"executor {executor}")
    recovery = (receipt.get("trajectory") or {}).get("recovery") or {}
    if (recovery.get("targets") or {}).get(arm) != strategy:
        notes.append(f"{arm} is not a {strategy} recovery: {recovery.get('targets')}")
    after = receipt.get("after", {}).get(arm, {}).get("pose_xyz_xyzw")
    if not isinstance(after, list) or len(after) != 7:
        notes.append("recovery lacks an after pose")
    else:
        distance, angle = _pose_error(anchor, after)
        if distance > REACH_POSITION_TOLERANCE_M or angle > REACH_ORIENTATION_TOLERANCE_RAD:
            notes.append(f"recovery residual {distance:.4f} m / {angle:.4f} rad from its anchor")
    return {"passed": not notes, "notes": notes}


def nominal_jaw_aperture_m(gripper_command: float) -> float:
    """Pad-face separation the pinned URDF/mesh geometry predicts for a command."""

    if not 0.0 <= float(gripper_command) <= 1.0:
        raise ValueError("gripper command must lie in [0, 1]")
    low, high = JAW_COMMAND_JOINT_RANGE_M
    joint = low + (high - low) * float(gripper_command)
    return max(
        0.0,
        round(JAW_APERTURE_PER_JOINT_M * joint + JAW_CLOSED_PAD_GAP_M, 6),
    )


def _jaw_axis_world(pose_xyz_xyzw: Sequence[float]) -> list[float]:
    """The controlled link's local +y expressed in world coordinates."""

    x, y, z, w = (float(pose_xyz_xyzw[index]) for index in (3, 4, 5, 6))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("controlled-link orientation is not a rotation")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Second column of the rotation matrix for xyzw.
    return [
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    ]


def measured_pad_face_separation_m(reading: Mapping[str, Any]) -> float:
    """Private measured pad-face gap for one arm's raw jaw reading.

    The two finger bodies' world origins sit on the joint origins, which travel
    along the controlled link's local +/-y as the prismatic joints open. Their
    separation along that axis minus the two mesh pad offsets is the gap
    between the pad faces, and it is the only aperture number in this file that
    is a measurement rather than a prediction.
    """

    bodies = reading.get("finger_body_position_w_m")
    if not isinstance(bodies, Mapping) or len(bodies) != 2:
        raise ValueError("a jaw reading needs exactly two finger body positions")
    axis = _jaw_axis_world(reading["controlled_link_pose_xyz_xyzw"])
    first, second = (list(bodies[name]) for name in sorted(bodies))
    projected = sum(
        (float(first[index]) - float(second[index])) * axis[index]
        for index in range(3)
    )
    return round(abs(projected) - 2.0 * JAW_MESH_PAD_OFFSET_M, 9)


def _two_finger_mean_m(fingers: Sequence[float]) -> float:
    """Both finger joints' mean in the primary coordinate (``common.two_finger_opening``)."""
    command = ARX_X5_CONTRACT["gripper_command"]
    multiplier, offset = float(command["mimic_multiplier"]), float(command["mimic_offset"])
    return (float(fingers[0]) + (float(fingers[1]) - offset) / multiplier) / 2


def derive_jaw_step(
    *, command: float, reading: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one probe step's measured, nominal and error fields."""

    measured = measured_pad_face_separation_m(reading)
    nominal = nominal_jaw_aperture_m(command)
    fingers = [float(value) for value in reading["gripper_finger_qpos"]]
    targets = [
        float(value) for value in reading["applied_gripper_joint_targets"]
    ]
    low, high = JAW_COMMAND_JOINT_RANGE_M
    return {
        "commanded_directive": float(command),
        "gripper_command": float(reading["gripper_command"]),
        "gripper": float(reading["gripper"]),
        "gripper_finger_qpos": fingers,
        "applied_gripper_joint_targets": targets,
        "measured_aperture_m": measured,
        "nominal_aperture_m": nominal,
        "aperture_error_m": round(measured - nominal, 9),
        "mimic_error_m": round(abs(fingers[0] - fingers[1]), 9)
        if len(fingers) >= 2 else None,
        # The public normalized reading must reproduce both finger joints'
        # mean on the command scale (E11G12-001; the second finger mapped back
        # through the X5 mimic); a divergence means the published state and
        # the plant disagree.
        "normalized_reading_error": round(
            float(reading["gripper"]) - (_two_finger_mean_m(fingers) - low) / (high - low), 9,
        ),
        "physics_steps": reading.get("physics_steps"),
        "native_actions": reading.get("native_actions"),
    }


def judge_jaw_aperture_sweeps(
    sweeps: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Pass/fail the free-air legs of ``robodojo_x5_jaw_aperture_probe_v1``.

    Every criterion is measured, none is assumed. A pass certifies the
    actuator, the mimic and the closed floor in free air; it certifies nothing
    about contact, which is why the record keeps a separate contact leg.
    """

    notes: list[str] = []
    if len(sweeps) != JAW_SWEEPS or any(len(sweep) < 2 for sweep in sweeps):
        return {
            "passed": False,
            "notes": [f"expected {JAW_SWEEPS} sweeps of at least two steps"],
        }
    for index, sweep in enumerate(sweeps):
        commands = [float(step["commanded_directive"]) for step in sweep]
        if commands != sorted(commands, reverse=True):
            notes.append(f"sweep {index} is not a descending command sequence")
        apertures = [float(step["measured_aperture_m"]) for step in sweep]
        if any(
            later > earlier + JAW_SWEEP_AGREEMENT_M
            for earlier, later in zip(apertures, apertures[1:])
        ):
            notes.append(f"sweep {index} aperture is not monotone non-increasing")
        for step in sweep:
            command = float(step["commanded_directive"])
            mimic = step.get("mimic_error_m")
            if mimic is None or float(mimic) > JAW_MIMIC_BOUND_M:
                notes.append(
                    f"sweep {index} command {command}: mimic error {mimic}"
                )
            if abs(float(step["normalized_reading_error"])) > GRIPPER_TOLERANCE:
                notes.append(
                    f"sweep {index} command {command}: public normalized reading "
                    f"diverges by {step['normalized_reading_error']}"
                )
            if any(
                value < JAW_FINGER_JOINT_LIMIT_M[0] - JAW_FINGER_UNDERSHOOT_BOUND_M
                for value in step["gripper_finger_qpos"]
            ):
                notes.append(
                    f"sweep {index} command {command}: finger below its own limit"
                )
            if command >= 1.0:
                if abs(float(step["aperture_error_m"])) > JAW_OPEN_APERTURE_TOLERANCE_M:
                    notes.append(
                        f"sweep {index} open aperture error "
                        f"{step['aperture_error_m']}"
                    )
            elif command <= 0.0 and (
                float(step["measured_aperture_m"]) > JAW_CLOSED_APERTURE_BOUND_M
            ):
                notes.append(
                    f"sweep {index} closed aperture "
                    f"{step['measured_aperture_m']} exceeds the closed bound"
                )
    levels = {float(step["commanded_directive"]) for sweep in sweeps for step in sweep}
    for command in sorted(levels):
        measured = [
            float(step["measured_aperture_m"])
            for sweep in sweeps for step in sweep
            if float(step["commanded_directive"]) == command
        ]
        if len(measured) != len(sweeps):
            notes.append(f"command {command} is not measured in every sweep")
        elif max(measured) - min(measured) > JAW_SWEEP_AGREEMENT_M:
            notes.append(
                f"command {command} hysteresis {max(measured) - min(measured):.6f} m"
            )
    return {"passed": not notes, "notes": notes}


def jaw_aperture_calibration(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a passing qualification record into the bindable calibration.

    The output is the exact value
    ``ora/agent/tools/arx_x5_gripper.ARX_X5_APERTURE_CALIBRATION``
    accepts. A record whose aperture case did not pass produces nothing, so a
    nominal number can never be laundered into the grasp compiler as a
    measurement.
    """

    cases = {
        str(case.get("case_id")): case
        for case in record.get("cases", [])
        if isinstance(case, Mapping)
    }
    case = cases.get(JAW_APERTURE_CASE_ID)
    if case is None or not case.get("passed"):
        raise ValueError(
            "no passing jaw-aperture case; the caveat stays and nothing binds"
        )
    steps = [
        step
        for sweep in case.get("sweeps", []) for step in sweep
        if isinstance(step, Mapping)
    ]
    if not steps:
        raise ValueError("a passing jaw-aperture case must carry its steps")
    apertures = [float(step["measured_aperture_m"]) for step in steps]
    contact = case.get("contact_leg")
    contact = dict(contact) if isinstance(contact, Mapping) else {
        "status": JAW_CONTACT_LEG_STATUS, "reason": JAW_CONTACT_LEG_REASON,
    }
    return {
        "schema": JAW_CALIBRATION_SCHEMA,
        "probe_id": JAW_APERTURE_PROBE_ID,
        "record_sha256": str(record["record_sha256"]),
        "source_commit": str(record["source_commit"]),
        "passed": True,
        "measured_maximum_aperture_m": round(max(apertures), 6),
        "measured_closed_aperture_m": round(min(apertures), 6),
        "contact_leg_passed": contact.get("status") == "passed",
        "contact_leg": contact,
        "arms": sorted(case.get("arms", [])),
    }


def summarize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Public, prompt-safe summary persisted per case."""
    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    trajectory = receipt.get("trajectory") if isinstance(receipt.get("trajectory"), Mapping) else None
    summary = {
        "schema": receipt.get("schema"), "action": receipt.get("action"),
        "execution_status": receipt.get("execution_status"), "motion_strategy": receipt.get("motion_strategy"),
        "failure_stage": receipt.get("failure_stage"), "failure_code": receipt.get("failure_code"),
        "executor": receipt.get("executor"), "state_changed": receipt.get("state_changed"),
        "native_actions": receipt.get("native_actions"), "round_index": receipt.get("round_index"),
        "terminal_after_action": receipt.get("terminal_after_action"),
        "command_invariant_verified": (receipt.get("command_invariant") or {}).get("verified")
        if isinstance(receipt.get("command_invariant"), Mapping) else None,
        "trajectory": None if trajectory is None else {
            key: trajectory[key] for key in ("status", "knots_planned", "knots_executed", "stop", "settle",
                                             "admission", "recovery", "undo") if key in trajectory},
        "arms": {},
    }
    for arm, effect in arms.items():
        if not isinstance(effect, Mapping):
            continue
        summary["arms"][arm] = {
            "targeted": effect.get("targeted"), "pose_error": effect.get("pose_error"),
            "gripper_error": effect.get("gripper_error"), "inactive_arm_drift": effect.get("inactive_arm_drift"),
            "motion_progress": effect.get("motion_progress"),
            "max_tracking_error_rad": ((trajectory or {}).get("arms") or {}).get(arm, {}).get("max_tracking_error_rad"),
        }
    return summary
