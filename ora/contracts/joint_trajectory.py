"""Joint-trajectory and rate-limited gripper actions of ``ora-motion-action.v3``.

``follow_joint_trajectory`` is one environment action whose arm motion is
smooth inside: the caller supplies, per arm, the joint target at the end of
each native action (a *knot*; one native action is 10 physics steps of 4 ms),
and the environment interpolates every physics step linearly between the
commanded knots with a velocity feed-forward equal to the knot slope. It
never restarts from the measured joints and never holds a stationary target
inside a tick. Every planning choice -- path, timing, inverse kinematics,
collision screening -- belongs to the caller (the toolkit lib); the
environment admits the knots (shape, finite values, live joint limits, a
per-knot step it can physically follow), executes them, guards tracking and
progress per physics step with caller-supplied bounds, and reports raw
measured residuals. ``part_of_move`` marks knots that end short of the
caller's goal (a move cut to fit its budget): the action then ends
``not_reached`` (``native_action_budget_exhausted``) with the measured
distance left to that goal, never as an arrival. ``expect_contact`` (a
deliberate push or press) turns off the growing-error tracking rule; the
tracking bound and the progress guard still apply. ``speed_segments`` names
the knots of each speed (per arm); the receipt reports what each spent.
``recovery`` (``targets``: arm -> ``backtrack_last_safe`` | ``home``;
``recovery_targets``: the restated anchors, as ``recover_motion`` takes them;
or ``targets``: arm -> ``retrace`` with ``retrace``: arm -> the undo
stack's depth and its top entry's start joints) marks a recovery planned
by the caller: the host admits it only while those anchors or stack tops
are current, and the trajectory keeps the previous anchor instead of
replacing it with its own start; a completed retrace pops its move.

``set_gripper`` keeps its name. Its legacy targets (``"open"``/``"close"``)
keep their behaviour; a per-arm object (``gripper_target``) selects the
rate-limited form: the drive target moves every physics step from the
previously commanded target at ``rate_per_s`` and never leads the measured
first finger by more than ``max_lead_m``, which bounds both the closing speed
and the squeeze; the command ends on arrival or a stall, and a close may first
wait for the arm to come to rest. ``hold_lead_m`` (optional, at most
``max_lead_m``) is the lead once the first finger has stopped (moved less than
``end.stall_still_step`` over a native action): a close may approach at one
lead and squeeze at a smaller one. ``arm_settle.first_reading``
``joint_velocity`` (optional) reads the arms' measured joint velocities before
the first settle tick, so an arm already at rest costs no native action
(``native_action``, the default, always looks once after one).

Shared by the agent (which builds these actions) and every environment (which
validates them); only RoboDojo executes them. Units: radians, metres, seconds;
gripper openings are normalized on the command scale (0 closed, 1 open).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

TRAJECTORY_ACTION = "follow_joint_trajectory"
#: Native actions one legacy primitive may use (the ARX X5 controller contract).
LEGACY_NATIVE_ACTION_CAP = 20
#: Native actions one trajectory or rate-limited gripper action may use (10 s).
MAX_TRAJECTORY_NATIVE_ACTIONS = 250
PHYSICS_STEPS_PER_NATIVE_ACTION = 10
PHYSICS_STEP_S = 0.004
NATIVE_ACTION_S = PHYSICS_STEPS_PER_NATIVE_ACTION * PHYSICS_STEP_S
#: The ARX X5 arm drive's velocity limit and the step it allows per native action.
JOINT_SPEED_LIMIT_RAD_S = 5.0
MAX_KNOT_STEP_RAD = JOINT_SPEED_LIMIT_RAD_S * NATIVE_ACTION_S
ARM_JOINTS = 6
ARMS = ("left", "right")
INTERPOLATIONS = ("knots_linear_ff",)
STOP_MODES = ("hold_measured",)
#: The SDK's own gripper limiter: 0.2 of the X5 command range (0.054 m).
SDK_GRIPPER_LEAD_M = 0.0108
CONTROLLER_INTERFACE = "knot_interpolation_v1"

DEFAULT_TRACKING_WINDOW_STEPS = 5
DEFAULT_PROGRESS = {"window_steps": 10, "min_ratio": 0.25, "min_commanded_rad": 0.002}
#: A growing tracking error: above ``min_error_rad`` and up by more than ``min_rise_rad`` over the last
#: ``over_steps`` physics steps, for ``sustained_steps`` consecutive steps (an arm pushed back by an obstacle).
#: Calibrated on the 2026-10-28 real-card runs: clean moves stay under 0.0055 rad and never rise like this
#: for more than 3 consecutive steps; a careful deliberate push rose about 0.001 rad per native action.
DEFAULT_TRACKING_GROWTH = {"min_error_rad": 0.004, "min_rise_rad": 0.0015, "over_steps": 10, "sustained_steps": 10}
DEFAULT_SETTLE = {"native_actions": 0, "joint_tolerance_rad": 0.002, "joint_speed_rad_s": 0.05}
DEFAULT_GRIPPER_END = {"tolerance": 0.03, "stall_window_native_actions": 3, "stall_still_step": 0.005}

_ARM_FIELDS = {"knots_rad", "gripper_knots", "expected_link6_xyz_xyzw"}
_FIELDS = {"arms", "interpolation", "guards", "gripper", "settle", "admission", "part_of_move", "speed_segments",
           "recovery", "expect_contact", "jaw_symmetry"}
RECOVERY_STRATEGIES = ("backtrack_last_safe", "home")
#: A recovery that retraces each arm's latest recorded move (recover_pose on the trajectory surfaces).
RETRACE_STRATEGY = "retrace"
#: How a target's orientation was chosen between the request and its half-turn jaw twin (echoed only).
JAW_SYMMETRY_LABELS = ("requested", "twin_nearer_current", "held_no_flip", "exact")
_SEGMENT_FIELDS = {"speed", "first_knot", "from_fraction_of_move", "path_left_m"}
_GUARD_FIELDS = {"tracking_rad", "tracking_window_steps", "progress", "on_stop", "tracking_growth"}
_GRIPPER_TARGET_FIELDS = {"opening", "rate_per_s", "max_lead_m", "hold_lead_m", "end", "arm_settle"}
#: How an ``arm_settle`` takes its first reading of the arms' speed (``native_action``: after one tick, the default).
ARM_SETTLE_FIRST_READINGS = ("native_action", "joint_velocity")


class TrajectoryContractError(ValueError):
    """A malformed trajectory or rate-limited gripper action, refused before any write."""


def _number(value: Any, label: str, *, low: float | None = None, high: float | None = None,
            positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrajectoryContractError(f"{label} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise TrajectoryContractError(f"{label} must be positive")
    if (low is not None and result < low) or (high is not None and result > high):
        raise TrajectoryContractError(f"{label} must lie in [{low}, {high}]")
    return result


def _integer(value: Any, label: str, *, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise TrajectoryContractError(f"{label} must be an integer in {low}..{high}")
    return value


def _vectors(value: Any, label: str, width: int) -> list[list[float]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_TRAJECTORY_NATIVE_ACTIONS:
        raise TrajectoryContractError(f"{label} must hold 1..{MAX_TRAJECTORY_NATIVE_ACTIONS} rows")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise TrajectoryContractError(f"{label}[{index}] must hold {width} numbers")
        rows.append([_number(item, f"{label}[{index}]") for item in row])
    return rows


def _arm(value: Any, arm: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "knots_rad" not in value or set(value) - _ARM_FIELDS:
        raise TrajectoryContractError(f"arms.{arm} takes knots_rad and optional {sorted(_ARM_FIELDS - {'knots_rad'})}")
    result: dict[str, Any] = {"knots_rad": _vectors(value["knots_rad"], f"arms.{arm}.knots_rad", ARM_JOINTS)}
    count = len(result["knots_rad"])
    if value.get("gripper_knots") is not None:
        knots = value["gripper_knots"]
        if not isinstance(knots, list) or len(knots) != count:
            raise TrajectoryContractError(f"arms.{arm}.gripper_knots must hold one opening per knot")
        result["gripper_knots"] = [_number(item, f"arms.{arm}.gripper_knots", low=0.0, high=1.0) for item in knots]
    if value.get("expected_link6_xyz_xyzw") is not None:
        poses = _vectors(value["expected_link6_xyz_xyzw"], f"arms.{arm}.expected_link6_xyz_xyzw", 7)
        if len(poses) != count:
            raise TrajectoryContractError(f"arms.{arm}.expected_link6_xyz_xyzw must hold one pose per knot")
        result["expected_link6_xyz_xyzw"] = poses
    return result


def _guards(value: Any, count: int) -> dict[str, Any]:
    value = {} if value is None else value
    if not isinstance(value, Mapping) or set(value) - _GUARD_FIELDS:
        raise TrajectoryContractError(f"guards takes {sorted(_GUARD_FIELDS)}")
    tracking = value.get("tracking_rad")
    if isinstance(tracking, list):
        if len(tracking) != count:
            raise TrajectoryContractError("guards.tracking_rad must hold one bound per knot")
        tracking = [_number(item, "guards.tracking_rad", positive=True) for item in tracking]
    elif tracking is not None:
        tracking = _number(tracking, "guards.tracking_rad", positive=True)
    progress = value.get("progress", DEFAULT_PROGRESS)
    if progress is not None:
        if not isinstance(progress, Mapping) or set(progress) != set(DEFAULT_PROGRESS):
            raise TrajectoryContractError(f"guards.progress takes exactly {sorted(DEFAULT_PROGRESS)}")
        progress = {"window_steps": _integer(progress["window_steps"], "guards.progress.window_steps", low=2, high=100),
                    "min_ratio": _number(progress["min_ratio"], "guards.progress.min_ratio", low=0.0, high=1.0),
                    "min_commanded_rad": _number(progress["min_commanded_rad"], "guards.progress.min_commanded_rad",
                                                 positive=True)}
    growth = value.get("tracking_growth", DEFAULT_TRACKING_GROWTH)
    if growth is not None:
        if not isinstance(growth, Mapping) or set(growth) != set(DEFAULT_TRACKING_GROWTH):
            raise TrajectoryContractError(f"guards.tracking_growth takes exactly {sorted(DEFAULT_TRACKING_GROWTH)}")
        growth = {"min_error_rad": _number(growth["min_error_rad"], "guards.tracking_growth.min_error_rad",
                                           positive=True),
                  "min_rise_rad": _number(growth["min_rise_rad"], "guards.tracking_growth.min_rise_rad", positive=True),
                  "over_steps": _integer(growth["over_steps"], "guards.tracking_growth.over_steps", low=1, high=100),
                  "sustained_steps": _integer(growth["sustained_steps"], "guards.tracking_growth.sustained_steps",
                                              low=1, high=100)}
    on_stop = value.get("on_stop", STOP_MODES[0])
    if on_stop not in STOP_MODES:
        raise TrajectoryContractError(f"guards.on_stop must be one of {list(STOP_MODES)}")
    return {"tracking_rad": tracking,
            "tracking_window_steps": _integer(value.get("tracking_window_steps", DEFAULT_TRACKING_WINDOW_STEPS),
                                              "guards.tracking_window_steps", low=1, high=100),
            "progress": None if progress is None else dict(progress), "on_stop": on_stop,
            "tracking_growth": None if growth is None else dict(growth)}


def _settle(value: Any) -> dict[str, Any]:
    value = {**DEFAULT_SETTLE, **dict(value or {})} if isinstance(value or {}, Mapping) else value
    if not isinstance(value, Mapping) or set(value) != set(DEFAULT_SETTLE):
        raise TrajectoryContractError(f"settle takes exactly {sorted(DEFAULT_SETTLE)}")
    return {"native_actions": _integer(value["native_actions"], "settle.native_actions", low=0, high=50),
            "joint_tolerance_rad": _number(value["joint_tolerance_rad"], "settle.joint_tolerance_rad", positive=True),
            "joint_speed_rad_s": _number(value["joint_speed_rad_s"], "settle.joint_speed_rad_s", positive=True)}


def _lead(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label, positive=True, high=SDK_GRIPPER_LEAD_M)


def _part_of_move(value: Any, arms: Mapping[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"fraction", "goals_xyz_xyzw"}:
        raise TrajectoryContractError("part_of_move takes exactly fraction and goals_xyz_xyzw")
    fraction = _number(value["fraction"], "part_of_move.fraction", low=0.0, high=1.0)
    goals = value["goals_xyz_xyzw"]
    if not isinstance(goals, Mapping) or not goals or set(goals) - set(arms):
        raise TrajectoryContractError("part_of_move.goals_xyz_xyzw names arms of this trajectory")
    return {"fraction": fraction,
            "goals_xyz_xyzw": {arm: _vectors([goals[arm]], f"part_of_move.goals_xyz_xyzw.{arm}", 7)[0]
                               for arm in ARMS if arm in goals}}


def _speed_segments(value: Any, arms: Mapping[str, Any], count: int) -> dict[str, list[dict[str, Any]]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value or set(value) - set(arms):
        raise TrajectoryContractError("speed_segments names arms of this trajectory")
    result = {}
    for arm in ARMS:
        if arm not in value:
            continue
        rows = value[arm]
        if not isinstance(rows, list) or not rows:
            raise TrajectoryContractError(f"speed_segments.{arm} must be a non-empty list")
        normalized, previous = [], 0
        for row in rows:
            if (not isinstance(row, Mapping) or not {"speed", "first_knot"} <= set(row)
                    or set(row) - _SEGMENT_FIELDS or not isinstance(row["speed"], str)
                    or not 0 < len(row["speed"]) <= 32):
                raise TrajectoryContractError(f"speed_segments.{arm} rows take speed, first_knot and optional "
                                              "from_fraction_of_move, path_left_m")
            first = _integer(row["first_knot"], f"speed_segments.{arm}.first_knot", low=previous + 1, high=count)
            if not normalized and first != 1:
                raise TrajectoryContractError(f"speed_segments.{arm} must start at knot 1")
            item = {"speed": row["speed"], "first_knot": first}
            for key in ("from_fraction_of_move", "path_left_m"):
                if row.get(key) is not None:
                    item[key] = _number(row[key], f"speed_segments.{arm}.{key}", low=0.0)
            normalized.append(item)
            previous = first
        result[arm] = normalized
    return result


def _recovery(value: Any, arms: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping) and set(value) == {"targets", "retrace"}:
        return _retrace(value, arms)
    if not isinstance(value, Mapping) or set(value) != {"targets", "recovery_targets"}:
        raise TrajectoryContractError("recovery takes targets and either recovery_targets or retrace")
    targets = value["targets"]
    if (not isinstance(targets, Mapping) or set(targets) != set(arms)
            or any(strategy not in RECOVERY_STRATEGIES for strategy in targets.values())):
        raise TrajectoryContractError(f"recovery.targets names every arm of this trajectory, each one of "
                                      f"{list(RECOVERY_STRATEGIES)}")
    from .execution_state import validate_recovery_targets
    try:
        anchors = validate_recovery_targets(targets, value["recovery_targets"])
    except ValueError as exc:
        raise TrajectoryContractError(f"recovery.{exc}") from exc
    return {"targets": {arm: str(targets[arm]) for arm in ARMS if arm in targets},
            "recovery_targets": {arm: anchors[arm] for arm in ARMS if arm in anchors}}


def _retrace(value: Mapping[str, Any], arms: Mapping[str, Any]) -> dict[str, Any]:
    """A retrace of each arm's latest undoable tool call: the undo stack depth and the call's start joints."""
    from .execution_state import UNDO_DEPTH

    targets, stated = value["targets"], value["retrace"]
    if (not isinstance(targets, Mapping) or set(targets) != set(arms)
            or any(strategy != RETRACE_STRATEGY for strategy in targets.values())
            or not isinstance(stated, Mapping) or set(stated) != set(arms)):
        raise TrajectoryContractError(f"a retrace names every arm of this trajectory as {RETRACE_STRATEGY!r}")
    result = {}
    for arm in ARMS:
        if arm not in stated:
            continue
        entry = stated[arm]
        if not isinstance(entry, Mapping) or set(entry) != {"moves", "arm_joint_qpos"}:
            raise TrajectoryContractError(f"recovery.retrace.{arm} takes exactly moves and arm_joint_qpos")
        result[arm] = {"moves": _integer(entry["moves"], f"recovery.retrace.{arm}.moves", low=1, high=UNDO_DEPTH),
                       "arm_joint_qpos": _vectors([entry["arm_joint_qpos"]], f"recovery.retrace.{arm}.arm_joint_qpos",
                                                  ARM_JOINTS)[0]}
    return {"targets": {arm: RETRACE_STRATEGY for arm in ARMS if arm in targets}, "retrace": result}


def speed_segment_spend(segments: list[Mapping[str, Any]], executed: int) -> list[dict[str, Any]]:
    """Each speed segment with the native actions (knots) it spent of ``executed`` knots."""
    result = []
    for index, segment in enumerate(segments):
        end = segments[index + 1]["first_knot"] - 1 if index + 1 < len(segments) else executed
        result.append({**dict(segment), "native_actions": max(0, min(end, executed) - segment["first_knot"] + 1)})
    return result


def validate_trajectory_arguments(arguments: Any) -> dict[str, Any]:
    """The normalized arguments of one ``follow_joint_trajectory``, or ``TrajectoryContractError``."""
    if not isinstance(arguments, Mapping) or "arms" not in arguments or set(arguments) - _FIELDS:
        raise TrajectoryContractError(f"follow_joint_trajectory arguments take arms and optional {sorted(_FIELDS - {'arms'})}")
    arms = arguments["arms"]
    if not isinstance(arms, Mapping) or not arms or set(arms) - set(ARMS):
        raise TrajectoryContractError("arms must name left, right or both")
    normalized = {arm: _arm(arms[arm], arm) for arm in ARMS if arm in arms}
    counts = {len(value["knots_rad"]) for value in normalized.values()}
    if len(counts) != 1:
        raise TrajectoryContractError("both arms of one trajectory share the same number of knots")
    count = counts.pop()
    interpolation = arguments.get("interpolation", INTERPOLATIONS[0])
    if interpolation not in INTERPOLATIONS:
        raise TrajectoryContractError(f"interpolation must be one of {list(INTERPOLATIONS)}")
    gripper = arguments.get("gripper") or {}
    if not isinstance(gripper, Mapping) or set(gripper) - {"max_lead_m"}:
        raise TrajectoryContractError("gripper takes only max_lead_m")
    admission = arguments.get("admission") or {}
    if not isinstance(admission, Mapping) or set(admission) - {"max_joint_step_rad"}:
        raise TrajectoryContractError("admission takes only max_joint_step_rad")
    jaw_symmetry = arguments.get("jaw_symmetry")
    if jaw_symmetry is not None and (
            not isinstance(jaw_symmetry, Mapping) or set(jaw_symmetry) - set(normalized)
            or any(label not in JAW_SYMMETRY_LABELS for label in jaw_symmetry.values())):
        raise TrajectoryContractError(
            f"jaw_symmetry maps arms of this trajectory to one of {list(JAW_SYMMETRY_LABELS)}")
    expect_contact = arguments.get("expect_contact", False)
    if not isinstance(expect_contact, bool):
        raise TrajectoryContractError("expect_contact must be true or false")
    return {"arms": normalized, "interpolation": interpolation,
            "guards": _guards(arguments.get("guards"), count),
            "gripper": {"max_lead_m": _lead(gripper.get("max_lead_m"), "gripper.max_lead_m")},
            "settle": _settle(arguments.get("settle")),
            "admission": {"max_joint_step_rad": _number(admission.get("max_joint_step_rad", MAX_KNOT_STEP_RAD),
                                                        "admission.max_joint_step_rad", positive=True,
                                                        high=MAX_KNOT_STEP_RAD)},
            "part_of_move": _part_of_move(arguments.get("part_of_move"), normalized),
            "speed_segments": _speed_segments(arguments.get("speed_segments"), normalized, count),
            "expect_contact": expect_contact,
            **({"jaw_symmetry": {arm: jaw_symmetry[arm] for arm in ARMS if arm in jaw_symmetry}}
               if jaw_symmetry else {}),
            **({"recovery": _recovery(arguments["recovery"], normalized)}
               if arguments.get("recovery") is not None else {})}


def validate_gripper_target(value: Any, arm: str) -> dict[str, Any]:
    """One arm's rate-limited ``set_gripper`` target (the object form)."""
    if not isinstance(value, Mapping) or "opening" not in value or set(value) - _GRIPPER_TARGET_FIELDS:
        raise TrajectoryContractError(f"targets.{arm} takes opening and optional "
                                      f"{sorted(_GRIPPER_TARGET_FIELDS - {'opening'})}")
    opening = value["opening"]
    if opening not in ("open", "close"):
        opening = _number(opening, f"targets.{arm}.opening", low=0.0, high=1.0)
    rate = value.get("rate_per_s")
    end = {**DEFAULT_GRIPPER_END, **dict(value.get("end") or {})}
    if set(end) != set(DEFAULT_GRIPPER_END):
        raise TrajectoryContractError(f"targets.{arm}.end takes exactly {sorted(DEFAULT_GRIPPER_END)}")
    settle = value.get("arm_settle")
    if settle is not None:
        keys = {"joint_speed_rad_s", "max_native_actions"}
        if not isinstance(settle, Mapping) or not keys <= set(settle) or set(settle) - keys - {"first_reading"}:
            raise TrajectoryContractError(f"targets.{arm}.arm_settle takes joint_speed_rad_s and max_native_actions "
                                          "and optionally first_reading")
        first = settle.get("first_reading")
        if "first_reading" in settle and first not in ARM_SETTLE_FIRST_READINGS:
            raise TrajectoryContractError(f"targets.{arm}.arm_settle.first_reading must be one of "
                                          f"{list(ARM_SETTLE_FIRST_READINGS)}")
        settle = {"joint_speed_rad_s": _number(settle["joint_speed_rad_s"], "arm_settle.joint_speed_rad_s",
                                               positive=True),
                  "max_native_actions": _integer(settle["max_native_actions"], "arm_settle.max_native_actions",
                                                 low=0, high=50),
                  **({"first_reading": first} if "first_reading" in settle else {})}
    lead = _lead(value.get("max_lead_m"), f"targets.{arm}.max_lead_m")
    hold = {}
    if "hold_lead_m" in value:
        hold_lead = _lead(value["hold_lead_m"], f"targets.{arm}.hold_lead_m")
        if hold_lead is not None and hold_lead > (SDK_GRIPPER_LEAD_M if lead is None else lead):
            raise TrajectoryContractError(f"targets.{arm}.hold_lead_m may not exceed max_lead_m")
        hold = {"hold_lead_m": hold_lead}
    return {"opening": opening,
            "rate_per_s": None if rate is None else _number(rate, f"targets.{arm}.rate_per_s", positive=True),
            "max_lead_m": lead, **hold,
            "end": {"tolerance": _number(end["tolerance"], "end.tolerance", positive=True, high=0.5),
                    "stall_window_native_actions": _integer(end["stall_window_native_actions"],
                                                            "end.stall_window_native_actions", low=1, high=50),
                    "stall_still_step": _number(end["stall_still_step"], "end.stall_still_step", positive=True)},
            "arm_settle": settle}


def rate_limited_gripper(action: Mapping[str, Any]) -> bool:
    """Whether a ``set_gripper`` action uses the rate-limited (object) target form."""
    targets = (action.get("arguments") or {}).get("targets")
    return (action.get("action") == "set_gripper" and isinstance(targets, Mapping)
            and any(isinstance(value, Mapping) for value in targets.values()))


def requires_joint_trajectory_backend(action: Mapping[str, Any]) -> bool:
    """Actions only a backend with the per-physics-step executor can run."""
    return action.get("action") == TRAJECTORY_ACTION or rate_limited_gripper(action)


def native_action_cap(action: Mapping[str, Any]) -> int:
    """The most native actions one admitted action may use."""
    return (MAX_TRAJECTORY_NATIVE_ACTIONS if requires_joint_trajectory_backend(action)
            else LEGACY_NATIVE_ACTION_CAP)


__all__ = [
    "ARMS", "ARM_JOINTS", "ARM_SETTLE_FIRST_READINGS", "CONTROLLER_INTERFACE", "DEFAULT_GRIPPER_END", "DEFAULT_PROGRESS", "DEFAULT_SETTLE",
    "DEFAULT_TRACKING_GROWTH", "DEFAULT_TRACKING_WINDOW_STEPS", "INTERPOLATIONS", "JAW_SYMMETRY_LABELS",
    "JOINT_SPEED_LIMIT_RAD_S",
    "LEGACY_NATIVE_ACTION_CAP",
    "MAX_KNOT_STEP_RAD", "MAX_TRAJECTORY_NATIVE_ACTIONS", "NATIVE_ACTION_S", "PHYSICS_STEPS_PER_NATIVE_ACTION",
    "PHYSICS_STEP_S", "RECOVERY_STRATEGIES", "RETRACE_STRATEGY", "SDK_GRIPPER_LEAD_M", "STOP_MODES", "TRAJECTORY_ACTION", "TrajectoryContractError",
    "native_action_cap", "rate_limited_gripper", "requires_joint_trajectory_backend", "speed_segment_spend",
    "validate_gripper_target", "validate_trajectory_arguments",
]
