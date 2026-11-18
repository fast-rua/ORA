"""End-effector pacing of RoboTwin's sparse moves (``motion_pace``, ``end_effector_speed_v1``).

Yulun's RoboDojo toolkit timing, through ORA's planner-neutral port
(``ora/environment/controllers/motion_pace.py``): every validated global plan
is re-timed so the public grip-centre TCP keeps the request's speed caps
(``ora.contracts.motion_speed``; none at the default ``fast``, where the joints
alone limit it), and before anything moves a plan whose TCP turns far more than
its start-to-goal turn (a wrist unwinding the long way round) or that needs more
physics steps than one action may spend is refused. The measured-convergence
executor writes one paced knot per stage and spends at least one physics step
on each, so a stage that takes longer only slows the move. The bounded local
DLS servo has no planned path: each iteration's step toward its target runs as
a straight joint line spread over as many physics steps as the request's
linear and turn caps need (``step_caps``; within the last 2 cm with
``slow_approaching``, careful's), with no ramps.

Everything here is a function of the session and the action; the session's
own files (``session.py``, ``common.py``) are untouched, because they bind the
seed scans' dependency hash.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from ora.environment.controllers.motion_pace import (
    PacedPath, StepCaps, pace_path, paces, speed_report, step_caps,
)

#: RoboTwin's SAPIEN physics step (``envs/_base_task.py`` ``set_timestep(1 / 250)``): one knot per step.
PHYSICS_STEP_S = 1.0 / 250.0
#: The ALOHA wrist joint (``fl_joint6`` / ``fr_joint6``) among the planner's six arm joints.
WRIST_JOINT_INDEX = 5
#: The joint ceilings of a paced route, so the default is no faster than the planner's own timing it replaces:
#: on 6903 recorded RoboTwin moves (2026-10-31) those reach 1.97 rad/s at p99, and at 1.1 rad/s^2 the paced
#: default keeps every joint and TCP speed percentile below the planner's and takes 1.31x its time at the median.
JOINT_SPEED_CEILING_RAD_S = 1.9
JOINT_ACCEL_CEILING_RAD_S2 = 1.1
#: The robot whose joint jerk limits the timing worker certifies a paced route at
#: (``motion_timing.JOINT_JERK_LIMITS``): none is published for the ALOHA arms, so the
#: conservative default runs, ``source: "unverified"`` in every receipt's ``motion.timing``.
ROBOT = "aloha_agilex"
#: The pacing's refusals; a receipt names them at failure stage ``pacing``.
PACING_REFUSAL_CODES = frozenset({"paced_path_wrist_unwind", "paced_motion_budget"})


def requested_speed(action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The move's speed request (wire form), or None for the default; only move_pose carries one."""

    arguments = action.get("arguments")
    speed = arguments.get("speed") if isinstance(arguments, Mapping) else None
    return speed if isinstance(speed, Mapping) else None


def tick_budget(profile: Mapping[str, Any]) -> int:
    """Physics steps one paced global route may plan (a control profile's ``profile``).

    The paced knots are the route's executed waypoints, so the validated plan's waypoint limit
    (``plan_validation.maximum_waypoints``) bounds them, as does the transaction's physics-step
    total less one waypoint's wait. A longer route is refused whole, never cut."""

    convergence = profile["motion_transaction"]["global_convergence"]
    return min(int(profile["plan_validation"]["maximum_waypoints"]),
               int(convergence["maximum_total_physics_steps"])
               - int(convergence["maximum_physics_steps_per_waypoint"]))


def pace_arm(rows: Any, fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], *,
             speed: Mapping[str, Any] | None, joint_limits: Sequence[Sequence[float]], budget_ticks: int,
             slow_approach: bool, arm: str) -> PacedPath:
    """One arm's validated joint path timed for the request, wrist-checked, within the budget.

    Raises ``motion_pace.PaceRefusal`` before anything moves."""

    wrist = joint_limits[WRIST_JOINT_INDEX]
    return pace_path(
        [[float(value) for value in row] for row in rows], fk, speed=speed, tick_s=PHYSICS_STEP_S,
        joint_speed_ceiling_rad_s=JOINT_SPEED_CEILING_RAD_S, joint_accel_ceiling_rad_s2=JOINT_ACCEL_CEILING_RAD_S2,
        wrist_joint_index=WRIST_JOINT_INDEX, wrist_limits_rad=(float(wrist[0]), float(wrist[1])),
        budget_ticks=int(budget_ticks), slow_approach=slow_approach, arm=arm, robot=ROBOT)


def synchronized_rows(paced: Mapping[str, PacedPath], starts: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Each arm's knots on one common timeline: an arm whose move is shorter is slowed uniformly.

    ``starts`` are the arms' path starts. Both arms then start and stop together, as the executor drives
    them; uniform slowing keeps every speed and acceleration cap of the shorter arm. Returns
    ``{arm: (ticks, joints) array}``."""

    import numpy as np

    rows = {arm: np.asarray(path.rows(), dtype=np.float64) for arm, path in paced.items()}
    count = max(len(values) for values in rows.values())
    result = {}
    for arm, values in rows.items():
        if len(values) == count:
            result[arm] = values
            continue
        # The knots are the positions at ticks 1..n and tick 0 is the start: sample the same motion at
        # count evenly spaced ticks instead, the last one exactly on the goal.
        timeline = np.vstack([np.asarray(starts[arm], dtype=np.float64), values])
        times = np.linspace(0.0, len(values), count + 1)[1:]
        result[arm] = np.column_stack([np.interp(times, np.arange(len(timeline)), timeline[:, joint])
                                       for joint in range(timeline.shape[1])])
        result[arm][-1] = values[-1]
    return result


def knot_velocities(start: Sequence[float], rows: Any) -> Any:
    """Joint velocity targets of one knot per physics step: central differences, zero at the goal."""

    import numpy as np

    values = np.vstack([np.asarray(start, dtype=np.float64), np.asarray(rows, dtype=np.float64)])
    velocity = np.zeros_like(values)
    if len(values) >= 3:
        velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * PHYSICS_STEP_S)
    return velocity[1:]


def motion_report(paced: Mapping[str, PacedPath], ticks: int) -> dict[str, Any]:
    """``motion`` of the receipt: the request and its caps with the planned peaks of every paced arm, each
    arm's wrist report, the common planned ticks and duration, and each arm's ``timing`` evidence (which
    implementation timed it; sealed evidence only, never in the policy's copy, ``policy_receipt``)."""

    reports = {arm: path.report() for arm, path in paced.items()}
    speed = dict(next(iter(reports.values()))["speed"])
    for key in ("planned_peak_linear_m_s", "planned_peak_angular_deg_s"):
        speed[key] = max(float(report["speed"][key]) for report in reports.values())
    timing = {arm: dict(report["timing"]) for arm, report in reports.items() if "timing" in report}
    return {"speed": speed, "wrist": {arm: dict(report["wrist"]) for arm, report in reports.items()},
            "planned_ticks": int(ticks), "duration_s": round(int(ticks) * PHYSICS_STEP_S, 6),
            **({"timing": timing} if timing else {})}


def request_report(speed: Mapping[str, Any] | None, *, slow_approach: bool = True) -> dict[str, Any]:
    """The request and its caps under this backend's joint ceilings, for a move with no paced route."""

    move, slow = paces(speed)
    move = move.within(JOINT_SPEED_CEILING_RAD_S, JOINT_ACCEL_CEILING_RAD_S2)
    slow = slow.within(JOINT_SPEED_CEILING_RAD_S, JOINT_ACCEL_CEILING_RAD_S2) if slow and slow_approach else None
    return speed_report(speed, move, slow)


def local_step_caps(action: Mapping[str, Any]) -> StepCaps | None:
    """The local servo's caps per physics step of the move's request (None: the default, uncapped)."""

    return step_caps(requested_speed(action), tick_s=PHYSICS_STEP_S)


def local_substeps(caps: StepCaps, translation: Sequence[float], rotation: Sequence[float],
                   remaining_m: float) -> int:
    """Physics steps one servo iteration's step needs under the caps, ``remaining_m`` from its target.

    Example: local_substeps(step_caps({"speed": "careful", "slow_approaching": False}, tick_s=0.004),
    [0.005, 0, 0], [0, 0, 0], 0.05) == 13."""

    import numpy as np

    linear, angular = caps.limits(remaining_m)
    needed = 1.0
    for vector, cap in ((translation, linear), (rotation, angular)):
        if cap is not None and cap > 0.0:
            needed = max(needed, float(np.linalg.norm(np.asarray(vector, dtype=np.float64))) / cap)
    return int(math.ceil(needed - 1e-9))


__all__ = [
    "JOINT_ACCEL_CEILING_RAD_S2", "JOINT_SPEED_CEILING_RAD_S", "PACING_REFUSAL_CODES", "PHYSICS_STEP_S", "ROBOT",
    "WRIST_JOINT_INDEX", "knot_velocities", "local_step_caps", "local_substeps", "motion_report",
    "pace_arm", "request_report", "requested_speed", "synchronized_rows", "tick_budget",
]
