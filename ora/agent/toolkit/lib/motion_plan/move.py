"""One arm move end to end: plan, time, fit, screen and run it as one smooth trajectory.

This is what move_pose does with its speed tiers, as public functions a tool
of any author can call with its own caps: ``plan`` takes any ``timing.Pace``
(and, optionally, a slower ``Pace`` for the last ``timing.SLOW_APPROACH_M`` of
each arm's path), ``run`` executes the plan as one ``follow_joint_trajectory``
and returns the receipt and the same ``motion`` summary move_pose returns.

``expect_contact`` is a deliberate touch, and it plans as one. The move is the
straight line the whole way, whatever its label and length, with the orientation
the line holds, never a free path around what it means to touch. The straight part of a contact is
exempt from the collision world where a free path is not: the fingers may touch what
they sweep through, and only the links behind them are checked
(``kinematics.contact_path``, ``contact_path_blocked``). A pushed object is thus no
obstacle and no mask is needed. The same check covers a ``contact_approach`` label
without ``expect_contact``. The same contact on LIBERO is its ``contact_approach``
interaction.

Replay: the inverse kinematics, FK, planned paths, timing and screens come from
recorded ``kinematics`` answers and the one trajectory is the recorded primitive;
the rest is float-deterministic computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...interface.api import DataInterface
from . import checks, execute, kinematics, paths, planner, route, timing

#: A joint-line bridge over an IK hole may leave the straight line by at most this much.
MAX_BRIDGE_DEVIATION_M = 0.005
ROUTES = ("interaction", "line", "straight", "joint", "staged", "global")
#: A contact path is checked at about this many rows, evenly along it.
CONTACT_CHECK_ROWS = 32


@dataclass(frozen=True)
class Planned:
    """A planned move, fitted to its budget and screened, ready for ``run``."""
    path: planner.PathPlan
    knots: Mapping
    fraction: float
    whole_native_actions: int
    expected: Mapping[str, list]
    settle: Mapping
    goals: Mapping[str, list]
    native_action_budget: int
    pace: timing.Pace
    slow_pace: timing.Pace | None = None
    #: Per arm: the knot (1-based) where its slow final segment starts and that segment's start fraction.
    slow_from: Mapping[str, dict] = field(default_factory=dict)
    #: The ``planned_peak_*`` of the timing.
    peaks: Mapping[str, float] = field(default_factory=dict)
    #: The screen's answer for the whole planned path (``kinematics.screen_path``; it was clear, or ``plan`` refused).
    screen: Mapping = field(default_factory=dict)


def _staged(api, starts, goals, joints, *, global_planner):
    lines = {arm: paths.line(starts[arm], goals[arm]) for arm in goals}
    precontact = {arm: path.pose(paths.precontact_fraction(path, translation_m=route.PRECONTACT_TRANSLATION_M,
                                                           rotation_rad=route.PRECONTACT_ROTATION_RAD))
                  for arm, path in lines.items()}
    first = _free(api, precontact, joints, global_planner=global_planner)
    handover = {arm: first.grid.joints[arm][-1].tolist() for arm in goals}
    second = planner.plan_line(api, starts=precontact, goals=goals, start_joints=handover,
                               max_deviation_m=MAX_BRIDGE_DEVIATION_M)
    return planner.PathPlan(timing.concatenate(first.grid, second.grid), "staged", dict(goals), None,
                            second.bridged, first.warnings + second.warnings)


def _free(api, goals, joints, *, global_planner):
    """The free route to ``goals``: the planner's path, or the joint line where no planner answers."""
    if global_planner:
        try:
            return planner.plan_global(api, goals=goals)
        except planner.PlannerUnavailable:
            pass
    return planner.plan_joint(api, goals=goals, start_joints=joints)


def path(api: DataInterface, *, route_kind: str, starts: Mapping[str, Sequence[float]],
         goals: Mapping[str, Sequence[float]], start_joints: Mapping[str, Sequence[float]],
         interactions: Mapping[str, str] | None = None, capabilities: Sequence[str] = (),
         global_planner: bool = False, expect_contact: bool = False) -> planner.PathPlan:
    """Replay: recorded kinematics answers. The path of one move (no timing yet).

    ``route_kind``: ``line`` (the straight line of the controlled point), ``straight`` (the same line that is
    never replaced by another path), ``joint`` (a joint-space line to one IK solution), ``global`` (the
    shared planner's collision-free path), ``staged`` (the free route to a pre-contact pose, then the straight
    final 40 mm), or ``interaction`` (move_pose's rule: the targets' shared interaction label picks one of them).
    With ``global_planner`` the free route is the planner's, and a straight line that cannot be followed takes it
    too, unless ``expect_contact``: a deliberate contact stays on the straight line, and with ``interaction`` it is
    the ``straight`` route, whatever the label and length. Raises ``checks.PlanRefusal``.
    Example: path(api, route_kind='line', starts=s, goals=g, start_joints=q)."""
    if route_kind not in ROUTES:
        raise ValueError(f"route_kind must be one of {ROUTES}")
    kind = "straight" if route_kind == "interaction" and expect_contact else route_kind
    if kind == "interaction":
        labels = set((interactions or {}).values())
        if len(labels) != 1:
            raise checks.PlanRefusal("both arms of one move need the same interaction", code="mixed_interactions",
                                     constraints={"interactions": dict(interactions or {})})
        kind = route.choose(labels.pop(), {arm: paths.line(starts[arm], goals[arm]) for arm in goals},
                            staged_available=route.STAGED_CAPABILITY in capabilities, global_available=global_planner)
        if kind == "refused":
            raise checks.PlanRefusal("a contact approach beyond 50 mm / 15 degrees needs the staged route",
                                     code=route.ROUTE_REFUSAL_CODE)
    if kind == "staged":
        return _staged(api, starts, goals, start_joints, global_planner=global_planner)
    if kind == "global":
        return _free(api, goals, start_joints, global_planner=True)
    if kind == "joint":
        return planner.plan_joint(api, goals=goals, start_joints=start_joints)
    if kind == "straight":
        return planner.plan_line(api, starts=starts, goals=goals, start_joints=start_joints,
                                 max_deviation_m=MAX_BRIDGE_DEVIATION_M)
    try:
        return planner.plan_line(api, starts=starts, goals=goals, start_joints=start_joints,
                                 max_deviation_m=MAX_BRIDGE_DEVIATION_M)
    except checks.PlanRefusal as refusal:
        if not global_planner or expect_contact or refusal.code not in route.LINE_FALLBACK_CODES:
            raise
    return _free(api, goals, start_joints, global_planner=True)


def _is_contact(found: planner.PathPlan, *, route_kind: str, interactions: Mapping[str, str] | None,
                expect_contact: bool) -> bool:
    """Whether ``found`` has a straight part that may touch: a contact by route, label or intent."""
    if found.route not in ("line", "staged"):
        return False
    return expect_contact or route_kind == "straight" or set((interactions or {}).values()) == {"contact_approach"}


def _check_contact(api, found: planner.PathPlan) -> None:
    """Refuse a contact whose straight part drives a link behind the fingers into the world."""
    first = found.grid.stops[-1] if found.grid.stops else 0
    for arm, rows in found.grid.joints.items():
        part = rows[first:]
        step = max(1, len(part) // CONTACT_CHECK_ROWS)
        answer = kinematics.contact_path(api, arm=arm, joints=[*part[:-1][::step].tolist(), part[-1].tolist()])
        if answer["status"] == "blocked":
            raise checks.PlanRefusal(
                f"this contact would drive {answer['limiting_link']} into what the cameras see, and only the "
                "fingers may touch; nothing was executed. Approach so the fingers reach it first",
                code="contact_path_blocked", constraints={
                    "arm": arm, "limiting_link": answer["limiting_link"],
                    "penetration_m": round(float(answer["penetration_m"]), 4)})


def plan(api: DataInterface, *, starts: Mapping[str, Sequence[float]], goals: Mapping[str, Sequence[float]],
         start_joints: Mapping[str, Sequence[float]], pace: timing.Pace, settle: Mapping, native_action_budget: int,
         route_kind: str = "line", interactions: Mapping[str, str] | None = None,
         capabilities: Sequence[str] = (), slow_pace: timing.Pace | None = None, global_planner: bool = False,
         expect_contact: bool = False) -> Planned:
    """Replay: recorded kinematics answers. Plan, time, fit to the budget and screen one move.

    ``pace`` caps the whole move; ``slow_pace`` (optional) also caps the last ``timing.SLOW_APPROACH_M`` of each
    arm's own path, entered without a stop. ``settle`` is the follow_joint_trajectory settle (its
    ``native_actions`` count toward ``native_action_budget``). Over the budget, the longest first part that
    fits is kept (``Planned.fraction`` < 1; ``run`` reports it as cut short, never as an arrival). Every knot
    is screened for self-collision, the table top and both arms in lockstep (``kinematics.screen_path``), the
    re-check of a planned path. ``global_planner`` and ``expect_contact``: ``path``.
    Raises ``checks.PlanRefusal`` (codes: paced_path_unreachable, paced_path_joint_jump, paced_path_wrist_unwind,
    paced_motion_budget, mixed_interactions, route_contact_approach_exceeds_local_bound, self_collision,
    table_collision, inter_arm_collision, contact_path_blocked, and the planner's goal_unreachable,
    goal_in_collision, goal_below_support, start_in_collision, start_outside_limits, path_blocked) or
    ``kinematics.KinematicsUnavailable``.
    Example: plan(api, starts=s, goals=g, start_joints=q, pace=timing.Pace(0.2, 40.0, 0.8, 150.0, 2.0, 12.0),
    settle={'native_actions': 3, 'joint_tolerance_rad': 0.002, 'joint_speed_rad_s': 0.05},
    native_action_budget=50)."""
    settle = dict(settle, native_actions=min(int(settle["native_actions"]), max(int(native_action_budget) - 1, 0)))
    found = path(api, route_kind=route_kind, starts=starts, goals=goals, start_joints=start_joints,
                 interactions=interactions, capabilities=capabilities, global_planner=global_planner,
                 expect_contact=expect_contact)
    if global_planner and _is_contact(found, route_kind=route_kind, interactions=interactions,
                                      expect_contact=expect_contact):
        _check_contact(api, found)
    timed = timing.timed(api, found.grid, pace, slow_pace=slow_pace,
                         budget=int(native_action_budget) - settle["native_actions"])
    knots = {"joints": timed.joints, "fractions": list(timed.fractions), "duration_s": timed.duration_s,
             "native_actions": timed.native_actions}
    screen = kinematics.screen_path(api, {arm: [list(start_joints[arm]), *rows] for arm, rows in timed.joints.items()})
    checks.screened(screen, knots["fractions"])
    expected = planner.expected_poses(api, found, timed.joints, knots["fractions"])
    return Planned(found, knots, timed.fraction, timed.needed_native_actions, expected, settle,
                   {arm: [float(v) for v in goal] for arm, goal in goals.items()}, int(native_action_budget), pace,
                   slow_pace, dict(timed.slow_from), dict(timed.peaks), screen)


def _deg_s(value):
    return None if value is None else round(math.degrees(value), 3)


def caps(pace: timing.Pace) -> dict:
    """Pure computation: a pace's speed caps as a result reports them. Example: caps(pace)['linear_m_s']."""
    return {"linear_m_s": pace.linear_m_s, "angular_deg_s": pace.angular_deg_s,
            "joint_speed_deg_s": _deg_s(pace.joint_speed_rad_s)}


def peaks(planned: Planned) -> dict:
    """Pure computation: the top speed and turn rate of the controlled point along a planned path.

    The timing's own, on any route, over the (possibly cut) path and its arms.
    Example: peaks(planned)['planned_peak_linear_m_s']."""
    return {key: round(float(value), 4 if "linear" in key else 2) for key, value in planned.peaks.items()}


def outcome(summary: Mapping) -> dict:
    """Pure computation: ``execute.summary`` with every angle labelled (radians; degrees on a degree surface).

    Example: outcome({'stop': {'error': 0.1}})['stop'] == {'joint_error_rad': 0.1}."""
    result = dict(summary)
    stop = result.get("stop")
    if isinstance(stop, dict):
        renamed = {"error": "joint_error_rad", "bound": "tracking_bound_rad"}
        result["stop"] = {renamed.get(key, key): value for key, value in stop.items()}
    settle = result.get("settle")
    if isinstance(settle, dict) and "joint_speed_rad_s" in settle:
        settle = {key: value for key, value in settle.items() if key != "joint_speed_rad_s"}
        settle["joint_speed_deg_s"] = _deg_s(summary["settle"]["joint_speed_rad_s"])
        result["settle"] = settle
    return result


#: The IK generator-state warning, stated once in the result's own words.
IK_UNRECORDED_WARNING = ("This run predates IK-generator recording, so the IK answers this move was planned with "
                         "were recomputed from the generator's initial state and may differ from the ones the "
                         "original run computed.")


def speed_segments(planned: Planned, labels: Mapping[str, str] | None = None) -> dict[str, list[dict]]:
    """Pure computation: per arm, the knots of each speed segment (``follow_joint_trajectory`` annotation).

    ``labels`` names the two paces (default ``{"move": "move", "slow": "slow"}``). Example:
    speed_segments(planned, {'move': 'fast', 'slow': 'careful'})."""
    labels = {"move": "move", "slow": "slow", **dict(labels or {})}
    result = {}
    for arm in planned.knots["joints"]:
        segments = [{"speed": labels["move"], "first_knot": 1}]
        slow = planned.slow_from.get(arm)
        if slow is not None:
            segments = ([] if slow["knot"] == 1 else segments) + [{
                "speed": labels["slow"], "first_knot": slow["knot"],
                "from_fraction_of_move": slow["fraction_of_move"], "path_left_m": slow["path_left_m"]}]
        result[arm] = segments
    return result


def run(api: DataInterface, planned: Planned, *, action_id: str, observation: Mapping, name: str = "this move",
        labels: Mapping[str, str] | None = None, expect_contact: bool = False,
        jaw_symmetry: Mapping[str, str] | None = None) -> dict:
    """Replay: the one recorded trajectory primitive. Run ``planned`` as one ``follow_joint_trajectory``.

    Returns ``receipt`` (the environment's, unprojected), ``remaining_budget`` and ``motion``: planned native
    actions, duration, the trajectory's status/stop/settle/residuals, ``speed_segments`` (each speed's start
    and the native actions it spent), joint-line bridges, ``ik_warning`` and ``budget_limited`` for a move cut
    to its budget. ``expect_contact`` marks a deliberate push or press (the growing-error tracking rule is off;
    the tracking bound still stops a real block; it plans as ``plan`` was told); ``motion.expect_contact`` echoes it. ``jaw_symmetry`` (per arm,
    a ``orientation.JAW_SYMMETRY_LABELS`` label) names how each target's orientation was chosen; the receipt
    echoes it.
    ``name`` and ``labels`` only word the result. Example: run(api, planned,
    action_id='action-4', observation=obs.identity.json())."""
    part = ({"fraction": planned.fraction, "goals_xyz_xyzw": dict(planned.goals)} if planned.fraction < 1.0
            else None)
    segments = speed_segments(planned, labels)
    action = execute.trajectory_action(action_id=action_id, observation=observation, knots=planned.knots["joints"],
                                       expected=planned.expected, settle=planned.settle, part_of_move=part,
                                       speed_segments=segments if planned.slow_pace is not None else None,
                                       expect_contact=expect_contact, jaw_symmetry=jaw_symmetry)
    result = execute.follow(api, action, settle_native_actions=planned.settle["native_actions"],
                            route=planned.path.route)
    summary = execute.summary(result["receipt"])
    motion = {"route": planned.path.route, "planned_native_actions": planned.knots["native_actions"],
              "settle_native_actions_allowed": planned.settle["native_actions"],
              "duration_s": planned.knots["duration_s"], **outcome(summary), "expect_contact": bool(expect_contact),
              "joint6": {arm: checks.joint6_extremes(planned.path.grid.joints[arm][0], rows)
                         for arm, rows in planned.knots["joints"].items()}}
    stop = motion.get("stop")
    if expect_contact and isinstance(stop, dict) and stop.get("reason") == "tracking":
        stop.setdefault("note", "stopped by the tracking bound despite expect_contact")
    if planned.path.bridged:
        motion["joint_line_bridges"] = list(planned.path.bridged)
    if planned.path.warnings:
        motion["ik_warning"] = IK_UNRECORDED_WARNING
    if planned.fraction < 1.0:
        reached = {arm: poses[-1] for arm, poses in planned.expected.items()}
        motion["budget_limited"] = {
            "fraction_of_move": round(planned.fraction, 3), "needed_native_actions": planned.whole_native_actions,
            "native_action_budget": planned.native_action_budget,
            "remaining_m": {arm: round(math.dist(reached[arm][:3], goal[:3]), 4)
                            for arm, goal in planned.goals.items()},
            "residuals_against": "cut_point",
            "note": (f"Cut short: the {planned.native_action_budget} native actions this call could spend cover "
                     f"only part of {name}, so the arm stopped on its path short of the target, remaining_m "
                     "away from it; the target was NOT reached. final_link6_error, the receipt's pose_error and "
                     "the pose target the arm now holds are those of that cut point.")}
    return {"receipt": result["receipt"], "remaining_budget": result["remaining_budget"], "motion": motion}


__all__ = ["IK_UNRECORDED_WARNING", "MAX_BRIDGE_DEVIATION_M", "Planned", "ROUTES", "caps", "outcome", "path", "peaks",
           "plan", "run", "speed_segments"]
