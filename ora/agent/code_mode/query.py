"""Queries, answered: the host side of ``robot.query`` (``library: extended``; the list and the documentation are
``query_spec.py``).

A query is a call of the toolkit lib a motion tool plans with -- ``motion_plan.kinematics`` (``ik``, ``fk``,
``screen_path``), ``motion`` (``reach_check``, ``probe_motion``, ``align_plan``) and ``observation``
(``robot_state``) -- over a ``DataInterface`` of the episode (``toolkit/host/query.query_api`` builds one that
records nothing). It reads the environment's robot model or the measured state, public data a robot instance could
read through that interface; it moves nothing and writes nothing. ``probe_motion`` is the one that plans: where the
motion tools plan on the policy side (``toolkit.host.motion.plans_moves``) it runs their own plan, so its verdict is the
real move's and not the environment's IK probe.

The script gives typed values and degrees: poses as ``[x, y, z, qx, qy, qz, qw]`` in ``world`` (metres, xyzw) and
every angle in degrees (``seed_joints_deg``, ``joints_deg``, ``knots_deg``, ``max_joint_step_deg``). The lib takes
radians and returns them, so the arguments are converted on the way in and the answer on the way out with the table
every policy-visible angle goes through (``composition/surface/units.py``): a result key ``..._rad`` of the lib is
``..._deg`` here, rounded as the tools' results are. Anything else of the lib's answer is passed on as it is.

``prepare`` checks a query's arguments and returns the call; it needs no environment, so a refusal for its arguments
costs nothing. The call, given the ``DataInterface``, raises a ``ToolError`` for everything else: ``query_unavailable``
where the environment cannot answer (no read-only bridge, a backend that serves no kinematics query), else the
environment's own refusal with its code, field and constraints.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition.surface import units
from ora.agent.toolkit.interface.types import InterfaceError, PublicToolFailure
from ora.agent.toolkit.lib import motion, observation
from ora.agent.toolkit.lib.motion_plan import checks, kinematics, orientation, timing
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.tools import _dry_run
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import _smooth
from ora.agent.toolkit.tools.move_pose.tool import validate_workspace
from ora.agent.error_classes import ToolError
from ora.contracts import alignment, reach_check
from ora.contracts import kinematics as kinematics_contract

from . import query_spec

#: A quaternion of a pose is a unit one within this much (as ``motion.probe`` and the motion tools admit).
NORM_BOUNDS = (0.98, 1.02)
VARIANT_POLICIES = ("exact", "parallel_jaw_equivalents")
Call = Callable[[Any], dict[str, Any]]


def _refuse(message: str, field: str, received: Any = None, **constraints: Any) -> ToolError:
    return ToolError(message, code="invalid_arguments", retryable=True, field=field, received=received,
                     constraints=constraints)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _refuse(f"{field} must be a finite number", field, repr(value)[:60])
    return float(value)


def _vector(value: Any, size: int, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise _refuse(f"{field} must be {size} numbers", field, f"{len(value)} items" if isinstance(value, (list, tuple))
                      else repr(value)[:60], length=size)
    return [_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _pose(value: Any, field: str) -> list[float]:
    pose = _vector(value, 7, field)
    norm = math.sqrt(sum(item * item for item in pose[3:]))
    if not NORM_BOUNDS[0] <= norm <= NORM_BOUNDS[1]:
        raise _refuse(f"{field} has a quaternion of norm {norm:.3f}", field, pose[3:],
                      norm_min=NORM_BOUNDS[0], norm_max=NORM_BOUNDS[1])
    return pose


def _rows(value: Any, field: str, maximum: int, row: Callable[[Any, str], list[float]], width: int) -> list[list[float]]:
    """A list of ``row``s (a single row of ``width`` numbers alone is one), 1 to ``maximum`` of them."""

    if isinstance(value, (list, tuple)) and len(value) == width and not any(
            isinstance(item, (list, tuple)) for item in value):
        value = [value]
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise _refuse(f"{field} must hold 1 to {maximum} entries of {width} numbers", field,
                      len(value) if isinstance(value, (list, tuple)) else repr(value)[:60], max_items=maximum)
    return [row(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _poses(value: Any, maximum: int) -> list[list[float]]:
    return _rows(value, "poses", maximum, _pose, 7)


def _joint_rows(value: Any, field: str, maximum: int) -> list[list[float]]:
    """Joint vectors in degrees as radians."""

    return [[math.radians(item) for item in joints]
            for joints in _rows(value, field, maximum, lambda item, name: _vector(item, 6, name), 6)]


def _arm(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse("arm must be a name such as left or right", "arm", repr(value)[:60])
    return value


def _choice(value: Any, field: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise _refuse(f"{field} must be one of {', '.join(allowed)}", field, repr(value)[:60], allowed=list(allowed))
    return value


# --- each query: its arguments checked and put in the lib's terms, then the call ---------------------------------

def _ik(arguments: Mapping[str, Any]) -> Call:
    seed = arguments.get("seed_joints_deg")
    step = _number(arguments.get("max_joint_step_deg", query_spec.DEFAULT_STEP_DEG), "max_joint_step_deg")
    if not 0 < step <= 360:
        raise _refuse("max_joint_step_deg must be above 0 and at most 360", "max_joint_step_deg", step)
    given = dict(arm=_arm(arguments["arm"]), poses=_poses(arguments["poses"], kinematics_contract.MAXIMUM_POSES),
                 seed=None if seed is None else [math.radians(item) for item in _vector(seed, 6, "seed_joints_deg")],
                 max_joint_step_rad=math.radians(step),
                 ranking=_choice(arguments.get("ranking", "nearest"), "ranking", kinematics_contract.RANKINGS),
                 screen=_choice(arguments.get("screen", "self_and_measured_other_arm"), "screen",
                                kinematics_contract.SCREENS))
    return lambda api: units.policy_degrees(kinematics.ik_chain(api, **given))


def _fk(arguments: Mapping[str, Any]) -> Call:
    arm = _arm(arguments["arm"])
    joints = _joint_rows(arguments["joints_deg"], "joints_deg", kinematics_contract.MAXIMUM_POSES)
    return lambda api: {"poses": kinematics.fk(api, arm=arm, joints=joints)}


def _screen_path(arguments: Mapping[str, Any]) -> Call:
    knots = arguments["knots_deg"]
    if not isinstance(knots, Mapping) or not knots:
        raise _refuse("knots_deg must map left and/or right to knots", "knots_deg", repr(knots)[:60])
    polylines = {_arm(arm): _joint_rows(rows, f"knots_deg.{arm}", kinematics_contract.MAXIMUM_POSES)
                 for arm, rows in knots.items()}
    if len({len(rows) for rows in polylines.values()}) != 1:
        raise _refuse("every arm needs the same number of knots", "knots_deg",
                      {arm: len(rows) for arm, rows in polylines.items()})
    return lambda api: units.policy_degrees(kinematics.screen_path(api, polylines))


def _reach_check(arguments: Mapping[str, Any]) -> Call:
    poses = _poses(arguments["poses"], reach_check.MAXIMUM_POSES)
    interaction = arguments.get("interaction", "free_space")
    interactions = interaction if isinstance(interaction, (list, tuple)) else [interaction] * len(poses)
    starts = arguments.get("from_index", [None] * len(poses))
    if len(interactions) != len(poses) or not isinstance(starts, (list, tuple)) or len(starts) != len(poses):
        raise _refuse("interaction and from_index are one value for all poses or one per pose", "interaction",
                      len(poses))
    for index, start in enumerate(starts):
        if start is not None and (isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < index):
            raise _refuse("from_index[%d] must be null or the index of an earlier pose" % index, "from_index", start)
    request = {"schema": reach_check.REQUEST_SCHEMA, "arm": _arm(arguments["arm"]), "queries": [
        {"pose_xyzw": pose, "interaction": _choice(kind, "interaction", reach_check.INTERACTIONS),
         "from_index": start} for pose, kind, start in zip(poses, interactions, starts)]}
    return lambda api: units.policy_degrees(motion.reach_check(api, JsonValue.of(request)).data.json())


def _planned_variant(api: Any, config: Mapping[str, Any], arm: str, variant_id: str, pose: list[float]) -> dict[str, Any]:
    """One variant of a probe, planned as a fast free-space ``move_pose`` is (``_smooth.plan_action``), with no budget."""

    action = {"arguments": {"targets": {arm: {"position_m": pose[:3], "orientation_xyzw": pose[3:]}},
                            "interactions": {arm: "free_space"}}}
    row: dict[str, Any] = {"variant_id": variant_id, "target_pose_xyzw": pose}
    current = observation.observe(api).manifest.json()["proprioception"]["end_effectors"][arm]["pose_xyz_xyzw"]
    try:
        validate_workspace(pose[:3], current[:3], config["motion"]["workspace_bounds_m"], "poses")
        _, planned, _ = _smooth.plan_action(api, action, config, budget=timing.UNBOUNDED_NATIVE_ACTIONS)
    except (checks.PlanRefusal, ToolRefusal) as refusal:
        return {**row, "reachable": False, "failure": {"code": refusal.code, "message": str(refusal)}}
    start, end = planned.path.grid.joints[arm][0], planned.knots["joints"][arm][-1]
    return {**row, "reachable": True, "route": planned.path.route,
            "trajectory_steps": planned.knots["native_actions"], "trajectory_duration_s": planned.knots["duration_s"],
            "joint_delta_rad": [float(last) - float(first) for first, last in zip(start, end)],
            "screening": _dry_run.screening(planned.screen)}


def _planned_probe(api: Any, arm: str, poses: Sequence[list[float]], policy: str) -> dict[str, Any]:
    """``probe_motion`` where the motion tools plan on the policy side: each pose gets the verdict the real move gets,
    the planner's when one answers, and nothing executes."""

    config = {"motion": api.planned_motion}
    results = []
    for index, pose in enumerate(poses):
        variants = [("variant-00-exact-+0deg", pose)]
        if policy == "parallel_jaw_equivalents":
            twin = orientation.jaw_twin(pose[3:], config["motion"]["local_axes"]["approach"])
            variants.append(("variant-01-jaw-flip-180deg", [*pose[:3], *twin]))
        rows = [_planned_variant(api, config, arm, variant_id, target) for variant_id, target in variants]
        admitted = sorted((row for row in rows if row["reachable"]), key=lambda row: row["trajectory_steps"])
        results.append({"query_id": f"query-{index}", "arm": arm, "reachable": bool(admitted),
                        "selected_variant_id": admitted[0]["variant_id"] if admitted else None, "variants": rows})
    return {"results": results, "world_state_changed": False}


def _probe_motion(arguments: Mapping[str, Any]) -> Call:
    arm = _arm(arguments["arm"])
    poses = _poses(arguments["poses"], query_spec.PROBE_ITEMS)
    policy = _choice(arguments.get("variant_policy", "exact"), "variant_policy", VARIANT_POLICIES)
    request = {"queries": [{"query_id": f"query-{index}", "arm": arm,
                            "pose": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}}
                           for index, pose in enumerate(poses)], "variant_policy": policy}

    def call(api: Any) -> dict[str, Any]:
        if getattr(api, "planned_motion", None) is not None:
            return units.policy_degrees(_planned_probe(api, arm, poses, policy))
        return units.policy_degrees(motion.probe_motion(api, JsonValue.of(request)).data.json())
    return call


def _align_plan(arguments: Mapping[str, Any]) -> Call:
    request = {"schema": alignment.REQUEST_SCHEMA, "target_ref": None,
               "approach_side": _choice(arguments.get("approach_side", "top"), "approach_side",
                                        alignment.APPROACH_SIDES),
               "arm": _arm(arguments["arm"]), "target_poses_xyzw": _poses(arguments["poses"], alignment.MAXIMUM_POSES)}
    return lambda api: units.policy_degrees(motion.align_plan(api, JsonValue.of(request)).data.json())


def _robot_state(arguments: Mapping[str, Any]) -> Call:
    def call(api: Any) -> dict[str, Any]:
        state = observation.proprioception(api)
        arms = {arm.name: {
            "pose": None if arm.pose is None else {"frame": arm.pose.frame, "xyz": list(arm.pose.position_m),
                                                    "xyzw": list(arm.pose.orientation_xyzw)},
            "joint_positions_deg": [units.degrees(joint) for joint in arm.joint_positions_rad],
            "gripper_normalized": arm.gripper_normalized} for arm in state.arms}
        public = state.public.json()
        for item in (public.get("end_effectors") or {}).values():
            if isinstance(item, dict):
                item.pop("arm_joint_qpos", None)  # radians without a unit in its name: joint_positions_deg above
        return {"arms": arms, "public": units.policy_degrees(public)}
    return call


_PREPARE: dict[str, Callable[[Mapping[str, Any]], Call]] = {
    "ik": _ik, "fk": _fk, "screen_path": _screen_path, "reach_check": _reach_check, "probe_motion": _probe_motion,
    "align_plan": _align_plan, "robot_state": _robot_state,
}


def prepare(name: Any, arguments: Any) -> Call:
    """Query ``name``'s arguments checked and put in the lib's terms; the call to make over a ``DataInterface``.

    Raises ``ToolError`` (``invalid_arguments``, ``unknown_query``) for a query that cannot be asked, before any
    environment is involved."""

    if name not in _PREPARE:
        raise ToolError(f"no query {name!r}", code="unknown_query", retryable=True, field="name",
                        received=repr(name)[:60], constraints={"allowed": list(query_spec.NAMES)})
    if not isinstance(arguments, Mapping):
        raise _refuse("a query's arguments are keyword arguments", "arguments", repr(arguments)[:60])
    allowed = query_spec.argument_names(name)
    stray = sorted(set(arguments) - set(allowed))
    if stray:
        raise _refuse(f"query {name} takes {', '.join(allowed) or 'no arguments'}, not {stray[0]}", stray[0],
                      allowed=list(allowed))
    absent = [item for item in query_spec.required(name) if item not in arguments]
    if absent:
        raise _refuse(f"query {name} needs {absent[0]}", absent[0], required=list(query_spec.required(name)))
    call = _PREPARE[name](arguments)

    def answered(api: Any) -> dict[str, Any]:
        try:
            return call(api)
        except PublicToolFailure as error:
            raise _environment(error.public()) from error
        except kinematics.KinematicsUnavailable as error:
            raise _kinematics(error) from error
        except InterfaceError as error:  # the interface could not bind the call to the current observation
            raise ToolError(str(error), code="query_unavailable", retryable=False) from error

    return answered


def _environment(error: Mapping[str, Any]) -> ToolError:
    """A public tool failure as the refusal a script reads; where it says the environment cannot answer, that."""

    code, message = str(error.get("code")), str(error.get("message"))
    if code == "tool_unavailable" or kinematics_contract.UNSUPPORTED in message:
        return ToolError(message, code="query_unavailable", retryable=False)
    return ToolError(message, code=code, retryable=bool(error.get("retryable")), field=error.get("field"),
                     received=error.get("received"), constraints=error.get("constraints"))


def _kinematics(error: Exception) -> ToolError:
    """The kinematics lib reports a failed answer as the repr of its envelope's error: the tool error in it, if any."""

    try:
        detail = ast.literal_eval(str(error))
    except (ValueError, SyntaxError):
        detail = None
    public = detail.get("tool_error") if isinstance(detail, dict) else None
    if isinstance(public, dict) and "message" in public:
        return _environment(public)
    code = detail.get("code") if isinstance(detail, dict) else None
    return ToolError(f"the environment's kinematics query failed{f' ({code})' if code else ''}",
                     code="query_unavailable", retryable=False)


__all__ = ["prepare"]
