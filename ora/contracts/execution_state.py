"""Private execution-request state contract (not a model-visible action).

The host owns these values. A worker uses a detached copy for one RPC only.
Recovery goals are explicit measured joint/pose targets selected by the host.
"""

from copy import deepcopy
import math
from typing import Any, Mapping

from .joint_trajectory import MAX_TRAJECTORY_NATIVE_ACTIONS

SCHEMA = "ora.execution-state.v1"
ARMS = ("left", "right")
FIELDS = {
    "pose_targets": ("exec.command", "pose_target_xyz_xyzw", ""),
    "gripper_directives": ("exec.command", "gripper_directive", ""),
    # The arm joint target the plant holds (a trajectory's last knot, or the
    # measured joints it stopped at); absent from runs recorded before it.
    "joint_targets": ("exec.command", "joint_target_rad", ""),
    "home_joints": ("exec.recovery", "arm_joint_qpos", ".home"),
    "home_poses": ("exec.recovery", "pose_xyz_xyzw", ".home"),
    "last_safe_joints": ("exec.recovery", "arm_joint_qpos", ".previous"),
    "last_safe_poses": ("exec.recovery", "pose_xyz_xyzw", ".previous"),
    # The joints measured at the end of every native action of the move that
    # set the previous anchor (first row: the anchor), written by
    # follow_joint_trajectory; absent when that move recorded none.
    "last_safe_paths": ("exec.recovery", "measured_path_rad", ".previous"),
    # The undo stack recover_pose retraces (``exec.recovery.<arm>.undo``,
    # ``moves`` oldest first, at most UNDO_DEPTH): one entry per arm-moving tool
    # call (its trajectories concatenated): the call's identity, its start joints
    # and pose, its measured path and, once a retrace was cut, its progress; a
    # completed retrace pops it. Absent from runs recorded before it existed.
    "undo_stacks": ("exec.recovery", "moves", ".undo"),
    "session_home_joints": ("exec.session_home", "arm_joint_qpos", ""),
    "session_home_poses": ("exec.session_home", "pose_xyz_xyzw", ""),
}


def unpack(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: {arm: deepcopy(values[key][member]) for arm in ARMS
                if (key := f"{prefix}.{arm}{suffix}") in values and member in values[key]}
        for field, (prefix, member, suffix) in FIELDS.items()
    }


def pack(fields: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field, (prefix, member, suffix) in FIELDS.items():
        for arm, value in fields.get(field, {}).items():  # joint_targets is absent before it existed
            if arm not in ARMS:
                raise ValueError("unknown execution-state arm")
            values.setdefault(f"{prefix}.{arm}{suffix}", {})[member] = deepcopy(value)
    return values


#: The members of a recovery anchor a recovery goal restates (never its recorded path).
ANCHOR_MEMBERS = ("arm_joint_qpos", "pose_xyz_xyzw")
#: The arm-moving tool calls an arm's undo stack keeps; a push beyond drops the oldest.
UNDO_DEPTH = 5
_UNDO_MEMBERS = {"tool_call_id", "tool", "ordinal", "arm_joint_qpos", "pose_xyz_xyzw", "measured_path_rad"}
_UNDO_OPTIONAL = {"undo_progress"}


def _vectors(value: Any, size: int) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(row, list) and len(row) == size
        and all(type(v) in (int, float) and math.isfinite(v) for v in row) for row in value)


def _retrace_goals(arguments: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Check a retrace's restated undo-stack tops against the state; a retrace lends no recovery goal."""
    for arm, stated in arguments["retrace"].items():
        moves = (values.get(f"exec.recovery.{arm}.undo") or {}).get("moves") or []
        if len(moves) != stated["moves"] or moves[-1]["arm_joint_qpos"] != list(stated["arm_joint_qpos"]):
            raise ValueError("stale_or_mismatched_recovery_target")
    return {}


def recovery_goals(action: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """The selected recovery anchors of ``recover_motion`` or of a recovery ``follow_joint_trajectory``.

    A recovery trajectory restates them in ``arguments.recovery`` (``targets``, ``recovery_targets``).
    """
    arguments = action.get("arguments") or {}
    if action.get("action") == "follow_joint_trajectory" and isinstance(arguments.get("recovery"), Mapping):
        arguments = arguments["recovery"]
    elif action.get("action") != "recover_motion":
        return {}
    if "retrace" in arguments:
        return _retrace_goals(arguments, values)
    targets = arguments.get("targets") or {arguments["arm"]: arguments.get("strategy", "backtrack_last_safe")}
    goals = {}
    for arm, strategy in targets.items():
        key = f"exec.recovery.{arm}.{'home' if strategy == 'home' else 'previous'}"
        if key in values and set(ANCHOR_MEMBERS).issubset(values[key]):
            goals[arm] = {**{member: deepcopy(values[key][member]) for member in ANCHOR_MEMBERS},
                          "strategy": strategy}
    if "recovery_targets" in arguments:
        supplied = validate_recovery_targets(targets, arguments["recovery_targets"])
        if supplied != goals:
            raise ValueError("stale_or_mismatched_recovery_target")
        return supplied
    return goals


def validate(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("execution request requires ora.execution-state.v1")
    budget = value.get("action_budget", 20)
    if type(budget) is not int or not 1 <= budget <= MAX_TRAJECTORY_NATIVE_ACTIONS:
        raise ValueError(f"execution action_budget must be an integer in 1..{MAX_TRAJECTORY_NATIVE_ACTIONS}")
    values, goals = value.get("values"), value.get("recovery_targets", {})
    if not isinstance(values, dict) or not isinstance(goals, dict):
        raise ValueError("invalid execution state values or recovery targets")
    for field, arms in unpack(values).items():
        for item in arms.values():
            if field == "undo_stacks":
                if (not isinstance(item, list) or len(item) > UNDO_DEPTH
                        or any(not isinstance(move, Mapping) or not _UNDO_MEMBERS <= set(move)
                               or set(move) - _UNDO_MEMBERS - _UNDO_OPTIONAL
                               or not _vectors(move["measured_path_rad"], 6)
                               or not _vectors([move["arm_joint_qpos"]], 6)
                               or not _vectors([move["pose_xyz_xyzw"]], 7) for move in item)):
                    raise ValueError("invalid undo stack")
                continue
            if field == "last_safe_paths":
                if (not isinstance(item, list) or not item
                        or any(not isinstance(row, list) or len(row) != 6
                               or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                               for row in item)):
                    raise ValueError("invalid recorded recovery path")
                continue
            vector = [item] if field == "gripper_directives" else item
            if (not isinstance(vector, list) or not vector
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError(f"invalid finite execution value: {field}")
            if "pose" in field and len(vector) != 7:
                raise ValueError("execution pose requires xyz/xyzw")
            if field == "joint_targets" and len(vector) != 6:
                raise ValueError("execution joint target requires six radians")
    for arm, target in goals.items():
        if arm not in ARMS or target.get("strategy") not in {"home", "backtrack_last_safe"}:
            raise ValueError("invalid recovery target identity")
        for member, size in (("arm_joint_qpos", 6), ("pose_xyz_xyzw", 7)):
            vector = target.get(member)
            if (not isinstance(vector, list) or len(vector) != size
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError(f"invalid explicit recovery target: {member}")
    return deepcopy(dict(value))


def validate_recovery_targets(strategies: Mapping[str, Any], targets: Any) -> dict[str, Any]:
    """An explicit restatement of selected recovery anchors, never arbitrary joints."""
    if not isinstance(targets, Mapping) or set(targets) != set(strategies):
        raise ValueError("recovery_targets must match the selected arms")
    for arm, value in targets.items():
        if (arm not in ARMS or not isinstance(value, Mapping)
                or set(value) != {"strategy", "arm_joint_qpos", "pose_xyz_xyzw"}
                or value["strategy"] != strategies[arm]):
            raise ValueError("recovery_targets must match each selected strategy and anchor fields")
    validated = validate({"schema": SCHEMA, "values": {}, "recovery_targets": dict(targets)})
    return validated["recovery_targets"]
