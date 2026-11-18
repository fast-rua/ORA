"""Public detail of a pre-motion refusal: RoboCasa's reach and collision admission, the YAM station's checks.

The worker's ``reach_check`` says where an arm path failed admission: the
path kind, how far along it, the IK residual or the penetration depth, and
which robot part would sink in (arm, base, gripper or held_object). A line's
IK refusal adds why (``cause``: ``ik_unsolved``, or ``joint_branch_jump`` with
the joint, its jump and the bound), the cut's pose (``at_pose``) and whether
the target itself is reachable from here (``endpoint``); a target out of reach
its distance from the arm mount, the bound and the mount height it was
measured from; a ``move_base`` the first contact with the depth of its footprint
or of the arm it carries (``robot_part``, the obstacle's height and position);
and a refusal a base move may answer the base poses from which the move is
admitted (``base_suggestions``, each a ``move_base`` target, and ``base_search``,
what the search asked). Only those fields cross; the worker's scope string and every
scene identity stay behind. The tool help promises them, so a refusal without
them would leave the policy nudging a target that no nearby pose can fix.
RoboCasa's ``torso`` block, the lift a move made before its arm path, crosses
with it (``torso``), whether the move then ran or was refused.

The YAM station's receipt says which check refused a plan (``failure_code``,
which the shared projection folds into ``planner_failed`` for every planner
stage) and where: the arm, the path waypoint or planned tick, the closest IK
miss, the distance to the table or to the other arm. A target IK cannot reach adds
the highest height the arm solves at the same xy and approach (``highest_reachable_z_m``,
None when none down to the table clearance), and the closest position at the same height and approach that
solves (``nearest_reachable_position_m``, with its distance ``nearest_reachable_offset_m``; None when none within
20 cm); an inter-arm refusal the arms the motion
moved (``moving_arms``) and the two nearest parts (``closest_parts``, the station
model's body names without the arm prefix). A plan the cell answered
without moving (``dry_run_read_only`` on a read-only cell, ``operator_declined``
on a live one) carries its summary, and a decline the operator's own words.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

STATION_RECEIPT_SCHEMA = "ora.yam-sparse-motion-receipt.v1"
#: Refusals the station answers with its plan, and the plan fields that cross.
_ANSWERED_WITH_PLAN = ("dry_run_read_only", "operator_declined")
_REFUSAL_NUMBERS = (("closest_position_error_m", 4), ("position_error_m", 4), ("distance_m", 4),
                    ("tcp_height_above_table_m", 4), ("joint_step_rad", 3), ("highest_reachable_z_m", 3),
                    ("nearest_reachable_offset_m", 3))
#: The station's arm parts (body names without the arm prefix) an inter-arm refusal may name.
_STATION_PART = re.compile(r"[a-z][a-z0-9_]{0,31}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def public_refusal_detail(record: Mapping[str, Any], schema: str) -> dict[str, Any]:
    """``{"admission_refusal": {...}}`` for a refused motion of ``schema``, and ``{"torso": {...}}`` for a move of it
    that lifted the torso first; else nothing.

    A YAM station receipt is answered by ``public_station_detail`` whatever ``schema`` names.
    """

    if record.get("schema") == STATION_RECEIPT_SCHEMA:
        return public_station_detail(record)
    if record.get("schema") != schema:
        return {}
    torso = _public_torso(record.get("torso"))
    value = record.get("reach_check")
    if not isinstance(value, Mapping):
        return {"torso": torso} if torso else {}
    detail: dict[str, Any] = {key: value[key] for key in _REACH_TEXTS if isinstance(value.get(key), str)}
    detail.update({key: int(value[key]) for key in ("waypoint_index", "waypoints_total", "joint")
                   if isinstance(value.get(key), int) and not isinstance(value.get(key), bool)})
    detail.update({key: round(float(value[key]), places) for key, places in _REACH_NUMBERS if _finite(value.get(key))})
    for key, size, places in (("at_pose", 7, 4), ("obstacle_xy_m", 2, 3)):
        if _vector(value.get(key), size):
            detail[key] = [round(float(item), places) for item in value[key]]
    residual = value.get("ik_residual")
    if isinstance(residual, Mapping) and _finite(residual.get("translation_m")) and _finite(residual.get("orientation_rad")):
        detail["ik_residual"] = {"translation_m": round(float(residual["translation_m"]), 4),
                                 "orientation_deg": round(math.degrees(float(residual["orientation_rad"])), 2)}
    suggestions = value.get("base_suggestions")
    if isinstance(suggestions, list):
        detail["base_suggestions"] = [_public_suggestion(item) for item in suggestions if isinstance(item, Mapping)]
    searched = value.get("base_search")
    if isinstance(searched, Mapping):
        detail["base_search"] = {key: searched[key] for key in _SEARCH_FIELDS if _finite(searched.get(key))}
    result: dict[str, Any] = {"admission_refusal": detail} if detail else {}
    return {**result, "torso": torso} if torso else result


#: The words, the numbers (with their decimals) and the base search's counts of a ``reach_check`` that cross.
_REACH_TEXTS = ("path_kind", "robot_part", "cause", "endpoint")
_REACH_NUMBERS = (("path_fraction", 3), ("penetration_m", 4), ("jump_rad", 4), ("bound_rad", 4),
                  ("distance_from_mount_m", 4), ("reach_m", 4), ("nearest_mount_height_m", 4),
                  ("obstacle_height_m", 3))
_SEARCH_FIELDS = ("stands", "footprint_clear", "asked", "ik_reached", "arm_blocked", "checked_in_world",
                  "max_travel_m")
_SUGGESTION_NUMBERS = (("x_m", 3), ("y_m", 3), ("yaw_deg", 1), ("torso_height_m", 3), ("distance_m", 3),
                       ("turn_deg", 1))


def _vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_finite(item) for item in value)


def _public_suggestion(value: Mapping[str, Any]) -> dict[str, Any]:
    """One base pose the arm reaches the target from: a ``move_base`` target, the torso height, the move and whether a
    path from the arm's current joints plans there."""
    found = {key: round(float(value[key]), places) for key, places in _SUGGESTION_NUMBERS if _finite(value.get(key))}
    path = value.get("arm_path")
    return {**found, "arm_path": path if isinstance(path, str) else None}


def _public_torso(value: Any) -> dict[str, Any]:
    """A move's lift: from, to and the target height, how it stopped, and whether the arm path was then admitted."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {key: round(float(value[key]), 4) for key in ("from_m", "to_m", "target_m")
                              if _finite(value.get(key))}
    if isinstance(value.get("stop_reason"), str):
        result["stop_reason"] = value["stop_reason"]
    if isinstance(value.get("arm_path_replanned"), bool):
        result["arm_path_replanned"] = value["arm_path_replanned"]
    return result


def public_station_detail(record: Mapping[str, Any]) -> dict[str, Any]:
    """The YAM station's refusal check, the plan of a motion it answered without moving, the operator's words."""

    code = record.get("failure_code")
    result: dict[str, Any] = {}
    refusal = record.get("refusal")
    if isinstance(code, str) and isinstance(refusal, Mapping):
        detail: dict[str, Any] = {"check": code}
        if isinstance(refusal.get("arm"), str):
            detail["arm"] = refusal["arm"]
        detail.update({key: int(refusal[key]) for key in ("waypoint_index", "tick")
                       if isinstance(refusal.get(key), int) and not isinstance(refusal.get(key), bool)})
        detail.update({key: round(float(refusal[key]), places) for key, places in _REFUSAL_NUMBERS
                       if _finite(refusal.get(key))})
        for key in ("highest_reachable_z_m", "nearest_reachable_position_m", "nearest_reachable_offset_m"):
            if key in refusal and refusal[key] is None:
                detail[key] = None
        nearest = refusal.get("nearest_reachable_position_m")
        if isinstance(nearest, list) and len(nearest) == 3 and all(_finite(value) for value in nearest):
            detail["nearest_reachable_position_m"] = [round(float(value), 3) for value in nearest]
        moving = refusal.get("moving_arms")
        if isinstance(moving, list) and all(arm in ("left", "right") for arm in moving):
            detail["moving_arms"] = list(moving)
        parts = refusal.get("closest_parts")
        if isinstance(parts, Mapping) and set(parts) == {"left", "right"} and all(
                isinstance(v, str) and _STATION_PART.fullmatch(v) for v in parts.values()):
            detail["closest_parts"] = {"left": parts["left"], "right": parts["right"]}
        result["admission_refusal"] = detail
    plan = record.get("plan")
    if code in _ANSWERED_WITH_PLAN and isinstance(plan, Mapping):
        result["plan"] = _public_plan(plan)
    if code == "operator_declined":
        words = record.get("operator_says")
        result["operator"] = {"decision": "declined",
                              "reason": words.strip() if isinstance(words, str) and words.strip() else None}
    return result


def _rounded(value: Any, places: int) -> Any:
    if _finite(value):
        return round(float(value), places)
    if isinstance(value, Mapping):
        return {str(key): _rounded(child, places) for key, child in value.items()}
    if isinstance(value, list):
        return [_rounded(child, places) for child in value]
    return value if isinstance(value, (str, bool)) or value is None else None


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan summary without its schema: arms, duration, waypoints and the checked minima."""

    keys = ("arms", "gripper", "ticks", "duration_s", "joint_space", "start_tcp", "waypoints",
            "min_inter_arm_distance_m", "min_table_distance_m", "max_joint_travel_rad")
    return {key: _rounded(plan[key], 4) for key in keys if key in plan}


__all__ = ["STATION_RECEIPT_SCHEMA", "public_refusal_detail", "public_station_detail"]
