"""Tool-local parallel-jaw headroom preference; inputs are measured joints and IK rows."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

CAPABILITY = "grasp_plan_jaw_symmetry_headroom_v1"

RULE = "reachable_then_larger_joint6_margin_then_smaller_joint_delta_then_compiled"

VARIANTS = ["exact", "jaw_flip"]

STAGES = ("pregrasp", "approach")

MARGIN_TIE_RAD = 0.001

DELTA_TIE_RAD = 0.001

CLAIM_BOUNDARY = (
    "Joint-limit margins of the backend's read-only IK solutions for the published pregrasp "
    "and contact poses at planning time; joint6 is the X5 wrist roll about the approach axis "
    "and is reported only when the measured joints prove it is the binding joint. An IK "
    "reference, not a carry guarantee: execution solves again, and lifting, carrying or "
    "rotating the wrist after the grasp moves every margin."
)

#: Joints 1-5 of the X5: the smaller of the URDF's 10 rad and the float32
#: articulation readback; joint6, the wrist roll about the approach axis.
ARX_X5_ARM_JOINT_LIMIT_RAD = 9.999999
ARX_X5_WRIST_ROLL_LIMIT_RAD = 3.14
#: A bound on |q| of joints 1-5 under which joint6 is the binding joint; the
#: 1 mrad covers the 0.01 mrad rounding of the published row numbers.
JOINT6_ISOLATION_BOUND_RAD = round(ARX_X5_ARM_JOINT_LIMIT_RAD - ARX_X5_WRIST_ROLL_LIMIT_RAD - 1e-3, 6)
#: Controlled frames whose wrist-roll table this module knows.
ISOLATION_BOUNDS = {"arx_x5_link6": JOINT6_ISOLATION_BOUND_RAD}
_EMPTY_HEADROOM = {stage: None for stage in STAGES}


def measured_arm_joints(observation: Any, arm: str) -> list[float] | None:
    """The six measured arm joints the observation publishes for ``arm``, or None."""
    proprioception = observation.get("proprioception") if isinstance(observation, Mapping) else None
    effectors = proprioception.get("end_effectors") if isinstance(proprioception, Mapping) else None
    record = effectors.get(arm) if isinstance(effectors, Mapping) else None
    joints = record.get("arm_joint_qpos") if isinstance(record, Mapping) else None
    if (not isinstance(joints, (list, tuple)) or len(joints) != 6
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in joints)):
        return None
    return [float(item) for item in joints]


def _stage_headroom(row: Mapping[str, Any], reach: float | None, bound: float | None) -> dict[str, Any]:
    margin = row.get("joint_limit_margin_rad")
    isolated = margin is not None and reach is not None and bound is not None and reach <= bound
    return {"min_joint_limit_margin_rad": margin, "joint6_margin_rad": margin if isolated else None}


def member_headroom(pregrasp: Mapping[str, Any], approach: Mapping[str, Any],
                    rule: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Both stages' margins for one member whose pregrasp and approach both solved.

    ``reach`` bounds |q| of joints 1-5 at each solution: the largest measured
    one plus every move along the chain that led there.
    """
    measured, bound = rule.get("measured_joints"), rule.get("isolation_bound_rad")
    reach = None if measured is None else max(abs(value) for value in measured[:5])
    chained = []
    for row in (pregrasp, approach):
        delta = row.get("joint_delta_rad")
        reach = None if reach is None or delta is None else reach + float(delta)
        chained.append(reach)
    return {stage: _stage_headroom(row, value, bound)
            for stage, row, value in zip(STAGES, (pregrasp, approach), chained)}


def _smallest(values: Sequence[Any]) -> float | None:
    return None if any(value is None for value in values) else min(float(value) for value in values)


def _compare(rows: Mapping[str, Sequence[Mapping[str, Any]]], joint6: Mapping[str, Any],
             headroom: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Both members solved: the larger margin, then the smaller move, then the compiled one."""
    isolated = all(joint6[name] is not None for name in VARIANTS)
    margin = joint6 if isolated else {
        name: _smallest([headroom[name][stage]["min_joint_limit_margin_rad"] for stage in STAGES])
        for name in VARIANTS}
    exact_margin, flip_margin = margin["exact"], margin["jaw_flip"]
    if (exact_margin is not None and flip_margin is not None
            and abs(exact_margin - flip_margin) > MARGIN_TIE_RAD):
        return ("exact" if exact_margin > flip_margin else "jaw_flip"), (
            "larger_joint6_margin" if isolated else "larger_joint_limit_margin")
    exact_move, flip_move = (rows[name][0].get("joint_delta_rad") for name in VARIANTS)
    if exact_move is not None and flip_move is not None and abs(exact_move - flip_move) > DELTA_TIE_RAD:
        return ("exact" if exact_move < flip_move else "jaw_flip"), "margin_tie_smaller_joint_delta"
    return "exact", "margin_and_joint_delta_tie"


def choose(rows: Mapping[str, Sequence[Mapping[str, Any]]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """The member to publish from each member's (pregrasp, approach) reach-check rows."""
    solved = [name for name in VARIANTS if rows[name][0]["reachable"] and rows[name][1]["reachable"]]
    headroom = {name: member_headroom(*rows[name], rule) for name in solved}
    joint6 = {name: (_smallest([headroom[name][stage]["joint6_margin_rad"] for stage in STAGES])
                     if name in headroom else None) for name in VARIANTS}
    if not solved:
        variant, reason = "exact", "no_reachable_variant"
    elif len(solved) == 1:
        variant, reason = solved[0], "only_reachable_variant"
    else:
        variant, reason = _compare(rows, joint6, headroom)
    return {
        "member": variant if solved else None,
        "jaw_symmetry": {"variant": variant, "reason": reason, "joint6_margin_rad": joint6},
        "joint_headroom": dict(headroom.get(variant, _EMPTY_HEADROOM)),
    }


def unavailable() -> dict[str, Any]:
    """The fields a candidate carries when the IK reference could not be taken."""
    return {
        "jaw_symmetry": {"variant": "exact", "reason": "ik_reference_unavailable",
                         "joint6_margin_rad": {name: None for name in VARIANTS}},
        "joint_headroom": dict(_EMPTY_HEADROOM),
    }


def summary() -> dict[str, str]:
    return {"rule": RULE, "claim_boundary": CLAIM_BOUNDARY}


