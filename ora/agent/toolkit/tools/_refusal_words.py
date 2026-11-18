"""The words of a pre-motion refusal's public detail, the same in a receipt's summary line and in a dry run's answer.

``_dry_run.py`` writes them into an infeasible dry-run record; ``session_core/action_summary.py`` into the summary line
of a receipt's effect, so the two say a refusal alike.

A backend that refuses a move before it moves says why in structured fields: RoboCasa's ``reach_check``, projected as
``admission_refusal`` (``session_core/refusal_detail.py``) and handed to a dry run as its refusal's constraints, and the
planner's diagnostic. These sentences restate those
fields and nothing else: where a straight line's IK stopped and why, whether the target itself is reachable from
here, which robot part would enter what the depth shows and how far along, how far a target out of reach lies from
the arm mount, where the base's footprint or the arm it carries would meet the depth on a base move, and the nearest
base pose the base-placement search found (``controllers/base_placement.py``). A field a detail does not carry adds
no words. Every number is a rounding of one it carries.

``first_recovery`` is the planner's own first sentence, unless it is one of the fixed hints a planner gives when it
found no cause (``NO_CAUSE``): those say neither why nor what plans.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: The fixed hints a planner's recovery opens with when it found no cause: the shared planner; RoboTwin's vendored
#: planning (``ik_planning_diagnostics``); RoboDojo's SDK solve (``ik.SELECTION_RECOVERY``). The tests read each from
#: its source.
NO_CAUSE = frozenset({
    "Try a different target position or orientation, or an intermediate pose.",
    "No candidate survived this bounded exact-pose IK solve. Solver failure does not identify collision or prove "
    "the target unreachable. Inspect the reported filters and consider a different exact pose or intermediate pose.",
    "Candidate filter details were not recorded; the cause is unknown. Consider a different exact pose or "
    "intermediate pose; failure does not prove unreachability.",
    "IK candidates remain subject to trajectory and execution admission.",
    "No valid endpoint in this bounded solve; inspect candidate pose and try an alternative. This is not proof that "
    "the target is unreachable.",
})
_ENDPOINT = {
    "reachable": "The target itself is reachable from here: the line is what fails.",
    "reachable_only_through_the_world": "The target itself is reachable from here only through what the depth shows.",
    "unreachable": "The target itself is not reachable from here either.",
}


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _millimetres(metres: float) -> str:
    value = metres * 1000.0
    return f"{value:.0f} mm" if abs(value) >= 10 else f"{value:.1f} mm"


def _point(value: Any) -> str | None:
    if isinstance(value, list) and len(value) >= 3 and all(_finite(item) is not None for item in value[:3]):
        return "(" + ", ".join(f"{float(item):.3f}" for item in value[:3]) + ")"
    return None


def first_recovery(diagnostic: Any) -> str | None:
    """The first ``recovery`` sentence of a planning diagnostic, unless it is a fixed hint (``NO_CAUSE``)."""
    recovery = diagnostic.get("recovery") if isinstance(diagnostic, Mapping) else None
    first = recovery[0] if isinstance(recovery, list) and recovery else None
    return first if isinstance(first, str) and first and first not in NO_CAUSE else None


def _line(detail: Mapping[str, Any]) -> list[str]:
    fraction = _finite(detail.get("path_fraction"))
    along = f" {fraction:.0%} of the way" if fraction is not None else ""
    at = _point(detail.get("at_pose"))
    where = f"{along}, at {at}" if at else along
    cause = detail.get("cause")
    if cause == "ik_unsolved":
        return [f"The straight line has no IK solution{where}."]
    jump, bound, joint = _finite(detail.get("jump_rad")), _finite(detail.get("bound_rad")), detail.get("joint")
    if cause == "joint_branch_jump" and jump is not None and bound is not None and isinstance(joint, int):
        return [f"The straight line's IK changes branch{where}: joint {joint} jumps {math.degrees(jump):.1f} deg, "
                f"more than {math.degrees(bound):.1f} deg."]
    return []


#: What a base move carries that may meet the depth on its way, as a refusal's ``robot_part`` names it.
_CARRIED = {"arm": "The arm", "held_object": "The held object"}


def _entered(detail: Mapping[str, Any]) -> list[str]:
    part, fraction = detail.get("robot_part"), _finite(detail.get("path_fraction"))
    if not isinstance(part, str) or fraction is None:
        return []
    depth = _finite(detail.get("penetration_m"))
    if detail.get("path_kind") == "base":
        height, at = _finite(detail.get("obstacle_height_m")), detail.get("obstacle_xy_m")
        where = (f", {height:.3f} m above the floor" if height is not None else "") + (
            f" at ({float(at[0]):.3f}, {float(at[1]):.3f})" if isinstance(at, list) and len(at) == 2
            and all(_finite(item) is not None for item in at) else "")
        if part in _CARRIED:
            return [f"{_CARRIED[part]}, carried at the arm's joints, meets what the depth shows {fraction:.0%} of the "
                    f"way{where}: retract the arm before the base moves."]
        return [f"The base's footprint meets what the depth shows {fraction:.0%} of the way{where}."]
    by = f" by {_millimetres(depth)}" if depth is not None else ""
    kind = " along the line" if detail.get("path_kind") == "line" else ""
    return [f"The {' '.join(part.split('_'))} enters what the depth shows{by} {fraction:.0%} of the way{kind}."]


def _reach(detail: Mapping[str, Any]) -> list[str]:
    distance, reach = _finite(detail.get("distance_from_mount_m")), _finite(detail.get("reach_m"))
    if distance is None or reach is None:
        return []
    return [f"The target is {distance:.3f} m from the arm mount at the nearest height the lift allows, beyond its "
            f"{reach:.2f} m reach."]


def _suggested(detail: Mapping[str, Any]) -> list[str]:
    suggestions = detail.get("base_suggestions")
    searched = detail.get("base_search") if isinstance(detail.get("base_search"), Mapping) else {}
    if not isinstance(suggestions, list):
        return []
    found = [item for item in suggestions if isinstance(item, Mapping)
             and all(_finite(item.get(key)) is not None for key in ("x_m", "y_m", "yaw_deg"))]
    if not found:
        travel, asked, reached = (_finite(searched.get(key)) for key in ("max_travel_m", "asked", "ik_reached"))
        within = f" within {travel:.1f} m" if travel is not None else ""
        counts = (f" ({asked:.0f} asked of the IK, {reached:.0f} reached)" if asked is not None and reached is not None
                  else "")
        blocked = _finite(searched.get("arm_blocked"))
        carried = (f"; the arm, carried at its joints, would meet the depth on the way to {blocked:.0f} of them: "
                   "retract it first" if blocked else "")
        return [f"No base pose{within} reaches it clear of the depth seen now{counts}{carried}."]
    best = found[0]
    distance, turn, torso = (_finite(best.get(key)) for key in ("distance_m", "turn_deg", "torso_height_m"))
    away = (f" ({distance:.3f} m and {turn:.1f} deg from here)" if distance is not None and turn is not None else "")
    height = f", torso {torso:.3f} m" if torso is not None else ""
    stand = (f"Standing at x {float(best['x_m']):.3f}, y {float(best['y_m']):.3f}, yaw {float(best['yaw_deg']):.1f} deg"
             f"{away}{height}")
    if best.get("arm_path") == "admitted":
        words = [f"{stand}, this move is admitted as this call checks it, clear of the depth seen now."]
    else:
        path = {"planned": "; a path from the arm's current joints plans there",
                "not_planned": "; no path from the arm's current joints plans there"}.get(best.get("arm_path"), "")
        words = [f"{stand}, the arm reaches it clear of the depth seen now{path}."]
    if len(found) > 1:
        words.append(f"{len(found) - 1} more in base_suggestions.")
    return words


def admission_words(detail: Any) -> list[str]:
    """The sentences for one refusal's detail (``admission_refusal``, or a dry run's raw refusal constraints)."""
    if not isinstance(detail, Mapping):
        return []
    endpoint = _ENDPOINT.get(detail.get("endpoint")) if isinstance(detail.get("endpoint"), str) else None
    return [*_reach(detail), *_line(detail), *([endpoint] if endpoint else []), *_entered(detail), *_suggested(detail)]


def torso_words(torso: Any) -> str | None:
    """A move's lift, as the receipt's ``torso`` block measured it, or ``None``."""
    if not isinstance(torso, Mapping):
        return None
    start, end = _finite(torso.get("from_m")), _finite(torso.get("to_m"))
    if start is None or end is None:
        return None
    return f"torso moved from {start:.3f} to {end:.3f} m first"


def refusal_words(detail: Any, diagnostic: Any = None) -> str:
    """The planner's own first sentence, then the detail's: what a refusal says of itself, or ``""``."""
    said = first_recovery(diagnostic)
    return " ".join([*([said] if said else []), *admission_words(detail)])


__all__ = ["NO_CAUSE", "admission_words", "first_recovery", "refusal_words", "torso_words"]
