"""move_pose's route policy as a planning rule (formerly inside the RoboDojo sparse executor).

Pure computation. The interaction label of a move_pose target chooses its
path, as the released executor did, at every speed (a speed only paces the path):

* ``free_space``: the shared planner's collision-free path to the target
  (``global``), or, where no planner runs, a joint-space line to one IK solution
  (canonical-first, up to a half-turn per joint), screened for self-collision
  and against the other arm;
* ``precision``: the straight Cartesian line when the whole change fits the
  local envelope (50 mm and 15 degrees), else as ``free_space``;
* ``contact_approach``: the straight line inside the envelope; beyond it, the
  ``free_space`` path to a pre-contact pose followed by the straight final
  segment (40 mm / 9.27 degrees) when the surface declares the staged
  approach, else refused (``route_contact_approach_exceeds_local_bound``).

What changed: every route now runs as one smooth ``follow_joint_trajectory``
(a staged approach as one trajectory with a stop at the hand-over), and the
straight line is the line itself, not one IK endpoint stepped through. A straight
line that cannot be followed (an IK hole, a branch change, a wrist unwinding) and
is no deliberate contact falls back to ``global`` (``LINE_FALLBACK_CODES``).
"""
from __future__ import annotations

import math

from .paths import Line

LOCAL_TRANSLATION_M = 0.05
LOCAL_ROTATION_RAD = math.radians(15.0)
#: The released staged approach's final local segment: the envelope less the arrival tolerance.
PRECONTACT_TRANSLATION_M = 0.05 - 0.01
PRECONTACT_ROTATION_RAD = math.radians(15.0) - 0.1
STAGED_CAPABILITY = "robodojo_staged_contact_approach_v1"
ROUTE_REFUSAL_CODE = "route_contact_approach_exceeds_local_bound"
#: The refusals of a straight line the global route can still reach the goal around.
LINE_FALLBACK_CODES = ("paced_path_unreachable", "paced_path_joint_jump", "paced_path_wrist_unwind")


def within_local_envelope(path: Line) -> bool:
    """Pure computation: whether a line fits the local envelope. Example: within_local_envelope(line(a, b))."""
    return path.length_m <= LOCAL_TRANSLATION_M + 1e-12 and path.angle_rad <= LOCAL_ROTATION_RAD + 1e-12


def choose(interaction: str, paths: dict[str, Line], *, staged_available: bool,
           global_available: bool = False) -> str:
    """Pure computation: ``global``, ``joint``, ``line``, ``staged`` or ``refused`` for one move_pose request.

    Example: choose('free_space', {'left': line(a, b)}, staged_available=True) == 'joint'."""
    local = all(within_local_envelope(path) for path in paths.values())
    free = "global" if global_available else "joint"
    if interaction == "free_space":
        return free
    if interaction == "precision":
        return "line" if local else free
    if local:
        return "line"
    return "staged" if staged_available else "refused"


__all__ = ["LINE_FALLBACK_CODES", "LOCAL_ROTATION_RAD", "LOCAL_TRANSLATION_M", "PRECONTACT_ROTATION_RAD", "PRECONTACT_TRANSLATION_M",
           "ROUTE_REFUSAL_CODE", "STAGED_CAPABILITY", "choose", "within_local_envelope"]
