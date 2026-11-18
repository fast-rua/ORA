"""Public detail of how a move was paced: its speed, the wrist report, or the pacing's refusal.

A backend that paces its routes (``ora/environment/controllers/motion_pace.py``;
LIBERO, RoboTwin, RoboCasa365 and RoboLab) reports, per move, the requested
speed with its caps and the planned peaks, the wrist joint's extremes along the
planned path (RoboTwin: per moved arm) and, for a route refused before anything
moved, the refusal's numbers. A sparse receipt carries that as ``motion``, a
LIBERO hybrid receipt (``recover_pose``) as ``pacing``. It crosses only where
the surface publishes the speed options (``end_effector_speed_v1``), whose help
names these fields.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

#: A surface whose pose tools take a speed (``ora/agent/toolkit/tools/move_pose/_smooth.py``).
CAPABILITY = "end_effector_speed_v1"
#: The pacing's refusals: ``plan_rejected`` at stage ``pacing``, the code kept as it is.
PACING_REFUSAL_CODES = frozenset({"paced_path_wrist_unwind", "paced_motion_budget"})
_MOTION_KEYS = ("speed", "wrist", "planned_ticks", "duration_s")


def refusal_stage(reason: Any) -> str:
    """The failure stage of a refused route: ``pacing`` for a pacing refusal, else ``planner``."""

    return "pacing" if reason in PACING_REFUSAL_CODES else "planner"


def public_motion_pacing(record: Mapping[str, Any], context: Any, *, key: str = "motion") -> dict[str, Any]:
    """``{"motion": ...}`` from the receipt's ``key``, or nothing.

    The refusal keeps its numbers; its code is the effect's ``failure_code`` and the help states its rule."""

    motion = record.get(key)
    if not context.supports(CAPABILITY) or not isinstance(motion, Mapping):
        return {}
    public = {name: copy.deepcopy(motion[name]) for name in _MOTION_KEYS if name in motion}
    refused = motion.get("refused")
    if isinstance(refused, Mapping):
        public["refused"] = {name: copy.deepcopy(value) for name, value in refused.items()
                             if name not in {"code", "message"}}
    return {"motion": public} if public else {}


__all__ = ["CAPABILITY", "PACING_REFUSAL_CODES", "public_motion_pacing", "refusal_stage"]
