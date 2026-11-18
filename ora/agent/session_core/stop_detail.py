"""How an arm or mobile-base move ended, as its executor named the stop, for the unified sparse projection.

RoboCasa and RoboLab keep the name of the stop that ended an arm move in the receipt's
``endpoint.stop_reason``: an arrival (RoboCasa ``settled``, RoboLab ``reached``), a contact stop
(``contact_detected``), a stop on a joint held back (RoboCasa ``joint_lag``, ``settle_stalled`` and
``settle_timeout``; RoboLab ``no_progress``), the budget or the episode. ``failure_code`` folds several of
these into one public code (``blocked`` on RoboCasa covers three), and a contact stop is ``reached``
wherever it stopped, so the name crosses as it is: ``detail.endpoint.stop_reason``. RoboCasa's
``move_base`` keeps its own in ``base.stop_reason`` (``reached``, ``base_contact``, ``no_progress``,
``budget_exhausted``, ``episode_ended``), which crosses beside the base's measured result.

The executor's other bookkeeping (knot counts, ticks) and every force stay Control-only. The joint a
RoboCasa stop names reaches ``detail.arms.<arm>.stop_evidence`` (:mod:`.stop_evidence`).
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _stop_reason(block: Any) -> str | None:
    reason = block.get("stop_reason") if isinstance(block, Mapping) else None
    return reason if isinstance(reason, str) and reason else None


def public_endpoint_stop(record: Mapping[str, Any]) -> dict[str, Any]:
    """``{"endpoint": {"stop_reason": ...}}`` where the receipt names how an arm move ended, else nothing."""

    reason = _stop_reason(record.get("endpoint"))
    return {"endpoint": {"stop_reason": reason}} if reason else {}


def public_base_detail(record: Mapping[str, Any]) -> dict[str, Any]:
    """A mobile-base move's measured result and how it ended (RoboCasa ``move_base``), or nothing."""

    base = record.get("base")
    if not isinstance(base, Mapping):
        return {}
    detail: dict[str, Any] = {}
    reason = _stop_reason(base)
    if reason:
        detail["stop_reason"] = reason
    moved = base.get("moved")
    if isinstance(moved, Mapping):
        detail["moved"] = {
            "x_m": round(float(moved.get("x_m", 0.0)), 4),
            "y_m": round(float(moved.get("y_m", 0.0)), 4),
            "yaw_deg": round(math.degrees(float(moved.get("yaw_rad", 0.0))), 2),
        }
    error = base.get("base_error")
    if isinstance(error, Mapping):
        detail["remaining_error"] = {
            "position_m": round(float(error.get("position_m", 0.0)), 4),
            "yaw_deg": round(math.degrees(float(error.get("yaw_rad", 0.0))), 2),
        }
    after = record.get("base_after")
    pose = after.get("pose_xy_yaw") if isinstance(after, Mapping) else None
    if isinstance(pose, list) and len(pose) == 3:
        detail["pose_after"] = {"x_m": round(float(pose[0]), 4), "y_m": round(float(pose[1]), 4),
                                "yaw_deg": round(math.degrees(float(pose[2])), 2)}
    return {"base": detail} if detail else {}


__all__ = ["public_base_detail", "public_endpoint_stop"]
