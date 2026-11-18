"""Where a joint trajectory stopped and which joint fell behind, from what the receipt measured.

A trajectory stop names the knot and the worst lag, in no joint. The receipt also keeps the
knots the arm was sent and how far each joint ended from the last one, so the commanded joints
at the stop and the measured joints after it give the lag joint by joint. A RoboCasa arm stop
on a joint held back (``endpoint.stop_reason`` ``joint_lag``, ``settle_stalled`` or
``settle_timeout``) records that joint and its lag itself, in ``endpoint``. Everything here is
the policy's own move and the arm's public joints. It says nothing about why the arm fell behind.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: A physics step's share of one native-action knot: the stop names the step inside its knot.
PHYSICS_STEPS_PER_KNOT = 10


def _commanded(knots: list[Any], knot: int, physics_step: Any) -> list[float] | None:
    """The joints commanded at stop ``knot`` (1 is the first of ``knots``), between that knot and the one before."""
    if not 1 <= knot <= len(knots):
        return None
    target = [float(value) for value in knots[knot - 1]]
    if knot == 1 or not isinstance(physics_step, int):
        return target
    before = [float(value) for value in knots[knot - 2]]
    share = min(physics_step, PHYSICS_STEPS_PER_KNOT) / PHYSICS_STEPS_PER_KNOT
    return [start + (end - start) * share for start, end in zip(before, target)]


def _endpoint_stop(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The same three, as an executor that names the joint itself records them in ``endpoint`` (RoboCasa).

    ``path_fraction`` is there only where the stop came before the last knot; a stop after it leaves the
    distance travelled to the measured translation."""
    endpoint, targets = record.get("endpoint"), record.get("target_arms")
    if not isinstance(endpoint, Mapping) or not isinstance(targets, list) or len(targets) != 1:
        return None
    joint, lag, fraction = endpoint.get("lagging_joint"), endpoint.get("lag_rad"), endpoint.get("path_fraction")
    if not (isinstance(joint, int) and not isinstance(joint, bool) and joint >= 1
            and isinstance(lag, (int, float)) and not isinstance(lag, bool) and math.isfinite(lag)):
        return None
    evidence: dict[str, Any] = {}
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and 0 <= fraction <= 1:
        evidence["path_fraction"] = round(float(fraction), 3)
    return str(targets[0]), {**evidence, "lagging_joint": joint, "lag_rad": round(float(lag), 6)}


def stop_evidence(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The stopped arm and its ``path_fraction``, ``lagging_joint`` (1 is the base joint) and ``lag_rad``, or ``None``."""
    if "trajectory" not in record:
        return _endpoint_stop(record)
    trajectory = record.get("trajectory")
    stop = trajectory.get("stop") if isinstance(trajectory, Mapping) else None
    if not isinstance(stop, Mapping) or stop.get("reason") not in ("tracking", "progress"):
        return None
    arm, planned, knot = stop.get("arm"), trajectory.get("knots_planned"), stop.get("knot")
    arguments = (record.get("arguments") or {}).get("arms") if isinstance(record.get("arguments"), Mapping) else None
    knots = (arguments.get(arm) or {}).get("knots_rad") if isinstance(arguments, Mapping) else None
    residual = ((trajectory.get("arms") or {}).get(arm) or {}).get("final_joint_error_per_joint_rad")
    if not (isinstance(planned, int) and planned and isinstance(knot, int) and isinstance(knots, list) and knots
            and isinstance(residual, list) and len(residual) == len(knots[-1])):
        return None
    commanded = _commanded(knots, knot, stop.get("physics_step"))
    if commanded is None:
        return None
    measured = [float(end) - float(left) for end, left in zip(knots[-1], residual)]
    lags = [abs(want - have) for want, have in zip(commanded, measured)]
    joint = max(range(len(lags)), key=lags.__getitem__)
    return str(arm), {"path_fraction": round(min(knot, planned) / planned, 3), "lagging_joint": joint + 1,
                      "lag_rad": round(lags[joint], 6)}


def with_stop_evidence(arms: Any, record: Mapping[str, Any]) -> Any:
    """``arms`` with the stopped arm's ``stop_evidence``, or as they were for a move that did not stop."""
    found = stop_evidence(record)
    if found is None or not isinstance(arms, Mapping) or found[0] not in arms:
        return arms
    return {**arms, found[0]: {**arms[found[0]], "stop_evidence": found[1]}}


__all__ = ["stop_evidence", "with_stop_evidence"]
