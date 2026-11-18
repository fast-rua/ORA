"""LIBERO's reading of a plan's captured IK seed populations.

The planner keeps every IK attempt's optimizer population (``detail.ik_attempts``),
which ``planner_port`` records. Constraint feasibility is aggregate: no per-term
collision cause is invented.
"""
from __future__ import annotations

from typing import Any, Mapping


#: The order an unaccepted CuRobo IK seed is attributed to one filter, in the
#: shared probe vocabulary (``contracts.planning_diagnostics.IK_CONSTRAINT_KEYS``):
#: a seed outside the raw joint position limits; else one the solver's own
#: constraint check marks infeasible (self-collision or the bound activation band,
#: which the SDK does not separate); else one that did not converge to the
#: pose within the solver's thresholds. ``joint_jump`` and ``pitch_joint_wrap``
#: are not LIBERO filters and stay zero.
TALLY_ORDER = ("joint_limit", "solver_unsuccessful", "fk_residual_exceeded")


def ik_candidate_tally(attempt: Any) -> dict[str, Any] | None:
    """Exclusive counts of one captured IK seed population, or None if none was captured.

    ``returned`` is every seed the solver optimised and ``valid_endpoint`` those
    both feasible and converged (what the solver accepts); every other seed is
    counted once, under the first ``TALLY_ORDER`` filter it fails, so the
    rejections sum to ``returned - valid_endpoint``. Seed arrays are captured
    finite by the planner, so ``non_finite_solution`` is zero.
    """
    from ora.contracts.planning_diagnostics import IK_CONSTRAINT_KEYS

    if not isinstance(attempt, Mapping) or attempt.get("status") != "captured":
        return None
    accepted, feasible, converged = (list(attempt[key]) for key in ("accepted", "feasible", "converged"))
    outside = attempt.get("outside_position_limits")
    outside = list(outside) if isinstance(outside, list) else [False] * len(accepted)
    if not len(accepted) == len(feasible) == len(converged) == len(outside):
        raise ValueError("captured IK seed arrays differ in length")
    rejections = {key: 0 for key in IK_CONSTRAINT_KEYS}
    for ok, fits, close, beyond in zip(accepted, feasible, converged, outside):
        if ok:
            continue
        rejections["joint_limit" if beyond else "solver_unsuccessful" if not fits
                   else "fk_residual_exceeded"] += 1
    return {"returned": len(accepted), "valid_endpoint": sum(bool(item) for item in accepted),
            "rejections": rejections, "limits_known": isinstance(attempt.get("outside_position_limits"), list)}


def public_summary(planned: Mapping[str, Any]) -> dict[str, Any] | None:
    from ora.contracts.planning_diagnostics import public_planning_diagnostics

    value = planned.get("planning_diagnostics")
    if not isinstance(value, Mapping):
        return None
    # The planner client validates this object. Never project the raw
    # per-seed positions or the canonical artifact's host path to the policy.
    return public_planning_diagnostics(value)
