"""Bounded public summaries of the probe results supplied through the interface."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

_IK_CANDIDATE_COUNT_KEYS = ("returned", "valid_endpoint", "continuous")


def _ik_candidates(variants: list[Any]) -> dict[str, Any]:
    """Sum the per-variant IK candidate counts; empty when none were kept.

    A count that any variant with diagnostics left unknown (``None``, as
    RoboTwin's ``continuous`` is when it applies no joint-jump filter) is
    withheld rather than summed: a partial sum would read as a measured total.
    """
    counts: Counter[str] = Counter()
    unknown: set[str] = set()
    outcomes: Counter[str] = Counter()
    for variant in variants:
        planner = variant.get("planner") if isinstance(variant, Mapping) else None
        diagnostics = planner.get("planning_diagnostics") if isinstance(planner, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        if isinstance(diagnostics.get("outcome"), str):
            outcomes[diagnostics["outcome"]] += 1
        raw = diagnostics.get("candidate_counts")
        for key in _IK_CANDIDATE_COUNT_KEYS:
            value = raw.get(key) if isinstance(raw, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                counts[key] += value
            elif isinstance(raw, Mapping) and key in raw:
                unknown.add(key)
    if not outcomes and not counts:
        return {}
    return {"ik_candidate_counts": {key: counts[key] for key in _IK_CANDIDATE_COUNT_KEYS
                                    if key in counts and key not in unknown},
            "ik_outcome_counts": dict(outcomes)}


def summarize_motion_probe(
    result: Mapping[str, Any], *, ik_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Summarize every query without copying per-seed solver arrays.

    Counts name variants unless the worker explicitly supplies constraint
    rejection counts. They never pretend to count IK seeds from a final code.
    ``ik_candidates`` adds the summed IK candidate counts and outcomes a v2
    feedback probe kept; released summaries, whose closed schemas admit no
    such keys, never receive them.
    """
    summary = []
    for query in result.get("results", []):
        variants = query.get("variants", [])
        failures: Counter[str] = Counter()
        constraints: Counter[str] = Counter()
        for variant in variants:
            failure = variant.get("failure")
            if isinstance(failure, Mapping) and isinstance(failure.get("code"), str):
                failures[failure["code"]] += 1
            planner = variant.get("planner")
            counts = planner.get("constraint_rejection_counts") if isinstance(planner, Mapping) else None
            if isinstance(counts, Mapping):
                for name, count in counts.items():
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        constraints[str(name)] += count
        reachable = query.get("reachable") is True
        # The worker may give an explicit mixed/planner/validation stage.
        stage = query.get("failure_stage")
        if not reachable and stage is None:
            stage = "plan_search_or_admission"
        summary.append({
            "query_id": query["query_id"], "arm": query["arm"],
            "reachable": reachable, "failure_stage": None if reachable else stage,
            "admitted_count": sum(variant.get("reachable") is True for variant in variants),
            "variant_count": len(variants), "plan_ref": query.get("plan_ref"),
            "variant_failure_counts": dict(failures),
            **({"constraint_rejection_counts": dict(constraints)} if constraints else {}),
            **(_ik_candidates(variants) if ik_candidates else {}),
        })
    return summary
