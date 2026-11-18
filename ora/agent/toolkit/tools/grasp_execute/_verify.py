"""Read-only verification of supplied measurements; no hidden worker calls or rendering."""
from typing import Any, Mapping, Sequence
from . import _localization as localization, _jaw_hold as jaw_hold

def measure(
    stage: str, observation: Mapping[str, Any],
    commitment: Mapping[str, Any], completed: Sequence[Mapping[str, Any]] = (),
    *, hold_signal: bool = True,
) -> dict[str, Any]:
    """Measure a read-only stage against the freshly captured observation."""

    effectors = observation.get("proprioception", {}).get("end_effectors", {})
    record = effectors.get(commitment["arm"]) if isinstance(effectors, Mapping) else None
    evidence = localization.closure_evidence(
        gripper_opening_m=localization.measured_jaw_separation_m(record),
        localization=commitment.get("localization"),
    )
    if hold_signal and evidence["strength"] == "unavailable":
        evidence["jaw_hold"] = jaw_hold.verify_stage_evidence(completed, commitment["arm"], record)
    semantics = commitment.get("semantics")
    semantics = (
        semantics if isinstance(semantics, Mapping)
        else dict(phrase=None, score=None, observation_id=None, selection="not_grounded_from_a_phrase")
    )
    return {
        "kind": "measurement", "status": "executed",
        "reason": f"verified_against_{observation.get('observation_id')}",
        "measurements": [
            {"quantity": "closure_strength", "value": evidence["strength"]},
            *([{"quantity": "jaw_hold", "value": evidence["jaw_hold"]["held"]}] if "jaw_hold" in evidence else []),
            {"quantity": "gripper_opening_m", "value": evidence["gripper_opening_m"]},
            {
                "quantity": "target_largest_extent_m",
                "value": evidence["target_largest_extent_m"],
            },
            # What the policy named, beside what the jaws measured. Neither is
            # a verdict on the other, and the pair is the whole evidence for a
            # question the geometry cannot reach.
            {"quantity": "committed_target_phrase", "value": semantics["phrase"]},
            {"quantity": "committed_target_score", "value": semantics["score"]},
            {
                "quantity": "committed_target_observation_id",
                "value": semantics["observation_id"],
            },
            {"quantity": "committed_target_selection", "value": semantics["selection"]},
        ],
        "verification": evidence,
    }
