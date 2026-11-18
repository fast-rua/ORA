"""What a receipt's copy for the policy leaves out: the evidence that names an implementation.

A backend's receipt is sealed evidence (``native/receipts/``) and says how a move was
made: which timing timed it (``motion.timing``, ``pacing.timing``: the implementation,
its library versions, the jerk limits' source) and which planner record planned it
(``planner.schema``, ``planner.canonical_artifact``: the planner's own record and the
path of its plan file; the YAM station's ``motion.planned_by``, which planner planned each arm). Nothing the policy can read names a library, so every copy of a
receipt that reaches a place the policy can read -- the observation an environment
publishes, the policy-state chain, the context a session reads, a script's robot view --
is ``policy_receipt`` of it. The public projections never read these keys, so what the
model is delivered does not change.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Per parent key, the child keys that are implementation evidence only: a paced move's timing ...
TIMING_EVIDENCE: dict[str, frozenset[str]] = {"motion": frozenset({"timing"}), "pacing": frozenset({"timing"})}
#: ... and the planner's own record (its schema, the path and hash of its plan file; the YAM station's per-arm
#: ``motion.planned_by``).
PLANNER_EVIDENCE: dict[str, frozenset[str]] = {"planner": frozenset({"schema", "canonical_artifact"}),
                                               "motion": frozenset({"planned_by"})}
IMPLEMENTATION_EVIDENCE: dict[str, frozenset[str]] = {
    key: TIMING_EVIDENCE.get(key, frozenset()) | PLANNER_EVIDENCE.get(key, frozenset())
    for key in (*TIMING_EVIDENCE, *PLANNER_EVIDENCE)}


def is_implementation_evidence(parent_key: str | None, key: str,
                               evidence: Mapping[str, frozenset[str]] = IMPLEMENTATION_EVIDENCE) -> bool:
    return key in evidence.get(parent_key or "", frozenset())


def policy_receipt(value: Any, *, parent_key: str | None = None,
                   evidence: Mapping[str, frozenset[str]] = IMPLEMENTATION_EVIDENCE) -> Any:
    """A copy of a receipt (or of any value holding one) without implementation evidence; never edits ``value``.

    Example: policy_receipt({"motion": {"speed": {}, "timing": {}}}) == {"motion": {"speed": {}}}."""

    if isinstance(value, Mapping):
        return {key: policy_receipt(child, parent_key=str(key), evidence=evidence) for key, child in value.items()
                if not is_implementation_evidence(parent_key, str(key), evidence)}
    if isinstance(value, list):
        return [policy_receipt(item, parent_key=parent_key, evidence=evidence) for item in value]
    return value


def names_timing(value: Any) -> bool:
    """Whether a value still holds a paced move's timing evidence (a copy the policy must not read)."""

    return policy_receipt(value, evidence=TIMING_EVIDENCE) != value


__all__ = ["IMPLEMENTATION_EVIDENCE", "PLANNER_EVIDENCE", "TIMING_EVIDENCE", "is_implementation_evidence",
           "names_timing", "policy_receipt"]
