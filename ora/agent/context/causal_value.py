"""What a tool result may carry into the policy's provider request, and the privacy rules Python owns.

The causal context the agent core sends a provider keeps a tool result's
decision-relevant semantics and drops its audit provenance:
``causal_provider_value`` filters a value, ``_causal_projection_violations``
audits one. The key sets and checkpoint patterns here are also what
``ora.agent.pi.domain`` sends the TypeScript context policy
(``ora-pi/src/context/projection.ts``), which applies them; Python stays their
owner. The context profile, retention and summary-policy values are the ones a
RunSpec's native policy may name.

This is the part of the retired Python Context pipeline (``projection.py``) that
non-test code still reached; the pipeline itself is the TypeScript core's, held
to the recorded fixtures in ``tests/data/context_differential``.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping

from ora.agent.memory.markers import (
    CONTEXT_ARTIFACT_REF_SCHEMA,
    LOSSLESS_CONTEXT_RETENTION,
    RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
    validate_recoverable_artifact_envelope,
)


# The Context values this tree composes.
CONTEXT_PROFILES = {"native_minimal_v1", "native_causal_context_v1"}
CONTEXT_RETENTIONS = {"rewrite_compact_v1", LOSSLESS_CONTEXT_RETENTION}
SUMMARY_POLICIES = {"disabled", "rollout_checkpoint_v1"}
# Retentions that close physical ActionSteps; each takes an exact
# ``context_management`` and the native causal context.
ACTION_STEP_RETENTIONS = {LOSSLESS_CONTEXT_RETENTION}


_CAUSAL_FORBIDDEN_KEYS = {
    "schema", "schema_version", "action_id", "receipt_id",
    "run_id", "event_id", "causal_parent_id", "plan_id", "sha256", "hash",
    "path", "artifact_path", "timestamp", "timestamps", "started_unix_s",
    "finished_unix_s", "worker", "transport", "transport_status",
    "reached", "verified", "state_changed",
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome",
}

# Tool results retain operational provenance in canonical evidence, but the
# provider view needs only decision-relevant semantics.  Keep this policy next
# to the causal-context validator so result projection and final request audit
# cannot drift into separate hand-maintained tool-name projectors.
# Two different things were in one set. ``diagnostics`` is content a producer
# builds for the policy -- grasp.plan's candidate ledger lives there -- so a
# result schema that declares it makes it promised evidence. ``backend`` is the
# tool's own model identity, version and config hash: provenance the policy
# cannot act on, and sixty schema sites declare it as an artefact of how those
# schemas were written rather than as a promise. It stays stripped.
#
# EXPIRY: when the v4 surface generation lands, remove ``backend`` from those
# declarations and delete this split -- at that point a declaration means a
# promise everywhere and the special case is dead weight.
_CAUSAL_RESULT_PROVENANCE_KEYS = {"backend"}
_CAUSAL_RESULT_CONTENT_KEYS = {"diagnostics"}
_CAUSAL_RESULT_OPERATIONAL_KEYS = _CAUSAL_RESULT_PROVENANCE_KEYS | _CAUSAL_RESULT_CONTENT_KEYS


def _logical_artifact_identity(value: Mapping[str, Any]) -> bool:
    reference = value.get("ref")
    digest = value.get("sha256")
    return bool(
        isinstance(reference, str)
        and isinstance(digest, str)
        and re.fullmatch(
            r"artifact://"
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}",
            reference,
        )
        and reference.endswith("/" + digest)
        and "path" not in value
        and "artifact_path" not in value
    )


def _recoverable_artifact_envelope_identity(value: Mapping[str, Any]) -> bool:
    if value.get("schema") != RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA:
        return False
    try:
        validate_recoverable_artifact_envelope(value)
    except ValueError:
        return False
    return True


def causal_provider_value(value: Any, *, declared: frozenset[str] = frozenset()) -> Any:
    """Recursively remove audit/provenance fields from one tool-result value.

    ``declared`` names the top-level properties a public result schema already
    promises. Those survive the *operational* filter, which exists to drop
    provenance that leaks in undeclared, not to overrule a schema. Without it the
    three layers disagree: a producer builds ``diagnostics`` deliberately for the
    policy — ``grasp_plan_public_result`` even adds
    ``compiler_returned_candidate_count`` to it — the result schema declares the
    property, and the projection then removes the whole block, so the candidate
    ledger a policy would need in order to cite an untried candidate never
    arrives.

    The privilege filter is not relaxed. ``_CAUSAL_FORBIDDEN_KEYS`` still applies
    to declared properties, and ``declared`` is deliberately not propagated into
    nested values: a nested ``backend`` stays provenance rather than becoming
    promised evidence.
    """

    if isinstance(value, Mapping):
        logical_artifact = _logical_artifact_identity(value)
        recoverable_envelope = _recoverable_artifact_envelope_identity(value)
        return {
            str(key): causal_provider_value(child)
            for key, child in value.items()
            if (
                str(key) not in _CAUSAL_FORBIDDEN_KEYS
                or logical_artifact and str(key) in {"schema", "sha256"}
                or recoverable_envelope and str(key) == "schema"
            )
            and (
                str(key) not in _CAUSAL_RESULT_OPERATIONAL_KEYS
                or (str(key) in declared
                    and str(key) not in _CAUSAL_RESULT_PROVENANCE_KEYS)
            )
        }
    if isinstance(value, list):
        return [causal_provider_value(child) for child in value]
    return copy.deepcopy(value)


def _causal_projection_violations(value: Any, *, location: str = "messages") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        logical_artifact = _logical_artifact_identity(value)
        recoverable_envelope = _recoverable_artifact_envelope_identity(value)
        if logical_artifact and value.get("schema") not in {
            None, CONTEXT_ARTIFACT_REF_SCHEMA,
        }:
            violations.append(f"{location}.schema")
        if (
            value.get("schema") == RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA
            and not recoverable_envelope
        ):
            violations.append(f"{location}.schema")
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _CAUSAL_FORBIDDEN_KEYS and not (
                logical_artifact and key in {"schema", "sha256"}
                or recoverable_envelope and key == "schema"
            ):
                violations.append(f"{location}.{key}")
            violations.extend(
                _causal_projection_violations(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _causal_projection_violations(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            violations.extend(
                _causal_projection_violations(decoded, location=f"{location}<json>")
            )
    return violations


_COMPACTOR_PRIVATE_KEYS = {
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome", "official_success",
    "official_verdict", "task_success",
}


def _compactor_private_violations(
    value: Any, *, location: str = "packet",
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _COMPACTOR_PRIVATE_KEYS:
                violations.append(f"{location}.{key}")
            violations.extend(
                _compactor_private_violations(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _compactor_private_violations(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            violations.extend(
                _compactor_private_violations(
                    decoded, location=f"{location}<json>"
                )
            )
    return violations


_CHECKPOINT_FORBIDDEN = re.compile(
    r"(?:\brun[_ -]?id\b|\bsha-?256\b|"
    r"\b(?:official[_-](?:success|verdict|outcome)|task[_-]success)\b|"
    r"\bpixel coordinates?\b|\bworld coordinates?\b|"
    r"\b(?:target|historical) pose\b|"
    r"\b(?:pose_xyz|position_m|delta_mm|point_uv|xyz_m)\b|"
    r"\b[xyz]\s*(?:=|~)|"
    r"(?:/home/|outputs/|native/agent-artifacts/)|\b[0-9a-f]{64}\b)",
    re.IGNORECASE,
)

_CHECKPOINT_OUTCOME_ASSERTION = re.compile(
    r"\b(?:official[ ]+(?:success|verdict|outcome)|task[ ]+success)\b"
    r"\s*(?:(?:is|was|=|:|returned|reported|indicated|showed)\s*)"
    r"(?:true|false|success(?:ful)?|fail(?:ed|ure)?)\b",
    re.IGNORECASE,
)


def checkpoint_forbidden_fragments(value: Mapping[str, Any]) -> list[str]:
    """Return deterministic replay/audit fragments rejected from a checkpoint."""

    serialized = json.dumps(value, ensure_ascii=False)
    return sorted({
        *(match.group(0) for match in _CHECKPOINT_FORBIDDEN.finditer(serialized)),
        *(match.group(0) for match in _CHECKPOINT_OUTCOME_ASSERTION.finditer(serialized)),
    })


def episode_index_keyframe(
    *, step: int, previous_action: Mapping[str, Any] | None,
) -> bool:
    """Decide the ``keyframe`` flag of one observation's Harness metadata.

    The flag selected the steps the ``native_context_v1`` profile's collapsed
    ``<episode_index>`` highlighted. That profile composes only at the commits
    that composed it; the flag stays on the observation metadata and nothing
    it decides is stated to the policy.

    It also read the per-action runtime notice, keeping an observation whose
    guard raised `warning`, `error` or `stagnation`, or whose head image moved
    more than four hash bits. The notice no longer carries any of those: the
    guard's booleans were retired, and its surviving image distance is not
    routed to the policy context. Step cadence and the previous action are
    what remain readable here.
    """

    if step == 0 or step % 10 == 0:
        return True
    return bool(
        isinstance(previous_action, Mapping)
        and (
            previous_action.get("action") == "set_gripper"
            or isinstance(previous_action.get("arguments"), Mapping)
            and previous_action["arguments"].get("gripper") is not None
        )
    )
