"""Recoverable-artifact markers: the envelope and pointer a Context payload is
replaced by, their schemas and validation, and the marker text that carries a
pointer through a provider message.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping

from .artifact_openers import ARTIFACT_READ_NAMES, recoverable_artifact_open


LOSSLESS_CONTEXT_RETENTION = "action_step_lossless_v4"
#: ``policy.context_tools``: the Context tools a context scheme provides.
#: ``artifact_read`` is the recoverable-artifact reader together with
#: ``observe(ref=...)``. A policy without the field (every baseline and scheme
#: before it) keeps the old rule: lossless retention provides the reader.
CONTEXT_TOOLS = frozenset({"artifact_read"})
CONTEXT_ARTIFACT_REF_SCHEMA = "ora.context-artifact-ref.v1"
RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA = (
    "ora.recoverable-artifact-envelope.v1"
)
RECOVERABLE_ARTIFACT_POINTER_SCHEMA = (
    "ora.recoverable-artifact-pointer.v1"
)
RECOVERABLE_ARTIFACT_TAG = "recoverable_artifact"
_REF_PATTERN = re.compile(
    r"artifact://(?P<scope>[A-Za-z0-9][A-Za-z0-9._-]{0,159})/"
    r"(?P<sha256>[0-9a-f]{64})"
)




def context_artifact_reader(policy: Mapping[str, Any]) -> bool:
    """Whether the plan advertises the recoverable-artifact reader tools."""

    if policy.get("context_retention") != LOSSLESS_CONTEXT_RETENTION:
        return False
    tools = policy.get("context_tools")
    return tools is None or "artifact_read" in tools


def recoverable_artifact_envelope(
    reference: Mapping[str, Any], *, source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode one policy-visible, path-free recovery instruction."""

    ref = reference.get("ref")
    sha256 = reference.get("sha256")
    media_type = reference.get("media_type")
    byte_count = reference.get("bytes")
    policy_scope = reference.get("policy_scope")
    kind = reference.get("kind")
    if (
        not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(policy_scope, str)
        or not isinstance(kind, str)
        or ref != f"artifact://{policy_scope}/{sha256}"
    ):
        raise ValueError("recoverable Artifact reference is invalid")
    envelope = {
        "schema": RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
        "artifact": {
            "ref": ref,
            "sha256": sha256,
            "kind": kind,
            "media_type": media_type,
            "bytes": byte_count,
            "policy_scope": policy_scope,
            "round": reference.get("round"),
            "action_step": reference.get("action_step"),
        },
        "source": copy.deepcopy(dict(source or {})),
        "open": recoverable_artifact_open(reference),
    }
    validate_recoverable_artifact_envelope(envelope)
    return envelope


def validate_recoverable_artifact_envelope(
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on drift between the reference and its advertised decoder."""

    if set(envelope) != {"schema", "artifact", "source", "open"}:
        raise ValueError("recoverable Artifact envelope fields are invalid")
    artifact = envelope.get("artifact")
    source = envelope.get("source")
    opener = envelope.get("open")
    if (
        envelope.get("schema") != RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA
        or not isinstance(artifact, Mapping)
        or set(artifact) != {
            "ref", "sha256", "kind", "media_type", "bytes", "policy_scope",
            "round", "action_step",
        }
        or not isinstance(source, Mapping)
        or not isinstance(opener, Mapping)
        or set(opener) != {"tool", "arguments"}
        or not isinstance(opener.get("arguments"), Mapping)
        or set(opener["arguments"]) != {"ref"}
    ):
        raise ValueError("recoverable Artifact envelope is malformed")
    ref = artifact.get("ref")
    sha256 = artifact.get("sha256")
    media_type = artifact.get("media_type")
    byte_count = artifact.get("bytes")
    scope = artifact.get("policy_scope")
    if (
        not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(artifact.get("kind"), str)
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(scope, str)
        or ref != f"artifact://{scope}/{sha256}"
        or not isinstance(opener.get("tool"), str)
        or opener.get("tool") not in (
            {"observe"} if media_type.startswith("image/") else ARTIFACT_READ_NAMES
        )
        or opener["arguments"].get("ref") != ref
    ):
        raise ValueError("recoverable Artifact envelope identity drift")
    for field in ("round", "action_step"):
        value = artifact.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"recoverable Artifact {field} is invalid")
    return copy.deepcopy(dict(envelope))


def recoverable_artifact_pointer(
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode only non-derivable recovery fields for one Artifact."""

    ref = reference.get("ref")
    sha256 = reference.get("sha256")
    policy_scope = reference.get("policy_scope")
    media_type = reference.get("media_type")
    byte_count = reference.get("bytes")
    match = _REF_PATTERN.fullmatch(ref) if isinstance(ref, str) else None
    if (
        not isinstance(ref, str)
        or match is None
        or sha256 != match.group("sha256")
        or policy_scope != match.group("scope")
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ValueError("recoverable Artifact pointer source is invalid")
    pointer = {
        "v": 1,
        "ref": ref,
        "mime": media_type,
        "bytes": byte_count,
        "at": [reference.get("round"), reference.get("action_step")],
        "open": recoverable_artifact_open(reference)["tool"],
    }
    return validate_recoverable_artifact_pointer(pointer)


def validate_recoverable_artifact_pointer(
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on one compact, directly callable Artifact pointer."""

    if set(pointer) != {"v", "ref", "mime", "bytes", "at", "open"}:
        raise ValueError("recoverable Artifact pointer fields are invalid")
    ref = pointer.get("ref")
    media_type = pointer.get("mime")
    byte_count = pointer.get("bytes")
    location = pointer.get("at")
    opener = pointer.get("open")
    if (
        pointer.get("v") != 1
        or not isinstance(ref, str)
        or _REF_PATTERN.fullmatch(ref) is None
        or not isinstance(media_type, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(location, list)
        or len(location) != 2
        or not isinstance(opener, str)
        or opener not in (
            {"observe"} if media_type.startswith("image/") else ARTIFACT_READ_NAMES
        )
    ):
        raise ValueError("recoverable Artifact pointer is malformed")
    for field, value in zip(("round", "action_step"), location, strict=True):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"recoverable Artifact pointer {field} is invalid")
    return copy.deepcopy(dict(pointer))


def _pointer_from_marker_value(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") == RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA:
        envelope = validate_recoverable_artifact_envelope(value)
        return recoverable_artifact_pointer({
            **envelope["artifact"], "open": envelope["open"],
        })
    return validate_recoverable_artifact_pointer(value)


def encode_recoverable_artifact_marker(value: Mapping[str, Any]) -> str:
    validated = _pointer_from_marker_value(value)
    return (
        f"<{RECOVERABLE_ARTIFACT_TAG}>"
        + json.dumps(
            validated, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        + f"</{RECOVERABLE_ARTIFACT_TAG}>"
    )


def decode_recoverable_artifact_marker(value: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("recoverable Artifact marker must be text")
    opening = f"<{RECOVERABLE_ARTIFACT_TAG}>"
    closing = f"</{RECOVERABLE_ARTIFACT_TAG}>"
    if not value.startswith(opening) or not value.endswith(closing):
        raise ValueError("recoverable Artifact marker framing is invalid")
    body = value[len(opening):-len(closing)]
    try:
        pointer = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("recoverable Artifact marker JSON is invalid") from exc
    if not isinstance(pointer, Mapping):
        raise ValueError("recoverable Artifact marker must contain an object")
    return _pointer_from_marker_value(pointer)
