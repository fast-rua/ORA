"""Manually numbered data-interface semantics, independent of source hashes."""
from __future__ import annotations

PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION,)
SCHEMA = "ora.toolkit.protocol.v1"
# Wildcards enumerate the affected public method families, including RSI-only
# privilege. Unknown versions cannot promise a narrower impact list.
METHOD_FAMILIES = ("perception.*", "state.*", "planner.*", "act.*", "episode.*",
                   "privileged.*", "artifacts.*", "compute.*", "context.*")


def metadata(*, recording_enabled: bool, implementation: dict | None = None) -> dict:
    return dict(schema=SCHEMA, protocol_version=PROTOCOL_VERSION,
                methods=list(METHOD_FAMILIES), recording_enabled=recording_enabled,
                implementation=implementation or {})


def compatibility(evidence: list[dict]) -> dict:
    """Version declarations are advisory; content/physical validation is separate."""
    evidence = evidence or [dict(location="toolkit-protocol.json", protocol_version=None)]
    versions = [entry.get("protocol_version") for entry in evidence]
    mismatch = any(value is not None and (type(value) is not int or value not in SUPPORTED_PROTOCOL_VERSIONS)
                   for value in versions)
    status = "mismatch" if mismatch else "unknown" if None in versions else "matched"
    warnings = []
    if status != "matched":
        warnings.append(dict(code="data_interface_protocol_" + status,
            source_versions=versions, current_version=PROTOCOL_VERSION,
            supported_versions=list(SUPPORTED_PROTOCOL_VERSIONS),
            affected_methods=list(METHOD_FAMILIES),
            reason="Protocol compatibility is unconfirmed; continue with current decoders and physical/evidence checks."))
    return dict(source_version=versions[0], current_version=PROTOCOL_VERSION,
                supported_versions=list(SUPPORTED_PROTOCOL_VERSIONS), status=status,
                protocol_matched=status == "matched", exact_replay_eligible=status == "matched",
                claim="Protocol agreement is necessary, not sufficient, for exact replay.",
                evidence=evidence, warnings=warnings)


def identity_warning(field, recorded, current, *, methods=METHOD_FAMILIES) -> dict:
    return dict(code="implementation_identity_mismatch", field=field,
                recorded=recorded, current=current, affected_methods=list(methods),
                reason="Implementation identity differs or is missing; continue and verify evidence/physical state.")
