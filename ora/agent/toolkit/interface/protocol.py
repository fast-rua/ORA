"""Read-only protocol audit; never backfill a source run or replace its hashes."""
import json
from pathlib import Path

from ora.contracts.toolkit_protocol import compatibility, identity_warning
from .blobs import IO_ROOT


def audit(root: Path, *, events=None) -> dict:
    root = Path(root)
    evidence, warnings = [], []
    for name in ("toolkit-protocol.json", f"{IO_ROOT}/manifest.json"):
        path = root / name
        if not path.exists() and name != "toolkit-protocol.json":
            continue
        try:
            record = json.loads(path.read_bytes()) if path.exists() else {}
            version = record.get("protocol_version")
        except (OSError, ValueError, AttributeError) as error:
            version = None
            warnings.append(dict(code="protocol_metadata_unreadable", location=name,
                                 error=f"{type(error).__name__}: {error}"))
        evidence.append(dict(location=name, protocol_version=version))
    if events is None:
        path = root / IO_ROOT / "calls.jsonl"
        # The normal reader still checks the journal for truncation/corruption.
        # Here only advisory metadata is collected, including on failed replay.
        try:
            events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        except (OSError, ValueError):
            events = []
    seen, method_versions = set(), set()
    for event in events:
        if not isinstance(event, dict):
            continue  # Normal readers reject malformed evidence; audit is advisory.
        version = event.get("protocol_version")
        key = json.dumps(version, sort_keys=True)
        if key not in seen:
            seen.add(key)
            evidence.append(dict(location=f"{IO_ROOT}/calls.jsonl", protocol_version=version))
        request = event.get("request")
        identity = request.get("identity") if isinstance(request, dict) else None
        if not isinstance(identity, dict):
            continue
        method, version = identity.get("method"), identity.get("method_version")
        key = (method, json.dumps(version, sort_keys=True))
        if (type(version) is not int or version != 1) and key not in method_versions:
            method_versions.add(key)
            warnings.append(identity_warning("method_version", version, 1, methods=[method]))
    result = compatibility(evidence)
    result["warnings"].extend(warnings)
    if warnings:
        result["exact_replay_eligible"] = False
    return result


def same_request(left, right):
    """Compare semantic fields; a decodable method-version label is advisory.

    Original semantic hashes are still verified by read_events. No evidence,
    action parameter, publication, service identity or input hash is rewritten.
    """
    return ({key: value for key, value in left.items() if key != "method_version"}
            == {key: value for key, value in right.items() if key != "method_version"})
