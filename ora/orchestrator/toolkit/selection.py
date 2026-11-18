"""Select public before-call evidence without importing RSI or a simulator."""
from __future__ import annotations

from pathlib import Path

from ora.agent.evidence.agent_events import library_op
from ora.agent.toolkit.interface.blobs import IO_ROOT
from ora.agent.toolkit.interface.journal import read_events
from ora.agent.toolkit.interface.types import AmbiguousRecord, CorruptRecord, NotRecorded, decode, digest


def read_local(root: Path, name: str) -> bytes:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise CorruptRecord("evidence path escapes source run")
    return path.read_bytes()


def reference(root: Path, name: str) -> dict:
    data = read_local(root, name)
    return dict(path=name, sha256=digest(data), bytes=len(data), recorded=True)


def rows(root: Path, name: str) -> list[dict]:
    if not (root / name).exists():
        return []
    return [decode(line) for line in read_local(root, name).splitlines()]


def catalog(root: Path) -> tuple[list[dict], list[dict], dict | None]:
    """All parent proposals in event order (a code-mode script's library requests are no tool calls: left out), or
    the single explicit legacy snapshot.

    A gap takes precedence over recorded calls. An unbound gap is reported
    separately and prevents execution; absence of gaps is not a completeness claim.
    """
    if not root.is_dir():
        raise NotADirectoryError("RUN must be a source run directory")
    events = rows(root, "agent-events.jsonl")
    seqs = [e["seq"] for e in events]
    if any(type(s) is not int or s < 0 for s in seqs) or any(a >= b for a, b in zip(seqs, seqs[1:])):
        raise CorruptRecord("agent event sequence must be strictly increasing")
    proposals = [e for e in events if e["type"] == "tool_call_proposed" and not library_op(e)]
    snapshot = None
    if not proposals and (root / "snapshot.json").exists():
        snapshot = decode(read_local(root, "snapshot.json"))
        if snapshot.get("schema") != "rsi.traj.snapshot.v0" or snapshot["boundary"]["kind"] != "before_call":
            raise NotRecorded("未记录: snapshot is not a before-call boundary")
        request = snapshot["replay_evidence"]["tool_request"]
        proposals = [dict(run_id=snapshot["run_id"], **{
            key: snapshot["boundary"].get(key) for key in ("seq", "tool_call_id", "model_call_id")},
            payload=dict(name=request["tool"], arguments=request["parameters"]))]
    ids = [p["tool_call_id"] for p in proposals]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise AmbiguousRecord("parent tool_call_id must be unique and nonempty")
    calls = rows(root, f"{IO_ROOT}/calls.jsonl")
    gaps = rows(root, f"{IO_ROOT}/gaps.jsonl")
    for gap in gaps:
        if gap.get("schema") != "ora.toolkit.gap.v1" or "tool_call_id" not in gap:
            raise CorruptRecord("invalid gap evidence")
    result = []
    for k, proposal in enumerate(proposals):
        tool_id = proposal["tool_call_id"]
        result.append(dict(call=snapshot["boundary"]["call"] if snapshot else k,
                           tool_call_id=tool_id, tool=proposal["payload"]["name"],
                           has_toolkit_io=any(c["identity"]["tool_call_id"] == tool_id for c in calls),
                           has_gap=any(g["tool_call_id"] == tool_id for g in gaps),
                           proposal=proposal))
    return result, events, snapshot


def list_calls(root: Path) -> dict:
    calls, _, _ = catalog(root)
    ids = {c["tool_call_id"] for c in calls}
    gaps = rows(root, f"{IO_ROOT}/gaps.jsonl")
    return dict(schema="ora.toolkit.call-list.v1", source_run=str(root),
                calls=[{k: v for k, v in c.items() if k != "proposal"} for c in calls],
                unbound_gaps=sum(g["tool_call_id"] is None for g in gaps),
                unknown_parent_gaps=sum(g["tool_call_id"] is not None and g["tool_call_id"] not in ids for g in gaps))


def select(root: Path, at: str) -> tuple[dict, list[dict], dict | None]:
    calls, events, snapshot = catalog(root)
    matches = [c for c in calls if c["tool_call_id"] == at or str(c["call"]) == at]
    if len(matches) > 1:
        raise AmbiguousRecord("--at matches both an ordinal and another parent id")
    if not matches:
        raise NotRecorded("未记录: parent tool call not found")
    return matches[0], events, snapshot


def recorded_entries(root: Path, selected: dict, cached_path: Path) -> list[dict]:
    """Validate the journal, preserving its per-invocation attempts and order."""
    if not selected["has_toolkit_io"]:
        return []
    path = f"{IO_ROOT}/calls.jsonl"
    # Freeze bytes once: a live source may append while the experiment runs.
    cached_path.write_bytes(read_local(root, path))
    entries = [e for e in read_events(cached_path)
               if e["identity"]["tool_call_id"] == selected["tool_call_id"]]
    if not entries:
        raise NotRecorded("未记录: selected toolkit invocation disappeared during selection")
    if any(e["identity"]["run_id"] != selected["proposal"]["run_id"] for e in entries):
        raise CorruptRecord("toolkit journal run differs from parent proposal")
    return entries


def legacy_snapshot(root: Path, selected: dict, events: list[dict]) -> dict:
    """Recover only the latest publication strictly before this parent proposal.

    No final state, task verdict, controller row or later observation is used.
    Unknown probes/receipts stay unrecorded rather than inferred from filenames.
    """
    proposal = selected["proposal"]
    publications = [e for e in events if e["type"] == "fresh_observation_published" and e["seq"] < proposal["seq"]]
    if not publications:
        raise NotRecorded("未记录: no observation published before this call")
    publication = max(publications, key=lambda e: e["seq"])
    oid = publication["payload"]["observation_id"]
    base = f"policy-observations/{oid}"
    manifest_ref = reference(root, f"{base}/observation.json")
    manifest = decode(read_local(root, manifest_ref["path"]))
    if manifest["observation_id"] != oid:
        raise CorruptRecord("publication/manifest identity mismatch")
    files = []
    for camera in manifest.get("cameras", {}).values():
        for ref in camera["files"].values():
            name = f"{base}/{ref['path']}"
            files.append(dict(ref, path=name, recorded=(root / name).is_file()))
    if (root / base / "proprioception.json").is_file():
        files.append(reference(root, f"{base}/proprioception.json"))
    boundary = dict(kind="before_call", call=selected["call"], **{
        key: proposal.get(key) for key in ("seq", "tool_call_id", "model_call_id")})
    return dict(schema="rsi.traj.snapshot.v0", run_id=proposal["run_id"], source_run=str(root), boundary=boundary,
                identity=dict(tool_call_id=selected["tool_call_id"], round_index=manifest["round_index"]),
                available_at_boundary=dict(observation=dict(recorded=True, id=oid, published_seq=publication["seq"],
                                                            manifest=manifest_ref, files=files), previous_receipts=[]),
                replay_evidence=dict(tool_request=dict(tool=selected["tool"], parameters=proposal["payload"]["arguments"])))
