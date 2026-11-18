"""Import snapshot.v0 without importing RSI or accessing a live simulator."""
from __future__ import annotations

import json
from pathlib import Path

from .blobs import BlobStore
from .journal import Journal, read_events
from .legacy import LegacyReplay, read_snapshot_reference
from .replay import Replay
from .runtime import Runtime
from .types import AccessDenied, CallContext, CorruptRecord, NotRecorded


def open_replay(*, run_root: Path, snapshot: dict, output_root: Path,
                services: dict, bindings: dict, allow_check_task: bool = False):
    """A source run, explicit boundary, and external-only service capabilities.

    Native-action snapshots remain useful to trajectory readers but have no
    interface cursor in v1. Output is separate; source evidence is never edited.
    """
    if snapshot.get("schema") != "rsi.traj.snapshot.v0":
        raise CorruptRecord("unsupported snapshot schema")
    if snapshot["boundary"].get("kind") != "before_call":
        raise NotRecorded("未记录: native-action boundary cannot locate interface cursor")
    run_root, output_root = Path(run_root).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(run_root) or run_root.is_relative_to(output_root):
        raise AccessDenied("source run and replay output must be disjoint")
    store = BlobStore(run_root)
    index = snapshot.get("toolkit_io")
    journal_path = None
    if index:
        if index.get("schema") != "ora.toolkit.snapshot-index.v1":
            raise CorruptRecord("unsupported snapshot index schema")
        if index.get("next_interface_ordinal") != 0 or index.get("exact_interface_replay") is not True:
            raise NotRecorded("未记录: snapshot does not establish a complete call boundary")
        ref = index["calls"]
        log_root = Path(index.get("snapshot_root", run_root)).resolve()
        read_snapshot_reference(log_root, ref)
        journal_path = (log_root / ref["path"]).resolve()
        selected = [event for event in read_events(journal_path)
                    if event["identity"]["tool_call_id"] == index["selected_tool_call_id"]
                    and event["identity"]["round_index"] == index["selected_round_index"]
                    and event["identity"]["run_id"] == snapshot["run_id"]]
        if not selected:
            raise NotRecorded("未记录: selected invocation is absent")
        first = selected[0]
        identity = first["identity"]
        if identity["tool_call_id"] != snapshot["identity"]["tool_call_id"]:
            raise CorruptRecord("snapshot tool identity differs from index")
        if identity["boundary"] != snapshot["boundary"]:
            raise CorruptRecord("snapshot boundary differs from recorded boundary")
        state = first["state_before"]
        permitted = first["request"]["identity"]["capabilities"]["check_task"]
        if permitted and not allow_check_task:
            raise AccessDenied("recorded call requires explicit check_task capability")
        context = CallContext.create(run_id=identity["run_id"], round_index=identity["round_index"],
                                     tool_call_id=identity["tool_call_id"], boundary=identity["boundary"],
                                     state=state, check_task=permitted)
    else:
        observation = snapshot["available_at_boundary"]["observation"]
        if not observation.get("recorded"):
            raise NotRecorded("未记录: no published observation")
        state = dict(observation=dict(observation_id=observation["id"],
                                      observation_sha256=observation["manifest"]["sha256"]))
        context = CallContext.create(run_id=snapshot["run_id"], round_index=snapshot["identity"]["round_index"],
                                     tool_call_id=snapshot["identity"]["tool_call_id"], boundary=snapshot["boundary"],
                                     state=state, check_task=allow_check_task)
    evaluation_context = CallContext.create(run_id=f"replay:{snapshot['run_id']}", round_index=context.json()["round_index"],
                                            tool_call_id=context.json()["tool_call_id"], boundary=context.json()["boundary"],
                                            state=state, check_task=context.json()["check_task"])
    evaluation = Runtime(context=evaluation_context, store=BlobStore(output_root, writable=True),
                         journal=Journal(output_root), adapters=services, bindings=bindings)
    if journal_path is not None:
        # Validate and register only the selected publication's public inputs.
        LegacyReplay(snapshot=snapshot, context=context, store=store,
                     evaluation=evaluation, bindings=bindings)
        return Replay(context=context, store=store, services=services, bindings=bindings,
                      evaluation=evaluation, journal_path=journal_path)
    return LegacyReplay(snapshot=snapshot, context=context, store=store, evaluation=evaluation, bindings=bindings)
