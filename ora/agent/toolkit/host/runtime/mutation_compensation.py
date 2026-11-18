"""Undo the action a failed mutation call persisted, so the observation stays open.

A mutation handler persists ``submitted-action.json`` before its result is
validated, and closes the current observation as it does (``_persist_action``
sets the pointer's ``open`` to false). A failure after that point -- the result
schema, a post-submit annotation -- used to leave the action persisted, which
closes the current observation for the rest of the turn, and was not logged.
The runtime takes a ``snapshot`` before the handler and, on any failure of a
mutation call, ``restore`` removes only the action this call created, its entry
in the submitted-mutation ledger, and the close of the pointer it made. An
action already pending before the call is kept, so a refused second mutation
cannot delete the first. A pointer that changed in any other way during the
call -- a fresh observation published after an executed action, a round end --
is left as it is. Holding-gate transitions are left as recorded: a refused
close deliberately records what it consumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

POINTER = "current-observation.json"


@dataclass(frozen=True)
class Snapshot:
    action_path: Path | None
    action_existed: bool
    submitted_count: int
    #: The observation pointer's bytes as the call found them, for a mutation call.
    pointer: bytes | None = None


def snapshot(host: Any, name: str, arguments: Mapping[str, Any]) -> Snapshot:
    control_root = getattr(host, "control_root", None)
    try:
        mutation = host.classify_call_effect(name, arguments) == "mutation"
    except Exception:  # noqa: BLE001 - an unclassifiable call submits nothing
        mutation = False
    action_path = control_root / "submitted-action.json" if mutation and isinstance(control_root, Path) else None
    pointer = action_path.parent / POINTER if action_path is not None else None
    return Snapshot(
        action_path=action_path,
        action_existed=action_path is not None and action_path.exists(),
        submitted_count=len(getattr(host, "_submitted_mutations", ()) or ()),
        pointer=pointer.read_bytes() if pointer is not None and pointer.is_file() else None,
    )


def restore(host: Any, taken: Snapshot) -> None:
    """Remove the action, ledger entry and pointer close the failed call made; keep anything older."""

    if taken.action_path is None:
        return
    if not taken.action_existed:
        taken.action_path.unlink(missing_ok=True)
        _reopen(taken)
    submitted = getattr(host, "_submitted_mutations", None)
    if isinstance(submitted, list):
        del submitted[taken.submitted_count:]


def _reopen(taken: Snapshot) -> None:
    """Put the pointer's bytes back when the only change since the snapshot is the close of an open pointer."""

    if taken.pointer is None:
        return
    path = taken.action_path.parent / POINTER
    try:
        before, now = json.loads(taken.pointer), json.loads(path.read_bytes())
    except (OSError, ValueError):
        return
    if not isinstance(before, dict) or before.get("open") is not True or now != {**before, "open": False}:
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(taken.pointer)
    os.replace(temporary, path)
