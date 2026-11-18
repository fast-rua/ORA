"""Every change of the worker's IK seed generator state, independent of toolkit I/O.

The RoboDojo worker reports ``exec.ik_sampler_state`` (the CuRobo seed
generator) in the execution state of every RPC response; any request that
solves IK advances it: ``ik_chain``, but equally ``reach_check``,
``align_plan``, ``probe_motion`` and every motion action. Publication
snapshots hold it only at publications, and toolkit I/O (off by default) only
inside the answers that carry seeding. This ledger keeps the value after each
request that changed it, so the run's evidence holds the generator state in
effect right before any tool call, not only at publications. Nothing restores
from it: a restore resumes the generators from the boundary snapshot's
``exec.ik_sampler_state``.

``state/exec-ledger.jsonl`` (``ora.toolkit.exec-ledger.v1``) is append-only
and hash-chained like ``state/tool-deltas.jsonl``: each row has ``sequence``,
``previous_sha256`` and ``sha256`` over its own body. Rows:

* ``kind: "bound"``: the value in effect when the owner bound the ledger
  (before the run's first tool call or publication; ``null`` when none);
* ``kind: "change"``: the value after the RPC request ``method`` changed it,
  with ``previous_value_sha256``.

Each row also carries ``key``, ``value`` and ``value_sha256`` (sha256 of the
canonical value bytes), the tool scope it was written in (``context``: the
parent ``model_call_id``/``tool_call_id``, an owner boundary, or ``null``
outside any), ``interface_ordinal`` (the data-interface ordinal of the call in
progress inside a parent, else ``null``) and ``tool_delta_count``: how many
``state/tool-deltas.jsonl`` rows existed when it was written, which orders it
against every tool call's ``begin`` row there. A failed write freezes the
writer and records ``state/exec-ledger.gaps.jsonl``; the writer never reopens
after a gap, so the changes after it are absent, never guessed. The seal
inventory covers both files like every file of the run.

``read_rows(run_root)`` is the reader.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .keys import StateError
from .store import StateStore, encode

SCHEMA = "ora.toolkit.exec-ledger.v1"
GAP_SCHEMA = "ora.toolkit.exec-ledger-gap.v1"
PATH = "state/exec-ledger.jsonl"
GAPS = "state/exec-ledger.gaps.jsonl"
KEYS = ("exec.ik_sampler_state",)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or path.parent.is_symlink():
        raise StateError("exec ledger cannot be a symlink")
    return path


def read_rows(root: Path) -> list[dict[str, Any]]:
    """Validate the chain, the sequence and each value's hash."""
    path = _path(Path(root), PATH)
    if not path.exists():
        raise StateError("未记录: exec ledger is absent")
    rows, previous = [], None
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise StateError("exec ledger has an incomplete tail")
            row = json.loads(raw)
            body = {key: value for key, value in row.items() if key != "sha256"}
            if (row.get("schema") != SCHEMA or row.get("sequence") != len(rows)
                    or row.get("previous_sha256") != previous or row.get("sha256") != _sha(encode(body))
                    or row.get("kind") not in {"bound", "change"} or row.get("key") not in KEYS
                    or row.get("value_sha256") != _sha(encode(row.get("value")))):
                raise StateError("exec ledger identity mismatch")
            rows.append(row)
            previous = row["sha256"]
    return rows


class ExecLedger:
    """The owner's one writer; the tool journal (if bound) supplies the call scope and ordering."""

    def __init__(self, root: Path, store: StateStore):
        self.root = Path(root).resolve()
        self.path = self.root / PATH
        self.store = store
        self.count, self.sha256, self.size, self.closed = 0, None, 0, False
        try:
            _path(self.root, PATH)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            rows = read_rows(self.root)
            self.count = len(rows)
            self.sha256 = rows[-1]["sha256"] if rows else None
            self.size = self.path.stat().st_size
            if (self.root / GAPS).exists():
                raise StateError("exec ledger recording previously failed; cannot reopen writer")
        except Exception as error:
            self._gap(error, "open")
            return
        if not rows:
            for key in KEYS:
                current = store._values.get(key)
                self._append("bound", key, None if current is None else json.loads(current), method=None,
                             previous_value_sha256=None)

    def _scope(self) -> tuple[Any, Any, int]:
        journal = self.store.tool_journal
        if journal is None:
            return None, None, 0
        active = journal.scope.get()
        context = None if active is None else active["context"]
        ordinal = None
        if active is not None and context.get("kind") == "parent":
            try:
                ordinal = active["ordinal"]()
            except Exception:  # noqa: BLE001 - evidence only; the row keeps the scope
                ordinal = None
        return context, ordinal, journal.count

    def _gap(self, error: Exception, stage: str) -> None:
        self.closed = True
        try:
            context, _, deltas = self._scope()
            value = {"schema": GAP_SCHEMA, "stage": stage, "sequence": self.count, "tool_delta_count": deltas,
                     "context": context, "time_unix_s": time.time(),
                     "error": {"type": type(error).__name__, "message": str(error)}}
            with (self.root / GAPS).open("ab") as stream:
                stream.write(encode(value) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:  # noqa: BLE001 - a failed gap record still leaves the writer closed
            pass

    def _append(self, kind: str, key: str, value: Any, *, method: str | None,
                previous_value_sha256: str | None) -> None:
        if self.closed:
            return
        try:
            context, ordinal, deltas = self._scope()
            body = {"schema": SCHEMA, "sequence": self.count, "previous_sha256": self.sha256, "kind": kind,
                    "key": key, "method": method, "context": context, "interface_ordinal": ordinal,
                    "tool_delta_count": deltas, "value": value, "value_sha256": _sha(encode(value)),
                    "previous_value_sha256": previous_value_sha256, "time_unix_s": time.time()}
            row = {**body, "sha256": _sha(encode(body))}
            raw = encode(row) + b"\n"
            with self.path.open("ab") as stream:
                if os.fstat(stream.fileno()).st_size != self.size:
                    raise StateError("exec ledger changed outside its owner")
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.count, self.sha256, self.size = self.count + 1, row["sha256"], self.size + len(raw)
        except Exception as error:  # noqa: BLE001 - evidence failures never change the run
            self._gap(error, kind)

    def accepted(self, method: str, previous: dict[str, bytes | None]) -> None:
        """After an RPC response's execution state is stored: one row per tracked key it changed."""
        for key in KEYS:
            before, after = previous.get(key), self.store._values.get(key)
            if after is not None and after != before:
                self._append("change", key, json.loads(after), method=method,
                             previous_value_sha256=None if before is None else _sha(before))


def bind_exec_ledger(store: StateStore, root: Path) -> ExecLedger:
    """Bind the owner's only ledger writer to its run, independently of toolkit I/O recording."""
    ledger = getattr(store, "exec_ledger", None)
    if ledger is None:
        ledger = store.exec_ledger = ExecLedger(root, store)
    elif ledger.root != Path(root).resolve():
        raise StateError("state owner already belongs to another run")
    return ledger


__all__ = ["GAPS", "GAP_SCHEMA", "KEYS", "PATH", "SCHEMA", "ExecLedger", "bind_exec_ledger", "read_rows"]
