"""Owner-recorded tool state changes between complete publication snapshots.

This journal has its own sequence. interface_ordinal identifies the next I/O
call, before that call, without consuming the toolkit-io cursor. State is
changed first; evidence failures freeze this writer and never change behavior.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time

from .keys import StateError, key_definition
from .store import StateStore, encode

SCHEMA = "ora.toolkit.tool-state-delta.v1"
PATH = "state/tool-deltas.jsonl"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _journal_path(root: Path) -> Path:
    path = root / PATH
    if path.is_symlink() or path.parent.is_symlink():
        raise StateError("tool state journal cannot be a symlink")
    return path


def read_rows(root: Path) -> list[dict]:
    """Validate the immutable prefix and every change's namespace and encoding."""
    path = _journal_path(root)
    if not path.exists():
        raise StateError("未记录: tool state delta journal is absent")
    previous = None
    rows = []
    active = None
    write_ordinal = 0
    interface_ordinal = 0
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise StateError("tool state delta journal has an incomplete tail")
            row = json.loads(raw)
            body = {key: value for key, value in row.items() if key != "sha256"}
            if (row.get("schema") != SCHEMA or type(row.get("sequence")) is not int
                    or row["sequence"] != len(rows)
                    or row.get("previous_sha256") != previous
                    or row.get("sha256") != _sha(encode(body))):
                raise StateError("tool state delta journal identity mismatch")
            if row.get("kind") not in {"begin", "end", "write"}:
                raise StateError("unknown tool state delta kind")
            context = row.get("context", {})
            if context.get("kind") == "parent":
                if not all(isinstance(context.get(key), str) and context[key]
                           for key in ("model_call_id", "tool_call_id")):
                    raise StateError("tool state parent identity is incomplete")
            elif context.get("kind") != "owner" or context.get("operation") not in {"initialize", "reset"}:
                raise StateError("tool state owner boundary is unspecified")
            if row["kind"] == "begin":
                if active is not None:
                    raise StateError("tool state parent boundary overlaps an unfinished invocation")
                active, write_ordinal, interface_ordinal = context, 0, 0
            elif active != context:
                raise StateError("tool state event is outside its invocation")
            if row["kind"] == "end":
                active = None
            if row["kind"] == "write":
                position = row.get("interface_ordinal")
                if (row.get("state_write_ordinal") != write_ordinal
                        or type(position) is not int or position < interface_ordinal
                        or row.get("interface_boundary") != "before"):
                    raise StateError("tool state interface order is invalid")
                write_ordinal += 1
                interface_ordinal = position
                for key, change in row["changes"].items():
                    if key_definition(key).writer != "tool":
                        raise StateError("tool state delta cannot write another namespace")
                    encode(change["value"])
            rows.append(row)
            previous = row["sha256"]
    return rows


class ToolJournal:
    """One writer per state owner. The scope comes from trusted dispatch."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / PATH
        self.scope = ContextVar("tool_state_delta_scope", default=None)
        self.count = 0
        self.sha256 = None
        self.closed = False
        self.size = 0
        self.active = False
        try:
            _journal_path(self.root)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            rows = read_rows(self.root)
            self.count = len(rows)
            self.sha256 = rows[-1]["sha256"] if rows else None
            self.size = self.path.stat().st_size
            if (self.root / "state/tool-deltas.gaps.jsonl").exists():
                raise StateError("tool state recording previously failed; cannot reopen writer")
            if rows and rows[-1]["kind"] != "end":
                raise StateError("tool state journal has an unfinished invocation")
        except Exception as error:
            self._gap(error, "open", {})

    def watermark(self) -> dict:
        return {"schema": SCHEMA, "count": self.count, "sha256": self.sha256,
                "complete": not self.closed}

    def _gap(self, error: Exception, stage: str, context: dict) -> None:
        self.closed = True
        try:
            value = {"schema": "ora.toolkit.tool-state-gap.v1", "stage": stage,
                     "sequence": self.count, "time_unix_s": time.time(),
                     "model_call_id": context.get("model_call_id"),
                     "tool_call_id": context.get("tool_call_id"),
                     "error": {"type": type(error).__name__, "message": str(error)}}
            with (self.root / "state/tool-deltas.gaps.jsonl").open("ab") as stream:
                stream.write(encode(value) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            pass

    def _append(self, kind: str, context: dict, **fields) -> None:
        if self.closed:
            return
        try:
            body = {"schema": SCHEMA, "sequence": self.count,
                    "previous_sha256": self.sha256, "kind": kind,
                    "context": context, **fields}
            row = {**body, "sha256": _sha(encode(body))}
            raw = encode(row) + b"\n"
            with self.path.open("ab") as stream:
                if os.fstat(stream.fileno()).st_size != self.size:
                    raise StateError("tool state journal changed outside its owner")
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.count += 1
            self.sha256 = row["sha256"]
            self.size += len(raw)
        except Exception as error:
            self._gap(error, kind, context)

    @contextmanager
    def boundary(self, context: dict, ordinal=lambda: 0):
        """Bracket a parent invocation or an explicit owner init/reset boundary."""
        try:
            context = json.loads(encode(context))
            if self.active:
                raise StateError("concurrent or recursive tool state invocation")
        except Exception as error:
            self._gap(error, "begin", {})
        active = {"context": context, "ordinal": ordinal, "writes": 0}
        was_active = self.active
        self.active = True
        token = self.scope.set(active)
        self._append("begin", active["context"])
        try:
            yield
        finally:
            self._append("end", active["context"])
            self.scope.reset(token)
            self.active = was_active

    def changed(self, previous: dict[str, bytes | None], values: dict[str, bytes]) -> None:
        active = self.scope.get()
        context = active["context"] if active is not None else {}
        if self.closed:
            return
        try:
            if active is None:
                raise StateError("tool state write has no trusted parent or owner boundary")
            ordinal = active["ordinal"]()
            if type(ordinal) is not int or ordinal < 0:
                raise StateError("invalid tool state interface ordinal")
            changes = {key: {"previous_sha256": _sha(previous[key]) if previous[key] is not None else None,
                             "value": json.loads(value)} for key, value in values.items()}
            self._append("write", context, interface_ordinal=ordinal, interface_boundary="before",
                         state_write_ordinal=active["writes"], changes=changes)
            active["writes"] += 1
        except Exception as error:
            self._gap(error, "write", context)


def bind_journal(store: StateStore, root: Path) -> ToolJournal:
    """Bind the owner's only writer, independently of toolkit I/O recording."""
    if store.tool_journal is None:
        store.tool_journal = ToolJournal(root)
    elif store.tool_journal.root != root.resolve():
        raise StateError("state owner already belongs to another run")
    return store.tool_journal


def before_call(store: StateStore, root: Path, *, model_call_id: str, tool_call_id: str) -> StateStore:
    """Make a private tool-writable overlay at a verified parent-before boundary.

The source state must be the last completed snapshot at or before the parent.
No missing delta, unknown initialization, or future value is inferred.
"""
    rows = read_rows(root)
    watermark = store.tool_watermark
    if not watermark or watermark.get("schema") != SCHEMA or not watermark.get("complete"):
        raise StateError("未记录: snapshot lacks a complete tool state watermark")
    count = watermark.get("count")
    if type(count) is not int or not 0 <= count <= len(rows):
        raise StateError("tool state watermark outside journal")
    if watermark["sha256"] != (rows[count - 1]["sha256"] if count else None):
        raise StateError("tool state watermark identity mismatch")
    matches = [row for row in rows if row["kind"] == "begin"
               and row["context"].get("kind") == "parent"
               and row["context"].get("model_call_id") == model_call_id
               and row["context"].get("tool_call_id") == tool_call_id]
    if len(matches) != 1:
        raise StateError("未记录: absent or ambiguous tool state parent boundary")
    stop = matches[0]["sequence"]
    if stop < count:
        raise StateError("snapshot is later than the requested parent boundary")
    overlay = StateStore()
    overlay._values = dict(store._values)
    overlay.complete_boundary = store.complete_boundary
    for row in rows[count:stop]:
        if row["kind"] != "write":
            continue
        for key, change in row["changes"].items():
            current = overlay._values.get(key)
            if change["previous_sha256"] != (_sha(current) if current is not None else None):
                raise StateError("tool state delta does not follow snapshot values")
            overlay._values[key] = encode(change["value"])
    return overlay


def journal_state(root: Path, *, model_call_id: str, tool_call_id: str) -> StateStore:
    """The tool state before one parent of a run that has no snapshot to start from.

    A journal begins at its owner's initialize boundary and holds every tool.* write after it, so an empty store at
    count zero is a complete snapshot of that namespace. The result is ``before_call``'s private overlay.
    """
    empty = StateStore(read_only=True)
    empty.tool_watermark = {"schema": SCHEMA, "count": 0, "sha256": None, "complete": True}
    return before_call(empty, root, model_call_id=model_call_id, tool_call_id=tool_call_id)


def before_interface(store: StateStore, root: Path, *, model_call_id: str,
                     tool_call_id: str, interface_ordinal: int) -> StateStore:
    """Overlay through writes before one verified interface invocation begins.

    The caller verifies the action-to-I/O mapping. Writes at this ordinal are
    before the interface call; later writes use the next ordinal. A full
    snapshot inside the same parent is supported without replaying old writes.
    """
    if type(interface_ordinal) is not int or interface_ordinal < 0:
        raise StateError("invalid interface boundary")
    records = read_rows(root)
    watermark = store.tool_watermark
    if not watermark or watermark.get("schema") != SCHEMA or not watermark.get("complete"):
        raise StateError("未记录: snapshot lacks a complete tool state watermark")
    count = watermark.get("count")
    if type(count) is not int or not 0 <= count <= len(records):
        raise StateError("tool state watermark outside journal")
    if watermark["sha256"] != (records[count - 1]["sha256"] if count else None):
        raise StateError("tool state watermark identity mismatch")
    matches = [row for row in records if row["kind"] == "begin"
               and row["context"].get("model_call_id") == model_call_id
               and row["context"].get("tool_call_id") == tool_call_id]
    if len(matches) != 1:
        raise StateError("未记录: absent or ambiguous tool state parent boundary")
    begin = matches[0]
    ends = [row for row in records[begin["sequence"] + 1:]
            if row["kind"] == "end" and row["context"] == begin["context"]]
    if not ends:
        raise StateError("未记录: incomplete tool state parent boundary")
    stop = ends[0]["sequence"]
    for row in records[begin["sequence"] + 1:stop]:
        if row["kind"] == "write" and row["interface_ordinal"] > interface_ordinal:
            stop = row["sequence"]
            break
    if count > stop:
        raise StateError("snapshot is later than the requested interface boundary")
    gaps = root / "state/tool-deltas.gaps.jsonl"
    if gaps.exists():
        for line in gaps.read_bytes().splitlines():
            gap = json.loads(line)
            if count <= gap.get("sequence", count) <= stop:
                raise StateError("未记录: tool state interval has a recording gap")
    overlay = StateStore()
    overlay._values = dict(store._values)
    overlay.complete_boundary = False
    for row in records[count:stop]:
        if row["kind"] != "write":
            continue
        for key, change in row["changes"].items():
            current = overlay._values.get(key)
            if change["previous_sha256"] != (_sha(current) if current is not None else None):
                raise StateError("tool state delta does not follow snapshot values")
            overlay._values[key] = encode(change["value"])
    return overlay


def unsettled_publication(store: StateStore, root: Path) -> str | None:
    """Report when a publication cannot establish settled tool state.

    A publication inside a parent can precede a zero-spend rollback or verify
    write. Do not guess which later writes belong to this published step.
    """
    watermark = store.tool_watermark
    if not watermark or watermark.get("schema") != SCHEMA or not watermark.get("complete"):
        return "publication has no complete tool-state watermark"
    try:
        records = read_rows(root)
    except StateError as error:
        if str(error).startswith("未记录"):
            return str(error)
        raise
    count = watermark.get("count")
    if type(count) is not int or not 0 <= count <= len(records):
        raise StateError("tool state watermark outside journal")
    if watermark["sha256"] != (records[count - 1]["sha256"] if count else None):
        raise StateError("tool state watermark identity mismatch")
    if count and records[count - 1]["kind"] != "end":
        return "publication occurred inside an unfinished tool/owner invocation; use --snapshot --at for settled tool state"
    return None
