"""Append-only invocation events, including failed and interrupted attempts."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ora.contracts.toolkit_protocol import PROTOCOL_VERSION, metadata

from .api import EFFECTS, EXTERNAL, METHODS
from .blobs import IO_ROOT, input_identity
from .types import AmbiguousRecord, CorruptRecord, canonical, decode, digest

EVENT_SCHEMA = "ora.toolkit.io-event.v1"


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        events = [decode(line) for line in path.read_bytes().splitlines()]
    except (ValueError, UnicodeError) as error:
        raise CorruptRecord("invalid or truncated I/O event") from error
    pending: dict[tuple, dict] = {}
    finished: set[tuple] = set()
    identities: dict[tuple, str] = {}
    last_ordinal: dict[tuple, int] = {}
    for sequence, event in enumerate(events):
        if event.get("schema") != EVENT_SCHEMA:
            raise CorruptRecord("unsupported I/O schema: no decoder")
        if event.get("sequence") != sequence:
            raise CorruptRecord("I/O sequence mismatch")
        identity = event["identity"]
        for phase in ("before", "after"):
            if identity.get(phase) != dict(tool_call_id=identity["tool_call_id"],
                                           interface_ordinal=identity["interface_ordinal"], phase=phase):
                raise CorruptRecord("logical boundary token mismatch")
        request = event["request"]
        semantic = request["identity"]
        if semantic.get("schema") != "ora.toolkit.request.v1":
            raise CorruptRecord("unsupported request schema: no decoder")
        if (set(request) != {"identity", "inputs"}
                or semantic.get("method") not in METHODS
                or {key: input_identity(ref) for key, ref in request["inputs"].items()} != semantic["inputs"]):
            raise CorruptRecord("invalid semantic request/actual input references")
        invocation = digest(canonical(identity))
        if invocation != event["invocation_id"]:
            raise CorruptRecord("invocation identity mismatch")
        slot = tuple(identity[key] for key in ("run_id", "round_index", "tool_call_id", "interface_ordinal"))
        if slot in identities and identities[slot] != invocation:
            raise AmbiguousRecord("duplicate interface ordinal")
        identities[slot] = invocation
        ordinal, attempt = identity["interface_ordinal"], event["attempt"]
        if type(ordinal) is not int or ordinal < 0 or type(attempt) is not int or attempt < 0:
            raise CorruptRecord("ordinal/attempt must be nonnegative integers")
        key = invocation, attempt
        if event["event"] == "begin":
            if key in pending or key in finished:
                raise AmbiguousRecord("duplicate invocation attempt")
            if pending:
                raise CorruptRecord("overlapping or interrupted invocation")
            previous = last_ordinal.get(slot[:3], -1)
            if ordinal not in (previous, previous + 1) or (ordinal == previous and attempt == 0):
                raise AmbiguousRecord("interface sequence skips or revisits an ordinal")
            if ordinal != previous and attempt != 0:
                raise CorruptRecord("new invocation must start with attempt zero")
            last_ordinal[slot[:3]] = ordinal
            if event["attempt"] and (invocation, event["attempt"] - 1) not in finished:
                raise CorruptRecord("retry without completed previous attempt")
            if digest(canonical(event["request"]["identity"])) != event["semantic_request_hash"]:
                raise CorruptRecord("request identity mismatch")
            pending[key] = event
        elif event["event"] == "end":
            begin = pending.pop(key, None)
            if begin is None:
                raise AmbiguousRecord("completion without unique begin")
            if any(begin[field] != event[field] for field in ("semantic_request_hash", "request", "identity", "state_before")):
                raise CorruptRecord("request changed between begin and end")
            if event.get("status") not in ("ok", "error", "timeout"):
                raise CorruptRecord("invalid completion status")
            source = "external_service" if semantic["method"] in EXTERNAL else "agent_control" if semantic["method"] in EFFECTS else "simulator"
            if event.get("source") != source or type(event.get("cache_hit")) is not bool:
                raise CorruptRecord("invalid source/cache-hit fields")
            finished.add(key)
        else:
            raise CorruptRecord("unknown I/O event")
    return events


class Journal:
    """A single writer; fail closed on duplicate identities, never retry a write."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / IO_ROOT / "calls.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = read_events(self.path)
        manifest = self.path.parent / "manifest.json"
        if not manifest.exists():
            manifest.write_bytes(canonical(metadata(recording_enabled=True)) + b"\n")
        self._closed = False
        self._bytes_written = sum(len(canonical(item)) + 1 for item in self.events)
        if self.events and self.events[-1]["event"] != "end":
            raise CorruptRecord("cannot continue an interrupted I/O writer")

    def append(self, event: dict) -> dict:
        if self._closed:
            raise CorruptRecord("journal write failed; writer cannot continue")
        value = dict(event, schema=EVENT_SCHEMA, sequence=len(self.events), protocol_version=PROTOCOL_VERSION)
        data = canonical(value) + b"\n"
        # Detect another writer before append. No multiwriter support is claimed.
        expected = self._bytes_written
        if self.path.exists() and self.path.stat().st_size != expected:
            raise CorruptRecord("journal changed outside its writer")
        try:
            with self.path.open("ab") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._closed = True
            raise
        self.events.append(value)
        self._bytes_written += len(data)
        return value
