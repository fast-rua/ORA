"""``stage(name, **fields)``: a tool notes a decision its result does not carry.

A toolkit tool calls ``stage`` where it decides something its result would not show: a candidate it tried and dropped,
a value it resolved from a default, a pool it cut to a cap. The call is a no-op unless a caller asks for the events
(``collecting``): in tool-call mode nobody does, so nothing is recorded, no result, ledger entry or journal event
changes, and the call costs a context-variable read. Code mode (``library: extended``) asks for them around each
script call, renders them with the result into the call's ``explain`` (``ora/agent/toolkit/explain.py``) and records
them with the call's record (``agent/code_mode/record.py``); they never enter a tool's result.

Fields are plain JSON values (NumPy values and tuples are converted); a tool passes values it already holds, never
anything it would compute only for the event. Where building the fields itself costs a loop, the tool asks
``active()`` first.
"""

from __future__ import annotations

import contextlib
import math
from contextvars import ContextVar
from typing import Any, Iterator

#: The most events one call records; further ones are dropped, so a tool's loop cannot grow a record without bound.
MAX_EVENTS = 64
_MAX_DEPTH = 6

_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar("toolkit_stages", default=None)


def active() -> bool:
    """Whether a caller is collecting stage events (use before building fields that need a loop)."""

    return _sink.get() is not None


def stage(name: str, **fields: Any) -> None:
    """Record ``{"stage": name, "fields": fields}`` for the call in progress; a no-op when nobody collects."""

    sink = _sink.get()
    if sink is None or len(sink) >= MAX_EVENTS:
        return
    sink.append({"stage": str(name), "fields": {str(key): _plain(value, 0) for key, value in fields.items()}})


@contextlib.contextmanager
def collecting() -> Iterator[list[dict[str, Any]]]:
    """Collect the stage events of the tool calls made inside the block, in order."""

    events: list[dict[str, Any]] = []
    token = _sink.set(events)
    try:
        yield events
    finally:
        _sink.reset(token)


def _plain(value: Any, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth < _MAX_DEPTH and isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    if depth < _MAX_DEPTH and isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_plain(item, depth + 1) for item in items]
    if hasattr(value, "tolist"):  # NumPy arrays and scalars
        return _plain(value.tolist(), depth + 1)
    return repr(value)


__all__ = ["MAX_EVENTS", "active", "collecting", "stage"]
