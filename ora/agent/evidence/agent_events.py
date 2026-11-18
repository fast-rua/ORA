"""The agent trace's event log (``ora.agent-event.v1``): its schema name, the
visibility vocabulary, the reader, and ``library_op``, which tells a code-mode
script's library request from a tool call.

The Pi core writes the log (``ora-pi/src/evidence/recorder.ts``); tests write
one with ``tests/support/agent/agent_events.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import TrajectoryError


AGENT_EVENT_SCHEMA = "ora.agent-event.v1"
VISIBILITY = {"exact", "derived", "opaque"}


def library_op(event: Mapping[str, Any]) -> str | None:
    """The op of a code-mode script's library request (``query``, ``explain``, in a sealed run ``inspect``), or None.

    Such a request (``library: extended``, ``agent/pi/code_mode.py``) is recorded as a nested call of the script,
    a ``tool_call_proposed``/``tool_result`` pair named ``robot.<op>`` whose ``payload.origin`` names the op. It is
    no tool call: a reader that lists, counts or replays tool calls leaves it out.
    """

    payload = event.get("payload")
    origin = payload.get("origin") if isinstance(payload, Mapping) else None
    op = origin.get("op") if isinstance(origin, Mapping) and origin.get("kind") == "code_mode" else None
    return op if isinstance(op, str) else None


def read_agent_events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root.resolve() / "agent-events.jsonl"
    if not path.is_file():
        raise TrajectoryError("agent-events.jsonl is missing")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(f"invalid agent event JSON at line {index + 1}") from exc
        if not isinstance(value, dict):
            raise TrajectoryError(f"agent event must be an object at line {index + 1}")
        events.append(value)
    return events
