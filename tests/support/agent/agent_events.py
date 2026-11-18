"""The agent-event recorder tests write traces with.

The Pi core writes production traces with its TypeScript recorder
(``ora-pi/src/evidence/recorder.ts``). This Python twin left ``ora.agent.evidence``
because only tests write a trace from Python; it appends the
``ora.agent-event.v1`` events and forward hash chain ``verify_agent_trace`` checks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence.agent_events import AGENT_EVENT_SCHEMA, VISIBILITY
from ora.agent.evidence.canonical import TrajectoryError, canonical_bytes, sha256_bytes


@dataclass
class TrajectoryRecorder:
    """Append-only, fsync-backed event recorder with a forward hash chain."""

    run_root: Path
    run_id: str
    policy_session_id: str

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.path = self.run_root / "agent-events.jsonl"
        if self.path.exists():
            raise TrajectoryError("refusing to append to an existing agent-events.jsonl")
        self.seq = 0
        self.previous_event_sha256: str | None = None

    def append(
        self,
        event_type: str,
        *,
        visibility: str = "exact",
        causal_parent_id: str | None = None,
        observation_step: int | None = None,
        model_call_id: str | None = None,
        tool_call_id: str | None = None,
        payload: Any = None,
        artifact_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if visibility not in VISIBILITY:
            raise TrajectoryError(f"invalid event visibility: {visibility}")
        body = {
            "schema": AGENT_EVENT_SCHEMA,
            "seq": self.seq,
            "event_id": f"agent-event-{self.seq:06d}",
            "type": event_type,
            "run_id": self.run_id,
            "policy_session_id": self.policy_session_id,
            "observation_step": observation_step,
            "model_call_id": model_call_id,
            "tool_call_id": tool_call_id,
            "causal_parent_id": causal_parent_id,
            "visibility": visibility,
            "unix_s": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            "previous_event_sha256": self.previous_event_sha256,
            "payload": payload,
            "artifact_ref": None if artifact_ref is None else dict(artifact_ref),
        }
        event = {**body, "event_sha256": sha256_bytes(canonical_bytes(body))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous_event_sha256 = event["event_sha256"]
        self.seq += 1
        return event
