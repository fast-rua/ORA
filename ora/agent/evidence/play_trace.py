"""The agent trace of a free-play run (``episode_mode: "play"``): complete, not benchmark-shaped.

A free-play run is Pi's own agent (ora-pi/src/play/): its model calls are Pi's
provider requests, its context is Pi's, and the agent may call Pi's coding
tools and run shell commands. Its trace is therefore checked for integrity and
completeness only, not against the benchmark's request, context and
decision grammar (``trajectory.verify_agent_trace``):

- identity: schema, sequence, unique ids, the hash chain, causal parents;
- artifacts: every referenced artifact exists with its recorded hash;
- completeness: every model call (``provider_request``) has its
  ``provider_response``, every tool call (``tool_call_proposed``) its
  ``tool_result``, every proposed robot action (``action_proposed``) a later
  ``environment_receipt``, and each operator message was delivered as the
  plan's operator control allows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .operator_messages import operator_trace_errors
from .agent_events import AGENT_EVENT_SCHEMA, read_agent_events
from .canonical import TrajectoryError, canonical_bytes, sha256_bytes
from .trace_verify import _artifact_refs, artifact_bytes

PLAY_TRACE_SCHEMA = "ora.play-trace-check.v1"


def verify_play_trace(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    try:
        events = read_agent_events(root)
    except TrajectoryError as exc:
        return {"schema": PLAY_TRACE_SCHEMA, "status": "invalid", "errors": [str(exc)], "event_count": 0}
    plan: dict[str, Any] = {}
    if (root / "plan.json").is_file():
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    errors: list[str] = list(operator_trace_errors(events, plan))
    known: set[str] = set()
    previous: str | None = None
    requests: dict[str, int] = {}
    tools: dict[str, int] = {}
    actions: list[int] = []
    receipts = 0
    for index, event in enumerate(events):
        if event.get("schema") != AGENT_EVENT_SCHEMA:
            errors.append(f"event:{index}:schema")
        if event.get("seq") != index:
            errors.append(f"event:{index}:sequence")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in known:
            errors.append(f"event:{index}:event_id")
        else:
            known.add(event_id)
        if event.get("previous_event_sha256") != previous:
            errors.append(f"event:{index}:previous_hash")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("event_sha256") != sha256_bytes(canonical_bytes(body)):
            errors.append(f"event:{index}:hash")
        previous = event.get("event_sha256") if isinstance(event.get("event_sha256"), str) else None
        parent = event.get("causal_parent_id")
        if parent is not None and parent not in known:
            errors.append(f"event:{index}:causal_parent")
        for artifact in [*_artifact_refs(event.get("artifact_ref")), *_artifact_refs(event.get("payload"))]:
            relative = artifact.get("path")
            data = artifact_bytes(root, relative) if isinstance(relative, str) else None
            if data is None:
                errors.append(f"event:{index}:artifact_missing")
            elif sha256_bytes(data) != artifact.get("sha256"):
                errors.append(f"event:{index}:artifact_hash")
        kind = event.get("type")
        call_id = event.get("model_call_id")
        tool_id = event.get("tool_call_id") or (event.get("payload") or {}).get("tool_call_id")
        if kind == "provider_request":
            if call_id in requests:
                errors.append(f"event:{index}:duplicate_request")
            requests[str(call_id)] = index
        elif kind in {"provider_response", "provider_error"}:
            if requests.pop(str(call_id), None) is None:
                errors.append(f"event:{index}:response_without_request")
        elif kind == "tool_call_proposed":
            tools[str(tool_id)] = index
        elif kind == "tool_result" and tool_id is not None:
            if tools.pop(str(tool_id), None) is None:
                errors.append(f"event:{index}:tool_result_without_call")
        elif kind == "action_proposed":
            actions.append(index)
        elif kind == "environment_receipt":
            receipts += 1
            if actions:
                actions.pop(0)
    errors.extend(f"event:{index}:request_without_response" for index in requests.values())
    errors.extend(f"event:{index}:tool_call_without_result" for index in tools.values())
    errors.extend(f"event:{index}:action_without_receipt" for index in actions)
    if events and events[0].get("type") != "session_started":
        errors.append("event:0:session_started")
    return {
        "schema": PLAY_TRACE_SCHEMA,
        "status": "valid" if not errors else "invalid",
        "errors": sorted(set(errors)),
        "event_count": len(events),
        "final_event_sha256": previous,
        "receipts": receipts,
    }


def is_play_plan(plan: Mapping[str, Any]) -> bool:
    return plan.get("episode_mode") == "play"


__all__ = ["PLAY_TRACE_SCHEMA", "is_play_plan", "verify_play_trace"]
