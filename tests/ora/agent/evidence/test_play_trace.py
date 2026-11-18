"""The free-play trace check (``ora/agent/evidence/play_trace.py``): integrity and completeness only.

A free-play run's ``agent-events.jsonl`` is valid when its events carry the
schema, consecutive sequence numbers, unique ids, an unbroken hash chain and
known causal parents; every referenced artifact exists with its recorded
hash; every provider request has its response, every tool call its result and
every proposed action a later environment receipt; and it starts with
``session_started``. Anything else makes it ``invalid`` with the event index
and the rule it broke.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ora.agent.evidence.canonical import canonical_bytes, sha256_bytes
from ora.agent.evidence.play_trace import PLAY_TRACE_SCHEMA, is_play_plan, verify_play_trace
from tests.support.agent.agent_events import TrajectoryRecorder


def _record(root: Path, *events: tuple[str, dict[str, Any]]) -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(root, run_id="play-run", policy_session_id="pi-session")
    for kind, fields in events:
        recorder.append(kind, **fields)
    return recorder


def _complete_turn(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """A session that asks the model once, runs one read and one action, and gets the receipt."""

    view = root / "artifacts" / "view.png"
    view.parent.mkdir(parents=True, exist_ok=True)
    view.write_bytes(b"\x89PNG fixture")
    artifact = {"path": "artifacts/view.png", "sha256": hashlib.sha256(view.read_bytes()).hexdigest(),
                "bytes": view.stat().st_size}
    return [
        ("session_started", {}),
        ("provider_request", {"model_call_id": "call-1", "causal_parent_id": "agent-event-000000"}),
        ("provider_response", {"model_call_id": "call-1", "causal_parent_id": "agent-event-000001"}),
        ("tool_call_proposed", {"tool_call_id": "t1", "model_call_id": "call-1"}),
        ("tool_result", {"tool_call_id": "t1", "artifact_ref": artifact}),
        ("action_proposed", {"tool_call_id": "t2", "payload": {"action": "close_gripper"}}),
        ("environment_receipt", {"payload": {"execution_status": "reached"}}),
    ]


def test_a_complete_session_is_valid(tmp_path: Path) -> None:
    recorder = _record(tmp_path, *_complete_turn(tmp_path))
    check = verify_play_trace(tmp_path)
    assert check == {"schema": PLAY_TRACE_SCHEMA, "status": "valid", "errors": [], "event_count": 7,
                     "final_event_sha256": recorder.previous_event_sha256, "receipts": 1}


def test_a_missing_or_unreadable_trace_is_invalid_with_the_reason(tmp_path: Path) -> None:
    check = verify_play_trace(tmp_path)
    assert check["status"] == "invalid" and check["event_count"] == 0
    assert check["errors"] == ["agent-events.jsonl is missing"]
    (tmp_path / "agent-events.jsonl").write_text("{not json\n", encoding="utf-8")
    assert verify_play_trace(tmp_path)["errors"] == ["invalid agent event JSON at line 1"]


def test_unanswered_requests_tools_and_actions_are_named_by_their_event(tmp_path: Path) -> None:
    _record(tmp_path,
            ("session_started", {}),
            ("provider_request", {"model_call_id": "call-1"}),
            ("tool_call_proposed", {"tool_call_id": "t1"}),
            ("action_proposed", {"tool_call_id": "t2"}),
            ("provider_response", {"model_call_id": "call-9"}),
            ("tool_result", {"tool_call_id": "t9"}))
    check = verify_play_trace(tmp_path)
    assert check["status"] == "invalid" and check["receipts"] == 0
    assert check["errors"] == sorted([
        "event:1:request_without_response", "event:2:tool_call_without_result",
        "event:3:action_without_receipt", "event:4:response_without_request",
        "event:5:tool_result_without_call",
    ])


def test_a_provider_error_answers_its_request_and_a_repeated_request_id_is_refused(tmp_path: Path) -> None:
    _record(tmp_path,
            ("session_started", {}),
            ("provider_request", {"model_call_id": "call-1"}),
            ("provider_request", {"model_call_id": "call-1"}),
            ("provider_error", {"model_call_id": "call-1"}))
    assert verify_play_trace(tmp_path)["errors"] == ["event:2:duplicate_request"]


def test_a_rewritten_event_breaks_its_hash_and_a_rehashed_one_the_chain_after_it(tmp_path: Path) -> None:
    _record(tmp_path, *_complete_turn(tmp_path))
    path = tmp_path / "agent-events.jsonl"
    original = path.read_text(encoding="utf-8")

    def rewrite(rehash: bool) -> list[str]:
        events = [json.loads(line) for line in original.splitlines()]
        events[2]["payload"] = {"edited": True}
        if rehash:
            body = {key: value for key, value in events[2].items() if key != "event_sha256"}
            events[2]["event_sha256"] = sha256_bytes(canonical_bytes(body))
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        return verify_play_trace(tmp_path)["errors"]

    assert rewrite(rehash=False) == ["event:2:hash"]
    assert rewrite(rehash=True) == ["event:3:previous_hash"]


def test_identity_sequence_and_causality_are_checked_per_event(tmp_path: Path) -> None:
    _record(tmp_path, ("session_started", {}), ("provider_request", {"model_call_id": "call-1"}),
            ("provider_response", {"model_call_id": "call-1", "causal_parent_id": "agent-event-999999"}))
    path = tmp_path / "agent-events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[1].update(schema="ora.agent-event.v0", seq=7, event_id=events[0]["event_id"])
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    errors = verify_play_trace(tmp_path)["errors"]
    assert {"event:1:schema", "event:1:sequence", "event:1:event_id", "event:1:hash",
            "event:2:causal_parent"} <= set(errors)


def test_a_missing_or_altered_artifact_is_refused(tmp_path: Path) -> None:
    _record(tmp_path, *_complete_turn(tmp_path))
    view = tmp_path / "artifacts" / "view.png"
    view.write_bytes(b"changed")
    assert verify_play_trace(tmp_path)["errors"] == ["event:4:artifact_hash"]
    view.unlink()
    assert verify_play_trace(tmp_path)["errors"] == ["event:4:artifact_missing"]


def test_a_trace_must_open_with_its_session(tmp_path: Path) -> None:
    _record(tmp_path, ("provider_request", {"model_call_id": "call-1"}),
            ("provider_response", {"model_call_id": "call-1"}))
    assert verify_play_trace(tmp_path)["errors"] == ["event:0:session_started"]


def test_an_operator_message_outside_a_play_plan_is_refused(tmp_path: Path) -> None:
    _record(tmp_path, ("session_started", {}),
            ("operator_message", {"payload": {"schema": "ora.ora-operator-command.v1", "type": "instruction"}}))
    (tmp_path / "plan.json").write_text(json.dumps({"episode_mode": "play", "experiment": "ora-play"}),
                                        encoding="utf-8")
    assert verify_play_trace(tmp_path)["errors"] == ["event:1:operator_message_contract"]


def test_only_a_play_episode_is_a_play_plan() -> None:
    assert is_play_plan({"episode_mode": "play"}) is True
    assert is_play_plan({"episode_mode": "benchmark"}) is False and is_play_plan({}) is False
