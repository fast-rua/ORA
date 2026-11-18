"""The evidence contract of a free-play session (``plan.play``, episode/play.py).

A free-play run is valid when its instruction records, scene resets and
counters join the event trace exactly, and it claims nothing a benchmark run
claims: no official verdict, no behavioral sample. It is never learnable
(``free_play_session``) and is evaluated without a benchmark's official
boundary checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..episode.play import ANSWER_SCHEMA, INSTRUCTION_SCHEMA, PLAY_SCHEMA

LEARNING_REASON = "free_play_session"


def is_play(plan: Mapping[str, Any]) -> bool:
    return plan.get("episode_mode") == "play" or isinstance(plan.get("play"), Mapping)


def play_trace_evidence(run_root: Path) -> Any:
    """The agent trace of a free-play run: identity, artifacts and completeness only."""

    from ora.agent.evidence.play_trace import verify_play_trace

    from .checks import AgentTraceEvidence

    if not (run_root / "agent-events.jsonl").is_file():
        return AgentTraceEvidence(reasons=[])
    check = verify_play_trace(run_root)
    return AgentTraceEvidence(reasons=[f"agent_trace:{error}" for error in check["errors"]], check=check)


def verify_play_session(
    run_root: Path, plan: Mapping[str, Any], result: Mapping[str, Any],
    events: list[Mapping[str, Any]],
) -> list[str]:
    if not is_play(plan):
        return ["play_result_without_plan"] if "play" in result else []
    errors: list[str] = []
    if (plan.get("experiment") != "ora-play" or plan.get("episode_mode") != "play"
            or "operator_control" not in plan or "session" in plan):
        errors.append("play_plan_binding_invalid")
    if result.get("official_success") is not None or result.get("verdict") is not None:
        errors.append("play_official_verdict_claimed")
    if (run_root / "native" / "official-verdict.json").exists():
        errors.append("play_official_verdict_persisted")
    if result.get("behavioral_sample_valid") is not False:
        errors.append("play_behavioral_sample_claimed")
    if result.get("termination") == "trusted_fault":
        return errors  # A fault result carries no play summary; its events stay as recorded.
    summary = result.get("play")
    if not isinstance(summary, Mapping) or summary.get("schema") != PLAY_SCHEMA:
        return [*errors, "play_summary_missing"]
    started = [event.get("record") for event in events if event.get("type") == "play_instruction_started"]
    finished = [event.get("record") for event in events if event.get("type") == "play_instruction_finished"]
    records = summary.get("instructions")
    if records != finished or summary.get("instruction_count") != len(finished):
        errors.append("play_instruction_record_drift")
    if not all(isinstance(item, Mapping) for item in started + finished):
        return [*errors, "play_instruction_record_invalid"]
    if [item.get("instruction_id") for item in started] != [item.get("instruction_id") for item in finished] or any(
        item.get("schema") != INSTRUCTION_SCHEMA or item.get("index") != index
        for index, item in enumerate(finished)
    ):
        errors.append("play_instruction_unclosed_or_unordered")
    messages = [event.get("payload", {}) for event in events if event.get("type") == "operator_message"]
    delivered = {message.get("id") for message in messages
                 if message.get("type") in {"instruction", "queued_instruction"}}
    # A message sent while an instruction ran joined it (episode.play: joined_messages); records
    # written before messages joined carry none, and their delivered instructions each opened one.
    joined = [item for record in finished for item in record.get("joined_messages") or []]
    opened = {item.get("instruction_id") for item in finished if item.get("delivered") is True}
    if (
        delivered - set(joined) != opened or len(joined) != len(set(joined)) or set(joined) & opened
        or not set(joined) <= {message.get("id") for message in messages}
    ):
        errors.append("play_instruction_delivery_mismatch")
    if summary.get("delivered_instruction_count") != len(opened):
        errors.append("play_instruction_delivery_mismatch")
    for field, total in (("decisions", "decisions"), ("refused_proposals", "refused_proposals"),
                         ("model_calls", "model_calls")):
        if sum(int(item.get(field) or 0) for item in finished) != result.get(total, 0):
            errors.append(f"play_{field}_mismatch")
    errors.extend(_answer_errors(finished, events))
    errors.extend(_scene_errors(plan["play"], summary, events))
    errors.extend(_operator_view_errors(run_root, events))
    return errors


def _answer_errors(finished: list[Any], events: list[Mapping[str, Any]]) -> list[str]:
    """Each ``play_answer`` answers a message that joined its instruction, once, as the instruction records."""

    by_id = {item.get("instruction_id"): item for item in finished if isinstance(item, Mapping)}
    answered: set[tuple[Any, Any]] = set()
    for event in events:
        if event.get("type") != "play_answer":
            continue
        record = event.get("record") if isinstance(event.get("record"), Mapping) else {}
        instruction = by_id.get(record.get("instruction_id"))
        key = (record.get("instruction_id"), record.get("message_id"))
        if (record.get("schema") != ANSWER_SCHEMA or instruction is None or key in answered
                or record.get("message_id") not in (instruction.get("joined_messages") or [])
                or record.get("instruction_index") != instruction.get("index")
                or not isinstance(record.get("observation_step"), int)):
            return ["play_answer_invalid"]
        answered.add(key)
    # Records written before answers existed carry no answered_messages.
    recorded = {(item.get("instruction_id"), message) for item in by_id.values()
                for message in item.get("answered_messages") or []}
    return [] if recorded == answered else ["play_answer_invalid"]


def _operator_view_errors(run_root: Path, events: list[Mapping[str, Any]]) -> list[str]:
    """Each ``operator_view_frame`` names a stored PNG under native/operator-view/ with its hash."""

    for event in events:
        if event.get("type") != "operator_view_frame":
            continue
        relative = str(event.get("path") or "")
        path = run_root / relative
        step = event.get("observation_step")
        if (
            not isinstance(step, int) or relative != f"native/operator-view/{step:06d}.png"
            or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != event.get("sha256")
        ):
            return ["play_operator_view_frame_invalid"]
    return []


def _scene_errors(play: Mapping[str, Any], summary: Mapping[str, Any], events: list[Mapping[str, Any]]) -> list[str]:
    scenes = summary.get("scenes")
    planned = play.get("scenes", [])
    resets = [event.get("record") for event in events if event.get("type") == "reset"]
    if (
        not isinstance(scenes, list) or not scenes
        or summary.get("scene_count") != len(scenes) or len(scenes) > len(planned)
        or len(resets) != len(scenes)
    ):
        return ["play_scene_record_invalid"]
    for index, (scene, reset) in enumerate(zip(scenes, resets)):
        if (
            not isinstance(scene, Mapping) or not isinstance(reset, Mapping)
            or scene.get("scene_index") != index
            or scene.get("seed") != planned[index].get("seed")
            or reset.get("seed") != planned[index].get("seed")
        ):
            return ["play_scene_seed_mismatch"]
    if sum(event.get("type") == "play_scene_reset" for event in events) != len(scenes) - 1:
        return ["play_scene_record_invalid"]
    return []


__all__ = ["LEARNING_REASON", "is_play", "verify_play_session"]
