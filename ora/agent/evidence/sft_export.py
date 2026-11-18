"""Export a verified agent trace as SFT records: one record per provider exchange,
with the tool surface, the decision attribution and the token accounting beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ora.agent.evidence import request_archive, sandbox_retention
from .canonical import TrajectoryError, _atomic_json, canonical_bytes, file_sha256, sha256_bytes
from .trace_privacy import (
    _allows_public_grasp_contact_pose, _forbidden_path, model_tool_arguments, script_call_ids,
    without_script_authored_messages,
)
from .agent_events import read_agent_events
from .trace_verify import verify_agent_trace


SFT_EXPORT_SCHEMA = "ora.sft-export.v1"
SFT_TOKEN_ACCOUNTING_SCHEMA = "ora.sft-token-accounting.v1"
#: Provider-reported usage fields copied into ``tokens.jsonl`` as reported (``null`` when absent).
USAGE_FIELDS = ("input_tokens", "output_tokens", "reasoning_output_tokens", "cached_input_tokens")


def _iter_provider_exchanges(events: Sequence[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    requests: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = event.get("model_call_id")
        if not isinstance(call_id, str):
            continue
        if event.get("type") == "provider_request":
            requests[call_id] = event
        elif event.get("type") == "provider_response" and call_id in requests:
            yield requests[call_id], event


def _canonical_provider_request_ref(event: Mapping[str, Any]) -> Any:
    payload = event.get("payload")
    canonical = payload.get("canonical_request_ref") if isinstance(payload, Mapping) else None
    return canonical if isinstance(canonical, Mapping) else event.get("artifact_ref")


def _surface_tool_sets(value: Any) -> tuple[set[str], set[str]]:
    """Locate the resolved ToolSurface snapshot's mutation/lifecycle sets."""

    if isinstance(value, Mapping):
        if (
            isinstance(value.get("profile_id"), str)
            and isinstance(value.get("mutation_tools"), list)
        ):
            return (
                {str(item) for item in value.get("mutation_tools", [])},
                {str(item) for item in value.get("lifecycle_tools", [])},
            )
        for child in value.values():
            found = _surface_tool_sets(child)
            if found != (set(), set()):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _surface_tool_sets(child)
            if found != (set(), set()):
                return found
    return set(), set()


def _decision_attribution(
    assistant: Mapping[str, Any],
    mutation_tools: set[str],
    lifecycle_tools: set[str],
) -> dict[str, Any]:
    """Attribute one exported decision (training-readiness boundary)."""

    calls = assistant.get("tool_calls")
    names: list[str] = []
    if isinstance(calls, list):
        for call in calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str):
                names.append(name)
    executor: str | None = None
    if any(name in mutation_tools for name in names):
        kind = "robot_action"
    elif any(name in lifecycle_tools for name in names):
        kind = "lifecycle"
    else:
        kind = "read_only_or_reflection"
    return {
        "decision_kind": kind,
        "delegated_executor": executor,
        "tool_call_names": names,
    }


def check_seal(root: Path) -> bool:
    """Whether the run is sealed; a sealed run's files must still match its seal."""

    seal_path = root / "seal.json"
    if seal_path.is_file():
        seal_record = json.loads(seal_path.read_text(encoding="utf-8"))
        for record in seal_record.get("inventory", []):
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise TrajectoryError("sealed inventory is invalid")
            if sandbox_retention.omitted_record(root, record):
                continue  # save_sandbox: false: hashes only; SFT export reads no sandbox file
            source = (root / record["path"]).resolve()
            present = source.is_file() and file_sha256(source) == record.get("sha256")
            # A provider request the landing archived is present when the archive holds its bytes.
            if not source.is_relative_to(root) or not (
                present or request_archive.inventory_record_present(root, record)
            ):
                raise TrajectoryError(f"sealed source artifact drift: {record['path']}")
    return seal_path.is_file()


def check_trace_integrity(root: Path) -> dict[str, Any]:
    """A sealed run's files still match its seal, and its agent trace verifies."""

    check_seal(root)
    verification = verify_agent_trace(root)
    if verification["status"] != "valid":
        raise TrajectoryError("invalid agent trace: " + ", ".join(verification["errors"]))
    return verification


def export_sft(run_root: Path, destination: Path) -> Path:
    """Export validated native exchanges as OpenAI-compatible tool-use JSONL.

    ``tokens.jsonl`` carries each record's model call and its provider-reported
    usage: what the serving model processed for that call, whose history may
    hold reasoning the record leaves out.
    """

    root = run_root.resolve()
    verification = check_trace_integrity(root)
    evaluation_path = root / "evaluation.json"
    if not evaluation_path.is_file():
        raise TrajectoryError(
            "SFT export requires a completed run evaluation; evaluate the "
            "run first"
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "valid":
        raise TrajectoryError(
            "SFT export refuses an invalid run evaluation: "
            + ", ".join(str(item) for item in evaluation.get("reasons", []))
        )
    eligibility = evaluation.get("learning_eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
        reasons = (
            eligibility.get("reasons", [])
            if isinstance(eligibility, Mapping) else ["missing"]
        )
        raise TrajectoryError(
            "SFT export refuses a learning-ineligible run: "
            + ", ".join(str(item) for item in reasons)
        )
    events = read_agent_events(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    allow_public_grasp_contact_pose = _allows_public_grasp_contact_pose(plan)
    script_calls = script_call_ids(plan, events)
    model_arguments = model_tool_arguments(events)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    mutation_tools, lifecycle_tools = _surface_tool_sets(plan)
    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    loss_masks: list[dict[str, Any]] = []
    token_entries: list[dict[str, Any]] = []
    for request_event, response_event in _iter_provider_exchanges(events):
        # Older native traces used the provider_request artifact for both the
        # canonical and wire request. New protocol adapters preserve each
        # separately and SFT always consumes the canonical OpenAI/ChatML IR.
        request_ref = _canonical_provider_request_ref(request_event)
        if not isinstance(request_ref, Mapping) or not isinstance(request_ref.get("path"), str):
            raise TrajectoryError("provider request lacks an exact artifact")
        request_bytes = request_archive.verified_bytes(root, request_ref)  # file or post-seal archive
        if request_bytes is None:
            raise TrajectoryError("canonical provider request artifact drift")
        request = json.loads(request_bytes.decode("utf-8"))
        response = response_event.get("payload")
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise TrajectoryError("provider exchange is not exportable")
        if response.get("accepted") is False:
            # Rejected malformed Provider output remains exact audit evidence,
            # but is never silently promoted into an SFT conversation.
            continue
        assistant = response.get("message")
        if not isinstance(assistant, Mapping):
            raise TrajectoryError("provider response lacks an assistant message")
        assistant = {
            key: value for key, value in assistant.items()
            if key != "reasoning_content" and not str(key).startswith("_")
        }
        if assistant.get("role") != "assistant":
            raise TrajectoryError("SFT target must be a Language-Model assistant response")
        record = {
            "messages": [*list(request.get("messages", [])), assistant],
            "tools": list(request.get("tools", [])),
        }
        forbidden = _forbidden_path(
            without_script_authored_messages(record, script_calls),
            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
            model_arguments=model_arguments,
        )
        if forbidden:
            raise TrajectoryError(f"privileged field in SFT export: {forbidden}")
        attributions.append({
            "index": len(records),
            **_decision_attribution(assistant, mutation_tools, lifecycle_tools),
        })
        loss_masks.append({
            "index": len(records),
            "target_author": "language_model",
            "target_message_index": len(record["messages"]) - 1,
            "message_loss_weights": [0] * (len(record["messages"]) - 1) + [1],
        })
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        token_entries.append({
            "index": len(records),
            "model_call_id": request_event.get("model_call_id"),
            "provider_usage": {field: usage.get(field) for field in USAGE_FIELDS},
        })
        records.append(record)

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    jsonl = destination / "conversations.openai-tools.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    attribution_jsonl = destination / "attribution.jsonl"
    with attribution_jsonl.open("w", encoding="utf-8") as stream:
        for item in attributions:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    attribution_counts: dict[str, int] = {}
    loss_mask_jsonl = destination / "loss-mask.jsonl"
    with loss_mask_jsonl.open("w", encoding="utf-8") as stream:
        for item in loss_masks:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    for item in attributions:
        attribution_counts[item["decision_kind"]] = (
            attribution_counts.get(item["decision_kind"], 0) + 1
        )
    tokens_jsonl = destination / "tokens.jsonl"
    with tokens_jsonl.open("w", encoding="utf-8") as stream:
        for item in token_entries:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    reported = [item["provider_usage"] for item in token_entries if isinstance(item["provider_usage"]["input_tokens"], int)]
    token_accounting = {
        "schema": SFT_TOKEN_ACCOUNTING_SCHEMA,
        "provider_usage": {
            "source": "provider_response.usage",
            "records": len(token_entries),
            "records_reported": len(reported),
            **{field: sum(item[field] or 0 for item in reported) for field in USAGE_FIELDS},
        },
        "claim_boundary": (
            "what the serving model reported for each exported call; the call's history may carry "
            "reasoning the record omits, so these are not the records' own token counts"
        ),
    }
    manifest = {
        "schema": SFT_EXPORT_SCHEMA,
        "run_id": result.get("run_id"),
        "source_commit": plan.get("source", {}).get("commit"),
        "trace_final_sha256": verification.get("final_event_sha256"),
        "context_profile": plan.get("policy", {}).get("context_profile"),
        "context_projection_sha256": sha256_bytes(canonical_bytes([
            event.get("payload") for event in events
            if event.get("type") == "context_projected"
        ])),
        "exact_request_sha256": [
            (_canonical_provider_request_ref(event) or {}).get("sha256")
            for event in events
            if event.get("type") == "provider_request"
        ],
        "official_success": result.get("official_success"),
        "termination": result.get("termination"),
        "record_count": len(records),
        "attribution_counts": attribution_counts,
        "token_accounting": token_accounting,
        "loss_mask_contract": {
            "level": "message",
            "target": "final_language_model_assistant_response_only",
            "history_tools_observations_and_receipts": "context_only",
            "tokenizer_compilation_required": True,
        },
        "files": [{
            "path": jsonl.name,
            "sha256": file_sha256(jsonl),
            "bytes": jsonl.stat().st_size,
        }, {
            "path": attribution_jsonl.name,
            "sha256": file_sha256(attribution_jsonl),
            "bytes": attribution_jsonl.stat().st_size,
        }, {
            "path": loss_mask_jsonl.name,
            "sha256": file_sha256(loss_mask_jsonl),
            "bytes": loss_mask_jsonl.stat().st_size,
        }, {
            "path": tokens_jsonl.name,
            "sha256": file_sha256(tokens_jsonl),
            "bytes": tokens_jsonl.stat().st_size,
        }],
    }
    if "context_retention" in plan.get("policy", {}):
        manifest["context_retention"] = plan["policy"]["context_retention"]
    _atomic_json(destination / "manifest.json", manifest)
    return destination
