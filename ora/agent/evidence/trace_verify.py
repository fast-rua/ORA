"""Verify a sealed agent trace: the hash chain, artifact bindings, action
discards, the actions the Supervisor refused inside an invocation, round
termination and the privacy check over every policy-visible event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import request_archive
from .operator_messages import operator_trace_errors
from .agent_events import AGENT_EVENT_SCHEMA, read_agent_events
from .canonical import TrajectoryError, canonical_bytes, sha256_bytes
from .trace_privacy import (
    SCRIPT_CALL_ORIGINS, SCRIPT_TOOLS, _allows_public_grasp_contact_pose, _forbidden_path, declared_argument_names,
    policy_argument_values, proposal_without_model_arguments, reply_tool_arguments, without_script_authored_fields,
    without_script_authored_messages,
)


#: Round budgets that may overtake a submitted action (``action_discarded``).
ACTION_DISCARD_REASONS = frozenset({"model_call_budget", "wall_time_budget", "robot_decision_budget"})
#: The receipt the Supervisor records for an action it refused before it started (episode/tool_execution.py).
SUPERVISOR_REFUSAL_SCHEMA = "ora.tool-action-refusal.v1"
#: The artifacts that repeat a script's result: the requests of the model and of its context compactor, and the
#: context packets whose ``compactor_messages`` the compactor's request repeats.
_REPEATS_SCRIPT_RESULTS = request_archive.REQUEST_KINDS | {
    "rollout_compactor_canonical_request", "rollout_compactor_wire_request",
    "lossless_context_semantic_packet", "episode_context_capsule_packet",
}
#: Policy activity that may not follow a discarded action before its round closes.
_ACTIVITY_AFTER_DISCARD = frozenset({
    "observation_received", "provider_request", "tool_call_proposed", "action_proposed",
    "environment_receipt",
})


def _artifact_refs(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str)
            and isinstance(value.get("bytes"), int)
        ):
            yield value
        for child in value.values():
            yield from _artifact_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_refs(child)


def artifact_bytes(root: Path, relative: str) -> bytes | None:
    """The bytes an artifact reference names: its file, or its copy in the run's request archive.

    After a seal, landing moves the provider requests into ``native/provider-requests.tar.zst``
    (``request_archive.compact_after_seal``); a verifier that only looked for files would
    call every archived artifact missing. A path outside the run is missing too.
    """
    if not (root / relative).resolve().is_relative_to(root):
        return None
    try:
        return request_archive.read_bytes(root, relative)
    except (FileNotFoundError, request_archive.RequestArchiveError):
        return None


def is_supervisor_refusal(receipt: Any) -> bool:
    """Whether a receipt body is the Supervisor's refusal of an action before it started: nothing ran and
    nothing was published, so the next action on the same observation (a code-mode script's next call, or in
    tool-call mode the model's next tool call) may carry its identity. The trace verifier, the Pi session, the
    world view, the Viewer and the Runtime metrics share this test."""

    return (isinstance(receipt, Mapping) and receipt.get("schema") == SUPERVISOR_REFUSAL_SCHEMA
            and receipt.get("status") == "refused" and receipt.get("native_actions") == 0)


def _supervisor_refusal(root: Path, event: Mapping[str, Any], pending: Mapping[str, Any]) -> bool:
    """Whether a receipt is the Supervisor's refusal of the pending action: zero spend, nothing published."""

    ref = event.get("artifact_ref")
    path = ref.get("path") if isinstance(ref, Mapping) else None
    data = artifact_bytes(root, path) if isinstance(path, str) else None
    if data is None or sha256_bytes(data) != ref.get("sha256"):
        return False
    try:
        receipt = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    payload = event.get("payload")
    return (
        is_supervisor_refusal(receipt)
        and receipt.get("action_id") == pending.get("action_id")
        and isinstance(payload, Mapping) and payload.get("action_id") == pending.get("action_id")
        and event.get("causal_parent_id") == pending.get("event_id")
    )


def _ledger_refusals(payload: Any) -> set[Any]:
    """The receipt events an invocation's execution ledger records as refused actions that used nothing."""

    execution = payload.get("execution") if isinstance(payload, Mapping) else None
    actions = execution.get("actions") if isinstance(execution, Mapping) else None
    return {item.get("receipt_event_id") for item in actions or []
            if isinstance(item, Mapping) and item.get("refusal") is not None
            and item.get("native_actions_used") == 0}


def verify_agent_trace(run_root: Path) -> dict[str, Any]:
    """Validate canonical identity, causality, pairing and artifact integrity."""

    root = run_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        events = read_agent_events(root)
    except TrajectoryError as exc:
        return {
            "status": "invalid", "errors": [str(exc)], "warnings": [],
            "event_count": 0,
        }
    previous_hash: str | None = None
    known_ids: set[str] = set()
    pending_tools: dict[str, dict[str, Any]] = {}
    seen_tool_ids: set[str] = set()
    mutation_by_model_call: dict[str, int] = {}
    requests: set[str] = set()
    responses: set[str] = set()
    # Requests an operator stop aborted in flight: a provider_error marked
    # interrupted, after session_interrupted, is their only answer.
    interrupted_requests: set[str] = set()
    session_interrupted = False
    request_steps: dict[str, Any] = {}
    request_refs: dict[str, Any] = {}
    request_ids: dict[str, Any] = {}
    failed_attempt_counts: dict[str, int] = {}
    terminal_attempt_failures: set[str] = set()
    accepted_mutations: dict[str, set[str]] = {}
    action_steps: set[Any] = set()
    awaiting_fresh: dict[str, Any] | None = None
    # A proposal a budget discarded, open until its round's recorded end.
    discarded: dict[str, Any] | None = None
    closed_rounds = 0
    mutation_tools = {"robot.submit_action"}
    # A code-mode script (plan policy.code_mode, agent/pi/code_mode.py) may act several times in its reply.
    code_tools: set[str] = set()
    code_calls: set[tuple[Any, Any]] = set()
    script_calls: set[str] = set()
    # The tool calls the model wrote in the replies so far, by id (trace_privacy.reply_tool_arguments): their
    # arguments are the model's own text wherever a reply, request or packet repeats them exactly.
    model_arguments: dict[str, str] = {}
    # Receipt events of actions the Supervisor refused, by the invocation they ran inside (a code-mode script or a
    # synchronous tool call); its execution ledger must record each.
    refusals: dict[tuple[Any, Any], set[Any]] = {}
    allow_public_grasp_contact_pose = False
    safe_field_names: frozenset[str] = frozenset()
    argument_names: frozenset[str] = frozenset()
    argument_values: dict[str, frozenset[str]] = {}
    plan: dict[str, Any] = {}
    plan_path = root / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            surface = plan.get("policy_composition", {}).get("tool_surface", {})
            allow_public_grasp_contact_pose = _allows_public_grasp_contact_pose(plan)
            configured = surface.get("mutation_tools")
            if isinstance(configured, list) and all(isinstance(name, str) for name in configured):
                mutation_tools = set(configured)
            # glue_code is the contract's key before the rename, kept so runs planned before it still verify.
            policy = plan.get("policy") or {}
            code_tools = set(SCRIPT_TOOLS) if policy.get("code_mode") or policy.get("glue_code") else set()
            capabilities = surface.get("capabilities")
            view_config = surface.get("view_config")
            automatic_views = (
                view_config.get("automatic_actual_views")
                if isinstance(view_config, Mapping) else None
            )
            if (
                isinstance(capabilities, list)
                and "automatic_current_auxiliary_contact_sheet_v1" in capabilities
                and isinstance(automatic_views, list)
                and "auxiliary_contact_sheet" in automatic_views
            ):
                # This exact field names a hash-bound RGB composition.  It is
                # not simulator contact evidence.  Descendants are still
                # scanned normally, so the exception cannot hide privileged
                # payloads beneath the derived-view label.
                safe_field_names = frozenset({"auxiliary_contact_sheet"})
            # Argument names the run's own tool surface declares (a toolset's `contact_speed_m_s`, I-0011) pass
            # only where they hold the policy's own argument (trace_privacy._policy_argument); their values are
            # still scanned.
            argument_names = declared_argument_names(surface)
            argument_values = policy_argument_values(surface, events)
        except (OSError, json.JSONDecodeError):
            errors.append("plan:invalid")

    errors.extend(operator_trace_errors(events, plan if plan_path.is_file() else {}))
    for index, event in enumerate(events):
        if event.get("schema") != AGENT_EVENT_SCHEMA:
            errors.append(f"event:{index}:schema")
        if event.get("seq") != index:
            errors.append(f"event:{index}:sequence")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in known_ids:
            errors.append(f"event:{index}:event_id")
        else:
            known_ids.add(event_id)
        if event.get("previous_event_sha256") != previous_hash:
            errors.append(f"event:{index}:previous_hash")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        expected = sha256_bytes(canonical_bytes(body))
        if event.get("event_sha256") != expected:
            errors.append(f"event:{index}:hash")
        previous_hash = event.get("event_sha256") if isinstance(event.get("event_sha256"), str) else None
        parent = event.get("causal_parent_id")
        if parent is not None and parent not in known_ids:
            errors.append(f"event:{index}:causal_parent")
        if event.get("type") == "provider_response":
            model_arguments.update(reply_tool_arguments(event.get("payload")))
        # The result of a script the model called itself (its proposal is in code_calls): the audit leaves out what
        # the script wrote (trace_privacy.SCRIPT_AUTHORED_FIELDS), here and where a later request repeats that
        # result; its calls' own results are audited in full.
        script_result = (event.get("artifact_ref") if event.get("type") == "tool_result"
                         and (event.get("model_call_id"), event.get("tool_call_id")) in code_calls else None)
        artifacts = list(_artifact_refs(event.get("artifact_ref")))
        artifacts.extend(_artifact_refs(event.get("payload")))
        for artifact in artifacts:
            relative = artifact.get("path")
            if not isinstance(relative, str):
                errors.append(f"event:{index}:artifact_path")
            else:
                data = artifact_bytes(root, relative)
                if data is None:
                    errors.append(f"event:{index}:artifact_missing")
                elif sha256_bytes(data) != artifact.get("sha256"):
                    errors.append(f"event:{index}:artifact_hash")
                elif artifact.get("media_type") == "application/json":
                    try:
                        artifact_value = json.loads(data.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        errors.append(f"event:{index}:artifact_json")
                    else:
                        if artifact is script_result:
                            artifact_value = without_script_authored_fields(artifact_value)
                        elif artifact.get("kind") in _REPEATS_SCRIPT_RESULTS:
                            artifact_value = without_script_authored_messages(artifact_value, script_calls)
                        forbidden_artifact = _forbidden_path(
                            artifact_value,
                            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                            safe_field_names=safe_field_names,
                            argument_values=argument_values,
                            argument_declarations=(argument_names if artifact.get("kind")
                                                   in request_archive.REQUEST_KINDS else frozenset()),
                            model_arguments=model_arguments,
                        )
                        if forbidden_artifact:
                            errors.append(
                                f"event:{index}:privileged_artifact_field:{forbidden_artifact}"
                            )

        event_type = event.get("type")
        call_id = event.get("model_call_id")
        tool_id = event.get("tool_call_id")
        payload = event.get("payload")
        forbidden_payload = _forbidden_path(
            (proposal_without_model_arguments(payload, tool_id, model_arguments, script_calls)
             if event_type == "tool_call_proposed" else payload),
            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
            safe_field_names=safe_field_names,
            argument_values=argument_values,
            model_arguments=model_arguments,
        )
        if forbidden_payload:
            errors.append(f"event:{index}:privileged_field:{forbidden_payload}")
        if event_type == "provider_request" and isinstance(call_id, str):
            if awaiting_fresh is not None:
                errors.append(f"event:{index}:provider_request_before_fresh_observation")
            requests.add(call_id)
            request_steps[call_id] = event.get("observation_step")
            request_refs[call_id] = event.get("artifact_ref")
            request_ids[call_id] = event.get("event_id")
        elif event_type == "provider_attempt_failed":
            sent_sha256 = payload.get("request_sha256") if isinstance(payload, Mapping) else None
            recorded_sha256 = (request_refs.get(call_id) or {}).get("sha256")
            if (
                not isinstance(call_id, str) or call_id not in requests
                or call_id in responses or call_id in terminal_attempt_failures
                or event.get("observation_step") != request_steps.get(call_id)
                or event.get("causal_parent_id") != request_ids.get(call_id)
            ):
                errors.append(f"event:{index}:provider_attempt_pairing")
            if not isinstance(payload, Mapping) or (
                payload.get("schema") != "ora.provider-attempt-failure.v1"
                or payload.get("model_call_id") != call_id
                or payload.get("tool_execution_started") is not False
                or type(payload.get("attempt")) is not int
                or payload.get("attempt") != failed_attempt_counts.get(call_id, 0) + 1
                or type(payload.get("retry_scheduled")) is not bool
                or sent_sha256 != recorded_sha256
                or event.get("artifact_ref") is None
            ):
                errors.append(f"event:{index}:provider_attempt_contract")
            if isinstance(call_id, str):
                failed_attempt_counts[call_id] = failed_attempt_counts.get(call_id, 0) + 1
                if isinstance(payload, Mapping) and payload.get("retry_scheduled") is False:
                    terminal_attempt_failures.add(call_id)
        elif event_type == "session_interrupted":
            session_interrupted = True
        elif (
            event_type == "provider_error"
            and isinstance(payload, Mapping) and payload.get("interrupted") is True
        ):
            if (
                not session_interrupted or not isinstance(call_id, str)
                or call_id not in requests or call_id in responses
                or call_id in interrupted_requests
                or event.get("observation_step") != request_steps.get(call_id)
                or event.get("causal_parent_id") != request_ids.get(call_id)
            ):
                errors.append(f"event:{index}:interrupted_provider_error")
            else:
                interrupted_requests.add(call_id)
        elif event_type == "provider_response" and isinstance(call_id, str):
            if call_id in interrupted_requests:
                errors.append(f"event:{index}:response_after_interrupted_request")
            if call_id in terminal_attempt_failures:
                errors.append(f"event:{index}:response_after_terminal_attempt_failure")
            if call_id in failed_attempt_counts and (
                not isinstance(payload, Mapping)
                or payload.get("attempts") != failed_attempt_counts[call_id] + 1
            ):
                errors.append(f"event:{index}:provider_attempt_count")
            responses.add(call_id)
            if request_steps.get(call_id) != event.get("observation_step"):
                errors.append(f"event:{index}:cross_observation_provider_response")
        elif event_type == "tool_call_proposed":
            if not isinstance(tool_id, str) or tool_id in seen_tool_ids:
                errors.append(f"event:{index}:tool_call_id")
            else:
                seen_tool_ids.add(tool_id)
                pending_tools[tool_id] = event
            origin = payload.get("origin") if isinstance(payload, Mapping) else None
            nested = (bool(code_tools) and isinstance(origin, Mapping)
                      and origin.get("kind") in SCRIPT_CALL_ORIGINS)
            if nested and (call_id, origin.get("parent_tool_call_id")) not in code_calls:
                errors.append(f"event:{index}:code_mode_call_without_script")
            if isinstance(payload, Mapping) and payload.get("name") in code_tools and not nested:
                code_calls.add((call_id, tool_id))
                script_calls.add(tool_id)
            if (
                isinstance(payload, Mapping)
                and payload.get("name") in mutation_tools | code_tools
                and payload.get("batch_valid") is not False
                and "read_only" not in (payload.get("effect"), payload.get("state_effect"))  # a dry run acts on nothing
            ):
                if isinstance(call_id, str):  # a script's calls count as its one script call
                    mutation_by_model_call[call_id] = mutation_by_model_call.get(call_id, 0) + (not nested)
                    accepted_mutations.setdefault(call_id, set()).add(str(tool_id))
        elif event_type == "tool_result":
            if not isinstance(tool_id, str) or tool_id not in pending_tools:
                errors.append(f"event:{index}:orphan_tool_result")
            else:
                proposed = pending_tools.pop(tool_id)
                if proposed.get("model_call_id") != call_id:
                    errors.append(f"event:{index}:cross_model_call_tool_result")
                if proposed.get("observation_step") != event.get("observation_step"):
                    errors.append(f"event:{index}:cross_observation_tool_result")
            refused = refusals.pop((call_id, tool_id), set())
            if refused - _ledger_refusals(payload):
                errors.append(f"event:{index}:refusal_not_in_execution")
        elif event_type == "action_proposed":
            if (
                not isinstance(call_id, str)
                or not isinstance(tool_id, str)
                or tool_id not in accepted_mutations.get(call_id, set())
            ):
                errors.append(f"event:{index}:action_without_unique_mutation")
            step = event.get("observation_step")
            if step in action_steps:
                errors.append(f"event:{index}:multiple_actions_for_observation")
            action_steps.add(step)
            awaiting_fresh = {
                "step": step, "receipt": False, "event_id": event.get("event_id"),
                "action_id": payload.get("action_id") if isinstance(payload, Mapping) else None,
                # The open invocation the action runs inside (a code-mode script's own call, or a synchronous tool
                # call executing its own actions), whose tool_result carries the execution ledger.
                "parent": ((call_id, tool_id) if isinstance(tool_id, str) and tool_id in pending_tools
                           and pending_tools[tool_id].get("model_call_id") == call_id else None),
            }
        elif event_type == "action_discarded":
            # The Supervisor ended the round on a budget before executing the
            # pending proposal: valid only as that proposal's direct closure.
            if (
                awaiting_fresh is None
                or awaiting_fresh.get("receipt") is not False
                or event.get("causal_parent_id") != awaiting_fresh.get("event_id")
                or event.get("observation_step") != awaiting_fresh.get("step")
                or not isinstance(payload, Mapping)
                or set(payload) != {"action_id", "reason"}
                or payload.get("action_id") != awaiting_fresh.get("action_id")
                or payload.get("reason") not in ACTION_DISCARD_REASONS
            ):
                errors.append(f"event:{index}:action_discarded_contract")
            else:
                discarded = {"index": index, "reason": payload["reason"], "round": closed_rounds}
                awaiting_fresh = None
        elif event_type == "environment_receipt":
            if awaiting_fresh is None:
                errors.append(f"event:{index}:receipt_without_action")
            elif awaiting_fresh.get("parent") is not None and _supervisor_refusal(root, event, awaiting_fresh):
                # The Supervisor refused the action before it started: nothing ran and nothing was published, so
                # no fresh observation is due and another action may follow on the same one, the script's next
                # call or, in tool-call mode, the model's next tool call (a zero-spend refusal leaves the turn
                # open). The invocation's execution ledger must record the refusal (checked at its result).
                refusals.setdefault(awaiting_fresh["parent"], set()).add(event.get("event_id"))
                action_steps.discard(awaiting_fresh["step"])
                awaiting_fresh = None
            else:
                awaiting_fresh["receipt"] = True
        elif event_type == "fresh_observation_published" and awaiting_fresh is not None:
            if awaiting_fresh.get("receipt") is not True:
                errors.append(f"event:{index}:fresh_observation_without_receipt")
            prior_step = awaiting_fresh.get("step")
            fresh_step = event.get("observation_step")
            if isinstance(prior_step, int) and (
                not isinstance(fresh_step, int) or fresh_step <= prior_step
            ):
                errors.append(f"event:{index}:observation_not_fresh")
            awaiting_fresh = None
        if discarded is not None and discarded["index"] != index:
            if event_type in _ACTIVITY_AFTER_DISCARD:
                errors.append(f"event:{index}:activity_after_discarded_action")
            elif event_type == "run_terminated":
                termination = payload.get("termination") if isinstance(payload, Mapping) else None
                if termination != discarded["reason"]:
                    errors.append(f"event:{index}:discarded_action_termination_mismatch")
                discarded = None
            elif event_type == "fresh_observation_published":
                # The next round of a persistent session: its round result
                # records the termination the discard must name.
                if _round_termination(root, discarded["round"]) != discarded["reason"]:
                    errors.append(f"event:{index}:discarded_action_termination_mismatch")
                discarded = None
        if event_type == "official_verdict":
            closed_rounds += 1

    for tool_id in sorted(pending_tools):
        errors.append(f"missing_tool_result:{tool_id}")
    # An invocation's refusals are checked against its ledger at its tool_result; a result that never came cannot
    # confirm them either.
    for _call_id, tool_id in sorted(refusals, key=str):
        errors.append(f"refusal_not_in_execution:{tool_id}")
    for call_id, count in mutation_by_model_call.items():
        if count > 1:
            errors.append(f"multiple_mutations:{call_id}")
    for call_id in sorted(responses - requests):
        errors.append(f"response_without_request:{call_id}")
    for call_id in sorted(requests - responses - interrupted_requests):
        errors.append(f"request_without_response:{call_id}")
    if awaiting_fresh is not None:
        errors.append("action_without_fresh_observation")
    if discarded is not None:
        errors.append("discarded_action_without_termination")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "event_count": len(events),
        "final_event_sha256": previous_hash,
    }


def _round_termination(root: Path, round_index: int) -> Any:
    """The termination a persistent session recorded for one of its rounds."""

    path = root / "rounds" / f"round-{round_index:04d}" / "result.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record.get("termination") if isinstance(record, Mapping) else None
