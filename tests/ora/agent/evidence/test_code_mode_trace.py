"""The agent-trace verifier counts a code-mode script as its reply's one mutation, and audits what the
script wrote itself only through its calls.

The robot calls a script makes (plan ``policy.code_mode``) are recorded as ``<call>#NNN`` with
``origin`` ``{kind, parent_tool_call_id, seq}``. The kind is ``code_mode``, or ``glue_code`` in runs
sealed before the rename; any other kind counts each call as a mutation of its own, and a nested call
names a script of the same reply. An action the Supervisor refused before it started publishes
nothing: the script, or in tool-call mode the model's next tool call, may act again on the same
observation, and the execution ledger of the invocation it ran inside records the refusal. The
privileged-name audit leaves out what a script printed, returned or handed back, the model's own
computation over its calls' results, each audited as its own event; the rest of its result stays audited,
in the result and where a later request of the model or the compactor repeats it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.trace_verify import verify_agent_trace
from tests.support.agent.agent_events import TrajectoryRecorder

CALL = "model-call-000001"
SCRIPT = "call-script"


def _run(root: Path, kind: str, *, parent: str = SCRIPT, plan_key: str = "code_mode") -> Path:
    """One reply whose run_python script moved the robot twice; ``plan_key`` names the plan's contract."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {plan_key: {"contract": "ora.code-mode-script.v1"}},
        "policy_composition": {"tool_surface": {"mutation_tools": ["move_to"]}},
    }), encoding="utf-8")
    recorder = TrajectoryRecorder(root, run_id="code-mode-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": CALL}
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    recorder.append("tool_call_proposed", tool_call_id=SCRIPT, **turn, payload={
        "name": "run_python", "arguments": {"code": "tools.move_to(...)"}, "batch_valid": True, "effect": "mutation"})
    for seq in (1, 2):
        nested = f"{SCRIPT}#{seq:03d}"
        recorder.append("tool_call_proposed", tool_call_id=nested, **turn, payload={
            "name": "move_to", "arguments": {}, "batch_valid": True, "effect": "mutation",
            "origin": {"kind": kind, "parent_tool_call_id": parent, "seq": seq}})
        recorder.append("tool_result", tool_call_id=nested, **turn, payload={"ok": True})
    recorder.append("tool_result", tool_call_id=SCRIPT, **turn, payload={"ok": True})
    return root


@pytest.mark.parametrize("kind", ["code_mode", "glue_code"])
def test_a_scripts_calls_count_as_its_one_call(tmp_path: Path, kind: str) -> None:
    verification = verify_agent_trace(_run(tmp_path / kind, kind))
    assert verification["status"] == "valid", verification["errors"]


def test_a_plan_from_before_the_rename_is_code_mode(tmp_path: Path) -> None:
    """Runs planned before the rename name the contract ``policy.glue_code`` and their calls' origin ``glue_code``."""

    verification = verify_agent_trace(_run(tmp_path / "glue", "glue_code", plan_key="glue_code"))
    assert verification["status"] == "valid", verification["errors"]
    result = {"ok": True, "status": "handed_back", "calls": [{"call": 1, "tool": "mark_point", "ok": True}],
              "handed_back": {"reason": "pregrasp_submitted", "data": {"contact": POINT}}}
    verification = verify_agent_trace(_script_result(tmp_path / "glue-authored", result, name="run_python",
                                                     plan_key="glue_code"))
    assert verification["status"] == "valid", verification["errors"]


def test_a_plan_without_either_key_counts_each_call(tmp_path: Path) -> None:
    errors = verify_agent_trace(_run(tmp_path / "none", "glue_code", plan_key="other"))["errors"]
    assert f"multiple_mutations:{CALL}" in errors


def _previews(root: Path, effect: str, *, acted: bool = False) -> Path:
    """One tool-call reply with two ``move_to`` previews of the given recorded effect."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy_composition": {"tool_surface": {"mutation_tools": ["move_to"]}},
    }), encoding="utf-8")
    recorder = TrajectoryRecorder(root, run_id="tool-call-run", policy_session_id="pi-session")
    turn = {"model_call_id": CALL, "observation_step": 0}
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    for call in ("call-1", "call-2"):
        recorder.append("tool_call_proposed", tool_call_id=call, **turn, payload={
            "name": "move_to", "arguments": {"dry_run": True}, "batch_valid": True, "effect": effect})
        if acted and call == "call-2":
            recorder.append("action_proposed", tool_call_id=call, **turn, payload={"action_id": "action-1"})
        recorder.append("tool_result", tool_call_id=call, **turn, payload={"ok": True})
    return root


def test_previews_in_one_reply_are_not_mutations(tmp_path: Path) -> None:
    verification = verify_agent_trace(_previews(tmp_path / "preview", "read_only"))
    assert verification["status"] == "valid", verification["errors"]


def test_two_mutations_in_one_reply_stay_invalid(tmp_path: Path) -> None:
    assert f"multiple_mutations:{CALL}" in verify_agent_trace(_previews(tmp_path / "moves", "mutation"))["errors"]


def test_a_read_only_call_cannot_publish_an_action(tmp_path: Path) -> None:
    errors = verify_agent_trace(_previews(tmp_path / "acted", "read_only", acted=True))["errors"]
    assert any(error.endswith("action_without_unique_mutation") for error in errors), errors


def test_calls_of_another_origin_kind_are_mutations_of_their_own(tmp_path: Path) -> None:
    assert verify_agent_trace(_run(tmp_path / "other", "other"))["errors"] == [f"multiple_mutations:{CALL}"]


@pytest.mark.parametrize("kind", ["code_mode", "glue_code"])
def test_a_nested_call_names_a_script_of_its_reply(tmp_path: Path, kind: str) -> None:
    errors = verify_agent_trace(_run(tmp_path / kind, kind, parent="call-elsewhere"))["errors"]
    assert errors == ["event:3:code_mode_call_without_script", "event:5:code_mode_call_without_script"]


def _refused_then_executed(root: Path, *, script: bool = True, receipt_schema: str = "ora.tool-action-refusal.v1",
                           ledger_refusal: str | None = "stale_or_mismatched_recovery_target",
                           result: bool = True) -> Path:
    """One reply whose script's first action the Supervisor refused and whose second, on the same observation
    and with the same identity, executed (``orchestrator/episode/tool_execution.py``); ``script=False`` makes
    the call a tool call of its own instead, ``result=False`` leaves out the script's result."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {"code_mode": {"contract": "ora.code-mode-script.v1"}},
        "policy_composition": {"tool_surface": {"mutation_tools": ["move_to"]}},
    }), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="code-mode-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": CALL}
    parent = SCRIPT if script else "call-move"
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    recorder.append("tool_call_proposed", tool_call_id=parent, **turn, payload={
        "name": "run_python" if script else "move_to", "arguments": {}, "batch_valid": True, "effect": "mutation"})
    receipts, ledger = [], []
    for seq, raw in enumerate((
        {"schema": receipt_schema, "action_id": "action-0", "status": "refused",
         "reason": "stale_or_mismatched_recovery_target", "native_actions": 0},
        {"action_id": "action-0", "status": "reached", "native_actions": 3},
    ), start=1):
        nested = f"{SCRIPT}#{seq:03d}"
        if script:
            recorder.append("tool_call_proposed", tool_call_id=nested, **turn, payload={
                "name": "move_to", "arguments": {}, "batch_valid": True, "effect": "mutation",
                "origin": {"kind": "code_mode", "parent_tool_call_id": SCRIPT, "seq": seq}})
        action = recorder.append("action_proposed", tool_call_id=parent, **turn, payload={
            "action_id": "action-0", "observation_id": "obs-0"})
        receipt = recorder.append(
            "environment_receipt", causal_parent_id=action["event_id"], observation_step=0,
            payload={"receipt_id": f"receipt-{seq}", "action_id": "action-0"},
            artifact_ref=store.put_json(raw, kind="environment_receipt"))
        receipts.append(receipt["event_id"])
        ledger.append({"action_id": "action-0", "receipt_event_id": receipt["event_id"],
                       "refusal": ledger_refusal if seq == 1 else None,
                       "native_actions_used": raw["native_actions"]})
        if seq == 2:
            recorder.append("fresh_observation_published", observation_step=1, payload={"observation_id": "obs-1"})
        if script:
            recorder.append("tool_result", tool_call_id=nested, **turn, payload={"ok": True})
    if result:
        recorder.append("tool_result", tool_call_id=parent, **turn, payload={
            "ok": True, "execution": {"schema": "ora.tool-execution.v1", "actions": ledger}})
    return root


def test_a_script_acts_again_after_the_supervisor_refused_its_action(tmp_path: Path) -> None:
    verification = verify_agent_trace(_refused_then_executed(tmp_path / "script"))
    assert verification["status"] == "valid", verification["errors"]


def test_a_receipt_that_is_not_the_supervisors_refusal_used_the_observation(tmp_path: Path) -> None:
    errors = verify_agent_trace(_refused_then_executed(tmp_path / "other", receipt_schema="other"))["errors"]
    assert "event:8:multiple_actions_for_observation" in errors


def test_a_script_refusal_its_ledger_does_not_record_is_invalid(tmp_path: Path) -> None:
    errors = verify_agent_trace(_refused_then_executed(tmp_path / "unrecorded", ledger_refusal=None))["errors"]
    assert errors == ["event:12:refusal_not_in_execution"]


def test_a_script_refusal_whose_result_never_came_is_reported(tmp_path: Path) -> None:
    """No ledger confirms the refusal: the trace names it besides the missing result."""

    errors = verify_agent_trace(_refused_then_executed(tmp_path / "no-result", result=False))["errors"]
    assert errors == [f"missing_tool_result:{SCRIPT}", f"refusal_not_in_execution:{SCRIPT}"]


def _tool_call_refusal(root: Path, *, receipt_schema: str = "ora.tool-action-refusal.v1",
                       ledger_refusal: str | None = "native_action_budget_exhausted") -> Path:
    """Tool-call mode on a synchronous toolkit surface: a ``grasp_execute`` whose first stage executed and whose
    second the Supervisor refused on its exhausted budget, then the model's next reply on the observation that
    refusal kept, a ``move_pose`` that executed (two RoboDojo toolkit runs of 2026-10-24 have this shape)."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {}, "policy_composition": {"tool_surface": {"mutation_tools": ["grasp_execute", "move_pose"]}},
    }), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="tool-call-run", policy_session_id="pi-session")

    def act(turn: dict, tool: str, action_id: str, raw: dict, ledger: list, *, fresh: int | None) -> None:
        action = recorder.append("action_proposed", tool_call_id=tool, **turn, payload={
            "action_id": action_id, "observation_id": f"obs-{turn['observation_step']}"})
        receipt = recorder.append(
            "environment_receipt", causal_parent_id=action["event_id"], observation_step=turn["observation_step"],
            payload={"action_id": action_id}, artifact_ref=store.put_json(raw, kind="environment_receipt"))
        ledger.append({"action_id": action_id, "receipt_event_id": receipt["event_id"],
                       "refusal": ledger_refusal if raw["status"] == "refused" else None,
                       "native_actions_used": raw["native_actions"]})
        if fresh is not None:
            recorder.append("fresh_observation_published", observation_step=fresh,
                            payload={"observation_id": f"obs-{fresh}"})

    for step, call, tool, name in ((0, "model-call-000001", "call-grasp", "grasp_execute"),
                                   (1, "model-call-000002", "call-move", "move_pose")):
        turn = {"observation_step": step, "model_call_id": call}
        recorder.append("provider_request", **turn)
        recorder.append("provider_response", **turn)
        recorder.append("tool_call_proposed", tool_call_id=tool, **turn, payload={
            "name": name, "arguments": {}, "batch_valid": True, "effect": "mutation"})
        ledger: list = []
        if name == "grasp_execute":
            act(turn, tool, "action-0:00-open", {"action_id": "action-0:00-open", "status": "reached",
                                                 "native_actions": 6}, ledger, fresh=1)
            act({**turn, "observation_step": 1}, tool, "action-0:01-pregrasp", {
                "schema": receipt_schema, "action_id": "action-0:01-pregrasp", "status": "refused",
                "reason": "native_action_budget_exhausted", "native_actions": 0}, ledger, fresh=None)
        else:
            act(turn, tool, "action-1", {"action_id": "action-1", "status": "reached", "native_actions": 4},
                ledger, fresh=2)
        recorder.append("tool_result", tool_call_id=tool, **turn, payload={
            "ok": True, "name": name, "execution": {"schema": "ora.tool-execution.v1", "actions": ledger}})
    return root


def test_a_tool_calls_refused_action_leaves_the_observation_to_the_next_reply(tmp_path: Path) -> None:
    """Tool-call mode closes a refused action on its refusal receipt as a script does: the next request and
    action on the observation it kept are no second action on that observation."""

    verification = verify_agent_trace(_tool_call_refusal(tmp_path / "tool-call"))
    assert verification["status"] == "valid", verification["errors"]


def test_a_tool_calls_receipt_that_is_not_a_refusal_used_the_observation(tmp_path: Path) -> None:
    errors = verify_agent_trace(_tool_call_refusal(tmp_path / "other", receipt_schema="other"))["errors"]
    assert errors == ["event:9:provider_request_before_fresh_observation",
                      "event:12:multiple_actions_for_observation"]


def test_a_tool_calls_refusal_its_ledger_does_not_record_is_invalid(tmp_path: Path) -> None:
    errors = verify_agent_trace(_tool_call_refusal(tmp_path / "unrecorded", ledger_refusal=None))["errors"]
    assert errors == ["event:8:refusal_not_in_execution"]


POINT = [-0.034, -0.049, 0.805]


def _script_result(root: Path, result: dict, *, child: dict | None = None, child_name: str = "mark_point",
                   name: str = "bash", code_mode: bool = True, plan_key: str = "code_mode",
                   request: dict | None = None, request_kind: str = "native_canonical_provider_request") -> Path:
    """One reply whose ``name`` call returned ``result``. In code mode a script tool's call made one read-only
    call of its own, ``child_name``, whose result is ``child`` (events: 0 request, 1 response, 2 the call, 3 its
    call, 4 that call's result, 5 the call's result); any other call nests nothing (3 is its result).
    ``request`` is the artifact of a next model call, the event after the call's result.
    ``code_mode=False`` is tool-call mode."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {plan_key: {"contract": "ora.code-mode-script.v1"}} if code_mode else {},
        "policy_composition": {"tool_surface": {"mutation_tools": ["move_pose"]}},
    }), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="code-mode-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": CALL}
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    recorder.append("tool_call_proposed", tool_call_id=SCRIPT, **turn, payload={
        "name": name, "arguments": {"command": "python step1.py"}, "batch_valid": True, "effect": "mutation"})
    if code_mode and name in ("bash", "run_python"):
        origin = {"kind": "code_mode", "parent_tool_call_id": SCRIPT, "seq": 1}
        recorder.append("tool_call_proposed", tool_call_id=f"{SCRIPT}#001", **turn, payload={
            "name": child_name, "arguments": {}, "batch_valid": True, "effect": "read_only", "origin": origin})
        recorder.append("tool_result", tool_call_id=f"{SCRIPT}#001", **turn, payload={"ok": True, "origin": origin},
                        artifact_ref=store.put_json(child or {"ok": True, "marks": []}, kind="native_tool_result"))
    recorder.append("tool_result", tool_call_id=SCRIPT, **turn, payload={"ok": True, "name": name},
                    artifact_ref=store.put_json(result, kind="native_tool_result"))
    if request is not None:
        turn = {"observation_step": 0, "model_call_id": "model-call-000002"}
        recorder.append("provider_request", **turn, artifact_ref=store.put_json(request, kind=request_kind))
        recorder.append("provider_response", **turn)
    return root


#: Names the model chose in what its script wrote, shaped as in the code-mode runs of 2026-10-27 to 10-29 the
#: audit invalidated (``$.handed_back.data.contact``, ``$.stdout<json>.red.cands[0].contact``).
AUTHORED = {
    "handed_back": ("bash", {"handed_back": {"reason": "pregrasp_submitted", "data": {"contact": POINT}}}),
    "handed_back_json": ("bash", {"handed_back": {"reason": "lifted", "data": {
        "out": json.dumps({"lift": {"contact": POINT}})}}}),
    "output": ("bash", {"output": json.dumps({"red": {"cands": [{"contact": POINT}]}})}),
    "stdout": ("run_python", {"stdout": json.dumps({"white": {"candidates": [{"contact": POINT}]}})}),
    "stderr": ("run_python", {"stderr": json.dumps({"contacts": [POINT]})}),
    "value": ("run_python", {"value": {"contact": POINT}}),
}


@pytest.mark.parametrize("field", sorted(AUTHORED))
def test_what_a_script_wrote_itself_is_not_privileged_by_its_name(tmp_path: Path, field: str) -> None:
    """The script's printed, returned and handed-back data are the model's computation over its calls' results,
    each audited as its own event, so a key the model named ``contact`` does not invalidate the run."""

    name, authored = AUTHORED[field]
    result = {"ok": True, "status": "completed", "calls": [{"call": 1, "tool": "mark_point", "ok": True}],
              "actions_executed": 0, "action_limit": 12, **authored}
    verification = verify_agent_trace(_script_result(tmp_path / field, result, name=name))
    assert verification["status"] == "valid", verification["errors"]


@pytest.mark.parametrize("child_name", ["mark_point", "run_python"])
def test_a_scripts_call_is_audited_in_full(tmp_path: Path, child_name: str) -> None:
    """A call the script made, one named as a script tool included, is no script of the model's: its result keeps
    the audit under the names the parent's result leaves out."""

    errors = verify_agent_trace(_script_result(tmp_path / child_name, AUTHORED["value"][1], name="run_python",
                                               child={"ok": True, "value": {"contact": POINT}},
                                               child_name=child_name))["errors"]
    assert errors == ["event:4:privileged_artifact_field:$.value.contact"]


@pytest.mark.parametrize(("field", "rest", "path"), [
    ("calls", {"calls": [{"call": 1, "tool": "mark_point", "ok": True, "receipt": {"contact": POINT}}]},
     "$.calls[0].receipt.contact"),
    ("handed_back", {"handed_back": {"reason": "done", "data": {}, "contact": POINT}}, "$.handed_back.contact"),
    ("script_error", {"script_error": {"type": "ValueError", "message": json.dumps({"contact": POINT})}},
     "$.script_error.message<json>.contact"),
])
def test_the_rest_of_a_scripts_result_is_audited(tmp_path: Path, field: str, rest: dict, path: str) -> None:
    """The calls the harness logged, any key of a handback besides its reason and data, and the error envelope."""

    errors = verify_agent_trace(_script_result(tmp_path / field, {"ok": True, **rest}))["errors"]
    assert errors == [f"event:5:privileged_artifact_field:{path}"]


def test_a_tool_calls_result_of_the_same_shape_is_audited(tmp_path: Path) -> None:
    """Tool-call mode: a plan without code mode has no script, whatever a tool is called."""

    errors = verify_agent_trace(_script_result(tmp_path / "tool-call", AUTHORED["value"][1], name="run_python",
                                               code_mode=False))["errors"]
    assert errors == ["event:3:privileged_artifact_field:$.value.contact"]


def test_another_tools_result_in_code_mode_is_audited(tmp_path: Path) -> None:
    """In code mode, the result of a tool that runs no script keeps the audit whatever its fields are called."""

    errors = verify_agent_trace(_script_result(tmp_path / "other", AUTHORED["value"][1], name="check_task"))["errors"]
    assert errors == ["event:3:privileged_artifact_field:$.value.contact"]


def _tool_message(call: str, result: dict) -> dict:
    return {"role": "tool", "name": "bash", "tool_call_id": call, "content": json.dumps(result)}


def _tool_result_block(call: str, result: dict) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call, "content": json.dumps(result)}]}


def _packet(call: str, result: dict) -> dict:
    head = json.dumps({"name": "bash", "role": "tool", "tool_call_id": call}, separators=(",", ":"))
    return {"role": "user", "content": [{"type": "text", "text": head}, {"type": "text", "text": json.dumps(result)}]}


#: How each kind of request holds a tool's result, and where the audit finds a field of it.
REQUESTS = {
    "canonical": (_tool_message, "native_canonical_provider_request", "content<json>"),
    "openai_wire": (_tool_message, "native_provider_wire_request", "content<json>"),
    "anthropic_wire": (_tool_result_block, "native_provider_wire_request", "content[0].content<json>"),
    "compactor_canonical": (_packet, "rollout_compactor_canonical_request", "content[1].text<json>"),
    "compactor_wire": (_packet, "rollout_compactor_wire_request", "content[1].text<json>"),
}
#: Shaped as run ora-glue-code-libero-pro-a29619033c3f--20261101T183916Z, refused sealing on 2026-11-01: a receipt the
#: script handed back under the name ``rc``, so no longer under the receipt's own path.
HANDED_BACK_RECEIPT = {"ok": True, "status": "completed", "calls": [{"call": 1, "tool": "mark_point", "ok": True}],
                       "handed_back": {"reason": "done", "data": {"rc": {"detail": {"contact_motion": {
                           "contact_estimated": True, "force_stop_threshold_n": 40.0}}}}}}


@pytest.mark.parametrize("shape", sorted(REQUESTS))
@pytest.mark.parametrize("authored", [HANDED_BACK_RECEIPT, *(field for _, field in AUTHORED.values())])
def test_a_request_repeats_a_scripts_result_under_the_same_audit(tmp_path: Path, shape: str, authored: dict) -> None:
    """The model reads its script's result in every later request and the compactor reads it in a packet. Each
    repeats what the result already passed, so neither invalidates the run."""

    message, kind, _ = REQUESTS[shape]
    verification = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, request={"messages": [message(SCRIPT, authored)]}, request_kind=kind))
    assert verification["status"] == "valid", verification["errors"]


#: The context packets the compactor's request repeats, which hold their messages under ``compactor_messages``.
PACKETS = ("lossless_context_semantic_packet", "episode_context_capsule_packet")
#: Shaped as run ora-glue-code-libero-pro-3d661e691237--20261104T011110Z, refused sealing on 2026-11-04 at
#: ``$.compactor_messages[0].content[108].text<json>.handed_back.data.p_contact``: the emergency checkpoint's packet
#: repeated a script's result whose handed-back data named a point ``p_contact``.
HANDED_BACK_POINT = {"ok": True, "status": "handed_back", "calls": [{"call": 1, "tool": "vision_ground", "ok": True}],
                     "handed_back": {"reason": "descend", "data": {"p_contact": POINT}}}


@pytest.mark.parametrize("kind", PACKETS)
def test_a_context_packet_repeats_a_scripts_result_under_the_same_audit(tmp_path: Path, kind: str) -> None:
    packet = {"compactor_messages": [_packet(SCRIPT, HANDED_BACK_POINT)]}
    verification = verify_agent_trace(_script_result(tmp_path / kind, {"ok": True}, request=packet,
                                                     request_kind=kind))
    assert verification["status"] == "valid", verification["errors"]


@pytest.mark.parametrize("kind", PACKETS)
@pytest.mark.parametrize(("call", "result", "where"), [
    (SCRIPT, {**HANDED_BACK_POINT, "calls": [{"call": 1, "tool": "vision_ground", "receipt": {"object_gt": POINT}}]},
     "calls[0].receipt.object_gt"),
    ("call-observe", {"value": {"p_contact": POINT}}, "value.p_contact"),
])
def test_the_rest_of_a_context_packet_is_audited(tmp_path: Path, kind: str, call: str, result: dict,
                                                 where: str) -> None:
    """The harness's part of a script's result, and another tool's result under the names a script's leaves out."""

    errors = verify_agent_trace(_script_result(tmp_path / "run", {"ok": True}, request={
        "compactor_messages": [_packet(call, result)]}, request_kind=kind))["errors"]
    assert errors == [f"event:6:privileged_artifact_field:$.compactor_messages[0].content[1].text<json>.{where}"]


PRIVATE_KEYS = ("object_gt", "segmentation_mask", "contact_force_n", "task_source", "success_predicate",
                "expert_program", "actor_state")


@pytest.mark.parametrize("shape", sorted(REQUESTS))
@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_a_private_key_in_the_rest_of_a_repeated_result_is_caught(tmp_path: Path, shape: str, key: str) -> None:
    """A receipt among the calls the harness logged, printed output beside it or not."""

    message, kind, where = REQUESTS[shape]
    result = {"ok": True, "output": "done", "calls": [{"call": 1, "tool": "mark_point", "receipt": {key: POINT}}]}
    errors = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, request={"messages": [message(SCRIPT, result)]},
        request_kind=kind))["errors"]
    assert errors == [f"event:6:privileged_artifact_field:$.messages[0].{where}.calls[0].receipt.{key}"]


@pytest.mark.parametrize("shape", sorted(REQUESTS))
def test_a_key_beside_a_handbacks_data_in_a_request_is_caught(tmp_path: Path, shape: str) -> None:
    message, kind, where = REQUESTS[shape]
    result = {"ok": True, "handed_back": {"reason": "done", "data": {}, "contact": POINT}}
    errors = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, request={"messages": [message(SCRIPT, result)]},
        request_kind=kind))["errors"]
    assert errors == [f"event:6:privileged_artifact_field:$.messages[0].{where}.handed_back.contact"]


@pytest.mark.parametrize("shape", sorted(REQUESTS))
@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_another_tools_result_in_a_request_is_audited(tmp_path: Path, shape: str, key: str) -> None:
    """Under the names a script's result leaves out too: the exemption goes by the call, not by the field."""

    message, kind, where = REQUESTS[shape]
    errors = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, request={"messages": [message("call-observe", {"value": {key: POINT}})]},
        request_kind=kind))["errors"]
    assert errors == [f"event:6:privileged_artifact_field:$.messages[0].{where}.value.{key}"]


@pytest.mark.parametrize("shape", sorted(REQUESTS))
def test_a_request_in_tool_call_mode_is_audited(tmp_path: Path, shape: str) -> None:
    """A plan without code mode has no script, whatever a tool is called."""

    message, kind, where = REQUESTS[shape]
    errors = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, code_mode=False,
        request={"messages": [message(SCRIPT, {"value": {"object_gt": POINT}})]}, request_kind=kind))["errors"]
    assert errors == [f"event:4:privileged_artifact_field:$.messages[0].{where}.value.object_gt"]


def test_a_packet_block_after_a_scripts_result_is_audited(tmp_path: Path) -> None:
    """Only the text a script result's heading names leaves the audit, not what the packet holds after it."""

    packet = _packet(SCRIPT, {"value": {"contact": POINT}})
    packet["content"].append({"type": "text", "text": json.dumps({"value": {"object_gt": POINT}})})
    errors = verify_agent_trace(_script_result(
        tmp_path / "run", {"ok": True}, request={"messages": [packet]},
        request_kind="rollout_compactor_wire_request"))["errors"]
    assert errors == ["event:6:privileged_artifact_field:$.messages[0].content[2].text<json>.value.object_gt"]


def test_what_a_request_says_outside_a_tools_result_is_audited(tmp_path: Path) -> None:
    request = {"messages": [_tool_message(SCRIPT, {"value": {"contact": POINT}}),
                            {"role": "user", "content": json.dumps({"object_gt": POINT})}]}
    errors = verify_agent_trace(_script_result(tmp_path / "run", {"ok": True}, request=request))["errors"]
    assert errors == ["event:6:privileged_artifact_field:$.messages[1].content<json>.object_gt"]
