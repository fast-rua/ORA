"""Argument names a run's own tool surface declares pass the privileged-name audit only where they hold the
policy's own argument (I-0011): its tool call, the echoes of that call's value or the declared default, and the
request's tool schema. The same key with another value, in a tool result or anywhere else, is still forbidden
evidence. A provider request reads a code-mode script's result as the audit reads that result.

The arguments the model wrote in a tool call are its own text under names it chose, so the audit leaves them out
where the model's provenance is exact: its reply, the proposal that parsed them, a call its script made, and every
request or packet that repeats the call with the same id and the same arguments. Echoes in a tool's result, calls
the context rewrote and calls the model never wrote stay audited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.trace_privacy import (
    _forbidden_path, declared_argument_names, model_tool_arguments, policy_argument_values,
    proposal_without_model_arguments, script_call_ids, without_script_authored_messages,
)
from ora.agent.evidence.trace_verify import verify_agent_trace
from tests.support.agent.agent_events import TrajectoryRecorder

SURFACE = {"definitions": [
    {"name": "slide_block_paced", "inputSchema": {"type": "object", "properties": {
        "arm": {"type": "string"}, "contact_speed_m_s": {"type": "number", "default": 0.05}}}},
    {"name": "compute", "inputSchema": {"type": "object"}},
]}
#: A surface declaring arguments named exactly a forbidden term.
PROBE = {"definitions": [{"name": "probe", "inputSchema": {"type": "object", "properties": {
    "contact": {"type": "boolean"}, "segmentation": {"type": "string"}}}}]}
ARGUMENTS = {"arm": "left", "contact_speed_m_s": 0.03}


def test_declared_argument_names_are_those_holding_a_forbidden_term_but_not_one():
    assert declared_argument_names(SURFACE) == {"contact_speed_m_s"}
    assert declared_argument_names(PROBE) == frozenset()
    assert declared_argument_names({}) == frozenset()


def test_a_declared_argument_passes_only_with_a_value_the_policy_wrote():
    names = declared_argument_names(SURFACE)
    values = policy_argument_values(SURFACE, [
        {"type": "tool_call_proposed", "payload": {"name": "slide_block_paced", "arguments": ARGUMENTS}},
        {"type": "tool_call_proposed", "payload": {"name": "push", "arguments": {"steps": [{"contact_speed_m_s": 2}]}}},
        {"type": "tool_result", "payload": {"contact_speed_m_s": 0.5}},
    ])
    assert values == {"contact_speed_m_s": {"0.03", "2.0", "0.05"}}
    probe_call = {"type": "tool_call_proposed", "payload": {"name": "probe", "arguments": {"contact": True}}}
    assert policy_argument_values(PROBE, [probe_call]) == {}
    assert _forbidden_path({"arguments": ARGUMENTS}) == "$.arguments.contact_speed_m_s"
    assert _forbidden_path({"arguments": ARGUMENTS}, argument_values=values) is None
    assert _forbidden_path({"echo": {"contact_speed_m_s": 2.0}}, argument_values=values) is None
    assert _forbidden_path({"contact_speed_m_s": 0.5}, argument_values=values) == "$.contact_speed_m_s"
    # Values under the key are still scanned, and a declaration counts only where a request declares its tools.
    leak = {"arguments": {"contact_speed_m_s": {"contact_force": 3.0}}}
    nested = policy_argument_values(SURFACE, [{"type": "tool_call_proposed", "payload": leak}])
    assert _forbidden_path(leak, argument_values=nested) == "$.arguments.contact_speed_m_s.contact_force"
    declared = {"tools": [{"function": {"parameters": {"properties": {"contact_speed_m_s": {"type": "number"}}}}}]}
    assert _forbidden_path(declared, argument_declarations=names) is None
    assert _forbidden_path({"result": declared}, argument_declarations=names) == (
        "$.result.tools[0].function.parameters.properties.contact_speed_m_s")


def _openai_request(surface: dict, messages: list) -> dict:
    return {"messages": messages, "tools": [{"type": "function", "function": {
        "name": d["name"], "parameters": d["inputSchema"]}} for d in surface["definitions"]]}


def _anthropic_request(surface: dict) -> dict:
    return {"messages": [{"role": "user", "content": "slide the block"}],
            "tools": [{"name": d["name"], "input_schema": d["inputSchema"]} for d in surface["definitions"]]}


def _tool_call_run(root: Path, result: dict, *, arguments: dict = ARGUMENTS, surface: dict = SURFACE) -> Path:
    """Tool-call mode: the model calls ``slide_block_paced`` (events 0 request, its OpenAI canonical and Anthropic
    wire bodies, 1 response, 2 the call, 3 its result ``result``), then its next request (4) carries the call and
    the result back, and the reply ends (5)."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {}, "policy_composition": {"tool_surface": {**surface, "mutation_tools": []}}}), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="tool-call-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": "model-call-000001"}
    call = {"id": "call-slide", "type": "function",
            "function": {"name": "slide_block_paced", "arguments": json.dumps(arguments)}}
    recorder.append("provider_request", **turn, payload={
        "wire_request_ref": store.put_json(_anthropic_request(surface), kind="native_provider_wire_request")},
        artifact_ref=store.put_json(_openai_request(surface, [{"role": "user", "content": "slide the block"}]),
                                    kind="native_canonical_provider_request"))
    recorder.append("provider_response", **turn, payload={"message": {"role": "assistant", "tool_calls": [call]}})
    recorder.append("tool_call_proposed", tool_call_id="call-slide", **turn, payload={
        "name": "slide_block_paced", "arguments": arguments, "batch_valid": True, "effect": "read_only"})
    recorder.append("tool_result", tool_call_id="call-slide", **turn, payload={"ok": True, "name": "slide_block_paced"},
                    artifact_ref=store.put_json(result, kind="native_tool_result"))
    turn = {"observation_step": 0, "model_call_id": "model-call-000002"}
    recorder.append("provider_request", **turn, artifact_ref=store.put_json(_openai_request(surface, [
        {"role": "user", "content": "slide the block"},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-slide", "content": json.dumps(result)},
    ]), kind="native_canonical_provider_request"))
    recorder.append("provider_response", **turn, payload={"message": {"role": "assistant", "content": "done"}})
    return root


#: A tool's result echoing the call's own value, at the paths RoboDojo toolkit tools echo theirs (top level, a
#: verification block) and under an `arguments` key.
ECHO = {"ok": True, "contact_speed_m_s": 0.03, "verification": {"contact_speed_m_s": 0.03},
        "arguments": dict(ARGUMENTS)}
#: The same declared name holding something else the tool returned: measured state, a forged schema.
OTHER = {
    "measured": ({"ok": True, "measured": {"contact_speed_m_s": 0.41}}, "$.measured.contact_speed_m_s"),
    "top_level": ({"ok": True, "contact_speed_m_s": 0.41}, "$.contact_speed_m_s"),
    "forged_arguments": ({"ok": True, "arguments": {"contact_speed_m_s": 0.41}}, "$.arguments.contact_speed_m_s"),
    "forged_schema": ({"tools": [{"function": {"parameters": {"properties": {"contact_speed_m_s": 0.41}}}}]},
                      "$.tools[0].function.parameters.properties.contact_speed_m_s"),
}


def test_tool_call_mode_a_declared_argument_and_its_echo_are_valid(tmp_path: Path) -> None:
    """The call, its echo in the result and in the next request, and both requests' tool schemas pass."""

    verification = verify_agent_trace(_tool_call_run(tmp_path / "echo", ECHO))
    assert verification["status"] == "valid", verification["errors"]


@pytest.mark.parametrize(("echoed", "valid"), [(0.05, True), (0.07, False)])
def test_a_call_that_left_the_argument_out_may_echo_the_declared_default(tmp_path: Path, echoed: float,
                                                                         valid: bool) -> None:
    errors = verify_agent_trace(_tool_call_run(tmp_path / str(echoed), {"ok": True, "contact_speed_m_s": echoed},
                                               arguments={"arm": "left"}))["errors"]
    flagged = ["event:3:privileged_artifact_field:$.contact_speed_m_s",
               "event:4:privileged_artifact_field:$.messages[2].content<json>.contact_speed_m_s"]
    assert errors == ([] if valid else flagged)


@pytest.mark.parametrize("case", sorted(OTHER))
def test_tool_call_mode_the_declared_name_holding_another_value_is_flagged(tmp_path: Path, case: str) -> None:
    result, path = OTHER[case]
    errors = verify_agent_trace(_tool_call_run(tmp_path / case, result))["errors"]
    assert errors == [f"event:3:privileged_artifact_field:{path}",
                      f"event:4:privileged_artifact_field:$.messages[2].content<json>{path[1:]}"]


def test_a_declared_argument_that_is_exactly_a_forbidden_term_is_not_exempt(tmp_path: Path) -> None:
    """`contact` names contact sensing whoever declares it: the harness's schema and the tool's echo, in its result
    and in the next request, are flagged. The model's call, in its reply, its proposal and the next request, is the
    model's own text."""

    errors = verify_agent_trace(_tool_call_run(tmp_path / "contact", {"ok": True, "contact": True},
                                               arguments={"contact": True}, surface=PROBE))["errors"]
    assert errors == [
        "event:0:privileged_artifact_field:$.tools[0].function.parameters.properties.contact",
        "event:0:privileged_artifact_field:$.tools[0].input_schema.properties.contact",
        "event:3:privileged_artifact_field:$.contact",
        "event:4:privileged_artifact_field:$.messages[2].content<json>.contact",
    ]


def _code_mode_run(root: Path, result: dict) -> Path:
    """Code mode: the model's run_python script calls ``slide_block_paced`` (events 0 request, 1 response, 2 the
    script, 3 its call, 4 that call's result ``result``, 5 the script's result)."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {"code_mode": {"contract": "ora.code-mode-script.v1"}},
        "policy_composition": {"tool_surface": {**SURFACE, "mutation_tools": []}}}), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="code-mode-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": "model-call-000001"}
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    recorder.append("tool_call_proposed", tool_call_id="call-script", **turn, payload={
        "name": "run_python", "arguments": {"code": "tools.slide_block_paced(arm='left', contact_speed_m_s=0.03)"},
        "batch_valid": True, "effect": "mutation"})
    origin = {"kind": "code_mode", "parent_tool_call_id": "call-script", "seq": 1}
    recorder.append("tool_call_proposed", tool_call_id="call-script#001", **turn, payload={
        "name": "slide_block_paced", "arguments": ARGUMENTS, "batch_valid": True, "effect": "read_only",
        "origin": origin})
    recorder.append("tool_result", tool_call_id="call-script#001", **turn, payload={"ok": True, "origin": origin},
                    artifact_ref=store.put_json(result, kind="native_tool_result"))
    recorder.append("tool_result", tool_call_id="call-script", **turn, payload={"ok": True, "name": "run_python"},
                    artifact_ref=store.put_json({"ok": True, "status": "completed", "calls": [
                        {"call": 1, "tool": "slide_block_paced", "ok": True}]}, kind="native_tool_result"))
    return root


def test_code_mode_a_scripts_declared_argument_and_its_echo_are_valid(tmp_path: Path) -> None:
    verification = verify_agent_trace(_code_mode_run(tmp_path / "echo", ECHO))
    assert verification["status"] == "valid", verification["errors"]


@pytest.mark.parametrize("case", sorted(OTHER))
def test_code_mode_the_declared_name_holding_another_value_is_flagged(tmp_path: Path, case: str) -> None:
    result, path = OTHER[case]
    errors = verify_agent_trace(_code_mode_run(tmp_path / case, result))["errors"]
    assert errors == [f"event:4:privileged_artifact_field:{path}"]


def test_only_a_scripts_result_loses_what_the_script_wrote() -> None:
    authored = json.dumps({"ok": True, "stdout": "x", "handed_back": {"reason": "r", "data": {}, "note": "n"}})
    request = {"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "script", "content": authored},
        {"role": "tool", "tool_call_id": "script", "content": "not json"},
        {"role": "tool", "tool_call_id": "script", "content": "[1, 2]"},
        {"role": "tool", "tool_call_id": ["script"], "content": authored},
        {"role": "tool", "tool_call_id": "other", "content": authored},
    ]}
    kept = without_script_authored_messages(request, {"script"})
    assert json.loads(kept["messages"][0]["content"]) == {"ok": True, "handed_back": {"note": "n"}}
    assert kept["messages"][1:] == request["messages"][1:] and kept["model"] == "m"
    assert without_script_authored_messages(request, frozenset()) is request
    assert without_script_authored_messages("text", {"script"}) == "text"


def test_the_model_called_the_scripts_of_a_code_mode_run() -> None:
    events = [
        {"type": "tool_call_proposed", "tool_call_id": "a", "payload": {"name": "bash"}},
        {"type": "tool_call_proposed", "tool_call_id": "b", "payload": {"name": "run_python"}},
        {"type": "tool_call_proposed", "tool_call_id": "a#001",
         "payload": {"name": "bash", "origin": {"kind": "code_mode", "parent_tool_call_id": "a", "seq": 1}}},
        {"type": "tool_call_proposed", "tool_call_id": "c", "payload": {"name": "check_task"}},
        {"type": "tool_result", "tool_call_id": "d", "payload": {"name": "bash"}},
    ]
    assert script_call_ids({"policy": {"code_mode": {"contract": "c"}}}, events) == {"a", "b"}
    assert script_call_ids({"policy": {"glue_code": {"contract": "c"}}}, events) == {"a", "b"}
    assert script_call_ids({"policy": {}}, events) == frozenset()


#: The `write` call of run ora-glue-code-robodojo-cb50f80ce61a--20261104T011111Z, refused sealing on 2026-11-04: the
#: model put a whole module into one malformed key (the value is empty), and a docstring in it says "pad contact".
MALFORMED_KEY = ('arguments=\n{"content": "\\"\\"\\"Grasp / carry / place primitives for the flat digit pieces.\\n'
                 "Roll: the jaw-closing line is laid along the piece's long axis, which puts the\\npad contact on "
                 'the two ends of the piece.\\n\\"\\"\\"\\nimport robot\\n", "path": "carry.py"}')
#: What a model may write as a call's arguments: the malformed key, a key exactly a forbidden term, and nested ones.
MODEL_ARGUMENTS = {
    "malformed_key": {MALFORMED_KEY: ""},
    "contact": {"contact": True},
    "nested": {"path": "probe.py", "segmentation": {"object_gt": [0.1, 0.2], "contact_points": []}},
}
#: The harness's answer to the malformed call (the run's `write` refusal).
REFUSAL = {"error": {"code": "invalid_arguments", "message": "write takes path and content", "retryable": True}}


def _call(call_id: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": "write", "arguments": json.dumps(arguments)}}


def _model_call_run(root: Path, arguments: dict, *, proposed: dict | None = None, repeated: dict | None = None,
                    result: dict = REFUSAL, code_mode: bool = False) -> Path:
    """The model's ``write`` call with ``arguments`` (events 0 request, 1 its reply, 2 the proposal of ``proposed``,
    default what the reply's arguments parse to, 3 its result ``result``), then the next request (4) and its reply
    (5). The next request repeats ``repeated`` (default the reply's call) in its OpenAI canonical body, its Anthropic
    wire body (a ``tool_use`` block) and the emergency checkpoint's packet (the assistant message as a text block)."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {"code_mode": {"contract": "ora.code-mode-script.v1"}} if code_mode else {},
        "policy_composition": {"tool_surface": {"mutation_tools": []}}}), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="model-call-run", policy_session_id="pi-session")
    call = _call("call-write", arguments)
    repeated = repeated or call
    turn = {"observation_step": 0, "model_call_id": "model-call-000001"}
    task = {"role": "user", "content": "carry the piece"}
    recorder.append("provider_request", **turn, artifact_ref=store.put_json(
        {"messages": [task]}, kind="native_canonical_provider_request"))
    recorder.append("provider_response", **turn, payload={"message": {"role": "assistant", "content": "",
                                                                      "tool_calls": [call]}})
    recorder.append("tool_call_proposed", tool_call_id="call-write", **turn, payload={
        "name": "write", "arguments": arguments if proposed is None else proposed, "batch_valid": True,
        "effect": "read_only"})
    recorder.append("tool_result", tool_call_id="call-write", **turn, payload={"ok": False, "name": "write"},
                    artifact_ref=store.put_json(result, kind="native_tool_result"))
    assistant = {"role": "assistant", "content": "", "tool_calls": [repeated]}
    tool_use = {"type": "tool_use", "id": repeated["id"], "name": "write",
                "input": json.loads(repeated["function"]["arguments"])}
    turn = {"observation_step": 0, "model_call_id": "model-call-000002"}
    recorder.append("provider_request", **turn, payload={
        "packet_ref": store.put_json({"compactor_messages": [{"role": "user", "content": [
            {"type": "text", "text": json.dumps(assistant, separators=(",", ":"))}]}]},
            kind="lossless_context_semantic_packet"),
        "wire_request_ref": store.put_json({"messages": [task, {"role": "assistant", "content": [tool_use]}, {
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-write",
                                         "content": json.dumps(result)}]}]}, kind="native_provider_wire_request"),
    }, artifact_ref=store.put_json({"messages": [task, assistant, {
        "role": "tool", "name": "write", "tool_call_id": "call-write", "content": json.dumps(result)}]},
        kind="native_canonical_provider_request"))
    recorder.append("provider_response", **turn, payload={"message": {"role": "assistant", "content": "done"}})
    return root


@pytest.mark.parametrize("code_mode", [False, True])
@pytest.mark.parametrize("case", sorted(MODEL_ARGUMENTS))
def test_the_arguments_the_model_wrote_are_not_privileged_by_their_names(tmp_path: Path, case: str,
                                                                        code_mode: bool) -> None:
    """Its reply, the proposal and every repeat of the call hold the model's own text, whatever its keys say."""

    verification = verify_agent_trace(_model_call_run(tmp_path / case, MODEL_ARGUMENTS[case], code_mode=code_mode))
    assert verification["status"] == "valid", verification["errors"]


def test_a_tools_echo_of_the_models_argument_is_audited(tmp_path: Path) -> None:
    """The harness wrote the result, wherever a request repeats it."""

    errors = verify_agent_trace(_model_call_run(tmp_path / "echo", {"contact": True}, result={
        "error": {"code": "invalid_arguments", "received": {"contact": True}}}))["errors"]
    assert errors == [
        "event:3:privileged_artifact_field:$.error.received.contact",
        "event:4:privileged_artifact_field:$.messages[2].content<json>.error.received.contact",
        "event:4:privileged_artifact_field:$.messages[2].content[0].content<json>.error.received.contact",
    ]


@pytest.mark.parametrize("repeated", [
    _call("call-write", {"contact": True, "contact_force_n": 3.1}),  # the same call, its arguments rewritten
    _call("call-other", {"contact": True}),  # a call the model never wrote
])
def test_a_call_the_model_did_not_write_is_audited(tmp_path: Path, repeated: dict) -> None:
    errors = verify_agent_trace(_model_call_run(tmp_path / "run", {"contact": True}, repeated=repeated))["errors"]
    assert errors == [
        "event:4:privileged_artifact_field:$.messages[1].tool_calls[0].function.arguments<json>.contact",
        "event:4:privileged_artifact_field:$.compactor_messages[0].content[0].text<json>.tool_calls[0].function"
        ".arguments<json>.contact",
        "event:4:privileged_artifact_field:$.messages[1].content[0].input.contact",
    ]


def test_a_proposal_that_differs_from_the_reply_is_audited(tmp_path: Path) -> None:
    errors = verify_agent_trace(_model_call_run(tmp_path / "run", {"contact": True},
                                                proposed={"contact": True, "contact_force_n": 3.1}))["errors"]
    assert errors == ["event:2:privileged_field:$.arguments.contact"]


def test_the_model_wrote_the_tool_calls_of_its_replies() -> None:
    events = [
        {"type": "provider_response", "payload": {"message": {"tool_calls": [
            _call("a", {"x": 1}), {"id": "b", "function": {"arguments": {"x": 1}}}, {"function": {"arguments": "{}"}},
            "not a call"]}}},
        {"type": "provider_request", "payload": {"message": {"tool_calls": [_call("c", {})]}}},
        {"type": "tool_call_proposed", "tool_call_id": "d", "payload": {"name": "write", "arguments": {}}},
    ]
    assert model_tool_arguments(events) == {"a": '{"x": 1}'}
    written = {"a": '{"contact": 1}'}
    assert proposal_without_model_arguments({"name": "w", "arguments": {"contact": 1}}, "a", written, ()) == {
        "name": "w"}
    for payload, call_id in (({"arguments": {"contact": 2}}, "a"), ({"arguments": {"contact": 1}}, "b")):
        assert proposal_without_model_arguments(payload, call_id, written, ()) is payload
    nested = {"arguments": {"contact": 1}, "origin": {"kind": "code_mode", "parent_tool_call_id": "script"}}
    assert proposal_without_model_arguments(nested, "script#001", {}, {"script"}) == {"origin": nested["origin"]}
    for scripts in ((), {"other"}):
        assert proposal_without_model_arguments(nested, "script#001", {}, scripts) is nested


def _script_call_run(root: Path, arguments: dict, *, code_mode: bool = True) -> Path:
    """The model's ``bash`` script called ``mark_point`` with ``arguments`` (events 0 request, 1 response, 2 the
    script, 3 its call, 4 that call's result, 5 the script's result)."""

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "policy": {"code_mode": {"contract": "ora.code-mode-script.v1"}} if code_mode else {},
        "policy_composition": {"tool_surface": {"mutation_tools": []}}}), encoding="utf-8")
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id="code-mode-run", policy_session_id="pi-session")
    turn = {"observation_step": 0, "model_call_id": "model-call-000001"}
    recorder.append("provider_request", **turn)
    recorder.append("provider_response", **turn)
    recorder.append("tool_call_proposed", tool_call_id="call-script", **turn, payload={
        "name": "bash", "arguments": {"command": "python mark.py"}, "batch_valid": True, "effect": "read_only"})
    origin = {"kind": "code_mode", "parent_tool_call_id": "call-script", "seq": 1}
    recorder.append("tool_call_proposed", tool_call_id="call-script#001", **turn, payload={
        "name": "mark_point", "arguments": arguments, "batch_valid": True, "effect": "read_only", "origin": origin})
    recorder.append("tool_result", tool_call_id="call-script#001", **turn, payload={"ok": True, "origin": origin},
                    artifact_ref=store.put_json({"ok": True, "marks": []}, kind="native_tool_result"))
    recorder.append("tool_result", tool_call_id="call-script", **turn, payload={"ok": True, "name": "bash"},
                    artifact_ref=store.put_json({"ok": True, "calls": [{"call": 1, "tool": "mark_point", "ok": True}]},
                                                kind="native_tool_result"))
    return root


def test_the_arguments_a_scripts_call_sent_are_the_models(tmp_path: Path) -> None:
    """A script the model wrote names its calls' arguments; outside code mode the same origin is no script's."""

    arguments = {"view": "wrist", "contact_point": [0.1, 0.2], "p_contact": {"u": 3, "v": 4}}
    verification = verify_agent_trace(_script_call_run(tmp_path / "code-mode", arguments))
    assert verification["status"] == "valid", verification["errors"]
    errors = verify_agent_trace(_script_call_run(tmp_path / "tool-call", arguments, code_mode=False))["errors"]
    assert errors == ["event:3:privileged_field:$.arguments.contact_point"]


def test_a_grasp_plan_pose_with_its_world_axes_is_public_only_as_the_exact_pair():
    """RO-27: a grasp_pose_world_axes_v1 pose adds approach_world and jaw_line_world, derived from its own quaternion;
    the contact_pose exemption takes them together, as three finite numbers each, and nothing else."""
    pose = {"position_m": [0.1, 0.2, 0.9], "orientation_xyzw": [0.0, 1.0, 0.0, 0.0]}
    axes = {"approach_world": [0.0, 0.0, -1.0], "jaw_line_world": [-1.0, 0.0, 0.0]}
    assert _forbidden_path({"candidates": [{"contact_pose": {**pose, **axes}}]},
                           allow_public_grasp_contact_pose=True) is None
    assert _forbidden_path({"contact_pose": {**pose, **axes}}) == "$.contact_pose"
    for damaged in ({**pose, "approach_world": axes["approach_world"]}, {**pose, **axes, "jaw_line_world": [1.0, 0.0]},
                    {**pose, **axes, "approach_world": [0.0, float("nan"), 1.0]}, {**pose, **axes, "contact_force": 1.0}):
        assert _forbidden_path({"contact_pose": damaged}, allow_public_grasp_contact_pose=True) == "$.contact_pose"
