"""The wire export: every call as sent, each image once, epochs, and the replay check that rebuilds them."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
from pathlib import Path

import pytest
from PIL import Image

from ora.agent.evidence import wire_export
from ora.agent.evidence.canonical import ArtifactStore, TrajectoryError
from ora.agent.evidence.host_paths import HostPaths
from ora.agent.evidence.wire_export import check_wire_export, export_wire
from tests.support.agent.agent_events import TrajectoryRecorder


def _png(shade: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (shade, shade, shade)).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


A, B, C, D = (_png(shade) for shade in (10, 20, 30, 40))


def _image(data: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _user(*images: str, text: str = "observation") -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, *(_image(item) for item in images)]}


def _assistant(name: str) -> dict:
    return {"role": "assistant", "content": "", "reasoning": f"thinking about {name}",
            "tool_calls": [{"id": f"call-{name}", "type": "function",
                            "function": {"name": name, "arguments": "{}"}}]}


def _tool(name: str) -> dict:
    return {"role": "tool", "tool_call_id": f"call-{name}", "content": f"{name} done"}


TRUNCATED = [  # a streamed reply cut off by the token limit inside a tool call
    {"choices": [{"delta": {"reasoning_content": "long "}, "finish_reason": None}]},
    {"choices": [{"delta": {"reasoning_content": "thought"}, "finish_reason": None}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-x", "type": "function",
                                            "function": {"name": "move", "arguments": "{\"pos"}}]}}]},
    {"choices": [{"delta": {}, "finish_reason": "length"}]},
    {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 16384}},
]


def _run(root: Path, calls: list[dict], *, compactions: int = 0, dangling: bool = False,
         evaluation: dict | None = None, sealed: bool = False, policy: dict | None = None) -> Path:
    """A run whose model calls send ``calls[i]['messages']`` (OpenAI wire unless ``anthropic`` is given), each
    answered by ``calls[i]['stream']`` when it is given."""

    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, "run", "session")
    for index in range(compactions):
        request = store.put_json({"model": "c", "messages": [{"role": "user", "content": [_image(A)]}]},
                                 kind="rollout_compactor_wire_request")
        recorder.append("context_compactor_attempt", model_call_id=f"context-compactor-{index:06d}",
                        artifact_ref=request, payload={"call_kind": "rollout_compaction", "model": "c",
                                                       "provider_protocol": "openai_chat_completions"})
        reply = store.put_json([{"choices": [{"delta": {"content": "summary"}, "finish_reason": "stop"}]}],
                               kind="rollout_compactor_wire_response")
        recorder.append("context_compactor_response", model_call_id=f"context-compactor-{index:06d}",
                        artifact_ref=reply, payload={"call_kind": "rollout_compaction", "finish_reason": "stop"})
    for index, call in enumerate(calls):
        call_id = f"model-call-{index:06d}"
        wire_body = call.get("anthropic") or {"model": "m", "temperature": 1.0, "stream": True,
                                              "messages": call["messages"], "tools": []}
        canonical = store.put_json({"model": "m", "messages": call["messages"], "tools": []},
                                   kind="native_canonical_provider_request")
        wire = store.put_json(wire_body, kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=call_id, artifact_ref=wire, payload={
            "model": "m", "canonical_request_ref": canonical,
            "provider_protocol": "anthropic_messages" if "anthropic" in call else "openai_chat_completions"})
        rejected = call.get("accepted") is False
        stream = store.put_json(call.get("stream") or (TRUNCATED if rejected else [
            {"choices": [{"delta": {"content": f"reply {index}"}}]}]), kind="native_provider_wire_response")
        payload = {"accepted": not rejected, "finish_reason": "invalid_tool_input" if rejected else "tool_calls",
                   "usage": {"input_tokens": 100 + index, "output_tokens": 5},
                   "message": {"role": "assistant", "content": "", "reasoning_content": f"r{index}",
                               "_provider_continuation_state": {"x": 1}}}
        if rejected:
            payload["output_rejection"] = {"code": "truncated_tool_call", "provider_stop_reason": "length"}
        recorder.append("provider_response", model_call_id=call_id, artifact_ref=stream, payload=payload)
    if dangling:
        request = store.put_json({"model": "m", "messages": [_user(D)]}, kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=f"model-call-{len(calls):06d}", artifact_ref=request,
                        payload={"model": "m", "provider_protocol": "openai_chat_completions"})
    (root / "plan.json").write_text(json.dumps({"source": {"commit": "test"}, "policy": policy or {}}))
    (root / "result.json").write_text(json.dumps({"run_id": "run", "official_success": True, "termination": "test"}))
    if evaluation is not None:
        (root / "evaluation.json").write_text(json.dumps(evaluation))
    if sealed:
        (root / "seal.json").write_text(json.dumps({"inventory": []}))
    return root


ELIGIBLE = {"status": "valid", "learning_eligibility": {"eligible": True, "reasons": []}}


def _rehash(destination: Path, name: str) -> None:
    """Edit-and-cover: give ``name``'s new bytes their sha256 in the manifest."""
    manifest = json.loads((destination / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == name:
            data = (destination / name).read_bytes()
            entry["sha256"], entry["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
    (destination / "manifest.json").write_text(json.dumps(manifest))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _calls() -> list[dict]:
    first = [{"role": "system", "content": "s"}, _user(A)]
    second = first + [_assistant("observe"), _tool("observe"), _user(B, A)]
    rewritten = [{"role": "system", "content": "s"}, _user(text="observation 0 is a pointer now"),
                 _assistant("observe"), _tool("observe"), _user(B)]
    tailed = rewritten + [_assistant("move"), _tool("move"), _user(C, text="camera tail")]
    rolled = tailed[:-1] + [_assistant("move2"), _tool("move2"), _user(D, text="camera tail")]
    return [{"messages": first}, {"messages": second}, {"messages": rewritten, "accepted": False},
            {"messages": tailed}, {"messages": rolled}]


def test_every_call_exports_as_sent_and_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()), tmp_path / "wire")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert (manifest["calls"], manifest["accepted_calls"], manifest["epochs"]) == (5, 4, 2)
    assert manifest["images"]["files"] == 4  # A appears three times and is stored once
    assert manifest["replay_check"] == {
        "identical": True, "calls": 5, "requests": 5, "requests_from_epochs": 5, "wire_requests": 0, "streams": 5,
        "provider_replies": 5, "compactor_requests": 0, "compactor_streams": 0, "images": 4, "tokens": 5,
        "failures": []}
    assert manifest["trace"]["status"] == "valid" and manifest["images"]["recorded_unsent"] == {}
    requests = _rows(destination / "requests.jsonl")
    first = requests[0]["request"]
    assert first["temperature"] == 1.0 and first["stream"] is True  # the wire body, options included
    image_ref = first["messages"][1]["content"][1]["image_url"]["url"]
    assert image_ref.startswith("images/") and (destination / image_ref).read_bytes() == base64.b64decode(A)
    assert requests[1]["request"]["messages"][2]["reasoning"] == "thinking about observe"
    assert [(row["epoch"], row["epoch_prefix"], row["epoch_tail"] is not None) for row in requests] == [
        (0, 2, False), (0, 5, False), (1, 5, False), (1, 7, True), (1, 9, True)]
    index = {item["path"]: item for item in _rows(destination / "images.jsonl")}
    assert index[image_ref]["calls"] == ["model-call-000000", "model-call-000001"]
    responses = _rows(destination / "responses.jsonl")
    assert responses[0]["message"] == {"role": "assistant", "content": "", "reasoning_content": "r0"}
    assert responses[0]["provider_reply"]["content"] == "reply 0"


def test_a_rejected_reply_keeps_the_provider_reply_and_the_rejection(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()), tmp_path / "wire")
    rejected = _rows(destination / "responses.jsonl")[2]
    assert rejected["accepted"] is False and rejected["finish_reason"] == "invalid_tool_input"
    assert rejected["output_rejection"] == {"code": "truncated_tool_call", "provider_stop_reason": "length"}
    assert rejected["provider_reply"] == {
        "content": None, "reasoning_content": "long thought", "finish_reason": "length",
        "tool_calls": [{"id": "call-x", "type": "function", "function": {"name": "move", "arguments": '{"pos'}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 16384}}


def test_compactor_calls_export_and_rebuild(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()[:2], compactions=2), tmp_path / "wire")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["compactor_calls"] == 2
    check = manifest["replay_check"]
    assert (check["identical"], check["compactor_requests"], check["compactor_streams"]) == (True, 2, 2)
    row = _rows(destination / "compactor.jsonl")[0]
    assert row["call_kind"] == "rollout_compaction" and row["reply"]["provider_reply"]["content"] == "summary"
    image_ref = row["request"]["messages"][0]["content"][0]["image_url"]["url"]
    entry = next(item for item in _rows(destination / "images.jsonl") if item["path"] == image_ref)
    assert entry["calls"] == ["model-call-000000", "model-call-000001", "context-compactor-000000",
                              "context-compactor-000001"]


def test_an_incomplete_run_exports_its_unanswered_request_with_its_images(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()[:2], dangling=True), tmp_path / "wire")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["trace"]["status"] == "incomplete"
    assert manifest["trace"]["errors"] == ["request_without_response:model-call-000002"]
    assert (manifest["calls"], manifest["unanswered_calls"], manifest["replay_check"]["identical"]) == (3, 1, True)
    last = _rows(destination / "requests.jsonl")[-1]
    assert last["answered"] is False and _rows(destination / "responses.jsonl")[-1]["provider_reply"] is None
    ref = last["request"]["messages"][0]["content"][1]["image_url"]["url"]
    assert (destination / ref).read_bytes() == base64.b64decode(D)
    assert manifest["images"]["recorded_unsent"] == {}


def test_learning_eligible_needs_a_seal_and_a_valid_eligible_evaluation(tmp_path: Path) -> None:
    def eligible(name: str, **run: object) -> bool:
        destination = export_wire(_run(tmp_path / name, _calls()[:1], **run), tmp_path / f"{name}-wire")
        return json.loads((destination / "manifest.json").read_text())["learning_eligible"]

    assert eligible("sealed", evaluation=ELIGIBLE, sealed=True) is True
    assert eligible("unsealed", evaluation=ELIGIBLE) is False
    assert eligible("invalid", evaluation={**ELIGIBLE, "status": "invalid"}, sealed=True) is False


def test_a_sealed_valid_run_this_verifier_rejects_is_labelled_not_refused(tmp_path: Path, monkeypatch) -> None:
    from ora.agent.evidence import wire_export

    monkeypatch.setattr(wire_export, "verify_agent_trace",
                        lambda root: {"status": "invalid", "errors": ["multiple_mutations"]})
    destination = export_wire(_run(tmp_path / "run", _calls()[:2], evaluation=ELIGIBLE, sealed=True),
                              tmp_path / "wire")
    trace = json.loads((destination / "manifest.json").read_text())["trace"]
    assert trace == {"status": "sealed_valid_unverified_here", "sealed": True, "errors": ["multiple_mutations"]}
    with pytest.raises(TrajectoryError, match="multiple_mutations"):  # unsealed: refused
        export_wire(_run(tmp_path / "unsealed", _calls()[:2], evaluation=ELIGIBLE), tmp_path / "unsealed-wire")


def test_the_check_needs_the_manifest_and_finds_covered_edits_against_the_run(tmp_path: Path) -> None:
    assert {item["check"] for item in check_wire_export(tmp_path / "missing")["failures"]} == {"missing_file"}
    run = _run(tmp_path / "run", _calls(), evaluation=ELIGIBLE, sealed=True)
    destination = export_wire(run, tmp_path / "wire")
    assert check_wire_export(destination, run)["identical"] is True
    responses = destination / "responses.jsonl"
    rows = _rows(responses)
    rows[0]["message"]["reasoning_content"] = "edited"  # ORA's own record, not the provider reply
    responses.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    _rehash(destination, "responses.jsonl")
    assert check_wire_export(destination)["identical"] is True  # the hashes alone are covered
    assert {item["check"] for item in check_wire_export(destination, run)["failures"]} == {"source_responses"}
    manifest = json.loads((destination / "manifest.json").read_text())
    (destination / "manifest.json").write_text(json.dumps({**manifest, "learning_eligible": False}))
    assert "source_manifest" in {item["check"] for item in check_wire_export(destination, run)["failures"]}
    (destination / "manifest.json").unlink()
    assert "missing_file" in {item["check"] for item in check_wire_export(destination)["failures"]}


def test_the_check_finds_a_changed_image_or_request(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()), tmp_path / "wire")
    image = sorted((destination / "images").iterdir())[0]
    image.write_bytes(image.read_bytes() + b"\0")
    report = check_wire_export(destination)
    assert report["identical"] is False and {item["check"] for item in report["failures"]} == {
        "requests", "requests_from_epochs", "images", "manifest_file"} - {"manifest_file"}


def test_the_check_finds_edited_replies_index_and_missing_images(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()), tmp_path / "wire")
    responses = destination / "responses.jsonl"
    rows = _rows(responses)
    rows[0]["provider_reply"]["reasoning_content"] = "edited"
    responses.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checks = {item["check"] for item in check_wire_export(destination)["failures"]}
    assert checks == {"provider_replies", "manifest_file"}
    index = destination / "images.jsonl"
    entries = _rows(index)
    entries[0]["calls"] = entries[0]["calls"][:1] + ["model-call-000009"]
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    assert "images" in {item["check"] for item in check_wire_export(destination)["failures"]}
    sorted((destination / "images").iterdir())[0].unlink()
    report = check_wire_export(destination)  # a structured failure, not an OSError
    assert any(item["check"] == "requests" and "FileNotFoundError" in item.get("detail", "")
               for item in report["failures"])


def test_the_export_is_group_readable_under_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o002)
    try:
        destination = export_wire(_run(tmp_path / "run", _calls()[:1]), tmp_path / "wire")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(destination.stat().st_mode) & 0o775 == 0o775
    assert stat.S_IMODE((destination / "manifest.json").stat().st_mode) == 0o664


def test_another_wire_protocol_exports_its_canonical_request_and_its_wire_body(tmp_path: Path) -> None:
    messages = [{"role": "system", "content": "s"}, _user(A)]
    anthropic = {"model": "m", "system": "s", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "observation"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": A}}]}]}
    destination = export_wire(_run(tmp_path / "run", [{"messages": messages, "anthropic": anthropic}]),
                              tmp_path / "wire")
    row = _rows(destination / "requests.jsonl")[0]
    assert row["request_source"] == "canonical" and row["request"]["messages"] == [
        messages[0], {**messages[1], "content": [messages[1]["content"][0], {
            "type": "image_url", "image_url": {"url": row["wire_request"]["messages"][0]["content"][1]["source"]["data"]}}]}]
    report = json.loads((destination / "manifest.json").read_text())["replay_check"]
    assert (report["identical"], report["wire_requests"]) == (True, 1)
    assert len(list((destination / "images").iterdir())) == 1


def test_a_privileged_field_refuses_the_export_and_leaves_nothing_behind(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [{"messages": [{"role": "user", "content": "x", "object_gt": [0, 0, 0]}]}])
    with pytest.raises(TrajectoryError, match="privileged"):
        export_wire(run, tmp_path / "wire")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run"]


def _script_run(root: Path, result: dict) -> Path:
    """A code-mode run whose request repeats the result of a script."""

    messages = [{"role": "tool", "name": "bash", "tool_call_id": "script", "content": json.dumps(result)}]
    return _run(root, [{"messages": messages}], policy={"code_mode": {"contract": "c"}})


def test_a_key_the_model_chose_in_a_scripts_result_exports_and_a_private_one_does_not(
        tmp_path: Path, monkeypatch) -> None:
    def proposals(events: list) -> list:
        return [{"type": "tool_call_proposed", "tool_call_id": "script", "payload": {"name": "bash"}}, *events]

    monkeypatch.setattr(wire_export, "_trace_status", lambda *_: {"status": "valid", "sealed": False, "errors": []})
    real_events = wire_export.read_agent_events
    monkeypatch.setattr(wire_export, "read_agent_events", lambda root: proposals(real_events(root)))
    authored = _script_run(tmp_path / "authored", {"ok": True, "stdout": json.dumps({"contact": [0.1, 0.2, 0.3]})})
    export_wire(authored, tmp_path / "authored-wire")
    private = _script_run(tmp_path / "private", {"ok": True, "calls": [{"receipt": {"object_gt": [0, 0, 0]}}]})
    with pytest.raises(TrajectoryError, match="privileged field in model-call-000000's request"):
        export_wire(private, tmp_path / "private-wire")


def test_a_call_the_model_wrote_exports_under_any_key_and_a_rewritten_one_does_not(
        tmp_path: Path, monkeypatch) -> None:
    """The arguments of a tool call in the model's reply are its own text, in the reply and in a later request."""

    def call(arguments: dict) -> dict:
        return {"id": "call-write", "type": "function",
                "function": {"name": "write", "arguments": json.dumps(arguments)}}

    written = call({"arguments=\n{\"content\": \"pad contact\"}": ""})

    def replies(events: list) -> list:
        return [{"type": "provider_response", "payload": {"message": {"role": "assistant", "tool_calls": [written]}}},
                *events]

    monkeypatch.setattr(wire_export, "_trace_status", lambda *_: {"status": "valid", "sealed": False, "errors": []})
    real_events = wire_export.read_agent_events
    monkeypatch.setattr(wire_export, "read_agent_events", lambda root: replies(real_events(root)))
    for name, repeated in (("model", written), ("rewritten", call({"contact": True}))):
        messages = [{"role": "user", "content": "carry"},
                    {"role": "assistant", "content": "", "tool_calls": [repeated]},
                    {"role": "tool", "tool_call_id": "call-write", "content": "write takes path and content"}]
        run = _run(tmp_path / name, [{"messages": messages}])
        if name == "model":
            export_wire(run, tmp_path / f"{name}-wire")
        else:
            with pytest.raises(TrajectoryError, match="privileged field in model-call-000000's request"):
                export_wire(run, tmp_path / f"{name}-wire")


def test_the_check_finds_manifest_count_edits_and_added_files(tmp_path: Path) -> None:
    destination = export_wire(_run(tmp_path / "run", _calls()), tmp_path / "wire")
    manifest_path = destination / "manifest.json"
    original = json.loads(manifest_path.read_text())

    def failures_after(change) -> set[str]:
        manifest_path.write_text(json.dumps(change(json.loads(json.dumps(original)))))
        found = {item["check"] for item in check_wire_export(destination)["failures"]}
        manifest_path.write_text(json.dumps(original))
        return found

    assert check_wire_export(destination)["identical"] is True
    assert failures_after(lambda m: {**m, "replay_check": {**m["replay_check"], "requests": 99}}) == {
        "manifest_replay_check"}
    assert failures_after(lambda m: {**m, "note": "added"}) == {"manifest_keys"}
    assert failures_after(lambda m: {**m, "files": [{**m["files"][0], "note": 1}, *m["files"][1:]]}) == {
        "manifest_keys"}
    assert failures_after(lambda m: {**m, "files": [{**m["files"][0], "bytes": 1}, *m["files"][1:]]}) == {
        "manifest_file"}
    (destination / "notes.txt").write_text("x")
    (destination / "images" / "stray.png").write_bytes(b"x")
    assert {item["detail"] for item in check_wire_export(destination)["failures"]} == {"notes.txt", "images/stray.png"}


def test_an_unanswered_request_keeps_its_place_in_the_trace(tmp_path: Path) -> None:
    root = tmp_path / "run"
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, "run", "session")
    for call_id, answered in (("model-call-000000", False), ("model-call-000001", True)):
        request = store.put_json({"model": "m", "messages": [_user(text=call_id)]}, kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=call_id, artifact_ref=request,
                        payload={"model": "m", "provider_protocol": "openai_chat_completions"})
        if answered:
            recorder.append("provider_response", model_call_id=call_id,
                            payload={"accepted": True, "message": {"role": "assistant", "content": "ok"}})
    (root / "plan.json").write_text(json.dumps({"source": {}, "policy": {}}))
    (root / "result.json").write_text(json.dumps({"run_id": "run"}))
    destination = export_wire(root, tmp_path / "wire")
    assert [(row["model_call_id"], row["answered"]) for row in _rows(destination / "requests.jsonl")] == [
        ("model-call-000000", False), ("model-call-000001", True)]


def test_host_paths_redact_requests_streams_and_wire_bodies_and_the_check_derives_them_again(tmp_path: Path) -> None:
    seen = {"role": "tool", "tool_call_id": "call-observe", "content": 'File "/data/lab/py/lib/json/encoder.py"'}
    messages = [{"role": "system", "content": "s"}, _user(A), _assistant("observe"), seen]
    echo = [{"prompt_token_ids": [1, 2], "choices": [{"delta": {"reasoning_content": "it failed in /data/lab/py/lib"},
                                                      "token_ids": [3, 4]}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}]
    anthropic = {"model": "m", "system": "s", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "/data/lab/py/x"}]}]}
    run = _run(tmp_path / "run", [{"messages": messages, "stream": echo},
                                  {"messages": messages + [_user(B)], "anthropic": anthropic}])
    destination = export_wire(run, tmp_path / "wire", host_paths=HostPaths.for_run(run, ["/data/lab"]))
    assert all(b"/data/lab" not in path.read_bytes() for path in destination.iterdir() if path.is_file())
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [(row["file"], row["model_call_id"], row["field"], row["replacements"])
            for row in manifest["redaction"]["rows"]] == [
        ("requests.jsonl", "model-call-000000", "request", 1), ("streams.jsonl", "model-call-000000", "response", 1),
        ("requests.jsonl", "model-call-000001", "request", 1),
        ("requests.jsonl", "model-call-000001", "wire_request", 1)]
    # The reply echoed the path: its reasoning reads redacted, and no token id of either text is shipped.
    reply = _rows(destination / "responses.jsonl")[0]["provider_reply"]
    assert reply["reasoning_content"] == "it failed in <host-path>/py/lib"
    tokens = _rows(destination / "tokens.jsonl")[0]
    assert tokens["prompt"] is None and tokens["output"] is None
    # No sealed sha256 of a changed row ships; the redaction record has none either.
    requests = _rows(destination / "requests.jsonl")
    assert [(row["request_sha256"], row.get("wire_request_sha256")) for row in requests] == [(None, None)] * 2
    assert _rows(destination / "streams.jsonl")[0]["response_sha256"] is None
    assert "final_event_sha256" not in manifest["trace"] and "sha256" not in json.dumps(manifest["redaction"]["rules"])
    check = manifest["replay_check"]
    assert check["identical"] and (check["requests"], check["requests_redacted"], check["streams_redacted"],
                                   check["wire_requests_redacted"]) == (0, 2, 1, 1)
    against_run = check_wire_export(destination, run, redact_paths=["/data/lab"])
    assert against_run["identical"], against_run["failures"]
    assert "derived again from the sealed run" in against_run["redaction"]
    assert "redaction_rules" in {item["check"] for item in check_wire_export(destination, run)["failures"]}


def test_a_host_path_split_across_stream_chunks_refuses_the_redacted_export(tmp_path: Path) -> None:
    split = [{"choices": [{"delta": {"content": "see /data/"}}]}, {"choices": [{"delta": {"content": "lab/x"}}]}]
    run = _run(tmp_path / "run", [{"messages": [_user(A)], "stream": split}])
    with pytest.raises(TrajectoryError, match="split across stream chunks"):
        export_wire(run, tmp_path / "wire", host_paths=HostPaths([("path", "/data/lab")]))
    assert not (tmp_path / "wire").exists()
    assert check_wire_export(export_wire(run, tmp_path / "plain"), run)["identical"]
