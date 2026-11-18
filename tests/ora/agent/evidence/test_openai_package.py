"""The OpenAI package: standard conversations per epoch, calls, the wire export, and the tool that verifies them."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ora.agent.evidence import openai_package_tool as tool
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.host_paths import HostPaths
from ora.agent.evidence.openai_package import check_openai_package, export_openai
from tests.support.agent.agent_events import TrajectoryRecorder

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def _stream(prompt: list[int], output: list[int], usage: dict) -> list[dict]:
    return [
        {"prompt_token_ids": prompt, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "think"}, "token_ids": output[:1], "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c", "type": "function",
                                                            "function": {"name": "observe", "arguments": "{}"}}]},
                      "token_ids": output[1:], "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": usage},
    ]


def _user(text: str, image: bool = False) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, *([IMAGE] if image else [])]}


FIRST = [{"role": "system", "content": "s"}, _user("look", image=True)]


def _second(seen: str = "seen") -> list[dict]:
    return FIRST + [{"role": "assistant", "content": "", "reasoning": "think",
                     "tool_calls": [{"id": "c", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c", "content": seen}, _user("again")]


SECOND = _second()
REWRITTEN = [{"role": "system", "content": "s"}, _user("look (image now a pointer)"), _user("again")]


def _run(root: Path, *, replay_program: bool = True, seen: str = "seen", interpreter: Path | None = None) -> Path:
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, "run", "session")
    calls = [(FIRST, [1, 2, 3], [4, 5]), (_second(seen), [1, 2, 3, 4, 5, 6], [7, 8]), (REWRITTEN, [1, 9], [10, 11])]
    for index, (messages, prompt, output) in enumerate(calls):
        call_id = f"model-call-{index:06d}"
        wire = store.put_json({"model": "m", "stream": True, "messages": messages, "reasoning_effort": "low",
                               "tools": [{"type": "function", "function": {"name": "observe", "parameters": {}}}]},
                              kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=call_id, artifact_ref=wire,
                        payload={"model": "m", "provider_protocol": "openai_chat_completions"})
        stream = store.put_json(_stream(prompt, output, {"prompt_tokens": len(prompt), "completion_tokens": len(output)}),
                                kind="native_provider_wire_response")
        recorder.append("provider_response", model_call_id=call_id, artifact_ref=stream,
                        payload={"accepted": True, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": ""}})
    (root / "plan.json").write_text(json.dumps({
        "source": {"commit": "test"}, "runtime": {"kind": "synthetic"}, "task": "look", "seed": 3,
        "task_instruction": "Look twice.", "policy": {"model": "m", "reasoning_effort": "low"}}))
    (root / "result.json").write_text(json.dumps({"run_id": "run", "official_success": True, "termination": "test"}))
    if replay_program:
        (root / "replay-program").mkdir()
        (root / "replay-program" / "program.json").write_text('{"schema": "ora.replay-program.v1"}')
    if interpreter is not None:  # the code-mode interpreter preflight records, whose prefixes tracebacks carry
        (root / "preflight.json").write_text(json.dumps({"checks": [
            {"check": "code_mode_sandbox", "detail": {"interpreter": {"executable": str(interpreter)}}}]}))
    return root


def _interpreter(tmp_path: Path) -> tuple[Path, Path]:
    """A code-mode environment and the managed Python its pyvenv.cfg names, as uv lays them out."""
    base, environment = tmp_path / "uv" / "python" / "cpython-3.11.12-linux-x86_64-gnu", tmp_path / "envs" / "code-mode"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text(f"home = {base}/bin\nversion_info = 3.11.12\n")
    return environment / "bin" / "python", base


def _traceback(base: Path) -> str:
    return ('Traceback (most recent call last):\n  File "/work/push.py", line 3, in <module>\n'
            f'  File "{base}/lib/python3.11/json/__init__.py", line 179, in dump\n'
            f'  File "{base}/lib/python3.11/json/encoder.py", line 406, in _iterencode\n'
            "TypeError: Object of type ndarray is not JSON serializable")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_package_holds_standard_conversations_calls_and_the_wire_export(tmp_path: Path) -> None:
    package = export_openai(_run(tmp_path / "run"), tmp_path / "package")
    conversations = _rows(package / "conversations.jsonl")
    assert [item["epoch"] for item in conversations] == [0, 1]
    first = conversations[0]
    assert first["model_call_ids"] == ["model-call-000000", "model-call-000001"]
    # The epoch's last request, images inline, the reasoning under its usual name, then the epoch's final reply.
    assert first["messages"][:-1][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert first["messages"][2]["reasoning_content"] == "think" and "reasoning" not in first["messages"][2]
    assert first["messages"][-1] == {"role": "assistant", "content": "", "reasoning_content": "think",
                                     "tool_calls": [{"id": "c", "type": "function",
                                                     "function": {"name": "observe", "arguments": "{}"}}]}
    assert first["tools"][0]["function"]["name"] == "observe"
    calls = _rows(package / "calls.jsonl")
    assert len(calls) == 3 and calls[0]["token_counts"] == {
        "prompt": 3, "sampled": 2, "prompt_equal_to_usage": True, "sampled_equal_to_usage": True}
    reference = calls[0]["request"]["messages"][1]["content"][1]["image_url"]["url"]
    assert reference.startswith("wire/images/") and (package / reference).is_file()
    listed = {entry["path"] for entry in json.loads((package / "package.json").read_text())["files"]}
    assert listed == {"README.md", "calls.jsonl", "conversations.jsonl", "ora_package.py", "replay-program/program.json"}
    readme = (package / "README.md").read_text()
    assert "| Mode | tool calls |" in readme and "prompt ids on 3 of 3 calls (3 from the stream)" in readme
    report = check_openai_package(package, tmp_path / "run")
    assert report["identical"], report["failures"]


def test_the_shipped_tool_verifies_and_converts_with_the_standard_library(tmp_path: Path) -> None:
    package = export_openai(_run(tmp_path / "run"), tmp_path / "package")
    tree = ast.parse((package / "ora_package.py").read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module != "__future__"}
    assert imported and imported <= set(sys.stdlib_module_names)

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "ora_package.py", *args], cwd=package, capture_output=True, text=True)

    verified = run("verify")
    assert verified.returncode == 0 and "PACKAGE VERIFIED" in verified.stdout
    assert run("calls", str(tmp_path / "calls.openai.jsonl")).returncode == 0
    samples = _rows(tmp_path / "calls.openai.jsonl")
    assert samples[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png")
    assert samples[0]["messages"][-1]["role"] == "assistant"
    assert run("speculator", str(tmp_path / "spec.jsonl"), "--per-epoch").returncode == 0
    rows = _rows(tmp_path / "spec.jsonl")
    # Calls 0 and 1 extend each other token for token; call 2 starts over after the rewrite.
    assert [row["model_call_ids"] for row in rows] == [["model-call-000000", "model-call-000001"], ["model-call-000002"]]
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8] and rows[0]["loss_mask"] == [0, 0, 0, 1, 1, 0, 1, 1]
    assert run("images", str(tmp_path / "images")).returncode == 0
    assert len(list((tmp_path / "images").iterdir())) == 1


def test_an_edit_anywhere_fails_and_an_added_file_is_only_listed(tmp_path: Path) -> None:
    package = export_openai(_run(tmp_path / "run", replay_program=False), tmp_path / "package")
    (package / "notes.txt").write_text("mine")
    report = tool.verify(package)
    assert report["identical"] and report["unlisted"] == ["notes.txt"]
    conversations = _rows(package / "conversations.jsonl")
    conversations[0]["messages"][0]["content"] = "edited"
    (package / "conversations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in conversations))
    image = next((package / "wire" / "images").iterdir())
    image.write_bytes(image.read_bytes() + b"\0")
    (package / "wire" / "epochs.jsonl").write_text((package / "wire" / "epochs.jsonl").read_text().replace("look", "lock"))
    failed = {failure["check"] for failure in tool.verify(package)["failures"]}
    assert {"package_file", "conversations.jsonl", "image", "wire_file", "request_from_epoch"} <= failed


def test_a_package_against_another_run_fails_the_source_check(tmp_path: Path) -> None:
    package = export_openai(_run(tmp_path / "run"), tmp_path / "package")
    other = _run(tmp_path / "other")
    (other / "result.json").write_text(json.dumps({"run_id": "other", "official_success": False, "termination": "x"}))
    report = check_openai_package(package, other)
    assert not report["identical"]
    assert any(failure["check"].startswith("wire_source") for failure in report["failures"])


def test_the_package_records_the_wire_manifest_it_was_derived_from(tmp_path: Path) -> None:
    package = export_openai(_run(tmp_path / "run"), tmp_path / "package")
    recorded = json.loads((package / "package.json").read_text())["wire_manifest_sha256"]
    assert recorded == hashlib.sha256((package / "wire" / "manifest.json").read_bytes()).hexdigest()


def _verify(package: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "ora_package.py", "verify"], cwd=package, capture_output=True, text=True)


def manifest_of(package: Path) -> dict:
    return json.loads((package / "wire" / "manifest.json").read_text())


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_host_path_in_a_tool_result_is_redacted_recorded_and_verified(tmp_path: Path) -> None:
    python, base = _interpreter(tmp_path)
    run = _run(tmp_path / "run", seen=_traceback(base), interpreter=python)
    plain = export_openai(run, tmp_path / "plain")
    redacted = export_openai(run, tmp_path / "redacted", host_paths=HostPaths.for_run(run))
    assert str(base) in (plain / "wire" / "requests.jsonl").read_text()  # without the option nothing changes
    for name, data in _files(redacted).items():
        assert str(base).encode() not in data, name
    # The placeholder keeps the path's tail, so the traceback still reads; the sandbox's own /work stays.
    sent = _rows(redacted / "calls.jsonl")[1]["request"]["messages"][3]["content"]
    assert sent == _traceback(Path("<host-python>"))
    assert 'File "/work/push.py"' in sent
    manifest = json.loads((redacted / "wire" / "manifest.json").read_text())
    record = manifest["redaction"]
    assert {"rule": "python", "placeholder": "<host-python>", "replacements": 2} in record["rules"]
    assert record["replacements"] == sum(rule["replacements"] for rule in record["rules"]) == 2
    # Call 1's request changed; its stream only loses the prompt ids, which were the original text's.
    assert [(row["file"], row["model_call_id"], row["field"], row["replacements"]) for row in record["rows"]] == [
        ("requests.jsonl", "model-call-000001", "request", 2), ("streams.jsonl", "model-call-000001", "response", 0)]
    assert all(set(row) == {"file", "model_call_id", "field", "replacements", "redacted_sha256"}
               for row in record["rows"])
    # Nothing derived from the redacted bytes ships: not the prefix's hash, not a changed row's sealed sha256, not
    # the trace's final event hash; an unchanged row keeps its sealed sha256.
    sealed = {row["model_call_id"]: row["request_sha256"] for row in _rows(plain / "wire" / "requests.jsonl")}
    streams = {row["model_call_id"]: row["response_sha256"] for row in _rows(plain / "wire" / "streams.jsonl")}
    withheld = [hashlib.sha256(str(base).encode()).hexdigest(), sealed["model-call-000001"],
                streams["model-call-000001"], manifest_of(plain)["trace"]["final_event_sha256"]]
    for name, data in _files(redacted).items():
        assert not any(digest.encode() in data for digest in withheld), name
    rows = _rows(redacted / "wire" / "requests.jsonl")
    assert [row["request_sha256"] for row in rows] == [sealed["model-call-000000"], None, sealed["model-call-000002"]]
    assert _rows(redacted / "wire" / "streams.jsonl")[1]["response_sha256"] is None
    assert "final_event_sha256" not in manifest["trace"]
    tokens = {row["model_call_id"]: row for row in _rows(redacted / "wire" / "tokens.jsonl")}
    assert tokens["model-call-000001"]["prompt"] is None and tokens["model-call-000001"]["output"] is not None
    assert tokens["model-call-000002"]["prompt"]["prefix"]["model_call_id"] == "model-call-000000"
    assert _rows(redacted / "wire" / "streams.jsonl")[1]["response"][0]["prompt_token_ids"] is None
    readme = (redacted / "README.md").read_text()
    assert "## Redaction" in readme and "prompt ids on 2 of 3 calls" in readme and "cannot prove" in readme
    assert "## Redaction" not in (plain / "README.md").read_text()
    report = tool.verify(redacted)
    assert report["identical"], report["failures"]
    assert report["checks"]["request"] == 2 and report["checks"]["request_redacted"] == 1
    assert report["checks"]["stream_redacted"] == 1 and report["checks"]["redaction_ids"] == 1
    assert report["redaction"]["requests"] == 1 and "cannot prove" in report["redaction"]["statement"]
    verified = _verify(redacted)
    assert verified.returncode == 0, verified.stdout
    assert "PACKAGE VERIFIED AS REDACTED: 1 requests and 1 streams" in verified.stdout
    assert "1 of 3 requests and 1 streams are redacted" in verified.stdout and "cannot prove" in verified.stdout
    verified = _verify(plain)
    assert verified.stdout.endswith("PACKAGE VERIFIED\n") and "redaction" not in verified.stdout
    # With the sealed run the redaction is derived again and compared.
    against_run = check_openai_package(redacted, run)
    assert against_run["identical"], against_run["failures"]
    assert "derived again from the sealed run" in against_run["redaction"]["statement"]
    assert check_openai_package(plain, run)["identical"]


def test_a_package_without_host_paths_differs_only_by_its_redaction_record(tmp_path: Path) -> None:
    run = _run(tmp_path / "run")
    plain, screened = export_openai(run, tmp_path / "plain"), export_openai(
        run, tmp_path / "screened", host_paths=HostPaths.for_run(run))
    before, after = _files(plain), _files(screened)
    assert set(before) == set(after)
    differing = {name for name in before if before[name] != after[name]}
    assert differing == {"README.md", "package.json", "wire/manifest.json"}
    manifest = json.loads(after["wire/manifest.json"])
    record = manifest.pop("redaction")
    assert manifest == json.loads(before["wire/manifest.json"])
    assert record["rows"] == [] and record["replacements"] == 0 and record["rules"]
    assert all(rule["replacements"] == 0 for rule in record["rules"])
    added = set(after["README.md"].decode().splitlines()) - set(before["README.md"].decode().splitlines())
    assert added == {"| Host paths | screened with `--redact-host-paths`: none found |"}
    package = json.loads(after["package.json"])
    assert package.pop("redaction") == {"files": {"README.md": 0, "replay-program/program.json": 0}}
    assert set(package) == set(json.loads(before["package.json"]))
    verified = _verify(screened)
    assert verified.returncode == 0 and verified.stdout.endswith("PACKAGE VERIFIED\n")
    assert "redaction: screened for host paths (--redact-host-paths): none found" in verified.stdout
    assert check_openai_package(screened, run)["identical"]


def test_a_tampered_redacted_request_fails_and_an_unknown_prefix_fails_the_check(tmp_path: Path) -> None:
    python, base = _interpreter(tmp_path)
    run = _run(tmp_path / "run", seen=_traceback(base) + "\nsee /data/lab/notes.txt", interpreter=python)
    package = export_openai(run, tmp_path / "package", host_paths=HostPaths.for_run(run, ["/data/lab"]))
    assert "<host-path>/notes.txt" in (package / "wire" / "requests.jsonl").read_text()
    # The check against the run needs every recorded prefix again: one given with --redact-path is given again.
    assert check_openai_package(package, run, redact_paths=["/data/lab"])["identical"]
    missing = check_openai_package(package, run)
    assert "wire_redaction_rules" in {failure["check"] for failure in missing["failures"]}
    requests = package / "wire" / "requests.jsonl"
    requests.write_text(requests.read_text().replace("<host-python>/lib/python3.11/json/encoder.py",
                                                     "<host-python>/lib/python3.11/json/decoder.py"))
    report = tool.verify(package)
    assert {"request_redacted", "wire_file", "calls.jsonl"} <= {failure["check"] for failure in report["failures"]}
    verified = _verify(package)
    assert verified.returncode == 1 and "FAIL request_redacted: model-call-000001" in verified.stdout
    assert "PACKAGE VERIFIED" not in verified.stdout
