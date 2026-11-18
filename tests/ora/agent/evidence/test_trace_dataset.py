"""The trace dataset: one record per episode with its segments in order, each image once, and the shipped reader."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from ora.agent.evidence import trace_dataset
from ora.agent.evidence import trace_dataset_tool as tool
from ora.agent.evidence.canonical import ArtifactStore, TrajectoryError
from ora.agent.evidence.trace_dataset import default_episode_id, export_dataset, read_entries
from tests.support.agent.agent_events import TrajectoryRecorder

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"one picture").decode("ascii")
OTHER = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"another picture").decode("ascii")
TOOLS = [{"type": "function", "function": {"name": "observe", "parameters": {}}}]


def _image(data: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _user(text: str, *images: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, *(_image(item) for item in images)]}


def _stream(usage: dict) -> list[dict]:
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "think"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c", "type": "function",
                                                            "function": {"name": "observe", "arguments": "{}"}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": usage},
    ]


def _calls(seen: str = "seen", picture: str = PNG) -> list[list[dict]]:
    """Three calls: the second extends the first, the third is a rewritten context (a second segment)."""
    first = [{"role": "system", "content": "s"}, _user("look", picture)]
    second = first + [{"role": "assistant", "content": "", "reasoning": "think",
                       "tool_calls": [{"id": "c", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
                      {"role": "tool", "tool_call_id": "c", "content": seen}, _user("again")]
    rewritten = [{"role": "system", "content": "s"}, _user("look (image now a pointer)"), _user("again", OTHER)]
    return [first, second, rewritten]


def _run(root: Path, run_id: str, calls: list[list[dict]], *, code_mode: bool = False) -> Path:
    root.mkdir(parents=True)
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, run_id, "session")
    for index, messages in enumerate(calls):
        call_id = f"model-call-{index:06d}"
        wire = store.put_json({"model": "m", "stream": True, "messages": messages, "reasoning_effort": "low",
                               "tools": TOOLS}, kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=call_id, artifact_ref=wire,
                        payload={"model": "m", "provider_protocol": "openai_chat_completions"})
        stream = store.put_json(_stream({"prompt_tokens": 10 + index, "completion_tokens": 2}),
                                kind="native_provider_wire_response")
        recorder.append("provider_response", model_call_id=call_id, artifact_ref=stream,
                        payload={"accepted": True, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": ""}})
    (root / "plan.json").write_text(json.dumps({
        "source": {"commit": "0123456789abcdef"}, "runtime": {"kind": "synthetic"}, "task": "look", "seed": 3,
        "task_instruction": "Look twice.", "policy": {"model": "m", "reasoning_effort": "low", "code_mode": code_mode}}))
    (root / "result.json").write_text(json.dumps({"run_id": run_id, "official_success": True, "termination": "test"}))
    return root


def _episodes(dataset: Path) -> list[dict]:
    rows = []
    for shard in json.loads((dataset / "dataset.json").read_text())["episodes"]["shards"]:
        with gzip.open(dataset / shard["path"], "rt", encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle]
    return rows


def _dataset(tmp_path: Path, **options) -> tuple[Path, dict]:
    runs = [_run(tmp_path / "runs" / "exp-secret-a", "exp-secret-a", _calls()),
            _run(tmp_path / "runs" / "exp-secret-b", "exp-secret-b", _calls(seen="seen in exp-secret-b"), code_mode=True)]
    entries = [{"run": str(runs[0]), "episode_id": "first", "labels": {"within_step_limit": True}}, {"run": str(runs[1])}]
    report = export_dataset(entries, tmp_path / "ds", workers=1, **options)
    return tmp_path / "ds", report


def test_an_episode_holds_its_segments_in_order_and_every_call_reads_from_them(tmp_path: Path) -> None:
    dataset, report = _dataset(tmp_path)
    assert (report["runs"], report["exported"], report["refused"], report["failed"]) == (2, 2, [], [])
    first, second = _episodes(dataset)
    assert first["schema"] == tool.EPISODE_SCHEMA and first["episode_id"] == "first"
    assert second["episode_id"] == default_episode_id("exp-secret-b")
    assert (first["model"], first["benchmark"], first["task"], first["seed"], first["instruction"]) == (
        "m", "synthetic", "look", 3, "Look twice.")
    assert (first["mode"], second["mode"]) == ("tool_calls", "code")
    assert first["result"] == {"success": True, "termination": "test", "learning_eligible": False}
    assert first["labels"] == {"within_step_limit": True} and first["tools"] == TOOLS
    assert first["usage"] == {"model_calls": 3, "prompt_tokens": 33, "completion_tokens": 6, "reasoning_tokens": 0}
    assert first["ora_commit"] == "012345678"
    # Two segments: the two calls that extend each other, then the rewritten context.
    one, two = first["segments"]
    assert [call["prefix"] for call in one["calls"]] == [2, 5] and [call["prefix"] for call in two["calls"]] == [3]
    assert one["messages"][2]["reasoning_content"] == "think" and "reasoning" not in one["messages"][2]
    assert one["messages"][-1]["role"] == "assistant" and one["messages"][-1]["reasoning_content"] == "think"
    samples = list(tool.call_samples(first))
    assert [len(sample["messages"]) for sample in samples] == [2, 5, 3]
    assert samples[0]["response"] == one["messages"][2] and samples[1]["response"] == one["messages"][-1]
    assert samples[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2} and samples[0]["tools"] == TOOLS


def test_each_image_is_stored_once_and_read_back_from_its_shard(tmp_path: Path) -> None:
    dataset, report = _dataset(tmp_path)
    assert report["images"] == 2  # both runs sent the same two pictures
    rows = tool.read_jsonl(dataset / "images.jsonl")
    assert sorted(row["bytes"] for row in rows) == sorted(len(base64.b64decode(item)) for item in (PNG, OTHER))
    loaded = tool.Dataset(dataset)
    first = _episodes(dataset)[0]
    ref = first["segments"][0]["messages"][1]["content"][1]["image_url"]["url"]
    assert ref == f"images/{hashlib.sha256(base64.b64decode(PNG)).hexdigest()}.png"
    assert loaded.image_bytes(ref) == base64.b64decode(PNG)
    assert loaded.inline(first["segments"][0]["messages"][1])["content"][1] == _image(PNG)
    with tarfile.open(dataset / rows[0]["shard"]) as archive:  # a plain tar anyone can unpack
        assert sorted(archive.getnames()) == sorted(f"{row['sha256']}.{row['ext']}" for row in rows)
    index = tool.read_jsonl(dataset / "index.jsonl")
    assert [(row["episode_id"], row["images"], row["segments"], row["model_calls"], row["line"]) for row in index] == [
        ("first", 2, 2, 3, 0), (default_episode_id("exp-secret-b"), 2, 2, 3, 1)]


def test_nothing_in_the_dataset_names_a_run_and_the_map_is_beside_it(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)
    second = _episodes(dataset)[1]
    assert second["segments"][0]["messages"][3]["content"] == f"seen in {second['episode_id']}"
    assert second["redaction"]["run_id"] == 1
    for path in dataset.rglob("*"):
        if path.is_file() and path.suffix != ".tar":
            text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
            assert "exp-secret" not in text and str(tmp_path) not in text, path
    mapping = tool.read_jsonl(tmp_path / "ds.run-map.jsonl")
    assert [(row["episode_id"], row["run_id"]) for row in mapping] == [
        ("first", "exp-secret-a"), (second["episode_id"], "exp-secret-b")]
    assert not (tmp_path / "ds.work").exists()


def test_a_forbidden_string_refuses_the_episode_and_the_report_lists_it(tmp_path: Path) -> None:
    runs = [_run(tmp_path / "runs" / "a", "a-run", _calls()),
            _run(tmp_path / "runs" / "b", "b-run", _calls(seen="read from node-17"))]
    report = export_dataset([{"run": str(run)} for run in runs] + [{"run": str(tmp_path / "missing")}],
                            tmp_path / "ds", workers=1, forbid=["node-17"])
    assert report["exported"] == 1
    assert [(item["run_id"], item["error"]) for item in report["refused"]] == [("b-run", "forbidden text: 'node-17'")]
    assert [item["error"] for item in report["failed"]] == ["TrajectoryError: not a run directory"]
    assert json.loads((tmp_path / "ds.report.json").read_text())["refused"] == report["refused"]
    assert [row["episode_id"] for row in tool.read_jsonl(tmp_path / "ds" / "index.jsonl")] == [default_episode_id("a-run")]


def test_the_shipped_tool_verifies_and_converts_with_the_standard_library(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)

    def run(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", str(dataset / "ora_traces.py"), *arguments],
                              capture_output=True, text=True, cwd=tmp_path)

    verified = run("verify", "--images")
    assert verified.returncode == 0 and json.loads(verified.stdout)["identical"] is True
    assert json.loads(run("stats").stdout)["mode"] == {"code": 1, "tool_calls": 1}
    assert run("openai", "first", str(tmp_path / "openai.jsonl")).returncode == 0
    conversations = tool.read_jsonl(tmp_path / "openai.jsonl")
    assert len(conversations) == 2 and conversations[0]["messages"][1]["content"][1] == _image(PNG)
    assert conversations[0]["tools"] == TOOLS
    assert run("calls", "first", str(tmp_path / "calls.jsonl")).returncode == 0
    assert [len(row["messages"]) for row in tool.read_jsonl(tmp_path / "calls.jsonl")] == [2, 5, 3]
    assert run("images", str(tmp_path / "pictures"), "first").returncode == 0
    assert len(list((tmp_path / "pictures").iterdir())) == 2


def test_an_edit_fails_the_check(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)
    assert tool.verify(dataset, images=True)["identical"] is True
    index = dataset / "index.jsonl"
    index.write_text(index.read_text().replace('"images": 2', '"images": 3', 1))
    checks = {item["check"] for item in tool.verify(dataset)["failures"]}
    assert {"file_changed", "index_images"} <= checks
    (dataset / "notes.txt").write_text("mine")
    assert tool.verify(dataset)["files_not_listed"] == ["notes.txt"]


def test_an_interrupted_export_resumes_from_its_work_directory(tmp_path: Path, monkeypatch) -> None:
    run = _run(tmp_path / "runs" / "a", "a-run", _calls())
    job = {"entry": {"run": str(run)}, "work": str(tmp_path / "work" / "0000000"), "redact_host_paths": True,
           "redact_paths": [], "forbid": [], "settings": {"forbid": []}}
    first = trace_dataset._export_one(job)
    assert first["status"] == "exported" and "resumed" not in first
    monkeypatch.setattr(trace_dataset, "export_wire", lambda *_a, **_k: pytest.fail("exported again"))
    assert trace_dataset._export_one(job)["resumed"] is True
    monkeypatch.undo()
    # Other settings, or another run in that place, export again.
    assert "resumed" not in trace_dataset._export_one({**job, "settings": {"forbid": ["x"]}})


def test_the_run_list_is_paths_or_objects_and_ids_are_not_repeated(tmp_path: Path) -> None:
    listing = tmp_path / "runs.txt"
    listing.write_text('# comment\n/data/a\n\n{"run": "/data/b", "episode_id": "b", "labels": {"k": 1}}\n')
    assert read_entries(listing) == [{"run": "/data/a"}, {"run": "/data/b", "episode_id": "b", "labels": {"k": 1}}]
    listing.write_text('{"run": "/data/a", "note": "x"}\n')
    with pytest.raises(TrajectoryError, match="an entry is"):
        read_entries(listing)
    with pytest.raises(TrajectoryError, match="given twice"):
        export_dataset([{"run": "/a", "episode_id": "same"}, {"run": "/b", "episode_id": "same"}], tmp_path / "ds")
    with pytest.raises(TrajectoryError, match="no run exported"):
        export_dataset([{"run": str(tmp_path / "missing")}], tmp_path / "ds", workers=1)
    assert not (tmp_path / "ds").exists() and not (tmp_path / ".ds.staging").exists()
