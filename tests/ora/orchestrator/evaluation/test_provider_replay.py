"""Recorded replies and request comparison retain all bytes and deterministic image structure."""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ora.orchestrator.evaluation.provider_replay import Tape, compare_runs
from ora.orchestrator.evaluation.request_comparison import compare_requests


def run(root, requests, responses=()):
    root.mkdir()
    events = []
    for kind, values in (("provider_request", requests), ("provider_response", responses)):
        for i, value in enumerate(values):
            raw = json.dumps(value).encode()
            path = f"{kind}-{i}.json"
            (root / path).write_bytes(raw)
            events.append({"type": kind, "model_call_id": f"model-{i}", "observation_step": i,
                           "payload": {"accepted": True},
                           "artifact_ref": {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}})
    (root / "agent-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return root


def image(value):
    buf = io.BytesIO()
    Image.fromarray(value.astype(np.uint8)).save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def request(content):
    return json.dumps({"messages": [{"role": "user", "content": content}]}).encode()


def test_psnr_does_not_waive_labels_order_or_bytes():
    first = np.full((16, 16, 3), 80)
    noise = first.copy()
    noise[:8] += 1  # PSNR ~51.14 dB.
    before = request([{"type": "text", "text": "head"}, image(first)])
    after = request([{"type": "text", "text": "wrist"}, image(noise)])
    result = compare_requests(before, after)
    assert not result["byte_equal"] and not result["json_equal"]
    assert not result["images"][0]["label_equal"]
    assert 48 <= result["images"][0]["psnr_db"] <= 52
    assert result["images"][0]["classification"] == "rendering_noise_reference_48_52_db"
    assert result["images"][0]["pixels_above_16_fraction"] == 0
    large = compare_requests(before, request([{"type": "text", "text": "head"}, image(first + 80)]))
    assert large["images"][0]["classification"] == "below_rendering_noise_reference"
    missing = compare_requests(before, request([]))
    assert missing["image_counts"] == [1, 0] and missing["unmatched_images"]
    identical = compare_requests(before, before)
    assert identical["byte_equal"] and identical["images"][0]["pixels_equal"]


def test_tape_preserves_chunks_usage_and_consumes_once(tmp_path):
    chunks = [{"choices": [{"delta": {"tool_calls": [{"id": "original", "function": {"name": "move_to", "arguments": "{}"}}]}}]},
              {"choices": [], "usage": {"prompt_tokens": 123}}]
    root = run(tmp_path / "source", [], [chunks])
    tape = Tape(root, tmp_path / "out")
    raw = b'{"stream": true}'
    data, kind = tape.reply(raw)
    assert kind == "text/event-stream" and data.endswith(b"data: [DONE]\n\n")
    decoded = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith("data: {")]
    assert decoded == chunks
    assert (tmp_path / "out/request-0000.json").read_bytes() == raw
    with pytest.raises(ValueError, match="exhausted"):
        tape.reply(raw)
    with pytest.raises(ValueError, match="outside"):
        Tape(root, root / "mutate")


def test_complete_run_comparison_counts_missing_calls(tmp_path):
    a = run(tmp_path / "a", [{"messages": []}, {"messages": [{"content": "second"}]}])
    b = run(tmp_path / "b", [{"messages": []}])
    report = compare_runs(a, b, tmp_path / "diff")
    assert not report["byte_equal"] and report["requests"][0]["byte_equal"]
    assert report["unmatched_source_calls"] == ["model-1"]
    with pytest.raises(ValueError, match="outside"):
        compare_runs(a, b, a / "diff")


def test_comparison_reports_each_run_s_recorded_baseline(tmp_path):
    a = run(tmp_path / "old", [{"messages": []}])
    b = run(tmp_path / "new", [{"messages": []}])
    for path, identity, marker in ((a, "native_full_explicit_motion_direct_v4", True),
                                   (b, "a_retired_replay_baseline", False)):
        (path / "plan.json").write_text(json.dumps({"source": {"commit": "actual-checkout"}, "policy": {
            "baseline": {"id": identity, "sha256": "evidence"}, "require_live_marker": marker}}))
    report = compare_runs(a, b, tmp_path / "comparison")
    assert report["policy_configuration"]["candidate"]["baseline"]["id"] == "a_retired_replay_baseline"
    assert report["policy_configuration"]["candidate"]["implementation_commit"] == "actual-checkout"
    assert report["policy_configuration"]["candidate"]["require_live_marker"] is False
    assert report["byte_equal"]


@pytest.mark.parametrize("version", [1, 2, None])
def test_comparison_versions_are_warnings_not_provider_request_differences(tmp_path, version):
    source = run(tmp_path / "source", [{"messages": []}])
    candidate = run(tmp_path / "candidate", [{"messages": []}])
    if version is not None:
        (source / "toolkit-protocol.json").write_text(json.dumps({"protocol_version": version}))
    (candidate / "toolkit-protocol.json").write_text(json.dumps({"protocol_version": 1}))
    report = compare_runs(source, candidate, tmp_path / "comparison")
    assert report["byte_equal"]
    assert report["protocol_compatibility"]["source_version"] == version
    assert report["protocol_compatibility"]["current_version"] == 1
    assert report["candidate_protocol_compatibility"]["status"] == "matched"
    assert report["protocol_compatibility"]["status"] == {1: "matched", 2: "mismatch", None: "unknown"}[version]
