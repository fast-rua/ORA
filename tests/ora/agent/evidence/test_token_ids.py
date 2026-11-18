"""Token ids in the wire export: from a vLLM stream, from an engine's /tokenize, checked against usage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ora.agent.evidence.canonical import ArtifactStore, TrajectoryError
from ora.agent.evidence.token_ids import (
    TokenChain, Tokenizer, ids_sha256, prompt_ids, stream_token_ids, tokenize_payload,
)
from ora.agent.evidence.wire_export import check_wire_export, export_wire
from tests.support.agent.agent_events import TrajectoryRecorder

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def _stream(prompt: list[int] | None, output: list[int] | None, usage: dict) -> list[dict]:
    """A vLLM stream with ``return_token_ids``: prompt ids in the first chunk, each delta's sampled ids."""
    first: dict = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    if prompt is not None:
        first.update(prompt_token_ids=prompt, prompt_text="<rendered>")
    chunks = [first]
    for position, token in enumerate(output or [7]):
        choice = {"index": 0, "delta": {"content": f"t{position}"}, "finish_reason": None}
        if output is not None:
            choice["token_ids"] = [token]
        chunks.append({"choices": [choice]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop",
                                **({"token_ids": []} if output is not None else {})}]})
    chunks.append({"choices": [], "usage": usage})
    return chunks


def _run(root: Path, calls: list[tuple[list, list[dict]]]) -> Path:
    """A run whose call ``i`` sends ``calls[i][0]`` as messages and archives ``calls[i][1]`` as its stream."""
    store = ArtifactStore(root)
    recorder = TrajectoryRecorder(root, "run", "session")
    for index, (messages, stream) in enumerate(calls):
        call_id = f"model-call-{index:06d}"
        wire = store.put_json({"model": "m", "stream": True, "messages": messages, "reasoning_effort": "high"},
                              kind="native_provider_wire_request")
        recorder.append("provider_request", model_call_id=call_id, artifact_ref=wire,
                        payload={"model": "m", "provider_protocol": "openai_chat_completions"})
        response = store.put_json(stream, kind="native_provider_wire_response")
        recorder.append("provider_response", model_call_id=call_id, artifact_ref=response,
                        payload={"accepted": True, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}})
    (root / "plan.json").write_text(json.dumps({"source": {"commit": "test"}, "policy": {}}))
    (root / "result.json").write_text(json.dumps({"run_id": "run", "official_success": True, "termination": "test"}))
    return root


def _user(text: str, *, image: bool = False) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, *([IMAGE] if image else [])]}


FIRST = [{"role": "system", "content": "s"}, _user("look")]
SECOND = FIRST + [{"role": "assistant", "content": "ok"}, _user("again")]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_streamed_ids_export_with_usage_checks_and_the_append_relation(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [
        (FIRST, _stream([1, 2, 3], [4, 5], {"prompt_tokens": 3, "completion_tokens": 2})),
        (SECOND, _stream([1, 2, 3, 4, 5, 6], [8], {"prompt_tokens": 6, "completion_tokens": 1})),
    ])
    destination = export_wire(run, tmp_path / "wire")
    first, second = _rows(destination / "tokens.jsonl")
    assert first["prompt"] == {"source": "stream", "length": 3, "sha256": ids_sha256([1, 2, 3]), "images": 0,
                               "image_form": "expanded", "prefix": None, "suffix": [1, 2, 3]}
    assert first["output"]["ids"] == [4, 5] and first["checks"]["output_length"] is True
    # The second prompt is the first, its sampled tokens and one more: stored as a prefix and a suffix.
    assert second["prompt"]["prefix"] == {"model_call_id": "model-call-000000", "length": 3}
    assert second["prompt"]["suffix"] == [4, 5, 6] and second["extends_previous"] is True
    assert [ids for _, ids in prompt_ids([first, second])] == [[1, 2, 3], [1, 2, 3, 4, 5, 6]]
    tokens = json.loads((destination / "manifest.json").read_text())["tokens"]
    assert tokens["prompt_ids"] == {"stream": 2} and tokens["output_ids"] == 2
    assert tokens["prompt_length"] == {"true": 2} and tokens["extends_previous"] == {"none": 1, "true": 1}
    report = check_wire_export(destination, run)
    assert report["identical"], report["failures"]
    assert report["tokens"] == 2


def test_a_length_that_differs_from_usage_is_recorded_not_dropped(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [(FIRST, _stream([1, 2, 3], [4], {"prompt_tokens": 4, "completion_tokens": 2}))])
    destination = export_wire(run, tmp_path / "wire")
    (row,) = _rows(destination / "tokens.jsonl")
    assert row["checks"] == {"prompt_length": False, "output_length": False, "tokenize_equals_stream": None}
    assert row["prompt"]["image_form"] is None
    assert check_wire_export(destination, run)["identical"]


def _tokenizer(answers: dict[str, list[int]], seen: list[dict]) -> Tokenizer:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"body": body, "authorization": request.headers.get("authorization")})
        return httpx.Response(200, json={"tokens": answers[body["messages"][-1]["content"][0]["text"]],
                                         "count": 0, "max_model_len": 9})
    return Tokenizer("https://engine.invalid/v1/tokenize", "test-secret-key",
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_tokenize_gives_prompt_ids_with_one_placeholder_per_image(tmp_path: Path) -> None:
    # sglang tokenizes before it expands images: one placeholder token per image, usage counts image_tokens.
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": {"image_tokens": 10}}
    run = _run(tmp_path / "run", [([{"role": "system", "content": "s"}, _user("look", image=True)],
                                   _stream(None, None, usage))])
    seen: list[dict] = []
    destination = export_wire(run, tmp_path / "wire", tokenizer=_tokenizer({"look": [1, 99, 2]}, seen))
    (row,) = _rows(destination / "tokens.jsonl")
    assert row["prompt"]["source"] == "tokenize" and row["prompt"]["image_form"] == "placeholder"
    assert row["checks"]["prompt_length"] is True and row["output"] is None
    # Only the chat-template inputs go to the engine, with the key as a bearer token; the effort goes where the
    # chat endpoint renders it from.
    assert set(seen[0]["body"]) == {"model", "messages", "chat_template_kwargs"}
    assert seen[0]["body"]["chat_template_kwargs"] == {"reasoning_effort": "high", "enable_thinking": True}
    assert seen[0]["authorization"] == "Bearer test-secret-key"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["tokens"]["tokenizer"]["url"] == "https://engine.invalid/v1/tokenize"
    assert "test-secret-key" not in (destination / "manifest.json").read_text()
    # The check cannot ask the engine again; it rebuilds the rows from the recorded ids.
    report = check_wire_export(destination, run)
    assert report["identical"], report["failures"]


def test_tokenize_next_to_streamed_ids_records_whether_they_agree(tmp_path: Path) -> None:
    usage = {"prompt_tokens": 3, "completion_tokens": 1}
    run = _run(tmp_path / "run", [(FIRST, _stream([1, 2, 3], [4], usage)),
                                  (SECOND, _stream([1, 2, 3, 4, 6], [5], {**usage, "prompt_tokens": 5}))])
    destination = export_wire(run, tmp_path / "wire",
                              tokenizer=_tokenizer({"look": [1, 2, 3], "again": [1, 2, 3, 4, 7]}, []))
    first, second = _rows(destination / "tokens.jsonl")
    assert first["checks"]["tokenize_equals_stream"] is True and "tokenized" not in first["prompt"]
    assert second["checks"]["tokenize_equals_stream"] is False
    assert second["prompt"]["source"] == "stream" and second["prompt"]["tokenized"]["ids"] == [1, 2, 3, 4, 7]
    assert check_wire_export(destination, run)["identical"]


def test_an_edited_token_row_fails_the_check(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [
        (FIRST, _stream([1, 2, 3], [4], {"prompt_tokens": 3, "completion_tokens": 1})),
        (SECOND, _stream([1, 2, 3, 4, 5], [6], {"prompt_tokens": 5, "completion_tokens": 1})),
    ])
    destination = export_wire(run, tmp_path / "wire")
    rows = _rows(destination / "tokens.jsonl")
    rows[1]["prompt"]["suffix"] = [4, 9]
    (destination / "tokens.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((destination / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == "tokens.jsonl":
            data = (destination / "tokens.jsonl").read_bytes()
            entry["sha256"], entry["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
    (destination / "manifest.json").write_text(json.dumps(manifest))
    report = check_wire_export(destination)
    assert not report["identical"]
    assert {failure["check"] for failure in report["failures"]} == {"tokens"}


def test_a_broken_prefix_chain_is_refused() -> None:
    chain = TokenChain()
    first = chain.row("a", {"messages": []}, _stream([1, 2], [3], {"prompt_tokens": 2}), None)
    second = chain.row("b", {"messages": []}, _stream([1, 2, 3], [4], {"prompt_tokens": 3}), None)
    second["prompt"]["prefix"]["model_call_id"] = "z"
    with pytest.raises(TrajectoryError, match="does not chain"):
        list(prompt_ids([first, second]))


def test_stream_token_ids_reads_streams_and_bodies() -> None:
    assert stream_token_ids(_stream([1], [2, 3], {})) == ([1], [2, 3])
    assert stream_token_ids(_stream(None, None, {})) == (None, None)
    # vLLM without return_token_ids: null ids in the first chunk and in every delta mean no ids, not empty ones.
    unasked = _stream(None, None, {})
    unasked[0].update(prompt_token_ids=None, prompt_text=None)
    for chunk in unasked:
        for choice in chunk["choices"]:
            choice["token_ids"] = None
    assert stream_token_ids(unasked) == (None, None)
    body = {"prompt_token_ids": [5, 6], "choices": [{"message": {"content": "x"}, "token_ids": [7]}]}
    assert stream_token_ids(body) == ([5, 6], [7])


@pytest.mark.parametrize(("body", "kwargs"), [
    ({"reasoning_effort": "low"}, {"reasoning_effort": "low", "enable_thinking": True}),
    ({"reasoning_effort": "none"}, {"reasoning_effort": "none", "enable_thinking": False}),
    ({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "max"}},
     {"reasoning_effort": "low", "enable_thinking": False}),
    ({"reasoning_effort": "auto", "chat_template_kwargs": {"enable_thinking": True}}, {"enable_thinking": True}),
    ({}, None),
])
def test_the_effort_is_rendered_from_the_template_kwargs_as_the_chat_endpoint_does(body, kwargs) -> None:
    payload = tokenize_payload({"model": "m", "messages": [], "temperature": 1.0, "tool_choice": "auto", **body})
    assert payload.get("chat_template_kwargs") == kwargs
    assert set(payload) <= {"model", "messages", "tools", "chat_template_kwargs"}
