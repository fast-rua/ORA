"""A compactor bound to its own Provider profile, over real loopback wires, on the Pi agent core.

One loopback server plays two models. ``/a`` speaks Anthropic Messages for the
policy (``model-a``, key A, scripted tool calls); its replies report a provider
input above the emergency watermark from the fifth call on, so the lossless
retention writes checkpoints. ``/b`` speaks OpenAI chat completions for the
compactor (``model-b``, key B), bound by ``policy.context_compactor`` to a
catalog profile with session affinity, and answers each call with a valid
checkpoint. Its base URL variable is left unset: the binding's endpoint serves.

Bound: every compaction request reached ``/b`` with model B, key B and the
compactor's own session id, and nothing else did; no policy request reached
``/b`` or carried key B; key B is written nowhere under the run root and the Pi
core got it only by environment; every compactor event names the binding; the
run evaluates valid. The bodies sent to both endpoints and the events equal
the golden ``compactor-bound`` (``tests/support/agent_core/golden.py``),
recorded from the Python Native core before ORA removed it.

Unbound (the default), the compactor stays on ``/a`` with model A, and the plan,
the Pi core's InitParams and every event carry no binding key at all.
"""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from tests.support.agent_core import episodes, golden
from tests.support.agent_core.replay.runtime import default_checkpoint

pytestmark = pytest.mark.usefixtures("pi_core")

KEY_A = "sk-ant-fake-policy-model-a-0123456789"
KEY_B = "sk-fake-compactor-model-b-9876543210"
B_KEY_ENV, B_URL_ENV = "ORA_TEST_COMPACTOR_B_API_KEY", "ORA_TEST_COMPACTOR_B_BASE_URL"
RUN_ID = "compactor-binding-compare"
ACTIONS = 6
INFLATED_FROM_CALL = 4
TOOLS = [
    *[step for index in range(ACTIONS) for step in (
        ("observe", {"views": ["agentview"]}),
        ("close_gripper" if index % 2 == 0 else "open_gripper", {}),
    )],
    ("finish_episode", {"outcome": "failure", "reason": "compactor binding probe", "attempts": ["scripted"]}),
]


def sse(event: dict[str, Any], *, named: bool) -> bytes:
    prefix = f"event: {event['type']}\n" if named else ""
    return f"{prefix}data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


def checkpoint(index: int) -> str:
    return json.dumps(default_checkpoint(index + 1), ensure_ascii=False, separators=(",", ":"))


def anthropic_reply(index: int, blocks: list[dict[str, Any]], *, input_tokens: int, stop: str) -> bytes:
    events: list[dict[str, Any]] = [{"type": "message_start", "message": {
        "id": f"msg_{index:03d}", "type": "message", "role": "assistant", "model": "model-a", "content": [],
        "usage": {"input_tokens": input_tokens}}}]
    for position, block in enumerate(blocks):
        events += [{"type": "content_block_start", "index": position, "content_block": block["start"]},
                   {"type": "content_block_delta", "index": position, "delta": block["delta"]},
                   {"type": "content_block_stop", "index": position}]
    events += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 40 + index}},
               {"type": "message_stop"}]
    return b"".join(sse(event, named=True) for event in events)


def policy_reply(index: int) -> bytes:
    """Model A: one tool call; from the fifth call on, a provider input above the L6 watermark."""
    name, arguments = TOOLS[min(index, len(TOOLS) - 1)]
    return anthropic_reply(index, [
        {"start": {"type": "text", "text": ""}, "delta": {"type": "text_delta", "text": f"Calling {name}."}},
        {"start": {"type": "tool_use", "id": f"toolu_{index:03d}", "name": name, "input": {}},
         "delta": {"type": "input_json_delta", "partial_json": json.dumps(arguments)}},
    ], input_tokens=230000 if index >= INFLATED_FROM_CALL else 3000 + 700 * index, stop="tool_use")


def policy_checkpoint_reply(index: int) -> bytes:
    """Model A answering an unbound compactor's tool-free call."""
    return anthropic_reply(index, [{"start": {"type": "text", "text": ""},
                                    "delta": {"type": "text_delta", "text": checkpoint(index)}}],
                           input_tokens=40000, stop="end_turn")


def compactor_reply(index: int) -> bytes:
    """Model B: one streamed checkpoint, then its usage."""
    chunk = {"id": f"cmpl-{index:03d}", "object": "chat.completion.chunk", "model": "model-b"}
    events = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": checkpoint(index)},
                               "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": {"prompt_tokens": 40000, "completion_tokens": 300, "total_tokens": 40300}},
    ]
    return b"".join(sse(event, named=False) for event in events) + b"data: [DONE]\n\n"


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def on(self, endpoint: str, *, tools: bool | None = None) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["endpoint"] == endpoint
                and (tools is None or ("tools" in json.loads(call["body"])) == tools)]


def handler_for(recorder: Recorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            endpoint = self.path.strip("/").split("/")[0]
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            with_tools = "tools" in json.loads(body)
            with recorder.lock:
                index = len(recorder.on(endpoint, tools=with_tools))
                recorder.calls.append({"endpoint": endpoint, "path": self.path, "body": body, "headers": {
                    key: self.headers.get(key) for key in ("x-api-key", "authorization", "x-session-id")}})
            if endpoint == "b":
                payload = compactor_reply(index)
            elif endpoint == "a":
                payload = (policy_reply if with_tools else policy_checkpoint_reply)(index)
            else:
                payload = b""
            self.send_response(200 if payload else 404)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    return Handler


@pytest.fixture
def server() -> Iterator[tuple[Recorder, int]]:
    recorder = Recorder()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(recorder))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield recorder, httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def init_params(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The InitParams every Pi core of the test received."""
    from ora.agent.pi.policy_session import PiPlanner

    captured: list[dict[str, Any]] = []
    start = PiPlanner.start

    def recording_start(self: Any) -> None:
        started = self.started
        start(self)
        if not started:
            captured.append(copy.deepcopy(self.init_params))

    monkeypatch.setattr(PiPlanner, "start", recording_start)
    return captured


def compactor_catalog(path: Path, endpoint: str) -> Path:
    path.write_text(yaml.safe_dump({"schema": "ora.native-provider-catalog.v1", "profiles": {"compactor-b": {
        "protocol": "openai_compatible", "endpoint": endpoint, "endpoint_transport": "loopback_http_v1",
        "model": "model-b", "model_env": "ORA_TEST_COMPACTOR_B_MODEL", "base_url_env": B_URL_ENV,
        "api_key_env": B_KEY_ENV, "proxy_env": [], "reasoning_effort": "low", "context_window_tokens": 262144,
        "max_output_tokens": 16384, "stream": True, "supports_images": True,
        "session_affinity": "x_session_id_run_v1",
        "qualification_schema": "ora.native-provider-probe.v1", "qualification_version": 1,
    }}}), encoding="utf-8")
    return path


def run(core: str, out: Path, port: int, *, bound: bool) -> dict[str, Any]:
    endpoint_a, endpoint_b = f"http://127.0.0.1:{port}/a/v1", f"http://127.0.0.1:{port}/b/v1"
    overrides: dict[str, Any] = {"model": "model-a", "endpoint": endpoint_a, "endpoint_transport": "loopback_http_v1",
                                 "require_live_marker": False}
    if bound:
        catalog = compactor_catalog(out / f"{core}-catalog.yaml", endpoint_b)
        overrides["context_compactor"] = {"provider_profile": "compactor-b", "provider_catalog": str(catalog)}

    def configure(plan: dict[str, Any]) -> dict[str, str | None]:
        policy = plan["policy"]
        return {policy["base_url_env"]: endpoint_a, policy["api_key_env"]: KEY_A, B_KEY_ENV: KEY_B, B_URL_ENV: None}

    # The remaining wall time reaches the checkpoints' exact-transcript hashes: pin it.
    with episodes.pinned_wall_time():
        return episodes.run_core(core, out, actions=ACTIONS, configure=configure, run_id=RUN_ID,
                                 policy_overrides=overrides)


def compactor_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in run["events"] if event["type"].startswith("context_compactor_")]


def bound_summary(core: str, out: Path, recorder: Recorder, port: int) -> dict[str, Any]:
    """The bound run on ``core``: the golden ``compactor-bound`` (the port and ``out`` normalized)."""

    endpoint_b = f"http://127.0.0.1:{port}/b/v1"
    binding = {"provider_profile": "compactor-b", "model": "model-b", "endpoint": endpoint_b}
    recorder.calls.clear()
    outcome = run(core, out, port, bound=True)
    assert outcome["core"] == core
    assert outcome["evaluation"]["status"] == "valid", outcome["evaluation"].get("reasons")
    assert {call["endpoint"] for call in recorder.calls} == {"a", "b"}, [call["path"] for call in recorder.calls]
    policy_calls, compactor_calls = recorder.on("a"), recorder.on("b")
    responses = [event for event in outcome["events"] if event["type"] == "context_compactor_response"]
    assert len(compactor_calls) == len(responses) > 0
    for call in compactor_calls:  # every compaction request: model B, key B, its own session id
        body = json.loads(call["body"])
        assert call["path"] == "/b/v1/chat/completions" and body["model"] == "model-b" and "tools" not in body
        assert call["headers"] == {"x-api-key": None, "authorization": f"Bearer {KEY_B}",
                                   "x-session-id": f"ora:{RUN_ID}:compactor"}
    for call in policy_calls:  # no policy request reached /b or carried key B; none was tool-free
        body = json.loads(call["body"])
        assert body["model"] == "model-a" and body["tools"]
        assert call["headers"]["x-api-key"] == KEY_A and KEY_B.encode() not in call["body"]
        assert KEY_B not in json.dumps(call["headers"])
    events = compactor_events(outcome)
    assert events and all(event["payload"]["provider_binding"] == binding for event in events)
    attempts = [event["payload"] for event in events if event["type"] == "context_compactor_attempt"]
    assert {(item["model"], item["provider_protocol"]) for item in attempts} == {
        ("model-b", "openai_chat_completions")}
    written = [path for path in outcome["root"].rglob("*") if path.is_file() and KEY_B.encode() in path.read_bytes()]
    assert written == [], written

    def normal(data: bytes) -> bytes:
        return data.replace(f"127.0.0.1:{port}".encode(), b"127.0.0.1:PORT").replace(str(out).encode(), b"OUT")

    return golden.summarize(
        [normal(call["body"]) for call in policy_calls + compactor_calls],
        json.loads(normal(json.dumps(outcome["events"]).encode())),
        policy_requests=len(policy_calls), compactor_requests=len(compactor_calls))


def test_the_bound_compactor_alone_reaches_its_endpoint(
    tmp_path: Path, server: tuple[Recorder, int], init_params: list[dict[str, Any]],
):
    recorder, port = server
    endpoint_b = f"http://127.0.0.1:{port}/b/v1"
    summary = bound_summary("pi", tmp_path, recorder, port)
    # The Pi core got the compactor's spec without its key (that travels by environment).
    (params,) = init_params
    assert params["compactor_provider"]["endpoint"] == endpoint_b and "api_key" not in params["compactor_provider"]
    assert params["compactor_provider"]["session_affinity_id"] == f"ora:{RUN_ID}:compactor"
    assert KEY_B not in json.dumps(params, default=str)
    assert golden.check("compactor-bound", summary) == []


def test_without_a_binding_the_compactor_stays_on_the_policy_provider_and_no_key_is_written(
    tmp_path: Path, server: tuple[Recorder, int], init_params: list[dict[str, Any]],
):
    recorder, port = server
    for core in ("pi",):
        recorder.calls.clear()
        outcome = run(core, tmp_path, port, bound=False)
        assert outcome["evaluation"]["status"] == "valid", outcome["evaluation"].get("reasons")
        plan = json.loads((outcome["root"] / "plan.json").read_text(encoding="utf-8"))
        assert "context_compactor" not in plan["policy"]
        assert {call["endpoint"] for call in recorder.calls} == {"a"}
        compactor_calls = recorder.on("a", tools=False)
        assert len(compactor_calls) == len([event for event in outcome["events"]
                                            if event["type"] == "context_compactor_response"]) > 0
        assert {json.loads(call["body"])["model"] for call in compactor_calls} == {"model-a"}
        assert all(call["headers"]["x-api-key"] == KEY_A for call in compactor_calls)
        assert compactor_events(outcome)
        assert all("provider_binding" not in json.dumps(event) for event in outcome["events"])
        session = json.loads((outcome["root"] / "native/agent-session.json").read_text(encoding="utf-8"))
        assert "context_compactor" not in session and "provider_binding" not in json.dumps(session)
    (params,) = init_params
    assert "compactor_provider" not in params and "context_compactor" not in params["policy"]
