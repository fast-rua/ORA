"""One Synthetic episode over a real Anthropic Messages wire on the Pi core, against its golden.

A scripted loopback Anthropic server answers every call with a signed thinking
block, a sentence and one ``tool_use``. The Pi core runs the plan through its
TypeScript provider under each recovery policy; every request body the server
received, and its headers, must equal the golden of that policy
(``tests/support/agent_core/golden.py``), recorded from the Python Native
core's ``AnthropicMessagesProvider`` before ORA removed it. That also shows the
signed-thinking continuation is replayed exactly.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.support.agent_core import episodes, golden

pytestmark = pytest.mark.usefixtures("pi_core")

API_KEY = "sk-ant-local-episode-test-abcdefghij"
TOOLS = [
    ("observe", {"views": ["agentview"]}),
    ("close_gripper", {}),
    ("observe", {"views": ["wrist"]}),
    ("open_gripper", {}),
    ("finish_episode", {"outcome": "failure", "reason": "anthropic wire probe", "attempts": ["scripted"]}),
]


def sse(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


def reply(index: int) -> bytes:
    name, arguments = TOOLS[min(index, len(TOOLS) - 1)]
    text = json.dumps(arguments)
    events = [
        {"type": "message_start", "message": {"id": f"msg_{index:03d}", "type": "message", "role": "assistant",
                                              "content": [], "usage": {"input_tokens": 3000 + 700 * index,
                                                                       "cache_read_input_tokens": 1024 * index}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": f"Step {index}: "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": f"call {name}."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": f"EqQBsig{index:03d}AAAA"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": f"Calling {name}."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": f"toolu_{index:03d}",
                                                                       "name": name, "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": text[:3]}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": text[3:]}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 60 + index}},
        {"type": "message_stop"},
    ]
    return b"".join(sse(event) for event in events)


class Recorder:
    """Request bodies and headers per core (the first path segment of the endpoint)."""

    def __init__(self) -> None:
        self.calls: dict[str, list[dict[str, Any]]] = {}
        self.lock = threading.Lock()


def handler_for(recorder: Recorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            core = self.path.strip("/").split("/")[0]
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            with recorder.lock:
                calls = recorder.calls.setdefault(core, [])
                index = len(calls)
                calls.append({"sha256": hashlib.sha256(body).hexdigest(), "body": body,
                              "headers": {key: self.headers.get(key) for key in
                                          ("content-type", "anthropic-version", "x-api-key", "x-session-id")}})
            payload = reply(index)
            self.send_response(200)
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


def wire_episode(core: str, out: Path, recorder: Recorder, port: int, recovery: str) -> dict[str, Any]:
    """The episode on ``core`` against the server; the golden ``anthropic-wire-<recovery>``."""

    def configure(plan: dict[str, Any]) -> dict[str, str]:
        endpoint = f"http://127.0.0.1:{port}/{core}/v1"
        policy = plan["policy"]
        assert policy["provider"] == "anthropic_messages"
        policy.update({"endpoint": endpoint, "require_live_marker": False, "provider_recovery": recovery})
        if recovery != "legacy_v1":
            policy["provider_recovery_window_s"] = 120.0
        return {policy["base_url_env"]: endpoint, policy["api_key_env"]: API_KEY}

    # The remaining wall-time budget inside the observation text is a clock reading.
    with episodes.pinned_wall_time():
        run = episodes.run_core(core, out, configure=configure, run_id="anthropic-episode-compare")
    assert run["core"] == core
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    calls = recorder.calls.get(core, [])
    assert len(calls) == len(TOOLS)
    return golden.summarize([call["body"] for call in calls], headers=[call["headers"] for call in calls])


@pytest.mark.parametrize("recovery", ["legacy_v1", "same_credential_deadline_v1", "first_token_deadline_v1"])
def test_anthropic_wire_requests_equal_the_golden(tmp_path: Path, server: tuple[Recorder, int],
                                                  recovery: str) -> None:
    recorder, port = server
    summary = wire_episode("pi", tmp_path, recorder, port, recovery)
    # The recovery policy changes no request byte: both deadline policies share one golden.
    name = "legacy_v1" if recovery == "legacy_v1" else "same_credential_deadline_v1"
    assert golden.check(f"anthropic-wire-{name}", summary) == []
