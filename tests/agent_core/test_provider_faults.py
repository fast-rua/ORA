"""Model and provider faults on the Pi agent core, end to end.

Each test runs one Synthetic episode on the Pi core, whose provider runs in
TypeScript, against a FakeProvider script or a loopback endpoint scripted per
request, and checks what the core admitted, dispatched and recorded, and what
the agent-trace verifier makes of the run:

* a tool call whose arguments are not JSON, or that names no function, is
  refused with a retryable error the next request carries, is never dispatched,
  and its proposal and its refusal name the same tool;
* a reply the output limit cut off is rejected with its whole tool batch, and
  the request is asked again without it, within the observation's call budget;
* a stream that breaks after content is asked again as the same request, each
  broken attempt recorded against the exact bytes sent; the trace verifier
  accepts it, names the request no attempt answered, and rejects an attempt
  record that disagrees with its request;
* the round's wall-time deadline is set when the turn starts: a read-only tool
  continuation keeps it.

They replace the Native-core cases of the provider tests
(``tests/ora/agent/providers``), which exercised the same contracts through the
Python provider loop.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pytest

from ora.agent.evidence.canonical import canonical_bytes
from ora.agent.evidence.trace_verify import verify_agent_trace
from tests.support.agent_core import episodes
from tests.support.agent_core.episodes import tool_reply

pytestmark = pytest.mark.usefixtures("pi_core")

API_KEY = "offline-primary-key"
#: ora/agent/providers/stream_failures.py (ora-pi/src/provider/stream_failures.ts).
STREAM_BREAK_RETRIES = 3
FINISH = {"outcome": "failure", "reason": "provider fault probe", "attempts": ["scripted"]}

Reply = Callable[[dict[str, Any]], Mapping[str, Any]]


# -- A scripted loopback endpoint ----------------------------------------------------------------


class Endpoint:
    """Answer request ``i`` with ``replies[i](body)`` (the last reply repeats).

    A reply is ``{"json": ...}`` (with ``status`` and ``headers``), or
    ``{"sse": [event, ...]}``, sent as one chunked event stream; with
    ``"break": True`` the socket closes after those events, before the stream
    ends. ``"delay": seconds`` holds the reply. Every request's raw body and
    arrival time is kept.
    """

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = replies
        self.bodies: list[bytes] = []
        self.times: list[float] = []
        self.lock = threading.Lock()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("content-length", "0")))
                with endpoint.lock:
                    index = len(endpoint.bodies)
                    endpoint.bodies.append(body)
                    endpoint.times.append(time.monotonic())
                reply = endpoint.replies[min(index, len(endpoint.replies) - 1)](json.loads(body))
                time.sleep(float(reply.get("delay", 0)))
                if "sse" in reply:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    for event in reply["sse"]:
                        self._chunk(event)
                    if reply.get("break"):
                        self.wfile.flush()
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.wfile.write(b"0\r\n\r\n")
                    return
                payload = json.dumps(reply["json"]).encode()
                self.send_response(int(reply.get("status", 200)))
                for name, value in (reply.get("headers") or {}).items():
                    self.send_header(name, value)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _chunk(self, event: Any) -> None:
                if isinstance(event, bytes):
                    data = event
                elif isinstance(event, tuple):  # (sse event name, data): Anthropic's framing
                    data = f"event: {event[0]}\ndata: {json.dumps(event[1])}\n\n".encode()
                else:
                    data = f"data: {json.dumps(event)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def requests(self) -> list[dict[str, Any]]:
        return [json.loads(body) for body in self.bodies]

    def __enter__(self) -> "Endpoint":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def openai_policy(port: int, **policy: Any) -> Callable[[dict[str, Any]], dict[str, str]]:
    """``configure`` for ``episodes.run_core``: the plan's provider is this loopback Chat Completions endpoint."""

    def configure(plan: dict[str, Any]) -> dict[str, str]:
        endpoint = f"http://127.0.0.1:{port}/v1"
        values = plan["policy"]
        values.update({"provider": "openai_compatible", "endpoint": endpoint, "require_live_marker": False,
                       **policy})
        return {values["base_url_env"]: endpoint, values["api_key_env"]: API_KEY,
                f"{values['api_key_env']}S": None}
    return configure


def anthropic_policy(port: int, *, standby: str, **policy: Any) -> Callable[[dict[str, Any]], dict[str, str]]:
    def configure(plan: dict[str, Any]) -> dict[str, str]:
        endpoint = f"http://127.0.0.1:{port}"
        values = plan["policy"]
        assert values["provider"] == "anthropic_messages"
        values.update({"endpoint": endpoint, "require_live_marker": False, **policy})
        return {values["base_url_env"]: endpoint, values["api_key_env"]: API_KEY,
                f"{values['api_key_env']}S": standby}
    return configure


def wire_tool(body: Mapping[str, Any], name: str) -> str:
    """The name ``name`` is sent under on this request's wire (OpenAI aliases dotted names)."""

    for tool in body.get("tools", []):
        wire = tool["function"]["name"] if "function" in tool else tool["name"]
        if wire == name or wire.startswith(name.replace(".", "_")):
            return wire
    raise AssertionError(f"{name} is not offered on this request")


def chat_tool_call(call_id: str, name: str, arguments: Any, *, finish: str = "tool_calls",
                   content: str = "", reasoning: str | None = None, output_tokens: int = 20) -> Reply:
    """One Chat Completions reply calling ``name`` (streamed when the request streams)."""

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        text = arguments if isinstance(arguments, str) else json.dumps(arguments)
        call = {"index": 0, "id": call_id, "type": "function",
                "function": {"name": wire_tool(body, name), "arguments": text}}
        usage = {"prompt_tokens": 100, "completion_tokens": output_tokens, "total_tokens": 100 + output_tokens}
        message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": [call]}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        if body.get("stream", True):
            return {"sse": [{"choices": [{"index": 0, "delta": message, "finish_reason": finish}], "usage": usage},
                            b"data: [DONE]\n\n"]}
        return {"json": {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}}
    return reply


def events_of(run: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in run["events"] if event["type"] == kind]


def rechain(root: Path, events: list[dict[str, Any]]) -> None:
    """Rewrite the trace with its hash chain recomputed (a tampered trace that still chains)."""

    previous = None
    for event in events:
        event["previous_event_sha256"] = previous
        event["event_sha256"] = hashlib.sha256(canonical_bytes(
            {key: value for key, value in event.items() if key != "event_sha256"})).hexdigest()
        previous = event["event_sha256"]
    (root / "agent-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events),
                                            encoding="utf-8")


# -- Malformed tool calls ------------------------------------------------------------------------

#: A read-only call with trailing garbage, a call naming no function, a mutation with bad JSON.
BAD_CALLS = {
    "observe": {"name": "observe", "arguments": '{"views": ["agentview"]}}'},
    None: {"arguments": "{}"},
    "close_gripper": {"name": "close_gripper", "arguments": "{}}"},
}


@pytest.mark.parametrize("signed", [False, True], ids=["scripted", "signed-loopback"])
@pytest.mark.parametrize("name", list(BAD_CALLS), ids=["read-only", "nameless", "mutation"])
def test_a_malformed_tool_call_is_refused_on_the_next_request_and_never_dispatched(
    tmp_path: Path, name: str | None, signed: bool,
) -> None:
    function = BAD_CALLS[name]
    bad = {"role": "assistant", "content": "", "tool_calls": [{"id": "bad-json", "type": "function",
                                                               "function": dict(function)}]}
    if not signed:
        script = [{"message": bad, "finish_reason": "tool_calls"},
                  tool_reply(("act", "close_gripper", {})),
                  tool_reply(("end", "finish_episode", FINISH))]
        run = episodes.run_core("pi", tmp_path, responses=script, actions=1)
        requests = [json.loads(request) for request in run["requests"]]
    else:
        def first(body: dict[str, Any]) -> dict[str, Any]:
            # A reasoning model's reply: the provider binds its reasoning to the call it made.
            message = {**bad, "reasoning_content": "Compute."}
            return {"json": {"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 10}}}

        with Endpoint([first, chat_tool_call("act", "close_gripper", {}),
                       chat_tool_call("end", "finish_episode", FINISH)]) as endpoint:
            run = episodes.run_core("pi", tmp_path, configure=openai_policy(endpoint.port, stream=False),
                                    actions=1)
        requests = endpoint.requests()
        if name is not None:
            # Arguments that do not decode to an object reject the whole reply at the provider, as on the
            # Anthropic wire and for an output-limited batch: neither the call nor the reasoning bound to it is
            # replayed (strict servers refuse a replayed malformed call with HTTP 400), and the next request asks
            # again on the same observation.
            assert run["core"] == "pi" and run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
            for message in requests[1]["messages"]:
                assert "bad-json" not in {call.get("id") for call in message.get("tool_calls") or []}
                assert message.get("tool_call_id") != "bad-json"
                assert message.get("reasoning_content") != "Compute."
            assert "invalid tool-input JSON" in json.dumps(requests[1]["messages"])
            response = events_of(run, "provider_response")[0]["payload"]
            assert response["accepted"] is False
            assert response["output_rejection"]["code"] == "invalid_tool_input_json"
            assert [block["reason"] for block in response["output_rejection"]["rejected_tool_blocks"]] == ["invalid_json"]
            assert "bad-json" not in {event["tool_call_id"] for event in events_of(run, "tool_call_proposed")}
            assert [event["tool_call_id"] for event in events_of(run, "action_proposed")] == ["act"]
            return
        # A signed nameless call (valid arguments) is replayed exactly: its call and its reasoning.
        replayed = next(m for m in requests[1]["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
        assert replayed["tool_calls"][0]["function"]["arguments"] == function["arguments"]
        assert replayed["reasoning_content"] == "Compute."
    assert run["core"] == "pi" and run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    reply = next(m for m in requests[1]["messages"] if m.get("tool_call_id") == "bad-json")
    result = json.loads(reply["content"])
    assert result["error"]["retryable"] is True
    assert result["error"]["code"] == ("invalid_arguments" if name else "invalid_tool_call")
    assert reply["name"] == (name or "invalid_tool_call")
    # The proposal and its refusal name one tool, or the ActionStep inventory refuses the pair.
    proposed = next(event for event in events_of(run, "tool_call_proposed") if event["tool_call_id"] == "bad-json")
    refused = next(event for event in events_of(run, "tool_result") if event["tool_call_id"] == "bad-json")
    assert proposed["payload"]["name"] == refused["payload"]["name"] == (name or "invalid_tool_call")
    assert refused["payload"]["ok"] is False
    # The call itself stays as the model sent it, in the trace and in the history the next
    # request carries (a provider reports a missing name as an empty one).
    sent = function if name is not None or not signed else {"name": "", **function}
    assert events_of(run, "provider_response")[0]["payload"]["message"]["tool_calls"][0]["function"] == sent
    prior = next(m for m in requests[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls"))
    assert prior["tool_calls"][0]["function"] == sent
    # Only the well-formed close_gripper was dispatched.
    assert [event["tool_call_id"] for event in events_of(run, "action_proposed")] == ["act"]


# -- Output-limited replies ----------------------------------------------------------------------


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("complete_json", [True, False])
def test_an_output_limited_tool_batch_is_rejected_whole_and_asked_again(
    tmp_path: Path, stream: bool, complete_json: bool,
) -> None:
    def capped(body: dict[str, Any]) -> dict[str, Any]:
        tool = wire_tool(body, "close_gripper")
        calls = [{"index": index, "id": call_id, "type": "function",
                  "function": {"name": tool, "arguments": text}}
                 for index, (call_id, text) in enumerate((("discard-complete", "{}"),
                                                          ("discard-tail", "{}" if complete_json else '{"x":')))]
        message = {"role": "assistant", "content": "Prepare a move.",
                   "reasoning_content": "unfinished private reasoning", "tool_calls": calls}
        usage = {"prompt_tokens": 100, "completion_tokens": 8192, "total_tokens": 8292}
        if body.get("stream", True):
            return {"sse": [{"choices": [{"index": 0, "delta": message, "finish_reason": "length"}],
                             "usage": usage}, b"data: [DONE]\n\n"]}
        return {"json": {"choices": [{"index": 0, "message": message, "finish_reason": "length"}], "usage": usage}}

    with Endpoint([capped, chat_tool_call("accepted-after-cap", "close_gripper", {}),
                   chat_tool_call("end", "finish_episode", FINISH)]) as endpoint:
        run = episodes.run_core("pi", tmp_path, configure=openai_policy(endpoint.port, stream=stream), actions=1)
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    requests = endpoint.requests()
    # The second request carries no part of the rejected reply, and says why.
    retried = requests[1]["messages"]
    assert not [m for m in retried if m["role"] == "assistant"]
    assert any("entire tool batch was discarded" in json.dumps(m.get("content")) for m in retried)
    responses = events_of(run, "provider_response")
    assert [event["payload"]["accepted"] for event in responses][:2] == [False, True]
    rejected = responses[0]["payload"]["output_rejection"]
    assert rejected["code"] == "truncated_tool_call" and rejected["discarded_tool_block_count"] == 2
    raw = (run["root"] / responses[0]["artifact_ref"]["path"]).read_text(encoding="utf-8")
    assert "discard-tail" in raw and "unfinished private reasoning" in raw
    proposals = [event["tool_call_id"] for event in events_of(run, "tool_call_proposed")]
    assert proposals[0] == "accepted-after-cap" and not any(call.startswith("discard") for call in proposals)


def test_the_retry_of_an_output_limited_batch_spends_a_model_call(tmp_path: Path) -> None:
    def capped(_body: dict[str, Any]) -> dict[str, Any]:
        return {"json": {"choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": "", "tool_calls": [{"id": "never-dispatch", "type": "function",
                                                                "function": {"name": "close_gripper",
                                                                             "arguments": '{"x":'}}]}}]}}

    def one_call(plan: dict[str, Any]) -> dict[str, str]:
        plan["budgets"]["max_model_calls"] = 1
        plan["policy"]["max_model_calls_per_observation"] = 1
        return configure(plan)

    with Endpoint([capped]) as endpoint:
        configure = openai_policy(endpoint.port, stream=False)
        run = episodes.run_core("pi", tmp_path, configure=one_call, actions=1)
    assert len(endpoint.bodies) == 1 and len(run["requests"]) == 1
    assert run["result"]["termination"] == "model_call_budget"
    assert not events_of(run, "tool_call_proposed") and not events_of(run, "action_proposed")
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")


# -- Streams that break after content ------------------------------------------------------------


def broken_chat_stream(call_id: str) -> Reply:
    """A Chat Completions stream cut after a reasoning delta and a complete tool call."""

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        call = {"index": 0, "id": call_id, "type": "function",
                "function": {"name": wire_tool(body, "close_gripper"), "arguments": "{}"}}
        return {"sse": [{"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "H"},
                                      "finish_reason": None}]},
                        {"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}],
                "break": True}
    return reply


@pytest.mark.parametrize(("answered", "output_token_parameter", "recovery"),
                         [(True, "max_completion_tokens", "same_credential_deadline_v1"),
                          (False, "max_tokens", "same_credential_deadline_v1"),
                          (True, "max_completion_tokens", "first_token_deadline_v1"),
                          (False, "max_tokens", "first_token_deadline_v1")])
def test_every_broken_attempt_joins_the_request_a_later_attempt_answered(
    tmp_path: Path, answered: bool, output_token_parameter: str, recovery: str,
) -> None:
    breaks = STREAM_BREAK_RETRIES + int(not answered)
    replies = [broken_chat_stream("discarded")] * breaks + [
        chat_tool_call("accepted", "close_gripper", {}), chat_tool_call("end", "finish_episode", FINISH)]
    with Endpoint(replies) as endpoint:
        run = episodes.run_core("pi", tmp_path, actions=1, configure=openai_policy(
            endpoint.port, provider_recovery=recovery, provider_recovery_window_s=120.0,
            output_token_parameter=output_token_parameter))
    root = run["root"]
    request = events_of(run, "provider_request")[0]
    wire = (root / request["artifact_ref"]["path"]).read_bytes()
    first_call = endpoint.bodies[:STREAM_BREAK_RETRIES + 1]
    assert len(first_call) == STREAM_BREAK_RETRIES + 1
    # The recorded wire request is the exact body every attempt sent.
    assert set(first_call) == {wire}
    assert output_token_parameter in json.loads(wire)
    failed = events_of(run, "provider_attempt_failed")
    assert len(failed) == (STREAM_BREAK_RETRIES if answered else STREAM_BREAK_RETRIES + 1)
    assert {event["payload"]["request_sha256"] for event in failed} == {request["artifact_ref"]["sha256"]}
    assert all(event["causal_parent_id"] == request["event_id"] for event in failed)
    assert all(event.get("tool_call_id") != "discarded" for event in run["events"])
    for event in failed:
        artifact = root / event["artifact_ref"]["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == event["artifact_ref"]["sha256"]
        evidence = json.loads(artifact.read_text(encoding="utf-8"))
        assert evidence["partial_event_count"] == 2 and API_KEY not in artifact.read_text(encoding="utf-8")
        assert event["payload"]["stream_break_retry_limit"] == STREAM_BREAK_RETRIES
        assert event["payload"]["recovery_policy"] == recovery
    check = verify_agent_trace(root)
    if answered:
        response = events_of(run, "provider_response")[0]
        assert response["payload"]["attempts"] == STREAM_BREAK_RETRIES + 1
        assert check["status"] == "valid", check["errors"]
        assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
        # An attempt record that disagrees with its request breaks the attempt contract.
        events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
        tampered = next(event for event in events if event["type"] == "provider_attempt_failed")
        tampered["payload"]["request_sha256"] = hashlib.sha256(b"a different request").hexdigest()
        rechain(root, events)
        assert f"event:{tampered['seq']}:provider_attempt_contract" in verify_agent_trace(root)["errors"]
    else:
        error = events_of(run, "provider_error")[0]
        assert error["payload"]["attempts"] == STREAM_BREAK_RETRIES + 1
        assert error["payload"]["accepted"] is False
        assert not events_of(run, "action_proposed")
        # No attempt answered: the unmatched request is the fault, not a gap.
        assert any(item.startswith("request_without_response:") for item in check["errors"])
        assert not any("provider_attempt" in item for item in check["errors"])


def anthropic_tool_use(call_id: str, name: str, *, cut: bool) -> Reply:
    """An Anthropic Messages stream with one ``tool_use`` block; ``cut`` closes it after the block starts."""

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        tool = wire_tool(body, name)
        start = [("message_start", {"type": "message_start", "message": {
                     "id": f"msg-{call_id}", "type": "message", "role": "assistant", "content": [],
                     "usage": {"input_tokens": 100}}}),
                 ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
                     "type": "tool_use", "id": call_id, "name": tool, "input": {}}})]
        if cut:
            return {"sse": start, "break": True}
        arguments = json.dumps(FINISH if name == "finish_episode" else {})
        return {"sse": start + [
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": arguments}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                               "usage": {"output_tokens": 20}}),
            ("message_stop", {"type": "message_stop"})]}
    return reply


@pytest.mark.parametrize("recover", [True, False])
def test_a_broken_anthropic_stream_never_executes_its_tool_use(tmp_path: Path, recover: bool) -> None:
    replies = [anthropic_tool_use("discarded", "close_gripper", cut=True)]
    if recover:
        replies += [anthropic_tool_use("accepted", "close_gripper", cut=False),
                    anthropic_tool_use("end", "finish_episode", cut=False)]
    with Endpoint(replies) as endpoint:
        run = episodes.run_core("pi", tmp_path, actions=1, configure=anthropic_policy(
            endpoint.port, standby="offline-standby-key"))
    root = run["root"]
    failed = events_of(run, "provider_attempt_failed")
    assert len(failed) == (1 if recover else 3)
    assert all(event.get("tool_call_id") != "discarded" for event in run["events"])
    assert len(events_of(run, "action_proposed")) == int(recover)
    for event in failed:
        artifact = root / event["artifact_ref"]["path"]
        evidence = json.loads(artifact.read_text(encoding="utf-8"))
        assert evidence["partial_events"][1]["content_block"]["id"] == "discarded"
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == event["artifact_ref"]["sha256"]
        assert API_KEY not in artifact.read_text(encoding="utf-8")
    check = verify_agent_trace(root)
    if not recover:
        terminal = events_of(run, "provider_error")
        assert len(terminal) == 1
        assert terminal[0]["payload"]["attempts"] == 3
        assert terminal[0]["payload"]["response_started"] is True
        assert terminal[0]["payload"]["accepted"] is False
        assert any("request_without_response" in error for error in check["errors"])
        return
    assert check["status"] == "valid", check["errors"]
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    tampered = next(event for event in events if event["type"] == "provider_attempt_failed")
    tampered["payload"]["tool_execution_started"] = True
    rechain(root, events)
    assert any("provider_attempt_contract" in error for error in verify_agent_trace(root)["errors"])


# -- The round's deadline ------------------------------------------------------------------------


@pytest.mark.parametrize("recovery", ["same_credential_deadline_v1", "first_token_deadline_v1"])
def test_a_read_only_continuation_keeps_the_rounds_wall_time_deadline(tmp_path: Path, recovery: str) -> None:
    """The first request takes 2 s and calls a read-only tool; every later one fails with a
    retryable 503. The provider retries until the round's deadline, which the turn set when it
    started (4 s after the observation was published, ``pinned_wall_time``), not 4 s after the
    continuation started: the last retry arrives within 4 s of the first request."""

    remaining = 4.0
    unavailable = {"status": 503, "headers": {"retry-after": "0.2"},
                   "json": {"error": {"message": "temporarily unavailable"}}}
    observe = chat_tool_call("read", "observe", {"views": ["agentview"]})
    with Endpoint([lambda body: {**observe(body), "delay": 2.0}, lambda _body: unavailable]) as endpoint:
        with episodes.pinned_wall_time(remaining):
            run = episodes.run_core("pi", tmp_path, actions=1, configure=openai_policy(
                endpoint.port, stream=False, provider_recovery=recovery,
                provider_recovery_window_s=120.0))
    times = endpoint.times
    assert len(times) >= 4, times  # the continuation was retried
    assert times[1] - times[0] >= 2.0
    assert times[-1] - times[0] <= remaining + 0.5, [round(t - times[0], 3) for t in times]
    error = events_of(run, "provider_error")
    assert error and error[-1]["payload"]["accepted"] is False
    assert not events_of(run, "action_proposed")
