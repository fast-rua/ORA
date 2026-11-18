"""A scripted OpenAI-compatible chat endpoint on loopback, for interactive demos.

    python -m tests.support.agent_core.fake_model_server PORT_FILE [--delay S]

Plays the default `ora interactive` Synthetic task: read the manual, then one
`robot.submit_action` per call through three poses (repeating). Each reply
carries a short sentence; an operator message in the request is acknowledged
in the next reply, so a delivered steer is visible in the transcript. `--delay`
holds every reply (time to press Esc during a request). Writes its port to
PORT_FILE and serves until killed.

`--play` follows `ora play`'s operator instructions instead (Pi's own
agent: the instruction is the latest plain user message): for the latest
instruction it makes two moves, then replies with a short text report and no
tool call (the end of that instruction). A message that ends in a question
mark (`?` or `？`) is the operator talking to it while it works: the next
reply answers it in a sentence and keeps working on the instruction (with the
next move); one containing "just answer" gets a text-only answer, and the
model goes on moving at its next call. An instruction containing "three"
makes three moves, one containing "keep"
moves until the operator stops it (Esc); one containing "bash" runs Pi's bash
tool once (a listing of the workspace and whether a credential variable is
visible) before reporting; one naming `@file` reads it with Pi's read tool;
one asking for a script writes, edits and runs it (write, edit, bash); one
naming ora-robot drives the robot from Pi's bash (`ora-robot call`: a look,
then a move); one naming codemode writes a codemode script that looks and
moves through `tools.*`; one asking it to "think too long" first answers with
reasoning only, cut off at the output limit (`finish_reason: "length"`, no text,
no tool call), then, after ORA's continuation, moves and reports as usual
("always think too long": every reply is cut off); a request without tools
(Pi's compaction or branch summary) gets a short summary. A streaming request (Pi's providers stream) is
answered as server-sent events.

In either mode, ORA's provider probe (``ora providers test``, launch
preflight's ``provider_marker``) is answered as a healthy provider answers it:
the marker, the colour of the image, the tool round trip.

On a closed-skills surface (one with ``observe`` and ``set_gripper``, the
default RoboDojo tools) an instruction naming the gripper closes the left
gripper once and one asking it to look calls ``observe`` once, each then
reporting.

In ORA's code mode (``ora play --code-mode``: the request's tools are ORA's
``bash``, ``read``, ``write`` and ``edit``, no ``codemode`` and no robot tool)
an instruction naming a program runs one ``bash`` command whose Python program
closes and opens the gripper through the ``robot`` library (a toolkit surface's
``close_gripper`` and ``open_gripper``), then reports what it printed; one
naming a slow program closes it, then waits a minute (time for the operator's Esc).
The images Pi sends after a tool result (a user message "Attached image(s) from
tool result") are not an instruction.

`--requests FILE` appends one JSON line per request: its model, whether it
carried tools, whether it was a probe call, and the request_options fields it
carried (`options`).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ora.agent.providers.openai_wire_options import REQUEST_OPTION_FIELDS

POSES = ([0, 0, 1.02], [0.05, 0, 1.05], [0, 0, 1.02])
#: The robot from the agent's shell: the run's ora-robot client, through the Supervisor.
ORA_ROBOT_COMMAND = (
    "ora-robot call observation.dump_state > /dev/null && ora-robot call robot.submit_action --json "
    "'{\"arm\": \"left\", \"pose\": {\"position_m\": [0.03, 0, 1.04], \"orientation_xyzw\": [0, 0, 0, 1]}}' "
    "| python3 -c 'import json, sys; out = json.load(sys.stdin); "
    "print(\"moved:\", out[\"ok\"], \"new step:\", out[\"observation\"][\"observation_step\"])'")
#: The robot from a codemode script (Pi's sandbox): each call a tool call the Supervisor runs.
CODEMODE_SCRIPT = (
    "const looked = JSON.parse(await tools.observation_dump_state({}));\n"
    "const moved = JSON.parse(await tools.robot_submit_action({arm: \"left\", pose: "
    "{position_m: [0, 0, 1.02], orientation_xyzw: [0, 0, 0, 1]}}));\n"
    "return {looked: looked.ok, moved: moved.ok, new_step: moved.observation.observation_step};")


#: A code-mode program (ORA's bash, its robot library on a toolkit surface): the gripper closes, then opens.
CODE_MODE_PROGRAM = (
    "python - <<'EOF'\nimport json\nfrom robot import tools\n"
    "closed = tools.close_gripper()\nopened = tools.open_gripper()\n"
    "print('program moved:', json.dumps(opened, sort_keys=True)[:120])\nEOF")
#: A code-mode program that closes the gripper and then waits (the operator stops it).
CODE_MODE_SLOW_PROGRAM = (
    "python - <<'EOF'\nimport json, time\nfrom robot import tools\n"
    "closed = tools.close_gripper()\nprint('program moved:', json.dumps(closed, sort_keys=True)[:120])\n"
    "print('waiting', flush=True)\ntime.sleep(60)\nprint('woke up')\nEOF")


def code_mode_tools(request: dict) -> bool:
    """Whether the request offers ORA's code mode: its bash and no codemode or direct robot tool."""
    names = {tool["function"]["name"] for tool in request.get("tools", [])}
    return "bash" in names and "codemode" not in names and not any("submit_action" in name for name in names)


def tool_named(request: dict, name: str) -> str:
    slug = name.replace(".", "_")
    for tool in request.get("tools", []):
        wire = tool["function"]["name"]
        if wire == name or wire.startswith(slug):
            return wire
    raise KeyError(f"no {name} tool in the request")


def operator_note(request: dict, seen: set[str]) -> str | None:
    for message in reversed(request.get("messages", [])):
        content = message.get("content")
        text = content if isinstance(content, str) else " ".join(
            block.get("text", "") for block in content or [] if isinstance(block, dict))
        match = re.search(r'<operator_message>\s*(\{.*?\})\s*</operator_message>', text or "", re.S)
        if match:
            note = json.loads(match.group(1))
            if note["id"] not in seen:
                seen.add(note["id"])
                return str(note["message"])
    return None


def _text(content) -> str:
    return content if isinstance(content, str) else " ".join(
        block.get("text", "") for block in content or [] if isinstance(block, dict))


def is_question(body: str) -> bool:
    return body.rstrip().endswith(("?", "？"))


def latest_question(request: dict) -> str | None:
    """A question the operator asked since the model last replied, or None."""
    for message in reversed(request.get("messages", [])):
        body = _text(message.get("content")).strip()
        if message.get("role") == "assistant":
            return None
        if message.get("role") == "user" and is_question(body):
            return body
    return None


#: The user message Pi sends a tool result's images in (not the operator's).
TOOL_IMAGES = "Attached image(s) from tool result"
#: ORA's continuation after a reply the output limit cut off (ora-pi/src/play/driver.ts).
TRUNCATED_CONTINUATION = "reached the Provider output limit"


def continuations_since_instruction(request: dict) -> int:
    """How many of ORA's output-limit continuations followed the latest instruction."""
    count = 0
    for message in reversed(request.get("messages", [])):
        body = _text(message.get("content"))
        if message.get("role") == "user" and TRUNCATED_CONTINUATION in body:
            count += 1
        elif message.get("role") == "user" and body.strip() and not body.strip().startswith("{") \
                and not is_question(body):
            break
    return count


def latest_instruction(request: dict) -> tuple[str | None, int, int]:
    """The latest operator instruction, the robot moves and the other tool calls the model made since.

    An instruction is a plain user message; an observation is a user message whose text is a
    JSON envelope (with its images); a question (`?`) asked while one runs leaves it as it is.
    """
    text, moves, other = None, 0, 0
    for message in request.get("messages", []):
        body = _text(message.get("content")).strip()
        if (message.get("role") == "user" and body and not body.startswith("{") and "<operator_message>" not in body
                and not body.startswith(TOOL_IMAGES)
                and TRUNCATED_CONTINUATION not in body and not (is_question(body) and text is not None)):
            text, moves, other = body, 0, 0
        elif message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if "submit_action" in call.get("function", {}).get("name", ""):
                    moves += 1
                else:
                    other += 1
    return text, moves, other


def last_tool_text(request: dict) -> str:
    for message in reversed(request.get("messages", [])):
        if message.get("role") == "tool":
            return _text(message.get("content"))
    return ""


def probe_reply(request: dict) -> tuple[str, str | None, dict] | None:
    """The provider probe's stages (provider_probe_stages.py), or None for any other request."""

    messages = request.get("messages") or [{}]
    last = _text(messages[-1].get("content"))
    for pattern in (r"Reply with exactly this marker: (\S+)", r"Reply with exactly (CONTINUED-[0-9a-f]+)"):
        match = re.search(pattern, last)
        if match:
            return match.group(1), None, {}
    if "Name the dominant color" in last:
        return "red", None, {}
    match = re.search(r"Call preflight_echo exactly once with token (\S+?)\.", last)
    if match:
        return "", "preflight_echo", {"token": match.group(1)}
    return None


def play_reply(request: dict, state: dict) -> tuple[str, str | None, dict]:
    if not request.get("tools"):
        return "Summary: the operator gave instructions; the agent moved the left arm and reported.", None, {}
    instruction, moves, other = latest_instruction(request)
    if instruction is None:
        return "No instruction yet; waiting for the operator.", None, {}
    if "think too long" in instruction.lower() and (
            "always" in instruction.lower() or continuations_since_instruction(request) == 0):
        # Reasoning only, cut off by the output limit: no text, no tool call.
        state["finish"] = "length"
        return "", None, {}
    if code_mode_tools(request):
        if other == 0:
            slow = "slow" in instruction.lower()
            return ("Looking and moving the left arm from a program.", "bash",
                    {"command": CODE_MODE_SLOW_PROGRAM if slow else CODE_MODE_PROGRAM})
        said = [line for line in last_tool_text(request).replace("\\n", "\n").splitlines() if "program moved" in line]
        return (f"Done with \"{instruction}\": the program ran ({'moved' if said else 'no move printed'}).", None, {})
    words = instruction.split()
    file_ref = next((word[1:].strip(".,") for word in words if word.startswith("@") and len(word) > 1), None)
    if file_ref is not None and other == 0:
        return (f"Reading {file_ref}.", "read", {"path": file_ref})
    if file_ref is not None:
        said = last_tool_text(request).strip().splitlines()
        return (f"Done with \"{instruction}\": {file_ref} says \"{said[0] if said else ''}\".", None, {})
    if "script" in instruction.lower():
        steps = [("Writing a small script.", "write", {"path": "hello.py", "content": "print('hello from play')\n"}),
                 ("Editing the script.", "edit", {"path": "hello.py", "edits": [
                     {"oldText": "hello from play", "newText": "hello from free play"}]}),
                 ("Running it.", "bash", {"command": "python3 hello.py"})]
        if other < len(steps):
            return steps[other]
        return (f"Done with \"{instruction}\": the script printed \"{last_tool_text(request).strip()}\".", None, {})
    if "ora-robot" in instruction and other == 0:
        return ("Driving the robot from the shell with ora-robot.", "bash", {"command": ORA_ROBOT_COMMAND})
    if "ora-robot" in instruction:
        return (f"Done with \"{instruction}\": {last_tool_text(request).strip()}", None, {})
    if "codemode" in instruction and other == 0:
        return ("Looking and moving from a codemode script.", tool_named(request, "codemode"),
                {"code": CODEMODE_SCRIPT})
    if "codemode" in instruction:
        said = last_tool_text(request).strip().splitlines()
        return (f"Done with \"{instruction}\": the script said {said[-1] if said else ''}", None, {})
    if "bash" in instruction.lower() and other == 0:
        return ("Checking the workspace with bash.", "bash",
                {"command": "ls; if [ -n \"${ORA_PI_API_KEY:-}\" ]; then echo 'key visible: yes'; "
                            "else echo 'key visible: no'; fi; echo \"credential variables: $(env | cut -d= -f1 | grep -ciE "
                            "'api_?key|token|secret' || true)\""})
    if "bash" in instruction.lower():
        return (f"Done with \"{instruction}\": I listed the workspace with bash; no API key is visible to it.",
                None, {})
    tools = {tool["function"]["name"] for tool in request.get("tools", [])}
    if {"observe", "set_gripper"} <= tools:
        # A closed-skills surface (RoboDojo's default): a look, or one gripper directive, then a report.
        if "gripper" in instruction.lower() and other == 0:
            return ("Closing the left gripper.", "set_gripper", {"targets": {"left": "close"}})
        if "look" in instruction.lower() and other == 0:
            return ("Looking at the table from the head camera.", "observe", {"views": ["head"]})
        if other:
            said = last_tool_text(request).strip().splitlines()
            return (f"Done with \"{instruction}\": {said[0][:160] if said else 'done'}", None, {})
    question = latest_question(request)
    if question is not None and "just answer" in question.lower():
        # A question to answer in words: a text-only reply, which the Supervisor takes as the answer to the
        # joined message (play_answer), not the instruction's report; the next act goes on moving.
        return (f"You asked \"{question}\": I have made {moves} move{'s' if moves != 1 else ''} so far and keep "
                "going.", None, {})
    if moves < (3 if "three" in instruction.lower() else 2) or "keep" in instruction.lower() or question is not None:
        pose = POSES[state["moves"] % len(POSES)]
        state["moves"] += 1
        said = (f"You asked \"{question}\": I have made {moves} move{'s' if moves != 1 else ''} so far and the arm is "
                f"where I last put it; I keep going. " if question is not None else "")
        return (f"{said}Working on \"{instruction}\": moving the left end effector to {pose}.",
                tool_named(request, "robot.submit_action"),
                {"arm": "left", "pose": {"position_m": pose, "orientation_xyzw": [0, 0, 0, 1]}})
    return (f"Done with \"{instruction}\": I moved the left end effector twice; the latest image shows the "
            "arm at the last commanded pose. Waiting for the next instruction.", None, {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port_file", type=Path)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--play", action="store_true", help="follow free-play instructions (ora play)")
    parser.add_argument("--requests", type=Path, default=None, help="log each request's model and request_options")
    args = parser.parse_args()
    state = {"calls": 0, "moves": 0}
    seen: set[str] = set()
    lock = threading.Lock()

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                probe = probe_reply(request)  # a probe is not part of the scripted conversation
                if args.requests is not None:
                    with args.requests.open("a", encoding="utf-8") as log:
                        log.write(json.dumps({
                            "model": request.get("model"), "tools": bool(request.get("tools")), "probe": probe is not None,
                            "options": {key: request[key] for key in sorted(REQUEST_OPTION_FIELDS) if key in request},
                        }, sort_keys=True) + "\n")
                state["calls"] += probe is None
                call = state["calls"]
                note = operator_note(request, seen)
                if probe is not None:
                    reply = probe
                elif args.play:
                    reply = play_reply(request, state)
                elif call == 1:
                    name, arguments, text = tool_named(request, "robot.read_manual"), {"section": "Actions"}, \
                        "I will read the action section of the robot manual first."
                else:
                    pose = POSES[state["moves"] % len(POSES)]
                    state["moves"] += 1
                    name = tool_named(request, "robot.submit_action")
                    arguments = {"arm": "left", "pose": {"position_m": pose, "orientation_xyzw": [0, 0, 0, 1]}}
                    text = f"Moving the left end effector to {pose}."
                if not args.play and probe is None:
                    if note:
                        text = f"Operator note received (\"{note}\"). " + text
                    reply = text, name, arguments
                cut_off = state.pop("finish", None)  # play_reply's output-limit stop, for this request only
            time.sleep(args.delay)
            text, name, arguments = reply
            finish = cut_off or ("tool_calls" if name else "stop")
            reasoning = "Weighing every way to do this, at length..." if finish == "length" else None
            message = {"role": "assistant", "content": text}
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            if name is not None:
                message["tool_calls"] = [{"id": f"call-{call}", "type": "function",
                                          "function": {"name": name, "arguments": json.dumps(arguments)}}]
            completion = 8192 if finish == "length" else 40
            usage = {"prompt_tokens": 1500 + 200 * call, "completion_tokens": completion,
                     "total_tokens": 1500 + 200 * call + completion}
            try:
                if request.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    first = {"role": "assistant", "reasoning_content": reasoning} if reasoning is not None \
                        else {"role": "assistant", "content": text}
                    for delta, reason in ([first, None],
                                          *([[{"tool_calls": [{"index": 0, **message["tool_calls"][0]}]}, None]]
                                            if name else []),
                                          [{}, finish]):
                        chunk = {"id": f"fake-{call}", "object": "chat.completion.chunk", "model": request.get("model"),
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
                        if reason is not None:
                            chunk["usage"] = usage
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                body = json.dumps({
                    "id": f"fake-{call}", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": finish, "message": message}], "usage": usage,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the caller aborted the request (Esc)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    args.port_file.write_text(str(server.server_port))
    server.serve_forever()


if __name__ == "__main__":
    main()
