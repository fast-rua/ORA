"""What a script asked of the library, on the record (``policy.code_mode.library: extended``).

A script's ``robot.query`` and ``robot.explain`` requests are no tool calls, but each is
recorded as a nested call of the script, as its tool calls are (``agent/pi/code_mode.py``, ``_code_mode_record_op``):
a ``tool_call_proposed``/``tool_result`` pair named ``robot.<op>``, effect ``read_only``, with ``origin.op``, what was
asked as its arguments (``{"name": ..., "arguments": ...}``) and what was returned, a refusal too, as its result.
``evidence.agent_events.library_op`` tells such a pair from a tool call. A ``docs`` lookup returns only static text
and is not recorded.

The result of every call an extended script makes (a dry run included) also carries the inputs its explanation is
rendered from (``payload.explain_inputs``, an artifact ``ora.code-mode-explain-inputs.v1``): the tool, the arguments,
what the script read of the call (its ``result``, a refused call's ``error``, an action's ``receipt``) and the stage
events its tool recorded (``toolkit/interface/stages.py``). ``render`` turns them into the explanation the script read,
so ``recorded_inputs`` lets ``explain`` answer for any call of the run, after a restart too, and
``check_explanations`` renders every explanation a script read again from the record alone (the run's evaluation runs it
for an extended plan). The inputs name their stage template by its digest: a text rendered from another template than
the checking code has cannot be held to it, and is left alone. So a sealed run's explanation, which ended with the
checkpoints of ``robot.inspect`` (retired on 2026-11-01), renders no more and is left alone by its renderer's digest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.agent.evidence.agent_events import library_op
from ora.agent.evidence.canonical import canonical_bytes, sha256_bytes
from ora.agent.evidence.trace_privacy import SCRIPT_TOOLS
from ora.agent.evidence.trace_verify import artifact_bytes
from ora.agent.toolkit import explain as toolkit_explain

EXPLAIN_INPUTS_SCHEMA = "ora.code-mode-explain-inputs.v1"
#: The ``tool_result`` payload field that holds a call's explain inputs, and their artifact kind.
EXPLAIN_INPUTS_FIELD = "explain_inputs"
EXPLAIN_INPUTS_KIND = "code_mode_explain_inputs"
#: What the explanation reads of the answer a script got (``toolkit_explain.render_answer``).
ANSWER_FIELDS = ("result", "error", "receipt")
#: The identity of the renderer's own code (``toolkit/explain.py``): a text rendered by other code than this is not held
#: to this code, as a text rendered from another template is not held to this one.
RENDERER_SHA256 = sha256_bytes(Path(toolkit_explain.__file__).read_bytes())


def template_sha256(tool: str) -> str | None:
    """The identity of the tool's stage template as this code has it (None without one)."""

    spec = toolkit_explain.template(tool)
    return None if spec is None else sha256_bytes(canonical_bytes(spec))


def explain_inputs(tool: str, arguments: Mapping[str, Any], answer: Mapping[str, Any],
                   events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The inputs a call's explanation is rendered from: the tool and its template's identity, its arguments as the
    script gave them, what the script read of it and its stage events (a copy, frozen as the call left them)."""

    return copy.deepcopy({"schema": EXPLAIN_INPUTS_SCHEMA, "tool": tool, "template_sha256": template_sha256(tool),
                          "renderer_sha256": RENDERER_SHA256,
                          "arguments": dict(arguments),
                          "answer": {key: answer[key] for key in ANSWER_FIELDS if key in answer},
                          "events": list(events)})


def render(inputs: Mapping[str, Any]) -> toolkit_explain.Explanation | None:
    """The call's explanation from its inputs alone; None for a tool without a stage template."""

    return toolkit_explain.render_answer(inputs["tool"], inputs["arguments"], inputs["answer"], inputs["events"])


def attached(text: str) -> str:
    """An explanation as a script's answer and its result's ``calls`` carry it: at most
    ``toolkit_explain.ATTACHED_CHARS`` characters, the middle cut."""

    text, limit = text.rstrip(), toolkit_explain.ATTACHED_CHARS
    return text if len(text) <= limit else text[: limit // 2] + f"\n[... {len(text) - limit} characters cut ...]\n" \
        + text[-limit // 2:]


def _events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "agent-events.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a writer cut off mid-line
        if isinstance(event, dict):
            events.append(event)
    return events


def _artifact(run_root: Path, ref: Any) -> Any:
    """The JSON an artifact reference names, if its bytes are there and match it; None otherwise."""

    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        return None
    data = artifact_bytes(run_root, ref["path"])
    if data is None or sha256_bytes(data) != ref.get("sha256"):
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _inputs_of(run_root: Path, event: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    inputs = _artifact(run_root, payload.get(EXPLAIN_INPUTS_FIELD))
    return inputs if isinstance(inputs, dict) and inputs.get("schema") == EXPLAIN_INPUTS_SCHEMA else None


def recorded_inputs(run_root: Path, call_id: str) -> dict[str, Any] | None:
    """The explain inputs the run's record holds for the script call ``call_id`` (``<parent call id>#NNN``), or None:
    no such call, a library request, or a call recorded without them (a ``base`` script's, one from before them)."""

    run_root = run_root.resolve()
    for event in reversed(_events(run_root)):
        if event.get("type") == "tool_result" and event.get("tool_call_id") == call_id:
            return _inputs_of(run_root, event)
    return None


def check_explanations(run_root: Path) -> list[str]:
    """Every explanation a script read, rendered again from the record alone: each ``robot.explain`` answer, and each
    explanation a script result's ``calls`` carry (a failed call's, attached by default). An error names the record
    whose text the explained call's recorded inputs do not reproduce, or that names a call without them."""

    run_root = run_root.resolve()
    events = _events(run_root)
    scripts = {(event.get("model_call_id"), event.get("tool_call_id")) for event in events
               if event.get("type") == "tool_call_proposed" and isinstance(event.get("payload"), Mapping)
               and event["payload"].get("name") in SCRIPT_TOOLS and "origin" not in event["payload"]}
    inputs: dict[Any, dict[str, Any] | None] = {}
    # (the record holding the text, the call it explains, the text, whether whole or as attached to a result)
    read: list[tuple[Any, Any, Any, bool]] = []
    for event in events:
        if event.get("type") != "tool_result":
            continue
        call_id, payload = event.get("tool_call_id"), event.get("payload")
        if library_op(event) == "explain":
            answer = _artifact(run_root, event.get("artifact_ref"))
            if isinstance(answer, Mapping) and answer.get("ok") is True:  # a refused request read no explanation
                read.append((call_id, answer.get("call_id"), answer.get("text"), True))
        elif isinstance(payload, Mapping) and EXPLAIN_INPUTS_FIELD in payload:
            inputs[call_id] = _inputs_of(run_root, event)
        elif (event.get("model_call_id"), call_id) in scripts:
            result = _artifact(run_root, event.get("artifact_ref"))
            for entry in result.get("calls") or () if isinstance(result, Mapping) else ():
                if isinstance(entry, Mapping) and isinstance(entry.get("explain"), str) and type(entry.get("call")) is int:
                    called = f"{call_id}#{entry['call']:03d}"
                    read.append((called, called, entry["explain"], False))
    errors = []
    for record, called, text, whole in read:
        recorded = inputs.get(called)
        if recorded is not None and (recorded.get("template_sha256") != template_sha256(recorded["tool"])
                                     or recorded.get("renderer_sha256") != RENDERER_SHA256):
            continue  # rendered from another template or by other renderer code than this: nothing to hold the text to
        explained = render(recorded) if recorded is not None else None
        if explained is None:
            errors.append(f"code_mode_explain_without_inputs:{record}")
        elif text != (explained.text if whole else attached(explained.text)):
            errors.append(f"code_mode_explain_not_reproduced:{record}")
    return errors


__all__ = ["ANSWER_FIELDS", "EXPLAIN_INPUTS_FIELD", "EXPLAIN_INPUTS_KIND", "EXPLAIN_INPUTS_SCHEMA", "attached",
           "RENDERER_SHA256", "check_explanations", "explain_inputs", "recorded_inputs", "render", "template_sha256"]
