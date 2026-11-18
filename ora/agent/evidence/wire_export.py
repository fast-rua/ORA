"""Export every model call of a run exactly as it was sent, each image once, with a replay proof.

``export_wire`` writes, for every provider exchange of a run:

- ``requests.jsonl``: the request body the provider received, images replaced
  by ``images/<sha256>.<ext>`` references (an OpenAI Chat Completions body for
  OpenAI-compatible providers; for another wire protocol the canonical
  OpenAI/ChatML request, with the wire body beside it);
- ``responses.jsonl``: ORA's record of the reply (accepted or rejected, with its
  output rejection) and ``provider_reply``, the reply parsed from the archived
  stream: content, reasoning, tool calls (truncated fragments included),
  the provider's finish reason and usage;
- ``streams.jsonl``: the archived provider response;
- ``compactor.jsonl``: the context compactor's calls, request, reply and stream;
- ``images/`` and ``images.jsonl``: each image sent to a model once, with the
  calls that sent it;
- ``epochs.jsonl``: the policy messages once per run of calls whose requests
  extend each other, so an append-only context is one conversation;
- ``tokens.jsonl``: each policy call's token ids where the run or the operator
  has them (``token_ids``): from a stream the engine sent with
  ``return_token_ids``, or from the engine's ``/tokenize`` when ``export_wire``
  is given a ``Tokenizer``; every call is checked against the engine's usage.

``check_wire_export`` rebuilds every request from ``requests.jsonl``, from
``epochs.jsonl`` and (for the compactor) from ``compactor.jsonl``, serializes it
canonically and requires the archived sha256; streams, provider replies, the
image index, the token rows and, once written, the manifest's file hashes are
checked as well. Token ids a ``Tokenizer`` gave cannot be asked again offline:
the check rebuilds every row from the recorded ids and the run's streams and
usage, so an edit is found, but it trusts the engine's answer.
``export_wire`` writes to a staging directory and moves it into place only when
the check passes.

With ``host_paths`` (``--redact-host-paths``) every request, stream and reply
record has the host paths of ``host_paths.RULES`` replaced by placeholders
before anything is derived from it; a call whose request changed carries no
prompt token ids, a stream whose text changed no sampled ids. Nothing derived
from the redacted bytes ships: a changed row's sealed sha256 is null and the
trace's final event hash is left out. The manifest's ``redaction`` records each
rule's replacements and each changed row's redacted sha256, which the check
takes. A prefix left after that (split across stream chunks) refuses the export.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.agent.evidence import request_archive, token_ids
from .agent_events import read_agent_events
from .canonical import TrajectoryError, _atomic_json, canonical_bytes, file_sha256, sha256_bytes
from .host_paths import HostPaths, Redaction, statement
from .sft_export import check_seal
from .trace_privacy import (
    _allows_public_grasp_contact_pose, _forbidden_path, model_tool_arguments, script_call_ids,
    without_script_authored_messages,
)
from .trace_verify import verify_agent_trace


WIRE_EXPORT_SCHEMA = "ora.sft-wire-export.v2"
OPENAI_PROTOCOL = "openai_chat_completions"
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
FILES = ("requests.jsonl", "responses.jsonl", "streams.jsonl", "compactor.jsonl", "epochs.jsonl", "images.jsonl",
         "tokens.jsonl")
#: An incomplete run (a request whose reply never arrived) exports its complete exchanges.
INCOMPLETE_ERROR = "request_without_response"
MANIFEST_KEYS = frozenset({
    "schema", "run_id", "source_commit", "trace", "official_success", "termination", "evaluation_status",
    "learning_eligible", "calls", "unanswered_calls", "accepted_calls", "compactor_calls", "epochs", "images",
    "tokens", "replay_check", "files",
})
COMPACTOR_REQUEST, COMPACTOR_RESPONSE, COMPACTOR_FAILED = (
    "context_compactor_attempt", "context_compactor_response", "context_compactor_failed")


class _Images:
    """Content-addressed image files under ``images/``, referenced from requests."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory  # None: index only, write nothing (a check against the source run)
        self.index: dict[str, dict[str, Any]] = {}

    def put(self, mime: str, data: str, call_id: str) -> str | None:
        """The reference for one base64 image, or None when it cannot round-trip."""

        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return None
        if base64.b64encode(raw).decode("ascii") != data:
            return None  # not canonical base64: keep it inline so the request rebuilds exactly
        digest = hashlib.sha256(raw).hexdigest()
        path = f"images/{digest}.{IMAGE_EXTENSIONS.get(mime, 'bin')}"
        entry = self.index.get(path)
        if entry is None:
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                (self.directory.parent / path).write_bytes(raw)
            entry = self.index[path] = {"path": path, "sha256": digest, "media_type": mime,
                                        "bytes": len(raw), "calls": []}
        elif entry["media_type"] != mime:
            return None
        if call_id not in entry["calls"]:
            entry["calls"].append(call_id)
        return path

    def externalize(self, value: Any, call_id: str) -> Any:
        """``value`` with each OpenAI data-URL or Anthropic base64 image replaced by its file reference."""

        if isinstance(value, list):
            return [self.externalize(item, call_id) for item in value]
        if not isinstance(value, Mapping):
            return value
        out = {key: self.externalize(item, call_id) for key, item in value.items()}
        image = value.get("image_url")
        if value.get("type") == "image_url" and isinstance(image, Mapping) and isinstance(image.get("url"), str):
            parts = _data_url(image["url"])
            if parts is not None:
                ref = self.put(parts[0], parts[1], call_id)
                if ref is not None:
                    out["image_url"] = {**image, "url": ref}
        source = value.get("source")
        if (value.get("type") == "image" and isinstance(source, Mapping) and source.get("type") == "base64"
                and isinstance(source.get("data"), str) and isinstance(source.get("media_type"), str)):
            ref = self.put(source["media_type"], source["data"], call_id)
            if ref is not None:
                out["source"] = {**source, "data": ref}
        return out


def _data_url(url: str) -> tuple[str, str] | None:
    header, sep, data = url.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        return None
    mime = header[len("data:"):-len(";base64")]
    return (mime, data) if mime in IMAGE_EXTENSIONS else None


def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("images/") and "/" not in value[len("images/"):]


def _ref_mime(ref: str) -> str:
    extension = ref.rsplit(".", 1)[-1]
    return next(mime for mime, ext in IMAGE_EXTENSIONS.items() if ext == extension)


def image_refs(value: Any) -> list[str]:
    """The image file references in a request, in order (repeats included)."""

    if isinstance(value, list):
        return [ref for item in value for ref in image_refs(item)]
    if not isinstance(value, Mapping):
        return []
    refs = [ref for item in value.values() for ref in image_refs(item)]
    image, source = value.get("image_url"), value.get("source")
    if value.get("type") == "image_url" and isinstance(image, Mapping) and _is_ref(image.get("url")):
        refs.insert(0, image["url"])
    if value.get("type") == "image" and isinstance(source, Mapping) and _is_ref(source.get("data")):
        refs.insert(0, source["data"])
    return refs


def internalize(value: Any, root: Path) -> Any:
    """Inverse of the export's image references: read each ``images/...`` file back into the request."""

    if isinstance(value, list):
        return [internalize(item, root) for item in value]
    if not isinstance(value, Mapping):
        return value
    out = {key: internalize(item, root) for key, item in value.items()}
    image = value.get("image_url")
    if value.get("type") == "image_url" and isinstance(image, Mapping) and _is_ref(image.get("url")):
        data = base64.b64encode((root / image["url"]).read_bytes()).decode("ascii")
        out["image_url"] = {**image, "url": f"data:{_ref_mime(image['url'])};base64,{data}"}
    source = value.get("source")
    if value.get("type") == "image" and isinstance(source, Mapping) and _is_ref(source.get("data")):
        out["source"] = {**source, "data": base64.b64encode((root / source["data"]).read_bytes()).decode("ascii")}
    return out


def provider_reply(response: Any) -> dict[str, Any] | None:
    """The reply as the provider sent it, from an archived OpenAI Chat Completions stream or body.

    Content and reasoning are concatenated deltas (sglang names reasoning
    ``reasoning_content``, vLLM ``reasoning``); tool calls are assembled by
    index, so a call cut off by the token limit keeps its fragment. The finish
    reason and usage are the provider's own. Another response shape gives None.
    """

    if isinstance(response, Mapping) and isinstance(response.get("choices"), list):
        choice = response["choices"][0] if response["choices"] else {}
        message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        return {
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content", message.get("reasoning")),
            "tool_calls": list(message.get("tool_calls") or []),
            "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
        }
    if (not isinstance(response, list) or not all(isinstance(chunk, Mapping) for chunk in response)
            or not any("choices" in chunk for chunk in response)):
        return None
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = usage = None
    for chunk in response:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") if isinstance(choice.get("delta"), Mapping) else {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for key in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    reasoning.append(delta[key])
            for part in delta.get("tool_calls") or []:
                if not isinstance(part, Mapping):
                    continue
                call = calls.setdefault(int(part.get("index", len(calls))),
                                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if part.get("id"):
                    call["id"] = part["id"]
                if part.get("type"):
                    call["type"] = part["type"]
                function = part.get("function") if isinstance(part.get("function"), Mapping) else {}
                if isinstance(function.get("name"), str):
                    call["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    call["function"]["arguments"] += function["arguments"]
    return {
        "content": "".join(content) if content else None,
        "reasoning_content": "".join(reasoning) if reasoning else None,
        "tool_calls": [calls[index] for index in sorted(calls)],
        "finish_reason": finish_reason, "usage": usage,
    }


def _epochs(calls: list[tuple[str, list[Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group calls whose messages extend each other; each call is ``base[:prefix]`` plus an optional tail.

    A call either extends its epoch's messages, or extends them except for its
    own last message (a rolling camera tail, which the previous call then also
    had). Anything else opens a new epoch. Every placement is checked against
    the call's messages; a call that would not rebuild gets an epoch of its own.
    """

    bases: list[list[Any]] = []
    placed: list[dict[str, Any]] = []
    for call_id, messages in calls:
        base = bases[-1] if bases else None
        if base is not None and messages[:len(base)] == base:
            bases[-1] = list(messages)
            placed.append({"model_call_id": call_id, "epoch": len(bases) - 1, "prefix": len(messages), "tail": None})
            continue
        previous = placed[-1] if placed else None
        if base is not None and previous is not None and messages and previous["epoch"] == len(bases) - 1:
            if previous["tail"] is None and previous["prefix"] == len(base) and base:
                # The previous call's own last message was a tail as well.
                shorter = base[:-1]
                others = [item for item in placed[:-1] if item["epoch"] == len(bases) - 1]
                if messages[:len(shorter)] == shorter and len(messages) > len(shorter) and all(
                        item["prefix"] <= len(shorter) for item in others):
                    previous["prefix"], previous["tail"] = len(shorter), base[-1]
                    base = bases[-1] = shorter
            if previous["tail"] is not None and messages[:-1][:len(base)] == base and len(messages) - 1 >= len(base):
                bases[-1] = list(messages[:-1])
                placed.append({"model_call_id": call_id, "epoch": len(bases) - 1,
                               "prefix": len(messages) - 1, "tail": messages[-1]})
                continue
        bases.append(list(messages))
        placed.append({"model_call_id": call_id, "epoch": len(bases) - 1, "prefix": len(messages), "tail": None})
    for (call_id, messages), item in zip(calls, placed):
        rebuilt = bases[item["epoch"]][:item["prefix"]] + ([item["tail"]] if item["tail"] is not None else [])
        if rebuilt != messages:
            bases.append(list(messages))
            item.update(epoch=len(bases) - 1, prefix=len(messages), tail=None)
    return [{"epoch": index, "messages": messages} for index, messages in enumerate(bases)], placed


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_bytes(row).decode("utf-8") + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _artifact(root: Path, ref: Any) -> tuple[Any, str] | None:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        return None
    data = request_archive.verified_bytes(root, ref)
    if data is None:
        raise TrajectoryError(f"provider artifact drift: {ref['path']}")
    return json.loads(data.decode("utf-8")), str(ref["sha256"])


def _clean(message: Any) -> Any:
    if not isinstance(message, Mapping):
        return None
    return {key: value for key, value in message.items() if not str(key).startswith("_")}


def _open_permissions(root: Path) -> None:
    """mkdtemp and mkstemp create 0700/0600; give the export the modes the umask allows."""

    umask = os.umask(0)
    os.umask(umask)
    setgid = root.parent.stat().st_mode & stat.S_ISGID
    for path in [root, *root.rglob("*")]:
        if path.is_dir():
            os.chmod(path, (0o777 & ~umask) | setgid)
        else:
            os.chmod(path, 0o666 & ~umask)


def _trace_status(root: Path, evaluation: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept a verified trace; an incomplete one; or a sealed one this tree reads differently. Refuse the rest."""

    sealed = check_seal(root)
    verification = verify_agent_trace(root)
    if verification["status"] == "valid":
        return {"status": "valid", "sealed": sealed, "final_event_sha256": verification.get("final_event_sha256")}
    errors = [str(item) for item in verification.get("errors", [])]
    if errors and all(item.startswith(INCOMPLETE_ERROR) for item in errors):
        return {"status": "incomplete", "sealed": sealed, "errors": errors}
    if sealed and isinstance(evaluation, Mapping) and evaluation.get("status") == "valid":
        # The seal still matches and the run's own evaluation found it valid; this
        # tree's verifier reads the trace differently. Every request below must
        # still rebuild to its sealed sha256.
        return {"status": "sealed_valid_unverified_here", "sealed": True, "errors": errors}
    raise TrajectoryError("invalid agent trace: " + ", ".join(errors))


def _request_row(root: Path, images: _Images, index: int, call_id: str, request_event: Mapping[str, Any],
                 payload: Mapping[str, Any], allow_grasp: bool, script_calls: frozenset[str],
                 model_arguments: Mapping[str, str], redaction: Redaction | None = None,
                 file: str = "requests.jsonl") -> tuple[dict[str, Any], Any, bool]:
    protocol = payload.get("provider_protocol") or OPENAI_PROTOCOL
    wire = _artifact(root, request_event.get("artifact_ref"))
    if wire is None:
        raise TrajectoryError(f"{call_id} lacks a wire request artifact")
    canonical = _artifact(root, payload.get("canonical_request_ref")) or wire
    # The OpenAI-format request is the wire body itself unless the provider spoke another protocol.
    body, body_sha = wire if protocol == OPENAI_PROTOCOL else canonical
    forbidden = _forbidden_path(without_script_authored_messages(body, script_calls),
                                allow_public_grasp_contact_pose=allow_grasp, model_arguments=model_arguments)
    if forbidden:
        raise TrajectoryError(f"privileged field in {call_id}'s request: {forbidden}")
    # A redacted value's sealed sha256 is null: it would confirm a guessed prefix put back into the text.
    body, replacements = (body, 0) if redaction is None else redaction.row(file, call_id, "request", body)
    row: dict[str, Any] = {
        "index": index, "model_call_id": call_id, "protocol": protocol,
        "request_source": "wire" if protocol == OPENAI_PROTOCOL else "canonical",
        "request_sha256": None if replacements else body_sha, "request": images.externalize(body, call_id),
    }
    if protocol != OPENAI_PROTOCOL:
        wire_body, wire_replacements = (wire[0], 0) if redaction is None else redaction.row(
            file, call_id, "wire_request", wire[0])
        replacements += wire_replacements
        row["wire_request_sha256"] = None if wire_replacements else wire[1]
        row["wire_request"] = images.externalize(wire_body, call_id)
    return row, body, replacements > 0


def _reply(root: Path, call_id: str, event: Mapping[str, Any] | None, allow_grasp: bool,
           model_arguments: Mapping[str, str], redaction: Redaction | None = None, file: str = "streams.jsonl",
           prompt_redacted: bool = False) -> tuple[dict, dict]:
    """ORA's record of one reply and its archived stream (its token ids null where redaction contradicts them)."""

    payload = event.get("payload") if isinstance(event, Mapping) and isinstance(event.get("payload"), Mapping) else {}
    message = _clean(payload.get("message"))
    forbidden = message is not None and _forbidden_path(message, allow_public_grasp_contact_pose=allow_grasp,
                                                        model_arguments=model_arguments)
    if forbidden:
        raise TrajectoryError(f"privileged field in {call_id}'s reply: {forbidden}")
    stream = _artifact(root, event.get("artifact_ref")) if isinstance(event, Mapping) else None
    record = {key: value for key, value in payload.items() if key != "message"}
    record["message"] = message
    response, changed = None if stream is None else stream[0], False
    if redaction is not None:
        record = redaction.text(record)[0]
        if stream is not None:
            response, changed = redaction.stream(file, call_id, response, prompt_redacted=prompt_redacted)
    record["provider_reply"] = None if response is None else provider_reply(response)
    return record, {"response_sha256": None if stream is None or changed else stream[1], "response": response}


def _policy_calls(events: list[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    """Every provider request in trace order with its reply, or None when nothing answered it."""

    order: list[str] = []
    requests: dict[str, Mapping[str, Any]] = {}
    replies: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = event.get("model_call_id")
        if not isinstance(call_id, str):
            continue
        if event.get("type") == "provider_request" and call_id not in replies:
            if call_id not in requests:
                order.append(call_id)
            requests[call_id] = event  # the request its reply answers
        elif event.get("type") == "provider_response" and call_id in requests:
            replies.setdefault(call_id, event)
    return [(requests[call_id], replies.get(call_id)) for call_id in order]


def _compactor_calls(events: list[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    attempts: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    replies: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = event.get("model_call_id")
        if not isinstance(call_id, str):
            continue
        if event.get("type") == COMPACTOR_REQUEST:
            attempts[call_id] = event
            order.append(call_id)
        elif event.get("type") in (COMPACTOR_RESPONSE, COMPACTOR_FAILED) and call_id in attempts:
            replies.setdefault(call_id, event)
    return [(attempts[call_id], replies.get(call_id)) for call_id in order]


def _unsent_images(events: list[Mapping[str, Any]], sent: set[str]) -> dict[str, int]:
    """Distinct recorded image artifacts, by event type, whose bytes no model request carried.

    Such an image was either never sent or sent in another encoding or size.
    """

    unsent: dict[str, set[str]] = {}
    for event in events:
        ref = event.get("artifact_ref")
        if (isinstance(ref, Mapping) and str(ref.get("media_type", "")).startswith("image/")
                and ref.get("sha256") not in sent):
            unsent.setdefault(str(event.get("type")), set()).add(str(ref.get("sha256")))
    return {kind: len(shas) for kind, shas in sorted(unsent.items())}


class _Replay:
    """An export's recorded ``/tokenize`` answers, handed back in trace order: a check cannot ask the engine again."""

    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self._rows = {row.get("model_call_id"): row for row in rows}
        self._chain = token_ids.prompt_ids(rows)

    def __call__(self, call_id: str, body: Mapping[str, Any]) -> list[int] | None:
        for rebuilt_id, ids in self._chain:
            if rebuilt_id == call_id:
                break
        else:
            return None
        prompt = (self._rows.get(call_id) or {}).get("prompt") or {}
        if isinstance(prompt.get("tokenized"), Mapping):
            return list(prompt["tokenized"].get("ids") or [])
        tokenized = prompt.get("source") == "tokenize" or (
            ((self._rows.get(call_id) or {}).get("checks") or {}).get("tokenize_equals_stream") is not None)
        return ids if tokenized else None


def _build(root: Path, images: _Images, tokenize: Any = None, tokenizer: Mapping[str, Any] | None = None,
           host_paths: HostPaths | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Every exported row of one run, and the manifest's description of it (without the check and file hashes).

    ``tokenize(model_call_id, body)`` gives a policy call's prompt ids when its
    stream has none (a ``token_ids.Tokenizer``, or a check's ``_Replay``);
    ``tokenizer`` describes it in the manifest. ``host_paths`` redacts every
    row (the module docstring); a redacted request is not tokenized.
    """

    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else None
    trace = _trace_status(root, evaluation)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    allow_grasp = _allows_public_grasp_contact_pose(plan)
    events = read_agent_events(root)
    script_calls = script_call_ids(plan, events)
    model_arguments = model_tool_arguments(events)
    requests, responses, streams, ordered, tokens = [], [], [], [], []
    chain = token_ids.TokenChain(tokenize)
    redaction = None if host_paths is None else Redaction(host_paths)
    for index, (request_event, response_event) in enumerate(_policy_calls(events)):
        call_id = str(request_event.get("model_call_id"))
        payload = request_event.get("payload") if isinstance(request_event.get("payload"), Mapping) else {}
        row, body, redacted = _request_row(root, images, index, call_id, request_event, payload, allow_grasp,
                                           script_calls, model_arguments, redaction)
        row["answered"] = response_event is not None
        requests.append(row)
        ordered.append((call_id, list(row["request"].get("messages") or [])))
        tokenizable = row["protocol"] == OPENAI_PROTOCOL and not redacted
        if response_event is None:  # an incomplete run's last request: exported, marked, with no reply
            responses.append({"index": index, "model_call_id": call_id, "answered": False, "accepted": None,
                              "message": None, "provider_reply": None})
            streams.append({"index": index, "model_call_id": call_id, "response_sha256": None, "response": None})
            tokens.append(chain.row(call_id, body, None, None, tokenizable=tokenizable))
            continue
        record, stream = _reply(root, call_id, response_event, allow_grasp, model_arguments, redaction,
                                prompt_redacted=redacted)
        responses.append({"index": index, "model_call_id": call_id, "answered": True,
                          "accepted": record.pop("accepted", True) is not False, **record})
        streams.append({"index": index, "model_call_id": call_id, **stream})
        tokens.append(chain.row(call_id, body, stream["response"], record["provider_reply"], tokenizable=tokenizable))
    compactor = []
    for index, (attempt, reply) in enumerate(_compactor_calls(events)):
        call_id = str(attempt.get("model_call_id"))
        payload = attempt.get("payload") if isinstance(attempt.get("payload"), Mapping) else {}
        row, _, redacted = _request_row(root, images, index, call_id, attempt, payload, allow_grasp, script_calls,
                                        model_arguments, redaction, "compactor.jsonl")
        record, stream = _reply(root, call_id, reply, allow_grasp, model_arguments, redaction, "compactor.jsonl",
                                redacted)
        compactor.append({**row, "call_kind": payload.get("call_kind"),
                          "reply_event": None if reply is None else reply.get("type"),
                          "reply": record, **stream})
    epochs, placement = _epochs(ordered)
    for row, place in zip(requests, placement):
        row["epoch"], row["epoch_prefix"], row["epoch_tail"] = place["epoch"], place["prefix"], place["tail"]
    index = sorted(images.index.values(), key=lambda item: item["path"])
    eligibility = (evaluation or {}).get("learning_eligibility") or {}
    summary = {
        "schema": WIRE_EXPORT_SCHEMA,
        "run_id": result.get("run_id"),
        "source_commit": plan.get("source", {}).get("commit"),
        "trace": trace,
        "official_success": result.get("official_success"),
        "termination": result.get("termination"),
        "evaluation_status": None if evaluation is None else evaluation.get("status"),
        # Consumers filter training data on this: a sealed run with a valid evaluation that found it eligible.
        "learning_eligible": bool(trace["sealed"] and isinstance(evaluation, Mapping)
                                  and evaluation.get("status") == "valid" and eligibility.get("eligible") is True),
        "calls": len(requests),
        "unanswered_calls": sum(1 for row in requests if not row["answered"]),
        "accepted_calls": sum(1 for item in responses if item["accepted"]),
        "compactor_calls": len(compactor),
        "epochs": len(epochs),
        "images": {
            "scope": "every image sent to a model; recorded images whose bytes no request carried are only counted",
            "files": len(index), "bytes": sum(item["bytes"] for item in index),
            "recorded_unsent": _unsent_images(events, {item["sha256"] for item in index}),
        },
        "tokens": token_ids.summarize(tokens, tokenizer),
    }
    tables = {"requests": requests, "responses": responses, "streams": streams, "compactor": compactor,
              "epochs": epochs, "images": index, "tokens": tokens}
    if redaction is not None:
        summary["redaction"] = redaction.record()
        if redaction.rows:  # the event chain hashes the sealed requests and streams
            summary["trace"] = {key: value for key, value in trace.items() if key != "final_event_sha256"}
        left = redaction.left(tables)
        if left is not None:
            raise TrajectoryError(f"a host path is left in {left} after redaction, split across stream chunks; "
                                  "this export cannot redact it")
    return tables, summary


def export_wire(run_root: Path, destination: Path, *, tokenizer: token_ids.Tokenizer | None = None,
                host_paths: HostPaths | None = None) -> Path:
    """Write the wire export of one run (see the module docstring) and prove it rebuilds.

    With ``tokenizer`` every OpenAI-protocol policy call whose stream has no
    prompt ids is tokenized by that engine, and one whose stream has them is
    tokenized too, to compare. With ``host_paths`` (``HostPaths.for_run``) the
    export is redacted and records it in the manifest's ``redaction``.
    """

    root = run_root.resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise TrajectoryError(f"export destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        tables, summary = _build(root, _Images(staging / "images"), tokenizer,
                                 None if tokenizer is None else tokenizer.describe(), host_paths)
        for name, rows in tables.items():
            _jsonl(staging / f"{name}.jsonl", rows)
        check = check_wire_export(staging, staged=True, redaction=summary.get("redaction"))
        if not check["identical"]:
            raise TrajectoryError("wire export does not rebuild the archived requests: " + json.dumps(check["failures"][:3]))
        _atomic_json(staging / "manifest.json", {
            **summary,
            "replay_check": check,
            "files": [{"path": name, "sha256": file_sha256(staging / name), "bytes": (staging / name).stat().st_size}
                      for name in FILES],
        })
        _open_permissions(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def check_wire_export(export_root: Path, run_root: Path | None = None, *, staged: bool = False,
                      redaction: Mapping[str, Any] | None = None, redact_paths: Iterable[str] = ()) -> dict[str, Any]:
    """Rebuild every request, stream, provider reply and image index entry; compare with the recorded hashes.

    A finished export must have its manifest and every file it lists. With
    ``run_root`` every row and the manifest's description are derived again from
    the source run and must be equal, so an edit to ORA's own record fields or to
    the manifest is found too. ``staged`` is the export's own check, before the
    manifest is written, given the export's ``redaction``; otherwise it is the
    manifest's, and ``run_root`` is redacted again with the prefixes
    ``HostPaths.for_run`` finds (``redact_paths``: the ``--redact-path`` prefixes
    the export was given). A redacted row is counted under its check's name with
    ``_redacted``.
    """

    root = Path(export_root)
    failures: list[dict[str, Any]] = []
    counts = {"requests": 0, "requests_from_epochs": 0, "wire_requests": 0, "streams": 0, "provider_replies": 0,
              "compactor_requests": 0, "compactor_streams": 0, "images": 0, "tokens": 0}
    changed: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    matched: set[tuple[Any, Any, Any]] = set()

    def fail(check: str, call_id: Any, detail: str | None = None) -> None:
        failures.append({"check": check, "model_call_id": call_id, **({"detail": detail} if detail else {})})

    def same(kind: str, call_id: str, build: Any, expected: str | None, place: tuple[str, str] | None = None) -> None:
        item = None if place is None else changed.get((place[0], call_id, place[1]))
        if item is not None:  # a redacted row: the sha256 of its redacted form; its sealed one is not shipped
            matched.add((place[0], call_id, place[1]))
            if expected is not None:
                fail("redaction_record", call_id, f"{place[0]} {place[1]}: a redacted row keeps its sealed sha256")
            kind, expected = f"{kind}_redacted", item.get("redacted_sha256")
            counts.setdefault(kind, 0)
        try:
            value = build()
        except OSError as exc:  # a missing or unreadable image file
            fail(kind, call_id, f"{type(exc).__name__}: {exc.filename or exc}")
            return
        if expected is not None and sha256_bytes(canonical_bytes(value)) == expected:
            counts[kind] += 1
        else:
            fail(kind, call_id)

    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if not staged:
        for name in ("manifest.json", *FILES):
            if not (root / name).is_file():
                fail("missing_file", None, name)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            listed = [entry.get("path") for entry in manifest.get("files", [])]
            if listed != list(FILES):
                fail("manifest_file", None, "files lists " + ", ".join(map(str, listed)))
            for entry in manifest.get("files", []):
                if set(entry) != {"path", "sha256", "bytes"}:
                    fail("manifest_keys", None, f"files entry {entry.get('path')}")
                path = root / str(entry.get("path"))
                if (not path.is_file() or file_sha256(path) != entry.get("sha256")
                        or path.stat().st_size != entry.get("bytes")):
                    fail("manifest_file", None, str(entry.get("path")))
            keys = MANIFEST_KEYS | ({"redaction"} if "redaction" in manifest else set())
            if set(manifest) != keys:
                fail("manifest_keys", None, ", ".join(sorted(set(manifest) ^ keys)))
            # The recorded check is the export's own: on intact files it recomputes to the same counts.
            recomputed = check_wire_export(root, staged=True, redaction=manifest.get("redaction"))
            if recomputed["identical"] and manifest.get("replay_check") != recomputed:
                fail("manifest_replay_check", None)
            # Nothing but the export's own files: an added file is an edit too.
            extra = sorted(item.name for item in root.iterdir()
                           if item.name not in {"manifest.json", "images", *FILES})
            if extra:
                fail("extra_file", None, ", ".join(extra))
        redaction = manifest.get("redaction")
    for item in (redaction or {}).get("rows") or []:
        changed[(item.get("file"), item.get("model_call_id"), item.get("field"))] = item
    requests = _read_jsonl(root / "requests.jsonl")
    epochs = {item["epoch"]: item["messages"] for item in _read_jsonl(root / "epochs.jsonl")}
    compactor = _read_jsonl(root / "compactor.jsonl")
    sent: dict[str, list[str]] = {}
    for row in requests:
        call_id = row["model_call_id"]
        same("requests", call_id, lambda: internalize(row["request"], root), row["request_sha256"],
             ("requests.jsonl", "request"))
        tail = [row["epoch_tail"]] if row.get("epoch_tail") is not None else []
        messages = epochs.get(row.get("epoch"), [])[:row.get("epoch_prefix", 0)] + tail
        same("requests_from_epochs", call_id, lambda: internalize({**row["request"], "messages": messages}, root),
             row["request_sha256"], ("requests.jsonl", "request"))
        if "wire_request" in row:
            same("wire_requests", call_id, lambda: internalize(row["wire_request"], root), row["wire_request_sha256"],
                 ("requests.jsonl", "wire_request"))
    for row in [*requests, *compactor]:
        for ref in image_refs([row.get("request"), row.get("wire_request")]):
            calls = sent.setdefault(ref, [])
            if row["model_call_id"] not in calls:
                calls.append(row["model_call_id"])
    replies = {row["model_call_id"]: row for row in _read_jsonl(root / "responses.jsonl")}
    for row in _read_jsonl(root / "streams.jsonl"):
        call_id = row["model_call_id"]
        if row["response"] is None:
            continue
        same("streams", call_id, lambda: row["response"], row["response_sha256"], ("streams.jsonl", "response"))
        if replies.get(call_id, {}).get("provider_reply") == provider_reply(row["response"]):
            counts["provider_replies"] += 1
        else:
            fail("provider_replies", call_id)
    for row in compactor:
        call_id = row["model_call_id"]
        same("compactor_requests", call_id, lambda: internalize(row["request"], root), row["request_sha256"],
             ("compactor.jsonl", "request"))
        if "wire_request" in row:
            same("wire_requests", call_id, lambda: internalize(row["wire_request"], root), row["wire_request_sha256"],
                 ("compactor.jsonl", "wire_request"))
        if row.get("response") is not None:
            same("compactor_streams", call_id, lambda: row["response"], row["response_sha256"],
                 ("compactor.jsonl", "response"))
            if row.get("reply", {}).get("provider_reply") != provider_reply(row["response"]):
                fail("provider_replies", call_id)
    for key in sorted(set(changed) - matched, key=str):
        fail("redaction_record", key[1], f"{key[0]} {key[2]}: no such row")
    token_rows = _read_jsonl(root / "tokens.jsonl")
    stream_rows = {row["model_call_id"]: row for row in _read_jsonl(root / "streams.jsonl")}
    if len(token_rows) != len(requests):
        fail("tokens", None, f"{len(token_rows)} rows for {len(requests)} requests")
    try:
        chain = token_ids.TokenChain(_Replay(token_rows))
        for row, recorded in zip(requests, token_rows):
            call_id = row["model_call_id"]
            stream = stream_rows.get(call_id) or {}
            expected = chain.row(call_id, row["request"], stream.get("response"),
                                 (replies.get(call_id) or {}).get("provider_reply"),
                                 tokenizable=row.get("protocol") == OPENAI_PROTOCOL)
            if canonical_bytes(expected) == canonical_bytes(recorded):
                counts["tokens"] += 1
            else:
                fail("tokens", call_id)
    except (TrajectoryError, KeyError, TypeError, AttributeError) as exc:
        fail("tokens", None, f"{type(exc).__name__}: {exc}")
    index = {item["path"]: item for item in _read_jsonl(root / "images.jsonl")}
    if (root / "images").is_dir():
        for item in sorted((root / "images").iterdir()):
            if f"images/{item.name}" not in index:
                fail("extra_file", None, f"images/{item.name}")
    for path in sorted(set(index) | set(sent)):
        entry = index.get(path)
        file = root / path
        if entry is None or entry.get("calls") != sent.get(path):
            fail("images", None, path)
        elif not file.is_file() or file_sha256(file) != entry["sha256"] or file.stat().st_size != entry["bytes"]:
            fail("images", None, path)
        else:
            counts["images"] += 1
    if run_root is not None:
        host_paths = None if redaction is None else HostPaths.for_run(Path(run_root).resolve(), redact_paths)
        try:
            tables, summary = _build(Path(run_root).resolve(), _Images(None), _Replay(token_rows),
                                     (manifest.get("tokens") or {}).get("tokenizer"), host_paths)
        except TrajectoryError as exc:  # the recorded token rows do not chain; already a tokens failure
            fail("source_tokens", None, str(exc))
            return {"identical": False, "calls": len(requests), **counts, "failures": failures}
        counts["source_rows"] = 0
        for name, rows in tables.items():
            exported = _read_jsonl(root / f"{name}.jsonl")
            for position, (expected, actual) in enumerate(zip(rows, exported)):
                if canonical_bytes(expected) == canonical_bytes(actual):
                    counts["source_rows"] += 1
                else:
                    fail(f"source_{name}", expected.get("model_call_id"), f"row {position}")
            if len(rows) != len(exported):
                fail(f"source_{name}", None, f"{len(exported)} rows, the run gives {len(rows)}")
        differing = sorted(key for key, value in summary.items() if manifest.get(key) != value)
        if "redaction" in differing:
            differing.remove("redaction")
            fail("redaction_rules", None, "the run redacted again here does not give the export's redaction: check "
                 "as the account that exported it, with each --redact-path it was given")
        if differing:
            fail("source_manifest", None, ", ".join(differing))
    report = {"identical": not failures, "calls": len(requests), **counts, "failures": failures}
    if redaction is not None and not staged:
        report["redaction"] = statement(redaction, len(requests), against_run=run_root is not None and not failures)
    return report


__all__ = ["WIRE_EXPORT_SCHEMA", "check_wire_export", "export_wire", "image_refs", "internalize", "provider_reply"]
