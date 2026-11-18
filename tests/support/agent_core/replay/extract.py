"""Turn one sealed ORA Native run into a self-contained replay fixture.

The sealed run is read-only evidence; this module only reads it (and streams
its ``native/provider-requests.tar.zst`` through ``zstd`` into a separate
archive directory). Everything the replay needs is written under the fixture
directory as JSON plus content-addressed files, so a non-Python consumer can
read it:

``fixture.json``          identity, counts and the extraction self-checks
``plan.json``             the resolved RunPlan (verbatim copy)
``result.json``           the run result delivered as ``run_termination``
``compiled-policy/``      the compiled prompt layers (verbatim copies)
``script.json``           the ordered Supervisor operations on the session
``contexts/step-N.json``  the exact ``context`` pointer published per step
``run-files/``            run-relative files the Supervisor had published
                          before each step (policy-state snapshots, camera
                          files, frozen current-visual / action-view images)
``tool-calls.json``       every proposed tool call; for dispatched ones the
                          result the live ToolRuntime returned
``artifacts/``            tool images and session-delivered tool results
``provider-calls.json``   every model call: recorded response and request ids
``provider/``             the recorded SSE events per model call
``requests/``             the archived wire and canonical request bytes
``expected/``             the original agent trace, Context-Artifact index and
                          agent-artifact inventory, for comparison

Run ``python -m tests.support.agent_core.replay.extract --help`` for options.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .runtime import (
    FIXTURE_SCHEMA, REPO_ROOT, ReplayMismatch, _install, build_offline_tool_runtime,
    build_provider, canonical_bytes, canonical_sha256, read_json, sha256_bytes, write_json,
)

#: Where the drivers write when not told: ``ORA_TEST_SCRATCH`` or the system temporary directory.
DEFAULT_SCRATCH = Path(os.environ.get("ORA_TEST_SCRATCH") or tempfile.gettempdir()) / "ora-agent-core-replay"

#: Run-relative trees whose files are immutable once published; anything the
#: session may read from disk for one step lives under one of them.
IMMUTABLE_RUN_PREFIXES = (
    "policy-state/steps/",
    "policy-observations/",
    "sandbox/workspace/inputs/current-visual/",
    "sandbox/workspace/inputs/action-observations/",
)


class SealedRun:
    """Read-only accessors over one sealed run directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.plan = read_json(self.root / "plan.json")
        self.result = read_json(self.root / "result.json")
        self.agent_events = [
            json.loads(line)
            for line in (self.root / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.run_events = [
            json.loads(line)
            for line in (self.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seal = read_json(self.root / "seal.json")
        self.inventory = {item["path"]: item for item in seal["inventory"]}
        self.sealed_source_commit = seal.get("sealed_source_commit")
        manifest = self.root / "native/provider-requests.manifest.json"
        self.request_manifest = read_json(manifest) if manifest.is_file() else None


def extract_request_archive(run: SealedRun, destination: Path) -> dict[str, Any]:
    """Stream the provider-request archive into ``destination``, member by member.

    Only the members the manifest lists are written, each verified against its
    recorded sha256; an existing complete extraction is reused.
    """

    manifest = run.request_manifest
    if manifest is None:
        return {"status": "absent", "members": 0}
    members = {item["path"]: item for item in manifest["members"]}
    destination = destination.resolve()
    if all(
        (destination / path).is_file()
        and sha256_bytes((destination / path).read_bytes()) == item["sha256"]
        for path, item in members.items()
    ):
        return {"status": "reused", "members": len(members), "root": str(destination)}
    archive = run.root / manifest["archive"]
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["archive_sha256"]:
        raise ReplayMismatch("provider request archive sha256 differs from its manifest")
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ReplayMismatch("zstd is required to read native/provider-requests.tar.zst")
    written = 0
    process = subprocess.Popen(
        [zstd, "-d", "--long=31", "-c", str(archive)], stdout=subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for member in tar:
                item = members.get(member.name)
                if item is None or not member.isfile():
                    raise ReplayMismatch(f"unexpected provider archive member: {member.name}")
                handle = tar.extractfile(member)
                assert handle is not None
                data = handle.read()
                if sha256_bytes(data) != item["sha256"] or len(data) != item["bytes"]:
                    raise ReplayMismatch(f"provider archive member differs: {member.name}")
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.wait() != 0:
            raise ReplayMismatch("zstd failed to decompress the provider request archive")
    if written != len(members):
        raise ReplayMismatch("provider request archive is missing members")
    return {"status": "extracted", "members": written, "root": str(destination)}


class _Blobs:
    """Artifact bytes from the sealed run or its extracted request archive."""

    def __init__(self, run: SealedRun, archive_root: Path) -> None:
        self.run = run
        self.archive_root = archive_root

    def read(self, ref: Mapping[str, Any]) -> bytes:
        relative = str(ref["path"])
        for root in (self.run.root, self.archive_root):
            path = root / relative
            if path.is_file():
                data = path.read_bytes()
                if sha256_bytes(data) != ref["sha256"]:
                    raise ReplayMismatch(f"artifact sha256 differs: {relative}")
                return data
        raise ReplayMismatch(f"artifact is neither sealed nor archived: {relative}")

    def json(self, ref: Mapping[str, Any]) -> Any:
        return json.loads(self.read(ref))


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _file_entry(destination: Path, relative: str, data: bytes) -> dict[str, Any]:
    _write_bytes(destination / relative, data)
    return {"path": relative, "sha256": sha256_bytes(data), "bytes": len(data)}


# -- provider calls -----------------------------------------------------------


def _reply_message(response: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """The assistant message and output rejection of one provider reply (as the core records them)."""

    message = response.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise ReplayMismatch("replayed provider response lacks an assistant message")
    rejection = response.get("output_rejection")
    if rejection is not None and not isinstance(rejection, Mapping):
        raise ReplayMismatch("replayed provider output rejection must be an object")
    return copy.deepcopy(dict(message)), rejection


def _response_payload(
    response: Mapping[str, Any], *, assistant: dict[str, Any], usage: Any,
    output_rejection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The ``provider_response`` trace payload an agent core records for one reply."""

    return {
        "message": assistant,
        "usage": usage,
        "finish_reason": response.get("finish_reason"),
        "attempts": response.get("attempts", 1),
        "accepted": output_rejection is None,
        **({"output_rejection": copy.deepcopy(dict(output_rejection))}
           if isinstance(output_rejection, Mapping) else {}),
    }


def _provider_calls(
    run: SealedRun, blobs: _Blobs, destination: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ora.agent.evidence.agent_events import library_op

    events = run.agent_events
    responses = {e["model_call_id"]: e for e in events if e["type"] == "provider_response"}
    tool_ids: dict[str, list[str]] = {}
    for event in events:
        if event["type"] == "tool_call_proposed" and not library_op(event):
            tool_ids.setdefault(event["model_call_id"], []).append(event["tool_call_id"])
    policy = run.plan["policy"]
    provider = build_provider(policy, run_id=str(events[0]["run_id"]))
    calls: list[dict[str, Any]] = []
    reproduced = 0
    try:
        for request in (e for e in events if e["type"] == "provider_request"):
            model_call_id = request["model_call_id"]
            response = responses[model_call_id]
            wire = blobs.read(request["artifact_ref"])
            canonical = blobs.read(request["payload"]["canonical_request_ref"])
            raw = blobs.read(response["artifact_ref"])
            entry = {
                "model_call_id": model_call_id,
                "observation_step": request["observation_step"],
                "wire_request": _file_entry(destination, f"requests/{model_call_id}.wire.json", wire),
                "canonical_request": _file_entry(
                    destination, f"requests/{model_call_id}.canonical.json", canonical,
                ),
                "raw_events": _file_entry(
                    destination, f"provider/{model_call_id}.raw-events.json", raw,
                ),
                "response": {
                    key: copy.deepcopy(response["payload"].get(key))
                    for key in ("finish_reason", "usage", "attempts", "accepted")
                },
                "response_payload_sha256": canonical_sha256(response["payload"]),
                "tool_call_ids": tool_ids.get(model_call_id, []),
            }
            # The real OpenAI-compatible accumulation over the recorded events
            # must reproduce the recorded provider_response payload exactly.
            recorded = json.loads(raw)
            _install(provider, lambda body, recorded=recorded: (copy.deepcopy(recorded), time.monotonic()))
            replayed = provider.complete(
                request={"model": policy["model"], "messages": [], "stream": True},
                model_call_id=model_call_id,
            )
            assistant, rejection = _reply_message(replayed)
            payload = _response_payload(
                replayed, assistant=assistant, usage=replayed.get("usage"),
                output_rejection=rejection,
            )
            entry["response_reproduced_by_accumulation"] = (
                canonical_sha256(payload) == entry["response_payload_sha256"]
            )
            reproduced += entry["response_reproduced_by_accumulation"]
            calls.append(entry)
    finally:
        provider.close()
    return calls, {"provider_responses_reproduced": reproduced, "provider_calls": len(calls)}


# -- tool calls -------------------------------------------------------------


def _image_suffix(media_type: str) -> str:
    return ".png" if media_type == "image/png" else ".img"


def _tool_execution(run: SealedRun, proposal: Mapping[str, Any]) -> dict[str, Any] | None:
    """A tool call's ledger on a synchronous surface, as the Supervisor's executor closed it; ``None`` elsewhere."""

    from ora.orchestrator.episode.tool_execution_log import ToolExecutionLog

    log = ToolExecutionLog(run.root, proposal["tool_call_id"], proposal["model_call_id"])
    return log.recover() if log.path.is_file() else None


def _tool_calls(
    run: SealedRun, blobs: _Blobs, destination: Path, tools: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ora.agent.evidence.agent_events import library_op
    from ora.agent.memory.artifact_openers import ARTIFACT_READ_NAMES
    from ora.agent.session_core.gateway import NativeToolGateway

    events = run.agent_events
    results ={e["tool_call_id"]: e for e in events if e["type"] == "tool_result"}
    actions: dict[str, list[dict[str, Any]]] = {}
    images: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["type"] == "tool_image":
            images.setdefault(event["tool_call_id"], []).append(event)
        elif event["type"] == "action_proposed":
            actions.setdefault(event["tool_call_id"], []).append(event)
    runtime_events = [e for e in run.run_events if e["type"] in {"tool_result", "tool_error"}]
    gateway = NativeToolGateway(tools)
    image_key_default = (
        "images_base64" if tools.surface.runtime_id == "native_atomic_causal" else None
    )
    cursor = 0
    calls: list[dict[str, Any]] = []
    verified = 0
    approximated = 0
    for proposal in (e for e in events if e["type"] == "tool_call_proposed" and not library_op(e)):
        payload = proposal["payload"]
        name, arguments = payload["name"], payload["arguments"]
        tool_call_id = proposal["tool_call_id"]
        delivered_ref = results[tool_call_id]["artifact_ref"]
        delivered = blobs.read(delivered_ref)
        delivered_entry = _file_entry(
            destination, f"artifacts/{delivered_ref['sha256']}.json", delivered,
        )
        record: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "model_call_id": proposal["model_call_id"],
            "observation_step": proposal["observation_step"],
            "name": name,
            "proposed_arguments": copy.deepcopy(arguments),
            "effect": payload.get("effect"),
            "batch_valid": payload.get("batch_valid"),
            "visual_feedback": payload.get("visual_feedback"),
            "dispatched": False,
            "source": None,
            "delivered_result": delivered_entry,
        }
        raw: dict[str, Any] | None = None
        if arguments is not None and name != "invalid_tool_call":
            if name in ARTIFACT_READ_NAMES or (name == "observe" and arguments.get("ref") is not None):
                record.update(dispatched=True, source="context_artifact_reader",
                              runtime_arguments=dict(gateway.arguments(name, arguments)))
            else:
                normalized = dict(gateway.arguments(name, arguments))
                candidate = runtime_events[cursor] if cursor < len(runtime_events) else None
                if (
                    candidate is not None and candidate["tool"] == name
                    and canonical_bytes(candidate["arguments"]) == canonical_bytes(normalized)
                ):
                    cursor += 1
                    record.update(dispatched=True, source="tool_runtime",
                                  runtime_arguments=copy.deepcopy(candidate["arguments"]))
                    if candidate["type"] == "tool_error":
                        record["outcome"] = {"error": copy.deepcopy(candidate["error"])}
                        raw = {"ok": False, "error": copy.deepcopy(candidate["error"])}
                    else:
                        result = copy.deepcopy(candidate["result"])
                        delivered_json = json.loads(delivered)
                        if "_runtime" in delivered_json:
                            result["_runtime"] = copy.deepcopy(delivered_json["_runtime"])
                        ordered = sorted(images.get(tool_call_id, []),
                                         key=lambda event: event["payload"]["image_index"])
                        image_entries = []
                        encoded = []
                        for event in ordered:
                            ref = event["artifact_ref"]
                            data = blobs.read(ref)
                            entry = _file_entry(
                                destination,
                                f"artifacts/{ref['sha256']}{_image_suffix(ref['media_type'])}",
                                data,
                            )
                            entry["media_type"] = ref["media_type"]
                            image_entries.append(entry)
                            encoded.append(base64.b64encode(data).decode("ascii"))
                        key = image_key_default or (
                            "image_base64" if len(encoded) == 1 else "images_base64"
                        )
                        outcome: dict[str, Any] = {"result": copy.deepcopy(result)}
                        if encoded:
                            outcome.update(images=image_entries, image_key=key)
                            result[key] = encoded if key == "images_base64" else encoded[0]
                        record["outcome"] = outcome
                        raw = result
                    proposed = actions.get(tool_call_id, ())
                    execution = _tool_execution(run, proposal)
                    if execution is not None:  # the tool ran its actions itself, inside this call
                        record["execution"] = execution
                        record["inline_actions"] = [blobs.json(event["artifact_ref"]) for event in proposed]
                    elif proposed:
                        record["submitted_action"] = blobs.json(proposed[-1]["artifact_ref"])
        if raw is None:
            # Not served by the ToolRuntime: the session's own refusal (or an
            # Artifact read). Only used to keep the repeated-failure tracker in
            # step for the verification below.
            raw = json.loads(delivered)
            approximated += 1
        # Verification: the gateway must turn the reconstructed raw result into
        # exactly the native_tool_result the session recorded.
        gateway_result = dict(gateway.record_proposed_result(name, proposal, raw))
        gateway_result.pop("image_base64", None)
        gateway_result.pop("images_base64", None)
        record["delivered_result_reproduced"] = (
            sha256_bytes(canonical_bytes(gateway_result)) == delivered_ref["sha256"]
        )
        verified += bool(record["delivered_result_reproduced"] and record["source"] == "tool_runtime")
        calls.append(record)
    if cursor != len(runtime_events):
        raise ReplayMismatch("ToolRuntime events remain unpaired with proposed tool calls")
    dispatched = sum(1 for call in calls if call["dispatched"])
    return calls, {
        "tool_calls": len(calls),
        "dispatched_tool_calls": dispatched,
        "tool_results_reconstructed": verified,
        "tool_results_not_from_runtime": approximated,
        "combined_feedback_batches": sum(
            1 for call in calls if call.get("visual_feedback") == "combined"
        ),
    }


#: Placeholder for the source run root inside a receipt; the replay substitutes its own.
RUN_ROOT_PLACEHOLDER = "${RUN_ROOT}"


def _rebase_receipt(receipt: Any, run: "SealedRun") -> tuple[Any, set[str]]:
    """Make absolute source-run paths in a receipt relative, and list the files they need.

    A prepared-grasp ``grasp_execute`` receipt names its intermediate
    (post-close, pre-lift) observation by an absolute ``observation_root``; the
    Native session reads that observation's manifest and RGB files. The replay
    runs under a different root, so the prefix becomes ``${RUN_ROOT}`` and the
    sealed files under every such observation root travel with the fixture.
    """

    prefix = str(run.root) + "/"
    files: set[str] = set()

    def visit(item: Any, key: str | None = None) -> Any:
        if isinstance(item, Mapping):
            return {name: visit(value, name) for name, value in item.items()}
        if isinstance(item, list):
            return [visit(value) for value in item]
        if isinstance(item, str) and item.startswith(prefix):
            relative = item[len(prefix):]
            if key == "observation_root":
                under = relative.rstrip("/") + "/"
                files.update(path for path in run.inventory if path.startswith(under))
            elif relative in run.inventory:
                files.add(relative)
            return RUN_ROOT_PLACEHOLDER + "/" + relative
        return item

    return visit(receipt), files


# -- Supervisor script and published disk state -------------------------------


def _referenced_run_files(value: Any, run_root: Path) -> set[str]:
    """Run-relative immutable files a published structure refers to."""

    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and 0 < len(item) < 512 and not item.startswith("/"):
            if ".." in Path(item).parts:
                return
            for candidate in (item, f"sandbox/workspace/{item}"):
                if candidate.startswith(IMMUTABLE_RUN_PREFIXES) and (run_root / candidate).is_file():
                    found.add(candidate)

    visit(value)
    return found


def _observation_terminal_flag(canonical_request: bytes) -> bool | None:
    """``terminal_lifecycle`` of the newest causal observation envelope in one request."""

    request = json.loads(canonical_request)
    for message in reversed(request.get("messages", [])):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list) or not content:
            continue
        first = content[0]
        if not isinstance(first, Mapping) or first.get("type") != "text":
            continue
        try:
            envelope = json.loads(first.get("text", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(envelope, Mapping) and "current_actual" in envelope:
            # observation_delivery_trim_v2 omits ``terminal_lifecycle: false``.
            return bool(envelope.get("terminal_lifecycle", False))
    return None


def _script(
    run: SealedRun, blobs: _Blobs, destination: Path, provider_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ora.agent.session_core.constants import NATIVE_CAUSAL_SLOTS

    plan = run.plan
    policy = plan["policy"]
    composition = policy.get("context_composition")
    task_instruction = str(plan.get("task_instruction", plan["task"])).strip()
    main_camera = str(plan["runtime"].get("main_camera", "head_camera"))
    observations = [e for e in run.run_events if e["type"] == "observation"]
    statuses = [e for e in run.run_events if e["type"] == "environment_status"]
    if len(observations) != len(statuses):
        raise ReplayMismatch("observation and environment_status events do not pair")
    by_step: dict[int, dict[str, Any]] = {}
    for observation, status in zip(observations, statuses):
        step = int(observation["policy_state"]["step"])
        by_step[step] = {"observation": observation, "status": status["record"]}
    canonical_by_call = {
        call["model_call_id"]: (destination / call["canonical_request"]["path"]).read_bytes()
        for call in provider_calls
    }
    max_model_calls = int(plan["budgets"]["max_model_calls"])
    ops: list[dict[str, Any]] = []
    published_files: set[str] = set()
    seal_verified = 0
    terminal_flags_checked = 0
    model_calls_before = 0
    turn: dict[str, Any] | None = None
    events = run.agent_events
    for index, event in enumerate(events):
        kind = event["type"]
        if kind == "fresh_observation_published":
            step = int(event["observation_step"])
            entry = by_step[step]
            observation_event = entry["observation"]
            context = observation_event["context"]
            context_path = f"contexts/step-{step:04d}.json"
            write_json(destination / context_path, context, indent=1)
            state_path = context["policy_state"]["path"]
            snapshot = read_json(run.root / state_path)
            files = sorted(
                (_referenced_run_files(context, run.root)
                 | _referenced_run_files(snapshot, run.root)
                 | {state_path}) - published_files
            )
            for relative in files:
                data = (run.root / relative).read_bytes()
                sealed = run.inventory.get(relative)
                if sealed is None or sealed["sha256"] != sha256_bytes(data):
                    raise ReplayMismatch(f"published file is not sealed evidence: {relative}")
                seal_verified += 1
                _write_bytes(destination / "run-files" / relative, data)
            published_files.update(files)
            observation_id = str(observation_event["observation_id"])
            manifest_path = run.root / f"policy-observations/{observation_id}/observation.json"
            manifest_bytes = manifest_path.read_bytes()
            observed = {**json.loads(manifest_bytes), "main_camera": main_camera}
            control = {
                "generation": hashlib.sha256(
                    json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:16],
                "observation_id": observation_id,
                "observation_manifest": "sandbox/workspace/inputs/current/observation.json",
                "observation_sha256": sha256_bytes(manifest_bytes),
                "open": True,
                "policy_state": copy.deepcopy(context["policy_state"]),
                "source_root": "sandbox/workspace/inputs/current",
            }
            ops.append({
                "op": "publish_step", "step": step, "files": files,
                "control_current_observation": control,
                "control_run_relative_keys": ["observation_manifest", "source_root"],
                "remove_control_files": ["submitted-action.json"],
            })
            ops.append({
                "op": "fresh_observation", "step": step, "observation_id": observation_id,
                "context": context_path,
            })
            turn = None
        elif kind == "observation_received":
            step = int(event["observation_step"])
            entry = by_step[step]
            context_path = f"contexts/step-{step:04d}.json"
            terminal = bool(entry["status"]["terminal"])
            observation_id = str(entry["observation"]["observation_id"])
            base = {
                "step": step, "observation_id": observation_id,
                "task_id": str(plan["task"]), "task_instruction": task_instruction,
                "context": context_path,
                "image_path": f"sandbox/workspace/{entry['observation']['context']['main_image']}",
                "image_paths": [],
            }
            if event["payload"].get("provider_turn") is False:
                follow = next(
                    e for e in events[index + 1:]
                    if e["type"] == "observation_preserved_without_policy_turn"
                )
                ops.append({**base, "op": "preserve_observation",
                            "terminal_lifecycle": bool(follow["payload"]["terminal_lifecycle"]),
                            "allow_reminder": False})
                turn = None
                continue
            turn = {
                **base, "op": "run_observation",
                "interaction_phase": "terminal" if terminal else "observation_ready",
                "allow_reminder": (not terminal) and max_model_calls - model_calls_before >= 2,
                "terminal_lifecycle": terminal and composition in NATIVE_CAUSAL_SLOTS,
                "expected": {"model_call_ids": [], "action_id": None},
            }
            ops.append(turn)
        elif kind == "provider_request" and turn is not None:
            turn["expected"]["model_call_ids"].append(event["model_call_id"])
            model_calls_before += 1
            if len(turn["expected"]["model_call_ids"]) == 1:
                flag = _observation_terminal_flag(canonical_by_call[event["model_call_id"]])
                if flag is not None:
                    if flag != turn["terminal_lifecycle"]:
                        raise ReplayMismatch(
                            f"step {turn['step']}: terminal_lifecycle {turn['terminal_lifecycle']} "
                            f"differs from the archived request ({flag})"
                        )
                    terminal_flags_checked += 1
        elif kind == "action_proposed" and turn is not None:
            turn["expected"]["action_id"] = event["payload"]["action_id"]
        elif kind == "environment_receipt":
            receipt, receipt_files = _rebase_receipt(blobs.json(event["artifact_ref"]), run)
            for relative in sorted(receipt_files - published_files):
                data = (run.root / relative).read_bytes()
                sealed = run.inventory.get(relative)
                if sealed is None or sealed["sha256"] != sha256_bytes(data):
                    raise ReplayMismatch(f"receipt file is not sealed evidence: {relative}")
                seal_verified += 1
                _write_bytes(destination / "run-files" / relative, data)
            published_files.update(receipt_files)
            ops.append({"op": "environment_receipt", "receipt": receipt,
                        "files": sorted(receipt_files)})
            turn = None
        elif kind == "official_verdict":
            ref = event["artifact_ref"]
            ops.append({"op": "official_verdict", "verdict": None if ref is None else blobs.json(ref)})
            turn = None
        elif kind == "run_terminated":
            for key, value in event["payload"].items():
                if run.result.get(key) != value:
                    raise ReplayMismatch(f"result.json {key} differs from run_terminated")
            ops.append({"op": "run_termination", "result": "result.json"})
            turn = None
        elif kind == "session_closed":
            ops.append({"op": "close"})
            turn = None
        elif kind in {"operator_message", "session_interrupted"}:
            raise NotImplementedError(f"replay does not script {kind} events")
    final_control = read_json(run.root / "control/current-observation.json")
    last_publish = next(op for op in reversed(ops) if op["op"] == "publish_step")
    control_check = {
        key: last_publish["control_current_observation"][key] == final_control.get(key)
        for key in ("generation", "observation_id", "observation_sha256", "policy_state")
    }
    return ops, {
        "published_files_verified_against_seal": seal_verified,
        "terminal_lifecycle_flags_checked_against_requests": terminal_flags_checked,
        "final_control_current_observation_matches": control_check,
    }


def _expected(run: SealedRun, destination: Path, archive_root: Path) -> dict[str, Any]:
    shutil.copyfile(run.root / "agent-events.jsonl", destination / "expected/agent-events.jsonl")
    index = run.root / "native/context-artifacts/index.json"
    if index.is_file():
        shutil.copyfile(index, destination / "expected/context-artifacts-index.json")
    kinds: dict[str, str] = {}
    for event in run.agent_events:
        ref = event.get("artifact_ref")
        if isinstance(ref, Mapping):
            kinds.setdefault(str(ref["sha256"]), str(ref["kind"]))
        payload = event.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("canonical_request_ref"), Mapping):
            ref = payload["canonical_request_ref"]
            kinds.setdefault(str(ref["sha256"]), str(ref["kind"]))
    inventory: dict[str, dict[str, Any]] = {}
    for root in (run.root, archive_root):
        directory = root / "native/agent-artifacts"
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            relative = f"native/agent-artifacts/{path.name}"
            sealed = run.inventory.get(relative)
            inventory[path.name] = {
                "name": path.name,
                "sha256": path.name.split(".")[0],
                "bytes": path.stat().st_size,
                "kind_from_trace": kinds.get(path.name.split(".")[0]),
                "sealed": sealed is not None,
            }
    write_json(destination / "expected/agent-artifacts.json",
               {"artifacts": sorted(inventory.values(), key=lambda item: item["name"])}, indent=1)
    return {"expected_agent_artifacts": len(inventory)}


def extract_fixture(
    source_run: Path,
    destination: Path = DEFAULT_SCRATCH / "fixture",
    *,
    archive_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the replay fixture for ``source_run`` into ``destination``."""

    started = time.monotonic()
    run = SealedRun(source_run)
    destination = Path(destination).resolve()
    if destination.exists():
        if not overwrite and any(destination.iterdir()):
            raise ReplayMismatch(f"fixture destination is not empty: {destination}")
        if (destination / "fixture.json").is_file() or not any(destination.iterdir()):
            shutil.rmtree(destination)
        else:
            raise ReplayMismatch(f"refusing to replace a non-fixture directory: {destination}")
    destination.mkdir(parents=True)
    for directory in ("expected", "compiled-policy", "contexts", "run-files", "artifacts",
                      "provider", "requests"):
        (destination / directory).mkdir()
    archive_root = Path(archive_root or destination.parent / "archive").resolve()
    archive = extract_request_archive(run, archive_root)
    blobs = _Blobs(run, archive_root)

    for name in ("plan.json", "result.json"):
        shutil.copyfile(run.root / name, destination / name)
    for path in sorted((run.root / "compiled-policy").iterdir()):
        shutil.copyfile(path, destination / "compiled-policy" / path.name)

    session_started = next(e for e in run.agent_events if e["type"] == "session_started")
    with tempfile.TemporaryDirectory(prefix=".offline-runtime-", dir=destination) as scratch:
        scratch_root = Path(scratch)
        (scratch_root / "control").mkdir()
        tools = build_offline_tool_runtime(
            run.plan, run_root=scratch_root,
            manual_path=destination / "compiled-policy/tool-manual.md",
        )
        provider_calls, provider_checks = _provider_calls(run, blobs, destination)
        tool_calls, tool_checks = _tool_calls(run, blobs, destination, tools)
        surface_profile = tools.surface.profile_id
        surface_runtime = tools.surface.runtime_id
    ops, script_checks = _script(run, blobs, destination, provider_calls)
    expected_checks = _expected(run, destination, archive_root)

    write_json(destination / "provider-calls.json", {
        "schema": "ora.pi-spike.native-replay-provider-calls.v1", "calls": provider_calls,
    }, indent=1)
    write_json(destination / "tool-calls.json", {
        "schema": "ora.pi-spike.native-replay-tool-calls.v1", "calls": tool_calls,
    }, indent=1)
    write_json(destination / "script.json", {
        "schema": "ora.pi-spike.native-replay-script.v1", "ops": ops,
    }, indent=1)
    turns = [op for op in ops if op["op"] == "run_observation"]
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "source_run": str(run.root),
        "run_id": run.agent_events[0]["run_id"],
        "policy_session_id": session_started["policy_session_id"],
        "plan_sha256": run.plan.get("plan_sha256"),
        "sealed_source_commit": run.sealed_source_commit,
        "extracted_with_repo": str(REPO_ROOT),
        "task_id": str(run.plan["task"]),
        "task_instruction": str(run.plan.get("task_instruction", run.plan["task"])).strip(),
        "tool_surface": {"profile": surface_profile, "runtime_id": surface_runtime},
        "session_started": session_started["payload"],
        "counts": {
            "script_ops": len(ops),
            "policy_turns": len(turns),
            "terminal_turns": sum(1 for op in turns if op["terminal_lifecycle"]),
            "provider_calls": len(provider_calls),
            "agent_events": len(run.agent_events),
            "published_steps": sum(1 for op in ops if op["op"] == "publish_step"),
        },
        "checks": {**provider_checks, **tool_checks, **script_checks, **expected_checks},
        "request_archive": archive,
        "extraction_seconds": round(time.monotonic() - started, 3),
    }
    write_json(destination / "fixture.json", manifest, indent=1)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=os.environ.get("ORA_TEST_SEALED_LIBERO"),
                        required=not os.environ.get("ORA_TEST_SEALED_LIBERO"),
                        help="the sealed run (default: ORA_TEST_SEALED_LIBERO)")
    parser.add_argument("--out", type=Path, default=DEFAULT_SCRATCH / "fixture")
    parser.add_argument("--archive", type=Path, default=None,
                        help="where to extract provider-requests.tar.zst (default: <out>/../archive)")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = extract_fixture(
        arguments.source, arguments.out, archive_root=arguments.archive,
        overwrite=arguments.overwrite,
    )
    print(json.dumps({key: manifest[key] for key in ("counts", "checks", "extraction_seconds")},
                     indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
