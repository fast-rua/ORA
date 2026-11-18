"""Extract a context-level replay from one sealed Pi-core run (read-only on the run).

    python -m tests.support.agent_core.context_replay.extract RUN_DIR --out REPLAY_DIR

Written in Python: the inputs are large JSONL files and a zstd tar archive,
which stream here without loading whole, and the run's robot vocabulary
(``context_domain``) is a Python function of the source tree. The runner that
consumes the replay is TypeScript (``ora-pi/test/support/context_replay.ts``).

Run it with the run's own source tree on ``PYTHONPATH`` for a faithful replay:
the domain constants it records come from the tree it imports.

How the per-call history is rebuilt: the Pi session holds the ORA history as
``ora.history`` entries (messages appended in history order, images as
artifact references) and an ``ora.state`` entry at every flush. The core
flushes at the start of every request, before projecting it, and the call
counter advances only after the projection, so the last ``ora.state`` whose
``core.call_sequence`` is N records exactly the history length the projection
of model call N received (``history_length``) and the pipeline state it
started from. The trace (``agent-events.jsonl``) gives everything else that
crossed into the context manager between two calls: tool images registered as
Context Artifacts, compactor calls (with their recorded responses) and the
Provider usage each response reported.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import SCHEMA

DATA_URL = re.compile(r"^data:([^;]+);base64,(.*)$", re.S)
#: The zstd binary that reads a sealed run's request archive (``ORA_ZSTD`` names another).
ZSTD = os.environ.get("ORA_ZSTD") or shutil.which("zstd")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def skeleton(value: Any) -> Any:
    """Large data URLs replaced by their sha256 and length (as ora-pi/test/context/replay.ts does)."""
    if isinstance(value, str):
        if len(value) >= 1024 and value.startswith("data:"):
            return f"<data {hashlib.sha256(value.encode('utf-8')).hexdigest()} {len(value)}>"
        return value
    if isinstance(value, list):
        return [skeleton(item) for item in value]
    if isinstance(value, dict):
        return {key: skeleton(item) for key, item in value.items()}
    return value


class RunFiles:
    """Run-relative files, read from the run or from its sealed request archive."""

    def __init__(self, run_dir: Path, scratch: Path) -> None:
        self.run_dir = run_dir
        self.scratch = scratch
        self.extracted: dict[str, Path] = {}
        manifest_path = run_dir / "native/provider-requests.manifest.json"
        self.archived: set[str] = set()
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.archive = run_dir / manifest["archive"]
            self.archived = {member["path"] for member in manifest["members"]}
        else:
            self.archive = None

    def prefetch(self, paths: Iterable[str]) -> None:
        """Extract archived members in one pass over the archive."""
        wanted = sorted({path for path in paths if path not in self.extracted
                         and not (self.run_dir / path).is_file() and path in self.archived})
        if not wanted:
            return
        target = self.scratch / "archive"
        target.mkdir(parents=True, exist_ok=True)
        listing = self.scratch / "members.txt"
        listing.write_text("".join(f"{path}\n" for path in wanted), encoding="utf-8")
        if ZSTD is None:
            raise FileNotFoundError("reading the request archive needs zstd on PATH (or ORA_ZSTD)")
        command = f"{ZSTD} -dc --long=31 {self.archive} | tar -x -C {target} -f - --files-from={listing}"
        subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True)
        for path in wanted:
            self.extracted[path] = target / path

    def path(self, relative: str) -> Path:
        direct = self.run_dir / relative
        if direct.is_file():
            return direct
        if relative not in self.extracted:
            self.prefetch([relative])
        if relative not in self.extracted:
            raise FileNotFoundError(f"{relative} is neither in the run nor in its request archive")
        return self.extracted[relative]

    def read(self, relative: str) -> bytes:
        return self.path(relative).read_bytes()

    def release(self, relative: str) -> None:
        path = self.extracted.pop(relative, None)
        if path is not None:
            path.unlink(missing_ok=True)


class Blobs:
    """Images stored once by sha256; each remembers the run-relative paths it is served at."""

    def __init__(self, out_dir: Path, files: RunFiles) -> None:
        self.dir = out_dir / "blobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = files
        self.index: dict[str, dict[str, Any]] = {}

    def _record(self, digest: str, suffix: str, media_type: str, size: int, relative: str) -> None:
        entry = self.index.setdefault(digest, {"file": f"blobs/{digest}{suffix}", "media_type": media_type,
                                               "bytes": size, "paths": []})
        if relative not in entry["paths"]:
            entry["paths"].append(relative)

    def add_ref(self, ref: dict[str, Any]) -> None:
        relative = str(ref["path"])
        digest = str(ref["sha256"])
        if digest not in self.index:
            data = self.files.read(relative)
            if sha256_hex(data) != digest or len(data) != int(ref["bytes"]):
                raise ValueError(f"artifact {relative} does not match its reference")
            suffix = Path(relative).suffix
            (self.dir / f"{digest}{suffix}").write_bytes(data)
        self._record(digest, Path(relative).suffix, str(ref.get("media_type", "")), int(ref["bytes"]), relative)

    def add_data_url(self, url: str) -> dict[str, Any] | None:
        match = DATA_URL.match(url)
        if not match:
            return None
        data = base64.b64decode(match.group(2), validate=True)
        if base64.b64encode(data).decode("ascii") != match.group(2):
            return None  # not canonical base64: keep the URL inline so its bytes stay exact
        digest = sha256_hex(data)
        suffix = ".png" if match.group(1) == "image/png" else ".img"
        relative = f"native/agent-artifacts/{digest}{suffix}"
        if digest not in self.index:
            (self.dir / f"{digest}{suffix}").write_bytes(data)
        self._record(digest, suffix, match.group(1), len(data), relative)
        return {"path": relative, "sha256": digest, "bytes": len(data), "media_type": match.group(1)}


def compact_images(message: dict[str, Any], blobs: Blobs) -> tuple[dict[str, Any], int]:
    """Images as ``ora_image_ref`` blocks backed by blobs (what ``dehydrate`` writes)."""
    content = message.get("content")
    if not isinstance(content, list):
        return message, 0
    converted = 0
    blocks = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "ora_image_ref":
            blobs.add_ref(block["ref"])
        elif (isinstance(block, dict) and block.get("type") == "image_url" and set(block) == {"type", "image_url"}
              and isinstance(block["image_url"], dict) and set(block["image_url"]) == {"url"}):
            ref = blobs.add_data_url(str(block["image_url"]["url"]))
            if ref is not None:
                block = {"type": "ora_image_ref", "ref": ref}
                converted += 1
        blocks.append(block)
    return {**message, "content": blocks}, converted


def read_session(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ora.history messages and ora.state snapshots on the session's active branch."""
    files = sorted((run_dir / "native/pi-session").glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one Pi session file, found {len(files)}")
    entries = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {entry["id"]: entry for entry in entries if "id" in entry}
    chain: list[dict[str, Any]] = []
    cursor: dict[str, Any] | None = next(entry for entry in reversed(entries) if "id" in entry)
    while cursor is not None:
        chain.append(cursor)
        cursor = by_id.get(cursor.get("parentId"))
    chain.reverse()
    history: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for entry in chain:
        if entry.get("type") != "custom":
            continue
        if entry.get("customType") == "ora.history":
            history.extend(json.loads(entry["data"]["pyjson"])["messages"])
        elif entry.get("customType") == "ora.state":
            state = json.loads(entry["data"]["pyjson"])
            if state["core"]["history_length"] != len(history):
                raise ValueError("an ora.state entry disagrees with the ora.history before it")
            states.append(state)
    return history, states


def iter_events(run_dir: Path) -> Iterator[dict[str, Any]]:
    with (run_dir / "agent-events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def response_text(raw_events: Any) -> str:
    """The assistant text of a recorded compactor response (OpenAI chunks or Anthropic events)."""
    parts: list[str] = []
    events = raw_events if isinstance(raw_events, list) else [raw_events]
    for event in events:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        if event.get("type") == "content_block_delta" and isinstance(event.get("delta"), dict) \
                and event["delta"].get("type") == "text_delta":
            parts.append(str(event["delta"].get("text", "")))
        if event.get("type") == "message" and isinstance(event.get("content"), list):  # non-streamed Messages
            parts.extend(str(block.get("text", "")) for block in event["content"]
                         if isinstance(block, dict) and block.get("type") == "text")
    return "".join(parts)


def provider_spec(plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    """``ora/agent/pi/factory.py::_provider_spec`` without credentials (they never reach the replay)."""
    policy = plan["policy"]
    affinity = f"ora:{run_id}:native" if policy.get("provider_session_affinity") == "x_session_id_run_v1" else None
    common = {
        "endpoint": policy.get("endpoint"),
        "recovery_policy": policy.get("provider_recovery", "legacy_v1"),
        "recovery_window_s": policy.get("provider_recovery_window_s"),
        "timeout_s": float(policy.get("timeout_s", 900)),
        "auth_scheme": policy.get("auth_scheme"),
        "output_token_parameter": policy.get("output_token_parameter"),
        "tool_strictness": policy.get("tool_strictness"),
        "reasoning_field": policy.get("reasoning_field"),
        "session_affinity_id": affinity,
    }
    if policy.get("provider") == "anthropic_messages":
        return {"kind": "anthropic_messages", **common, "credential_start_index": 0,
                "continuation_replay": policy.get("provider_continuation_replay", "exact_v1")}
    return {"kind": "openai_compatible", **common, "request_options": policy.get("request_options")}


def plan_mutation_tools(plan: dict[str, Any]) -> list[str]:
    """The surface's mutation tools (``PiPlanner.start``: ``surface.mutation_tools``)."""
    surface = (plan.get("policy_composition") or {}).get("tool_surface") or {}
    return list(surface.get("mutation_tools") or ["robot.submit_action"])


def context_domain_for(mutation_tools: list[str], code_mode: bool) -> dict[str, Any]:
    """The ``domain.context`` Python sends at init (``ora/agent/pi/policy_session.py``), from the tree imported."""
    import inspect

    from ora.agent.pi.domain import context_domain

    if "script_tools" not in inspect.signature(context_domain).parameters:  # trees before code mode's script tools
        return context_domain(mutation_tools=mutation_tools)
    from ora.agent.evidence.trace_privacy import SCRIPT_TOOLS

    return context_domain(mutation_tools=mutation_tools, script_tools=SCRIPT_TOOLS if code_mode else ())


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Extraction:
    def __init__(self, run_dir: Path, out_dir: Path, scratch: Path, *, requests: bool,
                 calls: int | None = None) -> None:
        self.run_dir = run_dir
        self.out_dir = out_dir
        self.files = RunFiles(run_dir, scratch)
        self.blobs = Blobs(out_dir, self.files)
        self.want_requests = requests
        self.call_limit = calls
        self.notes: list[str] = []

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        plan = json.loads((self.run_dir / "plan.json").read_text(encoding="utf-8"))
        history, states = read_session(self.run_dir)
        calls, extra = self.scan_events()
        if self.call_limit is not None:
            if self.call_limit < 1:
                raise ValueError("--calls must be positive")
            if self.call_limit < len(calls):
                self.notes.append(f"the first {self.call_limit} of {len(calls)} calls")
            calls = calls[:self.call_limit]
        self.bind_history(calls, history, states, extra)
        settings = self.settings(plan, calls, states, extra)
        replay_calls = self.call_records(calls, history, states)
        self.write_recorded(calls, states)
        replay = {
            "schema": SCHEMA,
            "source": {
                "run_id": extra["run_id"],
                "run_dir": str(self.run_dir),
                "source_commit": (plan.get("source") or {}).get("commit"),
                "source_dirty": (plan.get("source") or {}).get("dirty"),
                "policy_session_id": extra["policy_session_id"],
                "management_schema": (plan["policy"].get("context_management") or {}).get("schema"),
                "extracted_with": {"tree": str(Path(__file__).resolve().parents[4]),
                                   "commit": git_head(Path(__file__).resolve().parents[4])},
                "event_counts": extra["event_counts"],
            },
            "settings": settings,
            "blobs": self.blobs.index,
            "calls": replay_calls,
            "notes": self.notes,
            "extract_seconds": round(time.monotonic() - started, 3),
        }
        (self.out_dir / "replay.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
        return replay

    # -- Trace ---------------------------------------------------------------------------

    def scan_events(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        by_model_call: dict[str, dict[str, Any]] = {}
        observation_events: dict[int, str] = {}
        pending_compactor: list[dict[str, Any]] = []
        compactor_by_id: dict[str, dict[str, Any]] = {}
        pending_images: list[dict[str, Any]] = []
        group: dict[str, Any] | None = None
        counts: dict[str, int] = {}
        proposals: dict[str, list[str]] = {}
        response_events: dict[str, str] = {}
        extra: dict[str, Any] = {"run_id": None, "policy_session_id": None, "session_started": None}
        for event in iter_events(self.run_dir):
            kind = event["type"]
            counts[kind] = counts.get(kind, 0) + 1
            extra["run_id"] = extra["run_id"] or event.get("run_id")
            extra["policy_session_id"] = extra["policy_session_id"] or event.get("policy_session_id")
            if kind == "tool_image" and group is not None and event.get("tool_call_id") == group["tool_call_id"] \
                    and not (event.get("payload") or {}).get("origin") \
                    and (event.get("payload") or {}).get("image_index") == len(group["images"]):
                group["images"].append(event["artifact_ref"])
                continue
            if group is not None and group["images"]:
                pending_images.append(group)
            group = None
            if kind == "session_started":
                extra["session_started"] = event["payload"]
            elif kind == "observation_received":
                observation_events[int(event["observation_step"])] = event["event_id"]
            elif kind == "tool_result":
                group = {"step": event["observation_step"], "tool_call_id": event["tool_call_id"],
                         "event_id": event["event_id"], "images": []}
            elif kind in ("tool_call_proposed", "action_proposed") and event.get("model_call_id") \
                    and not (event.get("payload") or {}).get("origin"):
                # Script-made calls (payload.origin, code mode) arrive as remote events; the core
                # never adds them to the assistant's source ids.
                proposals.setdefault(event["model_call_id"], []).append(event["event_id"])
            elif kind == "context_compactor_attempt":
                payload = event["payload"]
                record = {"call_id": event["model_call_id"], "attempt_event_id": event["event_id"],
                          "canonical_request_ref": payload.get("canonical_request_ref"),
                          "wire_request_ref": event.get("artifact_ref"),
                          "source_set_sha256": payload.get("source_set_sha256"),
                          "response_event_id": None, "failed_event_id": None, "response": None, "error": None}
                compactor_by_id[event["model_call_id"]] = record
                pending_compactor.append(record)
            elif kind == "context_compactor_response":
                record = compactor_by_id[event["model_call_id"]]
                payload = event["payload"]
                record["response_event_id"] = event["event_id"]
                record["response"] = {"usage": payload.get("usage"), "finish_reason": payload.get("finish_reason"),
                                      "attempts": payload.get("attempts", 1), "latency_s": payload.get("latency_s"),
                                      "response_id": payload.get("response_id")}
                record["raw_response_ref"] = event.get("artifact_ref")
            elif kind == "context_compactor_failed":
                record = compactor_by_id[event["model_call_id"]]
                record["failed_event_id"] = event["event_id"]
                record["error"] = (event.get("payload") or {}).get("error")
            elif kind in ("context_projected", "context_exhausted"):
                step = int(event["observation_step"])
                call = {
                    "kind": "model_call" if kind == "context_projected" else "context_exhausted",
                    "model_call_id": event.get("model_call_id"),
                    "observation_step": step,
                    "observation_event_id": observation_events.get(step),
                    "event_id": event["event_id"],
                    "payload": event["payload"],
                    "compactor": pending_compactor,
                    "tool_image_registrations": pending_images,
                    "provider_request": None,
                    "provider_response": None,
                }
                pending_compactor, pending_images = [], []
                calls.append(call)
                if kind == "context_projected":
                    by_model_call[event["model_call_id"]] = call
            elif kind == "provider_request":
                by_model_call[event["model_call_id"]]["provider_request"] = {
                    "event_id": event["event_id"], "wire_ref": event.get("artifact_ref"),
                    "canonical_ref": (event.get("payload") or {}).get("canonical_request_ref")}
            elif kind == "provider_response":
                payload = event["payload"]
                response_events[event["model_call_id"]] = event["event_id"]
                by_model_call[event["model_call_id"]]["provider_response"] = {
                    "event_id": event["event_id"], "usage": payload.get("usage"),
                    "accepted": payload.get("accepted"), "finish_reason": payload.get("finish_reason")}
            elif kind == "provider_error":
                by_model_call[event["model_call_id"]]["provider_error"] = event.get("payload")
        if group is not None and group["images"]:
            pending_images.append(group)
        extra["event_counts"] = counts
        extra["trailing_compactor"] = pending_compactor
        extra["trailing_registrations"] = pending_images
        extra["proposals"] = proposals
        extra["response_events"] = response_events
        if counts.get("session_started", 0) != 1:
            self.notes.append(f"{counts.get('session_started', 0)} session_started events (agent process restarts are not replayed)")
        return calls, extra

    # -- History -------------------------------------------------------------------------

    def bind_history(self, calls: list[dict[str, Any]], history: list[dict[str, Any]],
                     states: list[dict[str, Any]], extra: dict[str, Any]) -> None:
        """Assign each call its history length and its pre-projection state."""
        real = 0
        for call in calls:
            matching = [index for index, state in enumerate(states) if state["core"]["call_sequence"] == real]
            if not matching:
                raise ValueError(f"no ora.state precedes the projection of call {real}")
            call["state_index"] = matching[-1]
            call["history_length"] = states[matching[-1]]["core"]["history_length"]
            if call["kind"] == "model_call":
                expected = f"model-call-{real:06d}"
                if call["model_call_id"] != expected:
                    raise ValueError(f"call {real} is {call['model_call_id']}, expected {expected}")
                real += 1
        # Assistant source ids gain the core's own proposals after the reply is recorded; the
        # flush before the next request holds them all. Verify against the trace.
        stale = 0
        for message in history:
            ids = message.get("_source_event_ids")
            if message.get("role") != "assistant" or not isinstance(ids, list) or not ids:
                continue
            model_call = next((mc for mc, eid in extra["response_events"].items() if eid == ids[0]), None)
            if model_call is None:
                continue
            expected = [ids[0], *extra["proposals"].get(model_call, [])]
            if ids != expected:
                stale += 1
        if stale:
            # Code mode: the script's own proposals are trace events the core never adds; a
            # restarted core (OraCore.restore) would add them. The replay keeps the flushed ids.
            self.notes.append(f"{stale} assistant messages' flushed source ids lack proposals the trace "
                              "attributes to their model call (calls the core did not record itself)")

    def call_records(self, calls: list[dict[str, Any]], history: list[dict[str, Any]],
                     states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        previous = 0
        converted = 0
        for index, call in enumerate(calls):
            length = call["history_length"]
            if length < previous:
                raise ValueError("history shrank between two calls")
            append = []
            for message in history[previous:length]:
                compacted, count = compact_images(message, self.blobs)
                converted += count
                append.append(compacted)
            previous = length
            for registration in call["tool_image_registrations"]:
                for image in registration["images"]:
                    self.blobs.add_ref(image)
            step = call["observation_step"]
            observation_index = None
            for position in range(length - 1, -1, -1):
                message = history[position]
                meta = message.get("_observation")
                ids = message.get("_source_event_ids")
                if isinstance(meta, dict) and meta.get("step") == step and isinstance(ids, list) and ids \
                        and ids[0] == call["observation_event_id"]:
                    observation_index = position
                    break
            if observation_index is None:
                self.notes.append(f"call {index}: the turn's observation message is not in its history")
            record = {
                "index": index,
                "kind": call["kind"],
                "model_call_id": call["model_call_id"],
                "observation_step": step,
                "observation_index": observation_index,
                "history_length": length,
                "history_append": append,
                "tool_image_registrations": [
                    {"step": item["step"], "tool_call_id": item["tool_call_id"], "images": item["images"]}
                    for item in call["tool_image_registrations"]],
                "compactor": [self.compactor_record(item) for item in call["compactor"]],
                "observed_usage": (call["provider_response"] or {}).get("usage") if call["kind"] == "model_call" else None,
                "response": None if call["provider_response"] is None else {
                    key: call["provider_response"][key] for key in ("accepted", "finish_reason")},
                "recorded": self.recorded(call),
            }
            if call.get("provider_error") is not None:
                record["provider_error"] = call["provider_error"]
            records.append(record)
        if converted:
            self.notes.append(f"{converted} inline data-URL images stored as blob references")
        return records

    def compactor_record(self, item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {key: item.get(key) for key in (
            "call_id", "attempt_event_id", "response_event_id", "failed_event_id", "source_set_sha256", "error")}
        request_ref = item.get("canonical_request_ref") or {}
        request = json.loads(self.files.read(str(request_ref["path"])))
        out["request_sha256"] = request_ref.get("sha256")
        out["messages_sha256"] = canonical_sha256(request["messages"][1:])
        out["system_sha256"] = canonical_sha256(request["messages"][:1])
        out["request_keys"] = sorted(request)
        if item.get("response") is not None:
            raw = json.loads(self.files.read(str(item["raw_response_ref"]["path"])))
            out["response"] = {**item["response"], "message": {"role": "assistant", "content": response_text(raw)}}
            out["raw_response_sha256"] = item["raw_response_ref"].get("sha256")
        else:
            out["response"] = None
        return out

    def recorded(self, call: dict[str, Any]) -> dict[str, Any]:
        payload = call["payload"]
        if call["kind"] != "model_call":
            return {"error": payload.get("error")}
        manifest = {key: value for key, value in payload.items()
                    if key not in ("exact_request_sha256", "wire_request_sha256")}
        request = call["provider_request"] or {}
        keys = ("canonical_request_sha256", "canonical_request_utf8_bytes", "provider_wire_request_sha256",
                "provider_wire_request_utf8_bytes", "request_sha256", "request_utf8_bytes", "history_sha256",
                "history_utf8_bytes", "projection_epoch", "cache_epoch", "request_relation",
                "pressure_estimated_input_tokens", "raw_estimated_input_tokens", "estimated_input_tokens",
                "tool_schema_sha256", "projection_transitions")
        out = {key: manifest.get(key) for key in keys}
        out.update(exact_request_sha256=payload.get("exact_request_sha256"),
                   wire_request_sha256=payload.get("wire_request_sha256"),
                   manifest_sha256=canonical_sha256(manifest),
                   canonical_ref_sha256=(request.get("canonical_ref") or {}).get("sha256"),
                   wire_ref_sha256=(request.get("wire_ref") or {}).get("sha256"))
        if out["canonical_ref_sha256"] != out["exact_request_sha256"]:
            self.notes.append(f"{call['model_call_id']}: provider_request canonical ref differs from context_projected")
        return out

    # -- Settings ------------------------------------------------------------------------

    def settings(self, plan: dict[str, Any], calls: list[dict[str, Any]], states: list[dict[str, Any]],
                 extra: dict[str, Any]) -> dict[str, Any]:
        policy = plan["policy"]
        started = extra["session_started"] or {}
        for key in ("model", "context_management", "context_window_tokens", "max_output_tokens", "context_profile",
                    "context_retention", "summary_policy", "context_composition", "reasoning_effort", "stream"):
            if key in started and canonical_bytes(started[key]) != canonical_bytes(policy.get(key)):
                self.notes.append(f"plan policy.{key} differs from the session_started payload")
        first = next(call for call in calls if call["kind"] == "model_call")
        ref = first["provider_request"]["canonical_ref"]
        request = json.loads(self.files.read(str(ref["path"])))
        tools = request.get("tools") or []
        if canonical_sha256(tools) != first["payload"].get("tool_schema_sha256"):
            raise ValueError("the first canonical request's tools differ from the manifest's tool_schema_sha256")
        mutation_tools = plan_mutation_tools(plan)
        domain = context_domain_for(mutation_tools, bool(policy.get("code_mode")))
        sources = states[first["state_index"]]["core"]["context_sources"]
        for call in calls:
            if canonical_bytes(states[call["state_index"]]["core"]["context_sources"]) != canonical_bytes(sources):
                self.notes.append("core.context_sources changed during the run (the replay uses the first)")
                break
        return {
            "run_id": extra["run_id"],
            "policy_session_id": extra["policy_session_id"],
            "policy": policy,
            "provider": provider_spec(plan, extra["run_id"]),
            "tools": {"definitions": tools, "mutation_tools": mutation_tools},
            "domain": {"context": domain, "artifact_reader_name": "artifact_read"},
            "context_sources": sources,
        }

    # -- Recorded evidence for diffs -------------------------------------------------------

    def write_recorded(self, calls: list[dict[str, Any]], states: list[dict[str, Any]]) -> None:
        with gzip.open(self.out_dir / "recorded-manifests.jsonl.gz", "wt", encoding="utf-8") as manifests, \
                gzip.open(self.out_dir / "recorded-states.jsonl.gz", "wt", encoding="utf-8") as snapshots:
            for call in calls:
                manifests.write(json.dumps({"model_call_id": call["model_call_id"], "kind": call["kind"],
                                            "manifest": call["payload"]}, ensure_ascii=False) + "\n")
                snapshots.write(json.dumps({"model_call_id": call["model_call_id"],
                                            "context": states[call["state_index"]].get("context")},
                                           ensure_ascii=False) + "\n")
        if not self.want_requests:
            return
        refs = [call["provider_request"]["canonical_ref"] for call in calls
                if call["kind"] == "model_call" and call["provider_request"]]
        self.files.prefetch(str(ref["path"]) for ref in refs)
        with gzip.open(self.out_dir / "recorded-requests.jsonl.gz", "wt", encoding="utf-8") as out:
            for call, ref in zip([c for c in calls if c["kind"] == "model_call" and c["provider_request"]], refs):
                data = self.files.read(str(ref["path"]))
                if sha256_hex(data) != ref["sha256"]:
                    raise ValueError(f"recorded request {ref['path']} does not match its sha256")
                out.write(json.dumps({"model_call_id": call["model_call_id"], "sha256": ref["sha256"],
                                      "skeleton": skeleton(json.loads(data))}, ensure_ascii=False) + "\n")
                self.files.release(str(ref["path"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="replay directory to create")
    parser.add_argument("--no-requests", action="store_true", help="skip recorded request skeletons")
    parser.add_argument("--scratch", type=Path, default=None, help="temporary directory (default: TMPDIR)")
    parser.add_argument("--calls", type=int, default=None, help="extract only the first N calls (a prefix)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not (run_dir / "seal.json").is_file():
        print(f"warning: {run_dir} has no seal.json", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="context-replay-", dir=args.scratch))
    try:
        replay = Extraction(run_dir, args.out, scratch, requests=not args.no_requests, calls=args.calls).run()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    calls = replay["calls"]
    print(json.dumps({
        "out": str(args.out), "calls": len(calls),
        "model_calls": sum(1 for call in calls if call["kind"] == "model_call"),
        "compactor_calls": sum(len(call["compactor"]) for call in calls),
        "blobs": len(replay["blobs"]), "notes": replay["notes"], "seconds": replay["extract_seconds"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
