"""Replay one sealed episode on the Pi agent core and check it.

The driver acts as the Supervisor (``EpisodeLoop``) for the fixture's script:
it publishes each step's disk state, delivers ``fresh_observation``,
``environment_receipt``, ``official_verdict`` and ``run_termination`` through
``PolicySession.accept_runtime_event``, runs each policy turn through
``PolicySession.act(PolicyTurnInput(...))`` and closes the session. It then
checks

* every wire request the replay produced against the archived one, byte for
  byte (``native_provider_wire_request`` sha256 and bytes), and the canonical
  requests the same way;
* the replayed ``agent-events.jsonl`` against the original, event by event, on
  ``type``, ``payload`` and ``artifact_ref`` (sha256, path, kind, bytes, media
  type) plus the envelope ids; ``unix_s``, ``monotonic_ns`` and the hash chain
  that covers them are excluded;
* the agent-artifact inventory and the Context-Artifact index.

``--inflate-prompt-tokens F`` scales every recorded ``prompt_tokens`` by F
(from ``--inflate-from-call`` on) and answers L6 compactor calls with a scripted
valid checkpoint; requests then diverge from the archive by design.

Run ``python -m tests.support.agent_core.replay.run_native --help`` for options.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from .extract import DEFAULT_SCRATCH
from .runtime import (
    Fixture, ReplayMismatch, ReplaySession, build_replay_session, canonical_bytes,
    compactor_responses, read_json, sha256_bytes, write_json,
)
#: Envelope fields that carry wall/monotonic time or the hash chain over it.
TIMING_FIELDS = frozenset({"unix_s", "monotonic_ns", "event_sha256", "previous_event_sha256"})
ENVELOPE_FIELDS = (
    "schema", "seq", "event_id", "run_id", "policy_session_id", "observation_step",
    "model_call_id", "tool_call_id", "causal_parent_id", "visibility",
)
REF_FIELDS = ("sha256", "path", "kind", "bytes", "media_type")
#: The marker that lets the driver replace a replay output directory it wrote.
OUTPUT_MARKER = ".ora-pi-spike-replay-output"


# -- disk reads -------------------------------------------------------------


class ReadAudit:
    """Record every file the process opens under one root (``sys.addaudithook``)."""

    _installed = False
    _active: "ReadAudit | None" = None

    def __init__(self, root: Path) -> None:
        self.root = str(Path(root).resolve()) + os.sep
        self.opens: dict[str, dict[str, int]] = {}

    def __enter__(self) -> "ReadAudit":
        if not ReadAudit._installed:
            sys.addaudithook(ReadAudit._hook)
            ReadAudit._installed = True
        ReadAudit._active = self
        return self

    def __exit__(self, *_exc: Any) -> None:
        ReadAudit._active = None

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Leave the Supervisor's own publishing out of the session's record."""

        ReadAudit._active = None
        try:
            yield
        finally:
            ReadAudit._active = self

    @staticmethod
    def _hook(event: str, args: tuple[Any, ...]) -> None:
        audit = ReadAudit._active
        if audit is None or event != "open" or not args:
            return
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(path)
        if not text.startswith(audit.root):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str):
            writing = any(char in mode for char in "wax+")
        else:
            writing = bool(int(flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        entry = audit.opens.setdefault(text[len(audit.root):], {"read": 0, "write": 0})
        entry["write" if writing else "read"] += 1

    def summary(self) -> dict[str, Any]:
        groups = (
            ("agent-events.jsonl", "agent-events.jsonl"),
            ("control/current-observation.json", "control/current-observation.json"),
            ("control/submitted-action.json", "control/submitted-action.json"),
            ("control/", "control/* (other)"),
            ("policy-state/steps/", "policy-state/steps/*"),
            ("policy-observations/", "policy-observations/*"),
            ("sandbox/workspace/inputs/current-visual/", "sandbox/workspace/inputs/current-visual/*"),
            ("sandbox/workspace/inputs/action-observations/",
             "sandbox/workspace/inputs/action-observations/*"),
            ("sandbox/", "sandbox/* (other)"),
            ("native/inputs/current-visual/", "native/inputs/current-visual/*"),
            ("native/agent-artifacts/", "native/agent-artifacts/*"),
            ("native/context-artifacts/", "native/context-artifacts/*"),
            ("native/", "native/* (other)"),
            ("compiled-policy/", "compiled-policy/*"),
            ("", "other"),
        )
        summary: dict[str, dict[str, int]] = {}
        for relative, counts in sorted(self.opens.items()):
            label = next(label for prefix, label in groups if relative.startswith(prefix))
            bucket = summary.setdefault(label, {"files_read": 0, "reads": 0,
                                                "files_written": 0, "writes": 0})
            if counts["read"]:
                bucket["files_read"] += 1
                bucket["reads"] += counts["read"]
            if counts["write"]:
                bucket["files_written"] += 1
                bucket["writes"] += counts["write"]
        policy_reads = sorted(
            relative for relative, counts in self.opens.items()
            if counts["read"] and relative.startswith((
                "control/", "policy-state/", "policy-observations/", "sandbox/",
            ))
        )
        return {"by_area": summary, "supervisor_state_files_read": policy_reads}


# -- the Supervisor script -----------------------------------------------------


def _publish_step(fixture: Fixture, run_root: Path, op: Mapping[str, Any]) -> None:
    for relative in op["files"]:
        target = run_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture.path(f"run-files/{relative}"), target)
    control = copy.deepcopy(dict(op["control_current_observation"]))
    for key in op.get("control_run_relative_keys", ()):
        control[key] = str(run_root / control[key])
    path = run_root / "control/current-observation.json"
    path.write_text(json.dumps(control, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    for name in op.get("remove_control_files", ()):
        (run_root / "control" / name).unlink(missing_ok=True)


def _with_run_root(value: Any, run_root: Path) -> Any:
    """Substitute this replay's run root for the fixture's ``${RUN_ROOT}`` placeholder."""

    if isinstance(value, Mapping):
        return {key: _with_run_root(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_run_root(item, run_root) for item in value]
    if isinstance(value, str) and value.startswith("${RUN_ROOT}/"):
        return str(run_root) + value[len("${RUN_ROOT}"):]
    return value


def _turn_input(fixture: Fixture, run_root: Path, op: Mapping[str, Any]) -> Any:
    from ora.agent import PolicyTurnInput

    context = read_json(fixture.path(op["context"]))
    return PolicyTurnInput(
        task_id=op["task_id"],
        task_instruction=op["task_instruction"],
        context=context,
        image_path=run_root / op["image_path"],
        observation_id=op["observation_id"],
        observation_step=op["step"],
        budget=dict(context.get("remaining_budget") or {}),
        interaction_phase=op.get("interaction_phase", "observation_ready"),
        fresh_observation_required=False,
        allow_reminder=bool(op["allow_reminder"]),
        terminal_lifecycle=bool(op["terminal_lifecycle"]),
        image_paths=tuple(run_root / path for path in op.get("image_paths", ())),
    )


def _event(kind: str, payload: Any, *, observation_id: Any = None, step: Any = None) -> Any:
    from ora.agent import PolicyRuntimeEvent

    return PolicyRuntimeEvent(
        kind=kind, payload=payload, observation_id=observation_id, observation_step=step,
    )


def run_script(
    fixture: Fixture, session: ReplaySession, audit: ReadAudit | None = None,
) -> list[dict[str, Any]]:
    """Execute every Supervisor operation; return one record per policy turn."""

    from ora.agent import ActionProposal, ExecutedActions

    run_root = session.run_root
    planner = session.planner
    turns: list[dict[str, Any]] = []
    observation_id: str | None = None
    step: int | None = None
    script = iter(fixture.script)
    upcoming = next(script, None)

    def advance() -> dict[str, Any]:
        nonlocal upcoming
        op, upcoming = upcoming, next(script, None)
        return op

    def run_tool_action(action: Mapping[str, Any], context: Mapping[str, Any], name: str) -> None:
        """One action a synchronous tool ran, in the order of ``ToolActionExecutor.execute_action``."""

        session.native.begin_tool_action(action["action_id"])
        session.native.record_tool_action(
            dict(action), dict(context), {"observation_id": observation_id, "step": step}, name,
        )
        perform(advance())  # its receipt, then the observation it published (none for a refused action)
        if upcoming is not None and upcoming["op"] == "publish_step":
            perform(advance())
            perform(advance())

    session.tools.run_tool_action = run_tool_action

    def perform(op: dict[str, Any]) -> None:
        nonlocal observation_id, step
        kind = op["op"]
        if kind == "publish_step":
            with audit.paused() if audit is not None else contextlib.nullcontext():
                _publish_step(fixture, run_root, op)
        elif kind == "fresh_observation":
            observation_id, step = op["observation_id"], op["step"]
            planner.accept_runtime_event(_event(
                "fresh_observation", read_json(fixture.path(op["context"])),
                observation_id=observation_id, step=step,
            ))
        elif kind == "run_observation":
            started = time.monotonic()
            outcome = planner.act(_turn_input(fixture, run_root, op))
            expected = op["expected"]
            action_id = (
                outcome.action.get("action_id")
                if isinstance(outcome, ActionProposal) and isinstance(outcome.action, Mapping)
                else outcome.execution["actions"][0]["action_id"] if isinstance(outcome, ExecutedActions)
                else None
            )
            turns.append({
                "step": op["step"],
                "terminal_lifecycle": op["terminal_lifecycle"],
                "outcome_kind": outcome.kind,
                "model_calls": outcome.model_calls,
                "expected_model_calls": len(expected["model_call_ids"]),
                "action_id": action_id,
                "expected_action_id": expected["action_id"],
                "matches": (
                    outcome.model_calls == len(expected["model_call_ids"])
                    and action_id == expected["action_id"]
                ),
                "seconds": round(time.monotonic() - started, 3),
            })
        elif kind == "preserve_observation":
            turn = _turn_input(fixture, run_root, op)
            planner.preserve_observation_without_policy_turn(**turn.runtime_kwargs())
        elif kind == "environment_receipt":
            with audit.paused() if audit is not None else contextlib.nullcontext():
                for relative in op.get("files", ()):
                    target = run_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fixture.path(f"run-files/{relative}"), target)
            planner.accept_runtime_event(_event(
                "environment_receipt", _with_run_root(op["receipt"], run_root),
                observation_id=observation_id, step=step,
            ))
        elif kind == "official_verdict":
            planner.accept_runtime_event(_event(
                "official_verdict", {"value": op["verdict"]},
                observation_id=observation_id, step=step,
            ))
        elif kind == "run_termination":
            planner.accept_runtime_event(_event(
                "run_termination", read_json(fixture.path(op["result"])),
                observation_id=observation_id, step=step,
            ))
        elif kind == "close":
            planner.close()
        else:
            raise ReplayMismatch(f"unknown script operation: {kind}")

    while upcoming is not None:
        perform(advance())
    return turns


# -- comparison ----------------------------------------------------------------


def _json_diff(expected: Any, actual: Any, path: str = "$", limit: int = 8) -> list[str]:
    """Up to ``limit`` JSON paths where two values differ (int and float distinct)."""

    out: list[str] = []

    def visit(left: Any, right: Any, where: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left:
                    out.append(f"{where}.{key}: only in replay")
                elif key not in right:
                    out.append(f"{where}.{key}: only in original")
                else:
                    visit(left[key], right[key], f"{where}.{key}")
                if len(out) >= limit:
                    return
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                out.append(f"{where}: length {len(left)} != {len(right)}")
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{where}[{index}]")
                if len(out) >= limit:
                    return
        elif canonical_bytes(left) != canonical_bytes(right):
            text_l, text_r = canonical_bytes(left)[:120], canonical_bytes(right)[:120]
            out.append(f"{where}: {text_l.decode('utf-8', 'replace')} != {text_r.decode('utf-8', 'replace')}")

    visit(expected, actual, path)
    return out


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _without_keys(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {key: _without_keys(item, keys) for key, item in value.items() if key not in keys}
    if isinstance(value, list):
        return [_without_keys(item, keys) for item in value]
    return value


def compare_events(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]], *,
    ignore_payload_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Event by event; ``ignore_payload_keys`` are dropped at every payload depth."""

    mismatches: list[dict[str, Any]] = []
    identical = 0
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected) or index >= len(actual):
            source = actual if index >= len(expected) else expected
            mismatches.append({
                "index": index, "type": source[index]["type"],
                "problem": "extra in replay" if index >= len(expected) else "missing from replay",
            })
            continue
        left, right = expected[index], actual[index]
        if ignore_payload_keys:
            left = {**left, "payload": _without_keys(left["payload"], ignore_payload_keys)}
            right = {**right, "payload": _without_keys(right["payload"], ignore_payload_keys)}
        problems: list[str] = []
        if left["type"] != right["type"]:
            problems.append(f"type {left['type']} != {right['type']}")
        if canonical_bytes(left["payload"]) != canonical_bytes(right["payload"]):
            problems.extend("payload " + item for item in _json_diff(left["payload"], right["payload"]))
        left_ref, right_ref = left.get("artifact_ref"), right.get("artifact_ref")
        if (left_ref is None) != (right_ref is None) or (
            left_ref is not None and any(left_ref.get(key) != right_ref.get(key) for key in REF_FIELDS)
        ):
            problems.append(
                f"artifact_ref {None if left_ref is None else left_ref.get('sha256')} "
                f"({None if left_ref is None else left_ref.get('kind')}) != "
                f"{None if right_ref is None else right_ref.get('sha256')}"
            )
        for key in ENVELOPE_FIELDS:
            if left.get(key) != right.get(key):
                problems.append(f"{key} {left.get(key)!r} != {right.get(key)!r}")
        unexpected = (set(left) ^ set(right)) - TIMING_FIELDS
        if unexpected:
            problems.append(f"envelope keys differ: {sorted(unexpected)}")
        if problems:
            mismatches.append({
                "index": index, "type": left["type"],
                "model_call_id": left.get("model_call_id"), "problems": problems,
            })
        else:
            identical += 1
    return {
        "original_events": len(expected), "replayed_events": len(actual),
        "identical": identical, "excluded_fields": sorted(TIMING_FIELDS),
        "ignored_payload_keys": sorted(ignore_payload_keys),
        "mismatches": mismatches,
    }


def compare_requests(fixture: Fixture, run_root: Path, actual_events: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [event for event in actual_events if event["type"] == "provider_request"]
    recorded = {call["model_call_id"]: call for call in fixture.provider_calls}
    wire_identical = canonical_identical = 0
    mismatches: list[dict[str, Any]] = []
    for event in requests:
        model_call_id = event["model_call_id"]
        call = recorded.get(model_call_id)
        replay_wire = (run_root / event["artifact_ref"]["path"]).read_bytes()
        canonical_ref = event["payload"]["canonical_request_ref"]
        replay_canonical = (run_root / canonical_ref["path"]).read_bytes()
        if call is None:
            mismatches.append({"model_call_id": model_call_id, "problem": "no recorded call"})
            continue
        archived_wire = fixture.path(call["wire_request"]["path"]).read_bytes()
        archived_canonical = fixture.path(call["canonical_request"]["path"]).read_bytes()
        wire_same = replay_wire == archived_wire
        canonical_same = replay_canonical == archived_canonical
        wire_identical += wire_same
        canonical_identical += canonical_same
        if not (wire_same and canonical_same):
            mismatch: dict[str, Any] = {
                "model_call_id": model_call_id,
                "wire_identical": wire_same,
                "canonical_identical": canonical_same,
                "replay_wire_sha256": sha256_bytes(replay_wire),
                "archived_wire_sha256": call["wire_request"]["sha256"],
            }
            if not wire_same:
                mismatch["wire_diff"] = _json_diff(json.loads(archived_wire), json.loads(replay_wire))
            mismatches.append(mismatch)
    return {
        "provider_calls_recorded": len(fixture.provider_calls),
        "provider_requests_replayed": len(requests),
        "wire_requests_byte_identical": wire_identical,
        "canonical_requests_byte_identical": canonical_identical,
        "mismatches": mismatches,
    }


def compare_artifacts(fixture: Fixture, run_root: Path) -> dict[str, Any]:
    expected = {item["name"]: item for item in read_json(fixture.path("expected/agent-artifacts.json"))["artifacts"]}
    produced = {path.name for path in (run_root / "native/agent-artifacts").iterdir() if not path.name.startswith(".")}
    missing = sorted(set(expected) - produced)
    extra = sorted(produced - set(expected))
    index_expected = fixture.path("expected/context-artifacts-index.json")
    index_actual = run_root / "native/context-artifacts/index.json"
    index_identical = (
        index_expected.is_file() and index_actual.is_file()
        and index_expected.read_bytes() == index_actual.read_bytes()
    )
    return {
        "expected": len(expected), "produced": len(produced),
        "missing": [{"name": name, "kind": expected[name]["kind_from_trace"]} for name in missing],
        "extra": extra,
        "context_artifact_index_identical": index_identical,
    }


def transitions(actual_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model call: projection epoch, layer transitions, L5/L6 admissions."""

    rows = []
    for event in actual_events:
        if event["type"] != "context_projected":
            continue
        manifest = event["payload"]
        emergency = manifest.get("emergency_compaction_admission") or {}
        rows.append({
            "model_call_id": event["model_call_id"],
            "observation_step": event["observation_step"],
            "projection_epoch": manifest.get("projection_epoch"),
            "transitions": [
                {"kind": item.get("kind"), "reasons": item.get("reasons")}
                for item in manifest.get("projection_transitions") or []
            ],
            "provider_usage_anchor_tokens": manifest.get("provider_usage_anchor_tokens"),
            "pressure_estimated_input_tokens": manifest.get("pressure_estimated_input_tokens"),
            "active_inline_images": manifest.get("active_inline_images"),
            "l6": {key: emergency.get(key) for key in ("status", "reason", "provider_input_tokens")},
            "l5": {key: (manifest.get("episode_checkpoint_admission") or {}).get(key)
                   for key in ("status", "reason")},
            "emergency_compaction_count": manifest.get("emergency_compaction_count"),
            "checkpoint_slot_count": manifest.get("checkpoint_slot_count"),
        })
    return rows


def compactor_calls(actual_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each L5/L6 compactor call with its run-relative request/response artifacts."""

    calls: dict[str, dict[str, Any]] = {}
    for event in actual_events:
        kind = event["type"]
        if kind == "context_compactor_attempt":
            calls[event["model_call_id"]] = {
                "compactor_call_id": event["model_call_id"],
                "attempt_event_id": event["event_id"],
                "wire_request": f"run/{event['artifact_ref']['path']}",
                "canonical_request": f"run/{event['payload']['canonical_request_ref']['path']}",
                "source_set_sha256": event["payload"]["source_set_sha256"],
            }
        elif kind in {"context_compactor_response", "context_compactor_failed"}:
            record = calls[event["model_call_id"]]
            record["outcome"] = kind
            if event.get("artifact_ref"):
                record["raw_events"] = f"run/{event['artifact_ref']['path']}"
            record["usage"] = (event.get("payload") or {}).get("usage")
        elif kind == "context_condensation":
            pending = [record for record in calls.values() if "next_main_model_call_id" not in record]
            for record in pending:
                record["next_main_model_call_id"] = event["model_call_id"]
                record["condensation_event_id"] = event["event_id"]
    return list(calls.values())


# -- the driver -------------------------------------------------------------


def _fresh_output(out_dir: Path) -> Path:
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        if not (out_dir / OUTPUT_MARKER).is_file():
            raise ReplayMismatch(f"refusing to replace a directory this driver did not write: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / OUTPUT_MARKER).write_text("written by tests.support.agent_core.replay.run_native\n")
    return out_dir


def trace_checks(run_root: Path) -> dict[str, Any]:
    """``verify_agent_trace`` and the lossless-context evidence check on one run."""

    from ora.agent.evidence.trace_verify import verify_agent_trace
    from ora.orchestrator.evaluation.lossless import _lossless_context_evidence

    trace = verify_agent_trace(run_root)
    lossless = _lossless_context_evidence(run_root)
    provenance = lossless.artifact_provenance
    return {
        "agent_trace": {
            "status": trace["status"], "errors": trace["errors"][:20],
            "error_count": len(trace["errors"]), "warnings": trace.get("warnings", [])[:20],
            "event_count": trace.get("event_count"),
        },
        "lossless_context": {
            "reasons": lossless.reasons[:20], "reason_count": len(lossless.reasons),
            "artifact_provenance_errors": provenance.get("errors", [])[:20],
            "artifact_provenance_gaps": provenance.get("evidence_gaps", [])[:20],
        },
    }


def call_timing(actual_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per model call, from the trace's monotonic clock (milliseconds).

    ``before_projection_ms``: from the previous event to ``context_projected``
    (tool replay, Context projection, request assembly and persistence);
    ``provider_ms``: ``provider_request`` to ``provider_response`` (recorded
    response accumulation); ``interval_ms``: between consecutive
    ``provider_request`` events.
    """

    rows: list[dict[str, Any]] = []
    previous_request: int | None = None
    for index, event in enumerate(actual_events):
        if event["type"] != "context_projected":
            continue
        model_call_id = event["model_call_id"]
        stamp = int(event["monotonic_ns"])
        before = int(actual_events[index - 1]["monotonic_ns"]) if index else stamp
        request = next(e for e in actual_events[index:] if e["type"] == "provider_request"
                       and e["model_call_id"] == model_call_id)
        response = next((e for e in actual_events[index:] if e["type"] == "provider_response"
                         and e["model_call_id"] == model_call_id), None)
        request_stamp = int(request["monotonic_ns"])
        rows.append({
            "model_call_id": model_call_id,
            "before_projection_ms": round((stamp - before) / 1e6, 3),
            "provider_ms": None if response is None else round(
                (int(response["monotonic_ns"]) - request_stamp) / 1e6, 3),
            "interval_ms": None if previous_request is None else round(
                (request_stamp - previous_request) / 1e6, 3),
        })
        previous_request = request_stamp
    return rows


def replay(
    fixture_dir: Path, out_dir: Path, *, core: str = "pi",
    inflate_factor: float | None = None, inflate_from_call: int = 0,
) -> dict[str, Any]:
    """Replay the fixture on ``core`` (``pi``) into ``out_dir/run``; return (and write) the report."""

    started = time.monotonic()
    fixture = Fixture.load(fixture_dir)
    out_dir = _fresh_output(out_dir)
    run_root = out_dir / "run"
    variant = "faithful" if inflate_factor is None else "usage_inflated"
    session = build_replay_session(
        fixture, run_root, core=core, inflate_factor=inflate_factor,
        inflate_from_call=inflate_from_call,
        compactor=None if inflate_factor is None else compactor_responses(),
    )
    built = time.monotonic()
    try:
        with ReadAudit(run_root) as audit:
            turns = run_script(fixture, session, audit)
    except Exception as exc:
        process = getattr(session.native, "process", None)
        if process is not None:
            try:
                process.wait(timeout=5)
            except Exception:
                pass
            # A negative code is the signal that ended the TypeScript core.
            exc.core_process_returncode = process.poll()  # type: ignore[attr-defined]
        raise
    finally:
        _release(session)
    finished = time.monotonic()
    actual = _load_events(run_root / "agent-events.jsonl")
    expected = _load_events(fixture.path("expected/agent-events.jsonl"))
    requests = compare_requests(fixture, run_root, actual)
    events = compare_events(expected, actual)
    policy = fixture.plan["policy"]
    hard_input_tokens = (
        int(policy["context_window_tokens"]) - int(policy["max_output_tokens"]) - 2048
    )
    report: dict[str, Any] = {
        "schema": "ora.pi-spike.native-replay-report.v1",
        "core": core,
        "variant": variant,
        "fixture": str(fixture.root),
        "run_root": str(run_root),
        "inflation": None if inflate_factor is None else {
            "prompt_tokens_factor": inflate_factor, "from_call_index": inflate_from_call,
            "emergency_high_watermark_tokens": int(
                hard_input_tokens * float(policy["context_management"]["emergency_fallback_ratio"])
            ),
            "hard_input_tokens": hard_input_tokens,
        },
        "compiled_policy_source": session.compiled_policy_source,
        "requests": requests,
        "events": events,
        "artifacts": compare_artifacts(fixture, run_root),
        "turns": {
            "total": len(turns), "matching": sum(turn["matches"] for turn in turns),
            "records": turns,
        },
        "tool_runtime": {
            "dispatched": len(session.tools.invocations),
            "unconsumed_recorded_calls": [call["tool_call_id"] for call in session.tools.remaining()],
        },
        "provider_transport": {
            "responses_served": sum(1 for event in actual if event["type"] == "provider_request"),
            "responses_recorded": len(fixture.provider_calls),
        },
        "compactor_calls": compactor_calls(actual),
        "context_transitions": transitions(actual),
        "trace_checks": trace_checks(run_root),
        "disk_reads": audit.summary(),
        "timing": {
            "build_s": round(built - started, 3),
            "episode_s": round(finished - built, 3),
            "per_call": call_timing(actual),
        },
        "seconds": None,
    }
    report["seconds"] = round(time.monotonic() - started, 3)
    write_json(out_dir / "report.json", report, indent=1)
    return report


def _release(session: ReplaySession) -> None:
    """Close a session the script left open (a failed run); stop its core process."""

    planner = session.native
    if getattr(planner, "closed", False):
        return
    try:
        session.planner.close()
    except Exception:
        pass
    process = getattr(planner, "process", None)
    if process is not None and process.poll() is None:
        process.kill()
        process.wait(timeout=10)


def headline(report: Mapping[str, Any]) -> dict[str, Any]:
    requests = report["requests"]
    events = report["events"]
    return {
        "core": report.get("core", "pi"),
        "variant": report["variant"],
        "wire_requests_byte_identical": f"{requests['wire_requests_byte_identical']}/{requests['provider_calls_recorded']}",
        "canonical_requests_byte_identical": f"{requests['canonical_requests_byte_identical']}/{requests['provider_calls_recorded']}",
        "events_identical": f"{events['identical']}/{events['original_events']} (replayed {events['replayed_events']})",
        "turns_matching": f"{report['turns']['matching']}/{report['turns']['total']}",
        "artifacts_missing": len(report["artifacts"]["missing"]),
        "artifacts_extra": len(report["artifacts"]["extra"]),
        "context_artifact_index_identical": report["artifacts"]["context_artifact_index_identical"],
        "compactor_calls": len(report["compactor_calls"]),
        "l6_admitted_at": [
            row["model_call_id"] for row in report["context_transitions"]
            if row["l6"]["status"] == "admitted"
        ],
        "agent_trace": report["trace_checks"]["agent_trace"]["status"],
        "lossless_reasons": report["trace_checks"]["lossless_context"]["reason_count"],
        "seconds": report["seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_SCRATCH / "fixture")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <scratch>/pi or <scratch>/inflated)")
    parser.add_argument("--inflate-prompt-tokens", type=float, default=None, metavar="FACTOR")
    parser.add_argument("--inflate-from-call", type=int, default=0, metavar="INDEX")
    arguments = parser.parse_args(argv)
    out = arguments.out or DEFAULT_SCRATCH / ("pi" if arguments.inflate_prompt_tokens is None else "inflated")
    report = replay(
        arguments.fixture, out, inflate_factor=arguments.inflate_prompt_tokens,
        inflate_from_call=arguments.inflate_from_call,
    )
    print(json.dumps(headline(report), indent=1))
    faithful_ok = (
        report["variant"] != "faithful"
        or (
            report["requests"]["wire_requests_byte_identical"]
            == report["requests"]["provider_calls_recorded"]
            == report["requests"]["provider_requests_replayed"]
            and not report["events"]["mismatches"]
        )
    )
    return 0 if faithful_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
