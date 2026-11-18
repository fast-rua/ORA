"""Freeze a completed-native-tick prefix and its pre-execution tool state.

No simulator import or future observation/configuration fallback. N counts
completed native ticks, not parent tools, receipts or zero-based samples.
"""
from copy import deepcopy
from pathlib import Path

from ora.contracts import validate_runtime_action
from ora.agent.toolkit.interface.gaps import require_no_gaps
from ora.agent.toolkit.interface.journal import read_events
from ora.agent.toolkit.interface.types import CorruptRecord, NotRecorded, canonical, decode, digest
from ora.agent.toolkit.state.deltas import before_interface
from ora.agent.toolkit.state.keys import StateError
from ora.orchestrator.episode.tool_execution_log import ToolExecutionLog
from .live_boundary import verified
from .selection import read_local, rows
from .snapshot_boundary import select_snapshot

SCHEMA = "ora.toolkit.native-boundary.v1"


def _traces(source):
    paths = sorted((source / "native/controller").glob("action-*.json"))
    result = []
    for index, path in enumerate(paths):
        if path.name != f"action-{index:04d}.json":
            raise NotRecorded("未记录: native controller sequence is not contiguous")
        receipt = decode(read_local(source, f"native/receipts/action-{index:04d}.json"))
        ref = receipt["control_trace"]
        if ref["path"] != f"controller/{path.name}":
            raise CorruptRecord("native trace path differs from receipt sequence")
        trace = decode(verified(source / "native", ref))
        if trace.get("schema") not in {"ora.robodojo-controller-trace.v1", "ora.robodojo-controller-trace.v2"}:
            raise NotRecorded("未记录: unsupported native controller trace")
        if trace["action_id"] != receipt["action_id"]:
            raise CorruptRecord("native trace/receipt action identity differs")
        result.append((receipt, trace))
    return result


def _action_boundary(source, native_receipt):
    action_id = native_receipt["action_id"]
    matches = []
    for path in sorted((source / "native/tool-executions").glob("*.json")):
        raw = decode(read_local(source, str(path.relative_to(source))))
        for action in raw.get("actions", []):
            if action.get("action_id") == action_id:
                matches.append((path, raw, action))
    if len(matches) != 1:
        raise NotRecorded("未记录: native action has no unique parent execution ledger")
    path, raw, action = matches[0]
    log = ToolExecutionLog(source, raw["tool_call_id"], raw["model_call_id"])
    if log.path != path:
        raise CorruptRecord("execution ledger filename differs from parent identity")
    # Validate the durable chain; an unknown physical tail is never guessed.
    log.recover()
    receipt = decode(verified(source, action["receipt_ref"]))
    nested = receipt.get("receipt") or {}
    nested_action = nested.get("action") if isinstance(nested, dict) else None
    ids = {value for value in (receipt.get("action_id"),
                              nested.get("action_id") if isinstance(nested, dict) else None,
                              nested_action.get("action_id") if isinstance(nested_action, dict) else None)
           if isinstance(value, str) and value}
    if not isinstance(nested, dict) or "control_trace" not in nested:
        raise NotRecorded("未记录: ledger receipt lacks its hash-bound native receipt")
    if nested != native_receipt:
        raise CorruptRecord("native receipt differs from trusted execution receipt artifact")
    if ids != {action_id}:
        raise CorruptRecord("execution receipt artifact belongs to another action")
    receipt_events = [row for row in rows(source, "agent-events.jsonl")
                      if row.get("event_id") == action["receipt_event_id"]]
    if (len(receipt_events) != 1 or receipt_events[0].get("type") != "environment_receipt"
            or receipt_events[0].get("payload", {}).get("action_id") != action_id
            or receipt_events[0].get("artifact_ref") != action["receipt_ref"]):
        raise NotRecorded("未记录: execution receipt has no unique trusted event binding")
    require_no_gaps(source, raw["tool_call_id"])
    events = read_events(source / "native/toolkit-io/calls.jsonl")
    beginnings = [row for row in events if row["event"] == "begin"
                  and row["request"]["identity"]["method"] == "execute_action"
                  and row["request"]["identity"]["parameters"].get("action", {}).get("action_id") == action_id
                  and row["identity"]["tool_call_id"] == raw["tool_call_id"]
                  and row["identity"]["boundary"].get("model_call_id") == raw["model_call_id"]]
    if len(beginnings) != 1:
        raise NotRecorded("未记录: native action has no unique execute_action begin")
    begin = beginnings[0]
    # The interface records the submitted action; the execution owner validates
    # it (including quaternion normalization) once before writing log.begin.
    # Compare that exact owner form, not approximate poses or a re-normalized ledger.
    try:
        submitted = validate_runtime_action(begin["request"]["identity"]["parameters"]["action"])
    except ValueError as error:
        raise CorruptRecord("invalid execution interface action: " + str(error)) from error
    if submitted != action["action"]:
        raise CorruptRecord("execution ledger action differs from interface request")
    boundary = select_snapshot(source, step=action["before_observation"]["step"], raw_tool_state=True)
    context = boundary.context.json()
    if any(begin["identity"].get(key) != context[key] for key in ("run_id", "round_index")):
        raise CorruptRecord("native action I/O identity differs from snapshot run/round")
    observed = context["state"]["observation"]
    if (observed["observation_id"] != action["before_observation"]["observation_id"]
            or begin["state_before"].get("observation") != observed):
        raise CorruptRecord("native action begins at a different publication")
    try:
        boundary.state = before_interface(
            boundary.state, source, model_call_id=raw["model_call_id"],
            tool_call_id=raw["tool_call_id"], interface_ordinal=begin["identity"]["interface_ordinal"])
    except StateError as error:
        if str(error).startswith("未记录"):
            raise NotRecorded(str(error)) from error
        raise CorruptRecord(str(error)) from error
    return boundary, dict(model_call_id=raw["model_call_id"], tool_call_id=raw["tool_call_id"],
                          interface_ordinal=begin["identity"]["interface_ordinal"],
                          invocation_id=begin["invocation_id"], action_id=action_id,
                          ledger_path=str(path.relative_to(source)), ledger_sha256=digest(path.read_bytes()))


def _zero(source, traces):
    rounds = {receipt.get("round_index", 0) for receipt, _ in traces}
    candidates = []
    for path in sorted((source / "state/steps").glob("step-*/snapshot.json")):
        document = decode(read_local(source, str(path.relative_to(source))))
        values = document["values"]
        round_index = values["env.observation"]["round_index"]
        rounds.add(round_index)
        if values.get("exec.native_actions", {}).get("used") == 0:
            candidates.append(document["step"])
    if len(rounds) != 1 or rounds != {0}:
        raise NotRecorded("未记录: native-action 0 is ambiguous across rounds")
    if not candidates:
        raise NotRecorded("未记录: no complete reset snapshot for native-action 0")
    return select_snapshot(source, step=min(candidates), raw_tool_state=True)


def _render_prefix(source, round_index, clock):
    events, current_round, offset, previous_clock = [], 0, None, None
    for index, event in enumerate(rows(source, "native/render-synchronization.jsonl")):
        value = event.get("simulation_time_offset_s", 0.0)
        if offset is not None and value != offset:
            if value < offset:
                raise CorruptRecord("render history offset moved backwards")
            current_round += 1
            previous_clock = None
        offset = value
        tick = event["physics_step"]
        if type(tick) is not int or (previous_clock is not None and tick < previous_clock):
            raise CorruptRecord("render history clock is invalid")
        previous_clock = tick
        # The selected endpoint is the end of the tick, before its later capture.
        if (current_round, tick) >= (round_index, clock):
            break
        events.append(dict(index=index, round_index=current_round, physics_clock=tick,
                           recorded_render_calls=len(event["samples"]), matched=event.get("matched")))
    return dict(status="recorded" if (source / "native/render-synchronization.jsonl").exists()
                else "unavailable", events=events, source="native/render-synchronization.jsonl")


def freeze(source: Path, output: Path, number: int):
    """N=0 reset; N>0 exactly the tick whose controller native_action equals N."""
    if type(number) is not int or number < 0:
        raise ValueError("--native-action must be a nonnegative completed-tick count")
    traces = _traces(source)
    selected = [(index, position, receipt, trace) for index, (receipt, trace) in enumerate(traces)
                for position, sample in enumerate(trace["samples"]) if sample["native_action"] == number]
    if number:
        if len(selected) != 1:
            raise NotRecorded("未记录: native-action is absent or ambiguous across rounds")
        index, position, receipt, trace = selected[0]
        state, mapping = _action_boundary(source, receipt)
        round_index = receipt.get("round_index", 0)
        if state.context.json()["round_index"] != round_index:
            raise CorruptRecord("native action differs from snapshot round")
        upto = index + 1
    else:
        state, mapping, round_index, upto = _zero(source, traces), None, 0, 0
        index, position, receipt, trace = None, None, None, None
    native = output / "source-native"
    native.mkdir()
    refs, derived = [], []

    def copy(name, data=None):
        original = read_local(source / "native", name)
        refs.append(dict(path="native/" + name, sha256=digest(original), bytes=len(original)))
        target = native / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original if data is None else data)
        derived.append(dict(path="source-native/" + name, sha256=digest(target.read_bytes()), bytes=target.stat().st_size))
        return decode(original) if name.endswith(".json") else original

    reset = copy("reset.json")
    copy("source-manifest.json")
    total, final_lines = 0, []
    for i, (item, control) in enumerate(traces[:upto]):
        name = f"controller/action-{i:04d}.jsonl"
        if control["physics_journal"]["path"] != name:
            raise CorruptRecord("native physics journal path differs from trace")
        data = verified(source / "native", control["physics_journal"])
        lines = data.splitlines(keepends=True)
        if len(lines) != len(control["samples"]) * 10:
            raise NotRecorded("未记录: native samples do not map to ten physics lines each")
        keep = position + 1 if i == index else len(control["samples"])
        final_lines = [decode(line) for line in lines[:keep * 10]]
        if any(row.get("action_id") != item["action_id"] for row in final_lines):
            raise CorruptRecord("native physics line action identity mismatch")
        for j, sample in enumerate(control["samples"][:keep]):
            group = final_lines[j * 10:(j + 1) * 10]
            if (any(row["native_action"] != sample["native_action"] for row in group)
                    or group[-1]["measured"] != sample["measured"]):
                raise CorruptRecord("native sample differs from its physics interval")
        truncated = b"".join(lines[:keep * 10])
        copy(name, truncated)
        control = dict(control, samples=control["samples"][:keep],
                       physics_journal={**control["physics_journal"], "sha256": digest(truncated)})
        if "bytes" in control["physics_journal"]:
            control["physics_journal"]["bytes"] = len(truncated)
        copy(f"controller/action-{i:04d}.json", canonical(control) + b"\n")
        # Replayer needs only identity/action/round/status, not the future receipt.
        minimal = {key: item.get(key) for key in ("action_id", "action", "round_index", "execution_status")}
        copy(f"receipts/action-{i:04d}.json", canonical(minimal) + b"\n")
        total += keep
    for i in range(1, round_index + 1):
        copy(f"round-resets/round-{i:04d}.json")
    values = state.state.access(reader="rsi").values()
    observed = values["env.observation"]
    manifest = decode(verified(source, observed["manifest"]))
    exec_values = {key: value for key, value in values.items() if key.startswith("exec.")}
    if number:
        sample = trace["samples"][position]
        for arm in ("left", "right"):
            feedback = sample["measured"].get(arm, {}).get("gripper_feedback", {})
            if feedback.get("requested_opening_normalized") is None:
                raise NotRecorded(f"未记录: native tick lacks {arm} gripper directive")
        clock, sim_steps = final_lines[-1]["physics_clock"], sample["physics_steps"]
        proprioception = dict(end_effectors=sample["measured"], single_arm=False)
        # This pre-action update happens before the first tick, even when the
        # action has no finished receipt at the selected physical boundary.
        # Only the pose routes call _remember_backtrack. Recovery reports a
        # global strategy too, but must retain the anchor it is recovering to,
        # as must a recovery trajectory (``trajectory.recovery``). The anchor's
        # recorded path is not rebuilt mid-action: a recovery from here falls
        # back to joint interpolation.
        if (receipt.get("action") in {"move_pose", "move_free_space", "follow_joint_trajectory"}
                and receipt.get("motion_strategy") is not None
                and not (receipt.get("trajectory") or {}).get("recovery")):
            for arm in receipt.get("target_arms", ()):
                before = receipt["before"][arm]
                exec_values[f"exec.recovery.{arm}.previous"] = {
                    key: before[key] for key in ("arm_joint_qpos", "pose_xyz_xyzw")}
    else:
        clocks = [row for row in rows(source, "native/observation-clocks.jsonl")
                  if row.get("observation_sequence") == manifest["sequence"]]
        if len(clocks) != 1 or clocks[0].get("read_only") is not True:
            raise NotRecorded("未记录: reset publication has no unique read-only physics clock")
        clock = clocks[0]["before_physics_steps"]
        if clock != clocks[0]["after_physics_steps"] or clock != reset["physics_steps_before_policy"]:
            raise CorruptRecord("reset snapshot clock differs from reset boundary")
        sim_steps, proprioception = 0, manifest["proprioception"]
    ground_truth = _ground_truth(source, round_index, clock)
    expected = dict(round_index=round_index, native_actions=number, physics_clock=clock,
                    sim_steps=sim_steps, proprioception=proprioception, objects=None if ground_truth is None else ground_truth["objects"],
                    native_receipts=upto, observation_sequence=manifest["sequence"] + 1,
                    previous_execution_receipt=manifest.get("previous_execution_receipt"))
    boundary = dict(schema=SCHEMA, expected=expected, native_ticks=total, source_native=str(native),
                    source_refs=refs, derived_refs=derived, action_mapping=mapping,
                    base_snapshot_sha256=state.snapshot_sha256, base_step=state.step,
                    exec_values=deepcopy(exec_values), ground_truth=ground_truth, render_history=_render_prefix(source, round_index, clock))
    (output / "native-boundary.json").write_bytes(canonical(boundary) + b"\n")
    return state, boundary


def _ground_truth(source, round_index, clock):
    """Private verifier evidence only, selected by the exact physical clock."""
    plan = decode(read_local(source, "plan.json"))
    root = source.parent.parent / "ground-truth" / source.name
    required = bool(plan.get("ground_truth_capture") or plan.get("runtime", {}).get("ground_truth_capture"))
    if not (root / "manifest.json").exists():
        if required:
            raise NotRecorded("未记录: enabled ground truth capture has no manifest")
        return None
    manifest = decode(read_local(root, "manifest.json"))
    if manifest.get("run_id") != source.name or manifest.get("errors"):
        raise CorruptRecord("ground truth identity or capture errors are invalid")
    matches = [row for row in rows(root, "ticks.jsonl") if row.get("round_index") == round_index
               and row.get("physics_clock") == clock]
    if not matches or any(row.get("objects") != matches[0].get("objects") for row in matches):
        raise NotRecorded("未记录: native tick has no unique captured object state")
    objects = matches[0].get("objects")
    if not isinstance(objects, dict) or any("error" in value for value in objects.values()):
        raise CorruptRecord("native tick object state is incomplete")
    return matches[0]
