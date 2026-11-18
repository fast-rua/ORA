"""Freeze a published before-call state and its completed native control prefix."""

import re
from pathlib import Path

from ora.agent.toolkit.interface.types import (
    CorruptRecord,
    NotRecorded,
    canonical,
    decode,
    digest,
)

from .selection import legacy_snapshot, read_local, rows


def verified(root, ref):
    data = read_local(root, ref["path"])
    if digest(data) != ref["sha256"] or ("bytes" in ref and len(data) != ref["bytes"]):
        raise CorruptRecord("live boundary artifact hash/length mismatch")
    return data


def freeze(source, output, selected, events):
    """No timestamp guessing: publication -> manifest -> last native receipt -> prefix.

    The observation clock must be uniquely recorded and read-only. A selected
    call in a half-published mutation is refused. Private GT is verifier-only;
    it never enters tool inputs or the public interface store.
    """
    snapshot = legacy_snapshot(source, selected, events)
    observation = snapshot["available_at_boundary"]["observation"]
    manifest = decode(verified(source, observation["manifest"]))
    sequence, round_index = manifest["sequence"], manifest["round_index"]
    clocks = [
        row
        for row in rows(source, "native/observation-clocks.jsonl")
        if row.get("observation_sequence") == sequence
    ]
    if len(clocks) != 1 or clocks[0].get("read_only") is not True:
        raise NotRecorded("live requires one recorded read-only observation clock")
    clock = clocks[0]["before_physics_steps"]
    if type(clock) is not int or clocks[0].get("after_physics_steps") != clock:
        raise CorruptRecord("observation advanced physics or has an invalid clock")
    previous = manifest.get("previous_execution_receipt")
    upto = 0
    if previous:
        match = re.fullmatch(r"receipts/action-(\d+)\.json", previous["path"])
        if match is None:
            raise CorruptRecord("unexpected native receipt reference")
        upto = int(match[1]) + 1
        recorded = decode(verified(source / "native", previous))
        if recorded != previous["receipt"]:
            raise CorruptRecord("published receipt differs from native receipt")
    state_events = [
        event
        for event in events
        if event["seq"] < selected["proposal"]["seq"]
        and event["type"] == "fresh_observation_published"
    ]
    if not state_events:
        raise NotRecorded("live requires a before-call observation publication")
    publication = state_events[-1]
    if any(
        event["type"] in {"action_proposed", "environment_receipt", "round_started"}
        for event in events
        if publication["seq"] < event["seq"] < selected["proposal"]["seq"]
    ):
        raise NotRecorded(
            "selected call is inside an unpublished mutation/reset boundary"
        )
    # A new round can have no current receipt, while previous rounds still
    # contribute to the session-global controller prefix.
    if round_index and not previous:
        for event in state_events[:-1]:
            ref = event["payload"].get("state_ref")
            if ref:
                old_state = decode(verified(source, ref))
                old_manifest = decode(
                    verified(source, old_state["observation"]["manifest"])
                )
                old_receipt = old_manifest.get("previous_execution_receipt")
                if old_receipt:
                    match = re.fullmatch(
                        r"receipts/action-(\d+)\.json", old_receipt["path"]
                    )
                    if (
                        match is None
                        or decode(verified(source / "native", old_receipt))
                        != old_receipt["receipt"]
                    ):
                        raise CorruptRecord(
                            "invalid previous-round publication receipt"
                        )
                    upto = max(upto, int(match[1]) + 1)
    state_ref = publication["payload"].get("state_ref")
    if not state_ref or not state_ref["path"].startswith("policy-state/steps/"):
        raise NotRecorded("live requires a published policy-state boundary")
    state = decode(verified(source, state_ref))
    if state["observation"]["id"] != manifest["observation_id"]:
        raise CorruptRecord("state/publication observation mismatch")
    if state["proprioception"] != manifest["proprioception"]:
        raise CorruptRecord("state/publication robot measurements disagree")
    native = output / "source-native"
    native.mkdir()
    refs = []

    def copy(name, expected=None):
        data = read_local(source / "native", name)
        if expected is not None and digest(data) != expected:
            raise CorruptRecord("native control prefix hash mismatch")
        target = native / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        refs.append(
            dict(path="source-native/" + name, sha256=digest(data), bytes=len(data))
        )
        return decode(data) if name.endswith(".json") else None

    reset = copy("reset.json")
    copy("source-manifest.json")
    receipts = []
    total = 0
    for index in range(upto):
        receipt = copy(f"receipts/action-{index:04d}.json")
        trace_ref = receipt["control_trace"]
        trace_name = f"controller/action-{index:04d}.json"
        if trace_ref["path"] != trace_name:
            raise CorruptRecord("receipt/control trace sequence mismatch")
        trace = copy(trace_name, trace_ref["sha256"])
        journal_name = f"controller/action-{index:04d}.jsonl"
        if (
            trace["action_id"] != receipt["action_id"]
            or trace["physics_journal"]["path"] != journal_name
        ):
            raise CorruptRecord("native controller identity mismatch")
        copy(journal_name, trace["physics_journal"]["sha256"])
        total += len(trace["samples"])
        receipts.append(receipt)
    for index in range(1, round_index + 1):
        copy(f"round-resets/round-{index:04d}.json")
    # Clock validation is completed by the restored worker before any tool call;
    # reset's initialization physics is distinct from policy sim_steps.
    expected = dict(
        observation_id=manifest["observation_id"],
        observation_sequence=sequence,
        round_index=round_index,
        physics_clock=clock,
        proprioception=manifest["proprioception"],
        native_receipts=upto,
        native_actions=sum(
            int(r.get("native_actions", 0))
            for r in receipts
            if r.get("round_index", 0) == round_index
        ),
    )
    gt_root = source.parent.parent / "ground-truth" / source.name
    gt = None
    plan = decode(read_local(source, "plan.json"))
    gt_required = bool(
        plan.get("ground_truth_capture")
        or plan.get("runtime", {}).get("ground_truth_capture")
    )
    if gt_required and not (gt_root / "manifest.json").is_file():
        raise NotRecorded(
            "ground truth capture was enabled but its manifest is missing"
        )
    if (gt_root / "manifest.json").exists():
        gt_manifest = decode(read_local(gt_root, "manifest.json"))
        if gt_manifest.get("run_id") != source.name or gt_manifest.get("errors"):
            raise CorruptRecord("ground truth capture identity/errors are invalid")
        matches = [
            row
            for row in rows(gt_root, "ticks.jsonl")
            if row.get("round_index") == round_index
            and row.get("physics_clock") == clock
        ]
        if not matches or any(
            row.get("objects") != matches[0].get("objects") for row in matches
        ):
            raise NotRecorded(
                "ground truth exists but no unambiguous pose row matches the boundary clock"
            )
        gt = matches[0].get("objects")
        if not isinstance(gt, dict) or any("error" in value for value in gt.values()):
            raise CorruptRecord("boundary ground truth has incomplete object poses")
    expected["objects"] = gt
    expected["previous_execution_receipt"] = previous
    expected["sim_steps"] = sum(
        int(r.get("physics_steps", 0))
        for r in receipts
        if r.get("round_index", 0) == round_index
    )
    from .render_history import schedule

    value = dict(
        schema="ora.toolkit.live-boundary.v1",
        # The restored worker continues from the publication's execution state, in
        # particular the IK seed generators (exec.ik_sampler_state; toolkit_restore).
        exec_values=_exec_values(source, state_ref),
        render_history=schedule(source, manifest, observation["manifest"], clock, round_index),
        expected=expected,
        native_ticks=total,
        source_native=str(native),
        source_refs=refs,
        snapshot=snapshot,
        state_ref=state_ref,
        reset_physics_steps=reset.get("physics_steps_before_policy"),
    )
    (output / "live-boundary.json").write_bytes(canonical(value) + b"\n")
    return value


def _exec_values(source, state_ref):
    """exec.* of the state snapshot published with ``state_ref``; empty for a run without one."""
    step = state_ref.get("step")
    if type(step) is not int:
        found = re.fullmatch(r"policy-state/steps/step-(\d+)\.json", state_ref.get("path", ""))
        step = int(found[1]) if found else None
    path = source / f"state/steps/step-{step:04d}/snapshot.json" if step is not None else None
    if path is None or not path.is_file():
        return {}
    document = decode(read_local(source, str(path.relative_to(source))))
    if (document.get("policy_state") or {}).get("sha256") != state_ref["sha256"]:
        raise CorruptRecord("state snapshot is not bound to the published policy state")
    return {key: value for key, value in document["values"].items() if key.startswith("exec.")}
