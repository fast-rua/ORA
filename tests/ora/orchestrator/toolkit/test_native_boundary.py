"""Completed native-tick axis, exact evidence links and no future receipt fields."""
import pytest

from ora.agent.toolkit.interface.types import CorruptRecord, NotRecorded, canonical, decode, digest
from ora.orchestrator.toolkit.native_boundary import freeze
from tests.support.agent.toolkit_native_snapshot import make_native


@pytest.mark.parametrize("number,upto,samples,step,marks", [
    (0, 0, 0, 0, {}), (1, 1, 1, 0, {"rim": [0, 2, 3]}),
    (2, 1, 2, 0, {"rim": [0, 2, 3]}), (3, 2, 1, 1, {"rim": [1, 2, 3]}),
    (4, 2, 2, 1, {"rim": [1, 2, 3]}),
])
def test_native_axis_zero_and_interval_endpoints(tmp_path, monkeypatch, number, upto, samples, step, marks):
    source = make_native(tmp_path / "source", monkeypatch)
    original = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "output"
    output.mkdir()
    state, boundary = freeze(source, output, number)
    assert state.step == step
    assert state.state.access(reader="rsi").read("tool.marks") == marks
    assert boundary["expected"]["native_receipts"] == upto
    assert boundary["expected"]["physics_clock"] == 10 + number * 10
    assert boundary["expected"]["native_actions"] == number
    assert boundary["native_ticks"] == number
    if number:
        trace = decode((output / f"source-native/controller/action-{upto - 1:04d}.json").read_bytes())
        assert len(trace["samples"]) == samples
        assert trace["samples"][-1]["native_action"] == number
        lines = (output / f"source-native/controller/action-{upto - 1:04d}.jsonl").read_bytes().splitlines()
        assert len(lines) == samples * 10
        receipt = decode((output / f"source-native/receipts/action-{upto - 1:04d}.json").read_bytes())
        assert "pose_targets_after" not in receipt and "command_targets_after" not in receipt
        assert boundary["exec_values"]["exec.recovery.left.previous"]["arm_joint_qpos"] != [-99] * 6
        assert boundary["action_mapping"]["interface_ordinal"] == 0
    assert original == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["absent", "duplicate_round", "no_io", "gap", "bad_hash", "bad_physics", "no_delta"])
def test_native_missing_ambiguous_or_corrupt_evidence_never_guesses(tmp_path, monkeypatch, kind):
    source = make_native(tmp_path / "source", monkeypatch)
    native = source / "native"
    number = 1
    error = NotRecorded
    if kind == "absent":
        number = 9
    elif kind == "no_io":
        (native / "toolkit-io/calls.jsonl").unlink()
    elif kind == "gap":
        (native / "toolkit-io/gaps.jsonl").write_bytes(canonical(dict(schema="ora.toolkit.gap.v1", tool_call_id="t-0")) + b"\n")
    elif kind == "no_delta":
        (source / "state/tool-deltas.jsonl").unlink()
    else:
        path = native / "controller/action-0001.json"
        trace = decode(path.read_bytes())
        if kind == "duplicate_round":
            for sample in trace["samples"]:
                sample["native_action"] -= 2
        elif kind == "bad_hash":
            trace["action_id"] = "tampered"
            error = CorruptRecord
        else:
            trace["samples"][0]["measured"] = {}
            number, error = 3, CorruptRecord
        path.write_bytes(canonical(trace))
        if kind != "bad_hash":
            receipt_path = native / "receipts/action-0001.json"
            receipt = decode(receipt_path.read_bytes())
            receipt["control_trace"]["sha256"] = digest(path.read_bytes())
            receipt["round_index"] = 1 if kind == "duplicate_round" else 0
            receipt_path.write_bytes(canonical(receipt))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(error):
        freeze(source, output, number)


def test_zero_is_ambiguous_with_multiple_recorded_rounds(tmp_path, monkeypatch):
    source = make_native(tmp_path / "source", monkeypatch)
    path = source / "native/receipts/action-0001.json"
    receipt = decode(path.read_bytes())
    receipt["round_index"] = 1
    path.write_bytes(canonical(receipt))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(NotRecorded, match="ambiguous across rounds"):
        freeze(source, output, 0)


def test_native_freeze_feeds_existing_control_reader_and_full_physics_verifier(tmp_path, monkeypatch):
    import shutil
    from ora.environment.backends.robodojo.control_replay import recorded_actions
    from ora.orchestrator.backends.robodojo.toolkit_live import verify_prefix
    source = make_native(tmp_path / "source", monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    _, boundary = freeze(source, output, 3)
    actions = recorded_actions(output / "source-native")
    assert [len(action["samples"]) for action in actions] == [2, 1]
    restored = tmp_path / "restored"
    shutil.copytree(output / "source-native", restored)
    report = dict(actual=boundary["expected"], replay=dict(ticks_recorded=3, ticks_replayed=3,
                                                         applied_target_mismatches=0, max_divergence={}))
    verified = verify_prefix(boundary, report, restored)
    assert verified["passed"] and verified["physics_prefix"]["lines_compared"] == 30
    path = restored / "controller/action-0001.jsonl"
    rows = [decode(line) for line in path.read_bytes().splitlines()]
    rows[0]["physics_clock"] += 1
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    assert not verify_prefix(boundary, report, restored)["passed"]


@pytest.mark.parametrize("kind,updated", [
    ("move_pose", True), ("move_free_space", True), ("recover_motion", False),
    ("set_gripper", False), ("wait", False),
])
def test_current_previous_anchor_changes_only_for_actual_pose_routes(tmp_path, monkeypatch, kind, updated):
    source = make_native(tmp_path / "source", monkeypatch, action_kind=kind)
    output = tmp_path / "output"
    output.mkdir()
    _, boundary = freeze(source, output, 1)
    previous = boundary["exec_values"]["exec.recovery.left.previous"]["arm_joint_qpos"]
    assert (previous != [-99] * 6) is updated


@pytest.mark.parametrize("difference", [None, "position", "orientation", "binding", "extra_field"])
def test_recorded_move_pose_normalization_matches_owner_once_and_rejects_other_changes(
    tmp_path, monkeypatch, difference
):
    from pathlib import Path
    from ora.contracts import validate_runtime_action
    fixture = decode((Path(__file__).resolve().parents[3] / "data/toolkit-native-action/move-pose-normalization.json").read_bytes())
    assert fixture["request"] != fixture["ledger_action"]
    assert validate_runtime_action(fixture["request"]) == fixture["ledger_action"]

    def change(action):
        if difference == "position":
            action["arguments"]["targets"]["left"]["position_m"][0] += 1e-10
        elif difference == "orientation":
            action["arguments"]["targets"]["left"]["orientation_xyzw"][0] += 1e-10
        elif difference == "binding":
            action["bound_observation_sha256"] = "0" * 64
        elif difference == "extra_field":
            action["unexpected"] = True

    source = make_native(tmp_path / "source", monkeypatch,
                         request_arguments=fixture["request"]["arguments"], ledger_transform=change)
    output = tmp_path / "output"
    output.mkdir()
    if difference:
        with pytest.raises(CorruptRecord, match="action differs from interface request"):
            freeze(source, output, 3)  # first tick of the next action, like real tick 37
    else:
        _, boundary = freeze(source, output, 3)
        assert boundary["action_mapping"]["action_id"] == "native-1"
        assert boundary["expected"]["native_actions"] == 3
