"""The incident worker's own logic: what the session is told, what a scene read says, how a prefix runs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.environment import incident_replay_worker as incident
from ora.environment.replay_worker import ReplayWorkerError


def _gripper(**arguments):
    return {"action": "set_gripper", "arguments": arguments}


def test_gripper_directives_read_both_receipt_forms_and_keep_the_last():
    receipts = [{"action": "move_pose", "arguments": {"targets": {"left": {"position_m": [0, 0, 0]}}}},
                _gripper(targets={"left": "close"}), _gripper(targets={"right": "close"}),
                _gripper(targets={"left": "open"})]
    assert incident.gripper_directives(receipts) == {"left": "open", "right": "close"}
    assert incident.gripper_directives([_gripper(arm="right", gripper=0.0)]) == {"right": "close"}
    assert incident.gripper_directives([_gripper(arm="right", gripper=0.0), _gripper(arm="right", gripper=1.0)]) == {
        "right": "open"}
    assert incident.gripper_directives([_gripper(arm="right", gripper=0.4)]) == {}  # a number without a name


def test_a_robotwin_session_learns_the_prefix():
    session = SimpleNamespace(gripper_command_targets={"left": "unknown", "right": "unknown"}, action_sequence=0,
                              physics_step_sequence=0, last_receipt=None)
    previous = {"path": "receipts/action-0002.json", "sha256": "0" * 64, "receipt": {}}
    told = incident.synchronise(session, backend="robotwin", receipts=[_gripper(targets={"left": "close"})],
                                previous=previous, steps_ran=781)
    assert session.physics_step_sequence == 781 and session.last_receipt == previous
    assert session.gripper_command_targets == {"left": "close", "right": "unknown"}
    assert session.action_sequence == 0  # the controller replay of this output starts here
    assert told == {"steps": 781, "gripper_directives": {"left": "close"}}


def test_a_libero_session_learns_the_prefix():
    session = SimpleNamespace(sim_steps=0, last_receipt=None, _raw_observation=lambda: {"fresh": True})
    incident.synchronise(session, backend="libero_pro", receipts=[_gripper(arm="right", gripper=0.0)], previous=None,
                         steps_ran=389)
    assert session.sim_steps == 389 and session._gripper_command_target == "close"
    assert session.last_raw_observation == {"fresh": True}


def test_a_robotwin_scene_names_articulations_too():
    pose = SimpleNamespace(p=[0.0, 0.0, 1.0], q=[1.0, 0.0, 0.0, 0.0])
    actor = SimpleNamespace(get_name=lambda: "can", get_pose=lambda: pose)
    pot = SimpleNamespace(get_name=lambda: "pot", get_root_pose=lambda: pose)
    scene = SimpleNamespace(get_all_actors=lambda: [actor], get_all_articulations=lambda: [pot])
    adapter = incident._RoboTwin(SimpleNamespace())
    adapter.environment = SimpleNamespace(scene=scene)
    assert adapter.scene() == {"can": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                               "articulation:pot": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]}


def test_scene_changes_name_what_moved_largest_first():
    still = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    moved = [0.1, 0.2, 0.34, 1.0, 0.0, 0.0, 0.0]
    half_turn = [0.1, 0.2, 0.3, 0.7071068, 0.0, 0.0, 0.7071068]
    report = incident.scene_changes({"bowl": still, "cup": still, "plate": still},
                                    {"bowl": still, "cup": moved, "plate": half_turn})
    assert report["bodies"] == 3
    assert [row["body"] for row in report["moved"]] == ["cup", "plate"]
    assert report["moved"][0]["moved_mm"] == pytest.approx(40.0)
    assert report["moved"][1]["turned_deg"] == pytest.approx(90.0, abs=0.01)
    assert incident.scene_changes(None, {"bowl": still}) is None


class _Adapter:
    def __init__(self, ran="controls"):
        self.session = SimpleNamespace(restart_round=lambda index: self.rounds.append(index))
        self.rounds, self.started_calls, self.ran = [], 0, ran

    def started(self):
        self.started_calls += 1

    def replay(self, entry, arrays, step):
        return self.ran

    def pose(self, arm):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _prefix(monkeypatch, adapter, steps, rounds):
    monkeypatch.setattr(incident.worker, "_artifact_arrays", lambda path, entry: {})
    manifest = {"entries": [{"action_sequence": index, "round_index": round_index}
                            for index, round_index in enumerate(rounds)]}
    return incident.replay_prefix(adapter, steps, Path("manifest.json"), manifest)


def test_the_prefix_compares_each_end_pose_and_follows_rounds(monkeypatch):
    adapter = _Adapter()
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    other = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result = _prefix(monkeypatch, adapter, [
        {"action_sequence": 0, "kind": "controls", "physics_steps": 40, "expect": {"right": pose}},
        {"action_sequence": 1, "kind": "controls", "physics_steps": 2, "expect": {"right": other}},
    ], rounds=[0, 1])
    assert result["physics_steps"] == 42 and adapter.rounds == [1] and adapter.started_calls == 1
    assert result["steps"][0]["arms"]["right"]["bit_identical"] is True
    assert result["steps"][1]["arms"]["right"]["bit_identical"] is False
    assert result["steps"][1]["arms"]["right"]["position_mm"] == pytest.approx(1.0)


def test_a_record_that_replays_as_another_kind_stops_the_prefix(monkeypatch):
    with pytest.raises(ReplayWorkerError, match="replays as skipped"):
        _prefix(monkeypatch, _Adapter(ran="skipped"), [{"action_sequence": 0, "kind": "controls"}], rounds=[0])
