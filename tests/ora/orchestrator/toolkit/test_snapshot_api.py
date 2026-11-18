"""Completed snapshot replay reads original bytes and refuses physical capabilities."""
import json

import pytest

from ora.agent.toolkit.interface.types import AccessDenied, CorruptRecord, NotRecorded, SimulatorRequired
from ora.agent.toolkit.lib.observation import observe, load_camera
from ora.orchestrator.toolkit.snapshot_boundary import select_snapshot
from ora.orchestrator.toolkit.snapshot_api import build
from tests.support.agent.toolkit_snapshot import make_source


def test_before_parent_uses_snapshot_and_private_tool_overlay(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    state = make_source(source, monkeypatch)
    original = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    boundary = select_snapshot(source, at="1")
    assert boundary.step == 1 and boundary.selected["tool_call_id"] == "t-1"
    assert boundary.state.access().read("tool.marks") == {"rim": [0, 2, 3]}
    api, backend = build(boundary, output, services={}, bindings={})
    observation = observe(api)
    camera = load_camera(api, observation.camera("head"))
    assert camera.rgb.numpy().shape == (480, 640, 3)
    rgb = observation.camera("head").rgb.json()
    assert api.artifacts.read_blob(rgb) == (source / rgb["path"]).read_bytes()
    assert api.state.check_task().json()["data"]["task_success"] is False
    assert api.privileged.read("privileged.task_check")["terminal"] is False
    api.state.write("tool.marks", {"trial": True})
    assert state.access().read("tool.marks") == {"rim": [1, 2, 3]}
    assert backend.context.json()["boundary"]["model_call_id"] == "m-1"
    with pytest.raises(NotRecorded):
        api.privileged.read("privileged.ground_truth")
    with pytest.raises(AccessDenied, match="disjoint"):
        build(boundary, source / "bad-output", services={}, bindings={})
    assert not (source / "bad-output").exists()
    assert callable(api.context.declare)
    api.context.declare({"role": "fixture"})
    assert (output / "context-attachments.jsonl").is_file()
    for method in (api.planner.probe_motion, api.planner.align_plan, api.planner.reach_check):
        with pytest.raises(SimulatorRequired):
            method(request={})
    with pytest.raises(AccessDenied, match="cannot execute"):
        api.act.execute_action({}, 20)
    assert api.episode.finish_episode(request={"outcome": "give_up"}).json()["data"]["executed"] is False
    assert api.episode.report_issue(issue={"message": "trial"}).json()["data"]["executed"] is False
    assert len((output / "episode-controls.jsonl").read_text().splitlines()) == 2
    assert original == {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_step_selection_cannot_read_future_publication_or_mutate_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    boundary = select_snapshot(source, step=0)
    api, _ = build(boundary, tmp_path / "out", services={}, bindings={})
    assert api.state.read("tool.marks") == {}
    future = select_snapshot(source, step=1).context.json()["state"]["observation"]
    with pytest.raises(NotRecorded, match="not published"):
        api.state.observation(observation=future)
    with pytest.raises(NotRecorded):
        api.state.receipt(action_id="future")
    with pytest.raises(AccessDenied):
        api.state.write("exec.command.left", {})


@pytest.mark.parametrize("damage", ["missing_snapshot", "missing_watermark", "hash", "duplicate_event"])
def test_snapshot_selection_never_falls_back_to_legacy(tmp_path, monkeypatch, damage):
    make_source(tmp_path, monkeypatch)
    snapshot = tmp_path / "state/steps/step-0000/snapshot.json"
    expected = CorruptRecord
    if damage == "missing_snapshot":
        snapshot.unlink()
        expected = NotRecorded
    elif damage == "hash":
        snapshot.chmod(0o644)
        snapshot.write_bytes(b"{}")
    elif damage == "missing_watermark":
        from ora.agent.toolkit.state.store import encode
        from ora.agent.toolkit.interface.types import digest
        value = json.loads(snapshot.read_bytes())
        value.pop("tool_state_watermark")
        value.pop("snapshot_sha256")
        value["snapshot_sha256"] = digest(encode(value))
        snapshot.chmod(0o644)
        snapshot.write_bytes(encode(value))
        expected = NotRecorded
    else:
        path = tmp_path / "agent-events.jsonl"
        rows = path.read_bytes().splitlines()
        path.write_bytes(b"\n".join([rows[0], *rows]) + b"\n")
    with pytest.raises(expected):
        select_snapshot(tmp_path, at="0")
