"""State ownership, immutable publication chains and native-byte loading."""

import hashlib
import json

import pytest

from ora.agent.toolkit.state.keys import StateAccessDenied, StateError
from ora.agent.toolkit.state.store import StateStore, encode
from ora.agent.toolkit.state.snapshot import export_snapshot, file_ref, load_publications, load_snapshot


def test_namespace_ownership_and_detached_values():
    state = StateStore()
    env = state.access(reader="supervisor", writer="environment")
    execution = state.access(reader="supervisor", writer="execution")
    tool = state.access(writer="tool")
    value = {"nested": [1]}
    tool.write("tool.held.left", value)
    value["nested"].append(2)
    assert tool.read("tool.held.left") == {"nested": [1]}
    tool.read("tool.held.left")["nested"].append(3)
    assert tool.read("tool.held.left") == {"nested": [1]}
    for key in ("env.terminal", "exec.command.left", "privileged.task_check"):
        with pytest.raises(StateAccessDenied):
            tool.write(key, {})
    with pytest.raises(StateAccessDenied):
        state.access(writer="execution")
    with pytest.raises(StateAccessDenied):
        env.write("exec.native_actions", {})
    execution.write("exec.session_home.left", {"arm_joint_qpos": [0.0]})
    env.write("privileged.task_check", {"task_success": True})
    with pytest.raises(StateAccessDenied):
        tool.read("privileged.task_check")
    assert "privileged.task_check" not in tool.values()
    assert state.access(reader="rsi").read("privileged.task_check")["task_success"]
    with pytest.raises(StateAccessDenied):
        state.access(privileged_keys=("privileged.task_check",))
    granted = state.access(privileged_keys=("privileged.task_check",), privileged_tools=("check_task",))
    assert granted.read("privileged.task_check")["task_success"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), (1,), {1: 2}, object()])
def test_invalid_json_and_atomic_update(bad):
    tool = StateStore().access(writer="tool")
    with pytest.raises(StateError):
        tool.update({"tool.first": 1, "tool.bad": bad})
    assert tool.values() == {}


def publication(root, state, step=0, previous=None):
    name = "policy-observations/observation-0000/image.png"
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unchanged native image bytes")
    ref = file_ref(root, name)
    state.access(reader="supervisor", writer="environment").update({
        "env.observation": {"step": step, "observation_id": "obs-0", "observation_sha256": ref["sha256"], "manifest": ref},
        "env.camera.head": {"files": {"rgb": ref}},
        "privileged.ground_truth": {"position": [0.123456789]},
    })
    body = {"schema": "ora.policy-state.v1", "run_id": "test", "step": step, "previous_state_sha256": previous,
            "observation": {"id": "obs-0", "manifest": ref}, "cameras": {"head": {"files": {"rgb": ref}}}}
    body["state_sha256"] = hashlib.sha256(encode(body)).hexdigest()
    name = f"policy-state/steps/step-{step:04d}.json"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(body))
    return {**file_ref(root, name, prefix="policy-state/steps/"), "step": step,
            "state_sha256": body["state_sha256"]}


def test_snapshot_roundtrip_no_image_copy_and_no_simulator(tmp_path):
    state = StateStore()
    policy = publication(tmp_path, state)
    state.access(writer="tool").write("tool.path", {"path": "/must/not/read"})
    marks = {"rim.1 / left": {"status": "solved", "xyz_m": [.1, .2, .3],
                             "sources": [{"observation_id": "observation-0000", "view": "head", "x": 30, "y": 20}]}}
    state.access(writer="tool").write("tool.marks", marks)
    with pytest.raises(StateError, match="incomplete"):
        export_snapshot(state, tmp_path, policy)
    state.publication_completed()
    export_snapshot(state, tmp_path, policy)
    loaded = load_snapshot(tmp_path, 0)
    assert loaded.access(reader="rsi").values() == state.access(reader="rsi").values()
    assert loaded.access().read("tool.marks") == marks
    name = "policy-observations/observation-0000/image.png"
    assert loaded.read_artifact(name) == (tmp_path / name).read_bytes()
    assert list((tmp_path / "state").rglob("*.png")) == []
    with pytest.raises(StateAccessDenied):
        loaded.store.access(writer="tool").write("tool.x", 1)
    with pytest.raises(StateError):
        loaded.read_artifact("/must/not/read")
    state.action_started()
    policy2 = publication(tmp_path, state, 1, policy["state_sha256"])
    state.publication_completed()
    export_snapshot(state, tmp_path, policy2)
    assert load_snapshot(tmp_path, 1).access().read("env.observation")["step"] == 1
    (tmp_path / name).write_bytes(b"drift")
    with pytest.raises(StateError, match="drift"):
        load_snapshot(tmp_path, 1)


def test_publications_validate_the_chain_once_and_keep_only_earlier_observations(tmp_path):
    state = StateStore()
    policy = publication(tmp_path, state)
    state.access(writer="tool").write("tool.marks", {"first": 1})
    state.publication_completed()
    export_snapshot(state, tmp_path, policy)
    state.action_started()
    policy2 = publication(tmp_path, state, 1, policy["state_sha256"])
    state.access(writer="tool").write("tool.marks", {"second": 2})
    state.publication_completed()
    export_snapshot(state, tmp_path, policy2)
    earlier, last = load_publications(tmp_path, 1)
    assert last.store._values == load_snapshot(tmp_path, 1).store._values
    # An earlier step grants its observation, never its tool or privileged values.
    assert set(earlier.access(reader="rsi").values()) == {"env.observation", "env.camera.head"}
    name = "policy-observations/observation-0000/image.png"
    assert earlier.read_artifact(name) == (tmp_path / name).read_bytes()
    (tmp_path / name).write_bytes(b"drift")
    with pytest.raises(StateError, match="drift"):  # an earlier file is checked when read
        earlier.read_artifact(name)
    with pytest.raises(StateError, match="drift"):  # the selected step's files are checked at load
        load_publications(tmp_path, 1)


def test_snapshot_chain_and_escape(tmp_path):
    state = StateStore()
    ref = publication(tmp_path, state)
    state.publication_completed()
    export_snapshot(state, tmp_path, ref)
    with pytest.raises(FileExistsError):
        export_snapshot(state, tmp_path, ref)
    for path in ("/etc/passwd", "policy-observations/../../escape", "elsewhere/file"):
        with pytest.raises(StateError):
            file_ref(tmp_path, path)
    second = publication(tmp_path, state, 1, "bad")
    with pytest.raises(StateError, match="aligned"):
        export_snapshot(state, tmp_path, second)
    target = tmp_path / "state/steps/step-0000/snapshot.json"
    body = json.loads(target.read_bytes())
    body["values"]["tool.forged"] = True
    target.chmod(0o644)
    target.write_text(json.dumps(body))
    with pytest.raises(StateError, match="identity"):
        load_snapshot(tmp_path, 0)


def test_snapshot_rejects_other_observation_at_the_same_step(tmp_path):
    state = StateStore()
    policy = publication(tmp_path, state)
    state.publication_completed()
    environment = state.access(reader="supervisor", writer="environment")
    identity = environment.read("env.observation")
    environment.write("env.observation", {**identity, "observation_id": "another-observation"})
    with pytest.raises(StateError, match="bound to this policy"):
        export_snapshot(state, tmp_path, policy)


def test_complete_snapshot_carries_tool_delta_watermark(tmp_path):
    from ora.agent.toolkit.state.deltas import before_call, bind_journal
    state = StateStore()
    journal = bind_journal(state, tmp_path)
    policy = publication(tmp_path, state)
    state.publication_completed()
    exported = export_snapshot(state, tmp_path, policy)
    assert exported["tool_state_watermark"] == journal.watermark()
    with journal.boundary({"kind": "owner", "operation": "initialize"}):
        state.access(writer="tool").write("tool.marks", {"rim": [1, 2, 3]})
    with journal.boundary({"kind": "parent", "model_call_id": "m1", "tool_call_id": "t1"}):
        state.access(writer="tool").write("tool.marks", {})
    loaded = load_snapshot(tmp_path, 0)
    overlay = before_call(loaded.store, tmp_path, model_call_id="m1", tool_call_id="t1")
    assert overlay.access().read("tool.marks") == {"rim": [1, 2, 3]}
    assert "tool.marks" not in loaded.access().values()
