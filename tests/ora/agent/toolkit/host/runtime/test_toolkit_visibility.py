"""Q58: robot historical selectors cannot retrieve intermediate primitive images."""
import json
import shutil

import pytest

from ora.agent.evidence.policy_state import PolicyStateStore, load_policy_state
from ora.agent.toolkit.host.observe import configuration
from ora.agent.toolkit.host.visibility import visible_steps
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.interface.types import canonical, digest
from tests.support.agent.toolkit_legacy import host
from tests.support.agent.toolkit_recording import proposal


def timeline(runtime):
    control, root, original = runtime._current()
    store = PolicyStateStore(runtime.control_root.parent, "run", "a" * 64)
    states = []
    for index in range(3):
        observation = json.loads(canonical(original))
        observation["observation_id"] = f"observation-{index:04d}"
        observation["sequence"] = index
        destination = root.parent / observation["observation_id"]
        shutil.copytree(root, destination, dirs_exist_ok=True)
        (destination / "observation.json").write_bytes(canonical(observation))
        state = store.publish(observation=observation, observation_root=destination,
                              cause="reset" if index == 0 else "after_action",
                              task_instruction="fixture", previous_public_action=None,
                              remaining_budget={"robot_decisions": 10, "model_calls": 20})
        states.append(state)
    control.update(observation_id=observation["observation_id"], source_root=str(destination),
                   observation_sha256=digest((destination / "observation.json").read_bytes()),
                   policy_state={"step": 2, "state_sha256": state["state_sha256"]})
    (runtime.control_root / "current-observation.json").write_bytes(canonical(control))
    runtime.state_root = store.root
    ledger = runtime.control_root.parent / "native/tool-executions/parent.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(canonical({"schema": "ora.tool-execution.v1", "finished": True, "pending": None,
                                 "initial_observation": {"step": 0, "observation_id": "observation-0000"},
                                 "final_observation": {"step": 2, "observation_id": "observation-0002"}}))
    return states, ledger


@pytest.mark.parametrize("name,args", [
    ("observe", {"step": 1, "views": ["head"]}),
    ("geometry_query", {"step": 1, "camera": "head", "points": [[300, 200]]}),
    ("vision_segment", {"step": 1, "camera": "head", "points": [[300, 200]]}),
    ("mark_point", {"step": 1, "view": "head", "points": [{"name": "hidden", "x": 300, "y": 200}]}),
])
def test_historical_routes_reject_internal_step_without_pixels(tmp_path, monkeypatch, name, args):
    runtime, client = host(tmp_path, monkeypatch)
    try:
        states, _ = timeline(runtime)
        config, publications = configuration(runtime)
        assert set(config["history"]) == {"0", "2"}
        assert "observation-0001" not in publications
        with pytest.raises(ToolError):
            runtime.invoke_with_context(name, args, call_context=proposal(runtime, name, args))
        assert runtime.toolkit_executor.requests == []
        # The operator's full state chain is untouched.
        assert load_policy_state(runtime.state_root.parent, 1)["observation"]["id"] == "observation-0001"
        assert runtime._select_observation(0)[0] == 0
        assert runtime._select_observation(2)[0] == 2
    finally:
        client.close()


def test_missing_or_unfinished_completion_does_not_grant_history(tmp_path, monkeypatch):
    runtime, client = host(tmp_path, monkeypatch)
    try:
        states, ledger = timeline(runtime)
        value = json.loads(ledger.read_bytes())
        value["finished"] = False
        ledger.write_bytes(canonical(value))
        assert visible_steps(runtime, states) == {0}
        ledger.unlink()
        assert visible_steps(runtime, states) == {0}
        ledger.write_text("broken")
        with pytest.raises(ToolError, match="visibility"):
            visible_steps(runtime, states)
    finally:
        client.close()


def test_hidden_target_and_query_refs_are_not_granted(tmp_path, monkeypatch):
    from ora.agent.toolkit.interface.blobs import BlobStore
    from ora.agent.toolkit.host.runtime.inputs import freeze
    runtime, client = host(tmp_path, monkeypatch)
    try:
        timeline(runtime)
        target = runtime.artifact_root / "segmented-target-index" / ("target-" + "0" * 24 + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical({"observation_id": "observation-0001"}))
        for directory, prefix in [("grounding-queries", "grounding-query-"), ("grasp-queries", "grasp-query-"),
                                  ("segmentation-batches", "segment-query-")]:
            manifest = runtime.artifact_root / directory / (prefix + "0" * 24) / "manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_bytes(canonical({"identity": {"observation_id": "observation-0001"}}))
        config, inputs = freeze(runtime, BlobStore(tmp_path, writable=True))
        assert inputs == {}
        assert config["current_target_refs"] == config["grounding_query_refs"] == config["grasp_query_refs"] == config["segmentation_query_refs"] == []
    finally:
        client.close()
