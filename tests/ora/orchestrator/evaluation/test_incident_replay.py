"""The host side of ``ora replay --incident``: which stack runs, what is asked again, what the report says."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.orchestrator.evaluation import incident_replay as incident


def _plans(kind="libero_pro"):
    source = {"task": "t", "seed": 1, "runtime": {
        "kind": kind, "suite": "libero_goal_task", "task_id": 9, "workspace_bounds_m": {"x": [0, 1]},
        "motion_planner": {"version": "v0.7.8", "world_model": "none"}, "required_libero_pro_revision": "a",
        "required_data_revision": "b", "required_revision": "c", "task": "t", "task_config": "demo"}}
    stack = {"runtime": {
        "kind": kind, "suite": "libero_10_task", "task_id": 1, "workspace_bounds_m": {"x": [2, 3]},
        "motion_planner": {"implementation": "curobo_v2", "world_model": "public_depth_voxel_v1"},
        "required_libero_pro_revision": "a", "required_data_revision": "b", "required_revision": "c", "task": "other",
        "task_config": "other", "execution_capture": {"enabled": True}, "workspace_profile": "stack-only"}}
    return source, stack


def test_the_stack_run_gives_the_motion_stack_and_the_source_keeps_the_episode():
    runtime = incident.stack_runtime(*_plans())
    assert runtime["motion_planner"]["implementation"] == "curobo_v2"
    assert (runtime["suite"], runtime["task_id"]) == ("libero_goal_task", 9)
    assert runtime["workspace_bounds_m"] == {"x": [0, 1]}
    assert "workspace_profile" not in runtime  # the source run named none
    assert runtime["execution_capture"] == {"enabled": False}


def test_the_robotwin_episode_is_kept_too():
    runtime = incident.stack_runtime(*_plans("robotwin"))
    assert (runtime["task"], runtime["task_config"]) == ("t", "demo")


@pytest.mark.parametrize("field", ["required_libero_pro_revision", "required_data_revision"])
def test_another_simulator_revision_cannot_reproduce_the_prefix(field):
    source, stack = _plans()
    stack["runtime"][field] = "other"
    with pytest.raises(incident.IncidentReplayError, match=field):
        incident.stack_runtime(source, stack)


def test_a_stack_run_of_another_backend_is_refused():
    source, _ = _plans()
    with pytest.raises(incident.IncidentReplayError, match="not libero_pro"):
        incident.stack_runtime(source, _plans("robotwin")[1])


def test_the_action_asked_again_is_the_receipts_target_without_its_stage():
    receipt = {"action": "move_pose", "action_id": "action-239c:02-approach",
               "arguments": {"targets": {"right": {"position_m": [0, 0, 1]}}, "interactions": {"right": "free_space"}}}
    assert incident.incident_action(receipt) == {
        "action_id": "action-239c", "action": "move_pose", "arguments": receipt["arguments"]}


@pytest.mark.parametrize("receipt, reason", [
    ({"action": "recover_motion", "action_id": "a", "arguments": {"targets": {"right": "home"}}}, "recorded poses"),
    ({"action": "move_pose", "action_id": "a", "arguments": {"plan_ref": "plan-1"}}, "plan_ref"),
])
def test_an_action_that_cannot_be_asked_twice_is_refused(receipt, reason):
    with pytest.raises(incident.IncidentReplayError, match=reason):
        incident.incident_action(receipt)


def test_the_prefix_report_names_every_pose_that_differs():
    same = {"bit_identical": True, "position_mm": 0.0}
    different = {"bit_identical": False, "position_mm": 0.4}
    report = incident._prefix_report({"prefix": {"physics_steps": 10, "steps": [
        {"action_sequence": 0, "arms": {"left": same, "right": same}},
        {"action_sequence": 1, "arms": {"left": different}},
    ]}})
    assert report["poses_checked"] == 3 and report["bit_identical"] is False
    assert report["different"] == [{"action_sequence": 1, "arm": "left", "position_mm": 0.4}]
    assert incident._prefix_report({"error": "x"}) is None


def _run(tmp_path: Path, kind="libero_pro", rows=()) -> Path:
    root = tmp_path / "run"
    (root / "native/receipts").mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"runtime": {"kind": kind}}))
    (root / "agent-events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def test_the_report_stays_out_of_the_run_and_out_of_old_results(tmp_path):
    root = _run(tmp_path)
    with pytest.raises(incident.IncidentReplayError, match="outside the run"):
        incident.replay_incident(root, root / "report", 0)
    used = tmp_path / "used"
    used.mkdir()
    (used / "old.json").write_text("{}")
    with pytest.raises(incident.IncidentReplayError, match="not empty"):
        incident.replay_incident(root, used, 0)


def test_a_simulator_incident_needs_a_stack_run(tmp_path):
    with pytest.raises(incident.IncidentReplayError, match="--stack-run"):
        incident.replay_incident(_run(tmp_path), tmp_path / "report", 0)


def test_the_parent_call_gives_the_tool_and_drops_retired_arguments(tmp_path):
    root = _run(tmp_path, "robodojo", [
        {"type": "tool_call_proposed", "tool_call_id": "call-1", "payload": {
            "name": "move_pose", "arguments": {"targets": {"right": {}}, "native_action_budget": 20}}},
        {"type": "action_proposed", "tool_call_id": "call-1", "payload": {"action_id": "action-1"}},
    ])
    assert incident._parent_call(root, "action-1") == ("call-1", "move_pose", {"targets": {"right": {}}})
    with pytest.raises(incident.IncidentReplayError, match="no tool call"):
        incident._parent_call(root, "action-2")


def test_a_robodojo_incident_needs_a_slot_and_a_toolkit_run(tmp_path):
    root = _run(tmp_path, "robodojo")
    with pytest.raises(incident.IncidentReplayError, match="--resource-profile"):
        incident.replay_incident(root, tmp_path / "a", 0)
    with pytest.raises(incident.IncidentReplayError, match="toolkit run"):
        incident.replay_incident(root, tmp_path / "b", 0, resource_profile="robodojo-019-g3s0")


def test_a_robodojo_branch_says_what_the_tool_did_and_what_moved(tmp_path):
    (tmp_path / "verification.json").write_text(json.dumps({"passed": True, "physics_prefix": {"lines_compared": 9}}))
    pose = {"position_m": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}
    pushed = {"position_m": [0.03, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}
    report = incident._robodojo_branch({"status": "ok", "execution": {
        "receipts": [{"receipt": {"action": "follow_joint_trajectory", "failure_code": "blocked", "noise": 1}}],
        "objects_before": {"t_0_1": pose}, "state": {"objects": {"t_0_1": pushed}},
        "check_task": {"verdict": {"task_success": False}}}}, tmp_path)
    assert report["prefix"]["bit_identical"] is True
    assert report["receipts"] == [{"action": "follow_joint_trajectory", "failure_code": "blocked"}]
    assert report["scene"]["moved"][0]["body"] == "t_0_1"
    assert report["scene"]["moved"][0]["moved_mm"] == pytest.approx(30.0)


def test_a_robodojo_branch_that_stopped_early_has_no_scene(tmp_path):
    report = incident._robodojo_branch({"status": "error", "error": {"type": "NotRecorded"}}, tmp_path)
    assert report["status"] == "error" and report["prefix"] is None and report["receipts"] == []
    assert report["scene"] is None


def test_a_robodojo_incident_moves_with_the_default_baseline():
    stack = incident.robodojo_stack()
    assert stack["runtime"]["motion_planner"]["implementation"] == "curobo_v2"
    assert stack["runtime"]["motion_timing"]["implementation"] == "toppra_ruckig"
    assert stack["tool_surface"].startswith("robodojo_placement_toolkit_v")
    assert stack["toolkit_grants"] is None  # the default surface has no official check to grant


def test_a_robodojo_incident_hands_the_tool_the_stack_and_reports_it(tmp_path, monkeypatch):
    root = _run(tmp_path, "robodojo", [
        {"type": "tool_call_proposed", "tool_call_id": "call-1", "payload": {"name": "move_pose", "arguments": {}}},
        {"type": "action_proposed", "tool_call_id": "call-1", "payload": {"action_id": "action-1"}},
    ])
    (root / "native/toolkit-io").mkdir(parents=True)
    (root / "native/receipts").mkdir(parents=True, exist_ok=True)
    (root / "native/receipts/action-0000.json").write_text(json.dumps({"action_id": "action-1:0", "action": "m"}))
    (root / "plan.json").write_text(json.dumps({"runtime": {"kind": "robodojo"}, "policy": {
        "tool_surface": {"profile": "robodojo_placement_toolkit_v2"}}}))
    before = (root / "plan.json").read_bytes()
    stack = {"baseline": "b", "tool_surface": "surface-now", "runtime": {
        "motion_planner": {"implementation": "curobo_v2"}, "motion_timing": {"implementation": "toppra_ruckig"}}}
    monkeypatch.setattr(incident, "robodojo_stack", lambda: stack)
    called = {}

    def experiment(**kwargs):
        called.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr("ora.orchestrator.toolkit.replay.experiment", experiment)
    report = incident.replay_incident(root, tmp_path / "report", 0, resource_profile="robodojo-019-g3s0")
    assert called["motion_stack"] is stack and called["live"] is True
    assert report["stack"] == {
        "baseline": "b", "tool_surface": "surface-now", "motion_planner": "curobo_v2",
        "motion_timing": "toppra_ruckig", "source": {
            "tool_surface": "robodojo_placement_toolkit_v2", "motion_planner": None, "motion_timing": None}}
    assert (root / "plan.json").read_bytes() == before


def test_a_robodojo_branch_says_what_the_worker_moved_with(tmp_path):
    motion = tmp_path / "live-worker/environment/motion-planner"
    motion.mkdir(parents=True)
    (tmp_path / "surface-compatibility.json").write_text(json.dumps({"current": {"profile_id": "surface-now"}}))
    (motion / "binding.json").write_text(json.dumps({"implementation": "curobo_v2"}))
    (motion / "plans.jsonl").write_text(json.dumps(
        {"arm": "left", "status": "planned", "reason": None, "world": {"sha256": "x"}}) + "\n")
    (motion / "timing.jsonl").write_text(json.dumps({"arm": "left", "timing": {"implementation": "toppra_ruckig"}}) + "\n")
    assert incident._motion_used(tmp_path) == {
        "tool_surface": "surface-now", "planner": "curobo_v2",
        "plans": [{"arm": "left", "status": "planned", "reason": None}], "timing": ["toppra_ruckig"]}


def test_a_worker_that_started_no_planner_reports_none(tmp_path):
    assert incident._motion_used(tmp_path) == {"tool_surface": None, "planner": None, "plans": [], "timing": []}
