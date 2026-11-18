"""Identical measured inputs retain the released verification classifications."""
from types import SimpleNamespace
import numpy as np
import pytest
from ora.agent.toolkit.host.skills import driver, lift_evidence
from ora.agent.toolkit.tools.grasp_execute import _verify, _lift_evidence, _stages
from ora.orchestrator.episode.gripper_stage import apply_dojo_gripper_stage


@pytest.mark.parametrize("frame", ["robodojo_episode_local_ee_link", "other"])
@pytest.mark.parametrize("opening", [None, 0.0, .002, .03, .2])
@pytest.mark.parametrize("hold_signal", [False, True])
def test_verify_matches_released_measurement_path(frame, opening, hold_signal):
    effector = {"pose_frame": frame, "gripper_opening_m": opening, "gripper": .4,
                "gripper_feedback": {"measured_opening_normalized": .4,
                                     "arrival_check": {"tolerance_normalized": .01}}}
    observation = {"observation_id": "obs-3", "proprioception": {"end_effectors": {"left": effector}}}
    commitment = {"arm": "left", "localization": {"schema": "ora.target-localization.v1",
                  "principal_extents_m": [.02, .04, .05]}}
    completed = [{"stage": "close", "measurements": [{"quantity": "gripper_outcome", "value": {
        "left": {"state": "stalled_before_target", "measured_opening_normalized": .4,
                 "effective_target_opening_normalized": 0., "sdk_limited_drive_target_normalized": .2}}}]}]
    host = SimpleNamespace(surface=SimpleNamespace(supports=lambda _: hold_signal))
    assert _verify.measure("verify", observation, commitment, completed, hold_signal=hold_signal) == (
        driver._measure(host, "verify", observation, commitment, completed))


@pytest.mark.parametrize("delta", [0., .03])
def test_lift_surface_classification_matches_released_calculation(delta):
    points = np.arange(90).reshape(30, 3) * .001
    kwargs = dict(points=points, before_scene=points, after_scene=points + [0, 0, delta],
                  before_position=np.zeros(3), after_position=np.array([0, 0, delta]),
                  before_rotation=np.eye(3), after_rotation=np.eye(3))
    assert _lift_evidence.compare_surfaces(**kwargs) == lift_evidence.compare_surfaces(**kwargs)


@pytest.mark.parametrize("stage", ["open", "close"])
@pytest.mark.parametrize("status,code", [("reached", None), ("not_reached", "gripper_stalled_before_target"),
                                         ("not_reached", "gripper_not_reached"), ("refused", "invalid")])
@pytest.mark.parametrize("verified", [False, True])
def test_gripper_continuation_matches_released_stage(stage, status, code, verified):
    raw = {"execution_status": status, "failure_code": code, "consumed": True,
           "native_actions": 4, "command_invariant": {"verified": verified},
           "gripper_directives": {"left": 0. if stage == "close" else 1.},
           "arms": {"left": {"gripper_outcome": {"state": "stalled_before_target"}}}}
    old = {"status": "executed", "reason": "executed", "measurements": []}
    apply_dojo_gripper_stage(old, raw, {"left": stage})
    public = {"native_actions": 4, "detail": {"arms": raw["arms"]},
              "stage_evidence": {**raw, "command_invariant_verified": verified}}
    new = _stages.stage_result(stage, "left", public)
    assert {key: new[key] for key in old} == old


def test_five_primitives_use_new_publications_and_verify_is_read_only(monkeypatch):
    from ora.agent.toolkit.lib.values import JsonValue
    from ora.agent.toolkit.interface.types import Response, canonical
    state = {"step": 0, "remaining": 100, "actions": []}
    def observation(api):
        step = state["step"]
        return SimpleNamespace(identity=JsonValue.of({"observation_id": f"obs-{step}",
            "observation_sha256": str(step) * 64}), manifest=JsonValue.of({
            "observation_id": f"obs-{step}", "proprioception": {"end_effectors": {"left": {
                "pose_xyz_xyzw": [0., 0., .3, 0., 0., 0., 1.], "gripper_opening_m": .02}}}}))
    def execute(action, budget):
        assert action["observation_id"] == f"obs-{state['step']}"
        assert action["bound_observation_sha256"] == str(state["step"]) * 64
        assert budget == 20
        state["actions"].append(action)
        state["step"] += 1
        state["remaining"] -= 2
        return Response(canonical({"status": "ok", "data": {"remaining_budget": state["remaining"],
            "receipt": {"action_id": action["action_id"], "native_actions": 2,
                        "stage_evidence": {"execution_status": "reached", "consumed": True,
                                           "command_invariant_verified": True}}}}))
    monkeypatch.setattr(_stages, "observe", observation)
    held = {"tool.held.left": None}
    api = SimpleNamespace(act=SimpleNamespace(execute_action=execute),
                          state=SimpleNamespace(read=held.__getitem__, write=held.__setitem__))
    config = {"execution": {"generation": 0, "workspace_bounds_m": {}}}
    pose = {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}
    commitment = {"arm": "left", "candidate_id": "candidate", "source_ref": "target",
                  "poses": {name: pose for name in ["pregrasp", "approach", "lift"]}}
    result = _stages.run(api, config, commitment, ["open", "pregrasp", "approach", "close", "lift", "verify"])
    assert len(state["actions"]) == 5
    assert [a["observation_id"] for a in state["actions"]] == [f"obs-{i}" for i in range(5)]
    assert result["status"] == "executed" and result["remaining_budget"] == 90
    assert result["verification"]["gripper_opening_m"] == .02
    assert all("stage_evidence" not in stage["receipt"] for stage in result["stages"])


def test_model_projection_does_not_mutate_semantic_receipt():
    from ora.agent.toolkit.tools._receipts import model_receipt
    semantic = {"status": "reached", "native_actions": 2, "action_id": "a",
                "stage_evidence": {"consumed": True}}
    assert model_receipt(semantic) == {"status": "reached", "native_actions": 2, "action_id": "a"}
    assert semantic["stage_evidence"] == {"consumed": True}
