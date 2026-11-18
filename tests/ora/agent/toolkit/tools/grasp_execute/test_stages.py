"""grasp_execute's stages on a backend that runs joint trajectories: the same sequence and bookkeeping,
each arm stage one fast trajectory routed by its interaction, open/close the rate-limited jaw command."""
import json
from types import SimpleNamespace

import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.tools.grasp_execute import _stages

START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
POSES = {"pregrasp": {"position_m": [0.3, 0.0, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
         "approach": {"position_m": [0.3, 0.0, 0.27], "orientation_xyzw": [0, 0, 0, 1]},
         "lift": {"position_m": [0.3, 0.0, 0.37], "orientation_xyzw": [0, 0, 0, 1]}}


def _response(data, method):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}).encode())


def _api(sent, state, screens):
    def kinematics(*, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        if request["operation"] == "screen_path":
            return _response(screens.pop(0) if screens else {"status": "clear"}, "kinematics")
        if request["operation"] == "fk":
            return _response({"rows": [{"index": i, "pose_xyz_xyzw": [*q[:3], 0, 0, 0, 1]}
                                       for i, q in enumerate(request["joints_rad"])]}, "kinematics")
        return _response({"rows": [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0, 0, 0]}
                                   for i, pose in enumerate(request["poses_xyz_xyzw"])]}, "kinematics")

    def execute_action(action, budget):
        sent.append((action, budget))
        evidence = {"execution_status": "reached", "consumed": True, "command_invariant_verified": True,
                    "failure_code": None, "gripper_directives": {}}
        return _response({"remaining_budget": 100 - 10 * len(sent), "receipt": {
            "status": "reached", "native_actions": 10, "action_id": action["action_id"],
            "stage_evidence": evidence}}, "execute_action")

    return SimpleNamespace(planner=SimpleNamespace(kinematics=kinematics),
                           act=SimpleNamespace(execute_action=execute_action),
                           state=SimpleNamespace(read=lambda key: state.get(key), write=state.__setitem__))


@pytest.fixture
def run(monkeypatch):
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": START, "arm_joint_qpos": [*START[:3], 0, 0, 0],
                     "gripper_command_target": "open"}}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    monkeypatch.setattr(_stages, "observe", lambda api: obs)
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose._smooth.observation.observe", lambda api: obs)
    monkeypatch.setattr(_stages._verify, "measure", lambda *a: {"verification": {"jaw_hold": {"held": "likely"}}})
    monkeypatch.setattr(_stages._lift_evidence, "measure", lambda *a, **k: {"lift": "probe"})

    def call(screens=(), poses=POSES):
        sent, state = [], {}
        config = {"execution": {"generation": 4, "workspace_bounds_m": {}},
                  "synchronous_execution": True,
                  "motion": {"joint_trajectory": True, "capabilities": ["robodojo_staged_contact_approach_v1"]}}
        commitment = {"arm": "left", "source_ref": "target-1", "candidate_id": "c1", "poses": poses}
        sequence = ["open", "pregrasp", "approach", "close", "lift", "verify"]
        result = _stages.run(_api(sent, state, list(screens)), config, commitment, sequence)
        return result, sent, state
    return call


def test_the_stage_sequence_is_unchanged_and_each_stage_is_one_action(run):
    result, sent, state = run()
    assert result["status"] == "executed" and [stage["stage"] for stage in result["stages"]] == [
        "open", "pregrasp", "approach", "close", "lift"]
    kinds = [action["action"] for action, _ in sent]
    assert kinds == ["set_gripper", "follow_joint_trajectory", "follow_joint_trajectory", "set_gripper",
                     "follow_joint_trajectory"]
    # open at full rate (an empty jaw), close rate- and lead-limited after an arm settle.
    opening = sent[0][0]["arguments"]["targets"]["left"]
    closing = sent[3][0]["arguments"]["targets"]["left"]
    assert opening["opening"] == "open" and opening["rate_per_s"] is None
    assert closing == {"opening": "close", "rate_per_s": 2.0, "max_lead_m": 0.005,
                       "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 5}}
    assert all(budget <= 250 for action, budget in sent if action["action"] == "follow_joint_trajectory")
    assert state["tool.held.left"]["target_ref"] == "target-1"
    assert [action["action_id"] for action, _ in sent][1] == "action-4:01-pregrasp"


def test_a_long_pregrasp_spends_the_sequence_budget_and_arrives(run):
    """The released primitive's 20 ended this pregrasp short of its goal, so the grasp never closed."""
    far = {**POSES, "pregrasp": {"position_m": [3.3, 0.0, 0.3], "orientation_xyzw": [0, 0, 0, 1]}}
    result, sent, _ = run(poses=far)
    pregrasp, budget = sent[1]
    assert budget > 20 and "part_of_move" not in pregrasp["arguments"]  # the whole move, not a cut part
    assert result["status"] == "executed"


def test_an_approach_the_screen_blocks_stops_the_grasp_before_it_moves(run):
    result, sent, state = run(screens=[{"status": "clear"}, {"status": "table_collision", "arm": "left",
                                                             "first_blocked_segment": 1}])
    assert result["status"] == "stopped" and result["stages"][-1]["stage"] == "approach"
    assert result["stages"][-1]["status"] == "not_reached" and result["stages"][-1]["reason"] == "table_collision"
    assert result["receipt"]["status"] == "refused_before_motion" and result["receipt"]["native_actions"] == 0
    assert len(sent) == 2  # open and pregrasp ran; nothing was sent for the approach
