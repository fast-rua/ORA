"""return_home: lift, then a screened joint unwind to the reset joints, as one trajectory; refusals; replay.

Toy kinematics (joints 0-2 are the link6 position, IK writes the position into them); the state holds the home
anchors as the host writes them.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.sim_exec.replay import Actions
from ora.agent.toolkit.interface.types import ActionNotExecuted, NotRecorded, Response
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.recover_pose import _trajectory
from ora.agent.toolkit.tools.return_home import tool

HOME = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
START = [0.3, 0.0, 0.2, 0.5, 0.0, 0.0]
RIGHT = [0.3, -0.4, 0.2, 0.0, 0.0, 0.0]

#: Measured residual (planned last knot minus measured joints) of an arm that arrived and settled.
RESIDUAL = [0.00015, -0.0001, 0.0, 0.0, 0.0, 0.0]


def _pose(joints):
    return [*joints[:3], 0.0, 0.0, 0.0, 1.0]


def _response(data, method):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}).encode())


def _api(executed, *, screens=(), ceiling=0.6, state=None, act=None, evidence=None, override=None):
    evidence, override = evidence or {}, override or {}
    screens = list(screens)

    def kinematics(*, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        if request["operation"] == "screen_path":
            return _response(screens.pop(0) if screens else {"status": "clear"}, "kinematics")
        if request["operation"] == "fk":
            rows = [{"index": i, "pose_xyz_xyzw": _pose(q)} for i, q in enumerate(request["joints_rad"])]
        else:
            seed = request["seed_joints_rad"]
            rows = [{"index": i, "status": "ok", "joints_rad": [*pose[:3], *seed[3:]]}
                    for i, pose in enumerate(request["poses_xyz_xyzw"])]
        return _response({"rows": rows}, "kinematics")

    def execute_action(action, budget):
        executed.append((action, budget))
        arms = action["arguments"]["arms"]
        count = len(next(iter(arms.values()))["knots_rad"])
        # As the tool receives it (pi.tool_rpc projection): no after-state, the settle and per-joint residuals.
        trajectory = {"status": "completed", "knots_planned": count, "knots_executed": count, "stop": None,
                      "settle": {"settled": True, "joint_error_rad": 0.00015, "joint_speed_rad_s": 0.03},
                      "arms": {arm: {"final_joint_error_per_joint_rad": evidence.get(arm, RESIDUAL)} for arm in arms}}
        trajectory.update(override)
        receipt = {"status": "reached", "stage_evidence": {"trajectory": trajectory}}
        return _response({"remaining_budget": 50 - budget, "receipt": receipt}, "execute_action")

    state = {f"exec.recovery.{arm}.home": {"arm_joint_qpos": q, "pose_xyz_xyzw": _pose(q)}
             for arm, q in (("left", HOME), ("right", RIGHT))} if state is None else state

    def read(key):
        if key not in state:
            raise NotRecorded(key)
        return state[key]

    config = {"synchronous_execution": True, "execution": {"generation": 5},
              "motion": {"joint_trajectory": True, "mutation_refusals": {},
                         "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, ceiling]}}}
    return SimpleNamespace(artifacts=SimpleNamespace(tool_context=lambda: config),
                           planner=SimpleNamespace(kinematics=kinematics),
                           act=act or SimpleNamespace(execute_action=execute_action),
                           state=SimpleNamespace(read=read))


@pytest.fixture(autouse=True)
def current(monkeypatch):
    joints = {"left": list(START), "right": list(RIGHT)}

    def observe(api):
        effectors = {arm: {"pose_xyz_xyzw": _pose(q), "arm_joint_qpos": list(q)} for arm, q in joints.items()}
        return SimpleNamespace(manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": effectors}}),
                               identity=SimpleNamespace(json=lambda: {"observation_id": "o",
                                                                      "observation_sha256": "s"}))

    monkeypatch.setattr(_trajectory, "observe", observe)
    return joints


def test_it_lifts_straight_up_then_unwinds_to_the_reset_joints_in_one_trajectory():
    executed = []
    result = tool.run(_api(executed), {"arms": ["left"]})
    action, budget = executed[0]
    knots = np.asarray(action["arguments"]["arms"]["left"]["knots_rad"])
    peak = int(np.argmax(knots[:, 2]))
    assert knots[peak, 2] == pytest.approx(0.3, abs=2e-3) and np.allclose(knots[:peak, :2], [0.3, 0.0])
    assert knots[-1] == pytest.approx(HOME) and budget <= 50 and len(executed) == 1
    assert action["arguments"]["recovery"] == {"targets": {"left": "home"}, "recovery_targets": {
        "left": {"strategy": "home", "arm_joint_qpos": HOME, "pose_xyz_xyzw": _pose(HOME)}}}
    assert result["action"] == "return_home" and result["status"] == "executed"
    assert result["compiled_pose"]["targets"]["left"] == {"strategy": "home", "method": "joint_interpolation"}
    progress = result["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["route"] == "retract_then_unwind" and progress["retract"]["status"] == "planned"
    assert progress["outcome"] == "arrived" and progress["unwind_order"] == "straight_joint_line"


def test_at_the_workspace_ceiling_it_unwinds_without_the_lift():
    executed = []
    result = tool.run(_api(executed, ceiling=0.205), {"arms": ["left"]})
    knots = np.asarray(executed[0][0]["arguments"]["arms"]["left"]["knots_rad"])
    assert np.max(knots[:, 2]) <= 0.3 + 1e-9 and knots[-1] == pytest.approx(HOME)
    assert result["receipt"]["arms"]["left"]["recovery_progress"]["retract"] == {
        "status": "skipped", "reason": "workspace_ceiling", "lift_m": pytest.approx(0.005)}


def test_when_only_the_path_without_the_lift_is_clear_it_skips_the_lift():
    executed = []
    blocked = [{"status": "table_collision", "arm": "left"}] * 3  # every order with the lift
    result = tool.run(_api(executed, screens=blocked), {"arms": ["left"]})
    progress = result["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["retract"]["status"] == "skipped" and progress["retract"]["reason"] == "path_blocked"
    knots = np.asarray(executed[0][0]["arguments"]["arms"]["left"]["knots_rad"])
    assert np.max(knots[:, 2]) <= 0.3 + 1e-9


def test_every_path_blocked_refuses_before_anything_moves():
    executed = []
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(executed, screens=[{"status": "self_collision"}] * 6), {"arms": ["left"]})
    assert error.value.code == "plan_rejected" and not executed
    assert error.value.constraints["failure_code"] == "recovery_self_collision_path"


def test_missing_reset_joints_spend_nothing():
    executed = []
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(executed, state={}), {"arms": ["left"]})
    assert error.value.constraints["failure_code"] == "recovery_state_unavailable" and not executed


def test_a_recorded_replay_plans_without_a_simulator_and_proposes_the_trajectory():
    with pytest.raises(ActionNotExecuted) as stop:
        tool.run(_api([], act=Actions()), {"arms": ["left", "right"]})
    assert stop.value.method == "execute_action" and stop.value.budget <= 50
    arms = stop.value.action["arguments"]["arms"]
    assert set(arms) == {"left", "right"} and arms["left"]["knots_rad"][-1] == pytest.approx(HOME)
    assert stop.value.action["arguments"]["recovery"]["targets"] == {"left": "home", "right": "home"}


def test_a_completed_and_settled_return_is_arrived_from_its_measured_evidence():
    """I-0005: the receipt the tool gets has no after-state; the trajectory evidence says it arrived."""
    result = tool.run(_api([], evidence={"left": [0.00005, 0.0, 0.0, 0.0, 0.0, 0.0]}), {"arms": ["left"]})
    progress = result["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["outcome"] == "arrived" and progress["joint_distance_end_rad"] == pytest.approx(0.00005)
    unknown = tool.run(_api([], override={"settle": None, "arms": {}}), {"arms": ["left"]})
    progress = unknown["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["outcome"] == "unknown" and progress["joint_distance_end_rad"] is None
