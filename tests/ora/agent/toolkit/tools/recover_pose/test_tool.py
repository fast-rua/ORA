"""recover_pose: undo the latest arm-moving tool call along its measured path (arms), else the released route.

Toy kinematics (joints 0-2 are the link6 position); the state holds the undo stacks
(``exec.recovery.<arm>.undo``) and anchors as the host writes them. The fake-plant run of the stack itself is
tests/ora/environment/backends/robodojo/test_robodojo_retrace_tools.py.
"""
import json
from types import SimpleNamespace

import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.sim_exec.replay import Actions
from ora.agent.toolkit.interface.types import ActionNotExecuted, NotRecorded, Response
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.recover_pose import _trajectory, tool

HOME = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
RIGHT = [0.3, -0.4, 0.2, 0.0, 0.0, 0.0]
CONFIG = {"synchronous_execution": True, "execution": {"generation": 9},
          "motion": {"joint_trajectory": True, "mutation_refusals": {},
                     "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 0.6]}}}

#: Measured residual (planned last knot minus measured joints) of an arm that arrived and settled.
RESIDUAL = [0.00015, -0.0001, 0.0, 0.0, 0.0, 0.0]


def _pose(joints):
    return [*joints[:3], 0.0, 0.0, 0.0, 1.0]


def _response(data, method):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}).encode())


def _api(state, executed, *, screens=(), config=None, act=None, evidence=None, override=None):
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
            rows = [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0.0, 0.0, 0.0]}
                    for i, pose in enumerate(request["poses_xyz_xyzw"])]
        return _response({"rows": rows}, "kinematics")

    def execute_action(action, budget):
        executed.append((action, budget))
        if action["action"] == "recover_motion":
            return _response({"remaining_budget": 20 - budget, "receipt": {"status": "reached"}}, "execute_action")
        arms = action["arguments"]["arms"]
        count = len(next(iter(arms.values()))["knots_rad"])
        # As the tool receives it (pi.tool_rpc projection): no after-state, the settle and per-joint residuals.
        trajectory = {"status": "completed", "knots_planned": count, "knots_executed": count, "stop": None,
                      "settle": {"settled": True, "joint_error_rad": 0.00015, "joint_speed_rad_s": 0.03},
                      "arms": {arm: {"final_joint_error_per_joint_rad": evidence.get(arm, RESIDUAL)} for arm in arms}}
        trajectory.update(override)
        receipt = {"status": "reached", "native_actions": count, "stage_evidence": {"trajectory": trajectory},
                   "arms": {arm: {"targeted": True, "motion_progress": {"outcome": "arrived"}} for arm in arms}}
        return _response({"remaining_budget": 50 - budget, "receipt": receipt}, "execute_action")

    def read(key):
        if key not in state:
            raise NotRecorded(key)
        return state[key]

    return SimpleNamespace(artifacts=SimpleNamespace(tool_context=lambda: config or CONFIG),
                           planner=SimpleNamespace(kinematics=kinematics),
                           act=act or SimpleNamespace(execute_action=execute_action),
                           state=SimpleNamespace(read=read))


@pytest.fixture
def current(monkeypatch):
    joints = {"left": list(START), "right": list(RIGHT)}

    def observe(api):
        effectors = {arm: {"pose_xyz_xyzw": _pose(q), "arm_joint_qpos": list(q)} for arm, q in joints.items()}
        return SimpleNamespace(manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": effectors}}),
                               identity=SimpleNamespace(json=lambda: {"observation_id": "o",
                                                                      "observation_sha256": "s"}))

    monkeypatch.setattr(tool, "observe", observe)
    monkeypatch.setattr(_trajectory, "observe", observe)
    return joints


def _path(begin, end, ticks=6):
    return [[a + (b - a) * k / ticks for a, b in zip(begin, end)] for k in range(ticks + 1)]


def _entry(path, ordinal):
    return {"tool_call_id": f"call-{ordinal}", "tool": "move_pose", "ordinal": ordinal, "arm_joint_qpos": path[0],
            "pose_xyz_xyzw": _pose(path[0]), "measured_path_rad": path}


def _state(*moves, right=()):
    """Undo stacks (oldest first) as the environment writes them, plus the anchors the released route reads."""
    state = {f"exec.recovery.{arm}.undo": {"moves": [_entry(path, index + 1) for index, path in enumerate(paths)]}
             for arm, paths in (("left", moves), ("right", right))}
    for arm, joints in (("left", HOME), ("right", RIGHT)):
        for name in ("home", "previous"):
            state[f"exec.recovery.{arm}.{name}"] = {"arm_joint_qpos": joints, "pose_xyz_xyzw": _pose(joints)}
    return state


def test_each_call_retraces_the_latest_move_and_restates_the_stack_top(current):
    first, second = [0.2, 0.0, 0.2, 0.0, 0.0, 0.0], [0.25, 0.0, 0.2, 0.0, 0.0, 0.0]
    executed = []
    result = tool.run(_api(_state(_path(first, second), _path(second, START)), executed), {"arms": ["left"]})
    action, budget = executed[0]
    assert len(executed) == 1 and action["action"] == "follow_joint_trajectory" and action["action_id"] == "action-9"
    assert budget <= 50 and action["arguments"]["arms"]["left"]["knots_rad"][-1] == pytest.approx(second)
    assert action["arguments"]["recovery"] == {"targets": {"left": "retrace"},
                                               "retrace": {"left": {"moves": 2, "arm_joint_qpos": second}}}
    assert "right" not in action["arguments"]["arms"]  # an omitted arm receives no drive target
    assert result["status"] == "executed" and result["action"] == "recover_pose"
    assert result["compiled_pose"]["targets"]["left"] == {"strategy": "retrace", "method": "measured_retrace"}
    progress = result["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["outcome"] == "arrived" and progress["route"] == progress["method"] == "measured_retrace"
    assert progress["undo"] == {"undone_call": {"tool_call_id": "call-2", "tool": "move_pose", "ordinal": 2},
                                "complete": True,
                                "undoable_calls": [{"tool_call_id": "call-1", "tool": "move_pose", "ordinal": 1}]}
    # After the environment popped that move, the next call undoes the one before it.
    current["left"] = list(second)
    tool.run(_api(_state(_path(first, second)), executed), {"arms": ["left"]})
    assert executed[1][0]["arguments"]["arms"]["left"]["knots_rad"][-1] == pytest.approx(first)
    assert executed[1][0]["arguments"]["recovery"]["retrace"]["left"]["moves"] == 1


@pytest.mark.parametrize("state, code, reason", [
    ({}, "retrace_not_recorded", "records no undo paths"),
    (_state(), "retrace_nothing_to_undo", "only the most recent 5 arm-moving tool calls per arm can be undone"),
    (_state(_path(HOME, [0.1, 0.0, 0.3, 0.0, 0.0, 0.0])), "retrace_arm_off_path", "no longer stands on the path"),
])
def test_without_a_move_to_retrace_it_refuses_and_names_the_other_tools(current, state, code, reason):
    executed = []
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(state, executed), {"arms": ["left"]})
    assert not executed and error.value.code == "plan_rejected" and not error.value.fields["retryable"]
    assert error.value.constraints["failure_code"] == code and error.value.constraints["native_actions"] == 0
    assert error.value.constraints["use_instead"] == ["return_home", "move_pose"]
    assert reason in str(error.value) and "return_home" in str(error.value) and "move_pose" in str(error.value)


def test_a_blocked_retrace_is_refused_with_no_fallback(current):
    executed = []
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(_state(_path(HOME, START)), executed, screens=[{"status": "inter_arm_collision"}]),
                 {"arms": ["left"]})
    assert error.value.constraints["failure_code"] == "recovery_inter_arm_collision_path" and not executed
    assert len(error.value.constraints["screens"]) == 1  # the retrace only: no joint-interpolation candidate


def test_a_cut_retrace_stays_on_top_and_the_next_call_continues_on_the_path(current):
    begin = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    path = _path(begin, [4.0, 0.0, 0.2, 0.0, 0.0, 0.0], ticks=400)  # 4 m: more than 250 native actions at normal
    current["left"] = list(path[-1])
    executed = []
    first = tool.run(_api(_state(path), executed), {"arms": ["left"]})
    knots = executed[0][0]["arguments"]["arms"]["left"]["knots_rad"]
    assert "part_of_move" in executed[0][0]["arguments"] and executed[0][1] <= 250
    progress = first["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["cut_short"]["native_action_budget"] == 250 and progress["undo"]["complete"] is False
    assert 0.2 < progress["undo"]["path_left_fraction"] < 0.8 and len(progress["undo"]["undoable_calls"]) == 1
    assert progress["outcome"] == "cut_short"
    current["left"] = list(knots[-1])  # the move is still on top; the arm stands on its path
    second = tool.run(_api(_state(path), executed), {"arms": ["left"]})
    assert second["receipt"]["arms"]["left"]["recovery_progress"]["undo"]["complete"] is True
    assert executed[1][0]["arguments"]["arms"]["left"]["knots_rad"][-1] == pytest.approx(begin)


def test_both_arms_undo_their_own_latest_moves_in_one_trajectory(current):
    executed = []
    tool.run(_api(_state(_path(HOME, START), right=[_path([0.3, -0.3, 0.2, 0, 0, 0], RIGHT)]), executed),
             {"arms": ["right", "left"]})
    arms = executed[0][0]["arguments"]["arms"]
    assert arms["left"]["knots_rad"][-1] == pytest.approx(HOME)
    assert arms["right"]["knots_rad"][-1] == pytest.approx([0.3, -0.3, 0.2, 0, 0, 0])
    assert executed[0][0]["arguments"]["recovery"]["targets"] == {"left": "retrace", "right": "retrace"}


@pytest.mark.parametrize("arguments", [{"arms": []}, {"arms": ["left", "left"]}, {"arms": ["up"]},
                                       {"arms": ["left"], "extra": 1}])
def test_arms_are_validated(current, arguments):
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(_state(_path(HOME, START)), []), arguments)
    assert error.value.code == "invalid_motion_arguments"


def test_a_recorded_replay_plans_without_a_simulator_and_proposes_the_retrace(current):
    """The snapshot replay's act capability stops the tool at its action: not_executed, nothing executed."""
    with pytest.raises(ActionNotExecuted) as stop:
        tool.run(_api(_state(_path(HOME, START)), [], act=Actions()), {"arms": ["left"]})
    assert stop.value.method == "execute_action" and stop.value.budget <= 50
    assert stop.value.action["action"] == "follow_joint_trajectory"
    assert stop.value.action["arguments"]["recovery"]["targets"] == {"left": "retrace"}


def test_without_joint_trajectories_the_retrace_is_unavailable(current):
    config = {**CONFIG, "motion": {**CONFIG["motion"], "joint_trajectory": False}}
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(_state(_path(HOME, START)), [], config=config), {"arms": ["left"]})
    assert error.value.code == "tool_unavailable"


def test_the_released_targets_route_runs_unchanged(current):
    """v1-v3 call recover_pose with targets: the environment's recover_motion, 20 native actions."""
    executed = []
    config = {"synchronous_execution": True, "execution": {"generation": 4}, "motion": {"mutation_refusals": {}}}
    result = tool.run(_api(_state(), executed, config=config), {"targets": {"left": "home"}})
    action, budget = executed[0]
    assert action["action"] == "recover_motion" and budget == 20
    assert action["arguments"] == {"targets": {"left": "home"}, "recovery_targets": {
        "left": {"strategy": "home", "arm_joint_qpos": HOME, "pose_xyz_xyzw": _pose(HOME)}}}
    assert result["compiled_pose"]["canonical_action"] == "recover_motion"
    # Even where joint trajectories run, targets keep the released route.
    executed.clear()
    tool.run(_api(_state(), executed), {"targets": {"left": "previous"}})
    assert executed[0][0]["action"] == "recover_motion"


def test_a_completed_and_settled_retrace_is_arrived_from_its_measured_evidence(current):
    """I-0005: the receipt the tool gets has no after-state; the trajectory evidence says it arrived."""
    begin = [0.25, 0.0, 0.2, 0.0, 0.0, 0.0]
    result = tool.run(_api(_state(_path(begin, START)), []), {"arms": ["left"]})
    progress = result["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["outcome"] == "arrived" and progress["joint_distance_end_rad"] == pytest.approx(0.00015)
    assert progress["undo"]["complete"] is True
    # Missing evidence is unknown, never "still moving".
    unknown = tool.run(_api(_state(_path(begin, START)), [], evidence={"left": None},
                            override={"settle": None}), {"arms": ["left"]})
    progress = unknown["receipt"]["arms"]["left"]["recovery_progress"]
    assert progress["outcome"] == "unknown" and progress["joint_distance_end_rad"] is None
    stalled = tool.run(_api(_state(_path(begin, START)), [], override={
        "status": "stopped", "stop": {"reason": "tracking"}}), {"arms": ["left"]})
    assert stalled["receipt"]["arms"]["left"]["recovery_progress"]["outcome"] == "stalled"


@pytest.mark.parametrize("name", ["recover_pose", "return_home"])
def test_inside_a_code_mode_script_a_recovery_plans_against_what_the_script_has_left(monkeypatch, name):
    """The recovery trajectory fits min(its budget, the episode's and the script's remaining native actions)."""
    seen = []

    def plan(api, **kwargs):
        seen.append(kwargs["native_action_budget"])
        return "planned"

    monkeypatch.setattr(_trajectory.recovery, "plan", plan)
    config = {**CONFIG, "episode_native_actions_remaining": 400, "script_native_actions_remaining": 17}
    assert _trajectory._plan(None, config, name, {}, {}, {}, {}) == "planned"
    assert _trajectory._plan(None, {**CONFIG, "episode_native_actions_remaining": 400}, name, {}, {}, {}, {}) \
        == "planned"
    assert seen == [17, 250]
