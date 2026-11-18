"""dry_run on recover_pose (arms) and return_home: the recovery plan the real call runs, returned instead of executed.

The same toy kinematics and state serve the real call and its dry run (``test_tool.py``), so the dry run is held to the
real call's own action: the end pose it predicts is the last pose the action expects, the route and method are the
receipt's. A dry run executes nothing; a plan the real call would refuse is its answer; the released anchor form
(``targets``), which the environment executes, has no dry run.
"""
import json
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.interface.types import NotRecorded, Response
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.recover_pose import _trajectory, tool as recover_pose
from ora.agent.toolkit.tools.return_home import tool as return_home
from tests.support.agent.paced_kinematics import paced

HOME = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
RIGHT = [0.3, -0.4, 0.2, 0.0, 0.0, 0.0]
CONFIG = {"synchronous_execution": True, "execution": {"generation": 9},
          "motion": {"joint_trajectory": True, "mutation_refusals": {},
                     "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 0.6]}}}


def _pose(joints):
    return [*joints[:3], 0.0, 0.0, 0.0, 1.0]


def _response(data):
    return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())


def _api(state, executed, *, screens=(), config=None):
    """Toy kinematics (joints 0-2 are the link6 position); the state holds the undo stacks and anchors."""
    screens = list(screens)

    def kinematics(*, request):
        if request["operation"] == "screen_path":
            return _response(screens.pop(0) if screens else {"status": "clear"})
        if request["operation"] == "fk":
            return _response({"rows": [{"index": i, "pose_xyz_xyzw": _pose(q)} for i, q in enumerate(request["joints_rad"])]})
        if request["operation"] == "pace_path":
            return _response(paced(request))
        return _response({"rows": [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0.0, 0.0, 0.0]}
                                   for i, pose in enumerate(request["poses_xyz_xyzw"])]})

    def execute_action(action, budget):
        executed.append((action, budget))
        arms = action["arguments"]["arms"]
        count = len(next(iter(arms.values()))["knots_rad"])
        trajectory = {"status": "completed", "knots_planned": count, "knots_executed": count, "stop": None,
                      "settle": {"settled": True, "joint_error_rad": 0.00015, "joint_speed_rad_s": 0.03},
                      "arms": {arm: {"final_joint_error_per_joint_rad": [0.00015, -0.0001, 0.0, 0.0, 0.0, 0.0]} for arm in arms}}
        receipt = {"status": "reached", "native_actions": count, "stage_evidence": {"trajectory": trajectory},
                   "arms": {arm: {"targeted": True, "motion_progress": {"outcome": "arrived"}} for arm in arms}}
        return _response({"remaining_budget": 50 - budget, "receipt": receipt})

    def read(key):
        if key not in state:
            raise NotRecorded(key)
        return state[key]

    return SimpleNamespace(artifacts=SimpleNamespace(tool_context=lambda: config or CONFIG),
                           planner=SimpleNamespace(kinematics=kinematics),
                           act=SimpleNamespace(execute_action=execute_action), state=SimpleNamespace(read=read))


@pytest.fixture
def current(monkeypatch):
    joints = {"left": list(START), "right": list(RIGHT)}

    def observe(api):
        effectors = {arm: {"pose_xyz_xyzw": _pose(q), "arm_joint_qpos": list(q)} for arm, q in joints.items()}
        return SimpleNamespace(manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": effectors}}),
                               identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))

    monkeypatch.setattr(recover_pose, "observe", observe)
    monkeypatch.setattr(_trajectory, "observe", observe)
    return joints


def _path(begin, end, ticks=6):
    return [[a + (b - a) * k / ticks for a, b in zip(begin, end)] for k in range(ticks + 1)]


def _state(*moves):
    """Undo stacks (oldest first) as the environment writes them, plus the anchors the released route reads."""
    state = {"exec.recovery.left.undo": {"moves": [
        {"tool_call_id": f"call-{index + 1}", "tool": "move_pose", "ordinal": index + 1, "arm_joint_qpos": path[0],
         "pose_xyz_xyzw": _pose(path[0]), "measured_path_rad": path} for index, path in enumerate(moves)]},
        "exec.recovery.right.undo": {"moves": []}}
    for arm, joints in (("left", HOME), ("right", RIGHT)):
        for name in ("home", "previous"):
            state[f"exec.recovery.{arm}.{name}"] = {"arm_joint_qpos": joints, "pose_xyz_xyzw": _pose(joints)}
    return state


def _dry(function, *arguments):
    with dry_run.dry():
        return function(*arguments)


def test_a_retrace_dry_run_names_the_call_it_would_undo_and_executes_nothing(current):
    state = _state(_path(HOME, START))
    real, plain = [], []
    result = recover_pose.run(_api(state, real), {"arms": ["left"]})
    record = _dry(recover_pose.run, _api(state, plain), {"arms": ["left"]})
    assert plain == [] and len(real) == 1
    action, _ = real[0]
    assert record["dry_run"] is True and record["feasible"] is True and record["action"] == "recover_pose"
    assert record["route"] == "measured_retrace"
    left = record["arms"]["left"]
    assert left["method"] == "measured_retrace"
    assert left["predicted_pose"] == pytest.approx(action["arguments"]["arms"]["left"]["expected_link6_xyz_xyzw"][-1], abs=1e-5)
    assert left["end_joints_deg"] == pytest.approx([v * 57.29577951308232 for v in HOME], abs=1e-3)
    assert left["undo"]["undone_call"] == {"tool_call_id": "call-1", "tool": "move_pose", "ordinal": 1}
    assert left["undo"]["complete"] is True and left["undo"]["undoable_calls"] == []
    assert record["motion"]["native_actions"] == len(action["arguments"]["arms"]["left"]["knots_rad"])
    assert result["receipt"]["arms"]["left"]["recovery_progress"]["undo"]["undone_call"] == left["undo"]["undone_call"]
    assert json.loads(json.dumps(record, allow_nan=False)) == record and "_rad" not in json.dumps(record)


def test_with_nothing_to_undo_the_dry_run_says_so_as_an_answer(current):
    executed = []
    record = _dry(recover_pose.run, _api(_state(), executed), {"arms": ["left"]})
    assert (record["feasible"], record["failure_code"], record["failure_stage"]) == (False, "retrace_nothing_to_undo", "plan")
    assert record["refusal"]["constraints"]["use_instead"] == ["return_home", "move_pose"] and executed == []
    with pytest.raises(ToolRefusal) as error:
        recover_pose.run(_api(_state(), executed), {"arms": ["left"]})
    assert error.value.constraints["failure_code"] == "retrace_nothing_to_undo"


def test_the_anchor_form_stops_at_the_action_for_the_environment_that_plans_it(current):
    executed = []
    api = SimpleNamespace(**{**vars(_api(_state(), executed)), "act": dry_run.NoActions()})
    with pytest.raises(dry_run.Proposed) as stopped:
        _dry(recover_pose.run, api, {"targets": {"left": "home"}})
    assert stopped.value.action["action"] == "recover_motion" and executed == []


def test_a_surface_whose_real_call_is_refused_has_no_dry_run_either(current):
    executed = []
    config = {"synchronous_execution": True, "execution": {"generation": 9}, "motion": {"mutation_refusals": {}}}
    for run, arguments in ((recover_pose.run, {"arms": ["left"]}), (return_home.run, {"arms": ["left"]})):
        with pytest.raises(ToolRefusal) as error:
            _dry(run, _api(_state(_path(HOME, START)), executed, config=config), arguments)
        assert error.value.code == dry_run.UNAVAILABLE
    assert executed == []


def test_a_return_home_dry_run_is_the_lift_and_unwind_the_real_call_runs(current):
    state = _state()
    real, plain = [], []
    result = return_home.run(_api(state, real), {"arms": ["left"]})
    record = _dry(return_home.run, _api(state, plain), {"arms": ["left"]})
    assert plain == [] and record["action"] == "return_home" and record["feasible"] is True
    action, _ = real[0]
    left = record["arms"]["left"]
    assert record["route"] == "retract_then_unwind" and left["method"] == "joint_interpolation"
    assert left["lift"] == "planned" and left["unwind_order"] == "straight_joint_line"
    assert left["predicted_pose"] == pytest.approx(action["arguments"]["arms"]["left"]["expected_link6_xyz_xyzw"][-1], abs=1e-5)
    assert left["end_joints_deg"] == pytest.approx([v * 57.29577951308232 for v in HOME], abs=1e-3)
    assert result["receipt"]["arms"]["left"]["recovery_progress"]["unwind_order"] == left["unwind_order"]
    assert record["screening"]["status"] == "clear" and record["screening"]["candidates_screened"] == 1
    assert "undo" not in left


def test_a_return_home_every_path_of_which_is_blocked_is_the_answer_not_an_error(current):
    executed = []
    api = _api(_state(), executed, screens=[{"status": "table_collision", "arm": "left"}] * 6)
    record = _dry(return_home.run, api, {"arms": ["left"]})
    assert (record["feasible"], record["failure_code"], record["failure_stage"]) == (
        False, "recovery_table_collision_path", "screening")
    assert record["refusal"]["code"] == "plan_rejected" and executed == []
    missing = _dry(return_home.run, _api({}, executed), {"arms": ["left"]})
    assert missing["failure_code"] == "recovery_state_unavailable" and missing["feasible"] is False
    with pytest.raises(ToolRefusal) as error:
        _dry(return_home.run, _api(_state(), executed), {"arms": ["up"]})
    assert error.value.code == "invalid_motion_arguments", "a malformed request is a refusal"
