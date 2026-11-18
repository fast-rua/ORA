"""recover_pose's undo stack on the fake plant, end to end: tool, host check, executor, restore.

Every non-recovery trajectory pushes an entry for its tool call on the arm's undo stack
(``exec.recovery.<arm>.undo``, the last 5 calls), or extends the entry when the same call already moved the
arm; recover_pose retraces the top entry and the executor pops it when every knot ran; a cut retrace stays on
top with its progress and continues. The toolkit tool runs against the fake plant's own kinematics query (FK
and screens; joints 0-2 are the link6 position) and executor; restore rebuilds the stacks from the recorded
receipts and traces exactly as the live session left them.
"""
from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface.types import NotRecorded, Response
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.recover_pose import _trajectory, tool
from ora.contracts.execution_state import FIELDS, UNDO_DEPTH, pack
from ora.environment.backends.robodojo import kinematics_query
from ora.environment.backends.robodojo.toolkit_restore import bind_bookkeeping
from tests.support.robodojo.backend import v3

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")


def _response(data, method):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}, default=float).encode())


def _values(sparse):
    return json.loads(json.dumps(pack({name: getattr(sparse, name) for name in FIELDS}), default=float))


def _api(sparse, tmp_path, monkeypatch, remaining=None):
    def observe(api):
        observation, _ = sparse.observe(tmp_path / "observations")
        effectors = sparse._proprioception()["end_effectors"]
        manifest = {"proprioception": {"end_effectors": json.loads(json.dumps(effectors, default=float))}}
        return SimpleNamespace(manifest=SimpleNamespace(json=lambda: manifest), identity=SimpleNamespace(
            json=lambda: {"observation_id": observation["observation_id"], "observation_sha256": "a" * 64}))

    def execute_action(action, budget):
        sparse.request_action_budget = budget
        try:
            receipt = sparse.execute(action)["receipt"]
        finally:
            del sparse.request_action_budget
        return _response({"remaining_budget": budget - receipt["native_actions"],
                          "receipt": {**receipt, "stage_evidence": {"trajectory": receipt.get("trajectory")}}},
                         "execute_action")

    def read(key):
        values = _values(sparse)
        if key not in values:
            raise NotRecorded(key)
        return values[key]

    monkeypatch.setattr(_trajectory, "observe", observe)
    config = {"synchronous_execution": True, "execution": {"generation": 1},
              "episode_native_actions_remaining": remaining,
              "motion": {"joint_trajectory": True, "mutation_refusals": {},
                         "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}}}
    return SimpleNamespace(
        artifacts=SimpleNamespace(tool_context=lambda: config),
        planner=SimpleNamespace(kinematics=lambda *, request: _response(
            kinematics_query.answer(sparse, request), "kinematics")),
        act=SimpleNamespace(execute_action=execute_action), state=SimpleNamespace(read=read))


def _joints(sparse):
    return [float(v) for v in sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"]]


def _move(sparse, tmp_path, joint, step=0.02, count=5, call=None):
    """One arm-moving tool call (``call``: its id; default a new one): a trajectory of ``count`` knots."""
    _move.calls = getattr(_move, "calls", 0) + 1
    sparse.tool_call = {"tool_call_id": call or f"call-{_move.calls}", "tool": "move_pose", "ordinal": _move.calls}
    start = _joints(sparse)
    knots, current = [], list(start)
    for _ in range(count):
        current = list(current)
        current[joint] += step
        knots.append(current)
    action = v3(sparse, tmp_path, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": knots}}}
    try:
        receipt = sparse.execute(action)["receipt"]
    finally:
        del sparse.tool_call
    assert receipt["arms"]["left"]["motion_progress"]["undo_depth"] == len(_stack(sparse))
    return start


def _stack(sparse):
    return sparse.undo_stacks["left"]


def test_three_calls_undo_three_moves_one_by_one(sparse, tmp_path, monkeypatch):
    starts = [_move(sparse, tmp_path, joint) for joint in (0, 1, 3)]
    assert len(_stack(sparse)) == 3
    for depth in (3, 2, 1):
        result = tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
        undo = result["receipt"]["arms"]["left"]["recovery_progress"]["undo"]
        assert undo["complete"] and len(undo["undoable_calls"]) == depth - 1
        assert result["receipt"]["arms"]["left"]["motion_progress"]["undo_depth"] == depth - 1
        assert len(_stack(sparse)) == depth - 1
        assert _joints(sparse) == pytest.approx(starts[depth - 1], abs=0.01)  # back where that move started
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
    assert error.value.constraints["failure_code"] == "retrace_nothing_to_undo"
    # Undoing never moved the previous anchor: it is still the last move's start.
    assert sparse.last_safe_joints["left"] == pytest.approx(starts[-1])


def test_a_cut_retrace_stays_on_top_and_continues(sparse, tmp_path, monkeypatch):
    start = _move(sparse, tmp_path, 0, step=0.03, count=12)
    tool.run(_api(sparse, tmp_path, monkeypatch, remaining=10), {"arms": ["left"]})
    assert len(_stack(sparse)) == 1 and sparse.last_receipt["receipt"]["trajectory"]["undo"]["arms"] == {
        "left": "cut"}
    progress = _stack(sparse)[-1]["undo_progress"]
    assert progress["retrace_calls"] == 1 and 0 < progress["fraction_undone"] < 1
    assert _joints(sparse)[0] > start[0] + 0.05  # part of the way back
    result = tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
    assert result["receipt"]["arms"]["left"]["recovery_progress"]["undo"]["complete"]
    assert _stack(sparse) == [] and _joints(sparse) == pytest.approx(start, abs=0.01)


def test_one_tool_call_is_one_undo_entry_and_undone_in_one_call(sparse, tmp_path, monkeypatch):
    first = _move(sparse, tmp_path, 0)
    start = _move(sparse, tmp_path, 1, call="grasp")  # e.g. grasp_execute's pregrasp, approach and lift
    _move(sparse, tmp_path, 3, call="grasp")
    _move(sparse, tmp_path, 0, step=0.01, call="grasp")
    assert len(_stack(sparse)) == 2 and _stack(sparse)[-1]["tool_call_id"] == "grasp"
    assert _stack(sparse)[-1]["arm_joint_qpos"] == pytest.approx(start)
    result = tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
    undo = result["receipt"]["arms"]["left"]["recovery_progress"]["undo"]
    assert undo["complete"] and undo["undone_call"]["tool_call_id"] == "grasp"
    assert _joints(sparse) == pytest.approx(start, abs=0.01) and len(_stack(sparse)) == 1
    assert _stack(sparse)[-1]["arm_joint_qpos"] == pytest.approx(first)


def test_the_stack_keeps_the_last_five_tool_calls(sparse, tmp_path, monkeypatch):
    starts = [_move(sparse, tmp_path, joint % 4) for joint in range(UNDO_DEPTH + 2)]
    stack = _stack(sparse)
    assert len(stack) == UNDO_DEPTH
    for move, start in zip(stack, starts[2:]):
        assert move["measured_path_rad"][0] == pytest.approx(start)


def test_the_arm_off_its_latest_path_is_refused(sparse, tmp_path, monkeypatch):
    _move(sparse, tmp_path, 0)
    sparse.environment.robot_manager.joints["left_arm"][1] += 0.1  # moved since by another route (a push)
    with pytest.raises(ToolRefusal) as error:
        tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
    assert error.value.constraints["failure_code"] == "retrace_arm_off_path"
    assert "return_home" in str(error.value)


def test_a_stale_stack_top_is_refused_by_the_host_check(sparse, tmp_path):
    from ora.contracts.execution_state import recovery_goals

    _move(sparse, tmp_path, 0)
    top = _stack(sparse)[-1]["measured_path_rad"][0]
    action = v3(sparse, tmp_path, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": [top]}}, "recovery": {
        "targets": {"left": "retrace"}, "retrace": {"left": {"moves": 2, "arm_joint_qpos": top}}}}
    with pytest.raises(ValueError, match="stale_or_mismatched_recovery_target"):
        recovery_goals(action, _values(sparse))
    action["arguments"]["recovery"]["retrace"]["left"]["moves"] = 1
    assert recovery_goals(action, _values(sparse)) == {}


def _restored(sparse, tmp_path, count, name):
    """bind_bookkeeping over the first ``count`` recorded receipts, on a fresh session at the round start."""
    native = tmp_path / f"prefix-{name}"
    for kind in ("receipts", "controller"):
        (native / kind).mkdir(parents=True)
        for index in range(count):
            for suffix in ("json", "jsonl") if kind == "controller" else ("json",):
                shutil.copyfile(sparse.output_root / kind / f"action-{index:04d}.{suffix}",
                                native / kind / f"action-{index:04d}.{suffix}")
    fresh = SimpleNamespace(output_root=tmp_path / f"restored-{name}", last_safe_joints=dict(sparse.home_joints),
                            last_safe_poses=dict(sparse.home_poses), last_safe_paths={}, undo_stacks={},
                            sim_steps=0, environment=SimpleNamespace(robot_manager=SimpleNamespace(
                                control_manager=SimpleNamespace(prev_control=[{}]))))
    for field in FIELDS:
        if not hasattr(fresh, field):
            setattr(fresh, field, getattr(sparse, field))
    bind_bookkeeping(fresh, dict(source_native=str(native), expected=dict(
        native_receipts=count, observation_sequence=0, round_index=0, previous_execution_receipt=None)))
    return pack({field: getattr(fresh, field) for field in FIELDS})


def test_restore_rebuilds_the_stack_exactly_after_every_action(sparse, tmp_path, monkeypatch):
    live = []

    def record(name):
        values = pack({field: getattr(sparse, field) for field in FIELDS})
        live.append((name, {key: value for key, value in values.items()
                            if key.startswith("exec.recovery.")}))

    for joint in range(UNDO_DEPTH + 1):
        _move(sparse, tmp_path, joint % 4, call=f"restore-{joint}")
        record(f"move {joint}")
    _move(sparse, tmp_path, 1, call=f"restore-{UNDO_DEPTH}")  # the same tool call again: extends its entry
    record("extend")
    tool.run(_api(sparse, tmp_path, monkeypatch, remaining=12), {"arms": ["left"]})
    assert "undo_progress" in sparse.undo_stacks["left"][-1]
    record("cut retrace")
    tool.run(_api(sparse, tmp_path, monkeypatch), {"arms": ["left"]})
    record("completed retrace")
    _move(sparse, tmp_path, 1)
    record("move after undo")
    assert len(live[-1][1]["exec.recovery.left.undo"]["moves"]) == UNDO_DEPTH
    for count, (name, expected) in enumerate(live, start=1):
        restored = _restored(sparse, tmp_path, count, count)
        assert {key: restored[key] for key in expected} == expected, name
