"""On a real cell a stop at the cell ends a script's later motions in that turn (robot_socket.py ``stop_scripts_on``)."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from ora.agent.pi.robot_socket import PiRobotSocket, ScriptCall


class _Socket(PiRobotSocket):
    """The part of a planner one script call touches; each action's receipt is the next of ``receipts``."""

    def __init__(self, play: dict[str, Any] | None, receipts: list[dict[str, Any]]) -> None:
        self.play, self.receipts, self.invoked, self.records = play, list(receipts), [], []
        self.robot = types.SimpleNamespace(script_calls=0, actions_this_turn=0, script_actions=0,
                                           stopped_this_turn=None)
        self._script_names = {"move_pose": "move_pose"}
        self.submitted_action, self.code_mode, self.run_root = None, None, Path("/run")
        self.last_fresh_observation_event_id = "obs-1"
        self.policy_session_id, self.last_action_step, self.last_action_event_id = "session", None, None
        self.gateway = types.SimpleNamespace(arguments=lambda name, args: args,
                                             classify_effect=lambda name, args: "mutation")
        self._script_executor = types.SimpleNamespace(admit=lambda: None)
        self.tools = types.SimpleNamespace(invoke_with_context=self._invoke)
        self.artifacts = types.SimpleNamespace(put_json=lambda value, kind: {"path": "a.json", "sha256": "0"})

    def _invoke(self, name: str, arguments: dict[str, Any], *, call_context: dict[str, Any]) -> dict[str, Any]:
        self.invoked.append(name)
        return {"ok": True, "status": "submitted", "directive": "action_submitted"}

    def _robot_step(self) -> int:
        return 1

    def _script_record(self, event_type: str, fields: Any) -> dict[str, Any]:
        self.records.append(event_type)
        return {"event_id": f"event-{len(self.records)}"}

    def _script_action(self, name: str, call_id: str, step: Any, origin: Any, *, model_call_id: Any = None):
        return {"receipt": self.receipts.pop(0), "observation": {}}


def _call() -> ScriptCall:
    return ScriptCall(connection=0, request_id=1, name="move_pose", arguments={"targets": {}}, origin={})


REAL_CELL = {"stop_scripts_on": ["guard_stop", "operator_declined"]}


def test_after_a_stopped_motion_the_scripts_later_motions_are_refused_until_the_turn_ends():
    socket = _Socket(REAL_CELL, [{"execution_status": "reached", "failure_code": None},
                                 {"execution_status": "not_reached", "failure_code": "guard_stop"}])
    assert socket._script_call(_call())["ok"] is True
    assert socket._script_call(_call())["receipt"]["failure_code"] == "guard_stop"
    refused = socket._script_call(_call())
    assert refused["ok"] is False and refused["error"]["code"] == "script_stopped_by_operator"
    assert socket.invoked == ["move_pose", "move_pose"], "the refused call never reached the tool"
    assert socket.robot.stopped_this_turn == "guard_stop"


def test_a_declined_motion_stops_the_script_too():
    socket = _Socket(REAL_CELL, [{"execution_status": "plan_rejected", "failure_code": "operator_declined"}])
    socket._script_call(_call())
    assert socket._script_call(_call())["error"]["code"] == "script_stopped_by_operator"


def test_off_a_real_cell_a_script_keeps_going_after_a_guard_stop():
    for play in (None, {"workspace": "/w"}):
        socket = _Socket(play, [{"failure_code": "guard_stop"}, {"failure_code": None}])
        socket._script_call(_call())
        assert socket._script_call(_call())["ok"] is True
        assert socket.robot.stopped_this_turn is None


def test_the_plan_names_the_stop_codes_on_a_real_cell_only(tmp_path, monkeypatch):
    from ora.agent.pi import factory

    monkeypatch.setattr(factory, "user_credential_store", lambda: tmp_path / "store" / "auth.json")
    for mode, expected in (("on", ["guard_stop", "operator_declined"]), ("read", ["guard_stop", "operator_declined"]),
                           ("sim", None)):
        init = factory._play_init({"episode_mode": "play", "play": {}, "runtime": {"rig_mode": mode}}, tmp_path)
        assert init.get("stop_scripts_on") == expected
    assert "stop_scripts_on" not in factory._play_init({"episode_mode": "play", "play": {}}, tmp_path)
