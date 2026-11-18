"""What is left of a script's allowance (``ora/agent/pi/code_mode.py``, ``library: extended``).

Every ``bash`` result of an extended contract states the world-changing calls, other calls and
queries the script had left (``allowance_left``); a base contract's result and the sandbox call are exactly as before.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from ora.agent.code_mode.contract import resolve_contract
from ora.agent.code_mode.shell import CommandOutcome
from ora.agent.pi import code_mode as session_module
from ora.agent.pi.code_mode import PiCodeMode


class Executor:
    def native_actions_per_action(self) -> int:
        return 1

    @contextlib.contextmanager
    def invocation(self, name, context, requested):
        yield SimpleNamespace(reentrant=False)


class Session(PiCodeMode):
    """A script's three world-changing calls and five other calls, without the robot socket."""

    def __init__(self, contract, root) -> None:
        self.code_mode, self.run_root, self.robot, self._serving = contract, root, None, 0
        self.tool_action_executor = Executor()
        self.tools = SimpleNamespace(control_root=root)  # the observation mount follows its control pointer

    def _code_mode_call(self, name, arguments, *, model_call, parent, state):
        state["actions" if name == "act" else "calls"] += 1
        return {"ok": True}


def _sandbox(monkeypatch, seen: dict) -> None:
    def run_command(command, *, handle_call, **options):
        seen.update(options)
        for name in ["act"] * 3 + ["look"] * 5:
            handle_call(name, {})
        return CommandOutcome(status="completed", exit_code=0)

    monkeypatch.setattr(session_module, "run_command", run_command)


@pytest.mark.parametrize("library", [None, "extended"])
def test_a_script_result_states_its_allowance_left_only_when_extended(tmp_path, monkeypatch, library) -> None:
    contract = {**resolve_contract({"max_calls_per_script": 20}), **({"library": library} if library else {})}
    seen: dict = {}
    _sandbox(monkeypatch, seen)
    session = Session(contract, tmp_path)
    result = session._code_mode_bash({"command": "python step.py"}, {"model_call_id": "model-1", "tool_call_id": "call-1"})
    assert result["status"] == "completed" and result["actions_executed"] == 3 and result["action_limit"] == 12
    assert seen["extended"] is (library == "extended")
    if library:
        assert result["allowance_left"] == {"actions": 9, "calls": 15, "queries": {"count": 32, "seconds": 180.0}}
    else:
        assert "allowance_left" not in result


def test_the_allowance_follows_the_cap_of_half_the_decisions_left(tmp_path, monkeypatch) -> None:
    seen: dict = {}
    _sandbox(monkeypatch, seen)
    session = Session({**resolve_contract(True), "library": "extended"}, tmp_path)
    session._code_mode_new_act({"robot_decisions": 8})  # at most 4 actions this script
    result = session._code_mode_bash({"command": "python step.py"}, {"model_call_id": "m", "tool_call_id": "c"})
    assert result["action_limit"] == 4 and result["allowance_left"] == {
        "actions": 1, "calls": 19, "queries": {"count": 32, "seconds": 180.0}}
