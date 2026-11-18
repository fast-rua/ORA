"""A mutation call that fails after submission leaves no action behind, and is logged.

Census item 10 / E6 (359a18b9): the manipulation tool runtime validated a result against
its schema outside the error path, so a failure there -- or any raise after the action
was persisted -- left ``submitted-action.json`` persisted (closing the observation for
the rest of the turn) and wrote no ``tool_error`` event. The RoboDojo stand-in host
(``tests/support/agent/stack_parity.py``) serves the tools from the toolkit, proposing
each action; what it answers is checked against its digest in
``tests/data/toolkit-goldens``, recorded while the handlers answered it alike.

Undoing the failed call also reopens the observation pointer it closed, on the stand-in's
real control root, so a corrected call can submit; a pointer that changed otherwise
during the call is left alone.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.error_classes import ToolError
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_recording import proposal

GOLDEN = Goldens(__file__)
#: A set_gripper result schema no result satisfies.
UNSATISFIABLE = {"type": "object", "properties": {"never": {"type": "string"}}, "required": ["never"]}


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


def _events(runtime):
    return [json.loads(line) for line in runtime.event_path.read_text().splitlines()]


def _runtime(root, monkeypatch):
    """The RoboDojo stand-in host, its tools served by the toolkit, proposing each action."""
    from tests.support.agent.stack_parity import on_toolkit, stand_in

    return on_toolkit(stand_in(root, "robodojo", monkeypatch))


def _call(runtime, name, arguments, index):
    return runtime.invoke_with_context(name, arguments, call_context=proposal(runtime, name, arguments, index))


def test_a_result_schema_failure_after_submission_unsubmits_and_is_logged(golden, tmp_path, monkeypatch):
    from dataclasses import replace

    runtime = _runtime(tmp_path / "run", monkeypatch)
    released = runtime.surface
    runtime.surface = replace(released, result_schemas={**released.result_schemas, "set_gripper": UNSATISFIABLE})
    action = runtime.control_root / "submitted-action.json"
    with pytest.raises(Exception):
        _call(runtime, "set_gripper", {"targets": {"left": "close"}}, 0)
    assert not action.exists()
    logged = _events(runtime)[-1]
    assert (logged["type"], logged["tool"]) == ("tool_error", "set_gripper")
    golden("logged_error", logged["error"], tmp_path / "run")


def test_a_corrected_call_after_the_failure_submits(tmp_path, monkeypatch):
    from dataclasses import replace

    runtime = _runtime(tmp_path / "run", monkeypatch)
    pointer = runtime.control_root / "current-observation.json"
    before = pointer.read_bytes()
    assert json.loads(before)["open"] is True
    released = runtime.surface
    runtime.surface = replace(released, result_schemas={**released.result_schemas, "set_gripper": UNSATISFIABLE})
    with pytest.raises(Exception):
        _call(runtime, "set_gripper", {"targets": {"left": "close"}}, 0)
    assert pointer.read_bytes() == before  # the control root is as the call found it
    runtime.surface = released
    _call(runtime, "set_gripper", {"targets": {"left": "close"}}, 1)
    assert (runtime.control_root / "submitted-action.json").exists()
    assert json.loads(pointer.read_bytes())["open"] is False


def _proposing(tmp_path, *, pending=False):
    """A control root with an open pointer, and a host whose next call is a mutation."""
    from types import SimpleNamespace

    control = tmp_path / "control"
    control.mkdir()
    pointer = {"open": not pending, "observation_id": "observation-1", "generation": 1, "observation_sha256": "a" * 64}
    (control / "current-observation.json").write_text(json.dumps(pointer) + "\n", encoding="utf-8")
    if pending:
        (control / "submitted-action.json").write_text('{"action_id": "action-0"}\n', encoding="utf-8")
    host = SimpleNamespace(control_root=control, _submitted_mutations=[],
                           classify_call_effect=lambda name, arguments: "mutation")
    return host, control / "current-observation.json", control / "submitted-action.json"


def _persist(host, pointer, action, **changed):
    """What ``_persist_action`` leaves: the action, and the pointer closed (plus any ``changed`` field)."""
    action.write_text('{"action_id": "action-1"}\n', encoding="utf-8")
    pointer.write_text(json.dumps({**json.loads(pointer.read_bytes()), "open": False, **changed}), encoding="utf-8")
    host._submitted_mutations.append("set_gripper")


def test_restore_reopens_only_the_pointer_the_call_closed(tmp_path):
    from ora.agent.toolkit.host.runtime import mutation_compensation

    host, pointer, action = _proposing(tmp_path)
    before = pointer.read_bytes()
    taken = mutation_compensation.snapshot(host, "set_gripper", {})
    _persist(host, pointer, action)
    mutation_compensation.restore(host, taken)
    assert not action.exists() and pointer.read_bytes() == before and host._submitted_mutations == []


def test_restore_leaves_a_pointer_replaced_during_the_call(tmp_path):
    from ora.agent.toolkit.host.runtime import mutation_compensation

    host, pointer, action = _proposing(tmp_path)
    taken = mutation_compensation.snapshot(host, "set_gripper", {})
    _persist(host, pointer, action, observation_id="observation-2", generation=2)  # a fresh publication's pointer
    replaced = pointer.read_bytes()
    mutation_compensation.restore(host, taken)
    assert pointer.read_bytes() == replaced


def test_restore_keeps_a_pending_action_and_its_closed_pointer(tmp_path):
    from ora.agent.toolkit.host.runtime import mutation_compensation

    host, pointer, action = _proposing(tmp_path, pending=True)
    before, pending = pointer.read_bytes(), action.read_bytes()
    taken = mutation_compensation.snapshot(host, "set_gripper", {})
    mutation_compensation.restore(host, taken)
    assert pointer.read_bytes() == before and action.read_bytes() == pending


def test_a_refused_second_mutation_keeps_the_pending_action(golden, tmp_path, monkeypatch):
    runtime = _runtime(tmp_path / "run", monkeypatch)
    _call(runtime, "set_gripper", {"targets": {"left": "close"}}, 0)
    action = runtime.control_root / "submitted-action.json"
    pending = action.read_bytes()
    with pytest.raises(ToolError) as refused:
        _call(runtime, "set_gripper", {"targets": {"right": "close"}}, 1)
    assert action.read_bytes() == pending
    golden("refusal", refused.value.as_dict(), tmp_path / "run")
    golden("pending_action", json.loads(pending), tmp_path / "run")
