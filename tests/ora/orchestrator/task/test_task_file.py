"""Task files: their schema, defaults and overrides."""
from pathlib import Path

import pytest

from ora.orchestrator.task.task_file import TaskFileError, apply_overrides, load_task, validate_task

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/tasks/robodojo-toolkit-v6.yaml"
MINIMAL = {"schema": "ora.task.v1", "simulator": "robodojo", "experiment": "x", "task": "stack_bowls",
           "model": "m"}


def test_the_shipped_example_is_a_task():
    task = load_task(EXAMPLE)
    assert (task["simulator"], task["seed"], task["policy"]["tool_surface"]) == (
        "robodojo", 0, "robodojo_placement_toolkit_v6")


def test_defaults_fill_in_and_keys_keep_the_schema_order():
    task = validate_task(MINIMAL)
    assert list(task)[:3] == ["schema", "simulator", "experiment"]
    assert (task["seed"], task["layout"], task["policy"], task["budgets"]) == (0, 0, {}, {})


@pytest.mark.parametrize("bad", [
    {"runtime": {}}, {"seed": True}, {"layout": -1}, {"policy": {"tool_allowlist": []}},
    {"budgets": {"max_robot_decisions": 0}}, {"schema": "ora.task.v0"},
])
def test_unknown_keys_and_wrong_types_are_refused(bad):
    with pytest.raises(TaskFileError):
        validate_task({**MINIMAL, **bad})


def test_overrides_change_schema_keys_only():
    task = apply_overrides(validate_task(MINIMAL), cell="align_blocks", layout=3, sets=[
        "policy.context_scheme=context_keep_written", "budgets.max_robot_decisions=40",
        'policy.toolkit_grants={"tools": ["check_task"], "state_keys": ["privileged.task_check"]}',
        "record_toolkit_io=true"])
    assert (task["task"], task["layout"], task["budgets"]["max_robot_decisions"]) == ("align_blocks", 3, 40)
    assert task["policy"]["toolkit_grants"]["tools"] == ["check_task"] and task["record_toolkit_io"] is True
    for assignment in ("runtime.gpu=3", "policy.provider_profile=x", "schema=ora.task.v2", "seed"):
        with pytest.raises(TaskFileError):
            apply_overrides(validate_task(MINIMAL), sets=[assignment])
    with pytest.raises(TaskFileError):
        apply_overrides(validate_task(MINIMAL), sets=["seed=one"])


def test_a_task_may_set_the_model_request():
    task = apply_overrides(validate_task(MINIMAL), sets=["policy.reasoning_effort=medium",
                                                         "policy.max_output_tokens=32768"])
    assert (task["policy"]["reasoning_effort"], task["policy"]["max_output_tokens"]) == ("medium", 32768)
    for bad in ({"reasoning_effort": ""}, {"max_output_tokens": 0}, {"max_output_tokens": True},
                {"max_output_tokens": "32768"}):
        with pytest.raises(TaskFileError):
            validate_task({**MINIMAL, "policy": bad})


def test_save_sandbox_defaults_on_and_is_a_boolean_override():
    assert validate_task(MINIMAL)["save_sandbox"] is True
    assert apply_overrides(validate_task(MINIMAL), sets=["save_sandbox=false"])["save_sandbox"] is False
    with pytest.raises(TaskFileError):
        apply_overrides(validate_task(MINIMAL), sets=["save_sandbox=sometimes"])
