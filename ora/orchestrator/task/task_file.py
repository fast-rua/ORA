"""``ora.task.v1``: what one task runs, written once and run many times.

A task file names the simulator, the task cell (``task``, ``seed``,
``layout``), the model by name, the experiment namespace, the policy
selection and budgets. It never names where a run executes: the card, the
resource profile, the model replica and its endpoint, the perception
services, the execution host and the admission cohort belong to each run
(``binding``). ``policy.reasoning_effort`` and ``policy.max_output_tokens``
set the model request over the bound Provider profile's values (the task
record keeps both). Overrides change keys the schema defines and nothing else:
``--task``, ``--seed`` and ``--layout``, or ``--set a.b=VALUE`` (VALUE is read
as YAML, so ``3``, ``true`` and ``{"tools": [...]}`` keep their types).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCHEMA = "ora.task.v1"
#: Top-level keys and their types; ``policy`` and ``budgets`` hold the nested keys below.
FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema": str, "simulator": str, "experiment": str, "task": str, "seed": int, "layout": int,
    "model": str, "policy": dict, "record_toolkit_io": bool, "ground_truth_capture": bool,
    "save_sandbox": bool, "budgets": dict,
}
REQUIRED = ("schema", "simulator", "experiment", "task", "model")
POLICY_FIELDS: dict[str, type | tuple[type, ...]] = {
    "baseline": str, "tool_surface": str, "context_scheme": str, "toolkit_grants": dict,
    "toolkit_receipt_delivery": str, "task_file": str,
    # The model request, over the Provider profile's values (ora.orchestrator.task.run).
    "reasoning_effort": str, "max_output_tokens": int,
}
BUDGET_FIELDS = ("max_robot_decisions", "max_model_calls", "wall_time_s", "max_sim_steps")
DEFAULTS = {"seed": 0, "layout": 0, "policy": {}, "record_toolkit_io": False,
            "ground_truth_capture": False, "save_sandbox": True, "budgets": {}}


class TaskFileError(ValueError):
    """A task file or an override does not fit ``ora.task.v1``."""


def validate_task(value: Any) -> dict[str, Any]:
    """The task with its defaults filled in; refuses unknown keys and wrong types."""

    if not isinstance(value, Mapping):
        raise TaskFileError("a task file is a mapping")
    task = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(dict(value))}
    if task.get("schema") != SCHEMA:
        raise TaskFileError(f"a task file has schema {SCHEMA}")
    unknown = sorted(set(task) - set(FIELDS))
    if unknown:
        raise TaskFileError(f"unknown task keys: {unknown}")
    missing = [key for key in REQUIRED if key not in task]
    if missing:
        raise TaskFileError(f"missing task keys: {missing}")
    for key, kind in FIELDS.items():
        # bool is an int: a seed of True is refused.
        if not isinstance(task[key], kind) or (kind is int and isinstance(task[key], bool)):
            raise TaskFileError(f"task.{key} must be {getattr(kind, '__name__', kind)}")
    for key, value in task["policy"].items():
        if (key not in POLICY_FIELDS or not isinstance(value, POLICY_FIELDS[key])
                or isinstance(value, bool) or (isinstance(value, (str, int)) and not value)
                or (isinstance(value, int) and value < 0)):
            raise TaskFileError(f"task.policy.{key} is not a {SCHEMA} policy field of its type")
    for key, value in task["budgets"].items():
        if key not in BUDGET_FIELDS or not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TaskFileError(f"task.budgets.{key} is not a positive budget of {BUDGET_FIELDS}")
    if task["seed"] < 0 or task["layout"] < 0:
        raise TaskFileError("task.seed and task.layout are non-negative")
    return {key: task[key] for key in FIELDS}


def load_task(path: Path) -> dict[str, Any]:
    """Read and validate one task file (YAML or JSON)."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise TaskFileError(f"{path} is not YAML: {error}") from error
    return validate_task(value)


def _set(task: dict[str, Any], assignment: str) -> None:
    path, sep, text = assignment.partition("=")
    keys = path.strip().split(".")
    if not sep or not all(keys):
        raise TaskFileError(f"--set takes KEY=VALUE or SECTION.KEY=VALUE, not {assignment!r}")
    value = yaml.safe_load(text) if text.strip() else ""
    if len(keys) == 1 and keys[0] in FIELDS and keys[0] != "schema":
        task[keys[0]] = value
    elif len(keys) == 2 and keys[0] == "policy" and keys[1] in POLICY_FIELDS:
        task["policy"][keys[1]] = value
    elif len(keys) == 2 and keys[0] == "budgets" and keys[1] in BUDGET_FIELDS:
        task["budgets"][keys[1]] = value
    else:
        raise TaskFileError(f"--set {path}: not a key of {SCHEMA}")


def apply_overrides(task: Mapping[str, Any], *, cell: str | None = None, seed: int | None = None,
                    layout: int | None = None, sets: Sequence[str] = ()) -> dict[str, Any]:
    """The task after ``--task``/``--seed``/``--layout`` and each ``--set``, validated again."""

    result = copy.deepcopy(dict(task))
    for key, value in (("task", cell), ("seed", seed), ("layout", layout)):
        if value is not None:
            result[key] = value
    for assignment in sets:
        _set(result, assignment)
    return validate_task(result)


def overrides_record(*, cell: str | None, seed: int | None, layout: int | None,
                     sets: Sequence[str]) -> dict[str, Any]:
    """The overrides as given, for the run's task record."""

    given = {"task": cell, "seed": seed, "layout": layout}
    return {**{key: value for key, value in given.items() if value is not None}, "set": list(sets)}
