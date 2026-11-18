"""A restore worker is configured from the source run alone: its seed, layout, layout file and task.

The restore worker of every toolkit live or native restore (and of motion replay) is built by
``motion.replay._worker_config`` from the source run's plan. A run on seed 1 restored with seed 0
loads another layout set: same layout id and file name, other objects (arrange_largest_number, 10-29).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.orchestrator.backends.robodojo.motion import replay
from ora.orchestrator.backends.robodojo.motion.replay import ReplayLaunchError, run_eval_seed

RUNTIME = {"task_variant": "arrange_largest_number", "eval_seed": 1, "layout_id": 7,
           "layout_file": "Eval_Layout/RoboDojo/arx_x5/1/arrange_largest_number_7.json",
           "layout_sha256": "f" * 64, "task_language_source": "live_reset_readback"}


def plan(**changes):
    value = {"seed": 1, "budgets": {}, "runtime": dict(RUNTIME), "policy": {"task_file": "agent/task.md"}}
    value.update(changes)
    return value


def test_the_worker_starts_with_the_runs_seed_and_scene(tmp_path, monkeypatch):
    built = {}

    def build(**arguments):
        built.update(arguments)
        return SimpleNamespace(worker_config={**arguments["runtime"], "eval_seed": arguments["seed"],
                                              "task": arguments["task"]})

    import ora.environment.factory as factory
    monkeypatch.setattr(factory, "build_robodojo_worker_session", build)
    native = tmp_path / "run/native"
    native.mkdir(parents=True)
    (tmp_path / "pass").mkdir()
    _, path = replay._worker_config(dict(RUNTIME), plan(), tmp_path / "pass", native)
    config = json.loads(Path(path).read_bytes())
    assert built["seed"] == 1 and config["eval_seed"] == 1
    for key in ("task_variant", "layout_id", "layout_file", "layout_sha256"):
        assert config[key] == RUNTIME[key]


@pytest.mark.parametrize("recorded,expected", [
    (dict(seed=1), 1), (dict(seed=0, runtime={**RUNTIME, "eval_seed": 0}), 0),
    (dict(seed=None, runtime={**RUNTIME, "eval_seed": 2}), 2),
    (dict(seed=None, runtime={k: v for k, v in RUNTIME.items() if k != "eval_seed"}), 0),
])
def test_the_runs_evaluation_seed(recorded, expected):
    assert run_eval_seed(plan(**recorded)) == expected


@pytest.mark.parametrize("recorded", [dict(seed=2), dict(seed=True), dict(seed=-1, runtime={**RUNTIME, "eval_seed": -1})])
def test_an_inconsistent_or_invalid_seed_is_refused(recorded):
    with pytest.raises(ReplayLaunchError):
        run_eval_seed(plan(**recorded))

