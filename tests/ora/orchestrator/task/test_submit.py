"""Scheduler requests for one run of a task."""
import json
from pathlib import Path

import pytest

from ora.orchestrator.task.submit import SubmitError, build_request, scheduler_url
from ora.orchestrator.task.task_file import validate_task

TASK = validate_task({"schema": "ora.task.v1", "simulator": "robodojo", "experiment": "x", "task": "stack_bowls",
                      "model": "m", "layout": 2})


def test_the_request_runs_ora_task_run_on_the_resolved_task_in_this_checkout():
    request = build_request(TASK, repo=Path("/repo"), defaults={"owner": "me", "window": "6h", "env": {"A": "1"}},
                            env={"B": "2"}, env_files={"KEY": "/secret"}, priority="P1", cpu=None)
    assert request["type"] == "submit" and request["owner"] == "me" and request["priority"] == "P1"
    assert "code" not in request  # no version: the checkout runs as it is
    assert request["ora"] == {"simulator": "robodojo", "model": "m", "task": "stack_bowls", "seed": 0, "layout": 2,
                              "experiment": "x"}
    assert request["resources"] == {"window": "6h"}
    run = request["run"]
    assert run["argv"][:4] == ["/repo/scripts/ora", "task", "run", "--resolved"]
    assert json.loads(run["argv"][4]) == TASK
    assert (run["cwd"], run["env"], run["env_files"]) == ("/repo", {"A": "1", "B": "2"}, {"KEY": "/secret"})
    with_summary = build_request(TASK, repo=Path("/repo"), defaults={"owner": "me"}, summary=Path("/shared/c.json"))
    assert with_summary["run"]["argv"][5:] == ["--summary", "/shared/c.json"]


def test_a_request_needs_an_owner_and_a_scheduler():
    with pytest.raises(SubmitError):
        build_request(TASK, repo=Path("/repo"), defaults={})
    with pytest.raises(SubmitError):
        scheduler_url({}, None, environ={})
    assert scheduler_url({"scheduler_url": "http://a/"}, None, environ={}) == "http://a"
    assert scheduler_url({}, None, environ={"ORA_SCHEDULER_URL": "http://b"}) == "http://b"
