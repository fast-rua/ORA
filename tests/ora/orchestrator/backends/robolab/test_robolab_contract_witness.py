"""The host side of RoboLab's seed-0 contract witness: the worker it starts and the review it fills.

The worker is scripted: nothing starts Isaac Sim. The record it writes stands in for one the real
worker's ``contract_witness.run_witness`` writes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ora.orchestrator.backends.robolab import contract_witness

ROOT = Path(__file__).resolve().parents[5]
TASK = "AnimalsInBinTask"


def _record(*, reset_success: bool = False, hidden: bool = False) -> dict:
    return {"schema": "ora.robolab-task-contract-witness.v2", "task": TASK, "seed": 0, "plan_sha256": "p" * 64,
            "all_cases_match": not reset_success,
            "cases": [{"case": "reset_state", "expected": False, "official_success": reset_success,
                       "matches": not reset_success},
                      {"case": "beside", "expected": False, "official_success": False, "matches": True},
                      {"case": "all_in", "expected": True, "official_success": True, "matches": True}],
            "visibility_at_reset": {"lizard_figurine": {"visible": True}, "grey_bin": {"visible": not hidden}}}


def _result(tmp_path: Path, record: dict) -> dict:
    path = tmp_path / contract_witness.WITNESS_DIRECTORY / f"{TASK}-s0-witness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return {"task": TASK, "path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "record": record}


def _review() -> dict:
    return json.loads((ROOT / contract_witness.review_path(TASK, root=ROOT)).read_text(encoding="utf-8"))


def test_a_matching_witness_fills_the_evidence_and_the_checks_it_settles(tmp_path):
    result = _result(tmp_path, _record())
    review = contract_witness.fill_review(dict(_review(), status="needs_review"), result, evidence_root=tmp_path)
    assert review["evidence"] == [{"role": "private_success_witness_and_counterexamples",
                                   "path": f"{contract_witness.WITNESS_DIRECTORY}/{TASK}-s0-witness.json",
                                   "sha256": result["sha256"]}]
    assert {key: review["checks"][key]["status"] for key in ("success_witness", "counterexamples",
                                                             "public_observability")} == {
        "success_witness": "passed", "counterexamples": "passed", "public_observability": "passed"}
    assert [item["expected"] for item in review["counterexamples"]] == [
        "witness case reset_state", "witness case beside", "source-derived"]
    assert review["status"] == "needs_review", "the witness fills checks; only promote marks a review reviewed"


def test_the_evidence_path_stays_relative_to_the_root_when_its_outputs_are_a_link(tmp_path):
    root, elsewhere = tmp_path / "root", tmp_path / "moved"
    (root / "outputs").mkdir(parents=True)
    (elsewhere / "ora-backends").mkdir(parents=True)
    (root / "outputs" / "ora-backends").symlink_to(elsewhere / "ora-backends")
    review = contract_witness.fill_review(_review(), _result(root, _record()), evidence_root=root)
    assert review["evidence"][0]["path"] == f"{contract_witness.WITNESS_DIRECTORY}/{TASK}-s0-witness.json"


def test_a_hidden_object_leaves_observability_to_a_reviewer_and_a_solved_reset_blocks(tmp_path):
    hidden = contract_witness.fill_review(_review(), _result(tmp_path, _record(hidden=True)), evidence_root=tmp_path)
    assert hidden["checks"]["public_observability"]["status"] == "pending"
    assert "grey_bin" in hidden["checks"]["public_observability"]["basis"]
    solved = contract_witness.fill_review(_review(), _result(tmp_path, _record(reset_success=True)),
                                          evidence_root=tmp_path)
    assert solved["checks"]["counterexamples"]["status"] == "failed"
    assert solved["compatibility"]["status"] == "blocked"
    assert solved["compatibility"]["blocked"][-1]["code"] == "solved_at_reset"


def test_a_worker_that_wrote_nothing_fails_the_witness_checks(tmp_path):
    review = contract_witness.fill_review(_review(), {"task": TASK, "error": "RobolabError: boom"},
                                          evidence_root=tmp_path)
    assert review["checks"]["success_witness"]["status"] == "failed"
    assert review["witness_result"]["blocked"][0]["code"] == "witness_worker_failed"


def test_a_later_witness_replaces_an_earlier_runs_blocker(tmp_path):
    failed = contract_witness.fill_review(_review(), {"task": TASK, "error": "EnvironmentRpcError: start failed"},
                                          evidence_root=tmp_path)
    assert failed["compatibility"]["status"] == "blocked"
    rerun = contract_witness.fill_review(failed, _result(tmp_path, _record()), evidence_root=tmp_path)
    assert rerun["compatibility"]["blocked"] == [] and rerun["compatibility"]["status"] != "blocked"
    assert rerun["checks"]["success_witness"]["status"] == "passed"


class _Worker:
    def __init__(self, job: Path) -> None:
        self.job, self.closed = job, False

    def start(self):
        job = json.loads(self.job.read_text(encoding="utf-8"))
        Path(job["output"]).write_text(json.dumps(_record()), encoding="utf-8")
        return {}

    def close(self):
        self.closed = True


def test_the_runner_starts_one_planner_free_worker_per_task_with_its_plan(tmp_path, monkeypatch):
    built = {}

    def build(**kwargs):
        built.update(kwargs)
        built["worker"] = _Worker(Path(kwargs["runtime"]["contract_witness_job"]))
        return built["worker"]

    import ora.environment.factory as factory

    monkeypatch.setattr(factory, "build_robolab_worker_session", build)
    result = contract_witness.run_task(TASK, gpu=3, evidence_root=tmp_path, root=ROOT)
    assert built["worker"].closed and result["record"]["all_cases_match"] is True
    runtime = built["runtime"]
    assert runtime["renderer_gpu"] == 3 and "motion_planner" not in runtime
    job = json.loads(Path(runtime["contract_witness_job"]).read_text(encoding="utf-8"))
    assert job["schema"] == contract_witness.JOB_SCHEMA and job["plan"] == _review()["witness_plan"]
    assert built["task"] == TASK and built["seed"] == 0


def test_shards_split_the_pending_list_across_cards():
    tasks = [f"T{index}" for index in range(7)]
    shards = [contract_witness._shard(tasks, f"{index}/3") for index in range(3)]
    assert sorted(sum(shards, [])) == sorted(tasks) and shards[0] == ["T0", "T3", "T6"]
