"""Partitions: a practice plan may not use an evaluation-set episode, and reports keep practice apart.

A fresh benchmark plan is partition evaluation. A practice plan is checked
against ora/environment/benchmarks/evaluation-sets-v1.json through
evaluation_set_hit and records the list's sha256; an environment without a
practice side and free play are refused. The run report adds a partition
group with the practice budget (runs, model calls, tokens, cost, hours) and
says practice runs are not evaluation results. Nothing here starts a
simulator or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.environment.benchmarks.evaluation_sets import MANIFEST
from ora.environment.benchmarks.libero_pro import PRACTICE_DRIVER, PRACTICE_SEED_FLOOR
from ora.orchestrator.spec import benchmark as benchmark_spec
from ora.orchestrator.reports import run_report
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.spec.partition import PARTITIONS, resolve_partition
from ora.orchestrator.spec.values import file_sha256
from tests.support.orchestrator import benchmark_specs as bench
from tests.support.orchestrator.run_report_fixture import round_record, write_run

ROOT = Path(__file__).resolve().parents[4]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())


def _robotwin_plan(workspace, monkeypatch, *, seed=100002, **spec_fields):
    report = bench._readiness_report(workspace / "reports", monkeypatch)
    spec, manifest = bench._robotwin(workspace, seed=seed, service_report=report,
                                     cohort_id="partition-fixture-g001")
    spec.update(spec_fields)
    benchmark_spec.compose(spec, seed_manifest=manifest)
    return resolve_run_spec(spec, root=ROOT)


def test_the_default_benchmark_plan_is_the_evaluation_partition(workspace, monkeypatch):
    plan = _robotwin_plan(workspace, monkeypatch)
    assert plan["partition"] == {"name": ACTIVATION["default_partition"]} == {"name": "evaluation"}


def test_a_default_baseline_libero_practice_plan_selects_the_practice_driver(workspace, monkeypatch):
    report = bench._readiness_report(workspace / "reports", monkeypatch)
    spec = bench._libero(service_report=report, cohort_id="partition-fixture-g001")
    spec.update(seed=PRACTICE_SEED_FLOOR + 1, partition="practice")
    spec["runtime"]["environment_driver"] = PRACTICE_DRIVER
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["environment_driver"] == PRACTICE_DRIVER
    assert plan["partition"] == {"name": "practice", "evaluation_sets_sha256": file_sha256(MANIFEST)}
    # The practice driver is the only substitute the baseline admits for its official driver.
    spec["runtime"]["environment_driver"] = "direct_libero_unregistered_v1"
    with pytest.raises((RunSpecError, ValueError), match="conflicts with runtime.environment_driver"):
        resolve_run_spec(spec, root=ROOT)


def test_practice_on_an_evaluation_seed_is_refused(workspace, monkeypatch):
    with pytest.raises(RunSpecError, match="evaluation range"):
        _robotwin_plan(workspace, monkeypatch, partition="practice")


def test_practice_below_the_evaluation_range_records_the_list_it_was_checked_against(workspace, monkeypatch):
    plan = _robotwin_plan(workspace, monkeypatch, seed=5, partition="practice")
    assert plan["partition"] == {"name": "practice", "evaluation_sets_sha256": file_sha256(MANIFEST)}
    evaluation = _robotwin_plan(workspace, monkeypatch, seed=5)
    assert evaluation["partition"] == {"name": "evaluation"}
    assert plan["plan_sha256"] != evaluation["plan_sha256"]


@pytest.mark.parametrize(("portable", "message"), [
    ({"runtime": {"kind": "robodojo"}, "seed": 0}, "no practice side"),
    ({"runtime": {"kind": "robolab"}, "seed": 7}, "no practice side"),
    ({"runtime": {"kind": "libero_pro"}, "seed": 3}, "official initial state"),
    ({"runtime": {"kind": "robocasa"}, "seed": 1}, "evaluation split"),
    ({"runtime": {"kind": "libero_pro"}, "seed": 3, "episode_mode": "play"}, "free-play"),
])
def test_practice_is_refused_where_no_practice_side_applies(portable, message):
    with pytest.raises(RunSpecError, match=message):
        resolve_partition({"partition": "practice"}, portable)


def test_practice_passes_the_libero_practice_driver_and_the_robocasa_pretrain_split():
    libero = {"runtime": {"kind": "libero_pro", "environment_driver": "direct_libero_practice_v1"}, "seed": 1000001}
    robocasa = {"runtime": {"kind": "robocasa", "split": "pretrain"}, "seed": 1}
    for portable in (libero, robocasa):
        assert resolve_partition({"partition": "practice"}, portable)["name"] == "practice"


def test_unknown_partitions_are_refused_and_the_vocabulary_is_fixed():
    assert PARTITIONS == ("evaluation", "practice")
    with pytest.raises(RunSpecError, match="unsupported partition"):
        resolve_partition({"partition": "training"}, {"runtime": {"kind": "robotwin"}, "seed": 5})


def _label(run: Path, partition: str | None) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if partition is not None:
        plan["partition"] = {"name": partition}
    (run / "plan.json").write_text(json.dumps(plan))


def test_the_report_keeps_practice_apart_and_reports_its_budget(tmp_path):
    success = round_record(success=True, termination="trusted_runtime_boundary", authority="environment",
                           upstream="official_success_latch")
    _label(write_run(tmp_path, "e1", rounds=[success]), "evaluation")
    _label(write_run(tmp_path, "p1", rounds=[round_record(decisions=4, calls=9, minutes=30.0)]), "practice")
    _label(write_run(tmp_path, "p2", rounds=[success]), "practice")
    _label(write_run(tmp_path, "old"), None)
    report = run_report.build_report(run_report.discover_runs(tmp_path), pricing={"models": {}}, corrections=[])
    groups = {item["key"]["partition"]: item for item in report["partitions"]}
    assert set(groups) == {"evaluation", "practice", "unlabelled"}
    assert (groups["practice"]["runs"], groups["practice"]["successes"]) == (2, 1)
    assert groups["practice"]["effort"]["model_calls"] >= 9
    assert groups["evaluation"]["runs"] == 1
    assert report["total"]["partitions"] == ["evaluation", "practice", "unlabelled"]
    markdown = run_report.render_markdown(report)
    assert "## Partitions" in markdown and "Practice runs are not evaluation results" in markdown
