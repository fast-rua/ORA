"""Result tiers: the plan's label for what the policy carried into the episode.

A fresh benchmark plan is ``zero_shot``; ``zero_shot_no_check`` is the same
baseline without ``check_task``, and the tier must match the allowlist both
ways. Tiers this tree cannot produce (``with_experience``, ``frozen_program``)
are refused. The run report groups results by tier, keeps plans recorded
before the field apart as ``unlabelled`` and counts ``check_task`` calls.
Nothing here starts a simulator or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.orchestrator.spec import benchmark as benchmark_spec
from ora.orchestrator.reports import run_report
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.spec.result_tier import RESULT_TIERS
from tests.support.orchestrator import benchmark_specs as bench
from tests.support.orchestrator.run_report_fixture import round_record, write_run

ROOT = Path(__file__).resolve().parents[4]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())


def _robotwin_plan(workspace, monkeypatch, **spec_fields):
    report = bench._readiness_report(workspace / "reports", monkeypatch)
    spec, manifest = bench._robotwin(workspace, service_report=report, cohort_id="result-tier-fixture-g001")
    spec.update(spec_fields)
    benchmark_spec.compose(spec, seed_manifest=manifest)
    return resolve_run_spec(spec, root=ROOT)


def test_the_default_benchmark_plan_is_zero_shot_and_advertises_check_task(workspace, monkeypatch):
    plan = _robotwin_plan(workspace, monkeypatch)
    assert plan["result_tier"] == ACTIVATION["default_result_tier"] == "zero_shot"
    assert "check_task" in plan["policy"]["tool_allowlist"]


def test_zero_shot_no_check_is_the_baseline_without_check_task(workspace, monkeypatch):
    default = _robotwin_plan(workspace, monkeypatch)
    plan = _robotwin_plan(workspace, monkeypatch, result_tier="zero_shot_no_check")
    assert plan["result_tier"] == "zero_shot_no_check"
    expected = [name for name in default["policy"]["tool_allowlist"] if name != "check_task"]
    assert plan["policy"]["tool_allowlist"] == expected
    assert plan["plan_sha256"] != default["plan_sha256"]


def test_withholding_check_task_without_the_no_check_tier_is_refused(workspace, monkeypatch):
    default = _robotwin_plan(workspace, monkeypatch)
    withheld = [name for name in default["policy"]["tool_allowlist"] if name != "check_task"]
    report = bench._readiness_report(workspace / "reports", monkeypatch)
    spec, manifest = bench._robotwin(workspace, service_report=report, cohort_id="result-tier-fixture-g001")
    spec["policy"]["tool_allowlist"] = withheld
    benchmark_spec.compose(spec, seed_manifest=manifest)
    with pytest.raises(RunSpecError):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize(("tier", "message"), [
    ("with_experience", "experience library binding"),
    ("frozen_program", "outside the ORA Runtime"),
    ("few_shot", "unsupported result_tier"),
])
def test_tiers_this_tree_cannot_produce_are_refused(workspace, monkeypatch, tier, message):
    with pytest.raises(RunSpecError, match=message):
        _robotwin_plan(workspace, monkeypatch, result_tier=tier)


def test_the_tier_vocabulary_is_fixed():
    assert RESULT_TIERS == ("zero_shot", "with_experience", "zero_shot_no_check", "frozen_program")


def _label(run: Path, tier: str | None) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if tier is not None:
        plan["result_tier"] = tier
    (run / "plan.json").write_text(json.dumps(plan))


def test_the_report_groups_by_tier_counts_check_task_and_keeps_unlabelled_runs_apart(tmp_path):
    success = round_record(success=True, termination="trusted_runtime_boundary", authority="environment",
                           upstream="official_success_latch")
    _label(write_run(tmp_path, "z1", rounds=[success],
                     events={"tools": [("check_task", True), ("check_task", True), ("move_pose", True)]}),
           "zero_shot")
    _label(write_run(tmp_path, "z2", events={"tools": [("check_task", True)]}), "zero_shot")
    _label(write_run(tmp_path, "n1", rounds=[success], events={"tools": [("move_pose", True)]}),
           "zero_shot_no_check")
    _label(write_run(tmp_path, "old", rounds=[success]), None)
    report = run_report.build_report(run_report.discover_runs(tmp_path), pricing={"models": {}}, corrections=[])
    tiers = {item["key"]["result_tier"]: item for item in report["result_tiers"]}
    assert set(tiers) == {"zero_shot", "zero_shot_no_check", "unlabelled"}
    assert (tiers["zero_shot"]["runs"], tiers["zero_shot"]["successes"], tiers["zero_shot"]["check_task_calls"]) \
        == (2, 1, 3)
    assert (tiers["zero_shot_no_check"]["successes"], tiers["zero_shot_no_check"]["check_task_calls"]) == (1, 0)
    assert tiers["unlabelled"]["runs"] == 1
    assert report["total"]["result_tiers"] == ["unlabelled", "zero_shot", "zero_shot_no_check"]
    markdown = run_report.render_markdown(report)
    assert "## Result tiers" in markdown and "not pooled" in markdown


def test_a_scripted_policy_keeps_zero_shot_without_the_allowlist_check():
    from ora.orchestrator.spec.result_tier import resolve_result_tier

    class Surface:
        definitions = [{"name": "check_task"}, {"name": "move_pose"}]

    scripted = {"kind": "scripted", "tool_allowlist": ["move_pose"]}
    assert resolve_result_tier({}, scripted, Surface(), None) == "zero_shot"
    with pytest.raises(RunSpecError, match="native_api policies only"):
        resolve_result_tier({"result_tier": "zero_shot_no_check"}, scripted, Surface(), None)
