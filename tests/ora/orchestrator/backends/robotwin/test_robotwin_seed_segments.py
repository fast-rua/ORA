"""Host-only RoboTwin seed-scan segments: planning, interim seals and the first-N merge.

No simulator is initialized; manifests and ledgers are fixtures in the shapes
``backends.robotwin.evaluation_seeds qualify`` and the oracle worker write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ora.orchestrator.backends.robotwin import evaluation_seeds as run
from ora.orchestrator.backends.robotwin import seed_manifest
from ora.orchestrator.backends.robotwin import seed_segments as segments
from tests.support.robotwin.evaluation_setting import _ledger_record, _official_robotwin_root


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = _official_robotwin_root(tmp_path)
    monkeypatch.setattr(run, "git", lambda root, *args: "" if args[0] == "status" else run.REQUIRED_REVISION)
    monkeypatch.setattr(run, "source_dependencies", lambda root, tasks, task_config: [])
    monkeypatch.setattr(run, "capacity", lambda: {"cpu_count": 16, "available_memory_gib": 64,
                                                 "gpus": {"6": {"free_memory_mib": 48000}, "7": {"free_memory_mib": 48000}}})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    return root


def test_prepare_splits_a_task_into_offset_segments(prepared):
    root = prepared
    plan = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"],
                       slots=["robotwin-gpu6", "robotwin-gpu7"], candidate_count=100, accepted_per_task=100,
                       task_config="demo_clean", start_offsets=[200, 300])
    cases = plan["qualification"]["cases"]
    assert [(c["case_id"], c["start_requested_seed"], c["exclusive_seed_limit"]) for c in cases] == [
        ("rotate_qrcode@100200", 100200, 100300), ("rotate_qrcode@100300", 100300, 100400)]
    assert [lane["tasks"] for lane in plan["qualification"]["parallel_execution"]["lanes"]] == [
        ["rotate_qrcode@100200"], ["rotate_qrcode@100300"]]
    assert plan["qualification"]["process_attempt_limit"] == 8
    explicit = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode", "click_bell"],
                           slots=["robotwin-gpu6", "robotwin-gpu7"], candidate_count=50, accepted_per_task=50,
                           task_config="demo_clean", segments=[("rotate_qrcode", 200), ("rotate_qrcode", 250),
                                                               ("click_bell", 200)])
    assert [c["case_id"] for c in explicit["qualification"]["cases"]] == [
        "rotate_qrcode@100200", "rotate_qrcode@100250", "click_bell@100200"]
    with pytest.raises(ValueError, match="segments"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["click_bell"],
                    slots=["robotwin-gpu6"], candidate_count=1, segments=[("rotate_qrcode", 200)])
    # A segment may carry its own length; the accepted target never exceeds it.
    sized = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["click_bell"],
                        slots=["robotwin-gpu6"], candidate_count=4, accepted_per_task=4, task_config="demo_clean",
                        segments=segments.parse_segments(["click_bell:8:4", "click_bell:12:1"]))
    assert [(c["start_requested_seed"], c["exclusive_seed_limit"], c["accepted_target"])
            for c in sized["qualification"]["cases"]] == [(100008, 100012, 4), (100012, 100013, 1)]
    with pytest.raises(ValueError, match="segments"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["click_bell"],
                    slots=["robotwin-gpu6"], candidate_count=4, segments=[("click_bell", 8, 5)])
    with pytest.raises(ValueError, match="start_offsets"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"],
                    slots=["robotwin-gpu6"], candidate_count=1, start_offsets=[5, 5])
    with pytest.raises(ValueError, match="process_attempt_limit"):
        run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["rotate_qrcode"],
                    slots=["robotwin-gpu6"], candidate_count=1, process_attempt_limit=0)


def test_a_practice_scan_counts_from_its_start_and_stays_below_the_evaluation_seeds(prepared):
    """Development seeds: offsets from practice_start, every case below the lowest official start."""
    root = prepared
    assert segments.PRACTICE_SEED_LIMIT == 100000

    def practice(**changes):
        arguments = dict(robotwin_root=root, python=Path(sys.executable), tasks=["click_bell", "move_can_pot"],
                         slots=["robotwin-gpu6", "robotwin-gpu7"], candidate_count=16, accepted_per_task=3,
                         purpose="development_diagnostic", practice_start=50000,
                         segments=segments.parse_segments(["click_bell:0:16", "move_can_pot:0:8"]))
        return run.prepare(**{**arguments, **changes})

    plan = practice()
    cases = plan["qualification"]["cases"]
    # Practice case ids always carry their first seed, so none reads like an official scan's.
    assert [(c["case_id"], c["start_requested_seed"], c["exclusive_seed_limit"], c["accepted_target"])
            for c in cases] == [("click_bell@50000", 50000, 50016, 3), ("move_can_pot@50000", 50000, 50008, 3)]
    assert {c["decision"] for c in cases} == {"execute_practice_seed_scan"}
    assert plan["practice_start"] == 50000 and plan["qualification"]["purpose"] == "development_diagnostic"
    official = run.prepare(robotwin_root=root, python=Path(sys.executable), tasks=["click_bell"],
                           slots=["robotwin-gpu6"], candidate_count=1)
    assert "practice_start" not in official and official["qualification"]["cases"][0]["case_id"] == "click_bell"
    with pytest.raises(ValueError, match="development_diagnostic"):
        practice(purpose="targeted_regression")
    with pytest.raises(ValueError, match="practice cases reach the evaluation seeds"):
        practice(practice_start=99990)
    for start in (100000, -1, True):
        with pytest.raises(ValueError, match="practice_start must be"):
            practice(practice_start=start)
    with pytest.raises(ValueError, match="no outer seed"):
        practice(outer_seed=1)


def test_prepare_keeps_a_venv_interpreter_path_unresolved(prepared, tmp_path):
    """A uv venv's bin/python is a symlink; resolving it would drop the venv."""
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    plan = run.prepare(robotwin_root=prepared, python=venv_python, tasks=["rotate_qrcode"],
                       slots=["robotwin-gpu6"], candidate_count=1)
    assert plan["robotwin_python"] == str(venv_python)


def test_capacity_leaves_out_a_gpu_that_needs_a_reset(monkeypatch):
    from ora.orchestrator.preflight.resource_inventory import GPU_MEMORY_PROBE_TIMEOUT_S
    seen = {}
    def nvidia_smi(*args, **kwargs):
        seen.update(kwargs)
        return "0, 143152, [N/A]\n1, 143080, 12\n"
    monkeypatch.setattr(run.subprocess, "check_output", nvidia_smi)
    assert list(run.capacity()["gpus"]) == ["1"]
    assert seen["timeout"] == GPU_MEMORY_PROBE_TIMEOUT_S  # the lease probe's bound


def _segment(directory, task, start, limit, accepted, rejected, revision="cf", producer="dep", practice_start=None):
    seeds = {str(s): {"reset_predicate_false": True, "oracle_success": True,
                      "post_success_physics_hold": True, "terminal_stability_steps": 250} for s in accepted}
    body = {"schema": run.SEED_MANIFEST_SCHEMA, "task": task, "task_config": "demo_clean",
            "robotwin_revision": revision, "source_commit": "c" * 40, "plan_sha256": f"plan-{start}",
            "producer_dependency_hash": producer, "start_requested_seed": start, "exclusive_seed_limit": limit,
            "accepted_target": limit - start, "accepted_seeds": accepted,
            "evaluated_seeds": sorted(accepted + rejected),
            "rejected": [{"requested_seed": s, "classification": "oracle_failure", "error": None} for s in rejected],
            "complete": True, "seeds": seeds, "required_stability_steps": 250, "ledger_sha256": f"l{start}"}
    if practice_start is not None:
        body["practice_start"] = practice_start
    body["manifest_sha256"] = run.digest(body)
    path = directory / f"{task}@{start}.json"
    path.write_text(json.dumps(body))
    return path


def test_merge_keeps_the_first_n_accepted_seeds_of_the_scanned_prefix(tmp_path):
    a = _segment(tmp_path, "rotate_qrcode", 100000, 100003, [100000, 100002], [100001])
    b = _segment(tmp_path, "rotate_qrcode", 100003, 100006, [100004, 100005], [100003])
    merged = segments.merge_seed_manifests([b, a], task_config="demo_clean", accepted_target=3)["rotate_qrcode"]
    assert merged["accepted_seeds"] == [100000, 100002, 100004] and merged["complete"]
    assert merged["evaluated_seeds"] == [100000, 100001, 100002, 100003, 100004]
    assert merged["exclusive_seed_limit"] == 100005 and merged["next_unscanned_seed"] is None
    assert [item["start_requested_seed"] for item in merged["merged_from"]] == [100000, 100003]
    # The merged manifest keeps the official schema: launch preflight and the
    # RunSpec builder admit its seeds by the one seed_admission rule.
    for seed, admitted in ((100004, True), (100003, False), (100005, False)):
        passed, _ = seed_manifest.seed_admission(merged, task="rotate_qrcode", seed=seed, task_config="demo_clean",
                                                 revision="cf")
        assert passed is admitted, seed
    # A gap before the target leaves the manifest incomplete at the gap.
    c = _segment(tmp_path, "rotate_qrcode", 100010, 100012, [100010, 100011], [])
    gapped = segments.merge_seed_manifests([a, c], task_config="demo_clean", accepted_target=3)["rotate_qrcode"]
    assert gapped["accepted_seeds"] == [100000, 100002] and not gapped["complete"]
    assert gapped["next_unscanned_seed"] == 100003
    (tmp_path / "x").mkdir()
    clash = _segment(tmp_path / "x", "rotate_qrcode", 100000, 100003, [100001], [100000, 100002])
    with pytest.raises(ValueError, match="conflicting"):
        segments.merge_seed_manifests([a, clash], task_config="demo_clean", accepted_target=3)
    # Under the reject policy every disagreeing seed counts as rejected, in any input order.
    for inputs in ([a, clash, b], [b, clash, a]):
        lenient = segments.merge_seed_manifests(inputs, task_config="demo_clean", accepted_target=3,
                                                conflicting_outcomes="reject")["rotate_qrcode"]
        assert lenient["conflicting_seeds"] == [100000, 100001, 100002]
        assert lenient["accepted_seeds"] == [100004, 100005] and not lenient["complete"]
        assert {r["requested_seed"]: r["classification"] for r in lenient["rejected"]}[100001] == "conflicting_outcomes"
        passed, _ = seed_manifest.seed_admission(lenient, task="rotate_qrcode", seed=100000,
                                                 task_config="demo_clean", revision="cf")
        assert passed is False   # a seed the expert solved only sometimes is never admitted
    # Agreeing re-scans merge under the reject policy and flag nothing.
    (tmp_path / "z").mkdir()
    again = _segment(tmp_path / "z", "rotate_qrcode", 100000, 100003, [100000, 100002], [100001])
    agreed = segments.merge_seed_manifests([a, again, b], task_config="demo_clean", accepted_target=3,
                                           conflicting_outcomes="reject")["rotate_qrcode"]
    assert agreed["accepted_seeds"] == [100000, 100002, 100004] and agreed["conflicting_seeds"] == []
    with pytest.raises(ValueError, match="conflicting_outcomes"):
        segments.merge_seed_manifests([a], task_config="demo_clean", conflicting_outcomes="accept")
    (tmp_path / "y").mkdir()
    other = _segment(tmp_path / "y", "rotate_qrcode", 100003, 100006, [100004], [100003, 100005],
                     revision="other")
    with pytest.raises(ValueError, match="robotwin_revision"):
        segments.merge_seed_manifests([a, other], task_config="demo_clean")


def test_practice_segments_merge_from_the_practice_start_and_never_with_official_ones(tmp_path):
    a = _segment(tmp_path, "click_bell", 50000, 50003, [50000, 50002], [50001], practice_start=50000)
    b = _segment(tmp_path, "click_bell", 50003, 50006, [50004], [50003, 50005], practice_start=50000)
    merged = segments.merge_seed_manifests([b, a], task_config="demo_clean", accepted_target=3,
                                           practice_start=50000)["click_bell"]
    assert merged["accepted_seeds"] == [50000, 50002, 50004] and merged["complete"]
    assert merged["start_requested_seed"] == 50000 and merged["practice_start"] == 50000
    passed, _ = seed_manifest.seed_admission(merged, task="click_bell", seed=50004, task_config="demo_clean", revision="cf")
    assert passed is True
    official = _segment(tmp_path, "click_bell", 100000, 100003, [100000], [100001, 100002])
    with pytest.raises(ValueError, match="practice segments merge only with --practice-start"):
        segments.merge_seed_manifests([a], task_config="demo_clean", accepted_target=1)
    with pytest.raises(ValueError, match="official segment"):
        segments.merge_seed_manifests([official], task_config="demo_clean", accepted_target=1, practice_start=50000)
    output = tmp_path / "merged"
    assert run.main(["merge", "--task-config", "demo_clean", "--accepted-per-task", "3", "--practice-start", "50000",
                     "--inputs", str(a), str(b), "--output", str(output)]) == 0
    assert json.loads((output / "click_bell.json").read_text())["accepted_seeds"] == [50000, 50002, 50004]
    # A practice plan's sealed manifests name their practice start.
    plan = {"schema": "ora.robotwin-evaluation-preparation.v1", "task_config": "demo_clean",
            "robotwin_revision": "cf", "source_commit": "c" * 40, "practice_start": 50000,
            "qualification": {"producer_dependency_hash": "dep", "cases": [
                {"task": "click_bell", "case_id": "click_bell@50000", "start_requested_seed": 50000,
                 "exclusive_seed_limit": 50004, "accepted_target": 1}]}}
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "prepared-plan.json").write_text(json.dumps({**plan, "plan_sha256": run.digest(plan)}))
    ledger = scan / "click_bell@50000" / "click_bell" / "seed-ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(_ledger_record(50000, accepted=True, hold=(True, 250, None))) + "\n")
    sealed = json.loads(segments.seal_interim(scan)["click_bell@50000"].read_text())
    assert sealed["practice_start"] == 50000 and sealed["accepted_seeds"] == [50000]


def test_interim_seal_reads_a_running_ledger_and_skips_a_torn_line(tmp_path):
    plan = {"schema": "ora.robotwin-evaluation-preparation.v1", "task_config": "demo_clean",
            "robotwin_revision": "cf", "source_commit": "c" * 40,
            "qualification": {"producer_dependency_hash": "dep", "cases": [
                {"task": "click_bell", "case_id": "click_bell@100000", "start_requested_seed": 100000,
                 "exclusive_seed_limit": 100050, "accepted_target": 50}]}}
    (tmp_path / "prepared-plan.json").write_text(json.dumps({**plan, "plan_sha256": run.digest(plan)}))
    ledger = tmp_path / "click_bell@100000" / "click_bell" / "seed-ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps(r) for r in (
        _ledger_record(100000, accepted=True, hold=(True, 250, None)),
        _ledger_record(100001, accepted=False, classification="scene_unstable"),
    )) + '\n{"requested_seed": 1000')
    written = segments.seal_interim(tmp_path)
    manifest = json.loads(written["click_bell@100000"].read_text())
    assert written["click_bell@100000"].parent.name == "seed-manifests-interim"
    assert manifest["accepted_seeds"] == [100000] and manifest["complete"] is False
    assert not (tmp_path / "seed-manifests").exists()
    assert json.loads(segments.seal_interim(tmp_path)["click_bell@100000"].read_text()) == manifest  # re-sealable
    merged = segments.merge_seed_manifests([written["click_bell@100000"]], task_config="demo_clean",
                                           accepted_target=1)
    assert merged["click_bell"]["accepted_seeds"] == [100000] and merged["click_bell"]["complete"]


def test_merge_compares_each_segments_task_closure_not_the_plan_wide_hash(tmp_path):
    root = tmp_path / "RoboTwin"
    def plan_dir(name, tasks, start, accepted, rejected, edit=None):
        closure = [{"path": str(root / "envs" / "_base_task.py"), "sha256": "base"}]
        closure += [{"path": str(root / "envs" / f"{t}.py"), "sha256": (edit or {}).get(t, t)} for t in tasks]
        plan = {"robotwin_root": str(root), "qualification": {"dependency_closure": closure, "checker_hash": "chk"}}
        directory = tmp_path / name / "seed-manifests"
        directory.mkdir(parents=True)
        path = _segment(directory, "click_bell", start, start + 3, accepted, rejected, producer=name)
        plan["plan_sha256"] = json.loads(path.read_text())["plan_sha256"]
        (tmp_path / name / "prepared-plan.json").write_text(json.dumps(plan))
        return path
    a = plan_dir("p1", ["click_bell", "lift_pot"], 100000, [100000, 100001], [100002])
    b = plan_dir("p2", ["click_bell"], 100003, [100003], [100004, 100005])
    merged = segments.merge_seed_manifests([a, b], task_config="demo_clean", accepted_target=3)["click_bell"]
    assert merged["accepted_seeds"] == [100000, 100001, 100003] and merged["dependency_scope"] == "task"
    c = plan_dir("p3", ["click_bell"], 100003, [100003], [100004, 100005], edit={"click_bell": "changed"})
    with pytest.raises(ValueError, match="task_dependency_hash"):
        segments.merge_seed_manifests([a, c], task_config="demo_clean")


def test_merge_and_seal_entry_points(tmp_path, capsys):
    qualification = tmp_path / "q1"
    (qualification / "seed-manifests-interim").mkdir(parents=True)
    _segment(qualification / "seed-manifests-interim", "rotate_qrcode", 100000, 100003, [100000, 100002], [100001])
    output = tmp_path / "merged"
    assert run.main(["merge", "--task-config", "demo_clean", "--accepted-per-task", "2",
                     "--inputs", str(qualification), "--output", str(output)]) == 0
    assert json.loads((output / "rotate_qrcode.json").read_text())["accepted_seeds"] == [100000, 100002]
    report = json.loads((output / "merge-report.json").read_text())
    assert report["tasks"]["rotate_qrcode"]["complete"] is True
    assert run.main(["merge", "--task-config", "demo_clean", "--accepted-per-task", "3",
                     "--inputs", str(qualification), "--output", str(tmp_path / "short")]) == 2
    (qualification / "x").mkdir()
    _segment(qualification / "x", "rotate_qrcode", 100000, 100003, [100001], [100000, 100002])
    assert run.main(["merge", "--task-config", "demo_clean", "--accepted-per-task", "1",
                     "--inputs", *map(str, sorted(qualification.glob("*/*.json"))),
                     "--output", str(tmp_path / "rejected"), "--conflicting-outcomes", "reject"]) == 2
    assert json.loads((tmp_path / "rejected" / "rotate_qrcode.json").read_text())["conflicting_seeds"] == [
        100000, 100001, 100002]
    with pytest.raises(SystemExit):
        run.main(["seal"])
    capsys.readouterr()
