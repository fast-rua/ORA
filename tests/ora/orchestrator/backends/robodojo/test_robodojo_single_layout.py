"""One RoboDojo episode on an official layout k of evaluation seed 0.

An unscheduled, one-round RoboDojo runtime may name any official layout
ordinal whose canonical file it binds; the worker resets round 0 (and an
in-place round reset) on that layout; the lane gate of that layout is tagged
``l<k>``; and the default builder composes the episode only where the reviewed
layout-portability table extends the task's registry-v10 contract to that
layout's bytes. Tasks whose instruction the layout generates take it from the
scene-smoke readback their lane gate certifies. Layout schedules of persistent
sessions are unchanged (tests/ora/environment/backends/robodojo/test_robodojo_layout_schedule.py). No simulator:
sessions, plants and qualification records are fakes; tests marked
``needs_assets`` read the pinned Assets layouts when present.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition.locked_assets import locked_paths
from ora.agent.composition.task_contract.robodojo_layouts import (
    INSTRUCTION_LAYOUT_READBACK, LAYOUT_PORTABILITY_PATH, PORTABILITY_SCHEMA, portable_layout,
)
from ora.agent.composition.task_contract.robodojo import REGISTRY_PATH, resolve_reviewed_contract
from ora.environment.backends.robodojo.session import RobodojoError, RobodojoSession
from ora.environment.benchmarks import robodojo
from ora.environment.benchmarks.robodojo_layouts import layout_cell_tag, layout_file_for
from ora.orchestrator.backends.robodojo import lane_qualification as lane
from ora.orchestrator.backends.robodojo import spec as robodojo_spec
from ora.orchestrator import robodojo_revalidation as rv
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.robodojo import runtime as robodojo_runtime

ROOT = Path(__file__).resolve().parents[5]
REGISTRY = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"]
TABLE = json.loads((ROOT / LAYOUT_PORTABILITY_PATH).read_text())
ASSETS = Path(os.environ.get("ROBODOJO_ASSET_ROOT", str(robodojo_runtime.ASSET_ROOT)))
needs_assets = pytest.mark.skipif(not (ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
                                  reason="pinned RoboDojo Assets absent")
#: The audited tasks whose instruction gen_instruction builds from the layout's objects.
READBACK_TASKS = {"stack_blocks_by_language", "classify_objects_by_language", "general_pickup", "pour_by_language"}
#: Tasks whose record holds on some of official layouts 0-9 only; the v3 and v5 review records say why.
PARTIAL_LAYOUTS = {"sweep_blocks": [0, 3, 4, 7], "hang_mugs_random": [0, 1, 2, 3, 4, 6, 7, 8, 9]}
#: The table's predecessors, whose entries v4 carries with their layouts and bytes unchanged (each removed
#: from the tree once nothing loaded it; Git history keeps them), and the review records of v3 and v4.
TABLE_V2 = "ora/policy/environments/robodojo/task-contracts/layout-portability-v2.json"
TABLE_V3 = "ora/policy/environments/robodojo/task-contracts/layout-portability-v3.json"
REVIEW_V3 = "ora/policy/environments/robodojo/task-contracts/reviews/layout-portability-v3-review.json"
REVIEW_V4 = "ora/policy/environments/robodojo/task-contracts/reviews/layout-portability-v4-review.json"
#: v5 (registry-v10) adds the reviewed *_random variants; v4 stays in the tree beside the frozen registry-v9.
TABLE_V4 = "ora/policy/environments/robodojo/task-contracts/layout-portability-v4.json"
REVIEW_V5 = "ora/policy/environments/robodojo/task-contracts/reviews/layout-portability-v5-review.json"
REGISTRY_V9 = "ora/policy/environments/robodojo/task-contracts/registry-v9.json"
QUALIFICATION = {"path": "outputs/single-layout-fixture/qualification/revalidation/gate.json", "sha256": "0" * 64}
EXPERIMENT = "single-layout-fixture"


@pytest.fixture(autouse=True)
def _registered_execution_host(monkeypatch):
    """RoboDojo runtimes bind the marker's registered host; tests run on the fixture host."""
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")


@pytest.fixture(autouse=True)
def _restore_service_endpoints():
    saved = {variable: os.environ.pop(variable, None) for _, variable in robodojo_spec.SERVICE_VARIABLES}
    yield
    for variable, value in saved.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


# --- host-safe admission -------------------------------------------------------------------------

def test_admission_takes_one_official_layout_whose_canonical_file_the_runtime_binds():
    from tests.support.robodojo.admission import _runtime

    base = _runtime()
    robodojo.validate_runtime_extension(base)  # layout 0 exactly as before
    single = {**base, "layout_id": 3, "layout_file": layout_file_for("arrange_largest_number", 0, 3)}
    robodojo.validate_runtime_extension(single)
    for change in ({"layout_file": base["layout_file"]}, {"layout_id": -1}, {"layout_id": True},
                   {"layout_file": layout_file_for("arrange_largest_number", 0, 4)}):
        with pytest.raises(robodojo.RoboDojoBenchmarkError, match="official layout ordinal"):
            robodojo.validate_runtime_extension({**single, **change})


def test_an_ordinal_the_pinned_assets_do_not_hold_is_refused(tmp_path, monkeypatch):
    from ora.orchestrator.backends.robodojo.runtime_config import runtime_configuration

    folder = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    for ordinal in range(5):
        (folder / f"stack_bowls_{ordinal}.json").write_text(json.dumps({"ordinal": ordinal}))
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    runtime = runtime_configuration("0", "stack_bowls", 4)
    assert (runtime["layout_id"], runtime["layout_file"]) == (4, layout_file_for("stack_bowls", 0, 4))
    assert runtime["layout_sha256"] == hashlib.sha256((folder / "stack_bowls_4.json").read_bytes()).hexdigest()
    assert runtime_configuration("0", "stack_bowls")["layout_id"] == 0
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="unavailable"):
        runtime_configuration("0", "stack_bowls", 5)


# --- the worker ------------------------------------------------------------------------------------

def test_the_worker_binds_round_zero_to_the_runplans_one_layout():
    from ora.environment.backends.robodojo.bridge import manifest_bindings

    assert manifest_bindings({"action_codec": "motion_action_v3"}, "stack_bowls")["layout_id"] == 0
    assert manifest_bindings({"action_codec": "motion_action_v3", "layout_id": 3}, "stack_bowls")["layout_id"] == 3
    for layout_id in (-1, True, "3"):
        with pytest.raises(RobodojoError, match="layout_id"):
            manifest_bindings({"action_codec": "motion_action_v3", "layout_id": layout_id}, "stack_bowls")


def test_an_unscheduled_session_resets_and_restarts_in_place_on_its_one_layout(tmp_path):
    from tests.support.robodojo.backend import Plant, _fake_source_manifest

    class SeedPlant(Plant):
        def reset(self, seed):
            assert len(seed) == 1 and seed[0] in self.harness_layout_ids
            self.seeds = [*getattr(self, "seeds", []), seed[0]]
            super().reset([0])

    plant = SeedPlant()
    session = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}, "layout_id": 3},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None),
                                            _fake_source_manifest(tmp_path)))
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    try:
        start = session.start()
        assert plant.harness_layout_ids == (3,) and plant.seeds == [3]
        assert start["layout_id"] == 3 and "layout_verification" not in start
        session.abort("qualification_boundary")
        session.persist_official_verdict()
        (session.output_root / "official-verdict.json").unlink()
        reset = session.restart_round(1, seed=0)
        assert reset["layout_id"] == 3 and reset["round_isolation"]["passed"] is True
        assert plant.seeds == [3, 3]
    finally:
        session.close()


# --- lane gate tags ----------------------------------------------------------------------------------

def test_the_lane_gate_and_cell_name_the_single_layout(monkeypatch, tmp_path):
    from tests.support.robodojo.lane_qualification import _configured

    assert layout_cell_tag(None) == "l0" and layout_cell_tag(None, 3) == "l3"
    assert layout_cell_tag({"rounds": [{"layout_id": 23}, {"layout_id": 29}]}, 7) == "l23.29"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="one layout ordinal"):
        layout_cell_tag(None, -1)
    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    assert lane.gate_path("lane-fixture", _configured()) == workspace / "gates/stack_bowls-l0-cuda0.json"
    assert lane.gate_path("lane-fixture", _configured(layout_id=3)) == workspace / "gates/stack_bowls-l3-cuda0.json"
    assert lane.gate_path("lane-fixture", _configured(layout_id=3, physics_device="cpu", cuda_visible_devices="4")) \
        == workspace / "gates/stack_bowls-l3-cpu-gpu4.json"


def test_a_single_layout_lane_reads_only_smokes_of_that_layout(monkeypatch, tmp_path):
    from tests.support.robodojo.lane_qualification import _configured, _smoke_dir

    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    layout0, layout3 = _configured(), _configured(layout_id=3, layout_file="f3", layout_sha256="s3")
    _smoke_dir(workspace, "robodojo-scene-smoke-l0", runtime=layout0, mtime=5)
    chosen = _smoke_dir(workspace, "robodojo-scene-smoke-l3", runtime=layout3, mtime=1)
    assert lane.smoke_candidates("lane-fixture", layout3) == [chosen]
    assert [path.parent.name for path in lane.smoke_candidates("lane-fixture", layout0)] == [
        "robodojo-scene-smoke-l0"]


def test_the_lane_entry_points_pass_the_single_layout(monkeypatch, capsys, tmp_path):
    from ora.orchestrator import robodojo_scene_smoke as scene

    calls = []
    monkeypatch.setattr(lane, "revalidate", lambda task, **kwargs: calls.append(kwargs) or tmp_path / "g.json")
    monkeypatch.setattr(scene, "smoke", lambda task, **kwargs: calls.append(kwargs) or {"passed": True})
    common = ["--gpu", "0", "--experiment", "lane-fixture", "--layout", "3"]
    assert lane.main(["smoke", "stack_bowls", *common, "--yes"]) == 0
    assert lane.main(["revalidate", "stack_bowls", *common, "--card", str(tmp_path / "card.json")]) == 0
    assert [(call["layout_id"], call["rounds"], call["layout_schedule"]) for call in calls] == [
        (3, 1, None), (3, 1, None)]
    seen = []
    monkeypatch.setattr(robodojo_spec, "qualification_runtime", lambda task, **kwargs: seen.append(kwargs) or {})
    lane._runtime("stack_bowls", gpu="0", resource_profile=None, rounds=1, layout_schedule=None)
    lane._runtime("stack_bowls", gpu="0", resource_profile=None, rounds=1, layout_schedule=None, layout_id=3)
    assert "layout_id" not in seen[0] and seen[1]["layout_id"] == 3


# --- the reviewed layout-portability table ---------------------------------------------------------

def test_the_portability_table_extends_registry_v10_records_unchanged_from_layout_zero():
    assert TABLE["schema"] == PORTABILITY_SCHEMA and TABLE["registry"] == REGISTRY_PATH
    assert (TABLE["robodojo_revision"], TABLE["assets_revision"]) == (
        robodojo.REVISIONS["robodojo"], robodojo.REVISIONS["assets"])
    assert LAYOUT_PORTABILITY_PATH in locked_paths(ROOT)
    # v3: every admitted task is audited; a layout its row list leaves out stays refused.
    assert set(TABLE["tasks"]) == set(REGISTRY) == set(robodojo_spec.task_roster(ROOT))
    for task, entry in TABLE["tasks"].items():
        record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
        assert entry["record"] == REGISTRY[task]["record"]
        assert entry["contract_layer_sha256"] == record["contract_layer"]["sha256"]
        # Every row binds the table's one asset revision, the record's and the runtime's (swap_T's included).
        assert "assets_revision" not in entry
        assert record["layout"]["assets_revision"] == TABLE["assets_revision"] == robodojo.REVISIONS["assets"]
        assert [row["layout_id"] for row in entry["layouts"]] == PARTIAL_LAYOUTS.get(task, list(range(10)))
        # A row binds evaluator assets only its layout holds, never one the record already hashes.
        bound = {item["path"] for item in record.get("evaluator_dependencies", [])}
        for row in entry["layouts"]:
            extra = row.get("evaluator_asset_dependencies", [])
            # fasten_screws' nut/screw colours and make_toast_random's second toaster model (00002).
            assert not extra or (task in {"fasten_screws", "make_toast_random"} and row["layout_id"] > 0), (
                task, row["layout_id"])
            assert not {item["path"] for item in extra} & bound
        assert all(row["layout_file"] == layout_file_for(task, 0, row["layout_id"]) for row in entry["layouts"])
        assert entry["layouts"][0] == {"layout_id": 0, "layout_file": record["layout"]["layout_file"],
                                       "layout_sha256": record["layout"]["layout_sha256"]}
    assert {task for task, entry in TABLE["tasks"].items()
            if entry["instruction"] == INSTRUCTION_LAYOUT_READBACK} == READBACK_TASKS


def test_the_v4_table_and_its_review_record_stay_beside_the_frozen_registry_v9():
    """v4 is what plans composed on registry-v9 named; v5 carries its 42 entries unchanged."""
    table = json.loads((ROOT / TABLE_V4).read_text())
    review = json.loads((ROOT / REVIEW_V4).read_text())
    assert table["review"]["record"] == REVIEW_V4 and {TABLE_V4, REVIEW_V4} <= locked_paths(ROOT)
    assert table["registry"] == REGISTRY_V9 != REGISTRY_PATH
    assert {task: TABLE["tasks"][task] for task in table["tasks"]} == table["tasks"]
    # v4 re-binds v3's approved entries to the registry-v9 records; that re-binding awaits the user's approval.
    assert (review["table"], review["registry"], review["predecessor"]["path"], review["status"]) == (
        TABLE_V4, REGISTRY_V9, REVIEW_V3, "pending_review")
    assert review["reissue"]["human_approval"]["status"] == "pending"
    assert table["review"]["human_approval"]["v4_reissue"]["status"] == "pending"
    assert (review["source"]["robodojo_revision"], review["source"]["assets_revision"]) == (
        robodojo.REVISIONS["robodojo"], robodojo.REVISIONS["assets"])
    # The user's approval reached this review relayed, with conditions; reachability stays unreviewed.
    approval = review["review"]["human_approval"]
    assert (approval["status"], approval["by"], approval["date"]) == ("approved", "the user", "2026-10-20")
    assert "relayed" in approval["provenance"] and approval["user_words"] and len(approval["conditions"]) == 4
    assert any(item.startswith("Reachability") for item in review["not_reviewed"])
    assert "Reachability" in table["review"]["not_reviewed"]
    # Nothing loads v2 or v3 any more: they left the tree and the lock.
    for path in (TABLE_V2, TABLE_V3, REVIEW_V3):
        assert not (ROOT / path).exists() and path not in locked_paths(ROOT)
    carried = set(review["carried_from_v2"])
    assert len(carried) == 29 and set(review["tasks"]) == set(table["tasks"]) - carried
    # Approval by scope: the thirteen reviewed here and, separately, the 29 carried from v2.
    scopes = table["review"]["human_approval"]
    assert (scopes["v3_tasks"]["status"], scopes["v3_tasks"]["date"]) == ("approved", "2026-10-20")
    assert set(scopes["v3_tasks"]["tasks"]) == set(review["tasks"])
    assert all(row["human_approval"].startswith("approved") for row in review["tasks"].values())
    assert (scopes["carried_from_v2"]["status"], scopes["carried_from_v2"]["date"]) == ("approved", "2026-10-20")
    assert set(scopes["carried_from_v2"]["tasks"]) == carried and scopes["carried_from_v2"]["user_words"]
    assert "relayed" in scopes["carried_from_v2"]["provenance"]
    assert review["carried_from_v2_approval"].startswith("approved")
    for task, row in review["tasks"].items():
        entry = table["tasks"][task]
        assert row["portable_layouts"] == [layout["layout_id"] for layout in entry["layouts"]]
        assert (row["record"], row["instruction_source"]) == (entry["record"], entry["instruction"])
        assert [layout["layout_sha256"] for layout in row["layouts"] if layout["portable"]] == [
            layout["layout_sha256"] for layout in entry["layouts"]]


def test_the_v5_review_record_covers_exactly_the_random_variants_it_adds():
    review = json.loads((ROOT / REVIEW_V5).read_text())
    v4 = json.loads((ROOT / TABLE_V4).read_text())
    assert TABLE["review"]["record"] == REVIEW_V5 and REVIEW_V5 in locked_paths(ROOT)
    assert (review["table"], review["registry"], review["predecessor"]["path"], review["status"]) == (
        LAYOUT_PORTABILITY_PATH, REGISTRY_PATH, REVIEW_V4, "approved")
    assert review["predecessor"]["sha256"] == hashlib.sha256((ROOT / REVIEW_V4).read_bytes()).hexdigest()
    assert (review["source"]["robodojo_revision"], review["source"]["assets_revision"]) == (
        robodojo.REVISIONS["robodojo"], robodojo.REVISIONS["assets"])
    # The 42 v4 entries are carried unchanged; v5 adds every registry-v10 *_random variant and nothing else.
    assert set(review["carried_from_v4"]) == set(v4["tasks"])
    variants = {task for task in REGISTRY if task.endswith("_random")}
    assert set(review["tasks"]) == set(TABLE["tasks"]) - set(v4["tasks"]) == variants
    # Approved by the user on 2026-11-14, relayed by the coordinator; the v4 re-issue scope is untouched.
    approval = TABLE["review"]["human_approval"]["v5_random_variants"]
    for scope in (approval, review["review"]["human_approval"]):
        assert (scope["status"], scope["by"], scope["date"]) == ("approved", "the user", "2026-11-14")
        assert "relayed" in scope["provenance"] and scope["user_words"]
    assert set(approval["tasks"]) == variants
    assert all(row["human_approval"].startswith("approved") for row in review["tasks"].values())
    assert TABLE["review"]["human_approval"]["v4_reissue"] == v4["review"]["human_approval"]["v4_reissue"]
    for task, row in review["tasks"].items():
        entry = TABLE["tasks"][task]
        assert (row["record"], row["instruction_source"]) == (entry["record"], entry["instruction"]) == (
            REGISTRY[task]["record"], "fixed")
        assert row["contract_layer"] == REGISTRY[task]["layer"]
        assert row["portable_layouts"] == [layout["layout_id"] for layout in entry["layouts"]]
        assert [layout["layout_id"] for layout in row["layouts"]] == list(range(10))
        assert [layout["layout_sha256"] for layout in row["layouts"] if layout["portable"]] == [
            layout["layout_sha256"] for layout in entry["layouts"]]


@needs_assets
@pytest.mark.parametrize("task", sorted(TABLE["tasks"]))
def test_every_audited_layout_is_the_pinned_upstream_ordinal_and_bytes(task):
    assets = ASSETS
    rows = TABLE["tasks"][task]["layouts"]
    paths = [robodojo.selected_layout(assets, task=task, eval_seed=0, layout_id=row["layout_id"]) for row in rows]
    robodojo_runtime.skip_lfs_pointers([path for path in paths if robodojo_runtime.is_lfs_pointer(path)],
                                       f"{task}'s other layouts were not checked either")
    for row, path in zip(rows, paths):
        assert path.relative_to(assets.resolve()).as_posix() == row["layout_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["layout_sha256"]
        for dependency in row.get("evaluator_asset_dependencies", []):
            assert hashlib.sha256((assets / dependency["path"]).read_bytes()).hexdigest() == dependency["sha256"]


# --- contract delivery -----------------------------------------------------------------------------------

def _contract_runtime(task, layout_id, **changes):
    record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
    rows = {row["layout_id"]: row for row in TABLE["tasks"].get(task, {}).get("layouts", [])}
    row = rows.get(layout_id, {"layout_file": layout_file_for(task, 0, layout_id), "layout_sha256": "e" * 64})
    return {"task_variant": task, "required_revision": record["source"]["robodojo_revision"],
            "required_asset_revision": record["layout"]["assets_revision"], "eval_seed": 0,
            "layout_id": layout_id, "layout_file": row["layout_file"], "layout_sha256": row["layout_sha256"],
            "expected_task_language": record["instruction"]["text"], **changes}


def _resolve(runtime):
    return resolve_reviewed_contract(runtime, project_root=ROOT, disclosure={"backend": "robodojo"})


@pytest.mark.parametrize("task", sorted(TABLE["tasks"]))
def test_an_audited_record_is_delivered_unchanged_on_each_audited_layout(task):
    layer0, disclosure0 = _resolve(_contract_runtime(task, 0))
    assert "layout_portability" not in disclosure0
    for layout_id in [row["layout_id"] for row in TABLE["tasks"][task]["layouts"]][1:]:
        runtime = _contract_runtime(task, layout_id)
        layer, disclosure = _resolve(runtime)
        assert layer == layer0 == REGISTRY[task]["layer"] and disclosure["delivered"] is True
        assert {key: value for key, value in disclosure.items() if key != "layout_portability"} == disclosure0
        portability = disclosure["layout_portability"]
        assert (portability["table"], portability["layout_id"], portability["layout_sha256"]) == (
            LAYOUT_PORTABILITY_PATH, layout_id, runtime["layout_sha256"])
        assert portability["instruction_source"] == TABLE["tasks"][task]["instruction"]
        other = {**runtime, "expected_task_language": "A layout-generated instruction."}
        if task in READBACK_TASKS:
            assert _resolve(other)[0] == layer
        else:
            with pytest.raises(RuntimeError, match="task instruction differs"):
                _resolve(other)


def test_preflight_rehashes_the_evaluator_assets_a_portability_row_binds(tmp_path):
    """fasten_screws layout 2 holds nut/screw indices 0 and 3, whose colour bytes the record does not hash."""
    from ora.agent.composition.task_contract.robodojo import verify_contract_dependencies

    row = TABLE["tasks"]["fasten_screws"]["layouts"][2]
    disclosure = _resolve(_contract_runtime("fasten_screws", 2))[1]
    assert disclosure["layout_portability"]["evaluator_dependencies"] == row["evaluator_asset_dependencies"]
    assert {item["path"].split("/")[4] for item in row["evaluator_asset_dependencies"]} == {"00000", "00003"}
    assert "evaluator_dependencies" not in _resolve(_contract_runtime("stack_bowls", 2))[1]["layout_portability"]
    record = json.loads((ROOT / REGISTRY["fasten_screws"]["record"]).read_text())
    source, assets = tmp_path / "source", tmp_path / "assets"
    fixture = hashlib.sha256(b"fixture").hexdigest()
    for dependency in [{"root": "robodojo", "path": record["source"]["source_path"]},
                       *record["evaluator_dependencies"], *row["evaluator_asset_dependencies"]]:
        target = (source if dependency["root"] == "robodojo" else assets) / dependency["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    record_bytes = json.dumps({**record, "source": {**record["source"], "file_sha256": fixture},
                               "evaluator_dependencies": [{**item, "sha256": fixture}
                                                          for item in record["evaluator_dependencies"]]}).encode()
    (tmp_path / "record.json").write_bytes(record_bytes)
    disclosure = {**disclosure, "record": "record.json", "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                  "layout_portability": {**disclosure["layout_portability"], "evaluator_dependencies": [
                      {**item, "sha256": fixture} for item in row["evaluator_asset_dependencies"]]}}
    checked = verify_contract_dependencies(disclosure, project_root=tmp_path, source_root=source, asset_root=assets)
    assert checked["checked_dependencies"] == 1 + len(record["evaluator_dependencies"]) + 8
    (assets / "Object/RoboDojo/Rigid/factory_nut/00003/object.usdz").write_bytes(b"recoloured")
    with pytest.raises(RuntimeError, match="factory_nut/00003/object.usdz"):
        verify_contract_dependencies(disclosure, project_root=tmp_path, source_root=source, asset_root=assets)


def test_an_unaudited_task_layout_or_byte_is_refused_its_contract():
    with pytest.raises(RuntimeError, match="fixture_task is not audited for layout portability"):
        portable_layout("fixture_task", 1, project_root=ROOT)
    # sweep_blocks layout 1 holds five blocks against its contract's three: not listed, so refused.
    with pytest.raises(RuntimeError, match="layout 1 of sweep_blocks is not an audited portable layout"):
        _resolve(_contract_runtime("sweep_blocks", 1))
    with pytest.raises(RuntimeError, match="layout 10 of stack_bowls is not an audited portable layout"):
        _resolve(_contract_runtime("stack_bowls", 10))
    with pytest.raises(RuntimeError, match="differs from its audited file or bytes"):
        _resolve(_contract_runtime("stack_bowls", 2, layout_sha256="f" * 64))
    with pytest.raises(RuntimeError, match="differs from its audited file or bytes"):
        _resolve(_contract_runtime("stack_bowls", 2, layout_file=layout_file_for("stack_bowls", 0, 3)))
    # Layout 0 keeps its reviewed binding: other bytes are refused as before.
    with pytest.raises(RuntimeError, match="differs from runtime layout_sha256"):
        _resolve(_contract_runtime("stack_bowls", 0, layout_sha256="f" * 64))


# --- RunSpec composition -----------------------------------------------------------------------------------

def _spec(task, **kwargs):
    kwargs.setdefault("gpu", "0")
    kwargs.setdefault("infrastructure_qualification", QUALIFICATION)
    return robodojo_spec.build_spec(task, experiment=EXPERIMENT, **kwargs)


@needs_assets
def test_a_fixed_instruction_task_composes_its_layout_with_the_record_instruction():
    record = robodojo_spec.task_contract("stack_bowls")["record"]
    assert _spec("stack_bowls", layout_id=0) == _spec("stack_bowls")
    spec = _spec("stack_bowls", layout_id=3)
    runtime, row = spec["runtime"], TABLE["tasks"]["stack_bowls"]["layouts"][3]
    assert spec["config"] == "stack_bowls-l3-s0" and "session" not in spec
    assert {key: runtime[key] for key in ("layout_id", "layout_file", "layout_sha256")} == row
    assert runtime["expected_task_language"] == spec["task_instruction"] == record["instruction"]["text"]
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["layout_id"] == 3
    assert plan["policy"]["task_contract"]["layout_portability"]["layout_id"] == 3
    qualification = robodojo_spec.qualification_runtime("stack_bowls", gpu="0", layout_id=3)
    assert qualification["layout_id"] == 3
    assert qualification["expected_task_language"] == record["instruction"]["text"]


#: The *_random variants registry-v10 contracts, each on its own <task>_random_<k>.json layouts.
RANDOM_VARIANTS = sorted(task for task in REGISTRY if task.endswith("_random"))


@needs_assets
@pytest.mark.parametrize("layout_id", [3, 4])
@pytest.mark.parametrize("task", RANDOM_VARIANTS)
def test_a_random_variant_composes_layouts_3_and_4_with_its_record_instruction(task, layout_id):
    record = robodojo_spec.task_contract(task)["record"]
    spec = _spec(task, layout_id=layout_id)
    runtime = spec["runtime"]
    row = next(row for row in TABLE["tasks"][task]["layouts"] if row["layout_id"] == layout_id)
    identity = ("layout_id", "layout_file", "layout_sha256")
    assert spec["config"] == f"{task}-l{layout_id}-s0" and "session" not in spec
    assert {key: runtime[key] for key in identity} == {key: row[key] for key in identity}
    assert row["layout_file"] == f"Eval_Layout/RoboDojo/arx_x5/0/{task}_{layout_id}.json"
    assert runtime["expected_task_language"] == spec["task_instruction"] == record["instruction"]["text"]
    assert runtime["native_action_limit"] == record["native_action_limit"] == robodojo.NATIVE_ACTION_LIMITS[task]
    plan = resolve_run_spec(spec, root=ROOT)
    contract = plan["policy"]["task_contract"]
    assert plan["runtime"]["layout_id"] == layout_id and contract["delivered"] is True
    assert contract["layout_portability"]["layout_id"] == layout_id
    assert contract["record"] == REGISTRY[task]["record"]
    # A row that binds evaluator assets only its layout holds hands them to fresh preflight.
    assert contract["layout_portability"].get("evaluator_dependencies", []) == row.get(
        "evaluator_asset_dependencies", [])


@pytest.mark.parametrize("layout_id", [3, 4])
@pytest.mark.parametrize("task", RANDOM_VARIANTS)
def test_a_random_variant_spec_resolves_its_contract_on_layouts_3_and_4_without_assets(monkeypatch, task, layout_id):
    """The same composition with the layout bytes taken from the reviewed table, so no Assets are needed."""
    from tests.support.robodojo.admission import _runtime
    from ora.environment.benchmarks.robodojo_physics import physics_device_for

    def runtime_configuration(gpu, selected, ordinal=0):
        runtime = _runtime()
        runtime.pop("infrastructure_qualification")
        runtime.pop("expected_task_language")
        runtime["public_environment_context"] = robodojo.public_context()
        row = next(row for row in TABLE["tasks"][selected]["layouts"] if row["layout_id"] == ordinal)
        runtime.update({k: row[k] for k in ("layout_id", "layout_file", "layout_sha256")}, task_variant=selected,
                       cuda_visible_devices=gpu, renderer_gpu=int(gpu),
                       native_action_limit=robodojo.NATIVE_ACTION_LIMITS[selected],
                       physics_device=physics_device_for(selected))
        return runtime

    monkeypatch.setattr(robodojo_spec, "runtime_configuration", runtime_configuration)
    spec = robodojo_spec.build_spec(task, experiment=EXPERIMENT, gpu="0", layout_id=layout_id)
    plan = resolve_run_spec(spec, root=ROOT)
    record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
    assert plan["runtime"]["layout_id"] == layout_id
    assert plan["runtime"]["expected_task_language"] == record["instruction"]["text"]
    contract = plan["policy"]["task_contract"]
    assert contract["delivered"] is True and contract["record"] == REGISTRY[task]["record"]
    assert contract["contract_sha256"] == record["contract_layer"]["sha256"]
    assert contract["layout_portability"]["layout_id"] == layout_id


@needs_assets
def test_a_single_layout_spec_refuses_what_its_layout_cannot_carry():
    with pytest.raises(ValueError, match="needs the scene-smoke lane gate of that layout"):
        _spec("stack_bowls", layout_id=3, infrastructure_qualification={"card": QUALIFICATION,
                                                                        "task": QUALIFICATION})
    with pytest.raises(ValueError, match="one round"):
        _spec("stack_bowls", layout_id=3, rounds=3)
    with pytest.raises(ValueError, match="layout 1 of sweep_blocks is not an audited portable layout"):
        _spec("sweep_blocks", layout_id=1)
    with pytest.raises(ValueError, match="not an audited portable layout"):
        _spec("stack_bowls", layout_id=10)
    with pytest.raises(ValueError, match="non-negative official ordinal"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", layout_id=-2)
    with pytest.raises(ValueError, match="renders its reviewed instruction on every layout"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", layout_id=3, task_instruction="Another.")


def _gate(tmp_path, monkeypatch, *, task, layout_id, instruction):
    monkeypatch.setattr(robodojo_spec, "WORKSPACE_ROOT", tmp_path)
    record = tmp_path / f"outputs/{EXPERIMENT}/qualification/revalidation/{task}-{layout_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"schema": rv.SCHEMA, "task_instruction": instruction,
                                  "configuration": {"task_variant": task, "layout_id": layout_id}}))
    return {"path": record.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(record.read_bytes()).hexdigest()}


@needs_assets
def test_a_layout_generated_instruction_comes_from_the_lane_gate_readback(tmp_path, monkeypatch):
    qualification = robodojo_spec.qualification_runtime("general_pickup", gpu="0", layout_id=2)
    assert qualification["layout_id"] == 2 and "expected_task_language" not in qualification
    instruction = "Pick up the pearl seashell by 10 cm."
    gate = _gate(tmp_path, monkeypatch, task="general_pickup", layout_id=2, instruction=instruction)
    spec = _spec("general_pickup", layout_id=2, infrastructure_qualification=gate)
    assert spec["runtime"]["expected_task_language"] == spec["task_instruction"] == instruction
    assert spec["config"] == "general_pickup-l2-s0"
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["policy"]["task_contract"]["layout_portability"]["instruction_source"] == "layout_readback"
    other = _gate(tmp_path, monkeypatch, task="general_pickup", layout_id=3, instruction=instruction)
    with pytest.raises(ValueError, match="binds no scene-smoke instruction of layout 2"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification=other)
    with pytest.raises(ValueError, match="unreadable"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification={**gate, "sha256": "1" * 64})
    with pytest.raises(ValueError, match="not a card/task pair"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification={"card": gate, "task": gate})


def test_the_spec_entry_point_passes_the_single_layout(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(robodojo_spec, "build_spec", lambda task, **kwargs: seen.update(kwargs) or {"spec": task})
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"path": "outputs/x.json", "sha256": "0" * 64}))
    assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0", "--layout", "3",
                               "--qualification-reference", str(gate)]) == 0
    assert seen["layout_id"] == 3 and seen["rounds"] is None and seen["layout_schedule"] is None


# --- revalidation: the readback binding ------------------------------------------------------------------

def test_a_plan_is_held_to_the_instruction_its_smoke_bound_or_read_back():
    bound, unbound = {"expected_task_language": "Stack."}, {}
    rv.scene_language({"expected_task_language": "Stack."}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="drift"):
        rv.scene_language({"expected_task_language": "Other."}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="drift"):
        rv.scene_language({}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="language mismatch"):
        rv.scene_language({"expected_task_language": "Stack."}, bound, "Read back otherwise.")
    # An unbound smoke certifies its readback, and admits only a plan that carries exactly it.
    rv.scene_language({}, unbound, "Pick up the scissors by 10 cm.")
    rv.scene_language({"expected_task_language": "Pick up the scissors by 10 cm."}, unbound,
                      "Pick up the scissors by 10 cm.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="language mismatch"):
        rv.scene_language({"expected_task_language": "Pick up the shovel by 10 cm."}, unbound,
                          "Pick up the scissors by 10 cm.")


@pytest.mark.parametrize("task", ["fasten_screws", "general_pickup"])
@pytest.mark.parametrize("layout_id", range(10))
def test_ungated_layouts_compose_and_resolve_without_qualification(monkeypatch, task, layout_id):
    from tests.support.robodojo.admission import _runtime
    from ora.environment.benchmarks.robodojo_physics import physics_device_for

    def runtime_configuration(gpu, selected, ordinal=0):
        runtime = _runtime()
        runtime.pop("infrastructure_qualification")
        runtime.pop("expected_task_language")
        runtime["public_environment_context"] = robodojo.public_context()
        row = next(row for row in TABLE["tasks"][selected]["layouts"] if row["layout_id"] == ordinal)
        runtime.update({k: row[k] for k in ("layout_id", "layout_file", "layout_sha256")}, task_variant=selected, cuda_visible_devices=gpu, renderer_gpu=int(gpu),
                       native_action_limit=robodojo.NATIVE_ACTION_LIMITS[selected],
                       physics_device=physics_device_for(selected))
        return runtime

    monkeypatch.setattr(robodojo_spec, "runtime_configuration", runtime_configuration)
    spec = robodojo_spec.build_spec(task, experiment=EXPERIMENT, gpu="0", layout_id=layout_id)
    assert "infrastructure_qualification" not in spec["runtime"]
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["layout_id"] == layout_id
    if task == "general_pickup" and layout_id > 0:
        assert "expected_task_language" not in plan["runtime"]
        assert plan["runtime"]["task_language_source"] == "live_reset_readback"
    else:
        assert plan["runtime"]["expected_task_language"] == spec["task_instruction"]
