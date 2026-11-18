"""ORA's own RoboTwin play scenes (``ora play --scene-set``): registry, audit record, admission and plan.

Offline: the play-scene audits are fixtures in the shape
``backends.robotwin.play_audit.build_record`` writes; no simulator, Provider or
resident service runs. The scenes' own build, settle and render evidence comes
from the audit worker on a GPU host (``backends/robotwin/play_scene_audit.py``).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from ora.environment.backends.robotwin import play_scene_audit
from ora.environment.benchmarks import robotwin_play_scenes as scene_sets
from ora.environment.backends.robotwin.common import PLAY_SIGNAL_POLICY
from ora.environment.benchmarks.robotwin import TASKS
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.backends.robotwin import play_audit as robotwin_play_audit
from ora.app.main import build_parser
from ora.app.play import play_spec
from ora.orchestrator.play.binding import bind_play_runtime
from ora.orchestrator.play.scenes import PlaySceneError, build_play_spec, list_play_scenes
from ora.orchestrator.backends.robotwin.play_audit import build_record, play_scene_admission
from ora.orchestrator.backends.robotwin.preflight import task_module
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.benchmark_specs import REVISION
from tests.support.orchestrator.play_spec import POLICY

ROOT = Path(__file__).resolve().parents[5]
SET = "tabletop-v1"
SCENES = ("table_setting_sequence", "shell_game_memory_3", "obstacle_rearrangement_6")
HOUSEHOLD = "household-v1"


def _seed(passed=True):
    return {"passed": passed, "built": True, "failures": [] if passed else ["cup:at_rest"], "object_count": 6,
            "cameras": {"head_camera": {"path": "head_camera.png", "sha256": "0" * 64, "std": 40.0}}}


def _record(scene="table_setting_sequence", seeds=(0, 1, 2), failed=(), revision=REVISION, **changes):
    summary = {"scene_source": scene_sets.source_binding(SET), "criteria": play_scene_audit.CRITERIA,
               "robotwin_sources": []}
    record = build_record(scene_set=SET, scene=scene, task_config="demo_clean", revision=revision,
                          summary=summary, seed_records={seed: _seed(seed not in failed) for seed in seeds},
                          source_commit="c" * 40, execution={"required_host": "test-host"}, created_unix_s=1.0)
    record.update(changes)
    return record


def _write(root: Path, stamp: str, record: dict) -> Path:
    path = root / stamp / record["scene_set"] / f"{record['scene']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def _audits(root: Path) -> Path:
    """Two audits: the newer one binds seed 1; failed, other-revision and tampered records admit nothing."""
    _write(root, "20261018T100000Z-1", _record(seeds=(0, 1, 2), failed=(2,)))
    _write(root, "20261018T110000Z-2", _record(seeds=(1, 3)))
    _write(root, "20261018T110000Z-2", _record("shell_game_memory_3", seeds=(5,)))
    _write(root, "20261018T110000Z-2", _record("obstacle_rearrangement_6", seeds=(7,), revision="0" * 40))
    tampered = _record("obstacle_rearrangement_6", seeds=(8,))
    tampered["accepted_seeds"] = [8, 9]
    _write(root, "20261018T120000Z-3", tampered)
    return root


# --- the registry --------------------------------------------------------------------------------

def test_the_tabletop_set_names_three_library_scenes_that_are_no_official_task():
    assert scene_sets.scene_sets() == (HOUSEHOLD, SET)
    definitions = scene_sets.scene_definitions(SET)
    assert tuple(item.name for item in definitions) == SCENES
    assert all(item.task_config == "demo_clean" and item.name not in TASKS for item in definitions)
    assert "034_knife/base0" in definitions[0].assets and "027_table-tennis/base1" in definitions[1].assets
    assert scene_sets.find_scene("shell_game_memory_3").scene_set == SET and scene_sets.find_scene("click_bell") is None
    with pytest.raises(scene_sets.PlaySceneSetError, match="unknown play scene set"):
        scene_sets.scene_definitions("tabletop-v0")
    with pytest.raises(scene_sets.PlaySceneSetError, match="has no scene"):
        scene_sets.scene_definition(SET, "click_bell")
    assert [path.relative_to(ROOT).as_posix() for path in scene_sets.source_files(SET)] == [
        "ora/environment/backends/robotwin/play_scenes/base.py",
        "ora/environment/backends/robotwin/play_scenes/tabletop_v1.py"]


def test_the_household_set_is_a_breakfast_table_and_a_kitchen_counter_bound_apart_from_tabletop(tmp_path):
    definitions = scene_sets.scene_definitions(HOUSEHOLD)
    assert tuple(item.name for item in definitions) == ("breakfast", "kitchen")
    assert all(item.task_config == "demo_clean" and item.name not in TASKS for item in definitions)
    breakfast, kitchen = definitions
    assert len(breakfast.assets) == 9 and len(kitchen.assets) == 8
    assert {"003_plate/base0", "076_breadbasket/base0", "002_bowl/base7"} <= set(breakfast.assets)
    assert {"104_board/base0", "067_steamer/base1", "062_plasticbox/base3"} <= set(kitchen.assets)
    # Nothing thin or flat to pinch: no knife, fork or spatula.
    assert not any(asset.startswith(("033_fork", "034_knife", "082_smallshovel"))
                   for item in definitions for asset in item.assets)
    assert scene_sets.find_scene("kitchen").scene_set == HOUSEHOLD
    assert [path.relative_to(ROOT).as_posix() for path in scene_sets.source_files(HOUSEHOLD)] == [
        "ora/environment/backends/robotwin/play_scenes/base.py",
        "ora/environment/backends/robotwin/play_scenes/household_v1.py"]
    # A household edit changes the household binding and leaves tabletop-v1's (and its audits) as they were.
    for path in {*scene_sets.source_files(SET), *scene_sets.source_files(HOUSEHOLD)}:
        copy = tmp_path / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    tabletop, household = scene_sets.source_binding(SET), scene_sets.source_binding(HOUSEHOLD)
    (tmp_path / "ora/environment/backends/robotwin/play_scenes/household_v1.py").write_text("# changed\n")
    assert scene_sets.source_binding(SET, root=tmp_path) == tabletop
    assert scene_sets.source_binding(HOUSEHOLD, root=tmp_path)["sha256"] != household["sha256"]


def test_the_source_binding_follows_the_scene_files_and_the_registry_is_host_safe(tmp_path):
    binding = scene_sets.source_binding(SET)
    assert binding == scene_sets.source_binding(SET) and binding["scene_set"] == SET
    for path in scene_sets.source_files(SET):
        copy = tmp_path / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    assert scene_sets.source_binding(SET, root=tmp_path) == binding
    (tmp_path / "ora/environment/backends/robotwin/play_scenes/tabletop_v1.py").write_text("# changed\n")
    assert scene_sets.source_binding(SET, root=tmp_path)["sha256"] != binding["sha256"]
    # The registry, the audit driver and play.scenes never import the simulator on the host.
    probe = ("import sys; import ora.orchestrator.play.scenes, ora.orchestrator.backends.robotwin.play_audit, "
             "ora.environment.benchmarks.robotwin_play_scenes, ora.environment.backends.robotwin.play_scene_audit; "
             "assert not {'sapien', 'envs', 'torch'} & set(sys.modules), sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)


# --- the audit record and its admission ---------------------------------------------------------

def test_an_audit_admits_its_accepted_seeds_for_the_current_sources_only(tmp_path):
    record = _record(failed=(2,))
    assert record["accepted_seeds"] == [0, 1] and record["verdict"] is None and record["expert_called"] is False
    assert record["seeds"]["0"]["cameras"]["head_camera"]["path"] == "table_setting_sequence/seed-0/cameras/head_camera.png"
    admit = dict(scene_set=SET, scene="table_setting_sequence", task_config="demo_clean", revision=REVISION)
    assert play_scene_admission(record, seed=0, **admit)[0]
    passed, detail = play_scene_admission(record, seed=2, **admit)
    assert not passed and detail["seed"]["failures"] == ["cup:at_rest"]
    assert not play_scene_admission(record, seed=5, **admit)[0]
    for change in ({"scene": "shell_game_memory_3"}, {"scene_set": None}, {"task_config": "demo_randomized"},
                   {"revision": "0" * 40}):
        assert not play_scene_admission(record, seed=0, **{**admit, **change})[0]
    tampered = {**record, "accepted_seeds": [0, 1, 2]}
    assert play_scene_admission(tampered, seed=2, **admit)[1]["audit_sha256_valid"] is False
    # A changed scene module needs a new audit.
    for path in scene_sets.source_files(SET):
        copy = tmp_path / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes() + b"# edited\n")
    passed, detail = play_scene_admission(record, seed=0, root=tmp_path, **admit)
    assert not passed and detail["scene_source_sha256"] != detail["current_scene_source_sha256"]
    # The official seed-manifest rule admits nothing an audit record carries.
    from ora.orchestrator.backends.robotwin.seed_manifest import seed_admission
    assert not seed_admission(record, task="table_setting_sequence", seed=0, task_config="demo_clean",
                              revision=REVISION)[0]


def test_the_audit_driver_runs_one_worker_per_scene_and_writes_its_records(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(robotwin_play_audit, "require_sources", lambda *args, **kwargs: "c" * 40)
    commands = []

    def runner(command, *, cwd, env, log, timeout):
        commands.append((command, env["CUDA_VISIBLE_DEVICES"], timeout))
        output = Path(command[command.index("--output-root") + 1])
        seeds = [int(value) for value in command[command.index("--seeds") + 1:]]
        for seed in seeds:
            (output / f"seed-{seed}").mkdir(parents=True)
            (output / f"seed-{seed}" / "record.json").write_text(json.dumps(_seed(seed != 1)))
        (output / "summary.json").write_text(json.dumps({
            "scene_source": scene_sets.source_binding(SET), "criteria": play_scene_audit.CRITERIA}))
        log.write_text("ok")
        return 0

    released = []
    report = robotwin_play_audit.audit(SET, ["table_setting_sequence", "shell_game_memory_3"], [0, 1],
                                       robotwin_root=tmp_path / "RoboTwin", robotwin_python=tmp_path / "python",
                                       resource_profile="robotwin-gpu3", destination=tmp_path / "audit",
                                       runner=runner, lease=lambda binding, scene_set: lambda: released.append(1))
    assert [command[command.index("--scene") + 1] for command, _, _ in commands] == [
        "table_setting_sequence", "shell_game_memory_3"]
    assert {device for _, device, _ in commands} == {"3"} and released == [1]
    written = json.loads((tmp_path / "audit" / SET / "shell_game_memory_3.json").read_text())
    assert written["accepted_seeds"] == [0] and written["execution"]["resource_profile"] == "robotwin-gpu3"
    assert report["records"]["shell_game_memory_3"]["rejected"] == {"1": ["cup:at_rest"]}
    assert play_scene_admission(written, scene_set=SET, scene="shell_game_memory_3", seed=0,
                                task_config="demo_clean", revision=REVISION)[0]


# --- what a host offers --------------------------------------------------------------------------

def test_a_scene_set_offers_only_the_seeds_an_audit_admits(tmp_path):
    root = _audits(tmp_path / "audits")
    scenes = list_play_scenes("robotwin", scene_set=SET, play_audit_root_dir=root)
    assert [(scene.task, scene.seed) for scene in scenes] == [
        ("shell_game_memory_3", 5), ("table_setting_sequence", 0), ("table_setting_sequence", 1),
        ("table_setting_sequence", 3)]
    newer = next(scene for scene in scenes if scene.seed == 1)
    assert newer.qualification["path"].startswith("20261018T110000Z-2/") and newer.scene_set == SET
    assert newer.qualification["kind"] == "robotwin_play_scene_audit"
    record = newer.record()
    assert record["scene_set"] == SET and record["task_config"] == "demo_clean"
    assert record["scene_id"] == "robotwin:tabletop-v1:table_setting_sequence:s1"
    assert record["label"] == "RoboTwin tabletop-v1 · table setting · seed 1"
    assert [scene.seed for scene in list_play_scenes("robotwin", scene_set=SET, task="shell_game_memory_3",
                                                     play_audit_root_dir=root)] == [5]
    # Without a scene set nothing changes: the official tasks' seed manifests, not the audits.
    assert list_play_scenes("robotwin", robotwin_workspace_root=tmp_path / "scans") == []
    with pytest.raises(PlaySceneError, match="scene sets are RoboTwin's"):
        list_play_scenes("libero", scene_set=SET)
    with pytest.raises(PlaySceneError, match="unknown play scene set"):
        list_play_scenes("robotwin", scene_set="tabletop-v0", play_audit_root_dir=root)
    with pytest.raises(PlaySceneError, match="ora play audit --env robotwin --scene-set tabletop-v1"):
        play_scenes.draw_play_scene("robotwin", random.Random(0), scene_set=SET,
                                    play_audit_root_dir=tmp_path / "none")
    draws = {play_scenes.draw_play_scene("robotwin", random.Random(seed), scene_set=SET,
                                         play_audit_root_dir=root).task for seed in range(20)}
    assert draws == {"shell_game_memory_3", "table_setting_sequence"}


def test_build_play_spec_names_the_audit_record_and_binds_the_set_only_at_resolution(tmp_path):
    root = _audits(tmp_path / "audits")
    spec = build_play_spec("robotwin", host="test-host", rng=random.Random(2), scene_set=SET,
                           play_audit_root_dir=root, resource_profile="robotwin-gpu0")
    scene = spec["play"]["scene"]
    assert scene["scene_set"] == SET and spec["task"] == scene["task"]
    # ORA's own scenes record their operator camera at the capture's highest rate.
    assert spec["runtime"] == {"kind": "robotwin", "required_host": "test-host", "resource_profile": "robotwin-gpu0",
                               "task": scene["task"], "task_config": "demo_clean",
                               "execution_capture": {"enabled": True, "fps": 30}}
    assert spec["qualification"]["manifest"].endswith(f"-{scene['qualification']['sha256'][:16]}.json")


def test_bind_play_runtime_binds_the_scene_set_and_no_benchmark_may_carry_it():
    runtime = {"kind": "robotwin", "task": "shell_game_memory_3", "task_config": "demo_clean",
               "environment_signal_policy": "lifecycle_only"}
    scene = {"environment": "robotwin", "task": "shell_game_memory_3", "task_config": "demo_clean", "seed": 5,
             "scene_set": SET, "qualification": {"kind": "robotwin_play_scene_audit"}}
    bound = bind_play_runtime(runtime, scene=scene)
    assert bound == {**runtime, "environment_signal_policy": PLAY_SIGNAL_POLICY, "play_scene_set": SET}
    assert bind_play_runtime(bound, scene=scene) == bound
    with pytest.raises(RunSpecError, match="play_scene_set loads a play scene"):
        bind_play_runtime({**runtime, "play_scene_set": SET}, scene=None)
    official = {**scene, "task": "click_bell", "scene_set": None}
    with pytest.raises(RunSpecError, match="must be play.scene.scene_set"):
        bind_play_runtime({**runtime, "task": "click_bell", "play_scene_set": SET}, scene=official)
    with pytest.raises(RunSpecError, match="has no scene"):
        bind_play_runtime({**runtime, "task": "click_bell"}, scene={**official, "scene_set": SET})
    with pytest.raises(RunSpecError, match="loads in scene configuration demo_clean"):
        bind_play_runtime({**runtime, "task_config": "demo_randomized"},
                          scene={**scene, "task_config": "demo_randomized"})
    with pytest.raises(RunSpecError, match="names a RoboTwin scene set"):
        bind_play_runtime({"kind": "libero_pro", "suite": "libero_goal", "task_id": 0},
                          scene={"environment": "libero", "task": "libero_goal-t00", "suite": "libero_goal",
                                 "task_id": 0, "seed": 0, "scene_set": SET, "qualification": {}})


# --- ora play --scene-set: the plan and its launch preflight --------------------------------------

def test_ora_play_draws_a_scene_of_the_set_and_preflight_readmits_its_seeds(tmp_path, monkeypatch, workspace):
    from ora.orchestrator.preflight import gates as preflight

    root = _audits(tmp_path / "audits")
    monkeypatch.setattr(play_scenes, "play_audit_root", lambda: root)
    monkeypatch.setattr(preflight, "resolve_workspace_root", lambda project_root: workspace)
    from ora.orchestrator.spec.inputs import provider_policy

    spec, chosen = play_spec("robotwin", policy=provider_policy(None), host="test-host", count=4,
                             rng=random.Random(3), draw={"draw_seed": 3},
                             scene="table_setting_sequence")  # the scene implies its set
    assert chosen.scene_set == SET and chosen.task == "table_setting_sequence"
    assert {scene["seed"] for scene in spec["play"]["scenes"]} <= {0, 1, 3}
    assert all(scene["scene_set"] == SET for scene in spec["play"]["scenes"])
    play_scenes.write_play_spec(spec, chosen, output=workspace / "play.json")
    plan = resolve_run_spec(spec)
    assert plan["runtime"]["play_scene_set"] == SET and plan["runtime"]["task_config"] == "demo_clean"
    assert plan["runtime"]["environment_signal_policy"] == PLAY_SIGNAL_POLICY
    assert plan["play"]["scene"]["scene_set"] == SET and plan["policy"]["task_contract"]["delivered"] is False
    check = preflight._qualification(plan)
    assert check["passed"] is True and check["detail"]["accepted"] is True
    assert task_module(plan["runtime"], plan["task"]) == "ora.environment.backends.robotwin.play_scenes.tabletop_v1"
    assert task_module({"kind": "robotwin"}, "click_bell") == "envs.click_bell"
    with pytest.raises(ValueError, match="use it with --env robotwin"):
        play_spec("libero", policy=POLICY, host="test-host", count=1, rng=random.Random(0), draw={},
                  scene_set=SET)
    args = build_parser().parse_args(["play", "--env", "robotwin", "--scene-set", SET, "--scene", "shell_game_memory_3"])
    assert (args.scene_set, args.scene) == (SET, "shell_game_memory_3")
    # A changed scene module fails preflight: the plan's audit no longer admits its seeds.
    monkeypatch.setattr(robotwin_play_audit, "source_binding", lambda scene_set, root: {"sha256": "f" * 64})
    assert preflight._qualification(plan)["passed"] is False


def test_the_scenes_of_one_session_share_their_scene_set(workspace):
    from ora.orchestrator.spec.play import _require_backend_scenes

    first = {"seed": 0, "task_config": "demo_clean", "scene_set": SET}
    _require_backend_scenes({"kind": "robotwin"}, [first, {**first, "seed": 1}])
    with pytest.raises(RunSpecError, match="share one scene configuration and scene set"):
        _require_backend_scenes({"kind": "robotwin"}, [first, {**first, "seed": 1, "scene_set": None}])


def test_the_audit_worker_reports_contacts_by_object_and_deepest_separation():
    from types import SimpleNamespace as N

    def body(name):
        return N(entity=N(get_name=lambda: name))

    contacts = [N(bodies=[body("cup"), body("table")], points=[N(separation=-0.0004), N(separation=0.0)]),
                N(bodies=[body("table"), body("cup")], points=[N(separation=-0.001)]),
                N(bodies=[body("cup"), body("fl_link6")], points=[N(separation=-0.003)]),
                N(bodies=[body("table"), body("ground")], points=[N(separation=0.0)])]
    environment = N(scene=N(get_contacts=lambda: contacts))
    report = play_scene_audit.contact_report(environment, {"cup": object()})
    assert report == [{"bodies": ["cup", "robot"], "min_separation_m": -0.003},
                      {"bodies": ["cup", "table"], "min_separation_m": -0.001}]
    assert round(play_scene_audit._rotation_deg([1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]), 3) == 90.0
