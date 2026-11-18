"""Play scenes: a qualified benchmark scene, loaded without its task (ora/orchestrator/play/scenes.py).

Offline: the LIBERO init-state files, the RoboTwin seed manifests and the
execution host are fixtures; no simulator, Provider or resident service runs.
``spec/play.py`` (ora play) resolves the play block and calls
``bind_play_runtime``; its plan and episode tests live with it.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from ora.environment.backends.libero_pro.direct import PLAY_DRIVER
from ora.environment.backends.robotwin.common import PLAY_SIGNAL_POLICY
from ora.environment.benchmarks.libero_pro_language import TASK_ROWS, base_task_language
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.backends.robotwin import play as robotwin_play
from ora.orchestrator.backends.libero_pro.configuration import validate_libero_configuration
from ora.orchestrator.play.binding import bind_play_runtime, verify_play_qualification
from ora.orchestrator.play.scenes import (
    PlaySceneError, build_play_spec, draw_play_scene, list_play_scenes, write_play_spec,
)
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.spec.runtime import _resolve_transport_and_signals
from tests.support.orchestrator.benchmark_specs import ACTIVATION, _official_manifest, _readiness_report
from tests.support.orchestrator.run_specs import native_spec

FILENAMES = {(family, task_id): name for family, task_id, name, *_ in TASK_ROWS}


def _libero_checkout(root: Path, cells=(("goal", 0), ("goal", 7), ("object", 2), ("spatial", 0))) -> Path:
    for family, task_id in cells:
        path = root / "libero/libero/init_files" / f"libero_{family}" / f"{FILENAMES[(family, task_id)]}.pruned_init"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{family}-{task_id}".encode())
    return root


def _scan(workspace_root: Path, scan: str, name: str, manifest: Path) -> Path:
    target = workspace_root / "qualification" / scan / "seed-manifests" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(manifest.read_bytes())
    return target


def _robotwin_records(root: Path) -> Path:
    """Two admitting scans, and manifests that admit nothing play may load."""
    scans = root / "robotwin-evaluation"
    made = root / "made"
    _scan(scans, "20261001T000000Z-a", "click_bell.json",
          _official_manifest(made / "a", "click_bell", "demo_clean", (100000, 100001)))
    _scan(scans, "20261002T000000Z-b", "click_bell@100050.json",
          _official_manifest(made / "b", "click_bell", "demo_clean", (100001, 100060)))
    _scan(scans, "20261002T000000Z-b", "adjust_bottle.json",
          _official_manifest(made / "c", "adjust_bottle", "demo_randomized", (100002,)))
    _scan(scans, "20261002T000000Z-b", "lift_pot.json",  # another revision
          _official_manifest(made / "d", "lift_pot", "demo_clean", (7,), revision="0" * 40))
    _scan(scans, "20261002T000000Z-b", "hanging_mug.json",  # its digest no longer matches
          _official_manifest(made / "e", "hanging_mug", "demo_clean", (8,), manifest_sha256="0" * 64))
    rejected = _official_manifest(made / "f", "beat_block_hammer", "demo_clean", (9,))
    body = json.loads(rejected.read_text())
    body["seeds"]["9"]["oracle_success"] = False
    from ora.orchestrator.evaluation.evidence_io import digest
    body["manifest_sha256"] = digest({key: value for key, value in body.items() if key != "manifest_sha256"})
    rejected.write_text(json.dumps(body))
    _scan(scans, "20261002T000000Z-b", "beat_block_hammer.json", rejected)
    return scans


# --- what a host offers --------------------------------------------------------------------------

def test_libero_scenes_are_the_init_states_of_local_records_in_the_play_suites(tmp_path):
    checkout = _libero_checkout(tmp_path / "LIBERO-PRO")
    scenes = list_play_scenes("libero", libero_checkout=checkout)
    assert sorted({scene.task for scene in scenes}) == ["libero_goal-t00", "libero_goal-t07", "libero_object-t02"]
    assert len(scenes) == 3 * 50 and {scene.seed for scene in scenes} == set(range(50))
    goal = next(scene for scene in scenes if scene.task == "libero_goal-t07")
    path = checkout / goal.qualification["path"]
    assert goal.qualification["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert goal.record()["qualification"]["kind"] == "libero_init_states" and "kitchen table" in goal.label
    assert len(list_play_scenes("libero", task="libero_goal-t07", libero_checkout=checkout)) == 50
    assert list_play_scenes("libero", environ={}) == []  # no LIBERO-Pro checkout on this host
    with pytest.raises(PlaySceneError, match="LIBERO_PRO_ROOT"):
        draw_play_scene("libero", random.Random(0), environ={})


def test_robotwin_scenes_are_only_seeds_a_local_manifest_admits_at_the_pinned_revision(tmp_path):
    scans = _robotwin_records(tmp_path)
    scenes = list_play_scenes("robotwin", robotwin_workspace_root=scans)
    assert [(scene.task, scene.task_config, scene.seed) for scene in scenes] == [
        ("adjust_bottle", "demo_randomized", 100002),
        ("click_bell", "demo_clean", 100000), ("click_bell", "demo_clean", 100001),
        ("click_bell", "demo_clean", 100060)]
    shared = next(scene for scene in scenes if scene.seed == 100001)
    assert shared.qualification["path"].startswith("qualification/20261002T")  # the newest scan binds it
    assert shared.qualification["sha256"] == hashlib.sha256(shared.source.read_bytes()).hexdigest()
    assert [scene.seed for scene in list_play_scenes(
        "robotwin", task="adjust_bottle", robotwin_workspace_root=scans)] == [100002]
    assert list_play_scenes("robotwin", task="click_bell", task_config="demo_randomized",
                            robotwin_workspace_root=scans) == []
    assert robotwin_play.robotwin_tasks(scans) == [
        "adjust_bottle", "beat_block_hammer", "click_bell", "hanging_mug", "lift_pot"]
    with pytest.raises(PlaySceneError, match="no local RoboTwin seed manifest"):
        draw_play_scene("robotwin", random.Random(0), robotwin_workspace_root=tmp_path / "empty")


def test_unknown_environments_offer_no_scene_and_robodojo_only_its_audited_scene_sets(tmp_path):
    # RoboDojo offers ORA's own scene sets only (test_robodojo_play_scenes.py); without an audit, nothing.
    assert list_play_scenes("robodojo", play_audit_root_dir=tmp_path / "no-audits") == []
    with pytest.raises(PlaySceneError, match="must be one of"):
        list_play_scenes("maniskill")
    assert [scene.scene_id for scene in list_play_scenes("synthetic")] == ["synthetic:synthetic-world:s0"]
    # Synthetic free play draws its world (the quickstart's capture), so the world view has frames.
    synthetic = build_play_spec("synthetic", host=None, rng=random.Random(0))
    assert synthetic["runtime"]["execution_capture"] == {"enabled": True}


def test_a_draw_picks_a_task_first_and_a_seeded_rng_repeats_it(tmp_path):
    scans = _robotwin_records(tmp_path)
    draws = [draw_play_scene("robotwin", random.Random(seed), robotwin_workspace_root=scans) for seed in range(60)]
    assert draws == [draw_play_scene("robotwin", random.Random(seed), robotwin_workspace_root=scans)
                     for seed in range(60)]
    # adjust_bottle has one admitted seed and click_bell three: each family is drawn about half the time.
    assert 18 <= sum(scene.task == "adjust_bottle" for scene in draws) <= 42
    checkout = _libero_checkout(tmp_path / "LIBERO-PRO")
    first = draw_play_scene("libero", random.Random(4), libero_checkout=checkout)
    assert first == draw_play_scene("libero", random.Random(4), libero_checkout=checkout)
    assert len({draw_play_scene("libero", random.Random(seed), libero_checkout=checkout) for seed in range(20)}) > 5


# --- the RunSpec ---------------------------------------------------------------------------------

def test_build_play_spec_is_reproducible_task_free_and_names_no_provider(tmp_path):
    checkout = _libero_checkout(tmp_path / "LIBERO-PRO")
    spec = build_play_spec("libero", host="test-host", rng=random.Random(5), libero_checkout=checkout,
                           resource_profile="libero-gpu4b")
    assert spec == build_play_spec("libero", host="test-host", rng=random.Random(5), libero_checkout=checkout,
                                   resource_profile="libero-gpu4b")
    scene = spec["play"]["scene"]
    assert scene in [item.record() for item in list_play_scenes("libero", libero_checkout=checkout)]
    assert spec["episode_mode"] == "play" and spec["experiment"] == "ora-play"
    assert spec["play"] == {"schema": "ora.play-session.v1", "scene": scene}
    assert {"environment", "task", "seed", "qualification", "scene_id", "label"} <= set(scene)
    assert not {"task_instruction", "policy", "operator_control"} & set(spec)
    assert (spec["task"], spec["seed"]) == (scene["task"], scene["seed"])
    assert spec["runtime"]["suite"] == scene["suite"] and spec["runtime"]["task_id"] == scene["task_id"]
    assert spec["runtime"]["resource_profile"] == "libero-gpu4b"
    # The backend's identity check still compares the scene's language; the policy never receives it.
    assert spec["runtime"]["expected_task_language"] == base_task_language(scene["suite"], scene["task_id"])
    scans = _robotwin_records(tmp_path)
    robotwin = build_play_spec("robotwin", host="test-host", rng=random.Random(5), robotwin_workspace_root=scans,
                               resource_profile="robotwin-gpu4")
    drawn = robotwin["play"]["scene"]
    assert robotwin["runtime"]["task"] == drawn["task"] and robotwin["runtime"]["task_config"] == drawn["task_config"]
    assert robotwin["qualification"]["manifest"].endswith(f"-{drawn['qualification']['sha256'][:16]}.json")
    with pytest.raises(PlaySceneError, match="ora play audit --env robodojo"):
        build_play_spec("robodojo", host="test-host", rng=random.Random(0), play_audit_root_dir=tmp_path / "none")


def _libero_runtime(tmp_path):
    """The resolved runtime of a LIBERO benchmark cell of a play scene's task."""
    checkout = _libero_checkout(tmp_path / "LIBERO-PRO")
    spec = build_play_spec("libero", host="test-host", rng=random.Random(1), libero_checkout=checkout,
                           resource_profile="libero-gpu4b")
    scene = spec["play"]["scene"]
    benchmark = {key: value for key, value in spec.items() if key not in {"episode_mode", "play"}}
    benchmark.update(policy=native_spec("libero_pro")["policy"],
                     task_instruction=base_task_language(scene["suite"], scene["task_id"]))
    plan = resolve_run_spec(benchmark)
    assert "episode_mode" not in plan and "play" not in plan  # the benchmark resolver is untouched
    return plan["runtime"], scene


def test_bind_play_runtime_selects_the_backend_play_binding_after_defaults(tmp_path):
    runtime, scene = _libero_runtime(tmp_path)
    assert runtime["environment_driver"] == "direct_libero_v1"
    assert bind_play_runtime(runtime, scene=None) == runtime  # a benchmark plan is unchanged
    bound = bind_play_runtime(runtime, scene=scene)
    assert bound == {**runtime, **ACTIVATION["play_scene_bindings"]["libero_pro"]}
    assert bound["environment_driver"] == PLAY_DRIVER
    validate_libero_configuration(dict(bound))
    with pytest.raises(RunSpecError, match="needs episode_mode play"):
        bind_play_runtime(bound, scene=None)
    with pytest.raises(RunSpecError, match="a scene of this runtime"):
        bind_play_runtime(runtime, scene={**scene, "task_id": (scene["task_id"] + 1) % 10})
    robotwin = {"kind": "robotwin", "task": "click_bell", "task_config": "demo_clean", "transport": "local_rpc",
                "environment_signal_policy": "lifecycle_only"}
    record = {"environment": "robotwin", "task": "click_bell", "task_config": "demo_clean", "seed": 100000,
              "qualification": {"kind": "robotwin_seed_manifest"}}
    bound = bind_play_runtime(robotwin, scene=record)
    assert bound["environment_signal_policy"] == PLAY_SIGNAL_POLICY == ACTIVATION["play_scene_bindings"][
        "robotwin"]["environment_signal_policy"]
    _resolve_transport_and_signals(dict(bound), task="click_bell")
    with pytest.raises(RunSpecError, match="supported only by robotwin"):
        _resolve_transport_and_signals({"kind": "libero_pro", "environment_signal_policy": PLAY_SIGNAL_POLICY},
                                       task="t")
    with pytest.raises(RunSpecError, match="binds runtime.environment_signal_policy"):
        bind_play_runtime({**robotwin, "environment_signal_policy": "task_recoverability_v1"}, scene=record)
    synthetic = {"kind": "synthetic"}
    assert bind_play_runtime(synthetic, scene=list_play_scenes("synthetic")[0].record()) == synthetic
    with pytest.raises(RunSpecError, match="a scene of this runtime"):
        bind_play_runtime({"kind": "robodojo", "task_variant": "make_toast"}, scene=record)


def test_a_robotwin_play_spec_binds_the_manifest_its_scene_names(workspace, monkeypatch):
    scans = _robotwin_records(workspace)
    scenes = list_play_scenes("robotwin", robotwin_workspace_root=scans)
    scene = next(item for item in scenes if item.task == "adjust_bottle")
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec = build_play_spec("robotwin", host="test-host", rng=random.Random(0), scene=scene,
                           resource_profile="robotwin-gpu4", service_report=report)
    write_play_spec(spec, scene, output=workspace / "play.json")
    assert set(spec["prerequisite"]["services"]) == {"segmentation", "grasp_proposals"}
    imported = workspace / spec["qualification"]["manifest"]
    assert imported.read_bytes() == scene.source.read_bytes()
    bound = {"required": True, "manifest": {"sha256": scene.qualification["sha256"]}}
    verify_play_qualification(spec["play"], bound)
    other = next(item for item in scenes if item.seed == 100000)
    with pytest.raises(RunSpecError, match="seed manifest its play.scene names"):
        verify_play_qualification(spec["play"], {"required": True, "manifest": {
            "sha256": other.qualification["sha256"]}})
