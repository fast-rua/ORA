"""A RoboTwin play scene: the task's scene loads and its official predicate is withheld."""
from __future__ import annotations

import pytest

from ora.environment.backends.robotwin.common import ENVIRONMENT_SIGNAL_POLICIES, PLAY_SIGNAL_POLICY
from ora.environment.backends.robotwin.session import RoboTwinLiveError, withhold_official_predicate
from tests.support.robotwin.correctness_repair import Plant, action, live_session


class ScenePlant(Plant):
    """The task object RoboTwin builds: ``setup_demo`` then its own ``check_success``."""

    def setup_demo(self, **kwargs):
        self.setup = kwargs


def _construct(tmp_path, monkeypatch, plant, *, policy):
    live = live_session(tmp_path, plant)
    live.environment_signal_policy = policy
    live.task_args_factory = lambda **kwargs: {}
    monkeypatch.setattr(live, "_verify_control_profile_sources", lambda args: None)
    live.environment = live._construct_environment(0)
    return live


def test_a_play_scene_never_latches_the_official_predicate_and_persists_a_null_verdict(tmp_path, monkeypatch):
    assert PLAY_SIGNAL_POLICY in ENVIRONMENT_SIGNAL_POLICIES
    plant = ScenePlant(success_step=1)
    live = _construct(tmp_path, monkeypatch, plant, policy=PLAY_SIGNAL_POLICY)
    assert live.play_scene and plant.setup["seed"] == 0
    receipt = live.execute(action())["receipt"]
    assert plant.steps == 300  # the gripper ran past the step its predicate holds at
    assert receipt["terminal_after_action"] is False
    assert plant.eval_success is False and plant.predicate_reads == 0
    assert live.status()["terminal"] is False
    live.abort("operator_quit")
    verdict = live.persist_official_verdict()
    assert verdict["official_success"] is None and live.official_terminal is None
    assert "withheld" in verdict["source"]
    plant.eval_success = True  # a latch the withheld predicate could not have made
    with pytest.raises(RoboTwinLiveError, match="withholds"):
        live.status()


def test_a_benchmark_scene_keeps_its_official_predicate(tmp_path, monkeypatch):
    plant = ScenePlant(success_step=1)
    live = _construct(tmp_path, monkeypatch, plant, policy="lifecycle_only")
    assert not live.play_scene
    live.execute(action())
    assert plant.eval_success is True and live.status()["reason"] == "official_success"
    assert live.persist_official_verdict()["official_success"] is True
    with pytest.raises(RoboTwinLiveError, match="no official predicate"):
        withhold_official_predicate(object())


# --- a scene of ORA's own scene sets (play_scenes) -------------------------------------------------

def _scene_set_session(tmp_path, **kwargs):
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession

    values = dict(robotwin_root=tmp_path, output_root=tmp_path / "native", task="shell_game_memory_3",
                  actual_seed=0, episode=0, task_config="demo_clean", play_scene_set="tabletop-v1",
                  environment_signal_policy=PLAY_SIGNAL_POLICY, task_factory=lambda task: ScenePlant())
    values.update(kwargs)
    return RoboTwinLiveSession(**values)


def test_a_scene_set_scene_loads_only_as_a_play_scene_of_its_set(tmp_path):
    assert _scene_set_session(tmp_path).play_scene
    with pytest.raises(RoboTwinLiveError, match="outside the registered RoboTwin session set"):
        _scene_set_session(tmp_path, play_scene_set=None)
    with pytest.raises(RoboTwinLiveError, match="load only as play scenes"):
        _scene_set_session(tmp_path, environment_signal_policy="lifecycle_only")
    with pytest.raises(RoboTwinLiveError, match="has no scene"):
        _scene_set_session(tmp_path, task="click_bell")
    with pytest.raises(RoboTwinLiveError, match="loads in scene configuration demo_clean"):
        _scene_set_session(tmp_path, task_config="demo_randomized")


def test_the_worker_builds_the_scene_from_ora_and_binds_its_sources(tmp_path, monkeypatch):
    import subprocess

    from ora.environment.backends.robotwin import session as session_module
    from ora.environment.benchmarks import robotwin_play_scenes as play_scenes

    root = tmp_path / "RoboTwin"
    for relative in ("envs/_base_task.py", "envs/camera/camera.py", "envs/robot/robot.py", "envs/robot/planner.py",
                     "task_config/demo_clean.yml", "task_config/_eval_step_limit.yml"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(relative)
    vcs = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*vcs, "init", "-q"], check=True)
    subprocess.run([*vcs, "add", "."], check=True)
    subprocess.run([*vcs, "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run([*vcs, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    built = []
    plant = ScenePlant()
    monkeypatch.setattr(play_scenes, "scene_instance",
                        lambda scene_set, task: built.append((scene_set, task)) or plant)
    live = _scene_set_session(tmp_path, robotwin_root=root, required_revision=revision,
                              task_factory=session_module._task_factory)
    manifest = live.source_manifest
    assert "envs/shell_game_memory_3.py" not in {item["path"] for item in manifest["files"]}
    assert manifest["play_scene_source"] == play_scenes.source_binding("tabletop-v1")
    live.task_args_factory = lambda **kwargs: {}
    assert live._construct_environment(4) is plant and built == [("tabletop-v1", "shell_game_memory_3")]
    assert plant.setup["seed"] == 4


@pytest.mark.parametrize("scene_set", [None, "tabletop-v1"])
def test_the_scene_set_reaches_the_isolated_worker_only_when_bound(tmp_path, monkeypatch, scene_set):
    from types import SimpleNamespace

    from ora.environment import factory, worker
    from ora.environment.execution_capture import resolve_execution_capture

    runtime = dict(kind="robotwin", execution_capture=resolve_execution_capture(None), startup_timeout_s=1,
                   request_timeout_s=1, task_config="demo_clean", required_revision="source",
                   workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                   environment_signal_policy=PLAY_SIGNAL_POLICY,
                   **({"play_scene_set": scene_set} if scene_set else {}))
    context = factory._BuildContext(runtime=runtime, output_root=tmp_path / "native", task="shell_game_memory_3",
                                    seed=0, max_sim_steps=10, redacted_environment_names=(),
                                    qualified_runtime_binding=None, action_codec="runtime_action_v3")
    monkeypatch.setattr(factory, "_required_environment_path", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(factory, "LocalRpcEnvironmentSession", lambda **kwargs: SimpleNamespace(**kwargs))
    config = {**factory._robotwin(context).worker_config, "shared_root": str(tmp_path)}
    assert ("play_scene_set" in config) is (scene_set is not None)  # a task's worker config keeps its keys
    monkeypatch.setattr("ora.environment.backends.robotwin.session.RoboTwinLiveSession", lambda **kwargs: kwargs)
    assert worker._build_session(config)["play_scene_set"] == scene_set
