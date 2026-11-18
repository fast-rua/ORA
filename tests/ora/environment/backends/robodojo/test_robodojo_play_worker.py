"""The RoboDojo free-play worker (backend ``robodojo_play``) without Isaac.

The worker's dispatch, the host-side config composer, the play bootstrap's
binding checks and layout regeneration, the play session's construction
rules and the reward-free lifecycle's refusals run here on fixtures; building
and judging a real scene is the play-scene audit's job on a GPU host.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from ora.environment.backends.robodojo.bridge import RobodojoError
from ora.environment.backends.robodojo.play_scenes import bootstrap, lifecycle
from ora.environment.backends.robodojo.play_scenes import session as play_session
from ora.environment.benchmarks import robodojo as benchmark
from ora.environment.benchmarks.robodojo_play_scenes import (
    PLAY_NATIVE_ACTION_LIMIT, WORKER_BACKEND, layout_sha256, scene_definition,
)
from tests.support.robodojo.play_scenes import SCENE, _asset_root

SET = "isaac-tabletop-v1"


def _runtime(**changes):
    runtime = {"kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
               "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
               "task_variant": "desk", "play_scene_set": SET, "physics_device": "cuda:0",
               "native_action_limit": PLAY_NATIVE_ACTION_LIMIT, "cuda_visible_devices": "3", "renderer_gpu": 3,
               "startup_timeout_s": 600, "request_timeout_s": 600, "close_timeout_s": 30,
               "play_layouts": {"0": "a" * 64, "4": "b" * 64}, "play_kernel_sha256": "k" * 64,
               "play_scene_source_sha256": "s" * 64}
    runtime.update(changes)
    return runtime


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name in ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(name, str(tmp_path / name))
    python = tmp_path / "runtime/env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setenv("ROBODOJO_PYTHON", str(python))
    return tmp_path


def test_the_worker_dispatches_robodojo_play_to_the_play_session(tmp_path, monkeypatch):
    from ora.environment import worker

    built = []
    monkeypatch.setattr(play_session, "RobodojoPlaySession",
                        lambda *, config, output_root: built.append((config["backend"], output_root)) or "play")
    config = {"backend": WORKER_BACKEND, "output_root": str(tmp_path / "run/native"), "shared_root": str(tmp_path)}
    assert worker._build_session(config) == "play"
    assert built == [(WORKER_BACKEND, (tmp_path / "run/native").resolve())]


def test_the_play_worker_config_is_the_benchmark_config_on_the_play_backend(roots, tmp_path):
    from ora.environment.factory import build_robodojo_worker_session
    from ora.environment.robodojo_play_factory import build_robodojo_play_worker_session, robodojo_play_worker_config

    config, python, environment = robodojo_play_worker_config(
        runtime=_runtime(), output_root=tmp_path / "native", task="desk", seed=4, max_sim_steps=50000,
        action_codec="motion_action_v3")
    assert config["backend"] == WORKER_BACKEND and config["task"] == "desk"
    assert (config["eval_seed"], config["play_seed"], config["qualification_binding"]) == (0, 4, None)
    assert config["play_layouts"] == {"0": "a" * 64, "4": "b" * 64} and environment == {"CUDA_VISIBLE_DEVICES": "3"}
    assert config["play_scene_source_sha256"] == "s" * 64  # the worker checks it (bootstrap.bound_scene_source)
    assert python == roots / "runtime/env/bin/python"
    assert config["runtime_directory"] == str(roots / "runtime")
    session = build_robodojo_play_worker_session(runtime=_runtime(), output_root=tmp_path / "native", task="desk",
                                                 seed=4, max_sim_steps=50000, action_codec="motion_action_v3")
    assert session.worker_config["backend"] == WORKER_BACKEND
    # The benchmark route is untouched: the same runtime through factory.py stays a robodojo worker.
    official = build_robodojo_worker_session(runtime=_runtime(), output_root=tmp_path / "native", task="desk",
                                             seed=0, max_sim_steps=50000, action_codec="motion_action_v3")
    assert official.worker_config["backend"] == "robodojo"
    from ora.environment.factory import EnvironmentFactoryError
    with pytest.raises(EnvironmentFactoryError, match="play_scene_set"):
        robodojo_play_worker_config(runtime=_runtime(play_scene_set=None), output_root=tmp_path, task="desk", seed=0,
                                    max_sim_steps=10, action_codec=None)


def test_free_play_plans_get_the_play_factory_and_everything_else_passes_through(monkeypatch):
    from ora.environment import robodojo_play_factory as factory
    from ora.orchestrator.episode import coordinator

    delegated = []
    monkeypatch.setattr(factory, "build_environment_session", lambda **kwargs: delegated.append(kwargs) or "env")
    assert factory.build_play_environment_session(runtime_config={"kind": "robotwin"}, output_root=Path("/x"),
                                                  task="t", seed=0, max_sim_steps=10) == "env"
    assert delegated and delegated[0]["runtime_config"] == {"kind": "robotwin"}
    with pytest.raises(factory.EnvironmentFactoryError, match="no infrastructure qualification binding"):
        factory.build_play_environment_session(runtime_config=_runtime(), output_root=Path("/x"), task="desk",
                                               seed=0, max_sim_steps=10, qualified_runtime_binding={"passed": True})
    chosen = []

    class Loop:
        def __init__(self, **kwargs):
            self.plan = kwargs["plan"]

        def run(self, sessions):
            return sessions

    monkeypatch.setattr(coordinator, "PlayLoop", Loop)
    monkeypatch.setattr(coordinator, "EpisodeLoop", Loop)
    monkeypatch.setattr(coordinator, "_bind_sessions", lambda episode, **kwargs: chosen.append(
        kwargs["environment_session_factory"]))
    coordinator.run_episode(run_root=Path("/x"), plan={"play": {}}, run_id="r")
    coordinator.run_episode(run_root=Path("/x"), plan={}, run_id="r")
    assert chosen == [factory.build_play_environment_session, coordinator.build_environment_session]


def test_the_play_bootstrap_binds_the_scene_device_and_regenerates_the_audited_layouts(tmp_path, monkeypatch):
    definition = scene_definition(SET, "desk")
    config = {"play_scene_set": SET, "task": "desk", "physics_device": "cuda:0",
              "native_action_limit": PLAY_NATIVE_ACTION_LIMIT}
    assert bootstrap.play_scene(config) == definition
    with pytest.raises(RobodojoError, match="runs on cuda:0 physics"):
        bootstrap.play_scene({**config, "physics_device": "cpu"})
    # The audit's contact pass may run a CUDA scene on CPU physics, and only on CPU.
    assert bootstrap.play_scene({**config, "physics_device": "cpu", "audit_physics_device": "cpu"}) == definition
    with pytest.raises(RobodojoError, match="only with cpu"):
        bootstrap.physics_device({"audit_physics_device": "cuda:0"}, definition)
    with pytest.raises(RobodojoError, match="native action limit"):
        bootstrap.play_scene({**config, "native_action_limit": 1050})
    with pytest.raises(RobodojoError, match="has no scene"):
        bootstrap.play_scene({**config, "task": "make_toast"})
    assets = _asset_root(tmp_path / "Assets")
    scene_file = tmp_path / "fixture.yml"
    scene_file.write_text(json.dumps(SCENE))
    fixture = SimpleNamespace(name="fixture", path=scene_file)
    from ora.environment.benchmarks import robodojo_play_layouts as layouts

    expected = layout_sha256(layouts.generate(SCENE, scene="fixture", seed=2, asset_root=assets)["layout"])
    loaded, generated = bootstrap.generate_layouts({"play_layouts": {"2": expected}}, fixture, assets, tmp_path / "out")
    assert set(loaded) == {2} and generated["layouts"]["2"]["layout_sha256"] == expected
    assert (tmp_path / "out/play-layouts/seed-2.json").is_file()
    assert set(generated["asset_metadata"]) == {item["metadata"] for item in generated["layouts"]["2"]["placements"]}
    with pytest.raises(RobodojoError, match="differs from the audited layout"):
        bootstrap.generate_layouts({"play_layouts": {"2": "0" * 64}}, fixture, assets)
    with pytest.raises(RobodojoError, match="needs runtime.play_layouts"):
        bootstrap.configured_layouts({})


def _session_config(**changes):
    config = {"task": "desk", "task_variant": "desk", "play_scene_set": SET, "physics_device": "cuda:0",
              "native_action_limit": PLAY_NATIVE_ACTION_LIMIT, "play_layouts": {"0": "a" * 64, "4": "b" * 64},
              "play_seed": 4, "eval_seed": 0, "action_codec": "motion_action_v3", "max_sim_steps": 50000,
              "execution_capture": {"enabled": False}}
    config.update(changes)
    return config


def test_the_play_session_admits_only_its_scene_seed_and_has_no_verdict(tmp_path, monkeypatch):
    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    import ora.environment.backends.robodojo.session as robodojo_session
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    session = play_session.RobodojoPlaySession(config=_session_config(), output_root=tmp_path)
    assert (session.task, session.play_seed, session.native_action_limit) == ("desk", 4, PLAY_NATIVE_ACTION_LIMIT)
    assert limits["desk"] == PLAY_NATIVE_ACTION_LIMIT  # this worker process only
    assert session._task_instruction() == "ORA free-play scene isaac-tabletop-v1/desk: no task."
    with pytest.raises(RobodojoError, match="no official verdict"):
        session.persist_official_verdict()
    assert session._official_outcome() == (None, False, "free_play_no_verdict")
    for change, message in (({"play_seed": 7}, "one of runtime.play_layouts"),
                            ({"physics_device": "cpu"}, "runs on cuda:0 physics"),
                            ({"task": "make_toast"}, "has no scene"),
                            ({"layout_schedule": {"rounds": []}}, "restart_round")):
        with pytest.raises(RobodojoError, match=message):
            play_session.RobodojoPlaySession(config=_session_config(**change), output_root=tmp_path)
    with pytest.raises(RobodojoError, match="collides with an official RoboDojo task"):
        play_session.admit_play_scene("make_toast")


def test_the_play_lifecycle_has_no_terminal_and_loads_only_audited_seeds():
    class TaskEnv:
        pass

    cls = lifecycle.play_scene_class(TaskEnv, layouts={0: {"Rigid": {}}, 4: {"Rigid": {}}}, settle_steps=500,
                                     step_limit=PLAY_NATIVE_ACTION_LIMIT)
    assert cls.__mro__[1:3] == (lifecycle.PlaySceneLifecycle, lifecycle.EvaluationLifecycle)
    assert cls.__mro__[3] is TaskEnv and cls.step_lim == PLAY_NATIVE_ACTION_LIMIT
    environment = object.__new__(cls)
    environment.harness_layout_ids = (0, 4)
    environment.num_envs = 1
    assert environment.advance_official_terminal() is None
    for seed in ([7], 4, [0, 4]):
        with pytest.raises(RobodojoError, match="audited seed"):
            environment.reset(seed=seed)
    environment.sim = object()
    environment.harness_loaded_layout_id = 0
    with pytest.raises(RobodojoError, match="relaunch_for_layout"):
        environment.reset(seed=[4])
    environment.take_action_cnt = [PLAY_NATIVE_ACTION_LIMIT]
    with pytest.raises(RobodojoError, match="native action limit"):
        environment.native_tick({})
    with pytest.raises(RobodojoError, match="native action limit"):
        environment.native_tick_sequence([{}] * 10)


def test_a_per_step_native_action_runs_in_a_play_scene_without_a_reward(monkeypatch):
    """The rate-limited jaws and smooth trajectories (``native_tick_sequence``) need no reward registration."""

    steps: list = []

    class Manager:
        def __init__(self):
            self.queue, self.prev_control = [], [{}]

        def push(self, envs, sequences):
            self.queue.extend(sequences[0])

        def pop(self, envs):
            return [self.queue.pop(0)]

        def get_empty(self, envs):
            return not self.queue

    class TaskEnv:
        def step(self, meta_control_list):
            steps.append(meta_control_list[0])

    monkeypatch.setattr(lifecycle, "_fixed_meta_control", lambda command, robots, previous, requested: (command, command))
    cls = lifecycle.play_scene_class(TaskEnv, layouts={0: {}}, settle_steps=0, step_limit=PLAY_NATIVE_ACTION_LIMIT)
    environment = object.__new__(cls)
    environment.reward_registered = False  # a play scene registers none, now or ever
    environment.take_action_cnt, environment.unstable_envs = [0], set()
    environment.obs_manager = SimpleNamespace(collect_interval=10)
    environment.robot_manager = SimpleNamespace(control_manager=Manager())
    environment.sim_step = lambda render: None
    captured = []
    environment.execution_capture_callback = lambda: captured.append(1)
    assert environment.native_tick_sequence(lambda index: {"left_gripper": index / 10}) is None
    assert [step["left_gripper"] for step in steps] == [index / 10 for index in range(10)]
    assert environment.take_action_cnt == [1] and len(captured) == 10
    with pytest.raises(RobodojoError, match="exactly 10"):
        environment.native_tick_sequence([{}] * 3)


def test_a_relaunch_measures_the_camera_clock_offset_within_its_tolerance():
    class TaskEnv:
        pass

    cls = lifecycle.play_scene_class(TaskEnv, layouts={0: {}}, settle_steps=0, step_limit=PLAY_NATIVE_ACTION_LIMIT)
    environment = object.__new__(cls)
    environment.sim = SimpleNamespace(sim=SimpleNamespace(current_time=15.764, current_time_step_index=3941))
    renders = []
    environment.render = lambda: renders.append(1)

    def views(time_s):
        reading = SimpleNamespace(get_data=lambda: {"simulationTime": time_s})
        return [SimpleNamespace(_harness_simulation_time=reading) for _ in range(3)]

    environment.harness_simulation_time_offset_s = 0.0
    assert environment._calibrate_epoch_clock() is None  # the first epoch keeps its own clock
    environment.harness_simulation_time_offset_s = 17.764
    environment.capture_manager = SimpleNamespace(tiled_cameras=views(15.764 + 17.764 + 0.076))
    clock = environment._calibrate_epoch_clock()
    assert clock["difference_s"] == pytest.approx(0.076) and len(renders) == 2
    assert environment.harness_simulation_time_offset_s == pytest.approx(17.84)
    environment.harness_simulation_time_offset_s = 17.764
    environment.capture_manager = SimpleNamespace(tiled_cameras=views(15.764 + 17.764 + 0.5))
    with pytest.raises(RobodojoError, match="tolerance"):
        environment._calibrate_epoch_clock()


# --- what the worker reads of a scene ------------------------------------------------------------
#
# A scene's audit binds its canonical index entry, its YAML and the generator
# (robodojo_play_scenes.source_binding), not the whole scene set. That is
# sound only if the worker reads nothing else of the set: the test below runs
# every worker step that touches scene data -- the play session's construction,
# the bootstrap's scene and source checks, the layout regeneration, the task
# config handed to upstream and the lifecycle's settle -- on a two-scene set,
# records every file it opens (a process audit hook, inert outside the
# recording) and shows that another scene's YAML and entry, and this scene's
# operator description, change nothing the worker uses.

_RECORDING: list[str] | None = None
_HOOKED = False


def _record_open(event, args):
    if _RECORDING is not None and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _RECORDING.append(os.fsdecode(args[0]))


@contextlib.contextmanager
def _opened_files():
    global _RECORDING, _HOOKED
    if not _HOOKED:
        sys.addaudithook(_record_open)
        _HOOKED = True
    _RECORDING = opened = []
    try:
        yield opened
    finally:
        _RECORDING = None


def _two_scene_set(root: Path) -> Path:
    from ora.environment.benchmarks import robodojo_play_scenes as registry

    directory = root / registry.SCENE_ROOT / registry.SCENE_SET_DIRECTORIES[SET]
    directory.mkdir(parents=True)
    entries = [{"name": name, "label": f"{name} table", "physics_device": "cpu", "settle_steps": 1500,
                "assets": ["Articulation/toaster", "Clutter/plate", "Rigid/cup", "Rigid/block"],
                "description": f"The {name} scene."} for name in ("alpha", "beta")]
    (directory / registry.INDEX_FILE).write_text(yaml.safe_dump({"scenes": entries}, sort_keys=False))
    for name in ("alpha", "beta"):
        (directory / f"{name}.yml").write_text(yaml.safe_dump(SCENE, sort_keys=False))
    generator = root / registry.GENERATOR_MODULE
    generator.parent.mkdir(parents=True)
    generator.write_bytes((registry.ROOT / registry.GENERATOR_MODULE).read_bytes())
    return directory


def _edit_entry(directory: Path, scene: str, **changes):
    data = yaml.safe_load((directory / "index.yml").read_text())
    next(entry for entry in data["scenes"] if entry["name"] == scene).update(changes)
    (directory / "index.yml").write_text(yaml.safe_dump(data, sort_keys=False))


def test_the_worker_reads_of_a_scene_only_its_own_entry_and_yaml(tmp_path, monkeypatch):
    from ora.environment.benchmarks import robodojo_play_layouts as layouts
    from ora.environment.benchmarks import robodojo_play_scenes as registry
    import ora.environment.backends.robodojo.session as robodojo_session

    tree = tmp_path / "ora"
    directory = _two_scene_set(tree)
    monkeypatch.setattr(registry, "ROOT", tree)
    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    assets = _asset_root(tmp_path / "Assets")
    expected = layout_sha256(layouts.generate(SCENE, scene="alpha", seed=2, asset_root=assets)["layout"])

    def worker(source_sha256=None):
        config = _session_config(task="alpha", task_variant="alpha", physics_device="cpu",
                                 play_layouts={"2": expected}, play_seed=2,
                                 play_scene_source_sha256=source_sha256 or registry.source_binding(SET, "alpha")["sha256"])
        with _opened_files() as opened:
            session = play_session.RobodojoPlaySession(config=config, output_root=tmp_path / "session")
            definition = bootstrap.play_scene(config)
            source = bootstrap.bound_scene_source(config, definition)
            loaded, generated = bootstrap.generate_layouts(config, definition, assets)
            task_config = bootstrap.scene_task_config(definition)
            scene_class = lifecycle.play_scene_class(type("TaskEnv", (), {}), layouts=loaded,
                                                     settle_steps=definition.settle_steps,
                                                     step_limit=PLAY_NATIVE_ACTION_LIMIT)
        reads = sorted({Path(path).resolve().relative_to(directory.resolve()).as_posix() for path in opened
                        if Path(path).resolve().is_relative_to(directory.resolve())})
        used = {"session": (session.task, session._task_instruction(), session.play_definition.physics_device),
                "definition": (definition.name, definition.scene_set, definition.scene_file,
                               definition.physics_device),
                "settle_steps": scene_class.harness_settle_steps, "layout": generated["layouts"]["2"]["layout_sha256"],
                "task_config": task_config, "source": source["sha256"]}
        return reads, used

    reads, used = worker()
    assert reads == ["alpha.yml", "index.yml"]  # never beta.yml
    assert used["layout"] == expected and used["settle_steps"] == 1500
    # Another scene's YAML and entry, and this scene's operator text, change nothing the worker uses.
    (directory / "beta.yml").write_text(yaml.safe_dump({**SCENE, "Rigid": SCENE["Rigid"][:1]}, sort_keys=False))
    _edit_entry(directory, "beta", settle_steps=999, physics_device="cuda:0", label="beta counter")
    _edit_entry(directory, "alpha", description="Something else.")
    assert worker() == (reads, used)
    # This scene's entry is read and bound: a settle edit reaches the lifecycle and changes the binding,
    # which the worker refuses against the old audited value.
    bound = used["source"]
    _edit_entry(directory, "alpha", settle_steps=1200)
    with pytest.raises(RobodojoError, match="differ from the audited binding"):
        worker(source_sha256=bound)
    reads_after, used_after = worker()
    assert reads_after == reads and used_after["settle_steps"] == 1200 and used_after["source"] != bound


# --- the operator's view -------------------------------------------------------------------------

def _project(point, view):
    from ora.environment.backends.robodojo.play_scenes.operator_view import look_at

    axes = look_at(view["eye"], view["target"], view["up"])
    relative = [p - e for p, e in zip(point, view["eye"])]
    depth = sum(a * b for a, b in zip(relative, axes["forward"]))
    pixels_per_mm = view["width"] / view["horizontal_aperture_mm"]
    u = view["width"] / 2 + view["focal_mm"] * pixels_per_mm * sum(a * b for a, b in zip(relative, axes["right"])) / depth
    v = view["height"] / 2 - view["focal_mm"] * pixels_per_mm * sum(a * b for a, b in zip(relative, axes["up"])) / depth
    return depth, u, v


def test_the_operator_camera_is_no_policy_camera_and_frames_both_arms_and_the_play_area():
    from ora.environment.backends.robodojo.play_scenes import operator_view as view
    from ora.environment.backends.robodojo.session import CAMERAS
    from ora.environment.benchmarks.robodojo_play_scenes import PLAY_REACH_ENVELOPE_XY_M, TABLE_TOP_Z_M, X5_BASES_XY_M
    from ora.orchestrator.robodojo_play_audit import spec_runtime

    policy = {*CAMERAS, *CAMERAS.values(), *benchmark.PUBLIC_CAMERAS,
              *spec_runtime(scene_definition(SET, "desk"), gpu="3", host="h")["policy_cameras"]}
    assert view.OPERATOR_CAMERA not in policy and view.OPERATOR_CAMERA_PRIM.startswith("/World/OraOperatorView/")
    # 1280 x 720 (16:9, square pixels), and both arm bases and the corners of the play area are in front of
    # it and in frame.
    spec = view.OPERATOR_VIEW
    assert (spec["width"], spec["height"]) == (1280, 720)
    # Beside the policy's three 640 x 480 cameras one capture frame stays inside the capture's pixel bound.
    from ora.environment.execution_capture import MAX_FRAME_BYTES
    assert 3 * 640 * 480 * 3 + spec["width"] * spec["height"] * 3 <= MAX_FRAME_BYTES
    (x0, x1), (y0, y1) = PLAY_REACH_ENVELOPE_XY_M["x"], PLAY_REACH_ENVELOPE_XY_M["y"]
    points = [(x, y, TABLE_TOP_Z_M) for x in (x0, x1) for y in (y0, y1)]
    points += [(x, y, TABLE_TOP_Z_M + 0.25) for x, y in X5_BASES_XY_M.values()]
    for point in points:
        depth, u, v = _project(point, spec)
        assert depth > 0 and 0 <= u <= spec["width"] and 0 <= v <= spec["height"], (point, u, v)
    assert view.operator_view_mode({}) == "capture" and view.operator_view_mode({"operator_view_mode": "off"}) == "off"
    with pytest.raises(ValueError, match="operator_view_mode"):
        view.operator_view_mode({"operator_view_mode": "policy"})
    with pytest.raises(RobodojoError, match="operator view mode"):
        lifecycle.play_scene_class(type("TaskEnv", (), {}), layouts={0: {}}, settle_steps=0,
                                   step_limit=PLAY_NATIVE_ACTION_LIMIT, operator_view_mode="policy")


class _OperatorOnly:
    """An environment that has an operator camera and nothing else: any other use fails the test."""

    def __init__(self, frame):
        self.frame, self.reads = frame, 0

    def operator_rgb(self):
        self.reads += 1
        return self.frame

    def __getattr__(self, name):
        raise AssertionError(f"operator_view touched environment.{name}")


def test_the_operator_view_reads_only_the_operator_camera(tmp_path, monkeypatch):
    import numpy as np

    from ora.environment.base import validate_operator_view
    import ora.environment.backends.robodojo.session as robodojo_session

    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    session = play_session.RobodojoPlaySession(config=_session_config(), output_root=tmp_path)
    assert session.operator_view() is None  # not started
    frame = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    session.environment = _OperatorOnly(frame)
    answer = validate_operator_view(session.operator_view())
    assert answer["camera"] == "operator_camera" and (answer["width"], answer["height"]) == (64, 48)
    assert session.environment.reads == 1  # no render, no physics step, no policy camera
    session.environment = _OperatorOnly(None)  # attached but not rendered yet
    assert session.operator_view() is None


class _CaptureWorld:
    """The capture's view of the world: the policy cameras through get_obs, the operator camera apart."""

    def __init__(self):
        import numpy as np

        self.operator_reads = 0
        self.execution_capture_callback = None
        self.policy = {source: np.full((24, 32, 3), index * 40, np.uint8)
                       for index, source in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))}
        self.operator = np.full((48, 64, 3), 200, np.uint8)

    def get_obs(self):
        return {"vision": {source: {"color": frame} for source, frame in self.policy.items()}}

    def operator_rgb(self):
        self.operator_reads += 1
        return self.operator


@pytest.mark.parametrize("mode", ["capture", "view", "off"])
def test_the_execution_capture_adds_the_operator_camera_only_in_capture_mode(tmp_path, monkeypatch, mode):
    import ora.environment.backends.robodojo.execution_capture as capture_mixin
    import ora.environment.backends.robodojo.session as robodojo_session
    import ora.environment.execution_capture as execution_capture

    recorders = []

    class Recorder:
        def __init__(self, root, *, backend, cameras, identity, config):
            self.cameras, self.frames, self.error = list(cameras), [], None
            recorders.append(self)

        def capture(self, pixels, **record):
            self.frames.append((record["phase"], record["action_id"], pixels()))

    monkeypatch.setattr(execution_capture, "ExecutionCapture", Recorder)
    monkeypatch.setattr(capture_mixin, "ExecutionCapture", Recorder)
    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    session = play_session.RobodojoPlaySession(
        config=_session_config(execution_capture={"enabled": True}, operator_view_mode=mode), output_root=tmp_path)
    world = _CaptureWorld()
    session.environment, session.source_hash = world, "s" * 64
    session.round_index, session.reset_physics_steps = 0, 0
    monkeypatch.setattr(session, "_clock", lambda: 25)
    session._start_execution_capture()
    session._execution_action = {"action_id": "action-7", "action": "move_pose"}
    session._capture_execution_boundary("action_start")
    (recorder,) = recorders
    policy = ["head", "left_wrist", "right_wrist"]
    assert recorder.cameras == (policy + ["operator_camera"] if mode == "capture" else policy)
    assert [(phase, action) for phase, action, _ in recorder.frames] == [("reset", None), ("action_start", "action-7")]
    for _, _, frames in recorder.frames:
        assert list(frames) == recorder.cameras
        assert all(frames[name] is world.policy[source] for name, source in
                   zip(policy, ("cam_head", "cam_left_wrist", "cam_right_wrist")))
    assert world.operator_reads == (2 if mode == "capture" else 0)
    # The policy's observation is get_obs's vision: the operator camera is not in it.
    assert "operator_camera" not in world.get_obs()["vision"]


def test_a_relaunch_releases_the_operator_camera_before_the_stage_closes(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle.operator_view, "release", lambda handle: calls.append(("release", handle)))
    monkeypatch.setattr(lifecycle.EvaluationLifecycle, "relaunch_for_layout",
                        lambda self, layout_id: calls.append(("relaunch", layout_id)) or {"layout_id": layout_id})
    cls = lifecycle.play_scene_class(type("TaskEnv", (), {}), layouts={0: {}, 4: {}}, settle_steps=0,
                                     step_limit=PLAY_NATIVE_ACTION_LIMIT)
    environment = object.__new__(cls)
    environment.harness_operator_view = {"camera": "operator_camera"}
    assert environment.relaunch_for_layout(4) == {"layout_id": 4}
    assert calls == [("release", {"camera": "operator_camera"}), ("relaunch", 4)]
    assert environment.harness_operator_view is None and environment.operator_rgb() is None
    # A failing release is recorded and the scene still changes.
    monkeypatch.setattr(lifecycle.operator_view, "release", lambda handle: (_ for _ in ()).throw(RuntimeError("gone")))
    environment.harness_operator_view = {"camera": "operator_camera"}
    assert environment.relaunch_for_layout(0) == {"layout_id": 0}
    assert environment.harness_operator_view_release_error == "RuntimeError: gone"


def test_the_operator_camera_is_released_by_its_render_product_path():
    from ora.environment.backends.robodojo.play_scenes import operator_view

    calls = []

    class Product:  # Replicator's HydraTexture: attach takes the object, detach its path
        path = "/Render/OmniverseKit/HydraTextures/operator"

        def destroy(self):
            calls.append("destroy")

    class Annotator:
        def detach(self, paths):
            assert all(isinstance(path, str) for path in paths), paths
            calls.append(("detach", tuple(paths)))

    operator_view.release({"product": Product(), "annotator": Annotator()})
    assert calls == [("detach", ("/Render/OmniverseKit/HydraTextures/operator",)), "destroy"]
    operator_view.release(None)


def _fake_physics_usd(monkeypatch, bodies):
    """Fake ``pxr`` and ``omni.usd``: a stage of prims under object roots, each a rigid body or not."""
    applied = {}

    class Attribute:
        def __init__(self, path, name):
            self.path, self.name = path, name

        def Set(self, value):
            applied.setdefault(self.path, {})[self.name] = value

        def Get(self):
            return None

    class Prim:
        def __init__(self, path, rigid, kinematic=False):
            self.path, self.rigid, self.kinematic = path, rigid, kinematic

        def IsValid(self):
            return True

        def __bool__(self):
            return True

        def HasAPI(self, api):
            return self.rigid and api is UsdPhysics.RigidBodyAPI

    prims = {path: Prim(path, rigid, kinematic) for path, rigid, kinematic in bodies}

    class RigidBodyAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetKinematicEnabledAttr(self):
            prim = self.prim
            return SimpleNamespace(Get=lambda: prim.kinematic)

    class PhysxRigidBodyAPI:
        def __init__(self, prim):
            self.prim = prim

        @classmethod
        def Apply(cls, prim):
            return cls(prim)

        def CreateMaxDepenetrationVelocityAttr(self):
            return Attribute(self.prim.path, "maxDepenetrationVelocity")

        def CreateMaxLinearVelocityAttr(self):
            return Attribute(self.prim.path, "maxLinearVelocity")

        def CreateMaxAngularVelocityAttr(self):
            return Attribute(self.prim.path, "maxAngularVelocity")

    class Stage:
        def GetPrimAtPath(self, path):
            return prims.get(path)

    UsdPhysics = SimpleNamespace(RigidBodyAPI=RigidBodyAPI)
    pxr = SimpleNamespace(
        PhysxSchema=SimpleNamespace(PhysxRigidBodyAPI=PhysxRigidBodyAPI), UsdPhysics=UsdPhysics,
        Usd=SimpleNamespace(PrimRange=lambda root: [prim for path, prim in prims.items()
                                                     if path == root.path or path.startswith(root.path + "/")]),
        UsdGeom=SimpleNamespace(GetStageMetersPerUnit=lambda stage: 1.0))
    usd = SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: Stage()))
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(usd=usd))
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    return applied


def test_play_scene_objects_get_the_physx_limits_and_the_robot_and_kinematic_bodies_do_not(monkeypatch):
    applied = _fake_physics_usd(monkeypatch, [
        ("/World/envs/env_0/rigid/owl", True, False),
        ("/World/envs/env_0/articulation/box", False, False),
        ("/World/envs/env_0/articulation/box/base", True, False),
        ("/World/envs/env_0/articulation/box/flap", True, False),
        ("/World/envs/env_0/articulation/box/anchor", True, True),
        ("/World/envs/env_0/Robot/link6", True, False),
    ])
    manager = SimpleNamespace(
        _rigid_and_dynamic_objects=[{"owl_0": SimpleNamespace(prim_path="/World/envs/env_0/rigid/owl")}],
        _articulation_objects=[{"box_0": SimpleNamespace(prim_path="/World/envs/env_0/articulation/box")}])
    record = lifecycle.limit_rigid_bodies(manager)
    assert record == {"schema": lifecycle.RIGID_BODY_LIMITS_SCHEMA, "max_depenetration_velocity_m_s": 0.5,
                      "max_linear_velocity_m_s": 3.0, "max_angular_velocity_deg_s": 720.0, "bodies": 3,
                      "meters_per_unit": 1.0}
    limits = {"maxDepenetrationVelocity": 0.5, "maxLinearVelocity": 3.0, "maxAngularVelocity": 720.0}
    assert applied == {"/World/envs/env_0/rigid/owl": limits, "/World/envs/env_0/articulation/box/base": limits,
                       "/World/envs/env_0/articulation/box/flap": limits}


def test_the_limits_follow_the_simulation_setup_and_every_later_spawn(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "limit_rigid_bodies",
                        lambda manager: calls.append(manager) or {"schema": "limits", "bodies": len(calls)})

    class TaskEnv:
        def _post_setup_scene(self, sim):
            calls.append(("setup", sim))

    cls = lifecycle.play_scene_class(TaskEnv, layouts={0: {}}, settle_steps=0, step_limit=PLAY_NATIVE_ACTION_LIMIT)
    environment = object.__new__(cls)
    manager = SimpleNamespace(spawn_scene_objects=lambda env_id, exclude_types=None: calls.append(("spawn", env_id)))
    environment.scene_manager, environment.task_name = manager, "toys"
    environment.obs_manager = SimpleNamespace(initialize=lambda env: calls.append("observations"))
    environment._post_setup_scene("sim")
    assert calls == [("setup", "sim"), "observations", manager]
    assert environment.harness_rigid_body_limits == {"schema": "limits", "bodies": 3}
    # The reset's own respawn (CPU physics respawns rigid objects) goes through the same limits.
    environment._limit_rigid_bodies_on_spawn()
    manager.spawn_scene_objects(0, exclude_types=["articulation"])
    assert calls[-2:] == [("spawn", 0), manager] and environment.harness_rigid_body_limits["bodies"] == 5


# --- the world between actions -------------------------------------------------------------------

class _Body:
    def __init__(self, linear, angular=(0.0, 0.0, 0.0), position=(0.0, 0.0, 0.8), quaternion=(1.0, 0.0, 0.0, 0.0)):
        self.linear, self.angular, self.position, self.quaternion = linear, angular, position, quaternion

    def get_linear_velocity(self):
        return list(self.linear)

    def get_angular_velocity(self):
        return list(self.angular)

    def _get_object_transform(self):
        return list(self.position), list(self.quaternion), "cpu"


class _Joints:
    def __init__(self, speeds, positions=(0.0,)):
        self.speeds, self.positions = speeds, positions

    def get_joint_velocities(self):
        return list(self.speeds)

    def get_joint_positions(self):
        return list(self.positions)


def test_held_physics_steps_are_captured_and_the_scene_objects_are_read_without_a_step():
    cls = lifecycle.play_scene_class(type("TaskEnv", (), {}), layouts={0: {}}, settle_steps=0,
                                     step_limit=PLAY_NATIVE_ACTION_LIMIT)
    environment = object.__new__(cls)
    calls = []
    environment.sim_step = lambda render=True: calls.append(("step", render))
    environment.execution_capture_callback = lambda: calls.append("capture")
    environment.unstable_envs = set()
    # No control and no native action: the instance has no robot, control manager or action counter to touch.
    environment.hold_steps(3)
    assert calls == [("step", False), "capture"] * 3
    environment.scene_manager = SimpleNamespace(
        _rigid_and_dynamic_objects=[{"mug_0": _Body((0.0, 0.02, 0.0)), "owl_0": _Body((0.0, 0.0, 0.0), (0.0, 0.0, 0.05)),
                                     "pad_0": object()}],
        _articulation_objects=[{"toaster_0": _Joints([0.0, -0.3], (0.1, 0.2)), "box_0": _Joints([0.01])}])
    motion = environment.object_motion()
    assert motion["moving"] == ["mug_0", "toaster_0"]
    assert (motion["max_linear_speed_m_s"], motion["max_angular_speed_rad_s"], motion["max_joint_speed"]) == (
        pytest.approx(0.02), pytest.approx(0.05), pytest.approx(0.3))
    poses = environment.object_poses()
    assert set(poses) == {"mug_0", "owl_0", "toaster_0", "box_0"} and poses["toaster_0"] == {"joints": [0.1, 0.2]}
    assert poses["mug_0"] == {"position": [0.0, 0.0, 0.8], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    assert calls == [("step", False), "capture"] * 3  # reading stepped nothing
    environment.unstable_envs = {0}
    with pytest.raises(RobodojoError, match="unstable"):
        environment.hold_steps(1)
    assert lifecycle.PLAY_POST_ACTION_SETTLE["max_physics_steps"] * benchmark.PHYSICS_DT_S == pytest.approx(2.0)


def test_a_window_moves_an_object_when_it_shifts_or_turns_beyond_its_bounds():
    import math

    rest = {"cup": {"position": [0.0, 0.0, 0.8], "quaternion": [1.0, 0.0, 0.0, 0.0]}, "lid": {"joints": [0.0]}}
    half = math.radians(1.5) / 2
    assert lifecycle.displaced(rest, rest) == []
    assert lifecycle.displaced(rest, {**rest, "cup": {"position": [0.0009, 0.0, 0.8],
                                                      "quaternion": [1.0, 0.0, 0.0, 0.0]}}) == []
    assert lifecycle.displaced(rest, {**rest, "cup": {"position": [0.0, 0.0012, 0.8],
                                                      "quaternion": [1.0, 0.0, 0.0, 0.0]}}) == ["cup"]
    assert lifecycle.displaced(rest, {**rest, "cup": {"position": [0.0, 0.0, 0.8],
                                                      "quaternion": [math.cos(half), math.sin(half), 0.0, 0.0]}}) == ["cup"]
    assert lifecycle.displaced(rest, {**rest, "lid": {"joints": [0.006]}}) == ["lid"]
    assert lifecycle.displaced(rest, {**rest, "new": {"joints": [1.0]}}) == []


class _SettleWorld:
    """A scene whose one object slides 1 mm per held physics step until ``rest_after`` steps.

    ``jitter`` keeps its velocity reading above the rest speeds while its pose stands still.
    """

    def __init__(self, rest_after, advance=True, jitter=False):
        self.clock, self.rest_after, self.advance, self.jitter, self.held = 0, rest_after, advance, jitter, []

    def hold_steps(self, count):
        self.held.append(count)
        self.clock += count if self.advance else 0

    def object_motion(self):
        moving = [] if self.clock >= self.rest_after and not self.jitter else ["block_1"]
        return {"max_linear_speed_m_s": 0.3 if moving else 0.0, "max_angular_speed_rad_s": 0.0,
                "max_joint_speed": 0.0, "moving": moving}

    def object_poses(self):
        travelled = 0.0 if self.jitter else min(self.clock, self.rest_after) * 0.001
        return {"block_1": {"position": [travelled, 0.0, 0.8], "quaternion": [1.0, 0.0, 0.0, 0.0]}}


def _settle_session(tmp_path, monkeypatch, world):
    import ora.environment.backends.robodojo.session as robodojo_session

    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    session = play_session.RobodojoPlaySession(config=_session_config(), output_root=tmp_path)
    session.environment = world
    monkeypatch.setattr(session, "_clock", lambda: world.clock)
    return session


def _settle_records(tmp_path):
    path = tmp_path / play_session.POST_ACTION_SETTLE_RECORD
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_an_action_is_followed_by_held_physics_until_the_scene_rests_inside_its_capture_window(tmp_path, monkeypatch):
    from ora.environment.backends.robodojo.session import RobodojoSession

    world = _SettleWorld(rest_after=120)
    session = _settle_session(tmp_path, monkeypatch, world)
    events = []

    def benchmark_action(self, action):
        events.append("receipt")
        self.action_sequence += 1
        return {"receipt": {"action_id": action["action_id"]}}

    monkeypatch.setattr(RobodojoSession, "_execute_without_capture", benchmark_action)
    monkeypatch.setattr(session, "_capture_execution_boundary",
                        lambda phase, force=True: events.append((phase, world.clock)))
    receipt = session.execute({"action_id": "a-1", "action": "move_pose"})
    assert receipt == {"receipt": {"action_id": "a-1"}}
    # The receipt first, then the settle (10-step windows up to the first still one), then action_end.
    assert events == [("action_start", 0), "receipt", ("action_end", 130)]
    assert world.held == [10] * 13 and session.sim_steps == 0  # not the session's physics budget
    (record,) = _settle_records(tmp_path)
    assert record["schema"] == play_session.POST_ACTION_SETTLE_SCHEMA
    assert (record["action_id"], record["action_sequence"], record["round_index"]) == ("a-1", 0, 0)
    assert (record["physics_steps"], record["simulated_s"], record["ended"], record["windows"]) == (
        130, 0.52, "at_rest", 13)
    assert (record["moving_at_start"], record["moving_at_end"], record["physics_step_start"]) == (["block_1"], [], 0)
    # A scene already at rest costs one reading and no physics step.
    assert session._settle_after_action("a-2", 1)["physics_steps"] == 0 and world.held == [10] * 13
    # A velocity that jitters while the pose stands still costs one window.
    world.jitter = True
    assert session._settle_after_action("a-3", 2)["physics_steps"] == 10


def test_the_settle_stops_at_its_cap_waits_for_a_sequence_and_refuses_a_clock_it_did_not_advance(tmp_path,
                                                                                                   monkeypatch):
    world = _SettleWorld(rest_after=10 ** 6)
    session = _settle_session(tmp_path, monkeypatch, world)
    cap = lifecycle.PLAY_POST_ACTION_SETTLE["max_physics_steps"]
    record = session._settle_after_action("a-1", 0)
    assert (record["physics_steps"], record["ended"], record["moving_at_end"]) == (cap, "step_limit", ["block_1"])
    # Between the children of a sequence the next child follows at once; after its last child the scene settles.
    session.action_sequences = [{"children": ["c-0", "c-1"], "next": 1}]
    assert session._settle_after_action("c-0", 1) is None and world.clock == cap
    session.action_sequences = [{"children": ["c-0", "c-1"], "next": 2}]
    assert session._settle_after_action("c-1", 2)["physics_steps"] == cap
    session.action_sequences = []
    world.advance = False
    with pytest.raises(RobodojoError, match="exactly its physics steps"):
        session._settle_after_action("a-3", 3)
    assert session.fault_reason == "robodojo_play_settle_clock_mismatch"
    assert [item["action_id"] for item in _settle_records(tmp_path)] == ["a-1", "c-1"]


# --- the shared motion base ----------------------------------------------------------------------
#
# A play session binds the shared planner and timing as the benchmark session does (shared_motion.start).
# The planner is a stand-in here: its binding, its forward kinematics, its IK and its close.

JOINTS = [0.1, 1.0, 1.0, 0.0, 0.0, 0.0]
#: The link6 pose the simulator's FK gives every joint row here; both arm bases sit at the world origin.
LINK6 = [0.3, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0]
PLANNER = {"implementation": "curobo_v2", "python_env": "CUROBO_V2_PYTHON", "root_env": "CUROBO_V2_ROOT",
           "required_revision": "c" * 40, "robot": "arx_x5", "world_model": "public_depth_voxel_v1"}


class _Planner:
    def __init__(self, *, offset_m=0.0, start_error=None):
        self.offset_m, self.start_error, self.ik_requests, self.closed = offset_m, start_error, [], 0

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        return {"implementation": "curobo_v2"}

    def forward_kinematics(self, rows):
        return [[LINK6[0] + self.offset_m, *LINK6[1:]] for _ in rows]

    def solve_ik(self, request):
        self.ik_requests.append(request)
        return {"status": "solved", "joint_limits": [[-3.0] * 6, [3.0] * 6],
                "results": [{"skipped": False, "solved": True, "joint_positions": JOINTS, "refusal": None}
                            for _ in request["queries"]]}

    def close(self):
        self.closed += 1


def test_the_play_worker_config_binds_the_planner_as_the_benchmark_config_does(roots, tmp_path, monkeypatch):
    from ora.environment.factory import EnvironmentFactoryError, build_robodojo_worker_session
    from ora.environment.robodojo_play_factory import robodojo_play_worker_config

    (tmp_path / "curobo").mkdir()
    monkeypatch.setenv("CUROBO_V2_PYTHON", str(roots / "runtime/env/bin/python"))
    monkeypatch.setenv("CUROBO_V2_ROOT", str(tmp_path / "curobo"))
    runtime = _runtime(motion_planner=dict(PLANNER))
    arguments = {"output_root": tmp_path / "native", "task": "desk", "max_sim_steps": 50000,
                 "action_codec": "motion_action_v3"}
    config, _, _ = robodojo_play_worker_config(runtime=runtime, seed=4, **arguments)
    official = build_robodojo_worker_session(runtime=runtime, seed=0, **arguments)
    assert config["motion_planner"] == official.worker_config["motion_planner"] == {
        **PLANNER, "python_executable": str(roots / "runtime/env/bin/python"), "root": str(tmp_path / "curobo")}
    assert "motion_planner" not in robodojo_play_worker_config(runtime=_runtime(), seed=4, **arguments)[0]
    monkeypatch.delenv("CUROBO_V2_ROOT")
    with pytest.raises(EnvironmentFactoryError, match="missing CUROBO_V2_ROOT"):
        robodojo_play_worker_config(runtime=runtime, seed=4, **arguments)


def _plant_session(tmp_path, monkeypatch, planner, **config):
    """A play session on a stand-in plant whose ``planner_client`` is ``planner``; not started."""
    import ora.environment.backends.robodojo.session as robodojo_session
    from ora.environment.backends.robodojo import reachability, shared_motion
    from ora.environment.controllers import motion_timing, robodojo_arx

    limits = dict(benchmark.NATIVE_ACTION_LIMITS)
    monkeypatch.setattr(benchmark, "NATIVE_ACTION_LIMITS", limits)
    monkeypatch.setattr(robodojo_session, "NATIVE_ACTION_LIMITS", limits)
    built, timing = [], []

    def client(configured, *, python_executable, root, output_root, robot_file):
        output_root.mkdir(parents=True, exist_ok=True)  # as MotionPlannerClient does
        built.append({"python_executable": python_executable, "root": root, "robot_file": robot_file})
        return planner

    monkeypatch.setattr(shared_motion, "planner_client", client)
    monkeypatch.setattr(motion_timing, "bind_preflight", lambda root: timing.append(Path(root)))
    monkeypatch.setattr(reachability, "fk_link6", lambda session, arm, joints: list(LINK6))
    monkeypatch.setattr(robodojo_arx, "validate_robot_facts", lambda facts: None)
    robots = [SimpleNamespace(arm_name=f"{arm}_arm", type="target", arm_joints_name=[f"joint{i}" for i in range(6)],
                              ee_link_name="link6", ee_type="gripper", gripper_scale=[0.0, 0.044],
                              gripper_move={"sign": 1, "mimic": [0, 1, 0]},
                              entity_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], curobo_yml_path="/x5/X5A.yml")
              for arm in ("left", "right")]
    manager = SimpleNamespace(robot_list=robots, get_end_effector_real_val=lambda robot, env_idx_list: [[0.02, 0.02]])
    world = SimpleNamespace(robot_manager=manager, scene_registration={"instances": {}})
    session = play_session.RobodojoPlaySession(
        config=_session_config(**config), output_root=tmp_path / "native",
        environment_factory=lambda config: (world, None, {"source_revisions": {}, "device_binding": {}}))
    session.exit_process_on_close = False  # an RPC worker's close would end this process
    session.gripper_directives = {}
    for name, value in (("_reset_environment", lambda *args, **kwargs: None), ("_runtime_embodiment", lambda: {}),
                        ("_bind_reset_state", lambda: None), ("_clock", lambda: 7),
                        ("_loaded_layout", lambda: {"seed": 4, "layout_sha256": "b" * 64,
                                                    "planned_layout_sha256": "b" * 64, "object_counts": {}}),
                        ("_proprioception", lambda: {"end_effectors": {
                            arm: {"arm_joint_qpos": list(JOINTS), "pose_xyz_xyzw": list(LINK6)}
                            for arm in ("left", "right")}})):
        monkeypatch.setattr(session, name, value)
    return session, built, timing


def _observe(session, root):
    """What an observation leaves the shared planner: each camera's depth and calibration, at the current clock."""
    import numpy as np

    for camera in ("head", "left_wrist", "right_wrist"):
        (root / camera).mkdir(parents=True)
        np.save(root / camera / "depth-m.npy", np.ones((48, 64), dtype=np.float32), allow_pickle=False)
        (root / camera / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": [[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]], "cam2world_cv": np.eye(4).tolist()}))
    session.last_observation_root, session.last_observation_clock = root, session._clock()
    session.last_observation_id = "robodojo:desk:layout-0:observation-0000"


def _reach(session):
    from ora.environment.backends.robodojo.reach_check import reach_check_batch

    return reach_check_batch(session, "left", [{"pose_xyzw": list(LINK6), "from_index": None}],
                             position_tolerance_m=0.005, orientation_tolerance_rad=0.05)["results"]


def test_a_play_session_starts_the_planner_its_plan_names_and_its_reach_check_answers_from_it(tmp_path, monkeypatch):
    from ora.environment.backends.robodojo import shared_motion

    planner = _Planner()
    config = {**PLANNER, "python_executable": "/curobo/bin/python", "root": "/curobo/src"}
    session, built, timing = _plant_session(tmp_path, monkeypatch, planner, motion_planner=config)
    session.start()
    assert session.motion_planner is planner and timing == [session.output_root]  # the timing preflight bound
    assert built == [{"python_executable": Path("/curobo/bin/python"), "root": Path("/curobo/src"),
                      "robot_file": Path("/x5/X5A.yml")}]
    binding = json.loads((session.output_root / "motion-planner/binding.json").read_text())
    assert binding["fk_parity"]["passed"] is True and binding["implementation"] == "curobo_v2"
    # No observation yet: the planner is there and waits for the world one publishes.
    assert shared_motion.plan_path(session, {"arm": "left", "goal_pose_xyz_xyzw": list(LINK6), "goal_joints": None}
                                   )["reason"] == "stale_observation"
    _observe(session, tmp_path / "observation")
    (row,) = _reach(session)
    assert (row["reachable"], row["reason"]) == (True, None) and len(planner.ik_requests) == 1
    assert planner.ik_requests[0]["start"]["joint_positions"] == JOINTS
    session.close()
    assert planner.closed == 1 and session.motion_planner is None


def test_a_play_plan_without_a_planner_starts_and_every_reach_row_says_no_planner(tmp_path, monkeypatch):
    session, built, timing = _plant_session(tmp_path, monkeypatch, None)
    session.start()
    assert session.motion_planner is None and built == [] and timing == [session.output_root]
    _observe(session, tmp_path / "observation")
    assert [(row["reachable"], row["reason"]) for row in _reach(session)] == [(False, "no_planner")]
    session.close()


@pytest.mark.parametrize("planner, message", [
    (_Planner(offset_m=0.01), "forward kinematics differs"),
    (_Planner(start_error=RuntimeError("motion planner worker readiness identity drift")), "readiness identity"),
])
def test_a_planner_that_does_not_start_or_disagrees_with_the_simulator_refuses_the_start(tmp_path, monkeypatch,
                                                                                          planner, message):
    session, _, _ = _plant_session(tmp_path, monkeypatch, planner, motion_planner={**PLANNER, "python_executable": "/p",
                                                                                     "root": "/r"})
    with pytest.raises(Exception, match=message):
        session.start()
    assert session.fault_reason == "robodojo_start_failed" and not session.started
    assert message in (session.output_root / "session-start-error.txt").read_text()
    session.close()  # a planner that started is closed with the session
    assert planner.closed == (1 if planner.start_error is None else 0)


def test_a_relaunch_drops_the_last_observations_world(tmp_path, monkeypatch):
    from ora.environment.backends.robodojo import shared_motion

    world = SimpleNamespace(reset=lambda seed: None, relaunch_for_layout=lambda seed: {"layout_id": seed},
                            step_lim=PLAY_NATIVE_ACTION_LIMIT, reward_registered=False)
    session = _settle_session(tmp_path, monkeypatch, world)  # the session's own _reset_environment
    monkeypatch.setattr(session, "_clock", lambda: 7)  # the new simulation's clock may read the last observation's
    monkeypatch.setattr(session, "_proprioception", lambda: {"end_effectors": {
        arm: {"arm_joint_qpos": list(JOINTS)} for arm in ("left", "right")}})
    session.last_observation_root, session.last_observation_clock = tmp_path / "observation", 7
    session._reset_environment("reset.txt")  # the start's reset keeps it
    assert session.last_observation_clock == 7
    assert session._reset_environment("relaunch.txt", relaunch_layout=0) == {"layout_id": 0}
    assert (session.last_observation_root, session.last_observation_clock) == (None, None)
    assert shared_motion.world_input(session, "left") == (None, JOINTS)
