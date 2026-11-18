"""Isolated Isaac boot and construction of one ORA RoboDojo play scene (worker only).

``build_play_environment`` is the play twin of ``backends/robodojo/bootstrap.
build_environment``. The Kit start is the benchmark's, call for call: the
same process-path binding, crash-report privacy, Vulkan compatibility layer,
node start gate, renderer device readback, CuRobo cache and resolved
environment configuration (``env_cfg/arx_x5.yml`` with its sim, scene, robot
and camera files and the RGB-D annotators). What differs is what a task would
supply:

- the source manifest (``play_source_manifest``) checks the same pinned
  revisions and clean checkouts and hashes the same robot and environment
  configuration files, but no task YAML or official ``Eval_Layout`` file: it
  records the scene's own source binding (its canonical index entry, its
  YAML and the layout generator: ``robodojo_play_scenes.source_binding``),
  which must equal the one the plan bound (``runtime.play_scene_source_sha256``,
  from the play-scene audit), hashes the fixture layout and every asset
  metadata file the generated layouts read, and regenerates each layout,
  which must equal the one the plan bound (``runtime.play_layouts``);
- the task config is the scene YAML's RoboDojo sections, the task name
  ``ora_play_<scene>``, the physics device and settle steps of the scene's
  index entry (``robodojo_play_scenes``), and PhysX stabilization is on (as
  upstream's teleop-sourced tabletop tasks run);
- the class is ``PlaySceneLifecycle`` over upstream ``TaskEnv``
  (``lifecycle.py``): no reward manager, no instruction, no official terminal.

What the worker reads of a scene is exactly what its source binding covers:
the scene's index entry (``play_scene``: name, set, physics device and YAML
path; ``settle_steps`` for the lifecycle), its YAML (``generate_layouts`` and
the task config above) and the generator. It reads no other scene's YAML and
no other entry's fields. The benchmark bootstrap is not imported or changed;
the play kernel identity (``ora/orchestrator/robodojo_play_kernel.py``) binds
both modules' code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping

PLAY_SOURCE_SCHEMA = "ora.robodojo-play-source-manifest.v2"
#: Upstream files a play scene's construction reads at the pinned revision (the benchmark's,
#: without the task YAML).
ENVIRONMENT_FILES = ("env_cfg/arx_x5.yml", "env_cfg/sim/sim_config.yml", "env_cfg/scene/default.yml",
                     "env_cfg/robot/dual_x5.yml", "env_cfg/camera/camera_config.yml",
                     "task/RoboDojo/config/_task.yml", "third_party/IsaacLab/apps/isaaclab.python.kit")
ROBOT_FILES = ("Robots/x5/robot_config.yml", "Robots/x5/curobo_tmp.yml", "Robots/x5/curobo.yml")
#: What a play scene changes in the resolved simulation configuration, recorded in the manifest.
PLAY_PHYSICS_OVERRIDES = {"sim.physx.enable_stabilization": True}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: The one physics override a worker config may carry: the play-scene audit's contact pass reads
#: PhysX contact reports, which only CPU physics delivers (``audit.py``). No plan can carry it.
AUDIT_PHYSICS_FIELD = "audit_physics_device"


def physics_device(config: Mapping[str, Any], definition: Any) -> str:
    """The scene's physics device, or the audit's CPU contact-pass override."""
    from ..bridge import RobodojoError

    override = config.get(AUDIT_PHYSICS_FIELD)
    if override is None:
        return str(definition.physics_device)
    if override != "cpu":
        raise RobodojoError("the play-scene audit overrides the physics device only with cpu")
    return "cpu"


def play_scene(config: Mapping[str, Any]) -> Any:
    """The scene definition this worker config names, checked against its runtime binding."""
    from ora.environment.benchmarks.robodojo_play_scenes import (
        PLAY_NATIVE_ACTION_LIMIT, PlaySceneSetError, scene_definition,
    )
    from ..bridge import RobodojoError

    try:
        definition = scene_definition(str(config.get("play_scene_set")), str(config.get("task")))
    except PlaySceneSetError as exc:
        raise RobodojoError(str(exc)) from None
    if config.get("physics_device") != physics_device(config, definition):
        raise RobodojoError(f"play scene {definition.name} runs on {definition.physics_device} physics")
    if config.get("native_action_limit") != PLAY_NATIVE_ACTION_LIMIT:
        raise RobodojoError("a play scene's native action limit is the play limit")
    return definition


def bound_scene_source(config: Mapping[str, Any], definition: Any) -> dict[str, Any]:
    """The scene's source binding, refused unless it is the one the plan carries (``play_scene_source_sha256``)."""
    from ora.environment.benchmarks.robodojo_play_scenes import source_binding
    from ..bridge import RobodojoError

    binding = source_binding(definition.scene_set, definition.name)
    if config.get("play_scene_source_sha256") != binding["sha256"]:
        raise RobodojoError(f"play scene {definition.name}'s sources (index entry, YAML or generator) differ "
                            "from the audited binding runtime.play_scene_source_sha256")
    return binding


def scene_task_config(definition: Any) -> dict[str, Any]:
    """The scene YAML's RoboDojo object blocks, as upstream's task config (``task_env``) receives them."""
    from ora.environment.benchmarks import robodojo_play_layouts as generator

    scene = generator.load_scene(definition.path)
    return {key: scene[key] for key in generator.SECTIONS if scene.get(key)}


def configured_layouts(config: Mapping[str, Any]) -> dict[int, str]:
    """``{seed: layout sha256}`` the plan bound (``runtime.play_layouts``), refused when absent or malformed."""
    from ..bridge import RobodojoError

    raw = config.get("play_layouts")
    if not isinstance(raw, Mapping) or not raw:
        raise RobodojoError("a play scene needs runtime.play_layouts (its audited seeds)")
    layouts: dict[int, str] = {}
    for seed, digest in raw.items():
        if not isinstance(seed, str) or not seed.isdigit() or not isinstance(digest, str) or len(digest) != 64:
            raise RobodojoError("runtime.play_layouts maps scene seeds to layout sha256")
        layouts[int(seed)] = digest
    return layouts


def generate_layouts(config: Mapping[str, Any], definition: Any, asset_root: Path,
                     output_root: Path | None = None) -> tuple[dict[int, Any], dict[str, Any]]:
    """Regenerate every bound seed's layout and require the bound bytes; write them as evidence."""
    from ora.environment.benchmarks.robodojo_play_scenes import layout_sha256
    from ..bridge import RobodojoError
    from ora.environment.benchmarks import robodojo_play_layouts as generator

    spec = generator.load_scene(definition.path)
    loaded: dict[int, Any] = {}
    records: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    for seed, expected in sorted(configured_layouts(config).items()):
        result = generator.generate(spec, scene=definition.name, seed=seed, asset_root=asset_root)
        digest = layout_sha256(result["layout"])
        if digest != expected:
            raise RobodojoError(f"generated layout of seed {seed} differs from the audited layout")
        loaded[seed] = result["layout"]
        metadata.update({path: _sha256(data) for path, data in result["metadata"].items()})
        record = {"seed": seed, "layout_sha256": digest, "placements": result["placements"],
                  "keep_out": result["keep_out"]}
        if output_root is not None:
            directory = Path(output_root) / "play-layouts"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"seed-{seed}.json").write_text(json.dumps(result["layout"], indent=1, sort_keys=True) + "\n")
            (directory / f"seed-{seed}.placements.json").write_text(
                json.dumps(record, indent=1, sort_keys=True) + "\n")
            record["path"] = f"play-layouts/seed-{seed}.json"
        records[str(seed)] = record
    return loaded, {"layouts": records, "asset_metadata": dict(sorted(metadata.items()))}


def play_source_manifest(config: Mapping[str, Any], *, output_root: Path | None = None
                         ) -> tuple[dict[str, Any], dict[int, Any], Any]:
    """The play scene's source manifest, its generated layouts and its definition."""
    import subprocess

    from ora.environment.benchmarks.robodojo import ACTION_CODECS, CONTROL_HZ, PHYSICS_DT_S, SENSOR_PROFILE
    from ora.environment.benchmarks.robodojo_play_scenes import PLAY_NATIVE_ACTION_LIMIT
    from ..bridge import RobodojoError, revision
    from .operator_view import OPERATOR_CAMERA, OPERATOR_VIEW, operator_view_mode
    from ora.environment.benchmarks.robodojo_play_layouts import FIXTURE_LAYOUT

    definition = play_scene(config)
    scene_source = bound_scene_source(config, definition)
    root = Path(config["robodojo_root"]).resolve()
    assets = Path(config["asset_root"]).resolve()
    roots = {"robodojo": root, "isaaclab": root / "third_party/IsaacLab", "curobo": root / "third_party/curobo",
             "xpolicylab": root / "XPolicyLab", "assets": assets}
    required_keys = {"robodojo": "required_revision", "isaaclab": "required_isaaclab_revision",
                     "curobo": "required_curobo_revision", "xpolicylab": "required_xpolicylab_revision",
                     "assets": "required_asset_revision"}
    actual, required = {}, {}
    for name, path in roots.items():
        expected = str(config.get(required_keys[name], ""))
        if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
            raise RobodojoError(f"missing exact source pin: {required_keys[name]}")
        actual[name], required[name] = revision(path), expected
        if actual[name] != expected:
            raise RobodojoError(f"source revision mismatch: {name}")
        dirty = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--", "."], capture_output=True, check=True)
        if dirty.stdout:
            raise RobodojoError(f"tracked source is modified: {name}")
    codec = config.get("action_codec")
    if codec not in ACTION_CODECS:
        raise RobodojoError("unsupported RoboDojo first-stage binding: action_codec")
    for field, pinned in (("eval_seed", 0), ("num_envs", 1), ("sensor_profile", SENSOR_PROFILE),
                          ("physics_dt", PHYSICS_DT_S), ("control_hz", CONTROL_HZ)):
        if config.get(field, pinned) != pinned:
            raise RobodojoError(f"unsupported RoboDojo play binding: {field}")
    loaded_files = {}
    for relative in ROBOT_FILES:
        path = assets / relative
        if not path.is_file():
            raise RobodojoError(f"required robot configuration is missing: {relative}")
        loaded_files["Assets/" + relative] = _sha256(path.read_bytes())
    template = (assets / "Robots/x5/curobo_tmp.yml").read_text()
    expected_curobo = template.replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
    if (assets / "Robots/x5/curobo.yml").read_text() != expected_curobo:
        raise RobodojoError("derived CuRobo configuration differs from exact path-template expansion")
    for relative in ENVIRONMENT_FILES:
        loaded_files[relative] = _sha256((root / relative).read_bytes())
    loaded_files["Assets/" + FIXTURE_LAYOUT] = _sha256((assets / FIXTURE_LAYOUT).read_bytes())
    layouts, generated = generate_layouts(config, definition, assets, output_root)
    loaded_files.update({"Assets/" + path: digest for path, digest in generated["asset_metadata"].items()})
    packages = {}
    for package in ("isaacsim", "torch", "numpy", "warp-lang", "nvidia-curobo", "starlette"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    if not str(packages["isaacsim"]).startswith("5.1.") or sys.version_info[:2] != (3, 11):
        raise RobodojoError("the isolated worker requires Python 3.11 and Isaac Sim 5.1")
    play_seed = config.get("play_seed")
    if type(play_seed) is not int or play_seed not in layouts:
        raise RobodojoError("the play seed must be one of the plan's audited seeds")
    manifest = {
        "schema": PLAY_SOURCE_SCHEMA, "task": definition.name, "scene_set": definition.scene_set,
        "play_seed": play_seed, "eval_seed": 0, "layout_id": 0, "num_envs": 1, "sensor_profile": SENSOR_PROFILE,
        "action_codec": codec, "physics_dt_s": PHYSICS_DT_S, "control_hz": CONTROL_HZ,
        "play_native_action_limit": PLAY_NATIVE_ACTION_LIMIT, "physics_device": physics_device(config, definition),
        "scene_physics_device": definition.physics_device,
        "settle_steps": definition.settle_steps,
        "scene_source": scene_source,
        "play_layouts": generated["layouts"],
        "source_revisions": actual, "required_revisions": required,
        "loaded_files": loaded_files, "installed_packages": packages,
        "play_physics_overrides": dict(PLAY_PHYSICS_OVERRIDES),
        # The operator's third-person camera (operator_view.py): never a policy camera.
        "operator_view": {"mode": operator_view_mode(config), "camera": OPERATOR_CAMERA,
                          "view": {key: list(value) if isinstance(value, tuple) else value
                                   for key, value in OPERATOR_VIEW.items()}},
    }
    return manifest, layouts, definition


def build_play_environment(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Launch one source-verified Isaac application and one play scene (see the module docstring)."""
    from ..bootstrap import KIT_INIT_GATE_TIMEOUT_S, bind_process_paths, nvidia_smi
    from ..bridge import (RobodojoError, _CRASH_REPORT_SETTINGS, _configure_crash_reporting,
                          _crash_reporting_readback, renderer_table_binding)
    from .lifecycle import play_scene_class

    runtime_directory = Path(config["runtime_directory"]).resolve()
    if not runtime_directory.is_absolute() or runtime_directory == Path("/"):
        raise RobodojoError("runtime_directory must be an explicit non-root path")
    bind_process_paths(runtime_directory)
    omni_config = runtime_directory / "config/omniverse"
    omni_config.mkdir(parents=True, exist_ok=True)
    os.environ["OMNI_CONFIG_PATH"] = str(omni_config)
    omni_paths = {"data_root": runtime_directory / "data/ov", "cache_root": runtime_directory / "cache/ov",
                  "logs_root": runtime_directory / "logs/ov"}
    for directory in omni_paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    config_text = "[paths]\n" + "".join(f"{key} = {json.dumps(str(value))}\n" for key, value in omni_paths.items())
    omni_config_file = omni_config / "omniverse.toml"
    if omni_config_file.exists():
        if tomllib.loads(omni_config_file.read_text()).get("paths") != {key: str(value) for key, value in omni_paths.items()}:
            raise RobodojoError("dedicated Omniverse path configuration differs from the runtime binding")
    else:
        omni_config_file.write_text(config_text)
    if config.get("accept_eula") is True:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    output_root = Path(config["output_root"]).resolve()
    crash_report_args, crash_privacy_file = _configure_crash_reporting(output_root)
    manifest, layouts, definition = play_source_manifest(config, output_root=output_root)
    from ..graphics_compat import activate, verify_activation
    from .. import start_gate

    init_gate = start_gate.gate("kit-init", start_gate.INIT_SLOTS, timeout_s=KIT_INIT_GATE_TIMEOUT_S)
    gate_record = init_gate.__enter__()
    graphics = activate(runtime_directory)
    manifest["graphics_compatibility"] = graphics
    manifest["loaded_files"].update({f"graphics-compat/{role}": reference["sha256"]
                                     for role, reference in graphics["files"].items()})
    root = Path(config["robodojo_root"]).resolve()
    sys.path.insert(0, str(root))
    globals_module = importlib.import_module("env.global_configs")
    assets = str(Path(config["asset_root"]).resolve())
    globals_module.ASSETS_PATH = assets
    globals_module.ROBOTS_PATH = str(Path(assets) / "Robots")
    globals_module.OBJECTS_PATH = str(Path(assets) / "Object/RoboDojo")
    from isaaclab.app import AppLauncher

    startup_log = output_root / "kit-startup.log"
    if startup_log.exists():
        raise RobodojoError("refusing to reuse an existing Kit startup log")
    startup_log.parent.mkdir(parents=True, exist_ok=True)
    experience = root / "third_party/IsaacLab/apps/isaaclab.python.kit"
    if not experience.is_file():
        raise RobodojoError("the pinned IsaacLab default experience is missing")
    device = physics_device(config, definition)
    launcher = AppLauncher(
        headless=True, enable_cameras=True, fast_shutdown=False, device=device,
        experience=str(experience),
        kit_args=(f"--/renderer/activeGpu={int(config.get('renderer_gpu', 0))} --/renderer/multiGpu/enabled=false "
                  f"--/renderer/multiGpu/maxGpuCount=1 --/renderer/activeCudaGpus=0 --/renderer/useCudaDeviceOrder=true "
                  f"--/plugins/carb.tasking.plugin/suppressTaskException=false "
                  f"--/log/file={startup_log} --portable-root {runtime_directory / 'kit'} {crash_report_args}"),
    )
    app = launcher.app
    try:
        import carb.settings

        verify_activation(runtime_directory, graphics)
        init_gate.__exit__(None, None, None)
        manifest["start_gate"] = gate_record

        import omni.structuredlog

        settings = carb.settings.get_settings()
        privacy_before = {key: settings.get(key) for key in _CRASH_REPORT_SETTINGS}
        settings.set_string("/structuredLog/privacySettingsFile", str(crash_privacy_file))
        if not omni.structuredlog.IStructuredLogSettings().load_privacy_settings():
            raise RobodojoError("SDK could not load the explicit episode privacy file")
        for category in ("performance", "personalization", "usage"):
            settings.set_bool(f"/privacy/{category}", False)
        manifest["crash_reporting"] = _crash_reporting_readback(settings, crash_privacy_file)
        manifest["crash_reporting"]["before_explicit_privacy_load"] = privacy_before
        settings.set_bool("/isaaclab/render/rtx_sensors", True)
        manifest["render_settings"] = {key: settings.get(key) for key in (
            "/isaaclab/cameras_enabled", "/isaaclab/render/offscreen",
            "/isaaclab/render/rtx_sensors", "/isaaclab/render/active_viewport",
            "/app/asyncRendering", "/omni/replicator/asyncRendering",
        )}
        if (manifest["render_settings"]["/isaaclab/render/offscreen"] is not True
                or manifest["render_settings"]["/isaaclab/render/rtx_sensors"] is not True):
            raise RobodojoError("RoboDojo external RGB-D sensors require offscreen RTX rendering")
        import torch
        import carb.tokens

        tokens = carb.tokens.get_tokens_interface()
        resolved_paths = {}
        for name in ("omni_global_cache", "omni_global_data", "omni_global_logs", "cache", "data", "logs"):
            resolved = tokens.resolve("${" + name + "}")
            path = Path(resolved)
            if not resolved or "${" in resolved or not path.is_absolute() or not path.resolve().is_relative_to(runtime_directory):
                raise RobodojoError(f"Kit runtime path escapes the dedicated runtime: {name}")
            resolved_paths[name] = str(path.resolve())
        manifest["runtime_paths"] = {"kit_tokens": resolved_paths, "runtime_directory": str(runtime_directory),
                                     "xdg_paths": {key: os.environ[key] for key in (
                                         "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR")},
                                     "omniverse_configuration": str(omni_config_file),
                                     "optix_cache": os.environ["OPTIX_CACHE_PATH"]}
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RobodojoError("the isolated worker must have exactly one logical CUDA device")
        gpu_output = nvidia_smi(["--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"])
        inventory = []
        for line in gpu_output.splitlines():
            index, gpu_uuid, pci_bus_id = [item.strip() for item in line.split(",")]
            inventory.append({"index": int(index), "uuid": gpu_uuid, "pci_bus_id": pci_bus_id})
        log_prefix = startup_log.read_bytes()
        device_binding = renderer_table_binding(
            log_prefix.decode("utf-8", errors="replace"), selected_index=int(config["renderer_gpu"]),
            cuda_uuid=str(torch.cuda.get_device_properties(0).uuid), inventory=inventory,
        )
        prefix_path = startup_log.with_name("kit-startup-readback.log")
        with prefix_path.open("xb") as stream:
            stream.write(log_prefix)
        prefix_path.chmod(0o444)
        device_binding["startup_log_prefix"] = {"path": prefix_path.name, "sha256": _sha256(log_prefix)}
        manifest["device_binding"] = device_binding
        import curobo._src.runtime as curobo_runtime

        curobo_cache = runtime_directory / "cache/curobo"
        curobo_cache.mkdir(parents=True, exist_ok=True)
        curobo_runtime.cache_dir = str(curobo_cache)
        manifest["runtime_paths"]["curobo_cache"] = curobo_runtime.cache_dir
        from omegaconf import OmegaConf
        from utils.pipeline_utils import process_config, process_randomization

        task_name = f"ora_play_{definition.name}"
        eval_config = OmegaConf.load(root / "env_cfg/arx_x5.yml")
        eval_config.update({"task_name": task_name, "seed": 0, "num_envs": 1})
        composed = OmegaConf.create({
            name: OmegaConf.load(root / f"env_cfg/{name}/{eval_config.config[name]}.yml")
            for name in ("sim", "scene", "robot", "camera")
        })
        composed.task_env = OmegaConf.create(scene_task_config(definition))
        composed.eval_cfg = eval_config
        composed = process_randomization(composed)
        composed, _ = process_config(composed, task_name=task_name)
        composed.sim.setdefault("physx", {})
        composed.sim.physx.enable_stabilization = True
        composed.sim.scene.num_envs = 1
        composed.sim.device = device
        composed.sim.seed = [0]
        composed.camera.default_frequency = 25
        composed.eval_cfg.observation.vision.update({
            "depth": True, "approximate_depth": False, "intrinsic_matrix": True, "extrinsic_matrix": True,
        })
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            composed.camera.annotator[camera].distance_to_image_plane_capture = {
                "type": "distance_to_image_plane", "device": "cpu",
            }
        if float(composed.sim.dt) != 0.004 or int(composed.sim.decimation) != 1:
            raise RobodojoError("upstream task physics cadence has changed")
        from env.environment.task_env import TaskEnv

        cls = play_scene_class(TaskEnv, layouts=layouts, settle_steps=definition.settle_steps,
                               step_limit=int(config["native_action_limit"]),
                               operator_view_mode=manifest["operator_view"]["mode"])
        from ..reset_diagnostics import stack_timer

        with (output_root / "environment-initialization-stack.txt").open("w") as stack_log, \
                stack_timer(120, stack_log):
            environment = cls(composed, app)
        environment.harness_output_root = output_root
        manifest["camera_capture"] = {
            "adapter": "single_environment_numpy_rgbd_v1", "num_envs": 1,
            "original_annotators": True, "pixels_resampled": False,
            "buffer_readback": "camera-buffer-readback.json",
            "texture_wait": "after_original_capture_reset_before_first_observation",
            "texture_wait_timeout_s": 180,
            "texture_wait_evidence": "render-readiness.jsonl",
            "intrinsics": "original_render_product_camera_params_projection",
            "frame_synchronization": "current_physics_time_and_common_reference_time_v1",
            "frame_synchronization_max_renders": 8,
            "frame_synchronization_min_renders": 2,
            "frame_synchronization_evidence": "render-synchronization.jsonl",
            "optical_pose_check": "render_product_matches_current_physical_prim_atol_1e-4",
        }
        manifest["resolved_environment_config"] = OmegaConf.to_container(composed, resolve=True)
        manifest["evaluator_input_adapter"] = None
        manifest["held_control_adapter"] = "per_step_resolved_meta_control_v1"
        return environment, app, manifest
    except BaseException:
        import traceback

        diagnostic = traceback.format_exc()
        (output_root / "startup-error.txt").write_text(diagnostic)
        print(diagnostic, file=sys.stderr, flush=True)
        init_gate.__exit__(None, None, None)
        # No in-process Kit close: a crashing Kit teardown can leave the card's Vulkan state
        # unusable. The worker process ends without it (session.close, play_scenes/session.py).
        raise


__all__ = ["AUDIT_PHYSICS_FIELD", "ENVIRONMENT_FILES", "PLAY_PHYSICS_OVERRIDES", "PLAY_SOURCE_SCHEMA",
           "bound_scene_source", "build_play_environment", "configured_layouts", "generate_layouts", "physics_device",
           "play_scene", "play_source_manifest", "scene_task_config"]
