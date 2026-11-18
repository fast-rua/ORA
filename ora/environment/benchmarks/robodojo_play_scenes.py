"""Host-safe registry of ORA's own RoboDojo free-play scenes: Isaac Sim tabletops, no task.

A scene set is a named group of scenes ORA owns. Each scene is a RoboDojo
task-config YAML of RoboDojo library objects (``Articulation``, ``Clutter`` and
``Rigid`` blocks) on RoboDojo's standard table with the dual ARX X5
embodiment and its three cameras. It has no reward, no success predicate, no
instruction and no step limit of its own: it exists only to be played in.
Public RoboDojo has no layout sampler, so ORA's generator
(``benchmarks/robodojo_play_layouts.py``, host-safe pure Python) turns the YAML and a seed into
a saved layout in RoboDojo's ``Eval_Layout`` record format, which the
unmodified scene manager replays. The isolated worker loads a scene through
its own backend id (``robodojo_play``, ``ora/environment/robodojo_play_factory.py``)
when the plan's runtime names the set (``runtime.play_scene_set``, bound only by
``play_scenes.bind_play_runtime`` in ``ora/orchestrator``); the RoboDojo checkout,
its pinned revision and the benchmark worker kernel are not changed.

This module is host-safe and code only: it never imports the simulator or the
worker package, and it holds no scene. A set's scenes are data, in the set's
``index.yml`` beside their YAMLs (``SCENE_SET_DIRECTORIES``): one entry per
scene (label, physics device, settle steps, assets, the operator's
description). A host offers a scene only at a seed a local play-scene audit
(``robodojo_play_scene_audit``, ``ora/orchestrator/robodojo_play_audit.py``)
admits for the current play kernel (``ora/orchestrator/robodojo_play_kernel.py``:
this module, the generator and the play modules, no scene data) and for the
scene's own source binding (``source_binding``: its canonical entry, its YAML
and the generator). So an edit to one scene's entry or YAML retires that
scene's audits and no other scene's.

``isaac-tabletop-v1`` is the default scene set of ``ora play --env robodojo``:
RoboDojo has no other play scenes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: The ORA repository root (this module is ``ora/environment/benchmarks/robodojo_play_scenes.py``).
ROOT = Path(__file__).resolve().parents[3]
SOURCE_SCHEMA = "ora.robodojo-play-scene-source.v2"
#: The worker module that turns a scene YAML and a seed into a saved layout.
GENERATOR_MODULE = "ora/environment/benchmarks/robodojo_play_layouts.py"
#: Where each scene set's YAMLs live (one directory per set).
SCENE_ROOT = "ora/environment/backends/robodojo/play_scenes/scenes"
#: Each scene set's directory under ``SCENE_ROOT``; its ``INDEX_FILE`` lists the set's scenes.
SCENE_SET_DIRECTORIES = {"isaac-tabletop-v1": "isaac_tabletop_v1"}
INDEX_FILE = "index.yml"
#: The fields of a scene's index entry (``SceneDefinition`` adds the set and the YAML path).
ENTRY_FIELDS = ("name", "label", "physics_device", "settle_steps", "assets", "description")
#: Entry fields no scene source binding covers: the operator's text, read by ``ora play`` only
#: (never by the worker, the audit or a plan).
UNBOUND_ENTRY_FIELDS = ("description",)
DEFAULT_SCENE_SET = "isaac-tabletop-v1"
#: The backend id the RoboDojo worker dispatches a play scene on (``environment/worker.py``).
WORKER_BACKEND = "robodojo_play"
#: A play scene's native-action limit: its environment's own step limit. ``ora play``'s
#: session budget (``max_sim_steps`` 50000, ten physics steps per native action) fits it exactly.
PLAY_NATIVE_ACTION_LIMIT = 5000
#: A play plan's ``runtime.execution_capture``: the capture's highest rate, so the operator's page and a
#: video of the run show the motion (an object knocked over, falling, rolling) rather than a few jumps. The
#: capture samples simulated time on the 4 ms physics grid, so 30 fps is one frame every ninth physics
#: step (27.8 frames per simulated second). Benchmark plans keep the default 10 fps.
PLAY_EXECUTION_CAPTURE = {"enabled": True, "fps": 30}
#: Physics devices a scene may declare (``robodojo_physics``): ``cpu`` for a scene with an
#: articulation (RoboDojo's ``ArticulationObject.initialize`` reads joint properties through
#: numpy and crashes on CUDA tensors), ``cuda:0`` for a rigid-only scene.
CPU_PHYSICS_DEVICE = "cpu"
CUDA_PHYSICS_DEVICE = "cuda:0"
#: RoboDojo's table top in the episode frame (``env_cfg/scene/default.yml``: 0.74 + 0.05 / 2)
#: and the table's x/y extent (1.4 x 1.1 m centred at (0, -0.05)).
TABLE_TOP_Z_M = 0.765
TABLE_XY_M = {"x": (-0.70, 0.70), "y": (-0.60, 0.50)}
#: The fixed ARX X5 base mounts (x, y) of ``env_cfg/robot/dual_x5.yml`` in the episode frame.
X5_BASES_XY_M = {"left": (-0.3, -0.45), "right": (0.3, -0.45)}
#: Where an audited object must rest: the union of the x/y ranges the official dual-X5 task
#: configs sample objects from and of the object centres in the official saved layouts. An object
#: resting in or on another one (its ``relative_plane`` names a label) is checked through its
#: table-level support: a pen leaning out of its holder is where the holder is.
PLAY_REACH_ENVELOPE_XY_M = {"x": (-0.45, 0.45), "y": (-0.30, 0.10)}
#: The largest centre-to-nearer-base distance among the official dual-X5 layout objects.
MAX_NEARER_BASE_DISTANCE_M = 0.613
#: The play-scene audit's worker contract, shared by the worker (``backends/robodojo/play_scenes/
#: audit.py``) and its host launcher (``ora/orchestrator/robodojo_play_audit.py``), which must not
#: import a process-isolated simulator backend. ``session`` runs on the scene's own physics device;
#: a CUDA scene adds ``contacts`` on CPU physics, where PhysX delivers contact reports.
AUDIT_PASSES = ("session", "contacts")
AUDIT_SEED_SCHEMA = "ora.robodojo-play-scene-audit-seed.v1"
AUDIT_WORKER_REPORT_SCHEMA = "ora.robodojo-play-scene-audit-worker.v1"
AUDIT_WORKER_REPORT_NAME = "audit-worker.json"
#: The audit's thresholds; every record carries them as ``criteria``.
AUDIT_CRITERIA = {
    "window_steps": 500,
    "max_settle_drift_m": 0.03,
    "max_settle_tilt_deg": 20.0,
    "left_table_below_top_m": 0.02,
    "max_window_displacement_m": 0.005,
    "max_window_peak_m": 0.01,
    "max_window_rotation_deg": 2.0,
    "max_joint_drift": 0.02,
    "max_penetration_m": 0.003,
    "min_gripper_clearance_m": 0.03,
    # IK reachability: align_plan's own poses (``ora/agent/tools/alignment.py``) along each declared
    # approach side, link6 standing off the object's near face so the open fingertips stop this far
    # short of it, the jaw rolled about the approach axis in these steps over the full turn.
    "ik_fingertip_clearance_m": 0.02,
    # When no side reaches an object: these sides' approaches tilted by these degrees, each planned from the
    # measured joints. Both come from the arms' side of the table: the top approach leans toward +y (azimuth
    # 90 degrees from +x), so the wrist stays nearer the bases, and the right approach (travelling +y) tilts
    # down. Of 64 tilted plans per object (10 and 20 degrees, 8 directions, 2 rolls, 2 arms) for the 24 objects
    # four audited scenes rejected, every object any of them reached was reached by one of these at roll 0; the
    # one left, the desk's pen holder at the far left corner, is reached at 30 degrees by the top approach.
    "tilt_approach_deg": [10.0, 20.0, 30.0],
    "tilt_sides": ["top", "right"],
    "tilt_top_azimuths_deg": [90.0],
    "tilt_rolls_deg": [0.0],
    "ik_roll_step_deg": 30.0,
    "min_image_std": 2.0,
    "physics_dt_s": 0.004,
}


class PlaySceneSetError(ValueError):
    """An unknown scene set or scene."""


@dataclass(frozen=True)
class SceneDefinition:
    """One free-play scene, from its set's index entry: what loads, for the operator and the audit."""

    scene_set: str
    name: str
    label: str
    #: The scene YAML, relative to the ORA repository root (``<set directory>/<name>.yml``).
    scene_file: str
    #: The IsaacLab physics device the scene runs on (an articulation needs ``cpu``).
    physics_device: str
    #: Physics steps (4 ms) the worker runs after each reset before the first observation.
    settle_steps: int
    #: ``<section>/<category>`` of every RoboDojo library object the scene places.
    assets: tuple[str, ...]
    #: What the operator is told the scene holds (``ora play``); never sent to the policy.
    description: str

    @property
    def path(self) -> Path:
        return ROOT / self.scene_file


def _set_directory(scene_set: str) -> str:
    try:
        return SCENE_SET_DIRECTORIES[scene_set]
    except KeyError:
        raise PlaySceneSetError(
            f"unknown RoboDojo play scene set {scene_set!r}; one of {list(scene_sets())}") from None


def index_file(scene_set: str) -> str:
    """The set's index (its scenes' entries), relative to the ORA repository root."""
    return f"{SCENE_ROOT}/{_set_directory(scene_set)}/{INDEX_FILE}"


def _entries(scene_set: str, root: Path | None) -> list[dict[str, Any]]:
    import yaml

    path = (ROOT if root is None else root) / index_file(scene_set)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlaySceneSetError(f"RoboDojo play scene set {scene_set} has no index: {exc}") from None
    entries = data.get("scenes") if isinstance(data, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise PlaySceneSetError(f"{index_file(scene_set)} lists no scenes")
    names = [entry.get("name") if isinstance(entry, Mapping) else None for entry in entries]
    if len(set(names)) != len(names):
        raise PlaySceneSetError(f"{index_file(scene_set)} names a scene twice")
    return entries


def _definition(scene_set: str, entry: Mapping[str, Any]) -> SceneDefinition:
    import re

    where = f"{index_file(scene_set)} entry {entry.get('name')!r}" if isinstance(entry, Mapping) else index_file(scene_set)
    if not isinstance(entry, Mapping) or set(entry) != set(ENTRY_FIELDS):
        raise PlaySceneSetError(f"{where} must have exactly the fields {list(ENTRY_FIELDS)}")
    name, settle, assets = entry["name"], entry["settle_steps"], entry["assets"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise PlaySceneSetError(f"{where}: a scene name is lower-case letters, digits and underscores")
    if entry["physics_device"] not in (CPU_PHYSICS_DEVICE, CUDA_PHYSICS_DEVICE):
        raise PlaySceneSetError(f"{where}: physics_device is {CPU_PHYSICS_DEVICE} or {CUDA_PHYSICS_DEVICE}")
    if isinstance(settle, bool) or not isinstance(settle, int) or settle <= 0:
        raise PlaySceneSetError(f"{where}: settle_steps is a positive integer")
    if (not isinstance(assets, list) or not assets or len(set(assets)) != len(assets)
            or not all(isinstance(item, str) and item.count("/") == 1 for item in assets)):
        raise PlaySceneSetError(f"{where}: assets lists distinct <section>/<category> names")
    if not all(isinstance(entry[field], str) and entry[field].strip() for field in ("label", "description")):
        raise PlaySceneSetError(f"{where}: label and description are text")
    return SceneDefinition(scene_set, name, entry["label"], f"{SCENE_ROOT}/{_set_directory(scene_set)}/{name}.yml",
                           entry["physics_device"], settle, tuple(assets), entry["description"])


def scene_sets() -> tuple[str, ...]:
    return tuple(sorted(SCENE_SET_DIRECTORIES))


def scene_definitions(scene_set: str, *, root: Path | None = None) -> tuple[SceneDefinition, ...]:
    """The set's scenes, in its index's order (read from ``root``'s index, default ``ROOT``, on every call)."""
    _set_directory(scene_set)
    return tuple(_definition(scene_set, entry) for entry in _entries(scene_set, root))


def scene_definition(scene_set: str, name: str, *, root: Path | None = None) -> SceneDefinition:
    _set_directory(scene_set)
    for entry in _entries(scene_set, root):
        if isinstance(entry, Mapping) and entry.get("name") == name:
            return _definition(scene_set, entry)
    raise PlaySceneSetError(f"RoboDojo play scene set {scene_set} has no scene {name!r}")


def find_scene(name: str, *, root: Path | None = None) -> SceneDefinition | None:
    """The scene of that name in any set (names are unique across sets), or None."""
    for scene_set in scene_sets():
        for definition in scene_definitions(scene_set, root=root):
            if definition.name == name:
                return definition
    return None


def scene_language(definition: SceneDefinition) -> str:
    """The scene's stand-in for a task language: what the worker reports and the plan binds.

    A RoboDojo plan binds ``runtime.expected_task_language`` and the worker
    reads its instruction back at reset. A play scene has no instruction, so
    both sides carry this fixed text instead; the policy never receives it
    (free play withholds the task layer).
    """
    return f"ORA free-play scene {definition.scene_set}/{definition.name}: no task."


def canonical_entry(definition: SceneDefinition) -> dict[str, Any]:
    """The scene's bound identity: its index entry without ``UNBOUND_ENTRY_FIELDS``, plus set and YAML."""
    return {"scene_set": definition.scene_set, "name": definition.name, "label": definition.label,
            "scene_file": definition.scene_file, "physics_device": definition.physics_device,
            "settle_steps": definition.settle_steps, "assets": list(definition.assets)}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_files(scene_set: str, scene: str, *, root: Path | None = None) -> tuple[str, ...]:
    """The ORA files whose bytes define one scene: its YAML and the layout generator (repository-relative)."""
    return tuple(sorted({GENERATOR_MODULE, scene_definition(scene_set, scene, root=root).scene_file}))


def source_binding(scene_set: str, scene: str, *, root: Path | None = None) -> dict[str, Any]:
    """``{schema, scene_set, scene, entry, entry_sha256, files: [{path, sha256}], sha256}`` of one scene.

    Everything of ORA's that decides what one scene's worker builds, apart
    from the play kernel's code: the scene's canonical index entry (its
    physics device, settle steps, label and assets), its YAML (the generator
    turns it into each seed's layout, and the worker also hands its object
    blocks to upstream's task config) and the generator. Another scene's
    entry or YAML is not in it. An audit records it, the worker records it in
    its source manifest and a plan binds its digest
    (``runtime.play_scene_source_sha256``).
    """
    definition = scene_definition(scene_set, scene, root=root)
    entry = canonical_entry(definition)
    base = ROOT if root is None else root
    files = [{"path": relative, "sha256": hashlib.sha256((base / relative).read_bytes()).hexdigest()}
             for relative in source_files(scene_set, scene, root=root)]
    body = {"schema": SOURCE_SCHEMA, "scene_set": scene_set, "scene": scene, "entry": entry,
            "entry_sha256": _canonical_sha256(entry), "files": files}
    return {**body, "sha256": _canonical_sha256(body)}


def layout_sha256(layout: Any) -> str:
    """The content identity of one generated saved layout (canonical JSON)."""
    return hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_play_runtime(runtime: Any) -> None:
    """Admit a RoboDojo play-scene runtime (``runtime.play_scene_set``) at RunSpec resolution.

    ``definitions.validate_runtime_extension`` routes a RoboDojo runtime that
    names a scene set here instead of the benchmark rule
    (``benchmarks/robodojo.validate_runtime_extension``), which admits only
    the official tasks. Everything a benchmark runtime pins stays pinned the
    same way; only the task identity, the layout and the step limit are the
    scene's.
    """
    from .robodojo import (
        PUBLIC_CAMERAS, REVISIONS, REVISION_FIELDS, RoboDojoBenchmarkError, source_revisions,
        validate_motion_profile_binding,
    )

    allowed = {
        "kind", "transport", "root_env", "python_env", "asset_root_env", *REVISION_FIELDS.values(),
        "required_host", "require_gpu", "eval_seed", "num_envs", "task_variant", "env_config",
        "sensor_profile", "physics_dt", "control_hz", "native_action_limit", "native_actions_per_command",
        "physics_device", "renderer_gpu", "cuda_visible_devices", "main_camera", "policy_cameras",
        "startup_timeout_s", "request_timeout_s", "close_timeout_s", "workspace_bounds_m",
        "environment_signal_policy", "execution_capture", "public_environment_context",
        "public_geometry_descriptor", "resource_binding", "expected_task_language", "workspace_profile",
        "environment_prompt_profile", "contact_approach_route", "recovery_route", "motion_planner", "motion_timing",
        # The ARX X5 motion profile the toolkit runs (benchmarks/robodojo.validate_motion_profile_binding).
        "motion_profile",
        # The play binding (play_scenes.bind_play_runtime).
        "play_scene_set", "play_layouts", "play_kernel_sha256", "play_scene_source_sha256",
    }
    unexpected = set(runtime) - allowed
    if unexpected:
        raise RoboDojoBenchmarkError(f"unsupported RoboDojo play runtime fields: {sorted(unexpected)}")
    try:
        definition = scene_definition(str(runtime.get("play_scene_set")), str(runtime.get("task_variant")))
    except PlaySceneSetError as exc:
        raise RoboDojoBenchmarkError(str(exc)) from None
    if runtime.get("transport") != "local_rpc":
        raise RoboDojoBenchmarkError("RoboDojo requires local_rpc")
    if source_revisions(runtime) != REVISIONS:
        raise RoboDojoBenchmarkError("RoboDojo play requires the reviewed code/fork/asset revisions")
    validate_motion_profile_binding(runtime.get("motion_profile"))
    exact = {
        "eval_seed": 0, "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1",
        "physics_dt": 0.004, "control_hz": 25, "native_action_limit": PLAY_NATIVE_ACTION_LIMIT,
        "native_actions_per_command": 20, "main_camera": "head", "require_gpu": True,
        "physics_device": definition.physics_device, "policy_cameras": list(PUBLIC_CAMERAS),
        "expected_task_language": scene_language(definition),
    }
    for key, expected in exact.items():
        value = runtime.get(key)
        if value != expected or isinstance(value, bool) != isinstance(expected, bool):
            raise RoboDojoBenchmarkError(f"runtime.{key} of a RoboDojo play scene must be {expected!r}")
    device = runtime.get("cuda_visible_devices")
    if not isinstance(device, str) or len(device) != 1 or device not in "01234567":
        raise RoboDojoBenchmarkError("runtime.cuda_visible_devices must select one GPU")
    if type(runtime.get("renderer_gpu")) is not int or runtime["renderer_gpu"] != int(device):
        raise RoboDojoBenchmarkError("runtime.renderer_gpu must select the physical GPU")
    for field in ("startup_timeout_s", "request_timeout_s", "close_timeout_s"):
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise RoboDojoBenchmarkError(f"runtime.{field} must be positive and finite")
    from ora.environment.workspace_profiles import ROBODOJO_ARX_X5_TABLETOP_V1
    from .robodojo import public_context

    profile = runtime.get("workspace_profile")
    if profile is not None and (not isinstance(profile, Mapping)
                                or profile.get("profile_id") != ROBODOJO_ARX_X5_TABLETOP_V1
                                or profile.get("environment_id") != "robodojo"):
        raise RoboDojoBenchmarkError(f"RoboDojo admits only the {ROBODOJO_ARX_X5_TABLETOP_V1} workspace profile")
    public = runtime.get("public_environment_context")
    if public is not None and public != public_context():
        raise RoboDojoBenchmarkError("RoboDojo public environment context drift")
    layouts = runtime.get("play_layouts")
    if layouts is not None and (not isinstance(layouts, Mapping) or not layouts or not all(
            isinstance(seed, str) and seed.isdigit() and isinstance(digest, str) and len(digest) == 64
            for seed, digest in layouts.items())):
        raise RoboDojoBenchmarkError("runtime.play_layouts maps scene seeds to layout sha256")
    for field in ("play_kernel_sha256", "play_scene_source_sha256"):
        identity = runtime.get(field)
        if identity is not None and (not isinstance(identity, str) or len(identity) != 64):
            raise RoboDojoBenchmarkError(f"runtime.{field} must be a sha256")


__all__ = [
    "AUDIT_CRITERIA", "AUDIT_PASSES", "AUDIT_SEED_SCHEMA", "AUDIT_WORKER_REPORT_NAME", "AUDIT_WORKER_REPORT_SCHEMA",
    "CPU_PHYSICS_DEVICE", "CUDA_PHYSICS_DEVICE", "DEFAULT_SCENE_SET", "ENTRY_FIELDS", "GENERATOR_MODULE",
    "INDEX_FILE", "MAX_NEARER_BASE_DISTANCE_M", "PLAY_EXECUTION_CAPTURE", "PLAY_NATIVE_ACTION_LIMIT",
    "PLAY_REACH_ENVELOPE_XY_M",
    "PlaySceneSetError", "SCENE_ROOT", "SCENE_SET_DIRECTORIES", "SceneDefinition", "TABLE_TOP_Z_M", "TABLE_XY_M",
    "UNBOUND_ENTRY_FIELDS", "WORKER_BACKEND", "X5_BASES_XY_M", "canonical_entry", "find_scene", "index_file",
    "layout_sha256", "scene_definition", "scene_definitions", "scene_language", "scene_sets", "source_binding",
    "source_files", "validate_play_runtime",
]
