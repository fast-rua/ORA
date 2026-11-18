"""RoboDojo's free-play scenes (``ora play --env robodojo``): ORA's own Isaac Sim scene sets only.

A scene is one of ORA's own scene sets (``ora/environment/benchmarks/
robodojo_play_scenes.py``: Isaac Sim tabletops of RoboDojo library objects,
no task; ``isaac-tabletop-v1`` by default) at a seed a local play-scene audit
admits for the current play kernel and the scene's current source binding
(``robodojo_play_audit.play_scene_admission``), from the audits under
``outputs/ora-play/qualification/play-scene-audits``. The plan runtime binds
``play_scene_set``, the scene's physics device, the generated layout of every
scene the session may load (``play_layouts``), the audited play kernel
(``play_kernel_sha256``) and the scene's audited source binding
(``play_scene_source_sha256``); the worker runs the scene on its own backend
(``robodojo_play``), so the benchmark worker kernel that RoboDojo
qualifications bind is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo_play_scenes import DEFAULT_SCENE_SET
from ora.orchestrator import robodojo_play_audit, robodojo_play_kernel
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.play import binding
from ora.orchestrator.play.scenes import EXPERIMENT, PlayScene, PlaySceneError
from ora.orchestrator.spec.values import RunSpecError

#: RoboDojo runtime fields only a play scene's binding may carry.
PLAY_FIELDS = ("play_layouts", "play_kernel_sha256", "play_scene_source_sha256")


def _robodojo_set_scenes(audit_root: Path, *, scene_set: str, task: str | None, root: Path,
                         play_kernel_sha256: str | None = None) -> list[PlayScene]:
    """The seeds of a RoboDojo scene set's scenes a local audit admits for the current play kernel and sources."""
    from ora.environment.benchmarks.robodojo_play_scenes import PlaySceneSetError, scene_definitions

    try:
        definitions = {definition.name: definition for definition in scene_definitions(scene_set, root=root)}
    except PlaySceneSetError as exc:
        raise PlaySceneError(str(exc)) from None
    paths = sorted(audit_root.glob(f"*/{scene_set}/*.json"), key=lambda path: (path.parent.parent.name, path.name),
                   reverse=True)
    if not paths:
        return []
    kernel = play_kernel_sha256 or robodojo_play_kernel.play_kernel_sha256(project_root=root)
    scenes: dict[tuple[str, int], PlayScene] = {}
    for path in paths:  # newest audit first (its directory is named by its UTC start time)
        try:
            data = path.read_bytes()
            record = json.loads(data)
        except (OSError, ValueError):
            continue
        if not isinstance(record, Mapping) or record.get("kind") != robodojo_play_audit.KIND:
            continue
        name = record.get("scene")
        if name not in definitions or path.stem != name or (task is not None and name != task):
            continue
        for seed in record.get("accepted_seeds") or ():
            if (isinstance(seed, bool) or not isinstance(seed, int) or (name, seed) in scenes
                    or not robodojo_play_audit.play_scene_admission(
                        record, scene_set=scene_set, scene=name, seed=seed, play_kernel_sha256=kernel,
                        root=root)[0]):
                continue
            qualification = {"kind": robodojo_play_audit.KIND, "schema": record.get("schema"),
                             "scene_set": scene_set, "path": path.relative_to(audit_root).as_posix(),
                             "sha256": hashlib.sha256(data).hexdigest(), "play_kernel_sha256": kernel,
                             "scene_source_sha256": record["scene_source"]["sha256"],
                             "layout_sha256": record["seeds"][str(seed)]["layout_sha256"]}
            scenes[(name, seed)] = PlayScene("robodojo", name, seed, qualification, source=path,
                                             scene_set=scene_set)
    return [scenes[key] for key in sorted(scenes)]


def _play_binding(runtime: Mapping[str, Any], scene: Mapping[str, Any],
                  scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The RoboDojo play binding: set, physics device, the scenes' layouts, the play kernel and the scene's sources.

    Every scene of a session is a seed of one scene, so each carries the same
    two audited identities: the play kernel (code) and that scene's own source
    binding (its index entry, YAML and the generator), and the plan binds both.
    """
    from ora.environment.benchmarks.robodojo_play_scenes import PlaySceneSetError, scene_definition

    field = binding.SCENE_SET_FIELD
    scene_set, task = scene.get("scene_set"), scene.get("task")
    seed = scene.get("seed")
    if (scene.get("environment") != "robodojo" or not isinstance(task, str) or task != runtime.get("task_variant")
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not isinstance(scene.get("qualification"), Mapping)):
        raise RunSpecError("play.scene must be a scene of this runtime with its qualification record")
    if not isinstance(scene_set, str):
        raise RunSpecError("a RoboDojo play scene is a scene of one of ORA's scene sets (play.scene.scene_set)")
    if runtime.get(field) not in (None, scene_set):
        raise RunSpecError(f"runtime.{field} must be play.scene.scene_set")
    try:
        definition = scene_definition(scene_set, task)
    except PlaySceneSetError as exc:
        raise RunSpecError(str(exc)) from None
    if runtime.get("physics_device") not in (None, definition.physics_device):
        raise RunSpecError(f"{scene_set} scene {task} runs on {definition.physics_device} physics")
    layouts: dict[str, str] = {}
    kernels, sources = set(), set()
    for item in scenes:
        qualification = item.get("qualification") if isinstance(item.get("qualification"), Mapping) else {}
        digest = qualification.get("layout_sha256")
        if (item.get("environment") != "robodojo" or item.get("task") != task or item.get("scene_set") != scene_set
                or not isinstance(digest, str) or len(digest) != 64
                or layouts.get(str(item.get("seed")), digest) != digest):
            raise RunSpecError("every RoboDojo play scene of a session is an audited seed of one scene")
        layouts[str(item["seed"])] = digest
        kernels.add(qualification.get("play_kernel_sha256"))
        sources.add(qualification.get("scene_source_sha256"))
    if len(kernels) != 1 or not isinstance(next(iter(kernels)), str) or str(seed) not in layouts:
        raise RunSpecError("a RoboDojo play session's scenes come from one audit of one play kernel")
    if len(sources) != 1 or not isinstance(next(iter(sources)), str) or len(next(iter(sources))) != 64:
        raise RunSpecError("a RoboDojo play session's scenes come from audits of one source binding of its scene")
    bound = {**runtime, field: scene_set, "physics_device": definition.physics_device,
             "play_layouts": dict(sorted(layouts.items(), key=lambda pair: int(pair[0]))),
             "play_kernel_sha256": next(iter(kernels)), "play_scene_source_sha256": next(iter(sources))}
    for name in PLAY_FIELDS:
        if runtime.get(name) not in (None, bound[name]):
            raise RunSpecError(f"runtime.{name} must be the audited play binding of the session's scenes")
    return bound


class RoboDojoPlay:
    """``registry.PlayBackend`` of RoboDojo."""

    environment = "robodojo"
    display_name = "RoboDojo"
    scene_sets = True
    #: The plan imports and binds the scene's play-scene audit.
    bound_record = True

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    play_audit_root_dir: Path | None = None, **_sources: Any) -> list[PlayScene]:
        # RoboDojo offers only ORA's own scene sets.
        return _robodojo_set_scenes(play_audit_root_dir or play_scenes.play_audit_root(),
                                    scene_set=scene_set or DEFAULT_SCENE_SET, task=task, root=root)

    def draw(self, rng: random.Random, listing: dict[str, Any]) -> PlayScene:
        scene_set = listing.get("scene_set") or DEFAULT_SCENE_SET
        scenes = play_scenes.list_play_scenes(self.environment, **{**listing, "scene_set": scene_set})
        if not scenes:
            raise PlaySceneError(
                f"no local play-scene audit admits a RoboDojo {scene_set} scene"
                + (f" {listing['task']}" if listing.get("task") else "")
                + " for the current play kernel and the pinned revisions; run "
                f"`./scripts/ora play audit --env robodojo --scene-set {scene_set}` on this host")
        return play_scenes.draw_from(scenes, rng)

    def label(self, scene: PlayScene) -> str:
        from ora.environment.benchmarks.robodojo_play_scenes import scene_definition

        return f"RoboDojo {scene.scene_set} · {scene_definition(str(scene.scene_set), scene.task).label} · seed {scene.seed}"

    def build_spec(self, spec: dict[str, Any], chosen: PlayScene, *, host: str | None,
                   resource_profile: str | None, service_report: Path | None, root: Path) -> None:
        from ora.environment.benchmarks.robodojo_play_scenes import scene_definition
        from ora.orchestrator.spec.benchmark import seed_manifest_reference
        from ora.orchestrator.preflight.resources import resolve_execution_resource

        placement = play_scenes.bind_placement(spec, self.environment, host=host, resource_profile=resource_profile,
                                               service_report=service_report, root=root)
        if chosen.source is None:
            raise PlaySceneError(f"scene {chosen.scene_id} carries no seed manifest")
        resource = resolve_execution_resource(placement["resource_profile"], runtime_kind="robodojo",
                                              required_host=placement["required_host"])
        gpu = str(resource["worker_environment"]["CUDA_VISIBLE_DEVICES"])
        definition = scene_definition(str(chosen.scene_set), chosen.task)
        spec["runtime"] = {**robodojo_play_audit.spec_runtime(definition, gpu=gpu, host=placement["required_host"]),
                           "resource_profile": placement["resource_profile"]}
        spec["qualification"] = {"required": True, "manifest": seed_manifest_reference(
            EXPERIMENT, chosen.task, chosen.source.read_bytes())}

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None:
        binding.refuse_scene_set(runtime)
        if any(runtime.get(field) is not None for field in PLAY_FIELDS):
            raise RunSpecError("runtime.play_layouts, play_kernel_sha256 and play_scene_source_sha256 load a "
                               "play scene; they need episode_mode play")

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _play_binding(runtime, scene, scenes)

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None:
        # Only ORA's own scene sets: each scene is a generated layout of one scene, and the worker
        # relaunches the simulation on it (CPU and CUDA alike), never an in-place restart.
        scene_set = runtime.get("play_scene_set")
        if scene_set is None or any(scene.get("scene_set") != scene_set for scene in scenes):
            raise RunSpecError("a RoboDojo free-play session's scenes are seeds of one scene of one of ORA's "
                               "scene sets (runtime.play_scene_set)")


PLAY = RoboDojoPlay()

__all__ = ["PLAY", "PLAY_FIELDS", "RoboDojoPlay"]
