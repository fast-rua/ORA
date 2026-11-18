"""RoboTwin's free-play scenes (``ora play --env robotwin``): admitted seeds, or ORA's own scene sets.

Without a scene set a scene is a seed that a local seed manifest admits for the
task, its scene configuration and the pinned revision (``seed_manifest.seed_admission``,
launch preflight's rule), from the seed scans under
``outputs/runtime-v1/robotwin-evaluation/qualification``. With ``scene_set``
(``ora play --scene-set``) it is a scene of one of ORA's own scene sets
(``ora/environment/benchmarks/robotwin_play_scenes.py``: tables with many
library objects and no task) at a seed a local play-scene audit admits
(``play_audit.play_scene_admission``), from the audits under
``outputs/ora-play/qualification/play-scene-audits``. Such a scene's plan
runtime also binds ``play_scene_set``, so the worker loads ORA's scene class
instead of ``envs.{task}``. The play signal policy (``play_scene_v1``) loads
either with the task's ``check_success`` withheld.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.backends.robotwin import play_audit, seed_manifest
from ora.orchestrator.play import binding
from ora.orchestrator.play.scenes import EXPERIMENT, PlayScene, PlaySceneError
from ora.orchestrator.spec.values import RunSpecError

#: The signal policy that loads a scene with its official predicate withheld.
PLAY_SIGNAL_POLICY = "play_scene_v1"


def robotwin_workspace() -> Path:
    """Where this host's RoboTwin seed scans live (``evaluation_seeds.WORKSPACE``)."""
    from ora.orchestrator.backends.robotwin.evaluation_seeds import WORKSPACE

    return WORKSPACE


def _manifest_paths(workspace: Path, task: str | None) -> list[Path]:
    names = ("*.json",) if task is None else (f"{task}.json", f"{task}@*.json")
    paths = {path for name in names for path in workspace.glob(f"qualification/*/seed-manifests/{name}")}
    # Newest scan first (its directory is named by its UTC start time).
    return sorted(paths, key=lambda path: (path.parent.parent.name, path.name), reverse=True)


def robotwin_tasks(workspace: Path | None = None) -> list[str]:
    """Tasks with at least one local seed manifest (not yet checked for an admitted seed)."""
    from ora.environment.benchmarks.robotwin import TASKS

    found = {path.name.removesuffix(".json").split("@", 1)[0]
             for path in _manifest_paths(workspace or robotwin_workspace(), None)}
    return sorted(found & set(TASKS))


def _robotwin_scenes(workspace: Path, *, task: str | None, task_config: str | None,
                     root: Path) -> list[PlayScene]:
    from ora.environment.benchmarks.robotwin import TASK_CONFIGS, TASKS
    from ora.orchestrator.spec.benchmark import robotwin_revision

    revision = robotwin_revision(root)
    scenes: dict[tuple[str, str, int], PlayScene] = {}
    for path in _manifest_paths(workspace, task):
        try:
            data = path.read_bytes()
            manifest = json.loads(data)
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, Mapping):
            continue
        name, config = manifest.get("task"), manifest.get("task_config")
        # A manifest that names no scene configuration or revision cannot say which scene it admitted.
        if (name not in TASKS or (task is not None and name != task) or config not in TASK_CONFIGS
                or (task_config is not None and config != task_config)
                or manifest.get("robotwin_revision") != revision
                or path.name.removesuffix(".json").split("@", 1)[0] != name):
            continue
        qualification = {"kind": "robotwin_seed_manifest", "schema": manifest.get("schema"),
                         "revision": revision,
                         "path": path.relative_to(workspace).as_posix(),
                         "sha256": hashlib.sha256(data).hexdigest()}
        for seed in manifest.get("accepted_seeds") or ():
            if (isinstance(seed, bool) or not isinstance(seed, int) or (name, config, seed) in scenes
                    or not seed_manifest.seed_admission(manifest, task=name, seed=seed, task_config=config,
                                                        revision=revision)[0]):
                continue
            scenes[(name, config, seed)] = PlayScene("robotwin", name, seed, qualification,
                                                     task_config=config, source=path)
    return [scenes[key] for key in sorted(scenes)]


def _robotwin_set_scenes(audit_root: Path, *, scene_set: str, task: str | None, task_config: str | None,
                         root: Path) -> list[PlayScene]:
    """The seeds of a scene set's scenes that a local audit admits for the current scene sources."""
    from ora.environment.benchmarks.robotwin_play_scenes import PlaySceneSetError, scene_definitions
    from ora.orchestrator.spec.benchmark import robotwin_revision

    try:
        definitions = {definition.name: definition for definition in scene_definitions(scene_set)}
    except PlaySceneSetError as exc:
        raise PlaySceneError(str(exc)) from None
    revision = robotwin_revision(root)
    # Newest audit first (its directory is named by its UTC start time).
    paths = sorted(audit_root.glob(f"*/{scene_set}/*.json"), key=lambda path: (path.parent.parent.name, path.name),
                   reverse=True)
    scenes: dict[tuple[str, int], PlayScene] = {}
    for path in paths:
        try:
            data = path.read_bytes()
            record = json.loads(data)
        except (OSError, ValueError):
            continue
        if not isinstance(record, Mapping):
            continue
        name, config = record.get("scene"), record.get("task_config")
        if (name not in definitions or path.stem != name or (task is not None and name != task)
                or config != definitions[name].task_config or (task_config is not None and config != task_config)):
            continue
        qualification = {"kind": play_audit.KIND, "schema": record.get("schema"), "revision": revision,
                         "scene_set": scene_set, "path": path.relative_to(audit_root).as_posix(),
                         "sha256": hashlib.sha256(data).hexdigest()}
        for seed in record.get("accepted_seeds") or ():
            if (isinstance(seed, bool) or not isinstance(seed, int) or (name, seed) in scenes
                    or not play_audit.play_scene_admission(record, scene_set=scene_set, scene=name, seed=seed,
                                                           task_config=config, revision=revision, root=root)[0]):
                continue
            scenes[(name, seed)] = PlayScene("robotwin", name, seed, qualification, task_config=config,
                                             source=path, scene_set=scene_set)
    return [scenes[key] for key in sorted(scenes)]


def _scene_set_binding(runtime: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, Any]:
    """``{play_scene_set: SET}`` for a scene of one of ORA's RoboTwin scene sets, else nothing."""
    from ora.environment.benchmarks.robotwin_play_scenes import PlaySceneSetError, scene_definition

    field = binding.SCENE_SET_FIELD
    scene_set = scene.get("scene_set")
    if runtime.get(field) not in (None, scene_set):
        raise RunSpecError(f"runtime.{field} must be play.scene.scene_set")
    if scene_set is None:
        return {}
    try:
        definition = scene_definition(str(scene_set), str(scene["task"]))
    except PlaySceneSetError as exc:
        raise RunSpecError(str(exc)) from None
    if runtime.get("task_config") != definition.task_config:
        raise RunSpecError(f"{scene_set} scene {definition.name} loads in scene configuration "
                           f"{definition.task_config}")
    return {field: scene_set}


class RoboTwinPlay:
    """``registry.PlayBackend`` of RoboTwin."""

    environment = "robotwin"
    display_name = "RoboTwin"
    scene_sets = True
    #: The plan imports and binds the scene's seed manifest or play-scene audit.
    bound_record = True

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    robotwin_workspace_root: Path | None = None, play_audit_root_dir: Path | None = None,
                    **_sources: Any) -> list[PlayScene]:
        if scene_set is not None:
            return _robotwin_set_scenes(play_audit_root_dir or play_scenes.play_audit_root(), scene_set=scene_set,
                                        task=task, task_config=task_config, root=root)
        return _robotwin_scenes(robotwin_workspace_root or robotwin_workspace(), task=task,
                                task_config=task_config, root=root)

    def draw(self, rng: random.Random, listing: dict[str, Any]) -> PlayScene:
        if listing.get("scene_set") is not None:
            scenes = play_scenes.list_play_scenes(self.environment, **listing)
            if not scenes:
                raise PlaySceneError(
                    f"no local play-scene audit admits a {listing['scene_set']} scene"
                    + (f" {listing['task']}" if listing.get("task") else "")
                    + " for the current scene sources and the pinned revision; run "
                    f"`./scripts/ora play audit --env robotwin --scene-set {listing['scene_set']}` on this host")
            return play_scenes.draw_from(scenes, rng)
        workspace = listing.get("robotwin_workspace_root") or robotwin_workspace()
        named = listing.pop("task", None)
        tasks = [named] if named is not None else robotwin_tasks(workspace)
        rng.shuffle(tasks)
        for task in tasks:  # the first task, in drawn order, with an admitted seed
            scenes = play_scenes.list_play_scenes(self.environment, task=task, **listing)
            if scenes:
                return play_scenes.draw_from(scenes, rng)
        raise PlaySceneError(f"no local RoboTwin seed manifest admits a seed at the pinned revision under "
                             f"{workspace / 'qualification'}; run a seed scan (ora run --env robotwin)")

    def label(self, scene: PlayScene) -> str:
        if scene.scene_set is not None:
            from ora.environment.benchmarks.robotwin_play_scenes import scene_definition

            return f"RoboTwin {scene.scene_set} · {scene_definition(scene.scene_set, scene.task).label} · seed {scene.seed}"
        return f"RoboTwin {scene.task} · {scene.task_config} · seed {scene.seed}"

    def build_spec(self, spec: dict[str, Any], chosen: PlayScene, *, host: str | None,
                   resource_profile: str | None, service_report: Path | None, root: Path) -> None:
        from ora.orchestrator.spec.benchmark import seed_manifest_reference

        placement = play_scenes.bind_placement(spec, self.environment, host=host, resource_profile=resource_profile,
                                               service_report=service_report, root=root)
        if chosen.source is None:
            raise PlaySceneError(f"scene {chosen.scene_id} carries no seed manifest")
        spec["runtime"] = {**placement, "task": chosen.task, "task_config": chosen.task_config}
        if chosen.scene_set is not None:
            # ORA's own scenes record their operator camera at the capture's highest rate (robotwin_play_scenes).
            from ora.environment.benchmarks.robotwin_play_scenes import PLAY_EXECUTION_CAPTURE

            spec["runtime"]["execution_capture"] = dict(PLAY_EXECUTION_CAPTURE)
        spec["qualification"] = {"required": True, "manifest": seed_manifest_reference(
            EXPERIMENT, chosen.task, chosen.source.read_bytes())}

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None:
        binding.refuse_play_field(runtime, "environment_signal_policy", PLAY_SIGNAL_POLICY)
        binding.refuse_scene_set(runtime)

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The play signal policy, and a scene set's ``play_scene_set``, for a scene of this task and configuration."""
        binding.require_scene(scene, {"environment": self.environment, "task": runtime.get("task"),
                                      "task_config": runtime.get("task_config")}, scene_set_allowed=True)
        bound = binding.bind_field(runtime, "environment_signal_policy", PLAY_SIGNAL_POLICY, {None, "lifecycle_only"})
        bound.update(_scene_set_binding(runtime, scene))
        return bound

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None:
        if any((scene.get("task_config"), scene.get("scene_set"))
               != (scenes[0].get("task_config"), scenes[0].get("scene_set")) for scene in scenes):
            # The worker resets to another seed of its own scene configuration and scene set.
            raise RunSpecError("a RoboTwin free-play session's scenes share one scene configuration and scene set")


PLAY = RoboTwinPlay()

__all__ = ["PLAY", "PLAY_SIGNAL_POLICY", "RoboTwinPlay", "robotwin_tasks", "robotwin_workspace"]
