"""LIBERO's free-play scenes (``ora play --env libero``): the initial states of two base suites.

The scenes are the initial states of LIBERO-Goal and LIBERO-Object, the two
base suites whose scenes hold the most things to handle. Every LIBERO-Goal
task is the same kitchen table (a bowl, a plate, cream cheese and a wine
bottle beside a cabinet with drawers, a stove and a wine rack), so its tasks
differ only in where things start; a LIBERO-Object scene is seven groceries
and a basket. LIBERO qualifies a start by publishing it: the record is the
task's init-state file in this host's pinned LIBERO-Pro checkout
(``LIBERO_PRO_ROOT``), bound by hash, and the scene is one of its 50 states.
The play driver (``direct_libero_play_v1``) loads it with the BDDL predicate
withheld.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.libero_pro import PLAY_DRIVER
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.play import binding
from ora.orchestrator.play.scenes import PlayScene, PlaySceneError
from ora.orchestrator.spec.values import RunSpecError

#: The LIBERO base suites a play scene is drawn from (see the module docstring).
PLAY_SUITES = ("libero_goal", "libero_object")
#: How a scene describes itself to the operator; never sent to the policy.
SCENE_NAMES = {"libero_goal": "kitchen table", "libero_object": "groceries and a basket"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _libero_instance(root: Path) -> Mapping[str, Any]:
    from ora.orchestrator.spec.inputs import default_instance

    return default_instance("libero_pro", root)[1]["runtime"]


def libero_root(environ: Mapping[str, str] | None = None, *, root: Path = PROJECT_ROOT) -> Path | None:
    """This host's LIBERO-Pro checkout, named by the default instance's root variable."""
    values = os.environ if environ is None else environ
    value = values.get(str(_libero_instance(root)["libero_pro_root_env"]))
    return Path(value).expanduser() if value else None


def _libero_scenes(checkout: Path | None, *, task: str | None, root: Path) -> list[PlayScene]:
    from ora.environment.benchmarks.libero_pro import OFFICIAL_CORE_INIT_STATES
    from ora.environment.benchmarks.libero_pro_language import TASK_ROWS

    if checkout is None:
        return []
    revision = str(_libero_instance(root)["required_libero_pro_revision"])
    scenes = []
    for family, task_id, filename, *_languages in TASK_ROWS:
        suite = family if family.startswith("libero_") else f"libero_{family}"
        name = f"{suite}-t{task_id:02d}"
        if suite not in PLAY_SUITES or (task is not None and name != task):
            continue
        relative = Path("libero/libero/init_files") / suite / f"{filename}.pruned_init"
        path = checkout / relative
        if not path.is_file():
            continue
        qualification = {"kind": "libero_init_states", "revision": revision, "path": relative.as_posix(),
                         "sha256": _sha256(path), "states": OFFICIAL_CORE_INIT_STATES}
        scenes.extend(PlayScene("libero", name, state, qualification, suite=suite, task_id=task_id)
                      for state in range(OFFICIAL_CORE_INIT_STATES))
    return scenes


class LiberoPlay:
    """``registry.PlayBackend`` of LIBERO-Pro."""

    environment = "libero"
    display_name = "LIBERO"
    scene_sets = False
    #: A scene's init-state record is named in the scene, not imported as the plan's manifest.
    bound_record = False

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    libero_checkout: Path | None = None, environ: Mapping[str, str] | None = None,
                    **_sources: Any) -> list[PlayScene]:
        checkout = libero_checkout if libero_checkout is not None else libero_root(environ, root=root)
        return _libero_scenes(checkout, task=task, root=root)

    def draw(self, rng: random.Random, listing: dict[str, Any]) -> PlayScene:
        scenes = play_scenes.list_play_scenes(self.environment, **listing)
        if not scenes:
            raise PlaySceneError("no local LIBERO init-state record: set LIBERO_PRO_ROOT to this host's pinned "
                                 "LIBERO-Pro checkout (./scripts/ora setup --env libero)")
        return play_scenes.draw_from(scenes, rng)

    def label(self, scene: PlayScene) -> str:
        return (f"LIBERO {SCENE_NAMES[str(scene.suite)]} · {scene.suite} task {scene.task_id} · "
                f"initial state {scene.seed}")

    def build_spec(self, spec: dict[str, Any], chosen: PlayScene, *, host: str | None,
                   resource_profile: str | None, service_report: Path | None, root: Path) -> None:
        from ora.environment.benchmarks.libero_pro_language import base_bddl_language, base_task_language

        placement = play_scenes.bind_placement(spec, self.environment, host=host, resource_profile=resource_profile,
                                               service_report=service_report, root=root)
        # The backend's identity check compares the loaded BDDL with this language; the policy never reads it.
        language = base_task_language(str(chosen.suite), int(chosen.task_id))
        spec["runtime"] = {**placement, "suite": chosen.suite, "task_id": chosen.task_id,
                           "expected_task_language": language}
        bddl = base_bddl_language(str(chosen.suite), int(chosen.task_id))
        if bddl != language:
            from ora.orchestrator.spec.benchmark import BASE_LANGUAGE_DIVERGENCE_REVIEW

            spec["runtime"]["task_language_divergence_ack"] = {
                "bddl_task_language": bddl, "policy_task_language": language,
                "review": BASE_LANGUAGE_DIVERGENCE_REVIEW}

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None:
        binding.refuse_play_field(runtime, "environment_driver", PLAY_DRIVER)
        binding.refuse_scene_set(runtime)

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The direct driver's play variant, for a scene of this runtime's suite and task."""
        binding.require_scene(scene, {"environment": self.environment, "suite": runtime.get("suite"),
                                      "task_id": runtime.get("task_id")}, scene_set_allowed=False)
        return binding.bind_field(runtime, "environment_driver", PLAY_DRIVER, {None, "direct_libero_v1"})

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None:
        if any(scene["seed"] >= 50 for scene in scenes):
            raise RunSpecError("a LIBERO free-play scene is an initial-state index below 50")
        if runtime.get("robustness_case") is not None and any(
                scene["seed"] != scenes[0]["seed"] for scene in scenes):
            # The backend rebinds no other initial state for a robustness case (libero_pro/session.py).
            raise RunSpecError("a LIBERO robustness case replays its own initial state in every scene")


PLAY = LiberoPlay()

__all__ = ["PLAY", "PLAY_SUITES", "SCENE_NAMES", "LiberoPlay", "libero_root"]
