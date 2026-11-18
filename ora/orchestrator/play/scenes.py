"""Play scenes (``ora play``): one qualified benchmark scene, loaded without its task.

A play RunSpec (``episode_mode: play``) loads a benchmark task's scene and
initial state and nothing else of the task: no task language, no completion
contract and no success predicate. The backend withholds the predicate itself
(LIBERO's ``direct_libero_play_v1`` driver, RoboTwin's ``play_scene_v1`` signal
policy, bound by ``play.binding.bind_play_runtime``), so nothing a task would
count as success ends or changes a scene: it ends only at the operator's reset
or quit, a budget or the environment's own step limit, and its verdict is
null. Scenes come only from what this host has qualified, and each backend
says what that is in its ``play`` hook (``backends/registry.py``):

- LIBERO (``backends/libero_pro/play.py``): the initial states of LIBERO-Goal
  and LIBERO-Object in this host's pinned LIBERO-Pro checkout, bound by hash.
- RoboTwin (``backends/robotwin/play.py``): the seeds a local seed manifest
  admits for the task, its scene configuration and the pinned revision, or,
  with ``scene_set`` (``ora play --scene-set``), a scene of one of ORA's own
  scene sets at a seed a local play-scene audit admits.
- RoboDojo (``backends/robodojo/play.py``): only ORA's own scene sets, at a
  seed a local play-scene audit admits for the current play kernel and the
  scene's current source binding.
- Synthetic (``backends/synthetic.py``): the built-in world, which needs no
  simulator or record.
- The YAM station (``backends/yam/play.py``): its one cell as it stands, no
  scene draw; ``/reset`` starts a new round from the arms' rest pose.

``list_play_scenes`` lists the scenes (``/reset`` and the TUI), ``draw_play_scene``
picks one with the caller's ``random.Random`` (a task first, then one of its
scenes), and ``build_play_spec`` returns the RunSpec of a drawn scene with
``play.scene`` (``ora.play-session.v1``). ``ora play`` (``ora/app/play.py``) adds the
Provider ``policy``, ``operator_control``, the session framing as
``task_instruction``, and ``play.scenes`` (the ``/reset`` candidates: the same
task, and for RoboTwin the same scene configuration, since a backend resets in
place only to another seed or initial state), ``instruction_budgets`` and
``draw``; ``spec/play.py`` resolves the block and binds the plan runtime
(``play.binding``). A scene's task id is for the backend and the evidence,
never for the policy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.registry import PlayBackend, canonical, find
from ..paths import PROJECT_ROOT, resolve_outputs_root, resolve_workspace_root

EXPERIMENT = "ora-play"
EPISODE_MODE = "play"
#: The play block ``spec/play.py`` resolves; ``build_play_spec`` fills its ``scene``.
PLAY_SESSION_SCHEMA = "ora.play-session.v1"
ENVIRONMENTS = ("synthetic", "libero", "robotwin", "robodojo", "yam")
#: Why an environment offers no play scene (every environment offers one now).
UNAVAILABLE: dict[str, str] = {}
#: Where the play-scene audits of ORA's own scene sets live, under the outputs root's ``ora-play``.
PLAY_AUDIT_DIRECTORY = "qualification/play-scene-audits"
#: One play session's whole budget; a LIBERO scene's simulator steps are its
#: step limit, a RoboTwin scene also ends at its task's own step limit.
PLAY_BUDGETS = {"max_robot_decisions": 400, "max_model_calls": 1600, "max_sim_steps": 50000,
                "wall_time_s": 14400}


class PlaySceneError(ValueError):
    """No qualified play scene is available, or a scene or request is invalid."""


@dataclass(frozen=True)
class PlayScene:
    """One loadable scene: a benchmark task's world at one seed or initial state."""

    environment: str
    task: str
    seed: int
    qualification: Mapping[str, Any] = field(compare=False)
    suite: str | None = None
    task_id: int | None = None
    task_config: str | None = None
    #: The local record file behind ``qualification`` (a RoboTwin seed manifest or play-scene audit).
    source: Path | None = field(default=None, compare=False)
    #: ORA's own scene set the scene belongs to (RoboTwin's ``scenes``, or RoboDojo's), if any.
    scene_set: str | None = None

    @property
    def family(self) -> str:
        """The scene's task family: the unit a draw picks first."""
        return self.task if self.task_config is None else f"{self.task_config}:{self.task}"

    @property
    def scene_id(self) -> str:
        if self.scene_set is not None:
            return f"{self.environment}:{self.scene_set}:{self.task}:s{self.seed}"
        return f"{self.environment}:{self.family}:s{self.seed}"

    @property
    def label(self) -> str:
        return _play(self.environment).label(self)

    def record(self) -> dict[str, Any]:
        """``play.scene`` (or one of ``play.scenes``): what loads and the record that admits it.

        ``environment``, ``task``, ``seed`` and ``qualification`` are always
        present; ``scene_id`` and ``label`` name it for the operator, and the
        LIBERO suite and task id or the RoboTwin scene configuration (and scene
        set) identify it against the runtime.
        """
        record: dict[str, Any] = {"environment": self.environment, "task": self.task, "seed": self.seed,
                                  "scene_id": self.scene_id, "label": self.label}
        if self.suite is not None:
            record.update(suite=self.suite, task_id=self.task_id)
        if self.task_config is not None:
            record["task_config"] = self.task_config
        if self.scene_set is not None:
            record["scene_set"] = self.scene_set
        record["qualification"] = json.loads(json.dumps(self.qualification))
        return record


def _play(environment: str) -> PlayBackend:
    """The play hook of the backend ``environment`` (an ``ora play --env`` name) names."""

    selected = find(environment)
    play = selected.play if selected is not None else None
    if play is None or play.environment != environment:
        raise PlaySceneError(f"play environment must be one of {list(ENVIRONMENTS)}, not {environment!r}")
    return play


def play_audit_root() -> Path:
    """Where this host's play-scene audits live: ``<outputs root>/ora-play/qualification/play-scene-audits``.

    The RoboTwin and RoboDojo play-scene audits write there (their ``audit_root``).
    """
    outputs = resolve_outputs_root(resolve_workspace_root(PROJECT_ROOT), project_root=PROJECT_ROOT)
    return outputs / EXPERIMENT / PLAY_AUDIT_DIRECTORY


# --- listing and drawing -------------------------------------------------------------------------

def _require_environment(environment: str) -> None:
    if environment not in ENVIRONMENTS:
        raise PlaySceneError(f"play environment must be one of {list(ENVIRONMENTS)}, not {environment!r}")
    if environment in UNAVAILABLE:
        raise PlaySceneError(UNAVAILABLE[environment])


def list_play_scenes(environment: str, *, task: str | None = None, task_config: str | None = None,
                     scene_set: str | None = None, robotwin_workspace_root: Path | None = None,
                     play_audit_root_dir: Path | None = None, libero_checkout: Path | None = None,
                     environ: Mapping[str, str] | None = None, root: Path = PROJECT_ROOT) -> list[PlayScene]:
    """Every scene this host has qualified for ``environment``, in a stable order.

    ``task`` (and for RoboTwin ``task_config``) narrow the list. For ``/reset``
    within one running scene pass the scene's own: a backend resets in place
    only to another seed or initial state of the same task and configuration.
    Listing all RoboTwin scenes reads every local seed manifest;
    ``draw_play_scene`` reads only the drawn task's. ``scene_set`` lists a
    RoboTwin scene set's audited scenes instead of the official tasks'.
    """
    _require_environment(environment)
    play = _play(environment)
    if scene_set is not None and not play.scene_sets:
        raise PlaySceneError(f"scene sets are RoboTwin's and RoboDojo's; {environment} has none")
    return play.list_scenes(task=task, task_config=task_config, scene_set=scene_set, root=root,
                            robotwin_workspace_root=robotwin_workspace_root,
                            play_audit_root_dir=play_audit_root_dir, libero_checkout=libero_checkout,
                            environ=environ)


def draw_play_scene(environment: str, rng: random.Random, **listing: Any) -> PlayScene:
    """One random qualified scene: a task (family) uniformly, then one of its scenes.

    Drawing the family first keeps a task with fifty admitted seeds from
    crowding out one with a single seed. The draw depends only on ``rng`` and
    the local records, so a seeded ``rng`` repeats it. Which records it reads,
    and what it says when none admits a scene, is the backend's.
    """
    _require_environment(environment)
    return _play(environment).draw(rng, dict(listing))


def draw_from(scenes: Sequence[PlayScene], rng: random.Random) -> PlayScene:
    """A family uniformly, then one of its scenes."""
    families = sorted({scene.family for scene in scenes})
    family = rng.choice(families)
    return rng.choice([scene for scene in scenes if scene.family == family])


# --- the RunSpec ---------------------------------------------------------------------------------

def build_play_spec(environment: str, *, host: str | None, rng: random.Random,
                    scene: PlayScene | None = None, resource_profile: str | None = None,
                    budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
                    root: Path = PROJECT_ROOT, **listing: Any) -> dict[str, Any]:
    """The play RunSpec of one scene (``scene``, else a random qualified one); ``policy`` is the caller's.

    ``host`` is the registered execution host (``ORA_EXECUTION_HOST`` when
    None; Synthetic needs none) and ``resource_profile`` its simulator slot
    (default: the host's first slot for the environment). ``service_report``
    binds the resident services as the benchmark builders do. A RoboTwin spec
    names the content-addressed import path of its seed manifest under
    ``outputs/ora-play``; ``write_play_spec`` writes the spec and imports it.
    """
    _require_environment(environment)
    chosen = scene if scene is not None else draw_play_scene(environment, rng, root=root, **listing)
    if chosen.environment != environment:
        raise PlaySceneError(f"scene {chosen.scene_id} is not a {environment} scene")
    spec: dict[str, Any] = {
        "schema": "ora.run-spec.v1", "experiment": EXPERIMENT, "episode_mode": EPISODE_MODE,
        "config": "play-" + chosen.scene_id.replace(":", "-"), "milestone": "PLAY",
        "output_workspace": f"outputs/{EXPERIMENT}", "task": chosen.task, "seed": chosen.seed,
        "evidence_tier": "diagnostic", "budgets": {**PLAY_BUDGETS, **dict(budgets or {})},
        "qualification": {"required": False},
        "play": {"schema": PLAY_SESSION_SCHEMA, "scene": chosen.record()},
    }
    _play(environment).build_spec(spec, chosen, host=host, resource_profile=resource_profile,
                                  service_report=service_report, root=root)
    return spec


def bind_placement(spec: dict[str, Any], environment: str, *, host: str | None, resource_profile: str | None,
                   service_report: Path | None, root: Path) -> dict[str, Any]:
    """A simulator scene's execution host and slot (the host's first slot by default), and the
    resident services of ``service_report`` as ``spec["prerequisite"]``, as the benchmark builders bind them."""

    from ..spec.benchmark import _placement
    from ..preflight.resources import ExecutionResourceError, default_slot

    kind = str(canonical(environment))
    try:
        slot = resource_profile or default_slot(kind, _host(host))
    except ExecutionResourceError as exc:
        raise PlaySceneError(str(exc)) from None
    placement = _placement(kind, host=host, resource_profile=slot)
    if service_report is not None:
        from ..spec.inputs import service_prerequisite

        spec["prerequisite"] = service_prerequisite(service_report, kind, root=root)
    return placement


def _host(host: str | None) -> str:
    from ..spec.benchmark import execution_host

    return execution_host(host)


def write_play_spec(spec: Mapping[str, Any], scene: PlayScene, *, output: Path | None = None) -> Path:
    """Write ``spec`` under ``outputs/ora-play/specs`` and import its seed manifest or play-scene audit."""
    from ..spec.benchmark import compose

    return compose(spec, seed_manifest=scene.source if _play(scene.environment).bound_record else None,
                   output=output)


__all__ = [
    "ENVIRONMENTS", "EPISODE_MODE", "EXPERIMENT", "PLAY_AUDIT_DIRECTORY", "PLAY_BUDGETS", "PLAY_SESSION_SCHEMA",
    "PlayScene", "PlaySceneError", "UNAVAILABLE", "bind_placement", "build_play_spec", "draw_from",
    "draw_play_scene", "list_play_scenes", "play_audit_root", "write_play_spec",
]
