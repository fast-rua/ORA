"""A free-play scene's plan binding: the runtime that loads it with its official predicate withheld.

``spec.play`` resolves a play RunSpec's ``play`` block and calls
``bind_play_runtime`` on the defaults-expanded runtime, then
``verify_play_qualification`` on the bound qualification. What a backend binds
is its ``play`` hook's (``backends/registry.py``): LIBERO's play driver,
RoboTwin's play signal policy (and a scene set's ``play_scene_set``), and
RoboDojo's scene set, physics device, layouts, play kernel and scene sources.
The helpers below are the rules those bindings share. The scenes themselves,
their listing and drawing, are ``scenes``'s.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..backends.registry import find
from ..spec.values import RunSpecError

#: The runtime field that loads a scene of ORA's own scene sets (RoboTwin's instead of ``envs.{task}``,
#: RoboDojo's always).
SCENE_SET_FIELD = "play_scene_set"


def bind_play_runtime(runtime: Mapping[str, Any], *, scene: Mapping[str, Any] | None,
                      scenes: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The plan runtime of a play scene (``scene``), or of a benchmark plan (``None``).

    The resolver calls this on the defaults-expanded runtime: the baseline
    template refuses a RunSpec that names a play binding itself. A benchmark
    runtime is returned unchanged and may not select a play binding. A play
    scene must be this runtime's (environment, task and its LIBERO suite and
    task id or RoboTwin scene configuration); its runtime is bound to the
    backend's play binding, which loads the scene with the official predicate
    withheld. A RoboTwin scene of one of ORA's scene sets also binds
    ``runtime.play_scene_set``, which no other plan may carry. A RoboDojo
    scene is always one of ORA's scene sets: its runtime binds the set, the
    scene's physics device, the generated layout of every scene the session
    may load (``scenes``, default ``[scene]``), the audited play kernel and
    the scene's audited source binding.
    """

    kind = runtime.get("kind")
    selected = find(kind)
    play = selected.play if selected is not None else None
    if scene is None:
        if play is not None:
            play.refuse_benchmark(runtime)
        else:
            refuse_scene_set(runtime)
        return dict(runtime)
    if play is None:
        raise RunSpecError(f"runtime.kind {kind!r} has no play scene")
    return play.bind_runtime(runtime, scene, list(scenes) if scenes is not None else [scene])


def verify_play_qualification(play: Mapping[str, Any], qualification: Mapping[str, Any]) -> None:
    """A RoboTwin or RoboDojo play scene's record is the one the plan binds (preflight re-admits its seeds)."""

    scene = play["scene"]
    selected = find(scene["environment"])
    backend_play = selected.play if selected is not None else None
    if backend_play is None or backend_play.environment != scene["environment"] or not backend_play.bound_record:
        return
    manifest = qualification.get("manifest")
    if (qualification.get("required") is not True or not isinstance(manifest, Mapping)
            or manifest.get("sha256") != scene["qualification"].get("sha256")):
        raise RunSpecError(f"a {backend_play.display_name} play scene "
                           "requires the seed manifest its play.scene names")


def refuse_scene_set(runtime: Mapping[str, Any]) -> None:
    """A benchmark runtime never carries ``play_scene_set``."""

    if runtime.get(SCENE_SET_FIELD) is not None:
        raise RunSpecError(f"runtime.{SCENE_SET_FIELD} loads a play scene; it needs episode_mode play")


def refuse_play_field(runtime: Mapping[str, Any], field: str, value: Any) -> None:
    """A benchmark runtime never selects the binding (``field: value``) that withholds the predicate."""

    if runtime.get(field) == value:
        raise RunSpecError(f"runtime.{field} {value} loads a play scene; it needs episode_mode play")


def require_scene(scene: Mapping[str, Any], expected: Mapping[str, Any], *, scene_set_allowed: bool) -> None:
    """``scene`` is one of this runtime's (``expected`` fields), with a task, a seed and its qualification record."""

    seed = scene.get("seed")
    if (any(scene.get(key) != item for key, item in expected.items()) or not isinstance(scene.get("task"), str)
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not isinstance(scene.get("qualification"), Mapping)):
        raise RunSpecError("play.scene must be a scene of this runtime with its qualification record")
    if not scene_set_allowed and scene.get("scene_set") is not None:
        raise RunSpecError("play.scene.scene_set names a RoboTwin scene set")


def bind_field(runtime: Mapping[str, Any], field: str, value: Any, replaceable: set[Any]) -> dict[str, Any]:
    """The runtime with ``field`` set to the play binding ``value``, which may replace only ``replaceable``."""

    if runtime.get(field) not in replaceable | {value}:
        raise RunSpecError(f"a play scene binds runtime.{field} {value}")
    return {**runtime, field: value}


__all__ = [
    "SCENE_SET_FIELD", "bind_field", "bind_play_runtime", "refuse_play_field", "refuse_scene_set",
    "require_scene", "verify_play_qualification",
]
