"""Host-side composition of a RoboDojo free-play worker (backend ``robodojo_play``).

``build_play_environment_session`` has ``factory.build_environment_session``'s
signature. A RoboDojo plan whose runtime names one of ORA's play scene sets
(``runtime.play_scene_set``, bound by ``play_scenes.bind_play_runtime``) gets
an isolated RoboDojo worker whose config selects the ``robodojo_play``
backend; every other plan is handed to ``build_environment_session``
unchanged. The coordinator uses it for free-play plans (``run_episode``), and
the play-scene audit builds its worker config the same way
(``robodojo_play_worker_config``).

The worker config is the benchmark RoboDojo config, built from the same
helpers (one device binding, public geometry, interpreter and roots, and the
shared motion planner's interpreter and source, which the session starts on
the worker's card), with ``backend`` ``robodojo_play``, ``eval_seed`` 0 and
the scene seed as ``play_seed``. ``factory.py`` is not changed: the benchmark
route stays byte for byte what every RoboDojo card and task qualification
binds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmarks.robodojo_play_scenes import WORKER_BACKEND
from .factory import (
    PROJECT_ROOT, EnvironmentFactoryError, _public_geometry_config, _required_environment_path,
    _robodojo_worker_environment, build_environment_session,
)
from .rpc import LocalRpcEnvironmentSession


def is_robodojo_play_runtime(runtime: Mapping[str, Any]) -> bool:
    return runtime.get("kind") == "robodojo" and runtime.get("play_scene_set") is not None


def robodojo_play_worker_config(*, runtime: Mapping[str, Any], output_root: Path, task: str, seed: int,
                                max_sim_steps: int, action_codec: str | None) -> tuple[dict[str, Any], Path,
                                                                                       dict[str, str]]:
    """The worker config, interpreter and worker environment of one play scene."""
    if not is_robodojo_play_runtime(runtime):
        raise EnvironmentFactoryError("a RoboDojo play worker needs runtime.play_scene_set")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EnvironmentFactoryError("a RoboDojo play scene seed is a non-negative integer")
    root = _required_environment_path(runtime, "root_env")
    assets = _required_environment_path(runtime, "asset_root_env")
    python = _required_environment_path(runtime, "python_env", preserve_symlink=True)
    worker = dict(runtime)
    worker.update(_public_geometry_config(runtime))
    worker.update({
        "backend": WORKER_BACKEND, "robodojo_root": str(root),
        "output_root": str(Path(output_root).resolve()),
        "runtime_directory": str(python.parent.parent.parent),
        "accept_eula": True, "asset_root": str(assets), "task": task,
        "eval_seed": 0, "play_seed": seed,
        "max_sim_steps": max_sim_steps, "action_codec": action_codec,
        "qualification_binding": None,
    })
    planner = runtime.get("motion_planner")
    if isinstance(planner, Mapping):
        # As factory.build_robodojo_worker_session binds it; shared_motion.start reads both paths.
        worker["motion_planner"] = {
            **dict(planner),
            "python_executable": str(_required_environment_path(planner, "python_env", preserve_symlink=True)),
            "root": str(_required_environment_path(planner, "root_env")),
        }
    return worker, python, _robodojo_worker_environment(runtime)


def build_robodojo_play_worker_session(
    *, runtime: Mapping[str, Any], output_root: Path, task: str, seed: int, max_sim_steps: int,
    action_codec: str | None, redacted_environment_names: Sequence[str] = (),
) -> LocalRpcEnvironmentSession:
    """Start the isolated RoboDojo worker on one play scene."""
    worker, python, environment = robodojo_play_worker_config(
        runtime=runtime, output_root=output_root, task=task, seed=seed, max_sim_steps=max_sim_steps,
        action_codec=action_codec)
    return LocalRpcEnvironmentSession(
        python_executable=python, project_root=PROJECT_ROOT, output_root=output_root,
        socket_root=PROJECT_ROOT / ".harness-runtime/env",
        startup_timeout_s=float(runtime["startup_timeout_s"]),
        request_timeout_s=float(runtime["request_timeout_s"]),
        close_timeout_s=float(runtime["close_timeout_s"]),
        redacted_environment_names=redacted_environment_names,
        worker_environment=environment, worker_config=worker,
    )


def build_play_environment_session(
    *, runtime_config: Mapping[str, Any], output_root: Path, task: str, seed: int, max_sim_steps: int,
    redact_environment_names: Sequence[str] = (), action_codec: str | None = None,
    qualified_runtime_binding: Mapping[str, Any] | None = None,
) -> Any:
    """``build_environment_session`` for a free-play plan: a RoboDojo play scene gets its own worker."""
    if not is_robodojo_play_runtime(runtime_config):
        return build_environment_session(
            runtime_config=runtime_config, output_root=output_root, task=task, seed=seed,
            max_sim_steps=max_sim_steps, redact_environment_names=redact_environment_names,
            action_codec=action_codec, qualified_runtime_binding=qualified_runtime_binding)
    if qualified_runtime_binding is not None:
        raise EnvironmentFactoryError("a RoboDojo play scene has no infrastructure qualification binding")
    from .definitions import EnvironmentDefinitionError, environment_definition
    from .session import DescribedEnvironmentSession

    if runtime_config.get("transport") != "local_rpc":
        raise EnvironmentFactoryError("RoboDojo requires local_rpc")
    try:
        descriptor = environment_definition("robodojo").resolve_descriptor(runtime_config, action_codec=action_codec)
    except EnvironmentDefinitionError as exc:
        raise EnvironmentFactoryError(str(exc)) from exc
    delegate = build_robodojo_play_worker_session(
        runtime=runtime_config, output_root=output_root, task=task, seed=seed, max_sim_steps=max_sim_steps,
        action_codec=action_codec, redacted_environment_names=redact_environment_names)
    return DescribedEnvironmentSession(delegate=delegate, planned_descriptor=descriptor,
                                       runtime_config=runtime_config, action_codec=action_codec)


__all__ = ["build_play_environment_session", "build_robodojo_play_worker_session", "is_robodojo_play_runtime",
           "robodojo_play_worker_config"]
