"""The YAM runtime of a RunSpec: its own fields, after the checks every runtime shares (spec.runtime)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ora.orchestrator.spec.runtime import bind_native_action_limit
from ora.orchestrator.spec.values import RunSpecError

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    from ora.environment.benchmarks.yam import (
        MAIN_CAMERA, PUBLIC_CAMERAS, YamBenchmarkError, task_instruction as benchmark_instruction,
        validate_runtime_extension,
    )

    if policy["action_codec"] != "motion_action_v3":
        raise RunSpecError("the YAM station supports motion_action_v3 only")
    if task != runtime.get("task") or seed != runtime.get("seed", 0):
        raise RunSpecError("YAM task/seed must match the episode identity")
    try:
        extension = validate_runtime_extension(runtime)
    except YamBenchmarkError as exc:
        raise RunSpecError(str(exc)) from exc
    if benchmark_instruction(task, runtime) != task_instruction:
        raise RunSpecError("YAM task instruction must match the runtime's task instruction")
    bind_native_action_limit(runtime, extension["native_action_limit"], "YAM")
    # The worker publishes all four cameras with the front one as its main camera, which the
    # Supervisor's context projection reads from the runtime.
    if (runtime.setdefault("main_camera", MAIN_CAMERA) != MAIN_CAMERA
            or runtime.setdefault("policy_cameras", list(PUBLIC_CAMERAS)) != list(PUBLIC_CAMERAS)):
        raise RunSpecError(f"YAM main_camera is {MAIN_CAMERA} and policy_cameras are {list(PUBLIC_CAMERAS)}")
    runtime.setdefault("startup_timeout_s", 180)
    runtime.setdefault("request_timeout_s", 3600 if extension["rig_mode"] == "on" else 600)
    runtime.setdefault("close_timeout_s", 120)


def revision_label(runtime: Mapping[str, Any]) -> str:
    """The plan display's runtime revision: the i2rt pin and the rig mode."""
    from ora.environment.benchmarks.yam import I2RT_REVISION

    return f"i2rt={I2RT_REVISION[:12]} rig={runtime.get('rig_mode', 'sim')}"


__all__ = ["resolve_runtime", "revision_label"]
