"""The RoboLab runtime of a RunSpec: its own fields, after the checks every runtime shares (spec.runtime)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.orchestrator.spec.runtime import bind_native_action_limit, bind_official_limit_multiple
from ora.orchestrator.spec.values import RunSpecError

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def task_limit(task: str) -> int:
    from ora.environment.benchmarks.robolab import TASKS
    return int(TASKS[task]["native_action_limit"])


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    # RoboLab drives motion_action_v3 only (absolute-IK Cartesian servo on the
    # Robotiq TCP). The Kit renderer device must equal the slot's CUDA device.
    if policy["action_codec"] != "motion_action_v3":
        raise RunSpecError("RoboLab supports motion_action_v3 only")
    if task != runtime.get("task") or seed != runtime.get("seed"):
        raise RunSpecError("RoboLab task/seed must match the episode identity")
    bind_native_action_limit(runtime, task_limit(task), "RoboLab")
    # As RoboDojo (user 2026-11-05: no simulation-time limit): the budgets end a run, never a native-action count;
    # a success past the official limit is recorded as success_after_official_limit.
    bind_official_limit_multiple(runtime, TOOL_BUDGET_LIMIT_MULTIPLE)
    if runtime.get("expected_task_language") != task_instruction:
        raise RunSpecError("RoboLab task instruction must match the official environment language")
    if isinstance(runtime.get("renderer_gpu"), bool) or not isinstance(runtime.get("renderer_gpu"), int):
        raise RunSpecError("RoboLab requires an integer runtime.renderer_gpu")
    runtime.setdefault("startup_timeout_s", 900)
    runtime.setdefault("request_timeout_s", 1200)
    runtime.setdefault("close_timeout_s", 120)


def revision_label(runtime: Mapping[str, Any]) -> str:
    """The plan display's runtime revision: the RoboLab pin and the renderer GPU."""
    return f"robolab={str(runtime.get('required_revision', ''))[:12]} gpu={runtime.get('renderer_gpu')}"


__all__ = ["resolve_runtime", "revision_label", "task_limit"]
