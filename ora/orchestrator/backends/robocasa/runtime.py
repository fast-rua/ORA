"""The RoboCasa365 runtime of a RunSpec: its own fields, after the checks every runtime shares (spec.runtime)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.orchestrator.spec.runtime import bind_native_action_limit, bind_official_limit_multiple
from ora.orchestrator.spec.values import RunSpecError

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def task_limit(task: str) -> int:
    from ora.environment.benchmarks.robocasa import TASKS
    return int(TASKS[task]["horizon_control_steps"])


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    # RoboCasa365 drives motion_action_v3 only (OSC Cartesian servo plus the
    # base servo behind move_base). Each seed samples its own kitchen, so the
    # episode language is the environment's own and is bound per cell.
    if policy["action_codec"] != "motion_action_v3":
        raise RunSpecError("RoboCasa supports motion_action_v3 only")
    if task != runtime.get("task") or seed != runtime.get("seed"):
        raise RunSpecError("RoboCasa task/seed must match the episode identity")
    bind_native_action_limit(runtime, task_limit(task), "RoboCasa")
    # As RoboDojo and RoboLab (user 2026-11-05: no simulation-time limit): the budgets end a run.
    bind_official_limit_multiple(runtime, TOOL_BUDGET_LIMIT_MULTIPLE)
    runtime.setdefault("startup_timeout_s", 300)
    runtime.setdefault("request_timeout_s", 600)
    runtime.setdefault("close_timeout_s", 60)
    language = runtime.get("expected_task_language")
    if language is not None and language != task_instruction:
        raise RunSpecError("RoboCasa task instruction must match the environment language")


def revision_label(runtime: Mapping[str, Any]) -> str:
    """The plan display's runtime revision: every pinned checkout."""
    revisions = runtime.get("required_revisions") or {}
    return "/".join(f"{name}={value[:12]}" for name, value in sorted(revisions.items()))


__all__ = ["resolve_runtime", "revision_label", "task_limit"]
