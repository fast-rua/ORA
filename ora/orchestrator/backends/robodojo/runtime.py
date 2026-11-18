"""The RoboDojo runtime of a RunSpec: its own fields and the session rules of its CPU-physics lane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.orchestrator.backends.robodojo.physics_attestation import cpu_round_restart_refusal
from ora.orchestrator.spec.runtime import bind_official_limit_multiple
from ora.orchestrator.spec.values import RunSpecError

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    # The unified RoboDojo instance drives the sparse v3 motion codec; the
    # registry's SPARSE_MOTION_QUALIFIED_ENVIRONMENTS row (bound to the
    # scripted RoboDojo motion-qualification record) is what actually
    # admits it, so this branch only excludes the codecs the backend has
    # no executor for.
    if policy["action_codec"] not in {"runtime_action_v3", "motion_action_v3"}:
        raise RunSpecError(
            "RoboDojo supports runtime_action_v3 and motion_action_v3"
        )
    if runtime.get("play_scene_set") is not None:
        # One of ORA's own play scenes: the episode seed is the scene's layout seed (admitted by
        # its play-scene audit, play.binding.bind_play_runtime); the evaluation seed stays 0.
        if task != runtime.get("task_variant") or runtime.get("eval_seed") != 0:
            raise RunSpecError("a RoboDojo play scene's task_variant must be the scene and eval_seed 0")
    elif task != runtime.get("task_variant") or seed != runtime.get("eval_seed"):
        raise RunSpecError("RoboDojo task/eval_seed must match the episode identity")
    else:
        bind_official_limit_multiple(runtime, TOOL_BUDGET_LIMIT_MULTIPLE)
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 180)
    runtime.setdefault("close_timeout_s", 30)
    if runtime.get("task_language_source") != "live_reset_readback" and runtime.get("expected_task_language") != task_instruction:
        raise RunSpecError("RoboDojo task instruction must match official environment language")
    # The staged contact route follows the surface whose help describes it.
    from ora.environment.benchmarks.robodojo import CONTACT_APPROACH_STAGED
    route = (CONTACT_APPROACH_STAGED if resolved_surface.supports(
        "robodojo_staged_contact_approach_v1") else None)
    if runtime.get("contact_approach_route", route) != route:
        raise RunSpecError("runtime.contact_approach_route disagrees with the ToolSurface")
    if route is not None:
        runtime["contact_approach_route"] = route
    recovery = ("retract_then_unwind" if resolved_surface.supports(
        "robodojo_retract_then_unwind_recovery_v1") else None)
    if runtime.get("recovery_route", recovery) != recovery:
        raise RunSpecError("runtime.recovery_route disagrees with the ToolSurface")
    if recovery is not None:
        runtime["recovery_route"] = recovery


def validate_session(runtime: Mapping[str, Any], session: Mapping[str, Any] | None) -> str | None:
    """The refusal of a session (None: one episode) this runtime cannot run, or None.

    Upstream's CPU reset respawns rigid objects that an in-place restart
    leaves gravity- and collision-disabled; a CPU session's later rounds must
    go through the scheduled relaunch (physics_attestation). A layout schedule
    plans exactly one layout per round.
    """

    refusal = cpu_round_restart_refusal(runtime, rounds=1 if session is None else session["rounds"])
    if refusal is not None:
        return refusal
    if runtime.get("layout_schedule") is not None and (
            session is None or len(runtime["layout_schedule"]["rounds"]) != session["rounds"]):
        return "runtime.layout_schedule must schedule exactly one layout per session round"
    return None


def provider_defaults(policy: dict[str, Any], runtime_spec: Mapping[str, Any]) -> None:
    """RoboDojo never passes through baseline expansion, so an omitted recovery
    selector left it on legacy retries; it inherits the maintained one."""

    policy.setdefault("provider_recovery", "first_token_deadline_v1")


def revision_label(runtime: Mapping[str, Any]) -> str:
    """The plan display's runtime revision: the source pins, then the play scene or the layout."""

    from ora.environment.benchmarks.robodojo import source_revisions

    revision = "/".join(value[:12] for value in source_revisions(runtime).values())
    if runtime.get("play_scene_set") is not None:
        return revision + f" play={runtime['play_scene_set']} kernel={str(runtime.get('play_kernel_sha256'))[:12]}"
    return revision + f" layout={runtime['layout_id']} sha256={runtime['layout_sha256'][:12]}"


__all__ = ["provider_defaults", "resolve_runtime", "revision_label", "validate_session"]
