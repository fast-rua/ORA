"""Publishing one fresh observation to the policy.

`publish_observation` captures (or accepts an already captured) public
observation, queries the trusted environment status, settles the receipt of
the action that preceded it, and writes the policy-state snapshot, the context
pointer and the observation events before notifying the PolicySession.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ora.agent import PolicyRuntimeEvent
from ora.agent.context.lossless.management import LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS
from ora.contracts.policy_receipt import names_timing
from ora.environment import validate_environment_status

from ..faults import SupervisorError
from .records import _append, _atomic_json
from .execution_state import publish_state
from ..evaluation.run_artifacts import _main_observation_image

if TYPE_CHECKING:
    from .loop import EpisodeLoop


def _session_round_budget_fields(
    policy: Mapping[str, Any], *, round_index: int, round_limit: int,
    multi_round: bool,
) -> dict[str, int]:
    """Give lossless ActionSteps an explicit Episode identity, including K=1."""

    management = policy.get("context_management")
    lossless_v4 = (
        isinstance(management, Mapping)
        and management.get("schema") in LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS
    )
    if not multi_round and not lossless_v4:
        return {}
    return {
        "session_round": round_index + 1,
        "session_rounds": round_limit,
    }


def publish_observation(
    loop: EpisodeLoop, cause: str, *,
    captured_observation: tuple[dict[str, Any], Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the next policy-visible observation; return (pointer, status)."""

    publish_started_ns = time.monotonic_ns()
    if captured_observation is None:
        observation, _images = loop.runtime.observe(loop.observation_root)
        observation = {
            **observation,
            "main_camera": loop.projector.main_camera,
        }
        source = loop.observation_root / observation["observation_id"]
    else:
        captured, source = captured_observation
        observation = {
            **captured,
            "main_camera": loop.projector.main_camera,
        }
    # Query the trusted Environment boundary before publishing the next
    # policy turn. The status record remains Control-only and is never
    # copied into policy state or context.
    environment_status = validate_environment_status(loop.runtime.status())
    # A free-play session has no task to check (check_task is not advertised there).
    if loop.tools.surface.supports("faithful_check_task") and "play" not in loop.plan:
        _atomic_json(loop.control / "policy-task-check.json", {
            "schema": "ora.openeta-task-check.v1",
            "terminal": bool(environment_status["terminal"]),
            "task_success": bool(
                environment_status["terminal"]
                and environment_status.get("reason") == "official_success"
            ),
            "source": environment_status["source"],
            "updated_unix_s": time.time(),
        })
    current_image = _main_observation_image(observation, source)
    runtime_notice: dict[str, Any] | None = loop.round_reset_notice
    loop.round_reset_notice = None
    post_action = loop.pending_execution["action"] if loop.pending_execution is not None else None
    public_execution_receipt = observation.get("previous_execution_receipt")
    if names_timing(public_execution_receipt):
        # The observation's files are the policy's: a backend embeds policy_receipt(receipt), never the sealed one.
        raise SupervisorError("the environment's observation embeds a receipt with its timing evidence")
    if loop.pending_execution is not None:
        public_execution_receipt = _settle_pending_execution(
            loop, observation, current_image,
        )
    remaining = _remaining_budget(loop)
    snapshot = loop.state_store.publish(
        observation=observation,
        observation_root=source,
        cause=cause,
        task_instruction=getattr(loop, "task_instruction", loop.plan.get("task_instruction", loop.plan["task"])),
        previous_public_action=loop.previous_public_action,
        remaining_budget=remaining,
        runtime_notice=runtime_notice,
        previous_execution_receipt=public_execution_receipt,
    )
    record = loop.state_store.steps[-1]
    publish_state(loop, snapshot, record, environment_status)
    state_ref = {
        "step": snapshot["step"],
        "path": record["path"],
        "sha256": record["sha256"],
        "state_sha256": snapshot["state_sha256"],
    }
    pointer = loop.projector.publish(
        observation=observation,
        observation_root=source,
        task_instruction=getattr(loop, "task_instruction", loop.plan.get("task_instruction", loop.plan["task"])),
        remaining_budget=remaining,
        policy_state=state_ref,
        runtime_notice=runtime_notice,
        previous_execution_receipt=public_execution_receipt,
        previous_action=post_action,
    )
    _append(loop.events, {
        "type": "observation",
        "observation_id": observation["observation_id"],
        "policy_state": state_ref,
        "context": pointer,
        "time": time.time(),
        "duration_s": round(
            max(0, time.monotonic_ns() - publish_started_ns) / 1e9, 6
        ),
        "duration_clock": "monotonic",
    })
    _append(loop.events, {
        "type": "environment_status",
        "record": environment_status,
        "time": time.time(),
    })
    loop.last_observation_image = current_image
    _announce_observation(loop, observation, snapshot, pointer)
    # Operator-only work after the policy has its observation (free play: the operator view).
    published = getattr(loop, "observation_published", None)
    if callable(published):
        published(int(snapshot["step"]))
    return pointer, environment_status


def _settle_pending_execution(
    loop: EpisodeLoop, observation: Mapping[str, Any], current_image: Path,
) -> dict[str, Any]:
    """Join the last receipt to this observation and return it."""

    pending = loop.pending_execution
    if loop.last_observation_image is None:
        raise SupervisorError(
            "the observation delta lacks the prior public image"
        )
    public_execution_receipt = pending["receipt"]
    _append(loop.events, {
        "type": "stagnation_observation",
        "record": loop.stagnation.observe(
            before_image=loop.last_observation_image,
            after_image=current_image,
        ),
        "time": time.time(),
    })
    loop.pending_execution = None
    return public_execution_receipt


def _remaining_budget(loop: EpisodeLoop) -> dict[str, Any]:
    """The round budget the policy sees beside this observation."""

    play_budget = getattr(loop, "play_remaining_budget", None)
    if callable(play_budget):
        return play_budget()  # free play: the session's remaining budget (play.py)
    budgets = loop.plan["budgets"]
    remaining = {
        "robot_decisions": (
            budgets["max_robot_decisions"] - loop.round_decisions
        ),
        # Inline tools publish before act returns and the Supervisor settles
        # its call count. Pi supplies the trusted current-act count separately.
        "model_calls": (
            budgets["max_model_calls"] - loop.round_model_calls
            - getattr(loop.planner, "inflight_model_calls", 0)
        ),
        "wall_time_s": max(
            0, round(
                budgets["wall_time_s"]
                - (time.time() - loop.round_started),
                3,
            )
        ),
        **loop.native_allowance.fields(),
    }
    remaining.update(_session_round_budget_fields(
        loop.policy, round_index=loop.round_index, round_limit=loop.round_limit,
        multi_round=loop.multi_round,
    ))
    return remaining


def _announce_observation(
    loop: EpisodeLoop, observation: Mapping[str, Any],
    snapshot: Mapping[str, Any], pointer: dict[str, Any],
) -> None:
    loop.interaction.publish_observation(
        observation_id=str(observation["observation_id"]),
        step=int(snapshot["step"]),
    )
    loop.deliver_policy_event(
        PolicyRuntimeEvent(
            kind="fresh_observation",
            payload=pointer,
            observation_id=str(observation["observation_id"]),
            observation_step=int(snapshot["step"]),
        ),
        stage="fresh_observation",
    )
    loop.publish_live("fresh_observation", {
        "observation_id": observation["observation_id"],
        "observation_step": snapshot["step"],
        "phase": loop.interaction.phase,
    })
