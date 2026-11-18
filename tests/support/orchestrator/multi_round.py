"""Synthetic persistent sessions on the Pi agent core, for Supervisor-level tests.

``spec()`` is the Synthetic ``native_atomic_causal_v1`` cell with the live
causal Context (``LIVE_CAUSAL_CONTEXT``) and one decision per round; the tests
set ``session`` for K rounds. The model is a ``FakeProvider`` script (the Pi
core runs it in TypeScript): one reply per provider request, in order, so a
script is the list of tool calls the model makes. What a provider stub used to
assert inside ``complete`` the tests now read back from the run's evidence
(``provider_requests``), and every run needs the Pi core
(``tests/support/agent_core/preconditions.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment import SyntheticSession
from ora.environment.base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status
from tests.support.orchestrator.native_synthetic_spec import synthetic_spec

ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = "synthetic-native-atomic-causal-cache-v1"
USAGE = {"input_tokens": 1000, "output_tokens": 20}

#: The live lossless ActionStep Context: management v6 as the default baseline
#: has it (the agent core refuses v4, v5 and v7).
LIVE_CAUSAL_CONTEXT: dict[str, Any] = {
    "context_retention": "action_step_lossless_v4",
    "summary_policy": "rollout_checkpoint_v1",
    "context_management": {
        "schema": "ora.native-context-management.v6",
        "causal_exact_tail_action_steps": 3, "inline_media_tail_action_steps": 2,
        "eager_large_result_bytes": 50000, "eager_preview_bytes": 2048,
        "eager_exact_deduplication": True, "trigger_inline_images": 24,
        "trigger_estimated_tokens": 48000, "trigger_provider_input_ratio": 0.7,
        "episode_checkpoint_trigger_tokens": 150000,
        "emergency_fallback_ratio": 0.8, "emergency_target_ratio": 0.65,
        "checkpoint_target_estimated_tokens": 1200, "checkpoint_max_estimated_tokens": 1600,
        "tail_aging_stride_action_steps": 4,
    },
}

FAILURE = {"outcome": "failure", "reason": "qualification round complete",
           "attempts": ["inspected the reset observation"]}


def live_exp01b_spec(config: str = BASE_CONFIG) -> dict[str, Any]:
    """The Synthetic Native composition with this tree's live causal Context."""

    spec = copy.deepcopy(synthetic_spec(config, seed=0))
    spec["policy"].update(copy.deepcopy(LIVE_CAUSAL_CONTEXT))
    return spec


def spec() -> dict[str, Any]:
    """One decision, two model calls and one simulator step per round."""

    value = live_exp01b_spec()
    value["experiment"] = "persistent-agent-session-test"
    value["output_workspace"] = "outputs/persistent-agent-session-test"
    value["milestone"] = "MULTI-ROUND-QUALIFICATION"
    value["runtime"]["terminal_actions"] = 1
    value["budgets"] = {"max_robot_decisions": 1, "max_model_calls": 2, "max_sim_steps": 1, "wall_time_s": 30}
    return value


# -- Model scripts: one reply per provider request ---------------------------------------------


def tool_reply(call_id: str, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A scripted assistant reply with one tool call."""

    return {"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments or {}), sort_keys=True)},
    }]}, "usage": dict(USAGE)}


def script(calls: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """``(name, arguments)`` per request; call ids ``<name>-<request index>``."""

    return [tool_reply(f"{name}-{index}", name, arguments) for index, (name, arguments) in enumerate(calls)]


def finish_each_round(rounds: int) -> list[dict[str, Any]]:
    """Each round's only request finishes it as a failure (``finish-round-<index>``)."""

    return [tool_reply(f"finish-round-{index}", "finish_episode",
                       {**FAILURE, "reason": f"qualification round {index} complete"})
            for index in range(rounds)]


def act_then_succeed(rounds: int) -> list[dict[str, Any]]:
    """Per round: ``close_gripper``, then ``finish_episode`` with outcome success."""

    replies = []
    for index in range(rounds):
        replies.append(tool_reply(f"close_gripper-{index}", "close_gripper"))
        replies.append(tool_reply(f"finish_episode-{index}", "finish_episode", {"outcome": "success"}))
    return replies


def close_every_request(requests: int, prefix: str = "close-round") -> list[dict[str, Any]]:
    """Every request closes the gripper (``<prefix>-<index>``)."""

    return [tool_reply(f"{prefix}-{index}", "close_gripper") for index in range(requests)]


def fake(responses: Sequence[Mapping[str, Any]]) -> Any:
    from ora.agent.providers.fake import FakeProvider

    return FakeProvider(responses=[copy.deepcopy(dict(item)) for item in responses])


# -- Running and reading back -------------------------------------------------------------------


def require_pi_core() -> None:
    from tests.support.agent_core.preconditions import require_pi_runtime

    require_pi_runtime()


def run(run_root: Path, plan: Mapping[str, Any], responses: Sequence[Mapping[str, Any]] | None, *,
        run_id: str = "multi-round-qualification", **options: Any) -> dict[str, Any]:
    """``run_episode`` on the Pi core with ``responses`` as the model; ``plan.json`` written first.

    ``responses=None`` leaves the plan's own provider (the test points it at a loopback endpoint).
    """

    from ora.orchestrator.episode.coordinator import run_episode

    require_pi_core()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provider = None if responses is None else fake(responses)
    return run_episode(run_root=run_root, plan=plan, run_id=run_id, provider_override=provider, **options)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agent_events(run_root: Path) -> list[dict[str, Any]]:
    return jsonl(run_root / "agent-events.jsonl")


def provider_requests(run_root: Path) -> list[dict[str, Any]]:
    """Every canonical provider request the core recorded, in order."""

    return [json.loads((run_root / event["payload"]["canonical_request_ref"]["path"]).read_text(encoding="utf-8"))
            for event in agent_events(run_root) if event["type"] == "provider_request"]


def rendered_messages(request: Mapping[str, Any]) -> str:
    return json.dumps(request.get("messages", []), ensure_ascii=False)


def announces_round(request: Mapping[str, Any], round_number: int) -> bool:
    """The request carries the ``session_round_start`` notice of ``round_number`` (1-based)."""

    rendered = rendered_messages(request)
    return "session_round_start" in rendered and (
        f'\\"round\\": {round_number}' in rendered or f'\\"round\\":{round_number}' in rendered)


def session_closed(run_root: Path) -> bool:
    return any(event["type"] == "session_closed" for event in agent_events(run_root))


# -- Environments ---------------------------------------------------------------------------------


class SuccessSyntheticSession(SyntheticSession):
    """Terminal on its first action with official success."""

    def status(self) -> dict[str, Any]:
        terminal = bool(self.terminal)
        reason = self.abort_reason or ("official_success" if terminal else None)
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if reason is not None else "running",
            "terminal": reason is not None,
            "reason": reason,
            "source": "synthetic_success_session",
            "policy_visibility": "control_only",
            "signals": [],
        })

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {"schema": "ora.synthetic-success-verdict.v1",
                   "official_success": bool(self.terminal and self.abort_reason is None)}
        path = self.output_root / "official-verdict.json"
        path.write_text(json.dumps(verdict, sort_keys=True) + "\n", encoding="utf-8")
        return verdict
