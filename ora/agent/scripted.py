"""Deterministic scripted policy: replays a fixed list of tool calls.

Selected by a RunSpec with ``policy.kind: scripted``; the tests drive the Runtime loop with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ora.agent.toolkit.host.runtime.base import RobotToolRuntime


class ScriptedPlannerError(RuntimeError):
    """A scripted action list is invalid for the current observation."""


@dataclass
class ScriptedPlanner:
    """Deterministic policy that needs no model."""

    tools: RobotToolRuntime
    actions: Sequence[Mapping[str, Any]]
    state_probe: bool = False
    index: int = 0

    def run_observation(self, **kwargs: Any) -> dict[str, Any]:
        if self.index >= len(self.actions):
            return {
                "submitted": False,
                "termination": "policy_no_action",
                "attempts": 1,
            }
        probe: dict[str, Any] | None = None
        if self.state_probe:
            context = kwargs.get("context", {})
            state_ref = (
                context.get("policy_state") if isinstance(context, Mapping) else None
            )
            if not isinstance(state_ref, Mapping) or not isinstance(
                state_ref.get("step"), int,
            ):
                raise ScriptedPlannerError("scripted state probe lacks a policy-state step")
            dumped = self.tools.invoke(
                "observation.dump_state", {"step": state_ref["step"]},
            )
            state = dumped.get("state")
            if not isinstance(state, Mapping) or state.get("step") != state_ref["step"]:
                raise ScriptedPlannerError("scripted state probe returned the wrong step")
            if self.index == 0:
                if state.get("previous_public_action") is not None:
                    raise ScriptedPlannerError(
                        "initial policy-state unexpectedly contains an action"
                    )
            elif not isinstance(state.get("previous_public_action"), Mapping):
                raise ScriptedPlannerError(
                    "policy-state chain lost the previous public action"
                )
            elif not isinstance(state.get("previous_execution_receipt"), Mapping):
                raise ScriptedPlannerError(
                    "policy-state chain lost the previous execution receipt"
                )
            probe = {
                "step": state["step"],
                "state_sha256": state.get("state_sha256"),
                "previous_action_present": (
                    state.get("previous_public_action") is not None
                ),
                "previous_receipt_present": (
                    state.get("previous_execution_receipt") is not None
                ),
            }
        arguments = dict(self.actions[self.index])
        self.index += 1
        tool_name = str(arguments.pop("tool", "robot.submit_action"))
        if tool_name not in self.tools.surface.mutation_tools:
            raise ScriptedPlannerError(
                "scripted action tool must select a surface mutation tool"
            )
        result = self.tools.invoke(tool_name, arguments)
        return {
            "submitted": True,
            "tool_result": result,
            "state_probe": probe,
            "attempts": 1,
        }
