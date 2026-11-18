"""The Runtime's own remaining-budget record for ``finish_episode``.

The toolkit serves ``compute``, ``report_issue``, ``check_task`` and
``finish_episode`` (``dispatch.install``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.

``load_policy_state`` comes from ``common``,
which reads it from ``core`` at call time.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.evidence.policy_state import PolicyStateError

from ora.agent.toolkit.host.runtime.common import load_policy_state


class LifecycleRuntimeMixin:
    """The Runtime's own remaining-budget record for ``finish_episode``."""

    def _runtime_remaining_budget(self) -> dict[str, Any]:
        """The Runtime's own remaining-budget count at this finish boundary.

        Read from the persisted policy state, never from anything the policy
        said. ``_finish`` refuses while an action proposal is pending, so
        the latest published step is the current decision boundary and its
        ``robot_decisions`` is exact. ``model_calls`` is the count as of that
        step and therefore excludes the turn now finishing; the record says so
        rather than implying a freshness it does not have.
        """

        unavailable = {
            "available": False,
            "source": "policy_state_unavailable",
            "robot_decisions": None,
            "model_calls": None,
        }
        if self.state_root is None:
            return unavailable
        try:
            state = load_policy_state(self.state_root.parent, None)
        except (PolicyStateError, OSError, ValueError):
            return unavailable
        budget = state.get("remaining_budget")
        if not isinstance(budget, Mapping):
            return unavailable
        record = {
            "available": True,
            "source": "runtime_policy_state_snapshot",
            "policy_state_step": state.get("step"),
            "policy_state_sha256": state.get("state_sha256"),
            "model_calls_freshness": "as_of_this_step_excludes_the_finishing_turn",
        }
        for key in (
            "robot_decisions", "model_calls", "wall_time_s",
            "session_round", "session_rounds",
        ):
            if key in budget:
                record[key] = budget[key]
        return record
