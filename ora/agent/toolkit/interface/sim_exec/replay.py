"""Phase-one snapshot replay has no online execution capability: an action is proposed, never executed."""
from dataclasses import dataclass
from ..types import ActionNotExecuted


@dataclass(frozen=True, slots=True)
class Actions:
    """The tool runs up to acting; its first action stops it with the proposal (``ActionNotExecuted``)."""

    def submit_action(self, *, action: dict, preparation: dict | None = None):
        raise ActionNotExecuted("submit_action", action, None)

    def execute_action(self, action: dict, budget: int):
        raise ActionNotExecuted("execute_action", action, budget)
