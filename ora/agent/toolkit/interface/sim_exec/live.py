"""Actions calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Actions:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def submit_action(self, *, action: dict, preparation: dict | None = None) -> Response:
        return self._backend.submit_action(action=action, preparation=preparation)

    def execute_action(self, action: dict, budget: int) -> Response:
        """Execute a primitive synchronously; return its receipt and remaining budget."""
        return self._backend.execute_action(action=action, budget=budget)


@dataclass(frozen=True, slots=True)
class SynchronousActions:
    """Budgeted execution only: legacy proposal submission is not a capability."""
    _backend: Any

    def execute_action(self, action: dict, budget: int) -> Response:
        return self._backend.execute_action(action=action, budget=budget)
