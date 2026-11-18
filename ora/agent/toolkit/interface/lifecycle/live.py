"""Episode calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Episode:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def report_issue(self, *, issue: dict) -> Response:
        return self._backend.report_issue(issue=issue)

    def finish_episode(self, *, request: dict) -> Response:
        return self._backend.finish_episode(request=request)
