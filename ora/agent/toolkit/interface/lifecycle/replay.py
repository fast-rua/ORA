"""Snapshot lifecycle is output-only, with no live Supervisor fallback."""
from dataclasses import dataclass
from typing import Callable
from ..types import AccessDenied, Response


@dataclass(frozen=True, slots=True)
class Episode:
    _record: Callable[[str, dict], Response] | None = None

    def report_issue(self, *, issue: dict) -> Response:
        """Record a diagnostic issue in replay output, never in the source run."""
        return self._invoke('report_issue', {'issue': issue})

    def finish_episode(self, *, request: dict) -> Response:
        """Record intent only; no live episode may be finished by snapshot code."""
        return self._invoke('finish_episode', {'request': request})

    def _invoke(self, method, parameters):
        if self._record is None:
            raise AccessDenied('replay lifecycle output recorder is unavailable')
        return self._record(method, parameters)
