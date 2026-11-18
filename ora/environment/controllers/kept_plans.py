"""The planner a session asks, keeping what a dry run asked so the real call runs the path the preview showed.

A dry run's answers (``previewing``) are kept, and the same request later returns the kept answer instead of solving
again. The key is the request itself, world hash and start joints included, so a kept plan cannot cross a change of the
world or of the robot; a request that differs in anything is solved as ever.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any, Iterator, Mapping

#: How many dry-run answers a session keeps; the oldest goes first.
KEPT_PLANS = 8


def _keyed(value: Any) -> Any:
    """A request without its file paths: a world is named by its hash, wherever its snapshot was written."""
    if isinstance(value, Mapping):
        return {str(key): _keyed(item) for key, item in value.items() if key != "path"}
    if isinstance(value, (list, tuple)):
        return [_keyed(item) for item in value]
    return value


def plan_key(method: str, request: Mapping[str, Any]) -> str:
    """Pure computation: the identity of one planner question."""
    encoded = json.dumps([method, _keyed(request)], sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class KeptPlans:
    """A planner that keeps what a dry run asked and answers the same question once from it."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self._kept: dict[str, Any] = {}
        self._previewing = False
        self.reused = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.planner, name)

    @contextlib.contextmanager
    def previewing(self) -> Iterator[None]:
        """The questions asked inside the block are a dry run's: their answers are kept."""
        self._previewing = True
        try:
            yield
        finally:
            self._previewing = False

    def _ask(self, method: str, request: Mapping[str, Any]) -> Any:
        key = plan_key(method, request)
        if self._previewing:
            self._kept[key] = getattr(self.planner, method)(request)
            while len(self._kept) > KEPT_PLANS:
                del self._kept[next(iter(self._kept))]
            return self._kept[key]
        if key in self._kept:
            self.reused += 1
            return self._kept.pop(key)
        return getattr(self.planner, method)(request)

    def plan(self, request: Mapping[str, Any]) -> Any:
        return self._ask("plan", request)

    def solve_ik(self, request: Mapping[str, Any]) -> Any:
        return self._ask("solve_ik", request)

    def stamped(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """``receipt`` saying the action ran a plan a dry run made, where it did; the count starts again."""
        if self.reused:
            receipt["ran_previewed_plan"] = True
        self.reused = 0
        return receipt


def previewing(planner: Any) -> Any:
    """``planner.previewing()`` where it keeps plans; a dry run with no planner still answers."""
    keep = getattr(planner, "previewing", None)
    return keep() if callable(keep) else contextlib.nullcontext()


def stamped(planner: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    """``KeptPlans.stamped`` where ``planner`` keeps plans; ``receipt`` otherwise."""
    stamp = getattr(planner, "stamped", None)
    return stamp(receipt) if callable(stamp) else receipt


__all__ = ["KEPT_PLANS", "KeptPlans", "plan_key", "previewing", "stamped"]
