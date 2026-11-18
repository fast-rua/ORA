"""The refusals of pacing a path: a move refused before any motion, whichever timing measured it.

``motion_pace`` and the timing worker (``ora/environment/planners/timing/``) raise
the same refusals with the same words, so a receipt does not depend on the timing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

WRIST_UNWIND_CODE = "paced_path_wrist_unwind"
BUDGET_CODE = "paced_motion_budget"


class PaceRefusal(ValueError):
    """A move refused before any motion (``code``, ``constraints``)."""

    def __init__(self, message: str, *, code: str, constraints: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code, self.constraints = code, dict(constraints or {})


def atomic_start(spans: Sequence[tuple[int, int]], nodes: int) -> tuple[int, tuple[int, int] | None]:
    """Where a cut after ``nodes`` segments moves back to so it falls inside no span, and the last span it left."""

    span = None
    while True:
        inside = next((item for item in spans if item[0] < nodes < item[1]), None)
        if inside is None:
            return nodes, span
        nodes, span = inside[0], inside


def budget_refusal(budget: int, needed: int) -> PaceRefusal:
    """The refusal of a move that needs more ticks than one action may spend and is not cut."""

    return PaceRefusal(
        f"this move needs {needed} control ticks at the requested speed, more than the "
        f"{int(budget)} one action may spend; nothing was executed. Use a faster speed or split the move",
        code=BUDGET_CODE, constraints={"tick_budget": int(budget), "needed_ticks": needed})


def cover_refusal(budget: int) -> PaceRefusal:
    """The refusal of a cut move whose budget does not cover even its first segment."""

    return PaceRefusal(f"a budget of {budget} ticks does not cover even the first part of the move",
                       code=BUDGET_CODE, constraints={"tick_budget": budget})


def unsplit_refusal(budget: int, needed: int, from_fraction: float, to_fraction: float) -> PaceRefusal:
    """The refusal of a cut move whose part that fits ends inside the large wrist turn it starts with."""

    return PaceRefusal(
        f"a budget of {budget} ticks cannot finish the large wrist turn the move starts with, and that is "
        "never cut part way; nothing was executed. Use a faster speed or split the move", code=BUDGET_CODE,
        constraints={"tick_budget": budget, "needed_ticks": needed,
                     "unsplit_span": {"from_fraction": round(float(from_fraction), 4),
                                      "to_fraction": round(float(to_fraction), 4)}})


__all__ = ["BUDGET_CODE", "PaceRefusal", "WRIST_UNWIND_CODE", "atomic_start", "budget_refusal", "cover_refusal",
           "unsplit_refusal"]
