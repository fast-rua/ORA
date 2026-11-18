"""Driving held arms home: the one joint-space drive to rest that every program holding the real arms uses.

An episode's worker (``backends/yam/real_rig.py``: the operator's home, and closing) and the operator tools
(``operator/actions.py``: ``ora cell home`` and the board's home button) both drive home through ``drive``;
each passes its own speed, tolerance and time limit (``HomeSettings``).

Before anything moves, the straight joint-space line from where the arms are to the goal is checked against
the table and between the arms (``path_problems``); a line that is not clear moves nothing. Then the arms
move together, the command ramped toward the goal at ``speed_rad_s``. Each tick:

- an arm whose control loop stopped ends the drive (``dead``);
- an arm whose measured joints lag the command by more than ``tracking_stop_rad`` ends it (``blocked``);
- once the command is at the goal, the drive ends ``at_rest`` when every measured joint is within
  ``tolerance_rad`` of it, or ``not_at_rest`` after ``settle_s`` short of it (an arm held in position control
  sags a little under gravity);
- after ``limit_s`` it ends ``timeout``.

``drive`` only commands; whenever the outcome is not ``at_rest`` the caller holds the arms where they are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

#: The closest the two arms may come anywhere on the way home.
INTER_ARM_MIN_M = 0.06
#: Poses checked along the straight line home.
PATH_SAMPLES = 60

Joints = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class HomeSettings:
    speed_rad_s: float
    period_s: float
    tracking_stop_rad: float
    tolerance_rad: float
    limit_s: float
    settle_s: float | None = None


@dataclass
class HomeOutcome:
    state: str          # at_rest | path_blocked | dead | blocked | not_at_rest | timeout
    detail: str = ""
    arms: list[str] = field(default_factory=list)
    off_rad: float | None = None


def path_problems(kinematics: Any, start: Joints, goal: Joints) -> list[str]:
    """What the straight joint-space line from ``start`` to ``goal`` runs into: the table, or the other arm.
    ``kinematics`` holds the table (``table_distance``) and both arms (``inter_arm_distance``); its state is
    left at the line's last pose, so a caller that keeps using it sets its state again."""

    for t in np.linspace(0.0, 1.0, PATH_SAMPLES):
        kinematics.set_state({arm: start[arm] + t * (goal[arm] - start[arm]) for arm in start})
        if len(start) > 1 and kinematics.inter_arm_distance() < INTER_ARM_MIN_M:
            return [f"the arms come within {INTER_ARM_MIN_M} m of each other at {t:.0%}"]
        for arm in start:
            distance = kinematics.table_distance(arm)
            if distance is not None and distance < 0.0:
                return [f"the {arm} arm passes through the table at {t:.0%}"]
    return []


def drive(goal: Joints, *, read: Callable[[], Joints], command: Callable[[Joints], None],
          alive: Callable[[], Mapping[str, bool]], wait: Callable[[], None], settings: HomeSettings,
          kinematics: Any = None) -> HomeOutcome:
    """Drive the arms in ``goal`` to it (module docstring). ``read`` measures the joints, ``command`` sends one
    tick's targets, ``wait`` paces one tick; with ``kinematics`` the line is checked first."""

    start = {arm: np.asarray(q, float).copy() for arm, q in read().items() if arm in goal}
    if kinematics is not None:
        problems = path_problems(kinematics, start, goal)
        if problems:
            return HomeOutcome("path_blocked", "; ".join(problems))
    running = alive()
    live = [arm for arm in start if running.get(arm, True)]
    if not live:
        return HomeOutcome("dead", "no arm's control loop runs")
    step = settings.speed_rad_s * settings.period_s
    commanded = {arm: start[arm] for arm in live}
    measured = start
    began, arrived = time.monotonic(), None
    while time.monotonic() - began < settings.limit_s:
        if all(np.max(np.abs(goal[arm] - commanded[arm])) < 1e-9 for arm in live):
            off = max(float(np.max(np.abs(goal[arm] - measured[arm]))) for arm in live)
            if off < settings.tolerance_rad:
                return HomeOutcome("at_rest", off_rad=off)
            arrived = arrived if arrived is not None else time.monotonic()
            if settings.settle_s is not None and time.monotonic() - arrived > settings.settle_s:
                return HomeOutcome("not_at_rest", f"{off:.3f} rad from rest", off_rad=off)
        wait()
        running = alive()
        dead = [arm for arm in live if not running.get(arm, True)]
        if dead:
            return HomeOutcome("dead", f"the {', '.join(dead)} control loop stopped", dead)
        commanded = {arm: commanded[arm] + np.clip(goal[arm] - commanded[arm], -step, step) for arm in live}
        command(commanded)
        measured = {arm: np.asarray(q, float) for arm, q in read().items()}
        lagging = [arm for arm in live if np.max(np.abs(commanded[arm] - measured[arm])) > settings.tracking_stop_rad]
        if lagging:
            return HomeOutcome("blocked", f"the {', '.join(lagging)} arm does not follow", lagging)
    return HomeOutcome("timeout", f"not at rest in {settings.limit_s:g} s")
