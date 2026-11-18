"""A plan a dry run made, kept for the real call that asks for exactly it on the same observation.

The key is the observation's identity and every input of the plan, so the same request on the same observation runs
the trajectory the preview showed, and any change of observation, state, budget or request plans as ever. The record
lives in the tool state on the policy side, finite JSON, one observation's worth: replay reads it back from there, and a
float survives the round trip exactly, so the trajectory that runs is byte for byte the one that was planned.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...interface.types import canonical, digest
from . import move, planner, recovery, timing

STATE_KEY = "tool.kept_plans"
#: Plans kept per observation; the oldest goes first.
LIMIT = 4


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    if isinstance(value, timing.Pace):
        return value.record()
    return float(value)


def key(identity: Mapping[str, Any], **inputs: Any) -> str:
    """Pure computation: what identifies one plan: the observation it was made on and everything it was made from."""
    return digest(canonical({"observation": _plain(identity), "inputs": _plain(inputs)}))


def recall(api: Any, identity: Mapping[str, Any], plan_key: str) -> dict[str, Any] | None:
    """The kept record of ``plan_key`` on this observation; None, as for an old run, when there is none."""
    try:
        kept = api.state.read(STATE_KEY)
    except Exception:  # a run or a replay without the key: nothing was kept
        return None
    if not isinstance(kept, Mapping) or kept.get("observation") != _plain(identity):
        return None
    return kept["plans"].get(plan_key)


def keep(api: Any, identity: Mapping[str, Any], plan_key: str, record: Mapping[str, Any]) -> None:
    """Keep a dry run's plan for this observation; a state that cannot take it only loses the reuse."""
    try:
        try:
            kept = api.state.read(STATE_KEY)
        except Exception:  # the first one of the run
            kept = None
        plans = dict(kept["plans"]) if (isinstance(kept, Mapping) and kept.get("observation") == _plain(identity)) else {}
        plans[plan_key] = record
        while len(plans) > LIMIT:
            del plans[next(iter(plans))]
        api.state.write(STATE_KEY, {"observation": _plain(identity), "plans": plans})
    except Exception:  # noqa: BLE001 - the preview's answer stands without it
        pass


def _knots(knots: Mapping[str, Any]) -> dict[str, Any]:
    return {"joints": _plain(knots["joints"]), "fractions": _plain(knots["fractions"]),
            "duration_s": float(knots["duration_s"]), "native_actions": int(knots["native_actions"])}


def move_record(planned: move.Planned) -> dict[str, Any]:
    """What ``move.run`` needs of a plan, as finite JSON."""
    path = planned.path
    return {"route": path.route, "goals": _plain(path.goals), "bridged": _plain(path.bridged),
            "warnings": _plain(path.warnings),
            "start_joints": {arm: _plain(rows[0]) for arm, rows in path.grid.joints.items()},
            "knots": _knots(planned.knots), "fraction": float(planned.fraction),
            "whole_native_actions": int(planned.whole_native_actions), "expected": _plain(planned.expected),
            "settle": _plain(planned.settle), "plan_goals": _plain(planned.goals),
            "native_action_budget": int(planned.native_action_budget), "pace": _plain(planned.pace),
            "slow_pace": _plain(planned.slow_pace), "slow_from": _plain(planned.slow_from),
            "peaks": _plain(planned.peaks), "screen": _plain(planned.screen)}


def _pace(value: Mapping[str, Any] | None) -> timing.Pace | None:
    return None if value is None else timing.Pace(**value)


def move_plan(record: Mapping[str, Any]) -> move.Planned:
    """The plan of ``move_record``: every field ``move.run`` and a dry run read, the path's grid its start alone."""
    grid = timing.Grid((0.0,), {arm: np.asarray([row], dtype=float) for arm, row in record["start_joints"].items()})
    path = planner.PathPlan(grid, record["route"], record["goals"], None, tuple(record["bridged"]),
                            tuple(record["warnings"]))
    return move.Planned(path, record["knots"], record["fraction"], record["whole_native_actions"], record["expected"],
                        record["settle"], record["plan_goals"], record["native_action_budget"], _pace(record["pace"]),
                        _pace(record["slow_pace"]), record["slow_from"], record["peaks"], record["screen"])


def recovery_record(planned: recovery.Planned, warnings: Any = ()) -> dict[str, Any]:
    """What ``recovery.run`` and its report need of a plan, as finite JSON."""
    return {"options": {arm: {"method": option.method, "detail": _plain(option.detail)}
                        for arm, option in planned.options.items()},
            "knots": _knots(planned.knots), "fraction": float(planned.fraction),
            "whole_native_actions": int(planned.whole_native_actions), "expected": _plain(planned.expected),
            "settle": _plain(planned.settle), "goals": _plain(planned.goals), "targets": _plain(planned.targets),
            "native_action_budget": int(planned.native_action_budget), "pace": _plain(planned.pace),
            "screens": _plain(planned.screens), "warnings": _plain(warnings)}


def recovery_plan(record: Mapping[str, Any]) -> recovery.Planned:
    """The plan of ``recovery_record``; a candidate's phases and the grid were only ever the way to it."""
    options = {arm: recovery.Option(value["method"], (), value["detail"]) for arm, value in record["options"].items()}
    return recovery.Planned(options, timing.Grid((0.0,), {}), record["knots"], record["fraction"],
                            record["whole_native_actions"], record["expected"], record["settle"], record["goals"],
                            record["targets"], record["native_action_budget"], _pace(record["pace"]),
                            tuple(record["screens"]))


__all__ = ["LIMIT", "STATE_KEY", "keep", "key", "move_plan", "move_record", "recall", "recovery_plan", "recovery_record"]
