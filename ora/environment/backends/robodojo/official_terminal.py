"""The official terminal rule of one RoboDojo episode (``EvaluationLifecycle.advance_official_terminal``).

This is upstream's ``is_episode_end`` for one environment, plus what an
extended RunPlan adds (``contracts/official_limit.py``): the episode may run on
past ``step_lim``, its official result stays the one at ``step_lim``, and a
later success is recorded apart.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ora.contracts.official_limit import limit_snapshot

from .bridge import RobodojoError


def _final_checks_registered(environment: Any) -> bool:
    """Whether the official limit itself would evaluate something: no shipped task registers this."""
    manager = environment.reward_manager
    return bool(manager.final_check_list[0] or manager.final_score_list[0])


def advance_official_terminal(environment: Any) -> dict[str, Any] | None:
    if not environment.reward_registered:
        raise RobodojoError("official reward lifecycle has not been initialized")
    if environment._official_terminal is not None:
        return deepcopy(environment._official_terminal)
    count = environment.take_action_cnt[0]
    # Upstream's final check runs once, at the official limit. A task that
    # registered one is decided there, so it ends there whatever the plan
    # allows; for every other task the check evaluates nothing new.
    final = (count >= environment.native_action_ceiling or not environment.success[0]
             or (count >= environment.step_lim and _final_checks_registered(environment)))
    reward = environment.reward_manager.get_reward(final_check=final)[0]
    if reward > 1 - 1e-3:
        environment.end_flag[0], environment.success[0] = True, True
    elif final:
        environment.end_flag[0], environment.success[0] = True, False
    if count >= environment.step_lim and not environment.end_flag[0] and environment.official_limit_snapshot is None:
        environment.official_limit_snapshot = limit_snapshot(count)
    if environment.end_flag[0]:
        if environment.official_limit_snapshot is None:
            environment._official_terminal = {"official_success": bool(environment.success[0]),
                                              "native_actions": count}
        else:
            # The official result was settled at the limit and a later success never changes it.
            environment._official_terminal = {"official_success": False, "native_actions": count,
                                              "success_after_official_limit": bool(environment.success[0])}
    return deepcopy(environment._official_terminal)


__all__ = ["advance_official_terminal"]
