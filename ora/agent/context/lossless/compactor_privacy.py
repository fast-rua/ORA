"""The finish refusals whose task-checker verdict the rollout Compactor never sees.

The main policy may call the faithful task checker, but a rollout checkpoint is
explicitly forbidden from carrying an official outcome. The Compactor packet is
the policy-visible history, so the checker's Boolean is withheld in the two
places the policy received it, preserving tool adjacency and evidence identity:

- a ``check_task`` result, replaced by a withheld marker;
- a ``finish_episode`` refusal coded ``task_success_latched`` or
  ``task_success_not_confirmed`` (``ora.agent.toolkit.host.finish.task_check``), which
  repeats the Boolean in ``error.constraints.task_success``: that one field is
  replaced by a marker, every other field stays.

Only these known places are masked. ``task_success`` (or any reserved key)
anywhere else stays a privacy violation that rejects the packet
(``causal_value._compactor_private_violations``). The policy's own context keeps
both results exactly as delivered. The Pi core applies the rule
(``ora-pi/src/context/projection.ts``, ``compactorHistoryMessage``) with the refusal
codes sent from here (``ora/agent/pi/domain.py``); the Python statement of it the
tests check against is ``tests/support/agent/compactor_verdicts.py``.
"""

from __future__ import annotations

FINISH_VERDICT_REFUSAL_CODES = frozenset({"task_success_latched", "task_success_not_confirmed"})

__all__ = ["FINISH_VERDICT_REFUSAL_CODES"]
