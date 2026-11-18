"""The plan's partition: official evaluation or practice.

A practice plan may not use an episode from any benchmark's evaluation set.
:func:`ora.environment.benchmarks.evaluation_sets.evaluation_set_hit` applies the
machine-readable list (``evaluation-sets-v1.json``) to the resolved plan; a hit
refuses the plan, and an environment the list does not cover has no practice
side, so practice there is refused too. A practice plan records the list's
sha256 so its evidence names the rules it was checked against. Free play is
never practice. Reports keep the two partitions apart (``run_report``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.benchmarks.evaluation_sets import MANIFEST, evaluation_set_hit

from .values import RunSpecError, file_sha256

PARTITIONS = ("evaluation", "practice")


def resolve_partition(spec: Mapping[str, Any], portable: Mapping[str, Any]) -> dict[str, Any]:
    """The ``partition`` block of a plan whose runtime, seed and episode mode are resolved."""

    name = spec.get("partition", "evaluation")
    if name not in PARTITIONS:
        raise RunSpecError(f"unsupported partition: {name}")
    if name == "evaluation":
        return {"name": name}
    if portable.get("episode_mode") == "play":
        raise RunSpecError("a free-play plan is not practice; it keeps partition evaluation")
    reason = evaluation_set_hit(portable)
    if reason is not None:
        raise RunSpecError(f"a practice plan may not use an evaluation-set episode: {reason}")
    return {"name": name, "evaluation_sets_sha256": file_sha256(MANIFEST)}


__all__ = ["PARTITIONS", "resolve_partition"]
