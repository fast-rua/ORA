"""The plan's result tier: what the policy carried into the episode beyond its prompt and tools.

Reports group results by tier and never pool across tiers. ``zero_shot`` is the
default; ``zero_shot_no_check`` is the same baseline without ``check_task``.
"""

from __future__ import annotations

from typing import Any, Mapping

from .values import RunSpecError

RESULT_TIERS = ("zero_shot", "with_experience", "zero_shot_no_check", "frozen_program")


def resolve_result_tier(
    spec: Mapping[str, Any], policy: Mapping[str, Any], surface: Any, play: Any,
) -> str:
    """The plan's result tier; refuses a tier the plan cannot honestly carry.

    ``zero_shot`` and ``zero_shot_no_check`` differ only in whether a surface
    that offers ``check_task`` advertises it, so the tier must match the
    allowlist both ways. ``with_experience`` needs an experience binding and
    ``frozen_program`` names results produced outside the Runtime; this tree
    has neither, so both are refused. Free play has no task and stays
    ``zero_shot``.
    """

    tier = spec.get("result_tier", "zero_shot")
    if tier not in RESULT_TIERS:
        raise RunSpecError(f"unsupported result_tier: {tier}")
    if tier == "with_experience":
        raise RunSpecError(
            "result_tier with_experience needs an experience library binding; "
            "this tree has none (policy.memory_policy is disabled)"
        )
    if tier == "frozen_program":
        raise RunSpecError(
            "result_tier frozen_program names results produced outside the ORA Runtime; "
            "a RunPlan cannot carry it"
        )
    if play is not None:
        if tier != "zero_shot":
            raise RunSpecError("a free-play plan keeps result_tier zero_shot")
        return tier
    if policy.get("kind") != "native_api":
        # Only a model sees the tool allowlist; a scripted policy has no check_task to withhold.
        if tier != "zero_shot":
            raise RunSpecError("result_tier zero_shot_no_check applies to native_api policies only")
        return tier
    offered = "check_task" in {item["name"] for item in surface.definitions}
    allowlist = policy.get("tool_allowlist")
    withheld = offered and isinstance(allowlist, list) and "check_task" not in allowlist
    if tier == "zero_shot_no_check" and not withheld:
        raise RunSpecError(
            "result_tier zero_shot_no_check needs a surface that offers check_task "
            "and a tool_allowlist without it"
        )
    if tier == "zero_shot" and withheld:
        raise RunSpecError(
            "a plan that withholds check_task must declare result_tier zero_shot_no_check"
        )
    return tier


__all__ = ["RESULT_TIERS", "resolve_result_tier"]
