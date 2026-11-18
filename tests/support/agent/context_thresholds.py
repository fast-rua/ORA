"""The L4 window share and the L5 formula label: test mirrors of the Pi core's thresholds.

The Pi core computes them in a run (``ora-pi/src/context/stages/pressure_externalize/trigger.ts``)
and records the formula in projection evidence (``resolved_episode_checkpoint_trigger_formula``,
``stages/episode_capsule/stage.ts``).
They were in ``ora/agent/context/lossless/thresholds.py`` until 2026-11-04, which
keeps the L5 and L6 triggers sealing reads; see its docstring for why L4 is a
share of the usable window above its absolute floor.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.context.lossless.management import WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS

#: The share of usable input at which L4 externalizes, recovered from the
#: absolute ``trigger_estimated_tokens`` every baseline carries: 48000 against
#: the 197952 usable tokens of the 200K window it was chosen for. Keeping the
#: share rather than the count is what makes the layer mean the same thing on a
#: model with a different window.
ESTIMATE_PRESSURE_WINDOW_SHARE = 48_000 / (200_000 - 2_048)

#: ``resolved_episode_checkpoint_trigger_formula`` in projection evidence.
ABSOLUTE_EPISODE_TRIGGER_FORMULA = "absolute_tokens"
WINDOW_SCALED_EPISODE_TRIGGER_FORMULA = (
    "max(absolute_tokens, int(hard_input_tokens * ratio))"
)


def estimate_pressure_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where L4 externalizes losslessly.

    The absolute count stays as a floor, so a 200K binding keeps exactly the
    behaviour its figure was picked for and nothing changes there; the share
    governs above it. Below about 620K of usable input this sits below
    ``thresholds.episode_checkpoint_trigger``; above it the image trigger, not
    this one, keeps L4 ahead of L5 (see ``ora/agent/context/lossless/thresholds.py``).
    """

    return max(
        int(management["trigger_estimated_tokens"]),
        int(hard_input_tokens * ESTIMATE_PRESSURE_WINDOW_SHARE),
    )


def episode_checkpoint_trigger_formula(management: Mapping[str, Any]) -> str:
    """The formula ``thresholds.episode_checkpoint_trigger`` applied, for evidence."""

    if management.get("schema") in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        return WINDOW_SCALED_EPISODE_TRIGGER_FORMULA
    return ABSOLUTE_EPISODE_TRIGGER_FORMULA


__all__ = [
    "ABSOLUTE_EPISODE_TRIGGER_FORMULA", "ESTIMATE_PRESSURE_WINDOW_SHARE", "WINDOW_SCALED_EPISODE_TRIGGER_FORMULA",
    "episode_checkpoint_trigger_formula", "estimate_pressure_trigger",
]
