# Copyright contributors to the Robot Harness Runtime project.
"""Where each context layer answers for the model it is bound to.

L4 externalizes losslessly at a share of the usable input window, with the
absolute ``trigger_estimated_tokens`` as a floor. Every shipped baseline
carries 48,000, chosen against a 200K context; read against a 1M model it is
4.7% of usable input instead of the 24% it was picked to be, so the share
governs above the floor.

L5 summarizes one closed Episode at an absolute Provider-input watermark. From
Context management v6 it is
``episode_checkpoint_trigger_tokens`` alone, 150,000 in every maintained
configuration, and no longer scales with the window: v5's
``max(tokens, 0.60 x hard)`` put it at 618,086 tokens on a 1M DeepSeek window
(1,030,144 usable), where it effectively never fired. v4/v5 keep that formula
here so their recorded runs, frozen archives and explicit historical
treatments read back with the exact threshold they ran with.

The property that matters is order: the lossless answer should be reached
before the summarizing one. In token terms that holds only up to about 620K of
usable input (L4's share reaches 150,000 at 618,600); at a 1M window L4's token
trigger is about 250,000 (249,792 at 1,030,144 usable), above L5. In practice
L4 still precedes L5 there, because its 24-inline-image trigger fires far
earlier than either token trigger. The v6 validator only checks the
window-independent part: the L4 floor below the absolute L5. L6 is unchanged
(80% of usable input, target 65%).

The Pi core computes every trigger in a run, in its stages
(``ora-pi/src/context/stages/pressure_externalize/trigger.ts``, ``episode_capsule``,
``emergency_checkpoint``).
Python keeps the L5 and L6 triggers that sealing's coverage check reads
(``ora/orchestrator/evaluation/context_coverage.py``); the L4 share and the L5
formula label, which only tests read, are in ``tests/support/agent/context_thresholds.py``.
"""

from __future__ import annotations

from typing import Any, Mapping

from .management import WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS


def episode_checkpoint_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where L5 closes an Episode: absolute from v6, window-scaled in v4/v5."""

    if management["episode_checkpoint_trigger_tokens"] is None:  # v10: by ratio alone
        return int(hard_input_tokens * float(management["episode_checkpoint_trigger_ratio"]))
    tokens = int(management["episode_checkpoint_trigger_tokens"])
    if management.get("schema") in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        return max(
            tokens,
            int(
                hard_input_tokens
                * float(management["episode_checkpoint_trigger_ratio"])
            ),
        )
    return tokens


def compaction_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where L6 summarizes: v8's absolute watermark capped by the window ratio.

    v4-v7 admit L6 only at ``emergency_fallback_ratio`` of the hard window, an
    emergency that a 1M window almost never reaches; v8 compacts at its own
    absolute watermark, and the ratio keeps a small window safe. v9 may leave
    ``compaction_trigger_tokens`` null: ratio-only, as v4-v7.
    """

    ratio_tokens = int(
        hard_input_tokens * float(management["emergency_fallback_ratio"])
    )
    if management.get("compaction_trigger_tokens") is not None:
        return min(int(management["compaction_trigger_tokens"]), ratio_tokens)
    return ratio_tokens


def exchange_aging_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where v9 ages exchanges: its absolute watermark capped by the window ratio.

    The validator keeps both figures below their L6 counterparts, so on every
    window the lossless aging is reached before the summarizing compaction.
    """

    return min(
        int(management["exchange_aging_trigger_tokens"]),
        int(hard_input_tokens * float(management["exchange_aging_trigger_ratio"])),
    )


__all__ = ["compaction_trigger", "episode_checkpoint_trigger", "exchange_aging_trigger"]
