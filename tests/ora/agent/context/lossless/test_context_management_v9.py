"""Context management v9: v8 plus exchange aging, the hard backstop and reasoning replay.

v9 keeps every v8 field with its v8 meaning and adds the aging watermark (an
absolute count capped by a hard-window ratio, both below L6's), the exact
exchange tail, the aged-payload thresholds, the backstop's exact tail and the
reasoning-replay mode, and a camera-image mode of its own (``exchange_window_v1``:
camera images age with their exchange). The agent core mirrors the validator
(ora-pi/src/context/management.ts) and the watermark (thresholds.ts).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS,
    CONTEXT_MANAGEMENT_V8_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V9_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V9_SCHEMA,
    EXCHANGE_AGING_SCHEMAS,
    POINTER_FREE_SCHEMAS,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.lossless.thresholds import compaction_trigger, exchange_aging_trigger
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[5]
#: The v9 schemes' management objects (the schemes were retired on 2026-11-05; v9 itself stays).
DEFAULT = retired_scheme("context_rolling_tail_v3")["policy"]["context_management"]
UNAGED = retired_scheme("context_rolling_tail_v3_unaged_reasoning")["policy"]["context_management"]
RECENT = retired_scheme("context_rolling_tail_v3_recent_reasoning")["policy"]["context_management"]
CAMERA_TAIL = retired_scheme("context_rolling_tail_v3_camera_tail")["policy"]["context_management"]


def test_v9_is_v8_plus_the_exchange_fields_and_shows_pointers():
    assert CONTEXT_MANAGEMENT_V9_SCHEMA in APPEND_ONLY_SCHEMAS & EXCHANGE_AGING_SCHEMAS
    assert CONTEXT_MANAGEMENT_V9_SCHEMA not in POINTER_FREE_SCHEMAS
    assert set(CONTEXT_MANAGEMENT_V8_FIELD_SPECS) < set(CONTEXT_MANAGEMENT_V9_FIELD_SPECS)
    assert set(CONTEXT_MANAGEMENT_V9_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V8_FIELD_SPECS) == {
        "exchange_aging_trigger_tokens", "exchange_aging_trigger_ratio", "exact_tail_exchanges",
        "aged_argument_bytes", "aged_result_bytes", "aged_preview_bytes", "backstop_exact_tail_exchanges",
        "reasoning_replay", "reasoning_tail_exchanges",
    }


@pytest.mark.parametrize("value", [DEFAULT, UNAGED, RECENT, CAMERA_TAIL])
def test_the_shipped_schemes_validate_unchanged(value):
    assert validate_context_management(value) == value
    assert {key: value[key] for key in ("reasoning_replay", "reasoning_tail_exchanges")} in (
        {"reasoning_replay": "all", "reasoning_tail_exchanges": None},
        {"reasoning_replay": "unaged_exchanges", "reasoning_tail_exchanges": None},
        {"reasoning_replay": "recent_exchanges", "reasoning_tail_exchanges": 6},
    )


def test_the_default_replays_aged_reasoning_and_the_comparison_arms_differ_only_there():
    assert DEFAULT["reasoning_replay"] == "all"
    for arm in (UNAGED, RECENT):
        changed = {key for key in DEFAULT if DEFAULT[key] != arm[key]}
        assert changed <= {"reasoning_replay", "reasoning_tail_exchanges"} and "reasoning_replay" in changed


def test_the_default_ages_camera_images_with_their_exchange_and_the_camera_tail_arm_differs_only_there():
    assert (DEFAULT["camera_images"], DEFAULT["camera_tail_observations"]) == ("exchange_window_v1", None)
    assert {key for key in DEFAULT if DEFAULT[key] != CAMERA_TAIL[key]} == {"camera_images", "camera_tail_observations"}
    assert (CAMERA_TAIL["camera_images"], CAMERA_TAIL["camera_tail_observations"]) == ("rolling_tail_v1", 2)


@pytest.mark.parametrize("change, match", [
    ({"exchange_aging_trigger_tokens": 150000}, "below the L6 watermark"),
    ({"exchange_aging_trigger_ratio": 0.8}, "below the L6 watermark"),
    ({"backstop_exact_tail_exchanges": 7}, "must not exceed exact_tail_exchanges"),
    ({"aged_preview_bytes": 1024}, "smaller than the aged argument and result"),
    ({"reasoning_replay": "none"}, "reasoning_replay must be one of"),
    ({"reasoning_tail_exchanges": 6}, "must be null for all"),
    ({"reasoning_replay": "unaged_exchanges", "reasoning_tail_exchanges": 6}, "must be null for unaged_exchanges"),
    ({"reasoning_replay": "recent_exchanges"}, "must be a positive integer for recent_exchanges"),
    ({"reasoning_tail_exchanges": 0}, "positive integer or null"),
    ({"exact_tail_exchanges": 0}, "must be positive"),
    ({"camera_tail_observations": 2}, "must be null for exchange_window_v1"),
    ({"camera_images": "rolling_tail_v1", "camera_tail_observations": 1}, "at least 2"),
])
def test_refusals(change, match):
    with pytest.raises(ContextManagementConfigError, match=match):
        validate_context_management({**DEFAULT, **change})


def test_a_v8_object_is_not_v9():
    v8 = {key: DEFAULT[key] for key in CONTEXT_MANAGEMENT_V8_FIELD_SPECS}
    with pytest.raises(ContextManagementConfigError, match="missing"):
        validate_context_management({**v8, "schema": CONTEXT_MANAGEMENT_V9_SCHEMA})


def test_aging_precedes_l6_on_every_window():
    for window, output in ((262_144, 16_384), (131_072, 8_192), (1_048_576, 16_384), (32_768, 4_096)):
        hard = window - output - 2048
        assert exchange_aging_trigger(DEFAULT, hard) < compaction_trigger(DEFAULT, hard)
    # 243,712 usable on a 262k window: aging at 100,000, L6 at 150,000.
    assert (exchange_aging_trigger(DEFAULT, 243_712), compaction_trigger(DEFAULT, 243_712)) == (100_000, 150_000)
    # A 131,072 window with 8,192 output tokens: the ratio governs.
    assert exchange_aging_trigger(DEFAULT, 120_832) == 60_416



def test_a_ratio_only_l6_compares_the_ratios():
    """v9's L6 may be ratio-only (compaction_trigger_tokens null): the aging watermark then stays below it by ratio."""
    ratio_only = {**DEFAULT, "compaction_trigger_tokens": None, "emergency_fallback_ratio": 0.6,
                  "emergency_target_ratio": 0.5}
    assert validate_context_management(ratio_only) == ratio_only
    assert validate_context_management({**ratio_only, "exchange_aging_trigger_tokens": 900_000})
    with pytest.raises(ContextManagementConfigError, match="below the L6 watermark"):
        validate_context_management({**ratio_only, "exchange_aging_trigger_ratio": 0.6})
    with pytest.raises(ContextManagementConfigError, match="emergency target < emergency fallback"):
        validate_context_management({**ratio_only, "emergency_target_ratio": 0.65})
    with pytest.raises(ContextManagementConfigError, match="must be positive or null"):
        validate_context_management({**ratio_only, "compaction_trigger_tokens": 0})
    # About 618k on a 1M window, 146k on a 262k one; aging stays below on every window.
    for window, output, l6 in ((1_048_576, 16_384, 618_086), (262_144, 16_384, 146_227)):
        hard = window - output - 2048
        assert compaction_trigger(ratio_only, hard) == l6
        assert exchange_aging_trigger(ratio_only, hard) < l6
    # v8 keeps its absolute L6.
    v8 = {key: DEFAULT[key] for key in CONTEXT_MANAGEMENT_V8_FIELD_SPECS}
    with pytest.raises(ContextManagementConfigError, match="must be positive"):
        validate_context_management({**v8, "schema": "ora.native-context-management.v8", "compaction_trigger_tokens": None})
