"""Context management v8: append-only between rollout compactions.

v8 is a closed field set of its own, not v7 plus a field: the L1-L4 fields are
gone, ``camera_images`` selects how fixed camera images are delivered, and L6
compacts at an absolute ``compaction_trigger_tokens`` capped by the
hard-window ratio. The agent core mirrors the validator
(ora-pi/src/context/management.ts) and the watermark (thresholds.ts).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS,
    CONTEXT_MANAGEMENT_V7_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V8_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V8_SCHEMA,
    LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS,
    TAIL_AGING_SCHEMAS,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.lossless.thresholds import compaction_trigger, episode_checkpoint_trigger
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[5]
#: context_rolling_tail_v2's management (the scheme retired on 2026-11-05; v8 itself stays).
ROLLING = retired_scheme("context_rolling_tail_v2")["policy"]["context_management"]
#: The append-only mode: what the retired scheme context_append_until_compact_v1 (2026-10-31) configured, the rolling
#: tail's management with every camera image inline until the compaction.
INLINE = {**ROLLING, "camera_images": "inline_until_compaction_v1", "camera_tail_observations": None}
V6 = json.loads((ROOT / "tests/data/default-expectations.json").read_text())["native_policy"]["context_management"]


def test_v8_drops_the_aging_fields_and_adds_the_camera_and_compaction_fields():
    assert CONTEXT_MANAGEMENT_V8_SCHEMA in LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS
    assert CONTEXT_MANAGEMENT_V8_SCHEMA in APPEND_ONLY_SCHEMAS
    assert CONTEXT_MANAGEMENT_V8_SCHEMA not in TAIL_AGING_SCHEMAS
    gone = set(CONTEXT_MANAGEMENT_V7_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V8_FIELD_SPECS)
    assert gone == {
        "inline_media_tail_action_steps", "eager_large_result_bytes", "eager_preview_bytes",
        "eager_exact_deduplication", "trigger_inline_images", "trigger_estimated_tokens",
        "trigger_provider_input_ratio", "tail_aging_stride_action_steps",
    }
    assert set(CONTEXT_MANAGEMENT_V8_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V7_FIELD_SPECS) == {
        "camera_images", "camera_tail_observations", "compaction_trigger_tokens",
    }


@pytest.mark.parametrize("value", [ROLLING, INLINE])
def test_the_shipped_schemes_validate_unchanged(value):
    assert validate_context_management(value) == value


@pytest.mark.parametrize("change, match", [
    ({"camera_images": "drop_all"}, "camera_images must be one of"),
    ({"camera_images": "exchange_window_v1", "camera_tail_observations": None}, "needs exchange aging"),
    ({"camera_tail_observations": 1}, "at least 2"),
    ({"camera_tail_observations": None}, "at least 2"),
    ({"camera_tail_observations": 0}, "positive integer or null"),
    ({"camera_tail_observations": True}, "positive integer or null"),
    ({"causal_exact_tail_action_steps": 2}, "must equal 3"),
    ({"emergency_target_ratio": 0.8}, "emergency target < emergency fallback"),
    ({"checkpoint_target_estimated_tokens": 2000}, "soft maximum"),
    ({"compaction_trigger_tokens": 0}, "must be positive"),
    ({"inline_media_tail_action_steps": 2}, "extra"),
])
def test_rolling_tail_refusals(change, match):
    with pytest.raises(ContextManagementConfigError, match=match):
        validate_context_management({**ROLLING, **change})


def test_inline_mode_takes_no_window_and_v7_fields_are_not_v8():
    with pytest.raises(ContextManagementConfigError, match="must be null"):
        validate_context_management({**INLINE, "camera_tail_observations": 2})
    without = {key: value for key, value in INLINE.items() if key != "compaction_trigger_tokens"}
    with pytest.raises(ContextManagementConfigError, match="missing"):
        validate_context_management(without)
    with pytest.raises(ContextManagementConfigError, match="drifted"):
        validate_context_management({**V6, "schema": CONTEXT_MANAGEMENT_V8_SCHEMA})


def test_l6_compacts_at_the_absolute_watermark_capped_by_the_window():
    hard = 1_048_576 - 16_384 - 2_048
    assert compaction_trigger(ROLLING, hard) == 150_000
    assert compaction_trigger(ROLLING, 15_904) == int(15_904 * 0.8)
    assert episode_checkpoint_trigger(ROLLING, hard) == 150_000
    # v4-v7 keep the ratio alone.
    assert compaction_trigger(V6, hard) == int(hard * 0.8)
