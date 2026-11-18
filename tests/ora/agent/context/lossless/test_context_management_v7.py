"""Context management v7: the pressure estimate's per-image charge is a field.

v4-v6 charge a fixed 4096 tokens per inline image. Against DeepSeek-V4.1-Flash
usage on 4,186 recorded RoboDojo requests one camera image cost about 530 input
tokens, so the toolkit v2 baseline stated 530 (retired 2026-11-01; v8 keeps the field). Every other field keeps its v6
meaning. The agent core refuses v7 since K2c; v8's value becomes the scheme's ``estimator.image_tokens``, which the
core's one estimate reads (ora-pi/src/context/stages/services/estimator.ts). Python still validates v7 for its sealed
runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.context.lossless.management import (
    CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V7_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V7_SCHEMA,
    IMAGE_ESTIMATE_TOKENS_FIELD,
    LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS,
    TAIL_AGING_SCHEMAS,
    WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS,
    ContextManagementConfigError,
    validate_context_management,
)

ROOT = Path(__file__).resolve().parents[5]
V6 = json.loads((ROOT / "tests/data/default-expectations.json").read_text())["native_policy"]["context_management"]
V7 = {**V6, "schema": CONTEXT_MANAGEMENT_V7_SCHEMA, IMAGE_ESTIMATE_TOKENS_FIELD: 530}


def test_v7_is_v6_plus_the_image_charge():
    assert set(CONTEXT_MANAGEMENT_V7_FIELD_SPECS) == set(CONTEXT_MANAGEMENT_V6_FIELD_SPECS) | {IMAGE_ESTIMATE_TOKENS_FIELD}
    assert CONTEXT_MANAGEMENT_V7_FIELD_SPECS[IMAGE_ESTIMATE_TOKENS_FIELD]["kind"] == "positive_int"
    assert CONTEXT_MANAGEMENT_V7_SCHEMA in LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS
    assert CONTEXT_MANAGEMENT_V7_SCHEMA in TAIL_AGING_SCHEMAS
    # v7 keeps v6's absolute L5 watermark.
    assert CONTEXT_MANAGEMENT_V7_SCHEMA not in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS
    assert validate_context_management(V7) == V7


@pytest.mark.parametrize("value", [0, -1, 1.5, None, "530"])
def test_v7_refuses_a_non_positive_or_non_integer_charge(value):
    with pytest.raises(ContextManagementConfigError, match=IMAGE_ESTIMATE_TOKENS_FIELD):
        validate_context_management({**V7, IMAGE_ESTIMATE_TOKENS_FIELD: value})


def test_closed_schemas_do_not_mix():
    missing = {key: value for key, value in V7.items() if key != IMAGE_ESTIMATE_TOKENS_FIELD}
    with pytest.raises(ContextManagementConfigError, match="missing"):
        validate_context_management(missing)
    with pytest.raises(ContextManagementConfigError, match="extra"):
        validate_context_management({**V6, IMAGE_ESTIMATE_TOKENS_FIELD: 530})


