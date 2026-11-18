"""Context management v10: v9 that may keep what the model wrote, and the context experiment's three schemes.

v10 keeps every v9 field and lets a policy leave arguments unaged (``aged_argument_bytes`` null: no assistant
message is rewritten), put L5 by ratio (``episode_checkpoint_trigger_ratio``), keep an aged result's structured
fields (``aged_result_preview``), count L6 and the exact tail in exchanges (``compaction_unit``) and keep the
unsigned reasoning of a rewritten message (``reasoning_binding``). The agent core mirrors the validator
(ora-pi/src/context/management.ts) and runs it as v9's stages (stages/scheme.ts ``schemeV10``,
ora-pi/test/keep_written.test.ts).

The experiment of 2026-11-04 ran three arms at the model's own window, each a scheme file named by behaviour:
A ``context_keep_written`` (P and T used it too), B ``context_age_scripts`` (A plus scripts over 1,024 bytes aged,
their unsigned reasoning rebound) and H ``context_keep_written_unaged_reasoning`` (A plus reasoning withheld from
what an aging ages). H is the default since then; B was retired on 2026-11-05 and stays a test fixture
(``tests/data/context_scheme_fixtures``) of the script aging and reasoning binding the student schemes keep. Each
resolves, validates and differs from A only in its own fields.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ora.agent.composition.context_scheme import load_context_scheme
from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS,
    CONTEXT_MANAGEMENT_V9_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V10_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V10_SCHEMA,
    EXCHANGE_AGING_SCHEMAS,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.lossless.thresholds import compaction_trigger, episode_checkpoint_trigger, exchange_aging_trigger
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[5]
ARMS = {
    "A": "context_keep_written",
    "B": "context_age_scripts",
    "H": "context_keep_written_unaged_reasoning",
}
#: What each arm changes against A.
ARM_CHANGES = {
    "A": {},
    "B": {"aged_argument_bytes": 1024, "reasoning_binding": "rebind_unsigned"},
    "H": {"reasoning_replay": "unaged_exchanges"},
}


def _scheme(scheme_id):
    return retired_scheme(scheme_id) if scheme_id == ARMS["B"] else load_context_scheme(ROOT, scheme_id)


def _management(scheme_id):
    return _scheme(scheme_id)["policy"]["context_management"]


A = _management(ARMS["A"])


def test_v10_is_v9_plus_four_fields():
    assert CONTEXT_MANAGEMENT_V10_SCHEMA in APPEND_ONLY_SCHEMAS & EXCHANGE_AGING_SCHEMAS
    assert set(CONTEXT_MANAGEMENT_V10_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V9_FIELD_SPECS) == {
        "episode_checkpoint_trigger_ratio", "aged_result_preview", "compaction_unit", "reasoning_binding",
    }
    assert set(CONTEXT_MANAGEMENT_V9_FIELD_SPECS) < set(CONTEXT_MANAGEMENT_V10_FIELD_SPECS)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_experiment_scheme_resolves_validates_and_differs_from_a_only_in_its_own_fields(arm):
    scheme = _scheme(ARMS[arm])
    management = scheme["policy"]["context_management"]
    assert validate_context_management(management) == management
    assert management["schema"] == CONTEXT_MANAGEMENT_V10_SCHEMA
    assert {key: value for key, value in management.items() if A[key] != value} == ARM_CHANGES[arm]
    assert scheme["policy"]["context_tools"] == ["artifact_read"]
    if arm != "B":  # the fixture of a retired scheme carries no prompt
        assert scheme["prompt"].endswith("keep-written.md")
    stages = scheme["policy"]["context_stages"]
    assert stages["segments"] == {"unit": "exchange", "exact_tail": 6, "exact_tail_unit": "exchange"}


def test_the_base_keeps_what_the_model_wrote_and_ages_data_by_exchange():
    assert A["aged_argument_bytes"] is None
    assert (A["aged_result_bytes"], A["aged_result_preview"]) == (8192, "structured")
    assert (A["reasoning_replay"], A["reasoning_binding"]) == ("all", "strict")
    assert (A["exchange_aging_trigger_tokens"], A["exchange_aging_trigger_ratio"], A["exact_tail_exchanges"]) == (
        100_000, 0.5, 6)
    assert A["camera_images"] == "exchange_window_v1" and A["compaction_unit"] == "exchange"
    # L5 and L6 by ratio: on the 1M window (hard limit 1,030,144) L5 sits near v9's 150,000 and L6 at 0.6.
    hard = 1_030_144
    assert (A["episode_checkpoint_trigger_tokens"], A["compaction_trigger_tokens"]) == (None, None)
    assert episode_checkpoint_trigger(A, hard) == int(hard * 0.15) == 154_521
    assert compaction_trigger(A, hard) == int(hard * 0.6)
    assert exchange_aging_trigger(A, hard) == 100_000


@pytest.mark.parametrize(("change", "message"), [
    ({"episode_checkpoint_trigger_tokens": 150_000}, "exactly one of episode_checkpoint_trigger_tokens"),
    ({"episode_checkpoint_trigger_ratio": None}, "exactly one of episode_checkpoint_trigger_tokens"),
    ({"episode_checkpoint_trigger_ratio": 0.6}, "episode_checkpoint_trigger_ratio < emergency_fallback_ratio"),
    ({"aged_result_preview": "head"}, "aged_result_preview must be one of"),
    ({"compaction_unit": "minute"}, "compaction_unit must be one of"),
    ({"reasoning_binding": "loose"}, "reasoning_binding must be one of"),
    ({"aged_argument_bytes": 256}, "aged_preview_bytes must be smaller"),
    ({"exchange_aging_trigger_ratio": 0.6}, "below the L6 watermark"),
])
def test_v10_refuses_what_v9_and_its_own_fields_refuse(change, message):
    with pytest.raises(ContextManagementConfigError, match=message):
        validate_context_management({**A, **change})


def test_a_v9_object_cannot_carry_a_v10_field_and_v10_needs_all_of_its_own():
    v9 = {key: value for key, value in A.items() if key in CONTEXT_MANAGEMENT_V9_FIELD_SPECS}
    with pytest.raises(ContextManagementConfigError, match="fields drifted"):
        validate_context_management({**v9, "schema": "ora.native-context-management.v9", "compaction_unit": "exchange"})
    with pytest.raises(ContextManagementConfigError, match="fields drifted"):
        validate_context_management({key: value for key, value in A.items() if key != "reasoning_binding"})
