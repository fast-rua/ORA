"""Context management v6: the absolute L5 watermark (user decision 2026-10-04).

"所有的 L5 改成 15 万": every maintained configuration admits an L5 Episode
Capsule at 150,000 Provider-input tokens, whatever the bound window. v5's
max(150,000, 0.60 x usable) put it at 618,086 on a 1M DeepSeek window, where
it effectively never fired. L4 and L6 are unchanged. v4/v5 keep their recorded
window-scaled watermark so sealed runs and frozen archives read back exactly.

The agent core's Context pipeline (ora-pi/src/context/native.ts) resolves these
watermarks with the same formulas; its L5 admission and L6 emergency at a
bound window are held by the recorded fixtures in tests/data/context_differential
(l5_capsule, l6_target_miss, stride_v6_4_slots_reset). Here the formulas
(:mod:`ora.agent.context.lossless.thresholds`, and the L4 share and L5 formula
label in ``tests/support/agent/context_thresholds.py``), the schema and the evidence check.
"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_CATALOGS, TOOLKIT_BASELINE_IDS
from ora.agent.composition.context_scheme import load_context_scheme
from ora.agent.context.lossless import thresholds
from ora.agent.context.lossless.management import (
    CONTEXT_MANAGEMENT_V5_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V6_SCHEMA,
    CONTEXT_MANAGEMENT_V7_SCHEMA,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.orchestrator.evaluation.context_coverage import coverage_violations
from tests.support.agent import context_thresholds as support_thresholds

ROOT = Path(__file__).resolve().parents[5]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())
#: The maintained block, as fresh configurations receive it.
V6 = ACTIVATION["native_policy"]["context_management"]
#: The same figures under the v5 schema this change replaced.
V5 = {
    **V6, "schema": "ora.native-context-management.v5",
    "episode_checkpoint_trigger_ratio": 0.60,
}
ONE_MILLION = 1_048_576
#: DeepSeek's usable input with the default 16,384-token output budget.
ONE_MILLION_USABLE = ONE_MILLION - 16_384 - 2_048


def _usable(context_window: int, max_output_tokens: int = 16_384) -> int:
    """Usable Provider input: the window less the output budget and the 2,048-token reserve."""
    return context_window - max_output_tokens - 2_048


def _recorded_v5_projection():
    """The last projection manifest of a Context v5 session the Python pipeline recorded."""
    fixture = json.loads(gzip.decompress(
        (ROOT / "tests/data/context_differential/stride_v5_4_two_rounds_anthropic.json.gz").read_bytes()))
    projected = [op for op in fixture["ops"] if op["op"] == "project" and op.get("result")]
    return copy.deepcopy(projected[-1]["result"]["manifest"])


def test_every_maintained_lossless_configuration_selects_the_absolute_watermark():
    from ora.agent.composition.surface.toolkit_profiles import TOOLKIT_BASELINES

    catalogs = {}
    for baseline_id, path in BASELINE_CATALOGS.items():
        raw = json.loads((ROOT / path).read_text())
        # A baseline that names its context scheme takes its context fields from it.
        policy = ({**raw["policy"], **load_context_scheme(ROOT, raw["context_scheme"])["policy"]}
                  if "context_scheme" in raw else raw["policy"])
        if policy["context_retention"] == "rewrite_compact_v1":
            assert baseline_id in TOOLKIT_BASELINE_IDS
            assert "context_management" not in policy
        else:
            assert policy["context_retention"] == "action_step_lossless_v4"
            catalogs[baseline_id] = policy["context_management"]
    for owner, management in {"default-activation": V6, **catalogs}.items():
        if management["schema"] == "ora.native-context-management.v10":
            # Keep-written context (test_context_management_v10), RoboDojo's default since 2026-11-04: only a
            # toolkit baseline selects it, and its L5 is by ratio alone (0.15 of the hard limit, about v9's
            # 150,000 on a 1M window), never v4/v5's window-scaled max(tokens, ratio).
            assert owner in TOOLKIT_BASELINE_IDS, owner
            assert validate_context_management(management) == management, owner
            assert (management["episode_checkpoint_trigger_tokens"],
                    management["episode_checkpoint_trigger_ratio"]) == (None, 0.15), owner
            continue
        assert "episode_checkpoint_trigger_ratio" not in management, owner
        assert validate_context_management(management) == management, owner
        if management["schema"] == CONTEXT_MANAGEMENT_V7_SCHEMA:
            # The toolkit v2 baseline sizes its absolute L5 for a 200K window
            # (test_context_management_v7); only a toolkit baseline selects v7.
            assert owner in TOOLKIT_BASELINES.values(), owner
            continue
        if management["schema"] in {"ora.native-context-management.v8", "ora.native-context-management.v9"}:
            # Append-only context (test_context_management_v8, _v9): only a toolkit
            # baseline selects it; its L5 keeps the absolute 150,000.
            assert owner in TOOLKIT_BASELINE_IDS, owner
            assert management["episode_checkpoint_trigger_tokens"] == 150_000, owner
            continue
        assert management["schema"] == CONTEXT_MANAGEMENT_V6_SCHEMA, owner
        assert management["episode_checkpoint_trigger_tokens"] == 150_000, owner


def test_l5_is_150k_not_618k_on_a_1m_window():
    assert _usable(ONE_MILLION) == ONE_MILLION_USABLE == 1_030_144
    assert thresholds.episode_checkpoint_trigger(V6, ONE_MILLION_USABLE) == 150_000
    assert support_thresholds.episode_checkpoint_trigger_formula(V6) == "absolute_tokens"
    # The replaced v5 watermark on the same binding.
    assert thresholds.episode_checkpoint_trigger(V5, ONE_MILLION_USABLE) == 618_086


def test_l5_admits_a_closed_episode_at_150k_on_a_1m_window():
    trigger = thresholds.episode_checkpoint_trigger(V6, ONE_MILLION_USABLE)
    assert 149_999 < trigger <= 150_000
    assert thresholds.episode_checkpoint_trigger(V5, ONE_MILLION_USABLE) > 150_000


def test_l5_is_150k_on_a_200k_window_where_v5_agreed():
    assert _usable(200_000) == 181_568
    assert thresholds.episode_checkpoint_trigger(V6, 181_568) == 150_000
    assert thresholds.episode_checkpoint_trigger(V5, 181_568) == 150_000


@pytest.mark.parametrize("window", [200_000, 262_144, ONE_MILLION])
def test_l4_and_l6_are_unchanged(window):
    assert V6["emergency_fallback_ratio"] == V5["emergency_fallback_ratio"] == 0.80
    assert V6["emergency_target_ratio"] == V5["emergency_target_ratio"] == 0.65
    assert support_thresholds.estimate_pressure_trigger(V6, _usable(window)) == (
        support_thresholds.estimate_pressure_trigger(V5, _usable(window)))


def test_token_order_of_l4_and_l5_holds_only_below_about_620k():
    """Decided, not guaranteed: above ~620K usable L4's token trigger passes L5."""
    share = support_thresholds.ESTIMATE_PRESSURE_WINDOW_SHARE
    assert support_thresholds.estimate_pressure_trigger(V6, 618_000) < 150_000
    assert support_thresholds.estimate_pressure_trigger(V6, ONE_MILLION_USABLE) == int(
        ONE_MILLION_USABLE * share) == 249_792
    assert thresholds.episode_checkpoint_trigger(V6, ONE_MILLION_USABLE) == 150_000


def test_v6_is_v5_without_the_l5_ratio():
    assert set(CONTEXT_MANAGEMENT_V5_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V6_FIELD_SPECS) == {
        "episode_checkpoint_trigger_ratio"}
    assert set(CONTEXT_MANAGEMENT_V6_FIELD_SPECS) < set(CONTEXT_MANAGEMENT_V5_FIELD_SPECS)
    with pytest.raises(ContextManagementConfigError, match="extra=\\['episode_checkpoint_trigger_ratio'\\]"):
        validate_context_management({**V6, "episode_checkpoint_trigger_ratio": 0.6})
    stride_free = {key: value for key, value in V6.items()
                   if key != "tail_aging_stride_action_steps"}
    with pytest.raises(ContextManagementConfigError, match="missing=\\['tail_aging_stride_action_steps'\\]"):
        validate_context_management(stride_free)
    with pytest.raises(ContextManagementConfigError, match="L4 estimate floor"):
        validate_context_management({**V6, "episode_checkpoint_trigger_tokens": 48_000})
    with pytest.raises(ContextManagementConfigError, match="must be positive"):
        validate_context_management({**V6, "episode_checkpoint_trigger_tokens": 0})
    with pytest.raises(ContextManagementConfigError, match="emergency target"):
        validate_context_management({**V6, "emergency_target_ratio": 0.75})


def test_recorded_v5_evidence_keeps_its_window_scaled_watermark():
    """HEAD still revalidates v5 evidence exactly; only maintained configs moved."""
    assert validate_context_management(V5)["schema"].endswith(".v5")
    manifest = _recorded_v5_projection()
    assert manifest["context_management"]["schema"].endswith(".v5")
    assert manifest["resolved_episode_checkpoint_trigger_formula"] == (
        "max(absolute_tokens, int(hard_input_tokens * ratio))")
    inventory = manifest["action_step_inventory"]
    assert coverage_violations(manifest, inventory) == []
    # The same v5 evidence bound to a 1M window carries 618,086, and only that.
    wide = {**manifest, "context_management": V5, "hard_input_tokens": ONE_MILLION_USABLE,
            "resolved_episode_checkpoint_trigger_tokens": 618_086}
    assert "lossless_context_resolved_threshold_invalid" not in coverage_violations(wide, inventory)
    # Evidence cannot be read back under the other schema's watermark.
    retimed = {**wide, "resolved_episode_checkpoint_trigger_tokens": 150_000}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(retimed, inventory)
    relabelled = {**wide, "context_management": dict(V6)}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(relabelled, inventory)
