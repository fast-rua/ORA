"""Context stage lists: the parameter schemas, v2 schemes and the management bridge."""
import copy
from pathlib import Path

import pytest

from ora.agent.composition.context_scheme import load_context_scheme
from ora.agent.composition.schemas.context_stages import (
    SCHEME_SETTINGS, STAGE_LIST_SCHEMA, check_stage_list, management_equivalent, resolve_stage_list, stage_schemas,
)
from ora.agent.context.lossless.management import validate_context_management
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[5]
STAGES = (
    "camera_delivery", "eager_externalize", "emergency_checkpoint", "episode_capsule", "exchange_aging",
    "hard_backstop", "materialize", "media_window", "over_limit_guard", "pressure_externalize", "reasoning_replay",
    "stride_aging",
)
#: What the two v2 schemes' v1 files held (git show bfcb3b6d9:ora/policy/context_schemes/...).
V6 = {
    "schema": "ora.native-context-management.v6", "causal_exact_tail_action_steps": 3,
    "inline_media_tail_action_steps": 2, "eager_large_result_bytes": 50000, "eager_preview_bytes": 2048,
    "eager_exact_deduplication": True, "trigger_inline_images": 24, "trigger_estimated_tokens": 48000,
    "trigger_provider_input_ratio": 0.7, "episode_checkpoint_trigger_tokens": 150000,
    "emergency_fallback_ratio": 0.8, "emergency_target_ratio": 0.65, "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600, "tail_aging_stride_action_steps": 4,
}
#: ``context_lossless_v6`` since 2026-11-14: v6 with its optional fields, the hard backstop that ages by itself and
#: truncated replies sent without their reasoning.
V6_BACKSTOP = {**V6, "backstop_exact_tail_exchanges": 2, "truncated_reply_reasoning": "withheld"}
V8 = {
    "schema": "ora.native-context-management.v8", "causal_exact_tail_action_steps": 3,
    "camera_images": "rolling_tail_v1", "camera_tail_observations": 2, "episode_checkpoint_trigger_tokens": 150000,
    "compaction_trigger_tokens": 150000, "emergency_fallback_ratio": 0.8, "emergency_target_ratio": 0.65,
    "checkpoint_target_estimated_tokens": 1200, "checkpoint_max_estimated_tokens": 1600, "image_estimate_tokens": 530,
}
#: What the reference line's v1 files held (git show 1a9c1b117:ora/policy/context_schemes/rolling-tail-v3*.json); the
#: v8 and v9 schemes were retired on 2026-11-05 and their stage lists are test fixtures now.
V9 = {
    **V8, "schema": "ora.native-context-management.v9", "camera_images": "exchange_window_v1",
    "camera_tail_observations": None, "exchange_aging_trigger_tokens": 100000, "exchange_aging_trigger_ratio": 0.5,
    "exact_tail_exchanges": 6, "aged_argument_bytes": 1024, "aged_result_bytes": 2048, "aged_preview_bytes": 256,
    "backstop_exact_tail_exchanges": 2, "reasoning_replay": "all", "reasoning_tail_exchanges": None,
}
V9_SCHEMES = {
    "context_rolling_tail_v3": V9,
    "context_rolling_tail_v3_camera_tail": {**V9, "camera_images": "rolling_tail_v1", "camera_tail_observations": 2},
    "context_rolling_tail_v3_recent_reasoning": {
        **V9, "reasoning_replay": "recent_exchanges", "reasoning_tail_exchanges": 6},
    "context_rolling_tail_v3_unaged_reasoning": {**V9, "reasoning_replay": "unaged_exchanges"},
}


def test_every_stage_has_one_parameter_schema_and_the_settings_have_theirs():
    schemas = stage_schemas(ROOT)
    assert sorted(schemas) == sorted((*STAGES, SCHEME_SETTINGS))
    for schema in schemas.values():
        assert schema["type"] == "object" and schema["additionalProperties"] is False


def _scheme_policy(scheme_id):
    return (load_context_scheme(ROOT, scheme_id) if scheme_id == "context_lossless_v6" else retired_scheme(scheme_id))[
        "policy"]


@pytest.mark.parametrize(("scheme_id", "management"), [
    ("context_lossless_v6", V6_BACKSTOP), ("context_rolling_tail_v2", V8),
])
def test_a_v2_scheme_resolves_to_its_stage_list_and_the_management_object_of_its_v1_file(scheme_id, management):
    """``context_rolling_tail_v2`` as its v1 file held it; ``context_lossless_v6`` as its v1 file plus the optional v6
    fields (2026-11-14)."""
    policy = _scheme_policy(scheme_id)
    assert policy["context_management"] == management
    assert list(policy["context_management"]) == list(management), "the evidence keeps its field order"
    assert validate_context_management(policy["context_management"]) == management
    stages = policy["context_stages"]
    assert stages["schema"] == STAGE_LIST_SCHEMA
    assert check_stage_list(stages, management) == stages
    params = {item["stage"]: item["params"] for item in stages["stages"]}
    # The schemas' defaults are filled in: the core receives every parameter.
    assert params["episode_capsule"]["retry_attempts"] == 4 and params["episode_capsule"]["enabled"] is True
    if scheme_id == "context_lossless_v6":
        assert params["materialize"] == {"reasoning": "exclude", "truncated_reply_reasoning": "withheld"}
        # It ages at L6's watermark (emergency_fallback_ratio of the hard limit), by the request's own estimate.
        assert params["hard_backstop"] == {"exact_tail_exchanges": 2, "aging": "own", "trigger_ratio": 0.8}
        assert params["hard_backstop"]["trigger_ratio"] == params["emergency_checkpoint"]["trigger_ratio"]
        assert "over_limit_guard" not in params
    else:
        assert params["materialize"] == {"reasoning": "exclude"}


def test_the_v6_stage_list_before_the_backstop_is_still_the_sealed_v6_object():
    """Plans resolved before 2026-11-14 carry the reactive guard and no materialize switch: the v6 object without the
    optional fields, so they keep validating and running as they ran."""
    stage_list = _v6_list()
    stage_list["stages"] = [
        {"stage": "over_limit_guard", "params": {"over_limit": "refuse"}} if item["stage"] == "hard_backstop"
        else {"stage": "materialize", "params": {"reasoning": "exclude"}} if item["stage"] == "materialize"
        else item for item in stage_list["stages"]
    ]
    assert management_equivalent(resolve_stage_list(ROOT, stage_list, label="test")) == V6
    assert validate_context_management(V6) == V6
    # Each optional field on its own.
    only_backstop = _v6_list()
    _stage(only_backstop, "materialize").pop("truncated_reply_reasoning")
    assert management_equivalent(only_backstop) == {**V6, "backstop_exact_tail_exchanges": 2}


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: _stage(s, "hard_backstop").pop("aging"), "does not age by itself"),
    (lambda s: _stage(s, "hard_backstop").update(trigger_ratio=0.7), "whose watermark is not L6's"),
    (lambda s: _stage(s, "hard_backstop").pop("trigger_ratio"), "whose watermark is not L6's"),
    (lambda s: _stage(s, "hard_backstop").update(exact_tail_exchanges=0), "hard_backstop"),
    (lambda s: _stage(s, "materialize").update(truncated_reply_reasoning="kept"), "materialize"),
])
def test_a_v6_backstop_stage_list_must_age_by_itself(change, message):
    stage_list = _v6_list()
    change(stage_list)
    with pytest.raises(ValueError, match=message):
        management_equivalent(resolve_stage_list(ROOT, stage_list, label="test"))


def test_the_optional_v6_fields_have_no_equivalent_beside_the_exchange_stages():
    stage_list = _v10_list()
    _stage(stage_list, "materialize")["truncated_reply_reasoning"] = "withheld"
    with pytest.raises(ValueError, match="truncated reply's reasoning withheld outside v6"):
        management_equivalent(stage_list)
    for extra in ({"aging": "own"}, {"trigger_ratio": 0.6}):
        stage_list = _v10_list()
        _stage(stage_list, "hard_backstop").update(extra)
        with pytest.raises(ValueError, match="ages by itself beside the exchange stages"):
            management_equivalent(stage_list)
    with pytest.raises(ValueError, match="truncated_reply_reasoning must be 'withheld'"):
        validate_context_management({**V6, "truncated_reply_reasoning": "kept"})
    with pytest.raises(ValueError, match="extra="):
        validate_context_management({**V8, "backstop_exact_tail_exchanges": 2})


@pytest.mark.parametrize("scheme_id", sorted(V9_SCHEMES))
def test_a_v9_scheme_resolves_to_its_stage_list_and_the_reference_management_object(scheme_id):
    management = V9_SCHEMES[scheme_id]
    policy = retired_scheme(scheme_id)["policy"]
    assert policy["context_management"] == management
    assert list(policy["context_management"]) == list(management), "the evidence keeps its field order"
    assert validate_context_management(policy["context_management"]) == management
    stages = policy["context_stages"]
    assert check_stage_list(stages, management) == stages
    assert stages["segments"] == {"unit": "exchange", "exact_tail": 3}
    params = {item["stage"]: item["params"] for item in stages["stages"]}
    assert params["exchange_aging"]["preview"] == {"head_bytes": 128, "tail_bytes": 128}
    assert params["reasoning_replay"]["tail_exchanges"] == management["reasoning_tail_exchanges"]


def test_a_v9_stage_list_needs_v9_references_and_an_even_preview_split():
    base = copy.deepcopy(retired_scheme("context_rolling_tail_v3")["policy"]["context_stages"])
    stage_list = copy.deepcopy(base)
    _stage(stage_list, "exchange_aging")["preview"] = {"head_bytes": 200, "tail_bytes": 56}
    with pytest.raises(ValueError, match="an aged preview split other than half and half"):
        management_equivalent(stage_list)
    stage_list = copy.deepcopy(base)
    stage_list["references"]["advertise_over_bytes"] = 50000
    with pytest.raises(ValueError, match="v9 references"):
        management_equivalent(stage_list)
    stage_list = copy.deepcopy(base)
    stage_list["segments"]["unit"] = "action_step"
    with pytest.raises(ValueError, match="another unit than exchange"):
        management_equivalent(stage_list)
    stage_list = copy.deepcopy(base)
    _stage(stage_list, "reasoning_replay")["tail_exchanges"] = 6
    with pytest.raises(ValueError, match="reasoning_replay"):
        resolve_stage_list(ROOT, stage_list, label="test")
    stage_list = copy.deepcopy(base)
    _stage(stage_list, "camera_delivery")["tail_observations"] = 2
    with pytest.raises(ValueError, match="camera_delivery"):
        resolve_stage_list(ROOT, stage_list, label="test")


def _v6_list():
    return copy.deepcopy(load_context_scheme(ROOT, "context_lossless_v6")["policy"]["context_stages"])


def _stage(stage_list, stage):
    return next(item for item in stage_list["stages"] if item["stage"] == stage)["params"]


def test_the_schemas_refuse_a_parameter_out_of_range_or_unknown():
    stage_list = _v6_list()
    _stage(stage_list, "stride_aging")["stride"] = 0
    with pytest.raises(ValueError, match="stride_aging"):
        resolve_stage_list(ROOT, stage_list, label="test")
    stage_list = _v6_list()
    _stage(stage_list, "media_window")["window"] = 2
    with pytest.raises(ValueError, match="media_window"):
        resolve_stage_list(ROOT, stage_list, label="test")
    stage_list = _v6_list()
    stage_list["references"]["pointers"] = False
    with pytest.raises(ValueError, match="settings"):
        resolve_stage_list(ROOT, stage_list, label="test")
    stage_list = _v6_list()
    stage_list["stages"].append({"stage": "action_digest", "params": {}})
    with pytest.raises(ValueError, match="unknown stage 'action_digest'"):
        resolve_stage_list(ROOT, stage_list, label="test")
    stage_list = _v6_list()
    stage_list["stages"].append(copy.deepcopy(stage_list["stages"][1]))
    with pytest.raises(ValueError, match="listed twice"):
        resolve_stage_list(ROOT, stage_list, label="test")


def test_a_stage_list_without_a_management_equivalent_does_not_resolve():
    stage_list = _v6_list()
    _stage(stage_list, "stride_aging").update(stride=1, strided=False)
    with pytest.raises(ValueError, match="unstrided aging has no management equivalent"):
        management_equivalent(resolve_stage_list(ROOT, stage_list, label="test"))
    stage_list = _v6_list()
    _stage(stage_list, "episode_capsule")["retry_attempts"] = 5
    with pytest.raises(ValueError, match="another L5 retry schedule"):
        management_equivalent(stage_list)
    stage_list = _v6_list()
    stage_list["stages"] = [item for item in stage_list["stages"] if item["stage"] != "episode_capsule"]
    with pytest.raises(ValueError, match="not the stages of a management version"):
        management_equivalent(stage_list)


def test_a_plan_stage_list_must_be_its_management_objects():
    stage_list = _v6_list()
    with pytest.raises(ValueError, match="is not the stage list of policy.context_management"):
        check_stage_list(stage_list, {**V6_BACKSTOP, "trigger_inline_images": 23})
    with pytest.raises(ValueError, match="is not the stage list of policy.context_management"):
        check_stage_list(stage_list, V6)


def test_an_exchange_unit_tail_or_checkpoint_has_no_management_equivalent_before_v10():
    """``segments.exact_tail_unit`` and ``emergency_checkpoint.unit`` validate; no version before v10 carries them."""
    base = copy.deepcopy(retired_scheme("context_rolling_tail_v3")["policy"]["context_stages"])
    stage_list = copy.deepcopy(base)
    stage_list["segments"].update(exact_tail=6, exact_tail_unit="exchange")
    _stage(stage_list, "emergency_checkpoint")["unit"] = "exchange"
    resolved = resolve_stage_list(ROOT, stage_list, label="test")
    with pytest.raises(ValueError, match="an exact tail or an emergency checkpoint in exchanges"):
        management_equivalent(resolved)
    _stage(stage_list, "emergency_checkpoint")["unit"] = "minute"
    with pytest.raises(ValueError, match="emergency_checkpoint"):
        resolve_stage_list(ROOT, stage_list, label="test")


V10_SCHEMES = ("context_keep_written", "context_age_scripts", "context_keep_written_unaged_reasoning")


@pytest.mark.parametrize("scheme_id", V10_SCHEMES)
def test_a_v10_scheme_resolves_to_its_stage_list_and_its_management_object(scheme_id):
    """The two v10 schemes, and context_age_scripts (arm B, retired 2026-11-05) as the fixture of script aging and
    the reasoning binding the student schemes keep."""
    policy = (retired_scheme(scheme_id) if scheme_id == "context_age_scripts" else load_context_scheme(ROOT, scheme_id))[
        "policy"]
    management = policy["context_management"]
    assert validate_context_management(management) == management
    assert management["schema"] == "ora.native-context-management.v10"
    stages = policy["context_stages"]
    assert check_stage_list(stages, management) == stages
    params = {item["stage"]: item["params"] for item in stages["stages"]}
    # The exact tail counts the aging's exchanges; the ActionStep tail keeps the value every version fixes.
    assert stages["segments"]["exact_tail"] == params["exchange_aging"]["exact_tail_exchanges"] == 6
    assert management["causal_exact_tail_action_steps"] == 3
    assert params["exchange_aging"]["keep_anchor"] is True and params["emergency_checkpoint"]["unit"] == "exchange"
    assert params["reasoning_replay"]["binding"] == management["reasoning_binding"]


def _v10_list():
    return copy.deepcopy(load_context_scheme(ROOT, "context_keep_written")["policy"]["context_stages"])


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: _stage(s, "exchange_aging").pop("result_preview"), "in exchanges before v10"),
    (lambda s: _stage(s, "exchange_aging").update(keep_anchor=False), "keeps the anchor exchange"),
    (lambda s: _stage(s, "emergency_checkpoint").update(unit="action_step"), "in different units"),
    (lambda s: s["segments"].update(exact_tail=5), "other than the aging's"),
    (lambda s: _stage(s, "reasoning_replay").pop("binding"), "do not state the reasoning binding"),
    (lambda s: _stage(s, "episode_capsule").update(trigger_tokens=150000), "by both tokens and ratio"),
])
def test_a_v10_stage_list_must_be_one_v10_object(change, message):
    stage_list = _v10_list()
    change(stage_list)
    with pytest.raises(ValueError, match=message):
        management_equivalent(stage_list)


def test_a_v9_stage_list_carries_no_v10_parameter():
    base = copy.deepcopy(retired_scheme("context_rolling_tail_v3")["policy"]["context_stages"])
    for change in (lambda s: _stage(s, "exchange_aging").update(argument_bytes=None),
                   lambda s: _stage(s, "reasoning_replay").update(binding="strict"),
                   lambda s: _stage(s, "exchange_aging").update(keep_anchor=True)):
        stage_list = copy.deepcopy(base)
        change(stage_list)
        with pytest.raises(ValueError, match="v9 stages with arguments that never age"):
            management_equivalent(stage_list)
