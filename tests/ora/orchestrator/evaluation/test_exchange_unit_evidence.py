"""The evidence of an exact tail and an emergency checkpoint that count in exchanges.

The projection records the tail (``exact_tail``: its size, where it starts, the anchor exchange where the current
observation arrived, and each exchange's inline images and the ActionSteps it spans) and the checkpoint's exchange
coverage (``emergency_exchange_coverage``: the exchanges it absorbed, the one before them it kept exact). The
evaluation checks the tail against its own counts, the coverage against the tail, the near tail against the
ActionSteps the tail spans, the delivered images against the exchanges the request still holds, and whether the
checkpoint is in the request by its exchange coverage
(``ora-pi/test/exchange_unit.test.ts`` writes such projections).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.context.lossless.thresholds import compaction_trigger, episode_checkpoint_trigger
from ora.orchestrator.evaluation.context_coverage import (
    APPEND_ONLY_MEDIA_POLICY,
    _append_only_violations,
    _binding_counts_invalid,
    coverage_violations,
)
from ora.orchestrator.evaluation.lossless import _exchange_checkpoint_active, _exchange_tail_reasons
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[4]
HARD = 195_904


def _tail(total=23, count=6, anchor=0, images=None, steps=None):
    images = images or [1] + [0] * (total - 1)
    steps = steps or [[0, 1]] + [[1, 1]] * (total - 1)
    return {
        "unit": "exchange", "exchanges": count, "total_exchanges": total, "from_exchange": max(0, total - count),
        "anchor_exchange": anchor,
        "exchange_inventory": [{"exchange_index": index, "action_steps": steps[index], "inline_images": images[index]}
                               for index in range(total)],
    }


def _projection(tail, *, summarized=17, kept=0, sha="a" * 64, inline=1, episode_through=None):
    return {
        "exact_tail": tail,
        "emergency_exchange_coverage": {"unit": "exchange", "summarized_exchanges": summarized,
                                        "through_exchange_index": summarized - 1 if summarized else None,
                                        "kept_exchange": kept},
        "emergency_checkpoint_sha256": sha,
        "episode_checkpointed_through_action_step": episode_through,
        "retained_exact_causal_action_steps": 0,
        "action_step_inventory": [{"action_step_index": 0, "inline_images": 1, "complete": True},
                                  {"action_step_index": 1, "inline_images": 0}],
        "active_inline_images": inline,
    }


def test_a_well_formed_exchange_tail_and_coverage():
    projection = _projection(_tail())
    assert _exchange_tail_reasons(projection) == []
    assert _exchange_checkpoint_active(projection, 0)


@pytest.mark.parametrize("change", [
    {"from_exchange": 16},                      # not the latest six
    {"anchor_exchange": 23},                    # past the last exchange
    {"exchanges": 0},
])
def test_a_tail_that_disagrees_with_its_counts_is_refused(change):
    assert _exchange_tail_reasons(_projection({**_tail(), **change})) == ["lossless_context_exact_tail_invalid"]


@pytest.mark.parametrize("summarized, kept", [(18, 0), (17, 17), (17, -1)])
def test_a_checkpoint_covers_only_exchanges_before_the_tail_and_keeps_one_before_its_frontier(summarized, kept):
    assert _exchange_tail_reasons(_projection(_tail(), summarized=summarized, kept=kept)) == [
        "lossless_context_exact_tail_invalid"]


def test_the_exchange_images_must_be_the_action_step_images():
    tail = _tail(images=[2] + [0] * 22)
    assert _exchange_tail_reasons(_projection(tail)) == ["lossless_context_exact_tail_invalid"]


def test_a_checkpoint_superseded_by_a_capsule_is_not_in_the_request():
    projection = _projection(_tail(), sha=None)
    # Every absorbed exchange lies in ActionSteps a capsule through ActionStep 1 covers.
    assert not _exchange_checkpoint_active(projection, 2)
    assert _exchange_checkpoint_active(projection, 1)


def _coverage_projection(inline, near=1):
    management = retired_scheme("context_rolling_tail_v3")["policy"]["context_management"]
    projection = {
        **_projection(_tail(images=[1, 2, 0, 0, 3] + [0] * 18), inline=inline),
        "resolved_near_tail_action_steps": near, "resolved_inline_image_target": None,
        "resolved_media_retention_policy": APPEND_ONLY_MEDIA_POLICY,
        "lifecycle_media_externalized_through_action_step": None,
        "pressure_externalized_through_action_step": None,
        "protected_inline_media_action_steps": [0, 1],
        "camera_delivery": {"schema": "ora.camera-delivery.v1", "camera_images": "exchange_window_v1",
                            "tail_observations": None, "retrieval": "observe.ref"},
        "exchange_aging": {"delivered_tool_images_externalized": 0, "delivered_camera_images_externalized": 0},
        "hard_input_tokens": HARD,
        "resolved_episode_checkpoint_trigger_tokens": episode_checkpoint_trigger(management, HARD),
        "resolved_compaction_trigger_tokens": compaction_trigger(management, HARD),
    }
    inventory = [{"action_step_index": 0, "inline_images": 1, "complete": True},
                 {"action_step_index": 1, "inline_images": 5}]
    return projection, inventory, management


@pytest.mark.parametrize("inline, valid", [(1, True), (6, False), (3, False)])
def test_the_request_holds_the_images_of_the_exchanges_the_checkpoint_left(inline, valid):
    """Exchanges 1 (2 images) and 4 (3) are absorbed; exchange 0 (1 image), the anchor, stays exact."""
    projection, inventory, management = _coverage_projection(inline)
    reasons = _append_only_violations(projection, inventory, management)
    assert ("lossless_context_image_target_invalid" in reasons) != valid


@pytest.mark.parametrize("near, valid", [(1, True), (3, False), (0, False)])
def test_the_near_tail_is_the_action_steps_the_exchange_tail_spans(near, valid):
    """The anchor (exchange 0, ActionStep 0) is older than the latest six exchanges, so the tail spans
    ActionStep 0 on: the one completed ActionStep, whatever ``causal_exact_tail_action_steps`` says."""
    projection, inventory, management = _coverage_projection(1, near=near)
    reasons = _append_only_violations(projection, inventory, management)
    assert ("lossless_context_near_window_invalid" in reasons) != valid


@pytest.mark.parametrize(("arguments", "binding", "unbound", "rebound", "invalid"), [
    (None, "strict", 0, 0, False),
    (None, "strict", 1, 0, True),         # arguments never age: nothing is rewritten
    (1024, "strict", 3, 0, False),
    (1024, "strict", 0, 2, True),         # strict never rebinds
    (1024, "rebind_unsigned", 1, 2, False),
    (1024, "rebind_unsigned", None, 2, True),
])
def test_the_rewritten_messages_reasoning_counts_agree_with_the_management(arguments, binding, unbound, rebound, invalid):
    management = {"aged_argument_bytes": arguments, "reasoning_binding": binding}
    record = {"reasoning_unbound_messages": unbound, "reasoning_rebound_messages": rebound}
    assert _binding_counts_invalid({"exact_tail": _tail()}, record, management) is invalid
    # An ActionStep tail's record (v9) carries neither count.
    assert _binding_counts_invalid({}, {}, management) is False


#: The first request of a RoboDojo code-mode run under context_keep_written (plan bef6e142fd8f, 2026-11-04),
#: its projection manifest cut to the fields the evaluation reads.
FIRST_REQUEST = ROOT / "tests/data/context_evidence/keep-written-first-request.json"


def test_the_first_request_holds_the_opening_observations_images_before_any_exchange():
    """Before the first assistant message there is no exchange: the opening observation's 3 camera images belong
    to none yet (exchange 0 takes them when it starts) and the exact tail holds them. Every v10 run's first
    projection looks like this, and the evaluation refused it (exact tail and image target)."""
    projection = json.loads(FIRST_REQUEST.read_text(encoding="utf-8"))
    inventory = projection["action_step_inventory"]
    assert projection["exact_tail"]["total_exchanges"] == 0 and projection["exact_tail"]["exchange_inventory"] == []
    assert projection["active_inline_images"] == sum(item["inline_images"] for item in inventory) == 3
    assert _exchange_tail_reasons(projection) == []
    assert coverage_violations(projection, inventory) == []
    # The ActionStep inventory still accounts for every image: one fewer or one more is refused.
    for images in (2, 4):
        assert "lossless_context_image_target_invalid" in coverage_violations(
            {**projection, "active_inline_images": images}, inventory)
    # Once an exchange exists, its inventory must hold the ActionSteps' images.
    tail = {**projection["exact_tail"], "total_exchanges": 1, "anchor_exchange": 0,
            "exchange_inventory": [{"exchange_index": 0, "action_steps": [0, 0], "inline_images": 0}]}
    assert _exchange_tail_reasons({**projection, "exact_tail": tail}) == ["lossless_context_exact_tail_invalid"]
