"""Context v9 coverage: how far reasoning may be withheld, and the images an aged exchange takes out.

``all`` withholds reasoning only inside the aged span (the backstop moves both frontiers together);
``unaged_exchanges`` moves the reasoning frontier with the aged one in every rewrite, so the two are equal;
``recent_exchanges`` slides its own window and may withhold past the aged span. Under
``camera_images: exchange_window_v1`` the request also carries an aged exchange's camera images as references,
which the record counts beside the tool-returned ones.
"""
from pathlib import Path

import pytest

from ora.agent.context.lossless.thresholds import compaction_trigger, episode_checkpoint_trigger, exchange_aging_trigger
from tests.support.agent.retired_context_schemes import retired_scheme
from ora.orchestrator.evaluation.context_coverage import (
    APPEND_ONLY_MEDIA_POLICY, EXCHANGE_AGING_RECORD_SCHEMA, _append_only_violations, _exchange_aging_violations,
)

ROOT = Path(__file__).resolve().parents[4]
HARD = 243_712


def _management(scheme_id):
    """A v9 scheme's management: the v9 schemes were retired on 2026-11-05 and are test fixtures now."""
    return retired_scheme(scheme_id)["policy"]["context_management"]


def _projection(management, aged, withheld):
    window = management["camera_images"] == "exchange_window_v1"
    return {
        "hard_input_tokens": HARD,
        "resolved_exchange_aging_trigger_tokens": exchange_aging_trigger(management, HARD),
        "exchange_aging": {
            "schema": EXCHANGE_AGING_RECORD_SCHEMA, "exchanges": 12, "aged_exchanges": aged,
            "aged_observation_groups": 0, "reasoning_withheld_exchanges": withheld,
            "exact_tail_exchanges": management["exact_tail_exchanges"],
            "reasoning_replay": management["reasoning_replay"],
            "delivered": {"arguments": 0, "tool_results": 0, "tool_images": 0, "observations": 0, "reasoning": 0,
                          **({"camera_images": 0} if window else {})},
            "delivered_tool_images_externalized": 0,
            **({"delivered_camera_images_externalized": 0} if window else {}),
        },
        "request_estimate": {"tokens": 90_000, "hard_input_tokens": HARD},
        "hard_backstop": None,
    }


@pytest.mark.parametrize("scheme_id, withheld, valid", [
    ("context_rolling_tail_v3", 0, True),
    ("context_rolling_tail_v3", 6, True),
    ("context_rolling_tail_v3", 7, False),
    ("context_rolling_tail_v3_unaged_reasoning", 6, True),
    ("context_rolling_tail_v3_unaged_reasoning", 5, False),
    ("context_rolling_tail_v3_unaged_reasoning", 7, False),
    ("context_rolling_tail_v3_recent_reasoning", 8, True),
])
def test_withheld_reasoning_against_the_aged_span(scheme_id, withheld, valid):
    management = _management(scheme_id)
    reasons = _exchange_aging_violations(_projection(management, 6, withheld), [{}] * 4, management)
    assert reasons == ([] if valid else ["lossless_context_exchange_aging_invalid"])


def _images_projection(management, *, inline, tool_images, camera_images):
    """Four observation groups of three inline images each (two cameras and a receipt image)."""
    window = management["camera_images"] == "exchange_window_v1"
    record = {"delivered_tool_images_externalized": tool_images}
    if camera_images is not None:
        record["delivered_camera_images_externalized"] = camera_images
    return {
        "resolved_near_tail_action_steps": 3, "resolved_inline_image_target": None,
        "resolved_media_retention_policy": APPEND_ONLY_MEDIA_POLICY,
        "lifecycle_media_externalized_through_action_step": None,
        "pressure_externalized_through_action_step": None,
        "protected_inline_media_action_steps": [0, 1, 2, 3],
        "camera_delivery": {
            "schema": "ora.camera-delivery.v1", "camera_images": management["camera_images"],
            "tail_observations": management["camera_tail_observations"],
            "retrieval": "observe.ref" if window else "observe.step",
        },
        "exchange_aging": record,
        "active_inline_images": inline,
        "hard_input_tokens": HARD,
        "resolved_episode_checkpoint_trigger_tokens": episode_checkpoint_trigger(management, HARD),
        "resolved_compaction_trigger_tokens": compaction_trigger(management, HARD),
    }


@pytest.mark.parametrize("inline, tool_images, camera_images, valid", [
    (12, 0, 0, True),    # nothing aged: every image inline
    (6, 2, 4, True),     # two aged observations: their receipt images and camera images are references
    (8, 2, 4, False),    # the request still carries what the record calls references
    (10, 2, None, False),  # exchange_window_v1 must count its camera references
])
def test_aged_camera_images_leave_the_inline_count_under_the_exchange_window(inline, tool_images, camera_images, valid):
    management = _management("context_rolling_tail_v3")
    inventory = [{"action_step_index": index, "inline_images": 3} for index in range(4)]
    projection = _images_projection(management, inline=inline, tool_images=tool_images, camera_images=camera_images)
    reasons = _append_only_violations(projection, inventory, management)
    assert reasons == ([] if valid else ["lossless_context_image_target_invalid"] * len(reasons)) and (valid or reasons)


def test_aged_camera_images_are_a_reference_only_after_an_exchange_aged():
    management = _management("context_rolling_tail_v3")
    projection = _projection(management, 0, 0)
    assert _exchange_aging_violations(projection, [{}] * 4, management) == []
    projection["exchange_aging"]["delivered_camera_images_externalized"] = 2
    assert _exchange_aging_violations(projection, [{}] * 4, management) == ["lossless_context_exchange_aging_invalid"]
    # The camera-tail arm records neither field: its camera images never age.
    tail = _management("context_rolling_tail_v3_camera_tail")
    assert _exchange_aging_violations(_projection(tail, 6, 0), [{}] * 4, tail) == []
    assert "camera_images" not in _projection(tail, 6, 0)["exchange_aging"]["delivered"]
