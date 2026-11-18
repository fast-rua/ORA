"""Source-version-aware validation of lossless Context coverage evidence."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS, CAMERA_EXCHANGE_WINDOW, CAMERA_ROLLING_TAIL, EXCHANGE_AGING_SCHEMAS, REASONING_REPLAY_ALL,
    REASONING_REPLAY_UNAGED,
)
from ora.agent.context.lossless.thresholds import (
    compaction_trigger, episode_checkpoint_trigger, exchange_aging_trigger,
)

#: ``resolved_media_retention_policy`` of an append-only (v8, v9) projection.
APPEND_ONLY_MEDIA_POLICY = "append_only_until_compaction_v1"
#: ``exchange_aging.schema`` of a v9 projection.
EXCHANGE_AGING_RECORD_SCHEMA = "ora.context-exchange-aging.v1"
#: ``exchange_aging.delivered`` of a v9 projection; ``camera_images`` only under ``exchange_window_v1``.
EXCHANGE_DELIVERED_FIELDS = frozenset({"arguments", "tool_results", "tool_images", "observations", "reasoning"})


def coverage_violations(
    projection: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    completed = sum(item.get("complete") is True for item in inventory)
    management = projection.get("context_management", {})
    if not isinstance(management, Mapping):
        return ["lossless_context_lifecycle_coverage_invalid"]
    if management.get("schema") in APPEND_ONLY_SCHEMAS:
        reasons = _append_only_violations(projection, inventory, management)
        if management.get("schema") in EXCHANGE_AGING_SCHEMAS:
            reasons.extend(_exchange_aging_violations(projection, inventory, management))
        return reasons
    lifecycle = projection.get("lifecycle_media_externalized_through_action_step")
    tail = management.get("inline_media_tail_action_steps")
    if "resolved_near_tail_action_steps" not in projection:
        # Frozen source evidence retains its original fixed-tail semantics.
        current_round = inventory[-1].get("session_round")
        current_completed = sum(item.get("complete") is True and item.get(
            "session_round"
        ) == current_round for item in inventory)
        count = completed - min(tail, current_completed) if type(tail) is int else 0
        if lifecycle != (count - 1 if count > 0 else None):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
        return reasons
    causal = management.get("causal_exact_tail_action_steps")
    target = management.get("trigger_inline_images")
    if type(tail) is not int or type(causal) is not int or type(target) is not int:
        return ["lossless_context_near_window_invalid"]
    near = max(tail, causal)
    eligible = max(0, completed - near)
    media_policy = projection.get("resolved_media_retention_policy")
    if projection.get("resolved_near_tail_action_steps") != near or projection.get(
        "resolved_inline_image_target"
    ) != target:
        reasons.append("lossless_context_near_window_invalid")
    if media_policy == "current_round_action_step_tail_v1":
        current_round = inventory[-1].get("session_round")
        current_completed = sum(
            item.get("complete") is True
            and item.get("session_round") == current_round
            for item in inventory
        )
        media_coverage = completed - min(tail, current_completed)
        expected_lifecycle = media_coverage - 1 if media_coverage else None
        if lifecycle != expected_lifecycle or projection.get(
            "protected_inline_media_action_steps"
        ) != [
            item["action_step_index"] for item in inventory
            if item["action_step_index"] >= media_coverage
        ]:
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    elif media_policy == "current_round_action_step_tail_stride_v1":
        if not _stride_media_coverage_admitted(management, lifecycle, inventory) or (
            projection.get("protected_inline_media_action_steps") != [
                item["action_step_index"] for item in inventory
                if item["action_step_index"] >= (
                    lifecycle + 1 if type(lifecycle) is int else 0
                )
            ]
        ):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    elif media_policy is None:
        # Target-based v4 evidence emitted before fixed Episode media tails
        # were restored. Keep its exact source semantics revalidatable.
        if projection.get("protected_inline_media_action_steps") != [
            item["action_step_index"] for item in inventory
            if item["action_step_index"] >= eligible
        ]:
            reasons.append("lossless_context_near_window_invalid")
        if lifecycle is not None and (
            type(lifecycle) is not int or not 0 <= lifecycle < eligible
        ):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    else:
        reasons.append("lossless_context_lifecycle_coverage_invalid")
    pressure = projection.get("pressure_externalized_through_action_step")
    if pressure is not None and (
        type(pressure) is not int or not 0 <= pressure < eligible
    ):
        reasons.append("lossless_context_lifecycle_coverage_invalid")
    frontier = max((projection.get(field) if type(projection.get(field)) is int else -1)
                   for field in ("lifecycle_media_externalized_through_action_step",
                                 "episode_checkpointed_through_action_step",
                                 "emergency_summarized_through_action_step"))
    inline = sum(item.get("inline_images", 0) for item in inventory
                 if item["action_step_index"] > frontier)
    if "backstop_exact_tail_exchanges" in management:
        # The backstop's tool-returned images left the request as references inside the inline ActionSteps.
        reasons.extend(_own_backstop_violations(projection, management))
        record = projection.get("hard_backstop_aging")
        externalized = record.get("delivered_tool_images_externalized") if isinstance(record, Mapping) else None
        inline -= externalized if _count(externalized) else 0
    target_violation = (
        media_policy is None and inline > target and frontier + 1 < eligible
    )
    if projection.get("active_inline_images") != inline or target_violation:
        reasons.append("lossless_context_image_target_invalid")
    hard = projection.get("hard_input_tokens")
    if type(hard) is not int or projection.get(
        "resolved_episode_checkpoint_trigger_tokens"
    ) != episode_checkpoint_trigger(management, hard):
        reasons.append("lossless_context_resolved_threshold_invalid")
    return reasons


def _stride_media_coverage_admitted(
    management: Mapping[str, Any], lifecycle: Any,
    inventory: Sequence[Mapping[str, Any]],
) -> bool:
    """Context v5 media may lag its minimum tail by less than one stride.

    The lag only ever keeps current-Episode media inline: a reset releases
    every prior-Episode image in the same projection, as v4 does.
    """

    stride = management.get("tail_aging_stride_action_steps")
    tail = management.get("inline_media_tail_action_steps")
    if type(stride) is not int or stride < 1 or type(tail) is not int:
        return False
    if lifecycle is not None and (type(lifecycle) is not int or lifecycle < 0):
        return False
    completed = sum(item.get("complete") is True for item in inventory)
    current_round = inventory[-1].get("session_round")
    current_completed = sum(
        item.get("complete") is True and item.get("session_round") == current_round
        for item in inventory
    )
    covered = lifecycle + 1 if lifecycle is not None else 0
    lag = completed - min(tail, current_completed) - covered
    return 0 <= lag < stride and (
        covered >= completed
        or inventory[covered].get("session_round") == current_round
    )


def _append_only_violations(
    projection: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
    management: Mapping[str, Any],
) -> list[str]:
    """Context v8: nothing ages, and the delivered images match the camera mode.

    The inventory counts every image of the history. Outside the compacted
    span, the rolling tail mode delivers each observation's camera images only
    in the tail, so the request holds the inventory's images minus those, plus
    the tail's. v9 also takes out the images of aged exchanges the request
    carries as references: tool-returned ones, and under ``exchange_window_v1``
    the observations' camera images.
    """

    reasons: list[str] = []
    completed = sum(item.get("complete") is True for item in inventory)
    exchange_near = _exchange_near_tail(projection, completed)
    if (
        projection.get("resolved_near_tail_action_steps")
        != (management.get("causal_exact_tail_action_steps") if exchange_near is None else exchange_near)
        or projection.get("resolved_inline_image_target") is not None
        or projection.get("resolved_media_retention_policy") != APPEND_ONLY_MEDIA_POLICY
    ):
        reasons.append("lossless_context_near_window_invalid")
    if (
        projection.get("lifecycle_media_externalized_through_action_step") is not None
        or projection.get("pressure_externalized_through_action_step") is not None
        or projection.get("protected_inline_media_action_steps")
        != [item["action_step_index"] for item in inventory]
    ):
        reasons.append("lossless_context_lifecycle_coverage_invalid")
    frontier = max(
        (projection.get(field) if type(projection.get(field)) is int else -1)
        for field in ("episode_checkpointed_through_action_step",
                      "emergency_summarized_through_action_step")
    )
    expected = sum(
        item.get("inline_images", 0) for item in inventory
        if item["action_step_index"] > frontier
    )
    shown_by_exchange = _exchange_unit_images(projection)
    if shown_by_exchange is not None:
        # An emergency checkpoint that counts in exchanges removes whole exchanges, not ActionSteps.
        expected = shown_by_exchange
    camera = projection.get("camera_delivery")
    if (
        not isinstance(camera, Mapping)
        or camera.get("schema") != "ora.camera-delivery.v1"
        or camera.get("camera_images") != management.get("camera_images")
        or camera.get("tail_observations") != management.get("camera_tail_observations")
    ):
        return [*reasons, "lossless_context_camera_delivery_invalid"]
    if management.get("camera_images") == CAMERA_ROLLING_TAIL:
        tail = camera.get("tail")
        references = camera.get("history_references")
        window = management.get("camera_tail_observations")
        if not isinstance(tail, list) or not isinstance(references, list) or type(window) is not int:
            return [*reasons, "lossless_context_camera_delivery_invalid"]
        counts = {
            item.get("action_step_index"): item.get("camera_images")
            for item in references if isinstance(item, Mapping)
        }
        tail_steps = [item.get("action_step_index") for item in tail if isinstance(item, Mapping)]
        tail_images = sum(
            counts.get(item.get("action_step_index"), 0) for item in tail if isinstance(item, Mapping)
        )
        if (
            tail_steps != [item["action_step_index"] for item in inventory][-window:]
            or camera.get("tail_images") != tail_images
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            reasons.append("lossless_context_camera_delivery_invalid")
        expected += tail_images - sum(
            count for step, count in counts.items()
            if type(step) is int and step > frontier and type(count) is int
        )
    if management.get("schema") in EXCHANGE_AGING_SCHEMAS:
        # v9: the images of aged exchanges are references, not inline images.
        record = projection.get("exchange_aging")
        fields = ["delivered_tool_images_externalized"]
        if management.get("camera_images") == CAMERA_EXCHANGE_WINDOW:
            fields.append("delivered_camera_images_externalized")
        for field in fields:
            aged_images = record.get(field) if isinstance(record, Mapping) else None
            if not _count(aged_images) or aged_images > expected:
                reasons.append("lossless_context_image_target_invalid")
                break
            expected -= aged_images
    if projection.get("active_inline_images") != expected:
        reasons.append("lossless_context_image_target_invalid")
    hard = projection.get("hard_input_tokens")
    if type(hard) is not int or projection.get(
        "resolved_episode_checkpoint_trigger_tokens"
    ) != episode_checkpoint_trigger(management, hard) or projection.get(
        "resolved_compaction_trigger_tokens"
    ) != compaction_trigger(management, hard):
        reasons.append("lossless_context_resolved_threshold_invalid")
    return reasons


def _exchange_near_tail(projection: Mapping[str, Any], completed: int) -> int | None:
    """With an exact tail in exchanges, the completed ActionSteps it spans: from the one holding its first
    message (the anchor exchange's when that is older than the latest exchanges) on. None for an ActionStep
    tail; -1, which no count equals, for a tail record that does not say."""

    tail = projection.get("exact_tail")
    if not isinstance(tail, Mapping) or tail.get("unit") != "exchange":
        return None
    inventory, start, anchor = tail.get("exchange_inventory"), tail.get("from_exchange"), tail.get("anchor_exchange")
    if not isinstance(inventory, list) or type(start) is not int:
        return -1
    if not inventory:
        return completed
    index = anchor if type(anchor) is int and anchor < start else start
    item = inventory[index] if 0 <= index < len(inventory) else None
    steps = item.get("action_steps") if isinstance(item, Mapping) else None
    if not isinstance(steps, list) or not steps or type(steps[0]) is not int:
        return -1
    return completed - min(steps[0], completed)


def _exchange_unit_images(projection: Mapping[str, Any]) -> int | None:
    """With an exact tail in exchanges (``exact_tail.unit: exchange``), the inline images of the exchanges the
    request still holds: those the emergency checkpoint has not absorbed (its kept exchange included) and the
    Episode capsules do not cover. None for an ActionStep tail, where a capsule's boundary splits an exchange
    (the manifest does not say which of its images that leaves), and before the first exchange (a run's first
    request): the opening observation's images belong to no exchange until exchange 0 starts, nothing can have
    been summarized yet, and the ActionStep inventory accounts for every image."""

    tail = projection.get("exact_tail")
    if not isinstance(tail, Mapping) or tail.get("unit") != "exchange":
        return None
    inventory = tail.get("exchange_inventory")
    if not isinstance(inventory, list) or not inventory:
        return None
    episode_through = projection.get("episode_checkpointed_through_action_step")
    episode = episode_through + 1 if type(episode_through) is int else 0
    coverage = projection.get("emergency_exchange_coverage")
    summarized = coverage.get("summarized_exchanges", 0) if isinstance(coverage, Mapping) else 0
    kept = coverage.get("kept_exchange") if isinstance(coverage, Mapping) else None
    active = projection.get("emergency_checkpoint_sha256") is not None and type(summarized) is int and summarized > 0
    shown = 0
    for index, item in enumerate(inventory):
        steps = item.get("action_steps") if isinstance(item, Mapping) else None
        if not isinstance(steps, list) or len(steps) != 2 or type(item.get("inline_images")) is not int:
            return None
        if steps[0] < episode <= steps[-1]:
            return None
        if steps[-1] < episode or active and index < summarized and index != kept:
            continue
        shown += item["inline_images"]
    return shown


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _binding_counts_invalid(
    projection: Mapping[str, Any], record: Mapping[str, Any], management: Mapping[str, Any],
) -> bool:
    """The unbound and rebound counts an exchange-tail record carries (v10), against what its management allows."""

    tail = projection.get("exact_tail")
    if not isinstance(tail, Mapping) or tail.get("unit") != "exchange":
        return False
    unbound, rebound = record.get("reasoning_unbound_messages"), record.get("reasoning_rebound_messages")
    return (
        not (_count(unbound) and _count(rebound))
        or management.get("aged_argument_bytes") is None and unbound + rebound > 0
        or management.get("reasoning_binding", "strict") == "strict" and rebound > 0
    )


def _exchange_aging_violations(
    projection: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], management: Mapping[str, Any],
) -> list[str]:
    """Context v9: the aging record agrees with its management, and no request was estimated over the limit.

    Frontiers only grow across projections and an aging moves its frontier by
    at least one batch (checked by the evaluation's projection walk); here
    each projection is checked on its own: the aged span leaves at least the
    backstop's exact exchanges, the observation frontier never reaches the
    camera tail's latest observations (the latest 2 without a tail), ``all``
    withholds reasoning only where an exchange aged, ``unaged_exchanges``
    withholds it from exactly the aged exchanges (every rewrite, the backstop
    included, moves both frontiers to the same exchange), nothing is a
    reference before any exchange aged (camera images included, which only
    ``exchange_window_v1`` ages and records), and the request the projection
    sent was estimated within the hard limit. The delivered image count is
    checked with v8's (``_append_only_violations``). With an exact tail in
    exchanges the record also counts the rewritten assistant messages sent
    without their reasoning (unbound) and with it bound to the rewrite
    (rebound): none where arguments never age, none rebound under a strict
    binding.
    """

    reasons: list[str] = []
    hard = projection.get("hard_input_tokens")
    record = projection.get("exchange_aging")
    if type(hard) is not int or projection.get(
        "resolved_exchange_aging_trigger_tokens"
    ) != exchange_aging_trigger(management, hard):
        reasons.append("lossless_context_resolved_threshold_invalid")
    if not isinstance(record, Mapping) or record.get("schema") != EXCHANGE_AGING_RECORD_SCHEMA:
        return [*reasons, "lossless_context_exchange_aging_invalid"]
    exchanges = record.get("exchanges")
    aged = record.get("aged_exchanges")
    observations = record.get("aged_observation_groups")
    window = management.get("camera_tail_observations")
    groups = sum(1 for _ in inventory)
    withheld = record.get("reasoning_withheld_exchanges")
    delivered = record.get("delivered")
    keep = management.get("backstop_exact_tail_exchanges")
    window_cameras = management.get("camera_images") == CAMERA_EXCHANGE_WINDOW
    delivered_fields = EXCHANGE_DELIVERED_FIELDS | ({"camera_images"} if window_cameras else set())
    image_fields = ["delivered_tool_images_externalized"] + (
        ["delivered_camera_images_externalized"] if window_cameras else [])
    if (
        not all(_count(value) for value in (exchanges, aged, withheld, observations))
        or observations > max(0, groups - (window if type(window) is int else 2))
        or type(keep) is not int
        or aged > max(0, exchanges - keep)
        or withheld > max(0, exchanges - 1)
        or record.get("exact_tail_exchanges") != management.get("exact_tail_exchanges")
        or record.get("reasoning_replay") != management.get("reasoning_replay")
        or management.get("reasoning_replay") == REASONING_REPLAY_ALL and withheld > aged
        or management.get("reasoning_replay") == REASONING_REPLAY_UNAGED and withheld != aged
        or not isinstance(delivered, Mapping)
        or set(delivered) != delivered_fields
        or not all(_count(value) for value in delivered.values())
        or aged == 0 and any(delivered[key] for key in delivered_fields - {"reasoning"})
        or aged == 0 and any(record.get(field) != 0 for field in image_fields)
        or withheld == 0 and aged == 0 and delivered["reasoning"]
        or _binding_counts_invalid(projection, record, management)
    ):
        reasons.append("lossless_context_exchange_aging_invalid")
    reasons.extend(_request_estimate_violations(projection))
    return reasons


def _request_estimate_violations(projection: Mapping[str, Any]) -> list[str]:
    """The request was sent at an estimate within the hard limit, and a backstop that ran left it there."""

    hard = projection.get("hard_input_tokens")
    estimate = projection.get("request_estimate")
    backstop = projection.get("hard_backstop")
    if (
        not isinstance(estimate, Mapping)
        or not _count(estimate.get("tokens"))
        or estimate.get("hard_input_tokens") != hard
        or type(hard) is int and estimate["tokens"] > hard
        or backstop is not None and (
            not isinstance(backstop, Mapping)
            or backstop.get("status") != "applied"
            or not _count(backstop.get("post_estimated_input_tokens"))
            or type(hard) is not int
            or backstop["post_estimated_input_tokens"] > hard
        )
    ):
        return ["lossless_context_request_over_hard_limit"]
    return []


#: ``hard_backstop_aging.delivered`` of a v6 projection whose backstop ages by itself.
OWN_BACKSTOP_DELIVERED_FIELDS = frozenset({"tool_results", "tool_images", "reasoning"})


def _own_backstop_violations(projection: Mapping[str, Any], management: Mapping[str, Any]) -> list[str]:
    """v6 with ``backstop_exact_tail_exchanges``: the backstop's frontier record and the request estimate.

    The frontier leaves at least the backstop's exact exchanges, nothing is a
    reference before it moved, and the request was sent within the hard limit.
    """

    record = projection.get("hard_backstop_aging")
    keep = management.get("backstop_exact_tail_exchanges")
    reasons = _request_estimate_violations(projection)
    if not isinstance(record, Mapping):
        return [*reasons, "lossless_context_hard_backstop_invalid"]
    exchanges, aged = record.get("exchanges"), record.get("aged_exchanges")
    delivered = record.get("delivered")
    images = record.get("delivered_tool_images_externalized")
    if (
        record.get("aging") != "own"
        or type(keep) is not int or record.get("exact_tail_exchanges") != keep
        or not all(_count(value) for value in (exchanges, aged, images))
        or aged > max(0, exchanges - keep)
        or not isinstance(delivered, Mapping) or set(delivered) != OWN_BACKSTOP_DELIVERED_FIELDS
        or not all(_count(value) for value in delivered.values())
        or aged == 0 and (any(delivered.values()) or images != 0)
    ):
        reasons.append("lossless_context_hard_backstop_invalid")
    return reasons
