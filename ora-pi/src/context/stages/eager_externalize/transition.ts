// Ported from `ora/agent/context/deferred_eager.py` of the retired Python Native core
// (robot-harness repository): cache transition evidence when L1/L2
// rewrite an older read-only result (eager_externalize).

/**
 * Its image and media counts are equal on both sides on purpose: this pass
 * externalizes read-only results, not media; what changes is the text of
 * already-delivered results, which the prefix check would otherwise refuse.
 */
export function transitionRecord(fields: {
  projectionEpoch: number; fromEpoch: number; before: number; after: number; eventIds: string[];
  prePressure: number; postPressure: number; preBytes: number; postBytes: number;
  inlineImages: number; mediaBytes: number;
}): Record<string, unknown> {
  return {
    schema: "ora.context-projection-transition.v4",
    kind: "lifecycle_read_only_externalization",
    projection_epoch: fields.projectionEpoch,
    from_epoch: fields.fromEpoch,
    to_epoch: fields.projectionEpoch,
    reasons: ["action_step_left_protected_near_window"],
    externalized_action_steps_before: fields.before,
    externalized_action_steps_after: fields.after,
    externalized_event_ids: [...fields.eventIds],
    semantic_loss: false,
    pre_inline_images: fields.inlineImages,
    post_inline_images: fields.inlineImages,
    pre_media_bytes: fields.mediaBytes,
    post_media_bytes: fields.mediaBytes,
    pre_estimated_input_tokens: fields.prePressure,
    post_estimated_input_tokens: fields.postPressure,
    pre_request_utf8_bytes: fields.preBytes,
    post_request_utf8_bytes: fields.postBytes,
  };
}
