// What the exchange stages share (exchange_aging, reasoning_replay,
// hard_backstop; README.md): one rewrite of the exchange frontiers recorded as a
// lossless projection transition, the changed-message counts it reports, the
// observation frontier an aging may move to, and the request estimate the hard
// backstop compares with the hard limit.
//
// The aged frontier and the observation frontier are exchange_aging's slice,
// the reasoning frontier reasoning_replay's; a stage moves another's frontier
// only through a notice. Frontiers only grow.

import { MIN_CAMERA_TAIL_OBSERVATIONS } from "../../management.ts";
import { inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf } from "../../projection.ts";
import type { CameraDeliveryParams } from "../camera_delivery/stage.ts";
import { CAMERA_DELIVERY, EXCHANGE_AGING, REASONING_REPLAY, TRANSITION_SCHEMA } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { ApplyContext, Transition } from "../stage.ts";

/** exchange_aging's slice. */
export interface ExchangeAgingState {
  /** Exchanges (from the first) aged. */
  aged: number;
  /** Observation groups (from the first) whose measured body may be aged. */
  observationGroups: number;
  /** Agings that changed the request. */
  agingCount: number;
  /** `keep_anchor`: the exchange before the frontier left as it is (the anchor when the last aging ran). */
  keptExchange: number | null;
}

/** reasoning_replay's slice. */
export interface ReasoningReplayState {
  /** Exchanges (from the first) whose assistant messages are sent without their continuation state. */
  withheld: number;
}

/** The pass that only camera_delivery `exchange_window_v1` renders. */
export const EXCHANGE_CAMERA_PASS = "exchange_camera_image_reference";

/** Changed messages per exchange pass of a render (camera images: exchange_window_v1 only). */
export function exchangeChanges(notes: ReadonlyMap<string, Record<string, any>>): Record<string, number> {
  const aging = notes.get(EXCHANGE_AGING)!;
  const reasoning = notes.get(REASONING_REPLAY)!;
  return {
    arguments: aging.render.arguments.length,
    tool_results: aging.render.results.length,
    tool_images: aging.render.toolImages.length,
    ...(aging.cameraImages ? { camera_images: aging.render.cameraImages.length } : {}),
    observations: aging.render.observations.length,
    reasoning: reasoning.withheld.length,
  };
}

function total(changes: Record<string, number>): number {
  return Object.values(changes).reduce((sum, value) => sum + value, 0);
}

/**
 * The observation groups outside the camera tail's latest observations now
 * (the latest 2 when the camera mode has no tail), whose measured body an
 * aging may age.
 */
export function observationGroupsOutsideCameraTail(view: RequestView): number {
  const camera = view.paramsOf<CameraDeliveryParams>(CAMERA_DELIVERY);
  const tail = camera?.tail_observations ?? MIN_CAMERA_TAIL_OBSERVATIONS;
  return Math.max(0, view.groups.length - tail);
}

/** The exchanges in the materialized history. */
export function exchangeCount(view: RequestView): number {
  return view.exchanges.starts.length;
}

/**
 * Move the frontiers (``move``, the caller's own slice and notices) and
 * re-render; when the delivered history changed, commit and return the
 * transition. Frontiers only grow, so each pass changes a superset of what it
 * changed before: an unchanged count means an unchanged request, and the
 * frontiers stay moved without a transition.
 */
export function exchangeRewrite(
  view: RequestView, context: ApplyContext, kind: string, fields: Record<string, any>, move: () => void,
): Transition | null {
  const aging = view.stateOf<ExchangeAgingState>(EXCHANGE_AGING)!;
  const reasoning = view.stateOf<ReasoningReplayState>(REASONING_REPLAY)!;
  const before = { aged: aging.aged, reasoning: reasoning.withheld, observations: aging.observationGroups };
  const preChanges = exchangeChanges(view.notes);
  const pre = view.sizeSnapshot();
  move();
  const rendered = context.render();
  const postChanges = exchangeChanges(rendered.notes);
  if (total(postChanges) === total(preChanges)) return null;
  const epoch = context.commit(rendered);
  return {
    kind,
    record: {
      schema: TRANSITION_SCHEMA,
      kind,
      projection_epoch: epoch,
      from_epoch: view.epochBefore,
      to_epoch: epoch,
      ...fields,
      exchanges: exchangeCount(view),
      aged_exchanges_before: before.aged,
      aged_exchanges_after: aging.aged,
      aged_observation_groups_before: before.observations,
      aged_observation_groups_after: aging.observationGroups,
      reasoning_withheld_exchanges_before: before.reasoning,
      reasoning_withheld_exchanges_after: reasoning.withheld,
      externalized_messages: postChanges,
      semantic_loss: false,
      pre_inline_images: pre.images,
      post_inline_images: inlineImageCount(view.requestView),
      pre_media_bytes: pre.media_bytes,
      post_media_bytes: inlineMediaBytesFromRefs(view.working),
      pre_estimated_input_tokens: pre.pressure,
      post_estimated_input_tokens: view.pressure,
      pre_request_utf8_bytes: pre.request_bytes,
      post_request_utf8_bytes: jsonBytesOf(view.requestView),
    },
  };
}

/** The request estimate the hard backstop compares with the hard limit, and its basis. */
export function requestEstimate(view: RequestView): [number, string] {
  if (view.providerAnchorApplies()) return [view.pressure, "provider_anchored_suffix_estimate"];
  const provider = view.usage.lastProviderInputTokens;
  const estimated = view.usage.lastEstimatedTokens;
  if (provider !== null && estimated !== null && estimated > 0) {
    // A rewritten epoch has no anchor yet: scale by the last measured ratio, never down.
    return [Math.ceil(view.pressureRaw * Math.max(1, provider / estimated)), "provider_ratio_scaled_estimate"];
  }
  return [view.pressure, "harness_estimate"];
}
