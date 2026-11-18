// When the lossless Context ages older ActionSteps (ported from
// `ora/agent/context/tail_aging.py` of the retired Python Native core).
//
// Unstrided (v4), every lifecycle frontier moves one ActionStep at every fresh
// observation. Strided (v5 on), the read-only and media tails are minimums
// that age in batches of `stride` ActionSteps; with stride 1 they reproduce v4
// byte for byte. L4 pressure is evaluated only while no aging is pending.

import { eq, get } from "../../pyutil.ts";

export const V4_MEDIA_POLICY = "current_round_action_step_tail_v1";
export const STRIDE_MEDIA_POLICY = "current_round_action_step_tail_stride_v1";
export const TAIL_AGING_RECORD_SCHEMA = "ora.context-tail-aging.v1";

/** One projection's aging decision; frontiers count ActionSteps. */
export class TailAging {
  /** Whether aging is strided (v5 on): it then reports itself in the manifest. */
  readonly strided: boolean;
  readonly stride: number;
  readonly readOnlyTarget: number;
  readonly mediaTarget: number;
  readonly readOnlyThrough: number;
  readonly mediaThrough: number;
  readonly reasons: string[];
  readonly aged: boolean;

  constructor(fields: {
    strided: boolean; stride: number; readOnlyTarget: number; mediaTarget: number;
    readOnlyThrough: number; mediaThrough: number; reasons: string[]; aged: boolean;
  }) {
    this.strided = fields.strided;
    this.stride = fields.stride;
    this.readOnlyTarget = fields.readOnlyTarget;
    this.mediaTarget = fields.mediaTarget;
    this.readOnlyThrough = fields.readOnlyThrough;
    this.mediaThrough = fields.mediaThrough;
    this.reasons = fields.reasons;
    this.aged = fields.aged;
  }

  /** Whether older steps stay unaged beyond a minimum tail. */
  get pending(): boolean {
    return this.readOnlyThrough < this.readOnlyTarget || this.mediaThrough < this.mediaTarget;
  }

  get mediaPolicy(): string {
    return this.strided ? STRIDE_MEDIA_POLICY : V4_MEDIA_POLICY;
  }

  /** Projection-manifest fields; v4 evidence keeps its exact field set. */
  manifestFields(): Record<string, unknown> {
    if (!this.strided) return {};
    return {
      resolved_tail_aging_stride_action_steps: this.stride,
      tail_aging: {
        schema: TAIL_AGING_RECORD_SCHEMA,
        stride_action_steps: this.stride,
        aged: this.aged,
        reasons: [...this.reasons],
        read_only_target_action_steps: this.readOnlyTarget,
        media_target_action_steps: this.mediaTarget,
        pending_read_only_action_steps: this.readOnlyTarget - this.readOnlyThrough,
        pending_media_action_steps: this.mediaTarget - this.mediaThrough,
        pressure_admission: this.pending ? "deferred_to_tail_aging" : "evaluated",
      },
    };
  }
}

/** Decide whether this projection ages, and to which frontiers. */
export function planTailAging(options: {
  strided: boolean; stride: number; mediaTail: number; inventory: Record<string, any>[]; completed: number;
  nearTail: number; readOnlyActive: number; mediaActive: number; allowTransition: boolean;
}): TailAging {
  const { strided, stride, mediaTail, inventory, completed, nearTail, readOnlyActive, mediaActive } = options;
  const currentRound = inventory[inventory.length - 1].session_round;
  const currentRoundCompleted = inventory.filter(
    (item) => get(item, "complete") === true && eq(get(item, "session_round"), currentRound),
  ).length;
  const readOnlyTarget = Math.max(0, completed - nearTail);
  const mediaTarget = completed - Math.min(mediaTail, currentRoundCompleted);
  const reasons: string[] = [];
  if (readOnlyTarget - readOnlyActive >= stride) reasons.push("read_only_stride_reached");
  if (mediaTarget - mediaActive >= stride) reasons.push("media_stride_reached");
  if (mediaActive < completed && !eq(get(inventory[mediaActive], "session_round"), currentRound)) {
    reasons.push("prior_episode_media_released");
  }
  const aged = options.allowTransition && reasons.length > 0;
  return new TailAging({
    strided, stride,
    readOnlyTarget, mediaTarget,
    readOnlyThrough: aged ? Math.max(readOnlyActive, readOnlyTarget) : readOnlyActive,
    mediaThrough: aged ? Math.max(mediaActive, mediaTarget) : mediaActive,
    reasons, aged,
  });
}
