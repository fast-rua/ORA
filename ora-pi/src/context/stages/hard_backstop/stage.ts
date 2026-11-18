// hard_backstop (phase guard): a request whose estimate exceeds the hard limit
// after every other stage has all but the latest `exact_tail_exchanges`
// exchanges aged, reasoning (and under camera_delivery `exchange_window_v1`
// camera images) included; one that still exceeds it ends the turn with
// `context_exhausted` before it is sent (README.md). It never sends a request
// it estimates over the limit, so the driver's rule for a request the Provider
// reported over the limit does not apply (`overLimit: own_estimate`). The one
// exchange rewrite that is not batched: it runs only for a request the
// Provider would otherwise refuse.
//
// Who renders the aged exchanges: with exchange stages in the scheme (v9,
// v10) the backstop moves exchange_aging's and reasoning_replay's frontiers by
// notices and they render; with `aging: own` (a scheme without them: v6) it
// keeps its own frontier and renders it at the end of the age phase
// (own_render.ts; the driver runs a guard's render there), so it reaches into
// the open ActionStep and the exact ActionStep tail, which no v6 layer ages.
// With `trigger_ratio` (v6: L6's watermark) it ages as soon as the request's
// own estimate reaches that share of the hard limit, not only over the limit.

import { inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf, messageEventIds, type Message } from "../../projection.ts";
import { PolicySessionError, clonePy, num, pyInt } from "../../pyutil.ts";
import { activeSizer, measuredPassRecord } from "../../request_size.ts";
import {
  exchangeCount, exchangeRewrite, observationGroupsOutsideCameraTail, requestEstimate,
} from "../exchange_aging/rewrite.ts";
import type { EmergencyAdmission } from "../emergency_checkpoint/stage.ts";
import {
  BACKSTOP_AGED, EMERGENCY_CHECKPOINT, EXCHANGE_AGING, HARD_BACKSTOP, REASONING_REPLAY, TRANSITION_SCHEMA,
} from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, ApplyContext, OverLimitRule, Rendered, Stage, Transition } from "../stage.ts";
import { type OwnBackstopRender, renderOwnBackstop } from "./own_render.ts";

export interface HardBackstopParams {
  /** The latest exchanges the backstop keeps exact. */
  exact_tail_exchanges: number;
  /** `own`: it keeps and renders its own aged frontier (a scheme without exchange stages); absent: theirs. */
  aging?: "own";
  /**
   * The share of the hard input limit at or above which it ages, by the
   * request's own estimate after every other stage (v6: L6's
   * `emergency_fallback_ratio`); absent: only over the hard limit.
   */
  trigger_ratio?: number;
}

export interface HardBackstopState {
  /** Requests the backstop rewrote to fit. */
  count: number;
  /** `aging: own`: the exchanges (from the first) it aged. */
  aged: number;
}

interface BackstopAdmission extends Admission {
  estimate: number;
  basis: string;
  /** The estimate it ages at: the watermark of `trigger_ratio`, or the hard limit. */
  watermark: number;
}

/**
 * L6's admission reasons that a later Provider report can turn into a
 * checkpoint: the report of the request before was below its watermark, or
 * was measured on an earlier epoch. (Without usage reports L6 never acts.)
 */
const SUMMARY_AWAITS_REPORT = new Set(["below_high_watermark", "projection_epoch_unanchored"]);

/** What an own render recorded: its input and the render. */
interface OwnNote {
  input: Message[];
  render: OwnBackstopRender;
}

/** Changed messages per pass of an own render (the transition's `externalized_messages`). */
function ownChanges(note: OwnNote | undefined): Record<string, number> {
  const render = note?.render;
  return {
    tool_results: render?.results.length ?? 0,
    tool_images: render?.toolImages.length ?? 0,
    reasoning: render?.reasoning.length ?? 0,
  };
}

function total(changes: Record<string, number>): number {
  return Object.values(changes).reduce((sum, value) => sum + value, 0);
}

export class HardBackstopStage implements Stage<HardBackstopParams, HardBackstopState, BackstopAdmission> {
  readonly id = HARD_BACKSTOP;
  readonly phase = "guard" as const;
  readonly transitions: readonly string[] = ["hard_backstop"];
  readonly requires: readonly string[];
  readonly excludes: readonly string[];
  readonly overLimit: OverLimitRule = "own_estimate";
  /** It reads L6's admission (whether L6 only waits for a report) when the scheme runs L6. */
  readonly after: readonly string[] = [EMERGENCY_CHECKPOINT];
  readonly params: HardBackstopParams;

  constructor(params: HardBackstopParams) {
    this.params = params;
    this.requires = this.own ? [] : [EXCHANGE_AGING, REASONING_REPLAY];
    this.excludes = this.own ? [EXCHANGE_AGING, REASONING_REPLAY] : [];
  }

  /** It ages by itself (`aging: own`). */
  get own(): boolean {
    return this.params.aging === "own";
  }

  init(): HardBackstopState {
    return { count: 0, aged: 0 };
  }

  stateFields(state: HardBackstopState): Record<string, unknown> {
    // Its own frontier is the snapshot's exchange frontier (state_fields.ts EXCHANGE_FIELDS).
    return { hardBackstopCount: state.count, ...(this.own ? { exchangeAgedCount: state.aged } : {}) };
  }

  restoreFields(fields: Record<string, any>): HardBackstopState {
    return { count: fields.hardBackstopCount, aged: this.own ? fields.exchangeAgedCount ?? 0 : 0 };
  }

  check(view: RequestView, state: HardBackstopState): void {
    if (this.own && state.aged > exchangeCount(view)) {
      throw new PolicySessionError("hard backstop frontier exceeds the exchanges of the history");
    }
  }

  /** `aging: own`: the history with every exchange before its frontier aged (the driver runs it after the age phase). */
  render(view: RequestView, state: HardBackstopState, history: Message[]): Rendered {
    if (!this.own) return { history };
    const render = view.memoize(this.id, history, String(state.aged), () => renderOwnBackstop(history, {
      agedExchanges: state.aged, previewBytes: view.services.references.preview.head_bytes,
      mutationTools: view.services.mutationTools, domain: view.services.domain,
    }));
    const note: OwnNote = { input: history, render };
    return { history: render.messages, note: note as unknown as Record<string, any> };
  }

  /** The estimate it ages at: `trigger_ratio` of the hard limit, or the hard limit itself. */
  watermark(hardInputTokens: number): number {
    const ratio = this.params.trigger_ratio;
    return ratio === undefined ? hardInputTokens : pyInt(hardInputTokens * ratio);
  }

  /**
   * A request estimated over the hard limit after every other stage; with
   * `trigger_ratio`, also one at or above its watermark that L6 cannot shrink
   * on the next report (nothing before its exact tail to absorb, as inside an
   * open ActionStep, or no report coming), once at least the exact tail's
   * exchanges more can age.
   */
  admit(view: RequestView, state: HardBackstopState): BackstopAdmission | null {
    const [estimate, basis] = requestEstimate(view);
    const hard = view.limits.hardInputTokens;
    const watermark = this.watermark(hard);
    if (this.params.trigger_ratio === undefined ? estimate <= hard : estimate < watermark) return null;
    if (estimate <= hard) {
      // Within the limit: not while L6 has something to absorb and only waits for the Provider's report of this
      // request (it summarizes on the next one), and only by at least the exact tail's exchanges, so a request
      // that stays above the watermark is not rewritten on every call.
      if (this.summaryPending(view)) return null;
      const target = Math.max(0, exchangeCount(view) - this.params.exact_tail_exchanges);
      if (target - state.aged < this.params.exact_tail_exchanges) return null;
    }
    return { admitted: true, record: {}, estimate, basis, watermark };
  }

  /** L6 would summarize once the Provider reports a request at its watermark: it has fresh ActionSteps to absorb. */
  private summaryPending(view: RequestView): boolean {
    const emergency = view.admission<EmergencyAdmission>(EMERGENCY_CHECKPOINT);
    if (emergency === null || emergency.admitted || !emergency.fresh) return false;
    return SUMMARY_AWAITS_REPORT.has(String(emergency.record.reason));
  }

  async apply(
    view: RequestView, state: HardBackstopState, admission: BackstopAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const hard = view.limits.hardInputTokens;
    const keep = this.params.exact_tail_exchanges;
    const target = Math.max(0, exchangeCount(view) - keep);
    const fields = {
      reasons: [admission.estimate > hard ? "estimated_input_over_hard_limit" : "estimated_input_at_or_above_watermark"],
      estimate_basis: admission.basis,
      hard_input_tokens: hard,
      backstop_exact_tail_exchanges: keep,
      ...(this.params.trigger_ratio === undefined ? {} : { watermark_tokens: admission.watermark }),
    };
    let transition: Transition | null = null;
    if (view.allowTransition) {
      transition = this.own ? this.ownRewrite(view, state, context, target, fields) : exchangeRewrite(
        view, context, "hard_backstop", fields, () => {
          context.notify(EXCHANGE_AGING, BACKSTOP_AGED, {
            target, observation_groups: observationGroupsOutsideCameraTail(view),
            anchor: view.exchangeTail?.anchor ?? -1,
          });
          context.notify(REASONING_REPLAY, BACKSTOP_AGED, { target });
        });
    }
    const [after, afterBasis] = requestEstimate(view);
    if (after > hard) {
      throw new PolicySessionError(
        `context_exhausted: estimated ${after} input tokens (${afterBasis}) after the hard backstop kept `
        + `only the latest ${keep} exchanges exact, hard limit ${hard}`,
      );
    }
    state.count += 1;
    Object.assign(admission.record, {
      status: "applied",
      estimate_basis: admission.basis,
      pre_estimated_input_tokens: admission.estimate,
      post_estimate_basis: afterBasis,
      post_estimated_input_tokens: after,
      hard_input_tokens: hard,
      ...(this.params.trigger_ratio === undefined ? {} : { watermark_tokens: admission.watermark }),
      backstop_exact_tail_exchanges: keep,
      hard_backstop_count: state.count,
    });
    return transition;
  }

  /**
   * `aging: own`: move the frontier to ``target`` and re-render; when the
   * delivered history changed, commit and return the transition (in the
   * exchange stages' record shape: the aged and the reasoning frontier are one).
   */
  private ownRewrite(
    view: RequestView, state: HardBackstopState, context: ApplyContext, target: number, fields: Record<string, any>,
  ): Transition | null {
    const before = state.aged;
    const preChanges = ownChanges(view.note(this.id) as unknown as OwnNote);
    const pre = view.sizeSnapshot();
    state.aged = Math.max(state.aged, target);
    const rendered = context.render();
    const postChanges = ownChanges(rendered.notes.get(this.id) as unknown as OwnNote);
    if (total(postChanges) === total(preChanges)) return null;
    const epoch = context.commit(rendered);
    return {
      kind: "hard_backstop",
      record: {
        schema: TRANSITION_SCHEMA,
        kind: "hard_backstop",
        projection_epoch: epoch,
        from_epoch: view.epochBefore,
        to_epoch: epoch,
        ...fields,
        aging: "own",
        exchanges: exchangeCount(view),
        aged_exchanges_before: before,
        aged_exchanges_after: state.aged,
        aged_observation_groups_before: 0,
        aged_observation_groups_after: 0,
        reasoning_withheld_exchanges_before: before,
        reasoning_withheld_exchanges_after: state.aged,
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

  /**
   * This request's backstop record (null when it did not run) and the estimate
   * it was sent at; with `aging: own`, its passes and its frontier's record.
   */
  manifest(view: RequestView, state: HardBackstopState, admission: BackstopAdmission | null): Record<string, unknown> {
    const [tokens, basis] = requestEstimate(view);
    const fields: Record<string, unknown> = {
      hard_backstop: admission === null ? null : clonePy(admission.record),
      request_estimate: { tokens, basis, hard_input_tokens: view.limits.hardInputTokens },
    };
    if (!this.own) return fields;
    const note = view.note(this.id) as unknown as OwnNote;
    const { render } = note;
    const sizer = activeSizer();
    const before = sizer.measure(note.input);
    const after = sizer.measure(render.messages);
    // One measurement for the three passes, credited to the first (as exchange_aging's passes are).
    const sections: [string, number[], Record<string, any>[]][] = [
      ["exchange_tool_result_reference", render.results, render.resultRefs],
      ["exchange_tool_image_reference", render.toolImages, render.toolImageRefs],
      ["exchange_reasoning_withheld", render.reasoning, []],
    ];
    const passes = sections.map(([name, changed, refs], index) => measuredPassRecord(
      name, index === 0 ? before : after, after, {
        changed_messages: changed, artifact_refs: refs,
        externalized_event_ids: messageEventIds(render.messages, changed),
      }));
    return {
      ...fields,
      passes: clonePy(passes),
      hard_backstop_aging: {
        aging: "own",
        ...(this.params.trigger_ratio === undefined ? {} : {
          watermark_tokens: this.watermark(view.limits.hardInputTokens),
        }),
        exchanges: exchangeCount(view),
        exact_tail_exchanges: this.params.exact_tail_exchanges,
        aged_exchanges: state.aged,
        delivered: ownChanges(note),
        delivered_tool_images_externalized: view.working.reduce(
          (sum, message) => sum + num(message._lossless_exchange_images ?? 0), 0),
      },
    };
  }
}
