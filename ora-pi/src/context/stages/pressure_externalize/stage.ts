// pressure_externalize (phase age, L4): under pressure, every remaining
// read-only result and older observation body before the read-only frontier
// becomes a recoverable reference (README.md).

import { type Message, inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf } from "../../projection.ts";
import { PolicySessionError, pyInt } from "../../pyutil.ts";
import { activeSizer } from "../../request_size.ts";
import { PRESSURE_EXTERNALIZE, STRIDE_AGING, TRANSITION_SCHEMA } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, ApplyContext, Rendered, Stage, Transition } from "../stage.ts";
import type { StrideAdmission } from "../stride_aging/stage.ts";
import { pressurePasses } from "./render.ts";
import { estimatePressureTrigger } from "./trigger.ts";

export interface PressureExternalizeParams {
  /** More inline images than this is pressure. */
  inline_images: number;
  /** The estimate watermark's floor, in tokens. */
  estimated_tokens: number;
  /** The estimate watermark as a share of the hard input limit (ESTIMATE_PRESSURE_WINDOW_SHARE). */
  window_share: number;
  /** Provider-reported input at or above this share of the hard input limit is pressure. */
  provider_input_ratio: number;
  /** false only when a context pack turns L4 off: never admitted, the passes render unchanged. */
  enabled: boolean;
}

export interface PressureExternalizeState {
  /** ActionSteps (from the first) externalized under pressure. */
  through: number;
}

interface PressureAdmission extends Admission {
  target: number;
  reasons: string[];
}

type PressureStage = Stage<PressureExternalizeParams, PressureExternalizeState, PressureAdmission>;

export class PressureExternalizeStage implements PressureStage {
  readonly id = PRESSURE_EXTERNALIZE;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = ["pressure_externalization"];
  readonly requires: readonly string[] = ["materialize", STRIDE_AGING];
  readonly excludes: readonly string[] = [];
  readonly after: readonly string[] = [STRIDE_AGING];
  readonly params: PressureExternalizeParams;

  constructor(params: PressureExternalizeParams) {
    this.params = params;
  }

  init(): PressureExternalizeState {
    return { through: 0 };
  }

  stateFields(state: PressureExternalizeState): Record<string, unknown> {
    return { pressureExternalizedActionStepCount: state.through };
  }

  restoreFields(fields: Record<string, any>): PressureExternalizeState {
    return { through: fields.pressureExternalizedActionStepCount };
  }

  check(view: RequestView, state: PressureExternalizeState): void {
    if (state.through > view.completedCount) {
      throw new PolicySessionError("pressure-GC coverage exceeds completed ActionSteps");
    }
  }

  render(view: RequestView, state: PressureExternalizeState, history: Message[]): Rendered {
    const passes = pressurePasses(history, activeSizer().measure(history), {
      through: state.through, mutationTools: view.services.mutationTools,
      previewBytes: view.services.references.preview.head_bytes, domain: view.services.domain,
    });
    return { history: passes.history, note: { passes: passes.records } };
  }

  /** Which thresholds this request crosses; none while the stride has aging pending. */
  private reasons(view: RequestView, plan: StrideAdmission["plan"]): string[] {
    const { params } = this;
    const hard = view.limits.hardInputTokens;
    const reasons: string[] = [];
    // Pending stride slack is not pressure; L4 waits for the next batch.
    if (!(view.allowTransition && !plan.pending)) return reasons;
    if (inlineImageCount(view.requestView) > params.inline_images) reasons.push("inline_images");
    if (view.pressure > estimatePressureTrigger(params.estimated_tokens, params.window_share, hard)) {
      reasons.push("estimated_tokens");
    }
    const providerTrigger = pyInt(hard * params.provider_input_ratio);
    const reported = view.usage.lastProviderInputTokens;
    if (view.providerAnchorApplies() && reported !== null && reported >= providerTrigger) {
      reasons.push("provider_input_tokens");
    }
    return reasons;
  }

  admit(view: RequestView, state: PressureExternalizeState): PressureAdmission | null {
    if (!this.params.enabled) return null;
    const plan = view.admission<StrideAdmission>(STRIDE_AGING)!.plan;
    const reasons = this.reasons(view, plan);
    if (!(reasons.length && plan.readOnlyThrough > state.through)) return null;
    return { record: {}, admitted: true, target: plan.readOnlyThrough, reasons };
  }

  async apply(
    view: RequestView, state: PressureExternalizeState, admission: PressureAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const pre = view.sizeSnapshot();
    const previous = state.through;
    state.through = admission.target;
    const trial = context.render();
    if (!trial.notes.get(this.id)!.passes.some((pass: Record<string, any>) => pass.changed_messages.length)) {
      return null;
    }
    const epoch = context.commit(trial);
    return {
      kind: "pressure_externalization",
      record: {
        schema: TRANSITION_SCHEMA,
        kind: "pressure_externalization",
        projection_epoch: epoch,
        from_epoch: view.epochBefore,
        to_epoch: epoch,
        reasons: admission.reasons,
        externalized_action_steps_before: previous,
        externalized_action_steps_after: state.through,
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

  /** Its two passes of the request's current render, the inline-image target and the pressure frontier. */
  manifest(view: RequestView, state: PressureExternalizeState): Record<string, unknown> {
    return {
      passes: view.note(this.id)!.passes,
      resolved_inline_image_target: this.params.inline_images,
      pressure_externalized_action_steps: state.through,
    };
  }
}
