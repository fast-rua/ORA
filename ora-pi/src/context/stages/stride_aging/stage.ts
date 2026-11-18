// stride_aging (phase age, first): when the age stages move their frontiers,
// and to where (README.md). It has no render and no state: its admission on
// each request is the aging plan the other age stages execute.

import { PolicySessionError } from "../../pyutil.ts";
import { EAGER_EXTERNALIZE, MEDIA_WINDOW, STRIDE_AGING } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, Stage } from "../stage.ts";
import { type TailAging, planTailAging } from "./plan.ts";

export interface StrideAgingParams {
  /** ActionSteps a frontier must lag its target by before it moves (A4). */
  stride: number;
  /**
   * true: the tails are minimums that age in batches of `stride`, and the
   * manifest reports the aging (v5 on). false: every frontier moves at each
   * fresh observation (v4's unstrided aging; `stride` must be 1).
   */
  strided: boolean;
}

/** The media window's parameters this stage reads. */
interface MediaWindowParams {
  tail_action_steps: number;
}

export interface StrideAdmission extends Admission {
  plan: TailAging;
}

type NoState = Record<string, never>;

export class StrideAgingStage implements Stage<StrideAgingParams, NoState, StrideAdmission> {
  readonly id = STRIDE_AGING;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = [];
  readonly requires: readonly string[] = ["materialize", MEDIA_WINDOW];
  readonly excludes: readonly string[] = [];
  readonly params: StrideAgingParams;

  constructor(params: StrideAgingParams) {
    this.params = params;
  }

  init(): NoState {
    return {};
  }

  stateFields(): Record<string, unknown> {
    return {};
  }

  restoreFields(): NoState {
    return {};
  }

  private mediaTail(view: RequestView): number {
    return view.paramsOf<MediaWindowParams>(MEDIA_WINDOW)!.tail_action_steps;
  }

  /**
   * Read-only bodies keep one exact near window: the scheme's exact tail or the
   * media tail, whichever is longer; media has its own current-Episode tail.
   */
  nearTail(view: RequestView): number {
    return Math.max(view.exactTail, this.mediaTail(view));
  }

  private frontier(view: RequestView, stageId: string): number {
    return view.stateOf<{ through: number }>(stageId)?.through ?? 0;
  }

  check(view: RequestView): void {
    const eligibleThrough = Math.max(0, view.completedCount - this.nearTail(view));
    if (this.frontier(view, EAGER_EXTERNALIZE) > eligibleThrough) {
      throw new PolicySessionError("read-only coverage enters protected near window");
    }
  }

  /** The plan, from the frontiers as this request found them; never applied. */
  admit(view: RequestView): StrideAdmission {
    const plan = planTailAging({
      strided: this.params.strided, stride: this.params.stride, mediaTail: this.mediaTail(view),
      inventory: view.inventory, completed: view.completedCount, nearTail: this.nearTail(view),
      readOnlyActive: this.frontier(view, EAGER_EXTERNALIZE), mediaActive: this.frontier(view, MEDIA_WINDOW),
      allowTransition: view.allowTransition,
    });
    return { record: plan.manifestFields(), admitted: false, plan };
  }

  manifest(view: RequestView, _state: NoState, admission: StrideAdmission | null): Record<string, unknown> {
    return {
      resolved_near_tail_action_steps: this.nearTail(view),
      resolved_media_retention_policy: admission!.plan.mediaPolicy,
      ...admission!.plan.manifestFields(),
    };
  }
}
