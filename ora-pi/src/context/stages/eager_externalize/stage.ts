// eager_externalize (phase age, L1/L2): once an ActionStep leaves the exact
// near window, its large read-only results, then its later exact duplicates,
// become recoverable references (README.md).

import {
  type Message, inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf, messageEventIds, providerMessage,
} from "../../projection.ts";
import { clonePy, pyEqual } from "../../pyutil.ts";
import { activeSizer } from "../../request_size.ts";
import { EAGER_EXTERNALIZE, STRIDE_AGING } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, ApplyContext, Rendered, Stage, Transition } from "../stage.ts";
import type { StrideAdmission } from "../stride_aging/stage.ts";
import { type EagerPasses, eagerPasses } from "./render.ts";
import { transitionRecord } from "./transition.ts";

export interface EagerExternalizeParams {
  /** A read-only result larger than this becomes a reference with the references' preview (L1). */
  large_result_bytes: number;
  /** Later exact duplicates become references without a preview (L2); the one option so far. */
  deduplicate: true;
}

export interface EagerExternalizeState {
  /** ActionSteps (from the first) whose read-only results are externalized. */
  through: number;
}

interface EagerAdmission extends Admission {
  target: number;
}

export class EagerExternalizeStage implements Stage<EagerExternalizeParams, EagerExternalizeState, EagerAdmission> {
  readonly id = EAGER_EXTERNALIZE;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = ["lifecycle_read_only_externalization"];
  readonly requires: readonly string[] = ["materialize", STRIDE_AGING];
  readonly excludes: readonly string[] = [];
  readonly after: readonly string[] = [STRIDE_AGING];
  readonly params: EagerExternalizeParams;

  constructor(params: EagerExternalizeParams) {
    this.params = params;
  }

  init(): EagerExternalizeState {
    return { through: 0 };
  }

  stateFields(state: EagerExternalizeState): Record<string, unknown> {
    return { lifecycleReadOnlyExternalizedActionStepCount: state.through };
  }

  restoreFields(fields: Record<string, any>): EagerExternalizeState {
    return { through: fields.lifecycleReadOnlyExternalizedActionStepCount };
  }

  render(view: RequestView, state: EagerExternalizeState, history: Message[]): Rendered {
    const passes = view.memoize<EagerPasses>(this.id, history, String(state.through), () => eagerPasses(
      history, activeSizer().measure(history), {
        through: state.through, mutationTools: view.services.mutationTools,
        largeResultBytes: this.params.large_result_bytes, previewBytes: view.services.references.preview.head_bytes,
        domain: view.services.domain,
      },
    ));
    return { history: passes.history, note: { passes: clonePy(passes.records) } };
  }

  /** Its two passes of the request's current render. */
  manifest(view: RequestView): Record<string, unknown> {
    return { passes: view.note(this.id)!.passes };
  }

  /** L1/L2 run only after a step leaves the shared near window, on the stride's schedule. */
  admit(view: RequestView, state: EagerExternalizeState): EagerAdmission | null {
    const plan = view.admission<StrideAdmission>(STRIDE_AGING)!.plan;
    if (!(view.allowTransition && plan.readOnlyThrough > state.through)) return null;
    return { record: {}, admitted: true, target: plan.readOnlyThrough };
  }

  async apply(
    view: RequestView, state: EagerExternalizeState, admission: EagerAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const previous = state.through;
    const prePressure = view.pressure;
    const preBytes = jsonBytesOf(view.requestView);
    state.through = admission.target;
    const trial = context.render();
    const changed: number[] = [];
    const count = Math.min(view.aged.length, trial.aged.length);
    for (let index = 0; index < count; index += 1) {
      if (!pyEqual(providerMessage(view.aged[index]), providerMessage(trial.aged[index]))) changed.push(index);
    }
    // Nothing the model sees changes: the frontier stays moved, the request and its epoch stay.
    if (!changed.length) return null;
    const epoch = context.commit(trial);
    return {
      kind: "lifecycle_read_only_externalization",
      record: transitionRecord({
        projectionEpoch: epoch, fromEpoch: view.epochBefore,
        before: previous, after: state.through,
        eventIds: messageEventIds(trial.aged, changed),
        prePressure, postPressure: view.pressure,
        preBytes, postBytes: jsonBytesOf(view.requestView),
        inlineImages: inlineImageCount(view.requestView),
        mediaBytes: inlineMediaBytesFromRefs(view.working),
      }),
    };
  }
}
