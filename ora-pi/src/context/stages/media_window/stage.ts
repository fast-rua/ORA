// media_window (phase age, L3): the media of ActionSteps outside the current
// Episode's latest tail become recoverable references (README.md).

import { type Message, inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf } from "../../projection.ts";
import { PolicySessionError, clonePy, sortedStrings } from "../../pyutil.ts";
import { activeSizer } from "../../request_size.ts";
import { CAMERA_DELIVERY, MEDIA_WINDOW, STRIDE_AGING, TRANSITION_SCHEMA } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, ApplyContext, Rendered, Stage, Transition } from "../stage.ts";
import type { StrideAdmission } from "../stride_aging/stage.ts";
import { type MediaPass, mediaPass } from "./render.ts";

export interface MediaWindowParams {
  /** Completed ActionSteps of the current Episode whose media stay inline. */
  tail_action_steps: number;
}

export interface MediaWindowState {
  /** ActionSteps (from the first) whose media are externalized. */
  through: number;
  /** Tool-image events externalized wherever they are (no producer adds one yet; carried for restore). */
  superseded: string[];
}

interface MediaAdmission extends Admission {
  target: number;
}

export class MediaWindowStage implements Stage<MediaWindowParams, MediaWindowState, MediaAdmission> {
  readonly id = MEDIA_WINDOW;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = ["lifecycle_media_externalization"];
  readonly requires: readonly string[] = ["materialize", STRIDE_AGING];
  readonly excludes: readonly string[] = [CAMERA_DELIVERY];
  readonly after: readonly string[] = [STRIDE_AGING];
  readonly params: MediaWindowParams;

  constructor(params: MediaWindowParams) {
    this.params = params;
  }

  init(): MediaWindowState {
    return { through: 0, superseded: [] };
  }

  stateFields(state: MediaWindowState): Record<string, unknown> {
    return {
      lifecycleMediaExternalizedActionStepCount: state.through,
      lifecycleSupersededToolImageEventIds: state.superseded,
    };
  }

  restoreFields(fields: Record<string, any>): MediaWindowState {
    return {
      through: fields.lifecycleMediaExternalizedActionStepCount,
      superseded: fields.lifecycleSupersededToolImageEventIds,
    };
  }

  check(view: RequestView, state: MediaWindowState): void {
    if (state.through > view.completedCount) {
      throw new PolicySessionError("lifecycle media coverage exceeds completed ActionSteps");
    }
  }

  render(view: RequestView, state: MediaWindowState, history: Message[]): Rendered {
    const key = JSON.stringify([state.through, [...state.superseded]]);
    const pass = view.memoize<MediaPass>(this.id, history, key, () => mediaPass(
      history, activeSizer().measure(history), { through: state.through, supersededToolImageEventIds: state.superseded },
    ));
    return { history: pass.history, note: { passes: [clonePy(pass.record)] } };
  }

  /** An Episode-aware fixed media tail, on the stride's schedule; a new Episode releases the earlier one's media. */
  admit(view: RequestView, state: MediaWindowState): MediaAdmission | null {
    const plan = view.admission<StrideAdmission>(STRIDE_AGING)!.plan;
    if (!(view.allowTransition && plan.mediaThrough > state.through)) return null;
    return { record: {}, admitted: true, target: plan.mediaThrough };
  }

  async apply(
    view: RequestView, state: MediaWindowState, admission: MediaAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const pre = view.sizeSnapshot();
    const previous = state.through;
    const superseded = sortedStrings(new Set(state.superseded));
    state.through = admission.target;
    state.superseded = superseded;
    const trial = context.render();
    if (!trial.notes.get(this.id)!.passes[0].changed_messages.length) return null;
    const epoch = context.commit(trial);
    return {
      kind: "lifecycle_media_externalization",
      record: {
        schema: TRANSITION_SCHEMA,
        kind: "lifecycle_media_externalization",
        projection_epoch: epoch,
        from_epoch: view.epochBefore,
        to_epoch: epoch,
        reasons: ["recent_media_tail_advanced"],
        externalized_action_steps_before: previous,
        externalized_action_steps_after: state.through,
        inline_media_tail_action_steps: this.params.tail_action_steps,
        protected_current_round: view.currentRound,
        superseded_tool_image_events_before: superseded,
        superseded_tool_image_events_after: [...superseded],
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

  /** Its pass of the request's current render, and the media frontier for the coverage fields. */
  manifest(view: RequestView, state: MediaWindowState): Record<string, unknown> {
    return { passes: view.note(this.id)!.passes, lifecycle_media_externalized_action_steps: state.through };
  }
}
