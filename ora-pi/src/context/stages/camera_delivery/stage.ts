// camera_delivery (phase deliver): where the fixed camera images of
// observations reach the model (README.md; delivery.ts does the rendering).

import type { Message } from "../../projection.ts";
import { clonePy, get, hasKey } from "../../pyutil.ts";
import { CAMERA_EXCHANGE_WINDOW, CAMERA_ROLLING_TAIL } from "../../management.ts";
import { PolicySessionError } from "../../pyutil.ts";
import { CAMERA_DELIVERY, EXCHANGE_AGING, MEDIA_WINDOW } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Rendered, Stage } from "../stage.ts";
import { type CameraDeliveryOptions, cameraTail, deliveredHistory } from "./delivery.ts";

export interface CameraDeliveryParams {
  /**
   * `rolling_tail_v1`, `inline_until_compaction_v1` or `exchange_window_v1`
   * (management.ts CAMERA_IMAGE_MODES; the last ages them with their exchange, exchange_aging).
   */
  mode: string;
  /** Rolling mode: the observations whose camera images the tail carries; the other modes: null. */
  tail_observations: number | null;
}

type NoState = Record<string, never>;

export class CameraDeliveryStage implements Stage<CameraDeliveryParams, NoState> {
  readonly id = CAMERA_DELIVERY;
  readonly phase = "deliver" as const;
  readonly transitions: readonly string[] = [];
  readonly requires: readonly string[] = ["materialize"];
  /** Both would decide where camera images go. */
  readonly excludes: readonly string[] = [MEDIA_WINDOW];
  readonly params: CameraDeliveryParams;

  constructor(params: CameraDeliveryParams) {
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

  /** `exchange_window_v1` leaves camera images to the exchange aging. */
  check(view: RequestView): void {
    if (this.params.mode === CAMERA_EXCHANGE_WINDOW && view.paramsOf(EXCHANGE_AGING) === null) {
      throw new PolicySessionError(`camera_delivery ${CAMERA_EXCHANGE_WINDOW} needs the exchange_aging stage`);
    }
    // The rolling tail counts observations by ActionStep; an exact tail in exchanges keeps camera images in place.
    if (this.params.mode === CAMERA_ROLLING_TAIL && view.exactTailUnit === "exchange") {
      throw new PolicySessionError(`camera_delivery ${CAMERA_ROLLING_TAIL} needs an exact tail in ActionSteps`);
    }
  }

  /** The retrieval a history reference names needs an observe tool taking ``step``. */
  private options(tools: Record<string, any>[]): CameraDeliveryOptions {
    const observe = tools.find((tool) => get(tool, "name") === "observe");
    const properties = get(get(observe ?? {}, "parameters") ?? {}, "properties");
    return {
      mode: this.params.mode,
      tailObservations: this.params.tail_observations,
      observeStep: properties !== null && typeof properties === "object" && hasKey(properties, "step"),
    };
  }

  render(view: RequestView, _state: NoState, history: Message[]): Rendered {
    const options = this.options(view.tools);
    const tail = cameraTail(view.materialized, options);
    return { history: deliveredHistory(history, options), appendix: tail.messages, note: tail.record };
  }

  manifest(view: RequestView): Record<string, unknown> {
    return { camera_delivery: clonePy(view.note(CAMERA_DELIVERY)) };
  }
}
