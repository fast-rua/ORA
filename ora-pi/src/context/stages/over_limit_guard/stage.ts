// over_limit_guard (phase guard): what happens to a request the Provider
// reported over the hard input limit (README.md). The driver applies the rule
// (driver.ts `overLimitRule`); this stage declares it (`overLimit`) and records
// each request it lets through.

import { clonePy } from "../../pyutil.ts";
import { OVER_LIMIT_GUARD } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, Stage, Transition } from "../stage.ts";

/** `refuse`: the turn ends with `context_exhausted`; `send_and_record`: sent with a warning record. */
export type OverLimitPolicy = "refuse" | "send_and_record";

export interface OverLimitGuardParams {
  over_limit: OverLimitPolicy;
}

export interface OverLimitGuardState {
  /** Requests sent although the Provider reported more input than the hard limit. */
  overLimitRequestCount: number;
}

export class OverLimitGuardStage implements Stage<OverLimitGuardParams, OverLimitGuardState> {
  readonly id = OVER_LIMIT_GUARD;
  readonly phase = "guard" as const;
  readonly transitions: readonly string[] = [];
  readonly requires: readonly string[] = [];
  readonly excludes: readonly string[] = [];
  readonly params: OverLimitGuardParams;

  constructor(params: OverLimitGuardParams) {
    this.params = params;
  }

  /** The driver's over-limit rule is this guard's policy. */
  get overLimit(): OverLimitPolicy {
    return this.params.over_limit;
  }

  init(): OverLimitGuardState {
    return { overLimitRequestCount: 0 };
  }

  stateFields(state: OverLimitGuardState): Record<string, unknown> {
    return { overLimitRequestCount: state.overLimitRequestCount };
  }

  restoreFields(fields: Record<string, any>): OverLimitGuardState {
    return { overLimitRequestCount: fields.overLimitRequestCount };
  }

  /** The driver has refused an over-limit request unless the policy sends it; then it is recorded. */
  admit(view: RequestView): Admission | null {
    if (view.overLimitInputTokens === null) return null;
    return {
      admitted: true,
      record: {
        status: "warning_sent_anyway",
        provider_input_tokens: view.overLimitInputTokens,
        hard_input_tokens: view.limits.hardInputTokens,
        estimated_input_tokens: view.pressure,
      },
    };
  }

  /** Counts the request; the delivered request does not change, so there is no transition. */
  async apply(_view: RequestView, state: OverLimitGuardState, admission: Admission): Promise<Transition | null> {
    state.overLimitRequestCount += 1;
    admission.record.over_limit_request_count = state.overLimitRequestCount;
    return null;
  }

  manifest(_view: RequestView, _state: OverLimitGuardState, admission: Admission | null): Record<string, unknown> {
    if (this.params.over_limit !== "send_and_record") return {};
    return { context_over_limit: admission === null ? null : clonePy(admission.record) };
  }
}
