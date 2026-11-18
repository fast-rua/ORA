// The estimator service: the one request-size estimate. Text is charged by its
// canonical bytes (`estimatedTokens`, bytes / 3); the pressure estimate charges
// each inline image `imageTokens` instead of its transport bytes; once the
// Provider has reported a request of the current projection epoch, the
// estimate is anchored on that report (the difference carries over to the
// next request of the epoch).

import {
  IMAGE_PRESSURE_ESTIMATE_TOKENS, estimatedTokens, inlineImageCount, pressureEstimatedTokens,
} from "../../projection.ts";

/** Tokens per inline image when a scheme names none (v6). */
export const DEFAULT_IMAGE_TOKENS = IMAGE_PRESSURE_ESTIMATE_TOKENS;

/** The Provider report an estimate may be anchored on. */
export interface UsageAnchor {
  /** The report was measured on the current projection epoch. */
  applies: boolean;
  inputTokens: number | null;
  estimatedTokens: number | null;
}

export interface CalibratedEstimate {
  tokens: number;
  delta: number;
  status: "provider_anchored_suffix_estimate" | "projection_epoch_unanchored" | "provider_usage_unavailable";
}

export class RequestEstimator {
  readonly imageTokens: number;

  constructor(imageTokens: number) {
    this.imageTokens = imageTokens;
  }

  /** Text bytes / 3, images by their bytes. */
  raw(request: unknown): number {
    return estimatedTokens(request);
  }

  /** Text bytes / 3, `imageTokens` per inline image. */
  pressure(request: unknown): number {
    return pressureEstimatedTokens(request, this.imageTokens);
  }

  /** The tokens the inline images of ``request`` are charged. */
  imageEstimate(request: unknown): number {
    return inlineImageCount(request) * this.imageTokens;
  }

  /** ``pressure`` anchored on the Provider's report of this epoch, when there is one. */
  calibrate(pressure: number, anchor: UsageAnchor): CalibratedEstimate {
    if (anchor.applies) {
      const delta = anchor.inputTokens! - anchor.estimatedTokens!;
      return { tokens: Math.max(1, pressure + delta), delta, status: "provider_anchored_suffix_estimate" };
    }
    return {
      tokens: Math.max(1, pressure), delta: 0,
      status: anchor.inputTokens !== null ? "projection_epoch_unanchored" : "provider_usage_unavailable",
    };
  }
}

/** A summary's own size estimate (text bytes / 3). */
export { estimatedTokens };
