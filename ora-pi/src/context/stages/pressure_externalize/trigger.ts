// L4's estimate watermark: the absolute floor, or a share of the hard input
// limit, whichever is larger.

import { pyInt } from "../../pyutil.ts";

/** 48000 of the 197952 usable tokens of the 200K window it was chosen for. */
export const ESTIMATE_PRESSURE_WINDOW_SHARE = 48_000 / (200_000 - 2_048);

export function estimatePressureTrigger(floorTokens: number, windowShare: number, hardInputTokens: number): number {
  return Math.max(floorTokens, pyInt(hardInputTokens * windowShare));
}
