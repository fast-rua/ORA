// Ported from `ora/agent/context/preview_shortfall.py` of the retired Python
// Native core: what an externalized preview is missing, stated in the record itself.
// ``keptBytes`` is given for a head-and-tail preview (stages/services/references.ts
// ``headTailPreview``), which is no prefix and no summary: the bytes of the content it keeps.

import { cpLen, cpPrefix, isMapping, utf8Len } from "./pyutil.ts";

export function previewShortfall(
  content: string, preview: string, value: unknown, keptBytes: number | null = null,
): Record<string, unknown> {
  const total = utf8Len(content);
  const kept = keptBytes ?? utf8Len(preview);
  const summarized = keptBytes === null && Boolean(preview) && preview !== cpPrefix(content, cpLen(preview));
  const shortfall: Record<string, unknown> = {
    preview_truncated: !summarized && kept < total,
    preview_bytes: kept,
    result_bytes: total,
  };
  if (summarized) shortfall.preview_is_summary = true;
  if (isMapping(value)) {
    const counts: Record<string, number> = {};
    let any = false;
    for (const key of Object.keys(value)) {
      if (Array.isArray(value[key])) {
        counts[key] = value[key].length;
        any = true;
      }
    }
    if (any) shortfall.item_counts = counts;
  }
  return shortfall;
}
