// L1/L2's render: large read-only results, then later exact duplicates, of the
// ActionSteps before the frontier become recoverable references (ported from
// the Python Native pipeline's lossless render).

import type { ContextDomain } from "../../domain_defaults.ts";
import { ARTIFACT_READ_NAMES } from "../../memory/artifacts.ts";
import { type Message, externalizeToolResults, messageEventIds } from "../../projection.ts";
import { type Measured, activeSizer, measuredPassRecord } from "../../request_size.ts";

export interface EagerPasses {
  history: Message[];
  size: Measured;
  records: Record<string, any>[];
  refs: Record<string, any>[];
}

export function eagerPasses(input: Message[], inputSize: Measured, options: {
  through: number; mutationTools: string[]; largeResultBytes: number; previewBytes: number; domain: ContextDomain;
}): EagerPasses {
  const sizer = activeSizer();
  const { domain } = options;
  let [working, largeIndexes, largeRefs] = externalizeToolResults(input, {
    throughActionSteps: options.through, mutationTools: options.mutationTools,
    selector: "large", thresholdBytes: options.largeResultBytes, previewBytes: options.previewBytes,
    excludeTools: ARTIFACT_READ_NAMES, domain,
  });
  let after = sizer.measure(working);
  const largeRecord = measuredPassRecord("l1_large_result_reference", inputSize, after, {
    changed_messages: largeIndexes, artifact_refs: largeRefs,
    externalized_event_ids: messageEventIds(working, largeIndexes),
  });
  const [dedupWorking, duplicateIndexes, duplicateRefs] = externalizeToolResults(working, {
    throughActionSteps: options.through, mutationTools: options.mutationTools,
    selector: "duplicate", thresholdBytes: options.largeResultBytes, previewBytes: 0,
    excludeTools: ARTIFACT_READ_NAMES, domain,
  });
  working = dedupWorking;
  const before = after;
  after = sizer.measure(working);
  const duplicateRecord = measuredPassRecord("l2_exact_duplicate_reference", before, after, {
    changed_messages: duplicateIndexes, artifact_refs: duplicateRefs,
    externalized_event_ids: messageEventIds(working, duplicateIndexes),
  });
  return { history: working, size: after, records: [largeRecord, duplicateRecord], refs: [...largeRefs, ...duplicateRefs] };
}
