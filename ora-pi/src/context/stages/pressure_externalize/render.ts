// L4's render: every remaining read-only result and older observation body of
// the ActionSteps before the frontier becomes a recoverable reference (ported
// from the Python Native pipeline's lossless render).

import type { ContextDomain } from "../../domain_defaults.ts";
import * as historicalObservation from "../../historical_observation.ts";
import { type Message, eligibleMessageIndexes, externalizeToolResults, messageEventIds } from "../../projection.ts";
import { type Measured, activeSizer, measuredPassRecord } from "../../request_size.ts";

export interface PressurePasses {
  history: Message[];
  size: Measured;
  records: Record<string, any>[];
  refs: Record<string, any>[];
}

export function pressurePasses(input: Message[], inputSize: Measured, options: {
  through: number; mutationTools: string[]; previewBytes: number; domain: ContextDomain;
}): PressurePasses {
  const sizer = activeSizer();
  const { domain } = options;
  const [remainingWorking, remainingIndexes, remainingRefs] = externalizeToolResults(input, {
    throughActionSteps: options.through, mutationTools: options.mutationTools,
    selector: "remaining", thresholdBytes: 0, previewBytes: options.previewBytes,
    domain,
  });
  let working = remainingWorking;
  let after = sizer.measure(working);
  const readOnlyRecord = measuredPassRecord("l4_read_only_result_reference", inputSize, after, {
    changed_messages: remainingIndexes, artifact_refs: remainingRefs,
    externalized_event_ids: messageEventIds(working, remainingIndexes),
  });
  const [observationWorking, observationIndexes, observationRefs] = historicalObservation.externalize(
    working, eligibleMessageIndexes(working, options.through).values(), domain,
  );
  working = observationWorking;
  const before = after;
  after = sizer.measure(working);
  const observationRecord = measuredPassRecord("l4_observation_reference", before, after, {
    changed_messages: observationIndexes, artifact_refs: observationRefs,
    externalized_event_ids: messageEventIds(working, observationIndexes),
  });
  return {
    history: working, size: after, records: [readOnlyRecord, observationRecord],
    refs: [...remainingRefs, ...observationRefs],
  };
}
