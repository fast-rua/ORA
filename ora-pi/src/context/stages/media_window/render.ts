// L3's render: media of the ActionSteps before the frontier (and of superseded
// tool-image events) become recoverable references, and an earlier round's
// static-facts slot ages with its round's media (ported from the Python Native
// pipeline's lossless render).

import { type Message, externalizeHistoricalMedia, messageEventIds } from "../../projection.ts";
import { sortedNumbers } from "../../pyutil.ts";
import { type Measured, activeSizer, measuredPassRecord } from "../../request_size.ts";
import * as roundStaticFacts from "../../round_static_facts.ts";

export interface MediaPass {
  history: Message[];
  size: Measured;
  record: Record<string, any>;
  refs: Record<string, any>[];
}

export function mediaPass(input: Message[], inputSize: Measured, options: {
  through: number; supersededToolImageEventIds: string[];
}): MediaPass {
  let [mediaWorking, mediaIndexes, mediaRefs] = externalizeHistoricalMedia(
    input, options.through, options.supersededToolImageEventIds,
  );
  // native_slots_v4: an earlier round's static-facts slot ages with its
  // round's media; the current round's slot never does.
  const [slotWorking, slotIndexes, slotRefs] = roundStaticFacts.externalizePriorRounds(mediaWorking, options.through);
  mediaWorking = slotWorking;
  if (slotIndexes.length) {
    mediaIndexes = sortedNumbers(new Set([...mediaIndexes, ...slotIndexes]));
    mediaRefs = [...mediaRefs, ...slotRefs];
  }
  const mediaAfter = activeSizer().measure(mediaWorking);
  const record = measuredPassRecord("l3_lifecycle_media_reference", inputSize, mediaAfter, {
    changed_messages: mediaIndexes, artifact_refs: mediaRefs,
    externalized_event_ids: messageEventIds(mediaWorking, mediaIndexes),
  });
  return { history: mediaWorking, size: mediaAfter, record, refs: mediaRefs };
}
