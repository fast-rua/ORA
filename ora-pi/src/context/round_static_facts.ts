// The Context-lifecycle half of `ora/agent/context/delivery/round_static_facts.py`
// (`round_static_facts_slot_v1`, composition `native_slots_v4`).
//
// The slot is `_protected_context`. Tail aging (L1-L4) never rewrites the
// current round's slot. An earlier round's slot is externalized to a
// recoverable pointer when its round's media are released (L3), and an
// emergency checkpoint (L6) that absorbs the round's first step carries the
// current round's slot forward. Building and reducing slots stays in Python.

import type { InlineMediaCache } from "./inline_media.ts";
import type { ContextArtifactCatalog } from "./memory/artifacts.ts";
import { recoverableArtifactPointer } from "./memory/artifacts.ts";
import {
  PolicySessionError, clonePy, dumpsCompact, eq, get, isMapping, pyEqual, sha256Hex, utf8, utf8Len,
} from "./pyutil.ts";

export const SLOT_KEY = "round_static_facts";
export const MESSAGE_FLAG = "_round_static_facts";
export const REF_KEY = "_round_static_facts_ref";
export const ORIGINAL_KEY = "_round_static_facts_original";
export const EXTERNALIZED_KEY = "_lossless_round_static_facts_externalized";
export const ARTIFACT_KIND = "context_round_static_facts";

export function isSlot(message: Record<string, any>): boolean {
  return isMapping(get(message, MESSAGE_FLAG));
}

export function slotRound(message: Record<string, any>): unknown {
  return isSlot(message) ? get(message[MESSAGE_FLAG], "round") ?? null : null;
}

/** Store a slot's exact text as a recoverable Artifact (lossless aging). */
export function materialize(
  message: Record<string, any>,
  options: { catalog: ContextArtifactCatalog | InlineMediaCache; roundIndex: number | null; actionStep: number | null },
): Record<string, any> | null {
  if (!isSlot(message)) return null;
  const content = get(message, "content");
  if (!(Array.isArray(content) && content.length === 1 && isMapping(content[0])
    && typeof get(content[0], "text") === "string")) {
    throw new PolicySessionError("round static facts slot is malformed");
  }
  const original: string = content[0].text;
  const ref = options.catalog.putBytes(utf8(original), {
    kind: ARTIFACT_KIND, suffix: ".json", mediaType: "application/json",
    roundIndex: options.roundIndex, actionStep: options.actionStep,
  });
  message[REF_KEY] = ref;
  message[ORIGINAL_KEY] = original;
  return ref;
}

function marker(message: Record<string, any>): string {
  return dumpsCompact({
    [SLOT_KEY]: { round: slotRound(message) },
    recoverable_round_static_facts: recoverableArtifactPointer(message[REF_KEY]),
    round_static_facts_externalized: true,
  });
}

function isObservation(message: Record<string, any>): boolean {
  return get(message, "role") === "user" && isMapping(get(message, "_observation"));
}

function currentRound(messages: Record<string, any>[]): unknown {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (isObservation(messages[index])) return get(messages[index], "_session_round") ?? null;
  }
  return null;
}

/** L3: a slot of an earlier round whose step's media aged becomes a pointer. */
export function externalizePriorRounds(
  messages: Record<string, any>[], throughActionSteps: number,
): [Record<string, any>[], number[], Record<string, any>[]] {
  const output = [...messages];
  const round = currentRound(output);
  const changed: number[] = [];
  const refs: Record<string, any>[] = [];
  let actionStep = -1;
  output.forEach((message, index) => {
    if (isObservation(message)) actionStep += 1;
    if (!isSlot(message) || !(0 <= actionStep && actionStep < throughActionSteps)
      || eq(slotRound(message), round) || !isMapping(get(message, REF_KEY))) {
      return;
    }
    const replaced = { ...message };
    replaced.content = [{ type: "text", text: marker(message) }];
    replaced[EXTERNALIZED_KEY] = true;
    output[index] = replaced;
    changed.push(index);
    refs.push(clonePy({ ...message[REF_KEY] }));
  });
  return [output, changed, refs];
}

/** A slot's original content, after checking its current text is exact. */
export function invariantContent(message: Record<string, any>): Record<string, any>[] | null {
  const ref = get(message, REF_KEY);
  if (!isSlot(message) || !isMapping(ref)) return null;
  const original = get(message, ORIGINAL_KEY);
  if (typeof original !== "string" || sha256Hex(utf8(original)) !== get(ref, "sha256")
    || !pyEqual(utf8Len(original), get(ref, "bytes") ?? null)) {
    throw new PolicySessionError("round static facts Artifact identity drift");
  }
  const expected = get(message, EXTERNALIZED_KEY) === true ? marker(message) : original;
  if (!pyEqual(get(message, "content") ?? null, [{ type: "text", text: expected }])) {
    throw new PolicySessionError("round static facts differ from their recoverable source");
  }
  return [{ type: "text", text: original }];
}

/** L6: the current round's slot, carried out of a summarized span. */
export function currentRoundSlots(absorbed: Record<string, any>[], round: unknown): Record<string, any>[] {
  return absorbed.filter((message) => isSlot(message) && eq(slotRound(message), round))
    .map((message) => clonePy({ ...message }));
}
