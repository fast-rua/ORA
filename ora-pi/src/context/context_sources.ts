// Ported from `ora/agent/native/context_sources.py` of the retired Python Native
// core (robot-harness repository): context-source records a typed-slot
// projection carries into its manifest (which slot every request byte came from).

import { canonicalBytesPy, get, num, sha256Hex, tryLoads, isMapping } from "./pyutil.ts";

export const NATIVE_SLOTS_V3 = "native_slots_v3";
export const NATIVE_SLOTS_V4 = "native_slots_v4";
export const NATIVE_CAUSAL_SLOTS = new Set([NATIVE_SLOTS_V3, NATIVE_SLOTS_V4]);
export const TOOL_IMAGE_CALL_LABELS = "tool_image_call_labels_v1";
export const COMPOSITION_CONTEXT_CAPABILITIES: Record<string, string[]> = {
  [NATIVE_SLOTS_V4]: ["round_static_facts_slot_v1", TOOL_IMAGE_CALL_LABELS],
};
export const OPERATOR_MESSAGE_PREFIX = "<operator_message>\n";
const CHECKPOINT_PREFIX = "<rollout_checkpoint>\n";

/** Add the per-request source records to ``projected.manifest`` in place. */
export function annotateContextSources(
  projected: Record<string, any>,
  options: {
    composition: string | null; observation: Record<string, any>;
    toolSchemas: Record<string, any>[]; contextSources: Record<string, any>[];
  },
): void {
  if (options.composition !== null && NATIVE_CAUSAL_SLOTS.has(options.composition)) {
    annotateV3(projected, options.observation, options.toolSchemas, options.contextSources);
    if (Object.prototype.hasOwnProperty.call(COMPOSITION_CONTEXT_CAPABILITIES, options.composition)) {
      annotateRoundStaticFacts(projected);
    }
    annotateOperatorMessages(projected);
  }
}

function nextOrder(records: Record<string, any>[]): number {
  return Math.max(...records.map((item) => Math.trunc(num(item.order)))) + 1;
}

function annotateOperatorMessages(projected: Record<string, any>): void {
  const records: Record<string, any>[] = projected.manifest.context_sources;
  let order = nextOrder(records);
  for (const message of projected.messages) {
    const content = get(message, "content");
    if (get(message, "role") === "user" && typeof content === "string" && content.startsWith(OPERATOR_MESSAGE_PREFIX)) {
      const data = canonicalBytesPy(message);
      records.push({
        kind: "operator_message", role: "user", order, status: "protected_human_guidance",
        bytes: data.length, sha256: sha256Hex(data),
      });
      order += 1;
    }
  }
}

function roundStaticFactsSlot(message: Record<string, any>): boolean {
  const content = get(message, "content");
  if (get(message, "role") !== "user" || !Array.isArray(content) || content.length !== 1) return false;
  const text = isMapping(content[0]) ? get(content[0], "text") : undefined;
  let body: unknown = null;
  if (typeof text === "string") {
    const decoded = tryLoads(text);
    if (!decoded.ok) return false;
    body = decoded.value;
  }
  const slot = isMapping(body) ? get(body, "round_static_facts") : undefined;
  return isMapping(slot) && Object.prototype.hasOwnProperty.call(slot, "current_actual");
}

/** ``native_slots_v4``: the protected round slots the history hash leaves out. */
function annotateRoundStaticFacts(projected: Record<string, any>): void {
  const records: Record<string, any>[] = projected.manifest.context_sources;
  let order = nextOrder(records);
  for (const message of projected.messages) {
    if (roundStaticFactsSlot(message)) {
      const data = canonicalBytesPy(message);
      records.push({
        kind: "round_static_facts", role: "user", order, status: "protected_round_slot",
        bytes: data.length, sha256: sha256Hex(data),
      });
      order += 1;
    }
  }
}

function rolloutCheckpointMessage(messages: Record<string, any>[]): Record<string, any> | null {
  return messages.find((message) => get(message, "role") === "user"
    && typeof get(message, "content") === "string" && message.content.startsWith(CHECKPOINT_PREFIX)) ?? null;
}

function annotateV3(
  projected: Record<string, any>, observation: Record<string, any>, toolSchemas: Record<string, any>[],
  contextSources: Record<string, any>[],
): void {
  const causalBytes = canonicalBytesPy(observation._causal_observation);
  const toolBytes = canonicalBytesPy(toolSchemas);
  const sourceRecords = contextSources.map((item, index) => ({ item, index }))
    .sort((a, b) => Math.trunc(num(a.item.order)) - Math.trunc(num(b.item.order)) || a.index - b.index)
    .map(({ item }) => item);
  // The optional checkpoint shifts every later order by one.
  const checkpointMessage = rolloutCheckpointMessage(projected.messages);
  if (checkpointMessage !== null) {
    const checkpointBytes = canonicalBytesPy(checkpointMessage);
    sourceRecords.push({
      kind: "rollout_context_checkpoint", role: "user", order: 3, status: "protected_current",
      bytes: checkpointBytes.length, sha256: sha256Hex(checkpointBytes),
    });
  }
  const dynamicOrder = 3 + (checkpointMessage !== null ? 1 : 0);
  sourceRecords.push(
    {
      kind: "history_projection", role: "mixed", order: dynamicOrder, status: "projected",
      bytes: projected.manifest.history_utf8_bytes, sha256: projected.manifest.history_sha256,
    },
    {
      kind: "causal_observation", role: "user", order: dynamicOrder + 1, status: "current_allowlisted",
      bytes: causalBytes.length, sha256: sha256Hex(causalBytes),
    },
    {
      kind: "tool_schemas", role: "provider_field", order: dynamicOrder + 2, status: "canonical",
      bytes: toolBytes.length, sha256: sha256Hex(toolBytes),
    },
  );
  projected.manifest.context_sources = sourceRecords;
  projected.manifest.episode_task_protected = true;
}
