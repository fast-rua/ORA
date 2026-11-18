// Request identity fields of a lossless projection manifest: the request, its
// stable and protected prefixes, the tool schemas, the history and the static
// context sources, each by sha256 (moved from native.ts).

import { type Message, providerMessage } from "../projection.ts";
import { canonicalLength, canonicalSha256, get, hasKey } from "../pyutil.ts";

/** The context sources of the fixed prefix: they never age. */
export const STATIC_PROMPT_SOURCES = new Set(["harness_system", "project_instructions", "episode_task"]);

/** Prefix hashes, protected sources and history of one projected request. */
export function requestIdentityParts(
  providerMessages: Message[], working: Message[], tools: Record<string, any>[],
  isStaticProtected: (message: Message) => boolean,
): Record<string, any> {
  const protectedIndexes = new Set<number>();
  working.forEach((message, index) => {
    if (get(message, "_protected_context") === true) protectedIndexes.add(index);
  });
  const stablePrefix: Message[] = [];
  for (const message of providerMessages) {
    if (get(message, "role") !== "system") break;
    stablePrefix.push(message);
  }
  const protectedMessages = [...protectedIndexes].sort((a, b) => a - b)
    .filter((index) => isStaticProtected(working[index]))
    .map((index) => providerMessages[index]);
  const sourceMessages: Record<string, Message> = {};
  for (const message of working) {
    const source = get(message, "_context_source");
    if (typeof source === "string" && get(message, "_rollout_checkpoint") !== true) {
      sourceMessages[source] = providerMessage(message);
    }
  }
  return {
    stable_prefix_hash: canonicalSha256({ messages: stablePrefix, tools: [...tools] }),
    protected_prefix_sha256: canonicalSha256({ messages: protectedMessages, tools: [...tools] }),
    protected_prefix_message_count: protectedMessages.length,
    source_messages: sourceMessages,
    history_messages: providerMessages.filter((_message, index) => !protectedIndexes.has(index)),
  };
}

export function sourceSha256(sourceMessages: Record<string, Message>, name: string): string | null {
  return hasKey(sourceMessages, name) ? canonicalSha256(sourceMessages[name]) : null;
}

/** Whether ``message`` belongs to the fixed prefix (a protected static context source). */
export function isFixedPrefix(message: Message): boolean {
  return get(message, "_protected_context") === true && STATIC_PROMPT_SOURCES.has(get(message, "_context_source"));
}

/** Request identity fields of a lossless manifest, in manifest order. */
export function losslessRequestIdentity(request: {
  providerMessages: Message[]; working: Message[]; tools: Record<string, any>[]; requestView: Record<string, any>;
}): Record<string, unknown> {
  const parts = requestIdentityParts(
    request.providerMessages, request.working, request.tools,
    (message) => STATIC_PROMPT_SOURCES.has(get(message, "_context_source")),
  );
  const sourceMessages = parts.source_messages;
  const historyMessages = parts.history_messages;
  return {
    request_sha256: canonicalSha256(request.requestView),
    stable_prefix_hash: parts.stable_prefix_hash,
    protected_prefix_sha256: parts.protected_prefix_sha256,
    protected_prefix_message_count: parts.protected_prefix_message_count,
    tool_schema_sha256: canonicalSha256([...request.tools]),
    history_sha256: canonicalSha256(historyMessages),
    history_utf8_bytes: canonicalLength(historyMessages),
    stable_system_sha256: canonicalSha256(hasKey(sourceMessages, "harness_system") ? sourceMessages.harness_system : []),
    project_context_sha256: sourceSha256(sourceMessages, "project_instructions"),
    episode_task_sha256: sourceSha256(sourceMessages, "episode_task"),
  };
}
