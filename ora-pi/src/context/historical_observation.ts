// `ora/agent/context/observation/historical.py`: recover old public observation
// bodies without rewriting causal outcomes.

import type { ContextDomain } from "./domain_defaults.ts";
import type { InlineMediaCache } from "./inline_media.ts";
import type { ContextArtifactCatalog } from "./memory/artifacts.ts";
import { recoverableArtifactPointer } from "./memory/artifacts.ts";
import {
  PolicySessionError, clonePy, dumpsCompact, get, hasKey, isMapping, pyEqual, sha256Hex, tryLoads,
  utf8, utf8Len,
} from "./pyutil.ts";

/** Store only the already public JSON envelope, never observation metadata. */
export function materialize(
  message: Record<string, any>,
  options: {
    catalog: ContextArtifactCatalog | InlineMediaCache; roundIndex: number | null;
    actionStep: number | null; domain: ContextDomain;
  },
): Record<string, any> | null {
  const content = get(message, "content");
  if (!(
    get(message, "role") === "user"
    && isMapping(get(message, "_observation"))
    && Array.isArray(content) && content.length
    && isMapping(content[0]) && get(content[0], "type") === "text"
  )) {
    return null;
  }
  const original = get(content[0], "text");
  if (typeof original !== "string") return null;
  const body = tryLoads(original);
  if (!body.ok) return null;
  if (!isMapping(body.value) || !isMapping(get(body.value, options.domain.observationBodyKey))) return null;
  const ref = options.catalog.putBytes(utf8(original), {
    kind: "context_observation", suffix: ".json", mediaType: "application/json",
    roundIndex: options.roundIndex, actionStep: options.actionStep,
  });
  message._context_observation_ref = ref;
  message._context_observation_original = original;
  return ref;
}

function marker(original: string, ref: Record<string, any>, domain: ContextDomain): string {
  const body = tryLoads(original);
  if (!body.ok) throw new PolicySessionError("observation body is not JSON");
  const value = body.value as Record<string, any>;
  // Keep comparison, last_action_effect, terminal verdict and lifecycle notices
  // verbatim; only the large measured body is replaced by its pointer.
  const actual = value[domain.observationBodyKey];
  const kept: Record<string, any> = {};
  for (const key of domain.observationIdentityKeys) {
    if (hasKey(actual, key)) kept[key] = clonePy(actual[key]);
  }
  value[domain.observationBodyKey] = kept;
  value.recoverable_observation = recoverableArtifactPointer(ref);
  value.observation_externalized = true;
  return dumpsCompact(value);
}

export function externalize(
  messages: Record<string, any>[], eligibleIndexes: Iterable<number>, domain: ContextDomain,
): [Record<string, any>[], number[], Record<string, any>[]] {
  // Copy-on-write: only a replaced message (and its content list) is copied.
  const output = [...messages];
  const changed: number[] = [];
  const refs: Record<string, any>[] = [];
  for (const index of [...eligibleIndexes].sort((a, b) => a - b)) {
    let message = output[index];
    const ref = get(message, "_context_observation_ref");
    const original = get(message, "_context_observation_original");
    if (!isMapping(ref) || typeof original !== "string") continue;
    const replacement = marker(original, ref, domain);
    if (utf8Len(replacement) >= utf8Len(original)) continue;
    message = { ...message };
    message.content = [...message.content];
    output[index] = message;
    message.content[0] = { type: "text", text: replacement };
    message._lossless_observation_externalized = true;
    changed.push(index);
    refs.push(clonePy({ ...ref }));
  }
  return [output, changed, refs];
}

/** Verify the original bytes or exact replacement before normalizing. */
export function invariantBlock(message: Record<string, any>, domain: ContextDomain): Record<string, any> | null {
  const ref = get(message, "_context_observation_ref");
  if (!isMapping(ref)) return null;
  const original = get(message, "_context_observation_original");
  if (typeof original !== "string" || sha256Hex(utf8(original)) !== get(ref, "sha256")
    || !pyEqual(utf8Len(original), get(ref, "bytes") ?? null)) {
    throw new PolicySessionError("observation Artifact identity drift");
  }
  const expected = get(message, "_lossless_observation_externalized") === true
    ? marker(original, ref, domain) : original;
  if (!pyEqual(message.content[0], { type: "text", text: expected })) {
    throw new PolicySessionError("observation body differs from recoverable source");
  }
  return { type: "text", text: original };
}
