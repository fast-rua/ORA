// `ora/agent/memory/artifact_openers.py`: one advertised spelling for Context
// Artifact recovery.

import { IMAGE_OPEN_TOOL } from "../domain_defaults.ts";
import { PyValueError, clonePy, get, hasKey, isMapping } from "../pyutil.ts";

export const ARTIFACT_READ_NAMES = new Set(["artifact.read", "artifact_read"]);

/** Python ``str(reference.get("media_type", ""))``. */
function mediaTypeText(reference: Record<string, any>): string {
  if (!hasKey(reference, "media_type")) return "";
  const value = reference.media_type;
  if (value === null || value === undefined) return "None";
  return typeof value === "string" ? value : String(value);
}

/** Use the catalog's delivered decoder; accept legacy unannotated refs. */
export function recoverableArtifactOpen(reference: Record<string, any>): {
  tool: string; arguments: { ref: unknown };
} {
  const image = mediaTypeText(reference).startsWith("image/");
  const opener = get(reference, "open");
  let name: unknown = isMapping(opener) ? get(opener, "tool") : undefined;
  if (name === undefined || name === null) name = image ? IMAGE_OPEN_TOOL : "artifact.read";
  const allowed = image ? new Set([IMAGE_OPEN_TOOL]) : ARTIFACT_READ_NAMES;
  if (typeof name !== "string" || !allowed.has(name)) {
    throw new PyValueError("recoverable Artifact decoder does not match media type");
  }
  if (!hasKey(reference, "ref")) throw new PyValueError("'ref'");
  return { tool: name, arguments: { ref: clonePy(reference.ref) } };
}

/** The spelling this surface would give the reader, had it composed it. */
export function artifactReadName(
  definitions: Record<string, any>[], canonicalToolNames: Iterable<string>,
): string {
  const canonical = new Set(canonicalToolNames);
  for (const item of definitions) {
    const raw = get(item, "name");
    const name = raw === undefined || raw === null ? "" : String(raw);
    if (name && canonical.has(name)) return "artifact_read";
  }
  return "artifact.read";
}

const PUBLIC_KEYS = ["schema", "ref", "sha256", "kind", "media_type", "bytes", "policy_scope"];

export function publicArtifactRef(
  entry: Record<string, any>,
  options: { readerName: string; roundIndex?: number | null; actionStep?: number | null },
): Record<string, any> {
  const reference: Record<string, any> = {};
  for (const key of PUBLIC_KEYS) {
    const value = get(entry, key);
    reference[key] = value === undefined ? null : clonePy(value);
  }
  reference.round = options.roundIndex ?? null;
  reference.action_step = options.actionStep ?? null;
  if (options.readerName !== "artifact.read") {
    const mediaType = get(entry, "media_type");
    const text = mediaType === undefined ? "" : mediaType === null ? "None" : String(mediaType);
    reference.open = {
      tool: text.startsWith("image/") ? IMAGE_OPEN_TOOL : options.readerName,
      arguments: { ref: reference.ref },
    };
  }
  return reference;
}
