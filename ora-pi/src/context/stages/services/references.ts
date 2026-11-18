// The references service: whether and how the model is shown what left the
// request (recoverable references, design-space D1 and D4), and the marker
// codec every stage writes and reads them with.
//
// A scheme's `references` setting is the one place that decides whether
// recoverable-artifact Pointers are shown at all (v8 shows none), which tool
// results advertise their Pointer when first delivered, and the preview an
// externalized result keeps. The preview strategies are the head (the first
// bytes, L1-L4) and the head and tail (`headTailPreview`, exchange_aging: a
// script prints its conclusion last).

import type { ContextDomain } from "../../domain_defaults.ts";
import {
  RECOVERABLE_ARTIFACT_TAG, decodeRecoverableArtifactMarker, encodeRecoverableArtifactMarker,
  recoverableArtifactPointer,
} from "../../memory/artifacts.ts";
import type { Message } from "../../projection.ts";
import { previewShortfall } from "../../preview_shortfall.ts";
import {
  clonePy, dumpsCompact, dumpsPy, get, hasKey, isMapping, tryLoads, utf8, utf8Len, utf8PrefixIgnore,
} from "../../pyutil.ts";

export interface ReferenceSettings {
  /** Whether the model is shown recoverable-artifact Pointers (summary slots name their transcript). */
  pointers: boolean;
  /** A tool result larger than this advertises its Pointer when first delivered; null: never. */
  advertise_over_bytes: number | null;
  /** The preview an externalized result keeps: its first `head_bytes` bytes (D1). */
  preview: { head_bytes: number };
}

/** A scheme that shows no Pointer (v8): nothing advertised, no preview. */
export const NO_POINTERS: ReferenceSettings = { pointers: false, advertise_over_bytes: null, preview: { head_bytes: 0 } };

/** A ToolSurface's declared semantic summary is kept whole as the preview up to this size. */
export const SUMMARY_PREVIEW_MAX_BYTES = 16384;

// -- the marker codec ------------------------------------------------------------

export { RECOVERABLE_ARTIFACT_TAG, decodeRecoverableArtifactMarker, encodeRecoverableArtifactMarker };

const MARKER_PREFIX = `<${RECOVERABLE_ARTIFACT_TAG}>`;

/** Whether a content block is a recoverable-artifact marker (materialization puts one before each image). */
export function isRecoverableMarker(block: unknown): boolean {
  return isMapping(block) && get(block, "type") === "text"
    && typeof get(block, "text") === "string" && String(block.text).startsWith(MARKER_PREFIX);
}

/**
 * A copy of ``message`` without the recoverable-artifact markers of its inline
 * images; null when it has none. Materialization puts each marker right before
 * its image, so a history that never removes an image (v8) loses every marker;
 * an image an exchange aging removed leaves its marker without an image, and
 * that marker is the reference the model is shown.
 */
export function withoutRecoverableMarkers(message: Message): Message | null {
  const content = get(message, "content");
  if (!Array.isArray(content)) return null;
  const inlineMarker = (block: unknown, index: number): boolean => isRecoverableMarker(block)
    && isMapping(content[index + 1]) && get(content[index + 1], "type") === "image_url";
  if (!content.some(inlineMarker)) return null;
  return { ...message, content: content.filter((block, index) => !inlineMarker(block, index)) };
}

/** Python ``str(message.get("_context_original_content", message.get("content", "")))``. */
export function originalContentText(message: Message): string {
  const value = hasKey(message, "_context_original_content")
    ? message._context_original_content
    : hasKey(message, "content") ? message.content : "";
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "None";
  return dumpsPy(value);
}

// -- previews ---------------------------------------------------------------------

/** The status fields a structured reference keeps, besides the domain's marker status keys. */
const STRUCTURED_FIELDS = ["ok", "status", "exit_code", "actions_executed"];
/** The last line of a failed command's output that names an error: ``Type: message``. */
const ERROR_LINE = /^([A-Za-z_][\w.]*): (.*)$/;
const ERROR_MESSAGE_CHARS = 300;

/** A failed result's error: its own `error` field, else the last ``Type: message`` line of its output. */
function resultError(value: Record<string, any>): Record<string, string> | null {
  const error = get(value, "error");
  if (isMapping(error)) {
    const type = get(error, "type") ?? get(error, "code");
    const message = get(error, "message");
    return {
      ...(typeof type === "string" ? { type } : {}),
      ...(typeof message === "string" ? { message: message.slice(0, ERROR_MESSAGE_CHARS) } : {}),
    };
  }
  if (typeof error === "string") return { message: error.slice(0, ERROR_MESSAGE_CHARS) };
  if (get(value, "ok") !== false) return null;
  const output = get(value, "output");
  if (typeof output !== "string") return null;
  const lines = output.split("\n").map((line) => line.trim()).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const match = lines[index].match(ERROR_LINE);
    if (match) return { type: match[1], message: match[2].slice(0, ERROR_MESSAGE_CHARS) };
  }
  return null;
}

/**
 * The reference an aged result leaves when it keeps its structured fields
 * (exchange_aging `result_preview: structured`): whether it succeeded, its
 * status, exit code and the actions it executed, the error type and message
 * of a failed one, its size and its Pointer. No text of its output.
 */
export function structuredToolMarker(
  message: Message, reference: Record<string, any>, domain: ContextDomain,
): string {
  const content = originalContentText(message);
  const decoded = tryLoads(content);
  const value = decoded.ok ? decoded.value : null;
  const fields: Record<string, any> = {};
  if (isMapping(value)) {
    for (const key of [...STRUCTURED_FIELDS, ...domain.markerStatusKeys]) {
      if (hasKey(value, key) && !hasKey(fields, key)) fields[key] = clonePy(value[key]);
    }
    const error = resultError(value);
    if (error !== null) fields.error = error;
  }
  return dumpsCompact({
    ...fields,
    result_bytes: utf8Len(content),
    recoverable_artifact: recoverableArtifactPointer(reference),
    externalized: true,
  });
}

/** A head-and-tail preview: the first ``head_bytes`` and the last ``tail_bytes`` UTF-8 bytes. */
export interface HeadTailPreview {
  head_bytes: number;
  tail_bytes: number;
}

const ELISION = /\n\[\.\.\. (\d+) bytes cut \.\.\.\]\n/g;

function continuationByte(byte: number | undefined): boolean {
  return byte !== undefined && (byte & 0xc0) === 0x80;
}

/**
 * ``text`` cut to its first ``head_bytes`` and last ``tail_bytes`` bytes, never
 * splitting a character, with an elision that says how many bytes lie between
 * them; text within the two is kept whole. Returns the bytes of ``text`` it keeps.
 */
export function headTailPreview(text: string, cut: HeadTailPreview): { preview: string; keptBytes: number } {
  const data = utf8(text);
  if (data.length <= cut.head_bytes + cut.tail_bytes) return { preview: text, keptBytes: data.length };
  let end = Math.max(0, cut.head_bytes);
  while (end > 0 && continuationByte(data[end])) end -= 1;  // a continuation byte at ``end``: its character was cut
  let start = data.length - cut.tail_bytes;
  while (start < data.length && continuationByte(data[start])) start += 1;
  const head = data.subarray(0, end);
  const tail = data.subarray(start);
  return {
    preview: `${head.toString("utf8")}\n[... ${start - end} bytes cut ...]\n${tail.toString("utf8")}`,
    keptBytes: head.length + tail.length,
  };
}

/** Whether ``preview`` is ``original`` whole, or its head and tail around an elision of exactly the bytes between. */
export function isHeadTailPreview(original: string, preview: string): boolean {
  if (preview === original) return true;
  const total = utf8Len(original);
  for (const match of preview.matchAll(ELISION)) {
    const head = preview.slice(0, match.index);
    const tail = preview.slice(match.index! + match[0].length);
    if (original.startsWith(head) && original.endsWith(tail)
      && utf8Len(head) + Number(match[1]) + utf8Len(tail) === total) return true;
  }
  return false;
}

/**
 * The reference an externalized tool result leaves: status keys, its preview
 * and its Pointer. ``preview`` is the head's bytes, or a head-and-tail preview.
 */
export function recoverableToolMarker(
  message: Message, reference: Record<string, any>, preview: number | HeadTailPreview, domain: ContextDomain,
): string {
  const content = originalContentText(message);
  const cut = typeof preview === "number" ? null : headTailPreview(content, preview);
  const budget = typeof preview === "number" ? preview : preview.head_bytes + preview.tail_bytes;
  let text = cut === null ? utf8PrefixIgnore(content, budget) : cut.preview;
  const status: Record<string, any> = {};
  const decoded = tryLoads(content);
  const value = decoded.ok ? decoded.value : null;
  if (isMapping(value)) {
    // A ToolSurface may declare a bounded semantic summary; keep all of it.
    const summary = get(value, "summary");
    if (budget > 0 && (isMapping(summary) || Array.isArray(summary))) {
      const summaryText = dumpsCompact(summary);
      if (utf8Len(summaryText) <= SUMMARY_PREVIEW_MAX_BYTES) text = summaryText;
    }
    for (const key of domain.markerStatusKeys) {
      if (hasKey(value, key)) status[key] = clonePy(value[key]);
    }
  }
  const pointer = recoverableArtifactPointer(reference);
  return dumpsCompact({
    ...status,
    ...previewShortfall(content, text, value, cut !== null && text === cut.preview ? cut.keptBytes : null),
    recoverable_artifact: pointer,
    preview: text,
    externalized: true,
  });
}
