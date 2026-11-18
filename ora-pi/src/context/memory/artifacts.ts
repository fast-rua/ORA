// `ora/agent/memory/artifacts.py`: Session-scoped, policy-readable references
// for recoverable Context payloads, and the `artifact_read` / `observe(ref)`
// readers over them.

import * as fs from "node:fs";
import * as path from "node:path";

import { ArtifactStore, type StorageRef } from "../../evidence/artifacts.ts";
import { IMAGE_OPEN_TOOL } from "../domain_defaults.ts";
import {
  PyValueError, canonicalBytesPy, clonePy, dumpsCompact, eq, get, hasKey, isInt, isMapping,
  loadsPy, num, pyEqual, pyFloat, pyRound, sha256Hex, sortedStrings, tryLoads, PyJsonDecodeError,
} from "../pyutil.ts";
import {
  ARTIFACT_READ_NAMES, artifactReadName, publicArtifactRef, recoverableArtifactOpen,
} from "./artifact_openers.ts";
import {
  JsonPointerError, RefError, boundedPointerValue, checkObserveRefArguments, checkReadArguments,
  jsonPointer, policyResolve, readRange,
} from "./artifact_reader.ts";
import { atomicJson, BatchedIndexPersistence } from "./index_persistence.ts";
import { validateSchemaInstance } from "./schema_validation.ts";
import { ToolError } from "./tool_errors.ts";

export { ARTIFACT_READ_NAMES, artifactReadName, recoverableArtifactOpen, publicArtifactRef };

export const LOSSLESS_CONTEXT_RETENTION = "action_step_lossless_v4";
export const CONTEXT_ARTIFACT_INDEX_SCHEMA = "ora.context-artifact-index.v1";
export const CONTEXT_ARTIFACT_REF_SCHEMA = "ora.context-artifact-ref.v1";
export const RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA = "ora.recoverable-artifact-envelope.v1";
export const RECOVERABLE_ARTIFACT_POINTER_SCHEMA = "ora.recoverable-artifact-pointer.v1";
export const RECOVERABLE_ARTIFACT_TAG = "recoverable_artifact";
export const ARTIFACT_READ_LIMIT_BYTES = 8192;
const REF_PATTERN = /^artifact:\/\/([A-Za-z0-9][A-Za-z0-9._-]{0,159})\/([0-9a-f]{64})$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export function refMatch(ref: string): { scope: string; sha256: string } | null {
  const match = REF_PATTERN.exec(ref);
  return match ? { scope: match[1], sha256: match[2] } : null;
}

function sameKeys(value: Record<string, any>, keys: string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => hasKey(value, key));
}

function isByteCount(value: unknown): boolean {
  return isInt(value) && num(value) >= 0;
}

function openerNames(mediaType: string): Set<string> {
  return mediaType.startsWith("image/") ? new Set([IMAGE_OPEN_TOOL]) : ARTIFACT_READ_NAMES;
}

/** Encode one policy-visible, path-free recovery instruction. */
export function recoverableArtifactEnvelope(
  reference: Record<string, any>, source: Record<string, any> | null = null,
): Record<string, any> {
  const ref = get(reference, "ref");
  const sha256 = get(reference, "sha256");
  const mediaType = get(reference, "media_type");
  const byteCount = get(reference, "bytes");
  const policyScope = get(reference, "policy_scope");
  const kind = get(reference, "kind");
  if (
    typeof ref !== "string" || refMatch(ref) === null
    || typeof sha256 !== "string" || !SHA256_PATTERN.test(sha256)
    || typeof mediaType !== "string" || !isByteCount(byteCount)
    || typeof policyScope !== "string" || typeof kind !== "string"
    || ref !== `artifact://${policyScope}/${sha256}`
  ) {
    throw new PyValueError("recoverable Artifact reference is invalid");
  }
  const envelope = {
    schema: RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
    artifact: {
      ref, sha256, kind, media_type: mediaType, bytes: byteCount, policy_scope: policyScope,
      round: get(reference, "round") ?? null,
      action_step: get(reference, "action_step") ?? null,
    },
    source: clonePy({ ...(source ?? {}) }),
    open: recoverableArtifactOpen(reference),
  };
  validateRecoverableArtifactEnvelope(envelope);
  return envelope;
}

/** Fail closed on drift between the reference and its advertised decoder. */
export function validateRecoverableArtifactEnvelope(envelope: Record<string, any>): Record<string, any> {
  if (!sameKeys(envelope, ["schema", "artifact", "source", "open"])) {
    throw new PyValueError("recoverable Artifact envelope fields are invalid");
  }
  const artifact = get(envelope, "artifact");
  const source = get(envelope, "source");
  const opener = get(envelope, "open");
  if (
    !eq(get(envelope, "schema"), RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA)
    || !isMapping(artifact)
    || !sameKeys(artifact, [
      "ref", "sha256", "kind", "media_type", "bytes", "policy_scope", "round", "action_step",
    ])
    || !isMapping(source)
    || !isMapping(opener)
    || !sameKeys(opener, ["tool", "arguments"])
    || !isMapping(get(opener, "arguments"))
    || !sameKeys(opener.arguments, ["ref"])
  ) {
    throw new PyValueError("recoverable Artifact envelope is malformed");
  }
  const ref = get(artifact, "ref");
  const sha256 = get(artifact, "sha256");
  const mediaType = get(artifact, "media_type");
  const byteCount = get(artifact, "bytes");
  const scope = get(artifact, "policy_scope");
  if (
    typeof ref !== "string" || refMatch(ref) === null
    || typeof sha256 !== "string" || !SHA256_PATTERN.test(sha256)
    || typeof get(artifact, "kind") !== "string"
    || typeof mediaType !== "string" || !isByteCount(byteCount)
    || typeof scope !== "string" || ref !== `artifact://${scope}/${sha256}`
    || typeof get(opener, "tool") !== "string"
    || !openerNames(mediaType).has(opener.tool)
    || !eq(get(opener.arguments, "ref"), ref)
  ) {
    throw new PyValueError("recoverable Artifact envelope identity drift");
  }
  for (const field of ["round", "action_step"]) {
    const value = get(artifact, field);
    if (value !== undefined && value !== null && !isByteCount(value)) {
      throw new PyValueError(`recoverable Artifact ${field} is invalid`);
    }
  }
  return clonePy({ ...envelope });
}

/** Encode only non-derivable recovery fields for one Artifact. */
export function recoverableArtifactPointer(reference: Record<string, any>): Record<string, any> {
  const ref = get(reference, "ref");
  const sha256 = get(reference, "sha256");
  const policyScope = get(reference, "policy_scope");
  const mediaType = get(reference, "media_type");
  const byteCount = get(reference, "bytes");
  const match = typeof ref === "string" ? refMatch(ref) : null;
  if (
    typeof ref !== "string" || match === null
    || sha256 !== match.sha256 || policyScope !== match.scope
    || typeof mediaType !== "string" || !isByteCount(byteCount)
  ) {
    throw new PyValueError("recoverable Artifact pointer source is invalid");
  }
  const pointer = {
    v: 1,
    ref,
    mime: mediaType,
    bytes: byteCount,
    at: [get(reference, "round") ?? null, get(reference, "action_step") ?? null],
    open: recoverableArtifactOpen(reference).tool,
  };
  return validateRecoverableArtifactPointer(pointer);
}

/** Fail closed on one compact, directly callable Artifact pointer. */
export function validateRecoverableArtifactPointer(pointer: Record<string, any>): Record<string, any> {
  if (!sameKeys(pointer, ["v", "ref", "mime", "bytes", "at", "open"])) {
    throw new PyValueError("recoverable Artifact pointer fields are invalid");
  }
  const ref = pointer.ref;
  const mediaType = pointer.mime;
  const byteCount = pointer.bytes;
  const location = pointer.at;
  const opener = pointer.open;
  if (
    !pyEqual(pointer.v, 1)
    || typeof ref !== "string" || refMatch(ref) === null
    || typeof mediaType !== "string" || !isByteCount(byteCount)
    || !Array.isArray(location) || location.length !== 2
    || typeof opener !== "string" || !openerNames(mediaType).has(opener)
  ) {
    throw new PyValueError("recoverable Artifact pointer is malformed");
  }
  const fields = ["round", "action_step"];
  location.forEach((value: unknown, index: number) => {
    if (value !== null && value !== undefined && !isByteCount(value)) {
      throw new PyValueError(`recoverable Artifact pointer ${fields[index]} is invalid`);
    }
  });
  return clonePy({ ...pointer });
}

function pointerFromMarkerValue(value: Record<string, any>): Record<string, any> {
  if (eq(get(value, "schema"), RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA)) {
    const envelope = validateRecoverableArtifactEnvelope(value);
    return recoverableArtifactPointer({ ...envelope.artifact, open: envelope.open });
  }
  return validateRecoverableArtifactPointer(value);
}

export function encodeRecoverableArtifactMarker(value: Record<string, any>): string {
  const validated = pointerFromMarkerValue(value);
  return `<${RECOVERABLE_ARTIFACT_TAG}>` + dumpsCompact(validated) + `</${RECOVERABLE_ARTIFACT_TAG}>`;
}

export function decodeRecoverableArtifactMarker(value: unknown): Record<string, any> {
  if (typeof value !== "string") throw new PyValueError("recoverable Artifact marker must be text");
  const opening = `<${RECOVERABLE_ARTIFACT_TAG}>`;
  const closing = `</${RECOVERABLE_ARTIFACT_TAG}>`;
  if (!value.startsWith(opening) || !value.endsWith(closing)) {
    throw new PyValueError("recoverable Artifact marker framing is invalid");
  }
  // Python ``value[len(opening):-len(closing)]`` (empty when they overlap).
  const body = value.length >= opening.length + closing.length
    ? value.slice(opening.length, value.length - closing.length) : "";
  const decoded = tryLoads(body);
  if (!decoded.ok) throw new PyValueError("recoverable Artifact marker JSON is invalid");
  if (!isMapping(decoded.value)) throw new PyValueError("recoverable Artifact marker must contain an object");
  return pointerFromMarkerValue(decoded.value);
}

// -- Tool definitions and result schemas --------------------------------------------

const READ_PROPERTIES = {
  ref: {
    type: "string", minLength: 1,
    description: "Exact Session-scoped artifact:// reference supplied by a recoverable-artifact pointer.",
  },
  offset: {
    type: ["integer", "null"], minimum: 0,
    description: "Starting UTF-8 byte offset for a text range; omit or pass null to start at zero.",
  },
  limit: {
    type: ["integer", "null"], minimum: 1, maximum: ARTIFACT_READ_LIMIT_BYTES,
    description: "Maximum bytes to read in one text range; omit or pass null for the bounded default.",
  },
  json_pointer: {
    type: ["string", "null"],
    description: "JSON Pointer selecting one value from a JSON artifact; do not combine with a byte range. Omit or pass null for byte-range reading.",
  },
};

export const ARTIFACT_READ_TOOL: Record<string, any> = {
  name: "artifact.read",
  description:
    "Retrieve exact text or JSON evidence from this Session's artifact store. "
    + "Use when: exact historical text, JSON, a state dump, receipt, or diagnostic "
    + "was replaced by an artifact:// reference. Effect: read-only and does not "
    + "change the robot or Memory. Returns: one bounded byte range, or one JSON "
    + "Pointer value, together with the verified content identity and continuation "
    + "offset. Does not: expose host paths, cross Session scope, bypass artifact "
    + "hash checks, or return images; reopen image references with observe(ref=...). "
    + "On error: correct the reference/range or re-ground from current evidence; "
    + "missing, tampered, wrong-media, and cross-scope references fail closed.",
  inputSchema: {
    type: "object",
    properties: READ_PROPERTIES,
    required: ["ref"],
    additionalProperties: false,
    oneOf: [
      {
        type: "object",
        properties: {
          ...clonePy(READ_PROPERTIES),
          json_pointer: { type: "null", description: "Omit or pass null for byte-range reading." },
        },
        required: ["ref"],
        additionalProperties: false,
      },
      {
        type: "object",
        properties: {
          ref: clonePy(READ_PROPERTIES.ref),
          json_pointer: { type: "string", description: "Existing JSON Pointer selecting one JSON value." },
          offset: { enum: [null, 0], description: "Omit in Pointer mode; null or the default zero is accepted." },
          limit: { enum: [null, 4096], description: "Omit in Pointer mode; null or the default 4096 is accepted." },
        },
        required: ["ref", "json_pointer"],
        additionalProperties: false,
      },
    ],
  },
};

export const ARTIFACT_READ_RESULT_SCHEMA: Record<string, any> = {
  type: "object",
  properties: {
    status: { type: "string", const: "ok" },
    ref: { type: "string", pattern: "^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}$" },
    sha256: { type: "string" },
    media_type: { type: "string" },
    bytes: { type: "integer" },
    offset: { type: "integer" },
    next_offset: { type: "integer" },
    eof: { type: "boolean" },
    content: { type: ["string", "null"] },
    json_value: {},
  },
  required: [
    "status", "ref", "sha256", "media_type", "bytes", "offset", "next_offset", "eof", "content",
    "json_value",
  ],
  additionalProperties: false,
};

export const HISTORICAL_OBSERVE_RESULT_SCHEMA: Record<string, any> = {
  type: "object",
  properties: {
    status: { type: "string", const: "actual_historical_artifact" },
    ref: { type: "string" },
    sha256: { type: "string", pattern: "^[0-9a-f]{64}$" },
    media_type: { type: "string", pattern: "^image/" },
    bytes: { type: "integer", minimum: 0 },
    round: { type: ["integer", "null"], minimum: 0 },
    action_step: { type: ["integer", "null"], minimum: 0 },
    views: {
      type: "array", items: { type: "string", const: "historical_artifact" }, minItems: 1, maxItems: 1,
    },
  },
  required: ["status", "ref", "sha256", "media_type", "bytes", "round", "action_step", "views"],
  additionalProperties: false,
};

export const CURRENT_OBSERVE_RESULT_SCHEMA: Record<string, any> = {
  type: "object",
  properties: {
    status: { type: "string", const: "actual" },
    views: { type: "array", items: { type: "string" } },
    view_dimensions: { type: "object" },
  },
  required: ["status", "views"],
  additionalProperties: false,
};

/** Add the orthogonal Context reader without changing a frozen ToolSurface. */
export function recoverableArtifactDefinitions(
  definitions: Record<string, any>[], canonicalToolNames: Iterable<string>,
): Record<string, any>[] {
  const values = definitions.map((item) => clonePy({ ...item }));
  const observe = values.find((item) => eq(get(item, "name"), IMAGE_OPEN_TOOL));
  if (!isMapping(observe)) throw new PyValueError("lossless Context requires an observe tool");
  const schema = get(observe, "inputSchema");
  const properties = isMapping(schema) ? get(schema, "properties") : undefined;
  if (!isMapping(properties)) throw new PyValueError("observe input schema is unavailable");
  const currentSchema = clonePy({ ...schema });
  const refSchema = {
    type: "string",
    description: "Exact Session-scoped artifact:// reference from a <recoverable_artifact> pointer.",
    pattern: "^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}$",
  };
  const historicalSchema = {
    type: "object",
    properties: { ref: clonePy(refSchema) },
    required: ["ref"],
    additionalProperties: false,
  };
  const rootProperties: Record<string, any> = clonePy({ ...properties });
  rootProperties.ref = clonePy(refSchema);
  observe.inputSchema = {
    type: "object",
    properties: rootProperties,
    oneOf: [currentSchema, historicalSchema],
    additionalProperties: false,
  };
  const description = get(observe, "description");
  observe.description = String(description === undefined ? "" : description === null ? "None" : description)
    .trimEnd()
    + " A <recoverable_artifact> pointer gives the exact historical mode: "
    + "pass its ref by itself to the named open tool to reopen that image. "
    + "The Runtime verifies Session scope, media type and SHA-256 before returning it.";
  const reader = clonePy(ARTIFACT_READ_TOOL);
  reader.name = artifactReadName(definitions, canonicalToolNames);
  values.push(reader);
  return values;
}

export function recoverableArtifactResultSchemas(
  resultSchemas: Record<string, any>, options: { readerName?: string | null; canonicalToolNames: Iterable<string> },
): Record<string, any> {
  const values: Record<string, any> = clonePy({ ...resultSchemas });
  let observe = get(values, "observe");
  if (observe === undefined || observe === null) {
    observe = { oneOf: [clonePy(CURRENT_OBSERVE_RESULT_SCHEMA)] };
    values.observe = observe;
  }
  if (!isMapping(observe)) throw new PyValueError("observe result schema must be an object");
  let branches = get(observe, "oneOf");
  if (!Array.isArray(branches)) {
    const metadata: Record<string, any> = {};
    for (const key of ["$schema", "$defs"]) if (hasKey(observe, key)) metadata[key] = clonePy(observe[key]);
    const current: Record<string, any> = {};
    for (const key of Object.keys(observe)) if (!hasKey(metadata, key)) current[key] = clonePy(observe[key]);
    observe = { ...metadata, oneOf: [current] };
    values.observe = observe;
    branches = observe.oneOf;
  }
  branches.push(clonePy(HISTORICAL_OBSERVE_RESULT_SCHEMA));
  const readerName = options.readerName
    || artifactReadName(Object.keys(resultSchemas).map((name) => ({ name })), options.canonicalToolNames);
  values[readerName] = clonePy(ARTIFACT_READ_RESULT_SCHEMA);
  return values;
}

// -- The Session catalog ---------------------------------------------------------------

function sourceSortKey(item: Record<string, any>): [number, number] {
  const round = get(item, "round");
  const step = get(item, "action_step");
  return [round === null || round === undefined ? -1 : num(round), step === null || step === undefined ? -1 : num(step)];
}

function compareKeys(left: [number, number], right: [number, number]): number {
  return left[0] - right[0] || left[1] - right[1];
}

function isInside(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

/** Python ``(run_root / relative).resolve()`` (non-strict). */
function resolveUnder(root: string, relative: string): string {
  const joined = path.resolve(root, relative);
  try {
    return fs.realpathSync(joined);
  } catch {
    return joined;
  }
}

export interface PutBytesOptions {
  kind: string;
  suffix: string;
  mediaType: string;
  roundIndex: number | null;
  actionStep: number | null;
}

/** Bind storage refs to stable logical refs within one PolicySession. */
export class ContextArtifactCatalog extends BatchedIndexPersistence {
  runRoot: string;
  policySessionId: string;
  artifacts: ArtifactStore;
  readerName: string;
  indexPath: string;
  private entries = new Map<string, Record<string, any>>();

  constructor(runRoot: string, policySessionId: string, artifacts: ArtifactStore, readerName = "artifact.read") {
    super();
    const absolute = path.resolve(runRoot);
    try {
      this.runRoot = fs.realpathSync(absolute);
    } catch {
      this.runRoot = absolute;
    }
    this.policySessionId = policySessionId;
    this.artifacts = artifacts;
    this.readerName = readerName;
    if (!ARTIFACT_READ_NAMES.has(this.readerName)) {
      throw new PyValueError("unsupported Context Artifact reader name");
    }
    if (refMatch(`artifact://${this.policySessionId}/` + "0".repeat(64)) === null) {
      throw new PyValueError("policy_session_id cannot form an artifact scope");
    }
    this.indexPath = path.join(this.runRoot, "native/context-artifacts/index.json");
    this.persist();
  }

  protected writeIndex(): void {
    const artifacts: Record<string, any> = {};
    for (const [key, entry] of this.entries) artifacts[key] = entry;
    atomicJson(this.indexPath, {
      schema: CONTEXT_ARTIFACT_INDEX_SCHEMA,
      policy_session_id: this.policySessionId,
      artifacts,
    }, this.artifacts.durability ?? null);
  }

  registerStorageRef(
    storageRef: Record<string, any>,
    options: { kind: string; roundIndex: number | null; actionStep: number | null },
  ): Record<string, any> {
    for (const [label, value] of [["round_index", options.roundIndex], ["action_step", options.actionStep]] as const) {
      if (value !== null && value !== undefined && !isByteCount(value)) {
        throw new PyValueError(`artifact ${label} is invalid`);
      }
    }
    const digest = get(storageRef, "sha256");
    const relative = get(storageRef, "path");
    const mediaType = get(storageRef, "media_type");
    const byteCount = get(storageRef, "bytes");
    if (
      typeof digest !== "string" || !SHA256_PATTERN.test(digest)
      || typeof relative !== "string" || typeof mediaType !== "string" || !isByteCount(byteCount)
    ) {
      throw new PyValueError("storage artifact reference is invalid");
    }
    const resolved = resolveUnder(this.runRoot, relative);
    let valid = !path.isAbsolute(relative) && !relative.split("/").includes("..")
      && isInside(this.runRoot, resolved);
    if (valid) {
      try {
        const stat = fs.lstatSync(resolved);
        valid = stat.isFile() && !stat.isSymbolicLink()
          && sha256Hex(fs.readFileSync(resolved)) === digest && stat.size === num(byteCount);
      } catch {
        valid = false;
      }
    }
    if (!valid) throw new PyValueError("storage artifact failed path/hash validation");
    const logical = `artifact://${this.policySessionId}/${digest}`;
    const existing = this.entries.get(logical);
    const entry: Record<string, any> = {
      schema: CONTEXT_ARTIFACT_REF_SCHEMA,
      ref: logical,
      sha256: digest,
      kind: options.kind,
      media_type: mediaType,
      bytes: byteCount,
      policy_scope: this.policySessionId,
      sources: [{ round: options.roundIndex ?? null, action_step: options.actionStep ?? null }],
      storage_ref: clonePy({ ...storageRef }),
    };
    if (existing !== undefined) {
      const keys = ["schema", "ref", "sha256", "kind", "media_type", "bytes", "policy_scope", "storage_ref"];
      const immutable: Record<string, any> = {};
      const candidate: Record<string, any> = {};
      for (const key of keys) {
        immutable[key] = get(existing, key) ?? null;
        candidate[key] = get(entry, key) ?? null;
      }
      // Storage is content-addressed: the same bytes stored as a tool image and as an
      // observation frame are one file; only the storage kind label differs. The entry
      // keeps its first storage ref (ora/agent/memory/catalog.py does the same).
      for (const value of [immutable, candidate]) {
        if (isMapping(value.storage_ref)) {
          value.storage_ref = Object.fromEntries(Object.entries(value.storage_ref).filter(([key]) => key !== "kind"));
        }
      }
      if (!canonicalBytesPy(immutable).equals(canonicalBytesPy(candidate))) {
        throw new PyValueError("logical artifact identity drift");
      }
      const source = { round: options.roundIndex ?? null, action_step: options.actionStep ?? null };
      if (!hasKey(existing, "sources")) existing.sources = [];
      const sources: Record<string, any>[] = existing.sources;
      if (!sources.some((item) => pyEqual(item, source))) {
        sources.push(source);
        // Stable sort by (round, action_step) with None as -1.
        const ordered = sources.map((item, index) => ({ item, index }))
          .sort((a, b) => compareKeys(sourceSortKey(a.item), sourceSortKey(b.item)) || a.index - b.index)
          .map(({ item }) => item);
        sources.splice(0, sources.length, ...ordered);
        this.persist();
      }
      return this.publicRef(existing, options.roundIndex, options.actionStep);
    }
    this.entries.set(logical, entry);
    this.persist();
    return this.publicRef(entry, options.roundIndex, options.actionStep);
  }

  putBytes(data: Buffer, options: PutBytesOptions): Record<string, any> {
    const storage: StorageRef = this.artifacts.putBytes(data, {
      kind: options.kind, suffix: options.suffix, mediaType: options.mediaType,
    });
    return this.registerStorageRef(storage, {
      kind: options.kind, roundIndex: options.roundIndex, actionStep: options.actionStep,
    });
  }

  publicRef(entry: Record<string, any>, roundIndex: number | null = null, actionStep: number | null = null): Record<string, any> {
    return publicArtifactRef(entry, { readerName: this.readerName, roundIndex, actionStep });
  }

  resolve(logicalRef: string): [Record<string, any>, Buffer] {
    const match = refMatch(logicalRef);
    if (match === null) throw new RefError("syntax", "artifact reference syntax is invalid");
    if (match.scope !== this.policySessionId) {
      throw new RefError("scope", "artifact reference is outside this PolicySession");
    }
    const entry = this.entries.get(logicalRef);
    if (entry === undefined) throw new RefError("unknown", "artifact reference is unavailable");
    const storage = get(entry, "storage_ref");
    const relative = isMapping(storage) ? get(storage, "path") : undefined;
    let data: Buffer | null = null;
    if (typeof relative === "string" && !path.isAbsolute(relative)
      && !relative.split("/").includes("..")) {
      const resolved = resolveUnder(this.runRoot, relative);
      if (isInside(this.runRoot, resolved)) {
        try {
          const stat = fs.lstatSync(resolved);
          if (stat.isFile() && !stat.isSymbolicLink()) data = fs.readFileSync(resolved);
        } catch {
          data = null;
        }
      }
    }
    if (data === null) throw new RefError("storage", "artifact storage is unavailable");
    if (sha256Hex(data) !== get(entry, "sha256") || !pyEqual(data.length, get(entry, "bytes"))
      || match.sha256 !== get(entry, "sha256")) {
      throw new RefError("hash", "artifact content hash does not match its reference");
    }
    return [clonePy(entry), data];
  }

  /** List policy-readable refs first observed in one physical Episode. */
  publicRefsForRound(roundIndex: number | null): Record<string, any>[] {
    if (roundIndex !== null && !isByteCount(roundIndex)) throw new PyValueError("Artifact round index is invalid");
    const values: Record<string, any>[] = [];
    for (const entry of this.entries.values()) {
      const sources = get(entry, "sources");
      if (!Array.isArray(sources)) continue;
      const matching = sources.filter((source) => isMapping(source) && eq(get(source, "round"), roundIndex));
      if (!matching.length) continue;
      const steps = matching.map((source) => get(source, "action_step")).filter((step) => isInt(step)).map(num);
      const reference = this.publicRef(entry, roundIndex, steps.length ? Math.min(...steps) : null);
      reference.open = recoverableArtifactOpen(reference);
      values.push(reference);
    }
    return values.sort((a, b) => {
      const left = a.action_step === null ? -1 : num(a.action_step);
      const right = b.action_step === null ? -1 : num(b.action_step);
      if (left !== right) return left - right;
      const l = String(a.ref);
      const r = String(b.ref);
      return l < r ? -1 : l > r ? 1 : 0;
    });
  }

  /** Every entry, for evidence (the index file's ``artifacts`` object). */
  entryMap(): Map<string, Record<string, any>> {
    return this.entries;
  }
}

// -- artifact_read / observe(ref) ------------------------------------------------------------

/** ``RecoverableArtifactToolRuntime._observe_ref``. */
export function observeRef(
  catalog: ContextArtifactCatalog, argumentsValue: Record<string, any>,
): Record<string, any> {
  checkObserveRefArguments(argumentsValue);
  const [entry, data] = policyResolve(catalog, String(argumentsValue.ref));
  const mediaType = String(entry.media_type);
  if (!mediaType.startsWith("image/")) {
    throw new ToolError(
      `observe(ref) requires an image artifact; use ${catalog.readerName} for text/JSON`,
      {
        code: "invalid_arguments", field: "ref", received: argumentsValue.ref,
        constraints: {
          media_type: mediaType,
          open: { tool: catalog.readerName, arguments: { ref: argumentsValue.ref } },
        },
      },
    );
  }
  const sources = get(entry, "sources");
  const source = Array.isArray(sources) && sources.length && isMapping(sources[0]) ? sources[0] : {};
  const result: Record<string, any> = {
    status: "actual_historical_artifact",
    ref: entry.ref,
    sha256: entry.sha256,
    media_type: mediaType,
    bytes: entry.bytes,
    round: get(source, "round") ?? null,
    action_step: get(source, "action_step") ?? null,
    views: ["historical_artifact"],
    images_base64: [data.toString("base64")],
  };
  const checked: Record<string, any> = { ...result };
  delete checked.images_base64;
  validateSchemaInstance(checked, HISTORICAL_OBSERVE_RESULT_SCHEMA);
  return result;
}

/** ``RecoverableArtifactToolRuntime._read_ref``: the ``artifact_read`` tool. */
export function artifactRead(
  catalog: ContextArtifactCatalog, argumentsValue: Record<string, any>,
): Record<string, any> {
  checkReadArguments(argumentsValue, ARTIFACT_READ_TOOL.inputSchema, ARTIFACT_READ_LIMIT_BYTES);
  const [entry, data] = policyResolve(catalog, String(argumentsValue.ref));
  const mediaType = String(entry.media_type);
  if (mediaType.startsWith("image/")) {
    throw new ToolError("ref names an image artifact; this reader returns text or JSON only", {
      code: "invalid_arguments", field: "ref", received: argumentsValue.ref,
      constraints: { media_type: mediaType },
    });
  }
  const pointer = argumentsValue.json_pointer ?? null;
  let offset = hasKey(argumentsValue, "offset") ? argumentsValue.offset : 0;
  const limit = hasKey(argumentsValue, "limit") ? argumentsValue.limit : 4096;
  if (pointer !== null) {
    if (mediaType !== "application/json") {
      throw new ToolError("json_pointer must be a string selecting a JSON artifact", {
        code: "invalid_arguments", field: "json_pointer", received: pointer,
        constraints: { allowed: { type: "string", media_type: "application/json" } },
      });
    }
    const value = loadsPy(data.toString("utf8"));
    let selected: unknown;
    try {
      selected = jsonPointer(value, pointer);
    } catch (error) {
      if (!(error instanceof JsonPointerError)) throw error;
      const rootKeys = isMapping(value) ? sortedStrings(Object.keys(value)) : [];
      throw new ToolError(`${error.message}; select a path from this artifact's root`, {
        code: "invalid_arguments", field: "json_pointer", received: pointer,
        constraints: {
          allowed: "an existing JSON Pointer; empty string selects the root",
          top_level_keys: rootKeys,
          root_type: isMapping(value) ? "object" : Array.isArray(value) ? "array" : "scalar",
          ...(Array.isArray(value) ? { root_array_length: value.length } : {}),
        },
      });
    }
    return {
      status: "ok", ref: entry.ref, sha256: entry.sha256, media_type: mediaType, bytes: entry.bytes,
      offset: 0, next_offset: data.length, eof: true, content: null,
      json_value: boundedPointerValue(selected, pointer, ARTIFACT_READ_LIMIT_BYTES),
    };
  }
  offset = offset === null || offset === undefined ? 0 : num(offset);
  const [text, nextOffset] = readRange(data, offset, limit === null || limit === undefined ? 4096 : num(limit));
  return {
    status: "ok", ref: entry.ref, sha256: entry.sha256, media_type: mediaType, bytes: entry.bytes,
    offset, next_offset: nextOffset, eof: nextOffset >= data.length, content: text, json_value: null,
  };
}

export interface ArtifactRetrievalRecord {
  tool: string;
  ref: unknown;
  status: "ok" | "rejected";
  error: string | null;
  duration_ms: unknown;
}

/**
 * ``RecoverableArtifactToolRuntime.invoke_with_context`` for the two readers:
 * runs the read, validates an ``artifact_read`` result, and reports one
 * ``context_artifact_retrieval`` payload to ``record``.
 */
export function invokeArtifactTool(
  catalog: ContextArtifactCatalog, name: string, argumentsValue: Record<string, any>,
  record: (payload: ArtifactRetrievalRecord) => void = () => {},
): Record<string, any> {
  const started = process.hrtime.bigint();
  const elapsed = () => pyFloat(pyRound(Number(process.hrtime.bigint() - started) / 1_000_000, 3));
  let result: Record<string, any>;
  try {
    result = ARTIFACT_READ_NAMES.has(name) ? artifactRead(catalog, argumentsValue) : observeRef(catalog, argumentsValue);
    if (ARTIFACT_READ_NAMES.has(name)) validateSchemaInstance(result, ARTIFACT_READ_RESULT_SCHEMA);
  } catch (error) {
    record({
      tool: name, ref: get(argumentsValue, "ref") ?? null, status: "rejected",
      error: error instanceof Error ? error.message : String(error), duration_ms: elapsed(),
    });
    throw error;
  }
  record({ tool: name, ref: get(argumentsValue, "ref") ?? null, status: "ok", error: null, duration_ms: elapsed() });
  return result;
}

/** Whether a call is answered by the Context readers rather than the robot tools. */
export function isArtifactToolCall(name: string, argumentsValue: Record<string, any>): boolean {
  return ARTIFACT_READ_NAMES.has(name)
    || (name === IMAGE_OPEN_TOOL && get(argumentsValue, "ref") !== undefined && get(argumentsValue, "ref") !== null);
}

export { PyJsonDecodeError };
