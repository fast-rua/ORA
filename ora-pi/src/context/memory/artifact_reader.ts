// `ora/agent/memory/artifact_reader.py`: argument refusals and the bounded read
// behind `artifact_read` and `observe(ref)`.

import {
  clonePy, dumpsPy, isInt, isMapping, num, pyEqual, pyRepr, sortedStrings, utf8,
  utf8DecodeError,
} from "../pyutil.ts";
import { SchemaViolation, ToolSurfaceError, validateSchemaInstance } from "./schema_validation.ts";
import { ToolError, schemaRefusal } from "./tool_errors.ts";

export const REF_PATTERN = "^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}$";
export const READ_ARGUMENTS = ["ref", "offset", "limit", "json_pointer"];

export type RefRefusalKind = "syntax" | "scope" | "unknown" | "storage" | "hash";

/** The catalog's own resolution failures, by refusal kind. */
export class RefError extends Error {
  kind: RefRefusalKind;
  constructor(kind: RefRefusalKind, message: string) {
    super(message);
    this.name = "RefError";
    this.kind = kind;
  }
}

export interface ResolvingCatalog {
  policySessionId: string;
  resolve(ref: string): [Record<string, any>, Buffer];
}

/** ``catalog.resolve`` for a policy call: each failure as its refusal. */
export function policyResolve(catalog: ResolvingCatalog, ref: string): [Record<string, any>, Buffer] {
  try {
    return catalog.resolve(ref);
  } catch (error) {
    if (error instanceof RefError) {
      throw refRefusal(error.kind, ref, error.kind === "scope" ? catalog.policySessionId : null);
    }
    throw error;
  }
}

export function refRefusal(kind: RefRefusalKind, ref: unknown, scope: string | null = null): ToolError {
  if (kind === "storage") {
    return new ToolError("the artifact store has no readable file for this reference", {
      code: "artifact_integrity_fault", retryable: false, field: "ref", received: ref,
    });
  }
  if (kind === "hash") {
    return new ToolError("artifact content no longer matches its reference hash", {
      code: "artifact_hash_mismatch", retryable: false, field: "ref", received: ref,
    });
  }
  const message = {
    syntax: "ref is not an artifact:// reference",
    scope: "ref belongs to another PolicySession",
    unknown: "no artifact in this PolicySession carries that ref",
  }[kind];
  const constraints: Record<string, unknown> = { pattern: REF_PATTERN };
  if (scope !== null) constraints.policy_session_id = scope;
  return new ToolError(message, { code: "invalid_arguments", field: "ref", received: ref, constraints });
}

function refuse(field: string, message: string, received: unknown, constraints: Record<string, unknown>): ToolError {
  return new ToolError(message, { code: "invalid_arguments", field, received, constraints });
}

function validate(argumentsValue: Record<string, any>, schema: Record<string, any>): void {
  try {
    validateSchemaInstance({ ...argumentsValue }, schema, { rootPath: "$input" });
  } catch (error) {
    if (error instanceof SchemaViolation) throw schemaRefusal(error);
    if (error instanceof ToolSurfaceError) throw schemaRefusal({ message: error.message });
    throw error;
  }
}

/** Each ``artifact_read`` argument by name, then the published schema. */
export function checkReadArguments(
  argumentsValue: Record<string, any>, schema: Record<string, any>, limit: number,
): void {
  const unknown = sortedStrings(Object.keys(argumentsValue).filter((key) => !READ_ARGUMENTS.includes(key)));
  if (unknown.length) {
    throw refuse(unknown[0], `artifact_read does not take ${unknown.join(", ")}`, unknown, {
      allowed: [...READ_ARGUMENTS],
    });
  }
  const ref = argumentsValue.ref;
  if (typeof ref !== "string" || !ref) {
    throw refuse("ref", "ref must be a non-empty artifact:// reference", ref ?? null, { pattern: REF_PATTERN });
  }
  const pointer = argumentsValue.json_pointer ?? null;
  if (pointer !== null && typeof pointer !== "string") {
    throw refuse("json_pointer", "json_pointer must be a string or null", pointer, { type: ["string", "null"] });
  }
  const offset = argumentsValue.offset ?? null;
  const size = argumentsValue.limit ?? null;
  if (offset !== null && (!isInt(offset) || num(offset) < 0)) {
    throw refuse("offset", "offset must be a non-negative integer byte offset or null", offset, { minimum: 0 });
  }
  if (size !== null && (!isInt(size) || !(1 <= num(size) && num(size) <= limit))) {
    throw refuse("limit", `limit must be an integer byte count in [1, ${limit}] or null`, size, {
      minimum: 1, maximum: limit,
    });
  }
  if (pointer !== null && (!(offset === null || pyEqual(offset, 0)) || !(size === null || pyEqual(size, 4096)))) {
    throw refuse(
      "json_pointer", "json_pointer cannot be combined with an explicit range; "
      + "omit offset and limit to select a JSON value",
      { json_pointer: pointer, offset, limit: size },
      { allowed: ["ref + json_pointer", "ref + optional offset/limit"] },
    );
  }
  validate(argumentsValue, schema);
}

/** ``observe``'s historical branch, as ``recoverableArtifactDefinitions`` publishes it. */
export const HISTORICAL_OBSERVE_INPUT = {
  type: "object", required: ["ref"], additionalProperties: false,
  properties: { ref: { type: "string", pattern: REF_PATTERN } },
};

/** ``observe(ref)`` takes ``ref`` alone. */
export function checkObserveRefArguments(argumentsValue: Record<string, any>): void {
  const keys = Object.keys(argumentsValue);
  if (!(keys.length === 1 && keys[0] === "ref")) {
    const others = sortedStrings(keys.filter((key) => key !== "ref"));
    throw refuse(others.length ? others[0] : "ref",
      "observe with ref reopens one historical image and takes ref alone",
      sortedStrings(keys), { allowed: ["ref"] });
  }
  validate(argumentsValue, HISTORICAL_OBSERVE_INPUT);
}

export class JsonPointerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JsonPointerError";
  }
}

/** RFC 6901 selection; ``""`` is the root. */
export function jsonPointer(value: unknown, pointer: string): unknown {
  if (pointer === "") return clonePy(value);
  if (!pointer.startsWith("/")) throw new JsonPointerError("json_pointer must be empty or start with /");
  let current: any = value;
  for (const raw of pointer.slice(1).split("/")) {
    const token = raw.replaceAll("~1", "/").replaceAll("~0", "~");
    if (isMapping(current)) {
      if (!Object.prototype.hasOwnProperty.call(current, token)) {
        throw new JsonPointerError("json_pointer does not exist");
      }
      current = current[token];
    } else if (Array.isArray(current)) {
      if (!/^[0-9]+$/.test(token) || Number(token) >= current.length) {
        throw new JsonPointerError("json_pointer does not exist");
      }
      current = current[Number(token)];
    } else {
      throw new JsonPointerError("json_pointer crosses a scalar");
    }
  }
  return clonePy(current);
}

function shape(value: unknown): Record<string, unknown> {
  if (isMapping(value)) return { value_type: "object", keys: sortedStrings(Object.keys(value)) };
  if (Array.isArray(value)) return { value_type: "array", length: value.length };
  return { value_type: "scalar" };
}

/** Refuse a selected value larger than one byte-range read. */
export function boundedPointerValue(selected: unknown, pointer: string, limit: number): unknown {
  const size = utf8(dumpsPy(selected, { ensureAscii: false, separators: [",", ":"] })).length;
  if (size <= limit) return selected;
  throw refuse("json_pointer",
    `the value at json_pointer ${pyRepr(pointer)} is ${size} bytes, more than the `
    + `${limit}-byte read limit`, pointer,
    { value_bytes: size, maximum_bytes: limit, ...shape(selected) });
}

/** One UTF-8 text range starting at ``offset``; refuses ranges that cannot decode. */
export function readRange(data: Buffer, offset: number, limit: number): [string, number] {
  if (offset > data.length) {
    throw refuse("offset", `offset ${offset} is past the artifact's ${data.length} bytes`, offset, {
      minimum: 0, maximum: data.length,
    });
  }
  let end = Math.min(data.length, offset + limit);
  for (;;) {
    const chunk = data.subarray(offset, end);
    const failure = utf8DecodeError(chunk);
    if (failure !== null) {
      if (failure.reason === "unexpected end of data" && end > offset) {
        end -= 1;
        continue;
      }
      if (failure.start === 0) {
        throw refuse("offset", "offset does not start a UTF-8 character", offset, {
          rule: "offset at a UTF-8 character boundary",
        });
      }
      throw new ToolError("artifact range is not UTF-8 text", {
        code: "artifact_integrity_fault", retryable: false, field: "ref",
      });
    }
    const text = chunk.toString("utf8");
    if (!text && offset < data.length) {
      throw refuse("limit", `limit ${limit} is shorter than the UTF-8 character at offset ${offset}`, limit, {
        minimum: 4,
      });
    }
    return [text, end];
  }
}
