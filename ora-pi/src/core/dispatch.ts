// ORA's tool-dispatch protocol, robot-agnostic half (TypeScript).
//
// Ports: native/tool_arguments.py (decoding, tool errors), native/tool_batch.py
// (execution copies, one-mutation batch rule), native/mutation_gates.py,
// composition/surface/null_optional_arguments.py (the gateway's null-as-omitted
// stripping), composition/surface/integral_arguments.py and session_core/repeated_failures.py.
// Which calls mutate the world is decided in Python (tools.classify); the
// tables of argument-refusal and infrastructure codes come from Python too.

import { clonePy, dumpsCompactSorted, parsePy, PyFloat, PyJsonError } from "../pyjson.ts";

type Json = any;

export interface DomainCodes {
  argumentRefusalCodes: Set<string>;
  infrastructureCodes: Set<string>;
}

export interface ToolErrorInit {
  message: string;
  code?: string;
  retryable?: boolean;
  field?: string | null;
  received?: Json;
  constraints?: Record<string, Json>;
  retryRequiresChangedArguments?: boolean | null;
}

/** `ToolError(...).as_dict()` */
export function toolErrorDict(init: ToolErrorInit, codes: DomainCodes): Record<string, Json> {
  const code = init.code ?? "invalid_arguments";
  let retry = init.retryRequiresChangedArguments ?? null;
  if (retry === null && codes.argumentRefusalCodes.has(code)) retry = true;
  const reported: Record<string, Json> = {
    code,
    message: init.message,
    retryable: init.retryable ?? true,
    field: init.field ?? null,
    received: init.received === undefined ? null : init.received,
    constraints: init.constraints ?? {},
    infrastructure: codes.infrastructureCodes.has(code),
  };
  if (retry !== null) reported.retry_requires_changed_arguments = Boolean(retry);
  return reported;
}

export class ToolCallError extends Error {
  readonly init: ToolErrorInit;
  constructor(init: ToolErrorInit) {
    super(init.message);
    this.init = init;
  }
}

/** `_tool_error(exc)` */
export function toolErrorResult(error: unknown, codes: DomainCodes): Record<string, Json> {
  const structured = error instanceof ToolCallError
    ? toolErrorDict(error.init, codes)
    : toolErrorDict({ message: String((error as Error)?.message ?? error), code: "internal_tool_error", retryable: false }, codes);
  return { ok: false, error: structured };
}

// -- Decoding -------------------------------------------------------------------

/** `tool_arguments._arguments`: the function name and decoded argument object. */
export function decodeArguments(call: Record<string, Json>): [string, Record<string, Json>] {
  const fn = call.function;
  if (typeof fn !== "object" || fn === null || typeof fn.name !== "string" || !fn.name) {
    throw new ToolCallError({ message: "tool call lacks a function name", code: "invalid_tool_call" });
  }
  const raw = Object.prototype.hasOwnProperty.call(fn, "arguments") ? fn.arguments : "{}";
  let value: Json;
  if (typeof raw === "object" && raw !== null && !Array.isArray(raw) && !(raw instanceof PyFloat)) {
    value = clonePy(raw);
  } else if (typeof raw === "string") {
    try {
      value = parsePy(raw);
    } catch (error) {
      if (!(error instanceof PyJsonError)) throw error;
      throw new ToolCallError({ message: "tool arguments are not valid JSON", field: "arguments", received: raw });
    }
  } else {
    throw new ToolCallError({ message: "tool arguments must be a JSON object", field: "arguments", received: raw });
  }
  if (typeof value !== "object" || value === null || Array.isArray(value) || value instanceof PyFloat) {
    throw new ToolCallError({ message: "tool arguments must decode to an object", field: "arguments", received: value });
  }
  return [String(fn.name), value];
}

export interface ParsedCall {
  call: Record<string, Json>;
  name: string | null;
  args: Record<string, Json> | null;
  effect: string | null;
  error: unknown | null;
}

/**
 * `tool_batch.parse_tool_calls`: decode every call into an execution copy. The
 * effect classifier runs per decodable call; a classifier error becomes that
 * call's parse error, exactly as the Python try-block does.
 */
export async function parseToolCalls(
  calls: Json[], modelCallId: string,
  adoptId: (index: number, callId: string) => void,
  classify: ((items: Array<{ name: string; arguments: Json }>) => Promise<Array<{ effect?: string; error?: Json }>>) | null,
): Promise<{ parsed: ParsedCall[]; names: string[] }> {
  const decoded: Array<{ call: Record<string, Json>; name?: string; args?: Record<string, Json>; error?: unknown }> = [];
  for (const [index, raw] of calls.entries()) {
    const call: Record<string, Json> = typeof raw === "object" && raw !== null ? { ...raw } : {};
    let callId = call.id;
    if (typeof callId !== "string" || !callId) {
      callId = `${modelCallId}-tool-${String(index).padStart(2, "0")}`;
      call.id = callId;
      adoptId(index, callId);
    }
    try {
      const [name, args] = decodeArguments(call);
      decoded.push({ call, name, args });
    } catch (error) {
      decoded.push({ call, error });
    }
  }
  const classifiable = decoded.filter((item) => item.error === undefined);
  const effects = classify && classifiable.length
    ? await classify(classifiable.map((item) => ({ name: item.name!, arguments: item.args! })))
    : [];
  const parsed: ParsedCall[] = [];
  const names: string[] = [];
  let effectIndex = 0;
  for (const item of decoded) {
    const call = item.call;
    if (item.error === undefined) {
      // Python rewrites the execution copy and records the name before it
      // classifies; a classifier error then keeps both (tool_batch.py).
      const fn = { ...(call.function ?? {}) };
      fn.name = item.name;
      fn.arguments = dumpsCompactSorted(item.args);
      call.type = "function";
      call.function = fn;
      names.push(item.name!);
      const classified = classify ? effects[effectIndex++] : { effect: undefined };
      if (classified && classified.error !== undefined) {
        parsed.push({ call, name: item.name!, args: null, effect: null, error: { classified: classified.error } });
        continue;
      }
      parsed.push({ call, name: item.name!, args: item.args!, effect: classified?.effect ?? null, error: null });
      continue;
    }
    const fn = call.function;
    const originalName = typeof fn === "object" && fn !== null && typeof fn.name === "string" ? fn.name : null;
    const name = originalName ?? "invalid_tool_call";
    if (originalName === null) {
      call.type = "function";
      call.function = { name, arguments: "{}" };
    }
    parsed.push({ call, name, args: null, effect: null, error: item.error });
  }
  return { parsed, names };
}

export interface BatchAdmission {
  refusedMutations: Set<number>;
  readsRun: number;
  typedEffects: boolean;
  mutationTools: Set<string>;
}

export function isMutation(admission: BatchAdmission, effect: string | null, name: string | null): boolean {
  return admission.typedEffects ? effect === "mutation" : name !== null && admission.mutationTools.has(name);
}

/** `tool_batch.admit_tool_batch`: a world mutation must be alone in its reply. */
export function admitToolBatch(
  parsed: ParsedCall[], names: string[], callCount: number, typedEffects: boolean, mutationTools: Set<string>,
): BatchAdmission {
  const mutationCount = typedEffects
    ? parsed.filter((item) => item.effect === "mutation").length
    : names.filter((name) => mutationTools.has(name)).length;
  const mutationBatch = mutationCount > 1 || (mutationCount === 1 && callCount > 1);
  const refused = new Set<number>();
  parsed.forEach((item, index) => {
    const mutation = typedEffects ? item.effect === "mutation" : item.name !== null && mutationTools.has(item.name);
    if (mutationBatch && mutation) refused.add(index);
  });
  const readsRun = parsed.filter((item, index) => !refused.has(index) && item.error === null).length;
  return { refusedMutations: refused, readsRun, typedEffects, mutationTools };
}

const CAUSAL_SLOTS = new Set(["native_slots_v3", "native_slots_v4"]);

/** A tool's own wording of the `episode_terminal` refusal (code mode's script tools, from Python). */
export interface TerminalRefusal {
  message: string;
  constraints?: Record<string, Json>;
}

/** `mutation_gates.mutation_protocol_error`; `wording`, where the tool has one, replaces its message and constraints. */
export function mutationProtocolError(
  composition: string | null, effect: string | null, terminalLifecycle: boolean, wording?: TerminalRefusal,
): ToolErrorInit | null {
  if (composition !== null && CAUSAL_SLOTS.has(composition) && effect === "mutation" && terminalLifecycle) {
    return {
      message: wording?.message ?? "world mutation is unavailable after the trusted terminal boundary",
      code: "episode_terminal", retryable: false, constraints: wording?.constraints,
    };
  }
  return null;
}

export function batchRejection(readsRun: number): ToolErrorInit {
  return {
    message: "a world-mutating tool must be the only tool call in an assistant response; this mutation was not "
      + "executed and the other calls in the response ran",
    code: "mutation_batch_rejected",
    constraints: { other_calls_executed: readsRun },
  };
}

// -- Argument normalisation (gateway.arguments) -------------------------------------

function isObject(value: unknown): value is Record<string, Json> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !(value instanceof PyFloat);
}

function resolveRef(schema: Json, defs: Record<string, Json>): Json {
  const reference = isObject(schema) ? schema.$ref : undefined;
  if (typeof reference === "string" && reference.startsWith("#/$defs/")) {
    const target = defs[reference.split("/").slice(2).join("/")];
    return isObject(target) ? resolveRef(target, defs) : schema;
  }
  return schema;
}

function objectBranches(schema: Record<string, Json>, defs: Record<string, Json>): Record<string, Json>[] {
  const branches: Record<string, Json>[] = isObject(schema.properties) ? [schema] : [];
  for (const keyword of ["oneOf", "anyOf"]) {
    for (const raw of schema[keyword] ?? []) {
      const branch = resolveRef(raw, defs);
      if (isObject(branch) && isObject(branch.properties)) branches.push(branch);
    }
  }
  return branches;
}

function optionalNames(schema: Record<string, Json>, defs: Record<string, Json>): Set<string> {
  const branches = objectBranches(schema, defs);
  const required = new Set<string>(schema.required ?? []);
  const names = new Set<string>();
  for (const branch of branches) {
    for (const name of branch.required ?? []) required.add(name);
    for (const name of Object.keys(branch.properties)) names.add(name);
  }
  return new Set([...names].filter((name) => !required.has(name)));
}

function deepEqual(left: Json, right: Json): boolean {
  return dumpsCompactSorted(left) === dumpsCompactSorted(right);
}

/** `null_optional_arguments.strip_null_optional` */
export function stripNullOptional(value: Json, schemaIn: Json, defs: Record<string, Json>): Json {
  const schema = resolveRef(schemaIn, defs);
  if (!isObject(schema)) return value;
  if (isObject(value)) {
    const branches = objectBranches(schema, defs);
    if (!branches.length) return value;
    const optional = optionalNames(schema, defs);
    const output: Record<string, Json> = {};
    let changed = false;
    for (const [key, childIn] of Object.entries(value)) {
      let child = childIn;
      if (child === null && optional.has(key)) {
        changed = true;
        continue;
      }
      const shapes = branches.filter((branch) => Object.prototype.hasOwnProperty.call(branch.properties, key))
        .map((branch) => branch.properties[key]);
      if (shapes.length && shapes.every((shape) => deepEqual(shape, shapes[0]))) {
        const stripped = stripNullOptional(child, shapes[0], defs);
        changed ||= stripped !== child;
        child = stripped;
      }
      output[key] = child;
    }
    return changed ? output : value;
  }
  if (Array.isArray(value) && isObject(schema.items)) {
    const items = value.map((item) => stripNullOptional(item, schema.items, defs));
    return items.some((item, index) => item !== value[index]) ? items : value;
  }
  return value;
}

function schemaTypes(node: Record<string, Json>): Set<string> {
  const declared = node.type;
  if (typeof declared === "string") return new Set([declared]);
  return Array.isArray(declared) ? new Set(declared.filter((item) => typeof item === "string")) : new Set();
}

function expand(nodes: Json[], root: Record<string, Json>): Record<string, Json>[] {
  const expanded: Record<string, Json>[] = [];
  const pending = nodes.filter(isObject);
  const seen = new Set<object>();
  while (pending.length) {
    const node = pending.pop()!;
    if (seen.has(node)) continue;
    seen.add(node);
    expanded.push(node);
    const reference = node.$ref;
    if (typeof reference === "string" && reference.startsWith("#/$defs/")) {
      const target = (root.$defs ?? {})[reference.slice("#/$defs/".length)];
      if (isObject(target)) pending.push(target);
    }
    for (const keyword of ["oneOf", "anyOf", "allOf"]) {
      const branches = node[keyword];
      if (Array.isArray(branches)) pending.push(...branches.filter(isObject));
    }
  }
  return expanded;
}

function coerce(value: Json, nodes: Json[], root: Record<string, Json>): Json {
  const applying = expand(nodes, root);
  if (value instanceof PyFloat || (typeof value === "number" && !Number.isInteger(value))) {
    const numeric = value instanceof PyFloat ? value.value : value;
    const types = new Set<string>();
    for (const node of applying) for (const type of schemaTypes(node)) types.add(type);
    if (types.has("integer") && !types.has("number") && Number.isFinite(numeric) && Number.isInteger(numeric)) {
      return Object.is(numeric, -0) ? 0 : numeric;
    }
    return value;
  }
  if (isObject(value)) {
    const changed: Record<string, Json> = {};
    for (const [key, item] of Object.entries(value)) {
      const children = [
        ...applying.filter((node) => isObject(node.properties) && Object.prototype.hasOwnProperty.call(node.properties, key)).map((node) => node.properties[key]),
        ...applying.filter((node) => isObject(node.additionalProperties)).map((node) => node.additionalProperties),
      ];
      const coerced = children.length ? coerce(item, children, root) : item;
      if (coerced !== item) changed[key] = coerced;
    }
    return Object.keys(changed).length ? { ...value, ...changed } : value;
  }
  if (Array.isArray(value)) {
    const children = applying.filter((node) => isObject(node.items)).map((node) => node.items);
    if (!children.length) return value;
    const coerced = value.map((item) => coerce(item, children, root));
    return coerced.some((item, index) => item !== value[index]) ? coerced : value;
  }
  return value;
}

/** Same spelling rules as ora/agent/composition/surface/tool_names.canonical (dotted ↔ underscore). */
export type CanonicalName = (name: string) => string;

export function normalizeArguments(
  name: string, args: Record<string, Json>, delivered: Record<string, Json>[], nullAsOmitted: boolean,
  canonical: CanonicalName,
): Record<string, Json> {
  const wanted = canonical(name);
  const definition = delivered.find((item) => canonical(String(item.name)) === wanted);
  let value = args;
  const schema = definition?.inputSchema;
  if (nullAsOmitted && isObject(schema)) {
    value = stripNullOptional(value, schema, isObject(schema.$defs) ? schema.$defs : {});
  }
  if (isObject(schema)) value = coerce(value, [schema], schema);
  return value;
}

// -- Repeated failures ----------------------------------------------------------------

export class RepeatedFailureTracker {
  private identity: string | null = null;
  private count = 0;

  reset(): void {
    this.identity = null;
    this.count = 0;
  }

  snapshot(): { identity: string | null; count: number } {
    return { identity: this.identity, count: this.count };
  }

  restore(state: { identity: string | null; count: number }): void {
    this.identity = state.identity;
    this.count = state.count;
  }

  /** The repetition record to add to this result's error, or null. */
  record(name: string, args: Record<string, Json> | null, result: Record<string, Json>): Record<string, Json> | null {
    const error = isObject(result.error) ? result.error : {};
    const code = error.code;
    let canonicalArgs: string | null = null;
    if (isObject(args)) {
      try {
        canonicalArgs = dumpsCompactSorted(args);
        if (/(?:^|[^"\w])(?:NaN|-?Infinity)(?:[^"\w]|$)/.test(canonicalArgs)) canonicalArgs = null;
      } catch {
        canonicalArgs = null;
      }
    }
    if (typeof code !== "string" || canonicalArgs === null) {
      this.reset();
      return null;
    }
    const identity = JSON.stringify([name, canonicalArgs, code]);
    this.count = identity === this.identity ? this.count + 1 : 1;
    this.identity = identity;
    if (this.count < 2) return null;
    const unchangedFails = error.retry_requires_changed_arguments === true;
    const note = `this identical request has failed ${this.count} times in a row with ${code}`;
    return {
      identical_failures_in_a_row: this.count,
      note: unchangedFails ? `${note}; sending it unchanged will fail again` : note,
    };
  }
}
