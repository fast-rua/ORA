// The JSON-Schema subset of `ora/agent/composition/surface/schema_validation.py`
// (validate_schema_instance), as the artifact readers use it.

import {
  isFloat, isInt, isMapping, isNumber, num, pyEqual, pyRepr, pythonRegex, sortedStrings,
  cpLen, dumpsPy,
} from "../pyutil.ts";
import { PyFloat } from "../../pyjson.ts";

export class ToolSurfaceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ToolSurfaceError";
  }
}

export class SchemaViolation extends ToolSurfaceError {
  path: string;
  keyword: string;
  constraints: Record<string, unknown>;
  received: Record<string, unknown>;

  constructor(message: string, options: {
    path: string; keyword: string; constraints: Record<string, unknown>;
    received: Record<string, unknown>;
  }) {
    super(message);
    this.name = "SchemaViolation";
    this.path = options.path;
    this.keyword = options.keyword;
    this.constraints = { ...options.constraints };
    this.received = { ...options.received };
  }
}

const VALIDATED_KEYWORDS = new Set([
  "$ref", "$defs", "type", "const", "enum", "minimum", "maximum",
  "exclusiveMinimum", "exclusiveMaximum", "oneOf", "anyOf", "allOf",
  "properties", "required", "additionalProperties", "minProperties",
  "maxProperties", "items", "minItems", "maxItems", "uniqueItems",
  "minLength", "maxLength", "pattern",
]);
const RESULT_KEYWORDS = new Set(
  [...VALIDATED_KEYWORDS].filter((keyword) => ![
    "anyOf", "allOf", "minProperties", "maxProperties", "uniqueItems", "maxLength", "pattern",
  ].includes(keyword)),
);

function jsonKind(instance: unknown): string {
  if (instance === null || instance === undefined) return "null";
  if (typeof instance === "boolean") return "boolean";
  if (isInt(instance)) return "integer";
  if (isFloat(instance)) return "number";
  if (typeof instance === "string") return "string";
  if (Array.isArray(instance)) return "array";
  return isMapping(instance) ? "object" : typeof instance;
}

function shape(instance: unknown): Record<string, unknown> {
  const result: Record<string, unknown> = { type: jsonKind(instance) };
  if (Array.isArray(instance)) result.length = instance.length;
  if (typeof instance === "string") result.length = cpLen(instance);
  if (isMapping(instance)) result.names = sortedStrings(Object.keys(instance));
  return result;
}

/** Python ``str`` of a schema bound (an int or float). */
function boundText(bound: unknown): string {
  return dumpsPy(bound instanceof PyFloat || typeof bound !== "number" ? bound : bound);
}

const patternCache = new Map<string, RegExp>();

export function validateSchemaInstance(
  value: unknown, schema: Record<string, any>, options: { rootPath?: string } = {},
): void {
  const rootPath = options.rootPath ?? "$result";
  const valueKind = rootPath === "$input" ? "input" : "result";
  const enforced = valueKind === "input" ? VALIDATED_KEYWORDS : RESULT_KEYWORDS;

  function check(instance: unknown, node: Record<string, any>, root: Record<string, any>, path: string): void {
    const fail = (keyword: string, message: string, constraints: Record<string, unknown>): never => {
      throw new SchemaViolation(`${path} ${message}`, {
        path, keyword, constraints, received: shape(instance),
      });
    };
    const reference = node.$ref;
    if (typeof reference === "string") {
      const prefix = "#/$defs/";
      if (!reference.startsWith(prefix)) throw new ToolSurfaceError(`unsupported schema reference at ${path}`);
      const target = (root.$defs ?? {})[reference.slice(prefix.length)];
      if (!isMapping(target)) throw new ToolSurfaceError(`unresolved schema reference at ${path}`);
      check(instance, target, root, path);
      return;
    }
    const declared = node.type;
    if (declared !== undefined && declared !== null) {
      const types: string[] = Array.isArray(declared) ? declared : [declared];
      const matches = types.some((kind) => ({
        null: instance === null || instance === undefined,
        object: isMapping(instance),
        array: Array.isArray(instance),
        string: typeof instance === "string",
        boolean: typeof instance === "boolean",
        integer: isInt(instance),
        number: isNumber(instance),
      } as Record<string, boolean>)[kind] ?? false);
      if (!matches) {
        fail("type", `has the wrong ${valueKind} type; expected ${types.join(" or ")}`, { type: [...types] });
      }
    }
    if ("const" in node && !pyEqual(instance ?? null, node.const)) {
      fail("const", `differs from ${valueKind} const`, { const: node.const });
    }
    if ("enum" in node && !(node.enum as unknown[]).some((item) => pyEqual(instance ?? null, item))) {
      fail("enum", `is outside the ${valueKind} enum ${pyRepr(node.enum)}`, { enum: [...node.enum] });
    }
    if (isNumber(instance)) {
      const actual = num(instance);
      if (isFloat(instance) && !Number.isFinite(actual)) {
        fail("type", `must be a finite ${valueKind} number`, { finite: true });
      }
      const rules: [string, (a: number, b: number) => boolean][] = [
        ["minimum", (a, b) => a < b],
        ["maximum", (a, b) => a > b],
        ["exclusiveMinimum", (a, b) => a <= b],
        ["exclusiveMaximum", (a, b) => a >= b],
      ];
      for (const [keyword, violates] of rules) {
        if (!(keyword in node)) continue;
        const bound = node[keyword];
        if (!isNumber(bound) || (isFloat(bound) && !Number.isFinite(num(bound)))) {
          throw new ToolSurfaceError(`${path} has an invalid ${keyword} bound`);
        }
        if (violates(actual, num(bound))) {
          fail(keyword, `violates ${valueKind} ${keyword} ${boundText(bound)}`, { [keyword]: bound });
        }
      }
    }
    const branches = node.oneOf;
    if (Array.isArray(branches)) {
      const matched: number[] = [];
      const failures: string[] = [];
      branches.forEach((branch, index) => {
        try {
          check(instance, branch, root, path);
        } catch (error) {
          if (!(error instanceof ToolSurfaceError)) throw error;
          failures.push(`oneOf[${index}]: ${error.message}`);
          return;
        }
        matched.push(index);
      });
      if (matched.length !== 1) {
        const detail = matched.length === 0 ? failures.join("; ") : `matched oneOf branches ${pyRepr(matched)}`;
        fail("oneOf", `must match exactly one ${valueKind} branch; ${detail}`, {
          oneOf: branches.length, matched,
        });
      }
      if (!("properties" in node) && !("items" in node)) return;
    }
    if (enforced.has("anyOf") && Array.isArray(node.anyOf)) {
      const failures: string[] = [];
      let ok = false;
      for (const [index, branch] of node.anyOf.entries()) {
        try {
          check(instance, branch, root, path);
          ok = true;
          break;
        } catch (error) {
          if (!(error instanceof ToolSurfaceError)) throw error;
          failures.push(`anyOf[${index}]: ${error.message}`);
        }
      }
      if (!ok) {
        fail("anyOf", `must match at least one ${valueKind} branch; ${failures.join("; ")}`, {
          anyOf: node.anyOf.length,
        });
      }
    }
    if (enforced.has("allOf") && Array.isArray(node.allOf)) {
      for (const branch of node.allOf) check(instance, branch, root, path);
    }
    if (isMapping(instance)) {
      const properties = node.properties ?? {};
      const required: string[] = node.required ?? [];
      const missing = sortedStrings(new Set(required.filter((name) => !(name in instance))));
      if (missing.length) {
        fail("required", `lacks required ${valueKind} fields ${pyRepr(missing)}`, {
          required: [...required], missing,
        });
      }
      if (node.additionalProperties === false) {
        const unknown = sortedStrings(Object.keys(instance).filter((name) => !(name in properties)));
        if (unknown.length) {
          fail("additionalProperties", `has unknown ${valueKind} fields ${pyRepr(unknown)}`, {
            allowed: sortedStrings(Object.keys(properties)), unknown,
          });
        }
      }
      for (const [keyword, violates] of [
        ["minProperties", (size: number, bound: number) => size < bound],
        ["maxProperties", (size: number, bound: number) => size > bound],
      ] as [string, (a: number, b: number) => boolean][]) {
        if (enforced.has(keyword) && isInt(node[keyword])
          && violates(Object.keys(instance).length, num(node[keyword]))) {
          fail(keyword, `violates ${valueKind} ${keyword} ${node[keyword]}`, { [keyword]: node[keyword] });
        }
      }
      const extra = node.additionalProperties;
      for (const name of Object.keys(instance)) {
        let childSchema = isMapping(properties) ? properties[name] : undefined;
        if (!isMapping(childSchema) && isMapping(extra)) childSchema = extra;
        if (isMapping(childSchema)) check(instance[name], childSchema, root, `${path}.${name}`);
      }
    }
    if (Array.isArray(instance)) {
      if (isInt(node.minItems) && instance.length < num(node.minItems)) {
        fail("minItems", `is shorter than minItems ${node.minItems}`, { minItems: node.minItems });
      }
      if (isInt(node.maxItems) && instance.length > num(node.maxItems)) {
        fail("maxItems", `is longer than maxItems ${node.maxItems}`, { maxItems: node.maxItems });
      }
      if (enforced.has("uniqueItems") && node.uniqueItems === true) {
        const encoded = instance.map((item) => dumpsPy(item, { sortKeys: true }));
        if (new Set(encoded).size !== encoded.length) {
          fail("uniqueItems", "repeats an item; items must be unique", { uniqueItems: true });
        }
      }
      const itemSchema = node.items;
      if (isMapping(itemSchema)) {
        instance.forEach((child, index) => check(child, itemSchema, root, `${path}[${index}]`));
      }
    }
    if (typeof instance === "string") {
      const length = cpLen(instance);
      if (isInt(node.minLength) && length < num(node.minLength)) {
        fail("minLength", `is shorter than minLength ${node.minLength}`, { minLength: node.minLength });
      }
      if (enforced.has("maxLength") && isInt(node.maxLength) && length > num(node.maxLength)) {
        fail("maxLength", `is longer than maxLength ${node.maxLength}`, { maxLength: node.maxLength });
      }
      if (enforced.has("pattern") && typeof node.pattern === "string") {
        let regex = patternCache.get(node.pattern);
        if (regex === undefined) {
          regex = pythonRegex(node.pattern, "");
          patternCache.set(node.pattern, regex);
        }
        if (!regex.test(instance)) {
          fail("pattern", `does not match pattern ${node.pattern}`, { pattern: node.pattern });
        }
      }
    }
  }

  check(value, schema, schema, rootPath);
}
