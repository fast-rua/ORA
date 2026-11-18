// Port of ora/agent/providers/openai_wire_options.py: profile-owned options of
// the OpenAI-compatible Chat Completions wire and usage normalization.

import { PyFloat, clonePy } from "../pyjson.ts";
import { PyError } from "./py.ts";

export const TOOL_STRICTNESS_MODES = new Set(["explicit_false"]);
export const REASONING_FIELDS = new Set(["reasoning_content", "reasoning"]);
/** Closed keys of a profile's `request_options` (openai_wire_options.REQUEST_OPTION_FIELDS). */
export const REQUEST_OPTION_FIELDS = new Set([
  "temperature", "top_p", "top_k", "chat_template_kwargs", "reasoning", "provider", "return_token_ids",
]);

type Json = { [key: string]: any };

function isMapping(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !(value instanceof PyFloat);
}

function has(value: Json, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

/** The wire tools with `"strict": false` stated on every function tool. */
export function projectToolStrictness(tools: unknown, toolStrictness: string | null): unknown {
  if (toolStrictness === null) return tools;
  if (!TOOL_STRICTNESS_MODES.has(toolStrictness)) {
    throw new PyError("ValueError", `unsupported tool_strictness: ${toolStrictness}`);
  }
  if (!Array.isArray(tools)) return tools;
  return tools.map((tool) => {
    if (isMapping(tool) && isMapping(tool.function)) {
      const copy = clonePy(tool);
      // `{**tool["function"], "strict": False}` keeps key order, overwrites strict in place.
      const fn: Json = { ...tool.function };
      fn.strict = false;
      copy.function = fn;
      return copy;
    }
    return tool;
  });
}

/** Python `repr()` of a str, as it appears inside a list in an error message. */
function pyStrRepr(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const char of text) {
    const code = char.codePointAt(0)!;
    if (char === "\\" || char === quote) out += `\\${char}`;
    else if (char === "\n") out += "\\n";
    else if (char === "\r") out += "\\r";
    else if (char === "\t") out += "\\t";
    else if (code < 0x20 || code === 0x7f) out += `\\x${code.toString(16).padStart(2, "0")}`;
    else out += char;
  }
  return out + quote;
}

/** Python `repr(sorted(keys))` for a list of str. */
function pySortedKeys(keys: Iterable<string>): string {
  return `[${[...keys].sort().map(pyStrRepr).join(", ")}]`;
}

/** `_json_value`: None, bool, str, a finite number, or lists and str-keyed mappings of them. */
function jsonValue(value: unknown): boolean {
  if (value === null || typeof value === "boolean" || typeof value === "string" || typeof value === "bigint") return true;
  if (value instanceof PyFloat) return Number.isFinite(value.value);
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(jsonValue);
  if (isMapping(value)) return Object.values(value).every(jsonValue);
  return false;
}

/** A Python int or float (not bool) as a JS number, or null. */
function pyNumber(value: unknown): number | null {
  if (value instanceof PyFloat) return value.value;
  if (typeof value === "bigint") return Number(value);
  return typeof value === "number" ? value : null;
}

/**
 * `validate_request_options`: a copy of valid profile `request_options`, or null for none;
 * a `ValueError` with Python's message otherwise.
 */
export function validateRequestOptions(value: unknown): Json | null {
  if (value === null || value === undefined) return null;
  if (!isMapping(value) || Object.keys(value).length === 0) {
    throw new PyError("ValueError", "request_options must be a non-empty mapping");
  }
  const unknown = Object.keys(value).filter((key) => !REQUEST_OPTION_FIELDS.has(key));
  if (unknown.length) {
    throw new PyError("ValueError", `unsupported request_options fields: ${pySortedKeys(unknown)}`);
  }
  if (!jsonValue(value)) throw new PyError("ValueError", "request_options must hold finite JSON values");
  if (has(value, "temperature")) {
    const temperature = pyNumber(value.temperature);
    if (temperature === null || !(temperature >= 0 && temperature <= 2)) {
      throw new PyError("ValueError", "request_options.temperature must be a number in 0..2");
    }
  }
  if (has(value, "top_p")) {
    const topP = pyNumber(value.top_p);
    if (topP === null || !(topP > 0 && topP <= 1)) {
      throw new PyError("ValueError", "request_options.top_p must be a number in (0, 1]");
    }
  }
  if (has(value, "top_k")) {
    // A Python int: a plain integral number or a bigint (an integral float crosses as PyFloat).
    const topK = value.top_k;
    const integer = typeof topK === "bigint" || (typeof topK === "number" && Number.isInteger(topK));
    const number = integer ? Number(topK) : null;
    if (number === null || !(number === -1 || number >= 1)) {
      throw new PyError("ValueError", "request_options.top_k must be -1 or a positive integer");
    }
  }
  for (const field of ["chat_template_kwargs", "reasoning", "provider"]) {
    if (has(value, field) && (!isMapping(value[field]) || Object.keys(value[field]).length === 0)) {
      throw new PyError("ValueError", `request_options.${field} must be a non-empty mapping`);
    }
  }
  if (has(value, "return_token_ids") && value.return_token_ids !== true) {
    throw new PyError("ValueError", "request_options.return_token_ids must be true");
  }
  return clonePy(value);
}

/** Apply profile-owned options to an assembled Chat Completions body. */
export function projectWireBody(
  body: Json, options: { toolStrictness: string | null; reasoningField?: string; requestOptions?: Json | null },
): Json {
  const reasoningField = options.reasoningField ?? "reasoning_content";
  if (!REASONING_FIELDS.has(reasoningField)) {
    throw new PyError("ValueError", `unsupported reasoning_field: ${reasoningField}`);
  }
  // A field the request already carries is refused rather than overwritten.
  const requestOptions = validateRequestOptions(options.requestOptions) ?? {};
  const collisions = Object.keys(requestOptions).filter((key) => has(body, key));
  if (collisions.length) {
    throw new PyError("ValueError", `request_options collide with request fields: ${pySortedKeys(collisions)}`);
  }
  Object.assign(body, requestOptions);
  // "No tools" is an absent field on this wire (vLLM rejects "tools": []).
  if (Array.isArray(body.tools) && body.tools.length === 0) delete body.tools;
  if (has(body, "tools")) body.tools = projectToolStrictness(body.tools, options.toolStrictness);
  if (reasoningField !== "reasoning_content") {
    const messages = has(body, "messages") ? body.messages : [];
    for (const message of Array.isArray(messages) ? messages : []) {
      if (isMapping(message) && has(message, "reasoning_content")) {
        const value = message.reasoning_content;
        delete message.reasoning_content;
        message[reasoningField] = value;
      }
    }
  }
  return body;
}

/** Provider-owned settings for short, tool-free checkpoint calls. */
export function checkpointRequestOptions(model: string): Json {
  if (["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"].includes(model)) {
    return { reasoning_effort: "low", response_format: { type: "json_object" } };
  }
  return {};
}

/** A reported token count, or null when absent, boolean or negative (`int(value)`). */
export function nonnegativeCount(value: unknown): number | bigint | null {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  const number = value instanceof PyFloat ? value.value : value;
  if (typeof number !== "number") return null;
  if (Number.isNaN(number)) return null;
  if (!(number >= 0)) return null;
  if (!Number.isFinite(number)) throw new PyError("OverflowError", "cannot convert float infinity to integer");
  return Math.trunc(number);
}

/** Chat Completions usage in the Harness's provider-neutral usage shape. */
export function normalizeUsage(value: unknown): Json | null {
  if (!isMapping(value)) return null;
  const details = isMapping(value.prompt_tokens_details) ? value.prompt_tokens_details : {};
  const completionDetails = isMapping(value.completion_tokens_details) ? value.completion_tokens_details : {};
  const cachedValue = has(details, "cached_tokens")
    ? nonnegativeCount(details.cached_tokens)
    : nonnegativeCount(value.prompt_cache_hit_tokens);
  const cacheReadReported = has(details, "cached_tokens") || has(value, "prompt_cache_hit_tokens");
  const output: Json = {
    input_tokens: nonnegativeCount(value.prompt_tokens),
    cached_input_tokens: cachedValue,
    output_tokens: nonnegativeCount(value.completion_tokens),
    // sglang reports reasoning at the top level of usage, OpenAI and vLLM in completion_tokens_details.
    reasoning_output_tokens: nonnegativeCount(
      has(completionDetails, "reasoning_tokens") ? completionDetails.reasoning_tokens : value.reasoning_tokens,
    ),
    total_tokens: nonnegativeCount(value.total_tokens),
  };
  if (Object.values(output).every((item) => item === null)) return null;
  const normalized: Json = {};
  for (const key of Object.keys(output)) normalized[key] = output[key] === null ? 0 : output[key];
  normalized.cache_read_input_tokens = cacheReadReported ? cachedValue : null;
  normalized.cache_creation_input_tokens = null;
  normalized.cache_read_status = cacheReadReported ? "reported" : "missing";
  normalized.cache_creation_status = "missing";
  return normalized;
}
