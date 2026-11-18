// Port of AnthropicMessagesProvider (ora/agent/providers/anthropic_messages.py) and
// ora/agent/providers/{anthropic_events,anthropic_history}.py: the exact
// Anthropic Messages wire projection, streamed-event accumulation, the
// signed-thinking continuation and the provider's retry and credential-ring
// behaviour. Error texts, retry delays (int vs float) and record fields follow
// Python exactly, because they are evidence.

import { canonicalBytesPy, clonePy, dumpsPy, parsePy, PyFloat, utf8Strict } from "../pyjson.ts";
import {
  attemptFailureRecord, echoFailure, partialCredentialEcho, recordProviderFailure, CREDENTIAL_ECHO_MESSAGE,
} from "./attempts.ts";
import {
  PROVIDER_CONTINUATION_MESSAGE_KEY, ProviderContinuationState, canonicalAssistantSha256,
} from "./continuation.ts";
import { rejectTruncatedTools } from "./openai.ts";
import {
  PolicySessionError, ProviderClosed, ProviderRecoveryDeadline, PyEOFError, StreamInterrupted, attr, pyFloat,
  pyRound6, pyStrip, pyTail, sha256Hex, type PyNum,
} from "./py.ts";
import {
  type Clock, LEGACY_RECOVERY_POLICY, RECOVERY_POLICIES, RecoveryWindow, SYSTEM_CLOCK, transientTransport,
} from "./recovery.ts";
import { providerFetch } from "./http.ts";
import { httpStatusFailure } from "./stream_failures.ts";
import { toolAlias } from "./tool_names.ts";
import {
  AttemptControl, ProviderAborted, abortedError, boundedRequest, decodeReplace, lines, loadsBytes, loadsText, post,
  readAll, type AttemptResult, type TransportBinding,
} from "./transport.ts";
import { nonnegativeCount } from "./wire_options.ts";

type Json = { [key: string]: any };

export const ANTHROPIC_PROTOCOL = "anthropic_messages";
export const ANTHROPIC_CONTINUATION_KIND = "anthropic_assistant_content_blocks.v1";
const CONTINUATION_REPLAY_MODES = new Set(["exact_v1"]);
const FORBIDDEN_FIELDS = ["input", "instructions", "system", "thinking", "output_config", "input_schema"];
const DATA_URL = /^data:([^;,]+);base64,(.*)$/s;
const KIMI_CODING_BASE_URLS = new Set([
  "https://api.kimi.com/coding",
  "https://api.kimi.com/coding/v1",
  "https://api.kimi.com/coding/v1/messages",
]);
const KIMI_TRANSIENT_SIGNATURE_REJECTION = "malformed encrypted reasoning content: invalid base64url encoding";
const AFFINITY_ID = /^[A-Za-z0-9:._-]{1,200}$/;

function isMapping(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && !(value instanceof PyFloat) && !(value instanceof Uint8Array);
}

function has(value: Json, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function get(value: Json, key: string, fallback: any = null): any {
  return has(value, key) ? value[key] : fallback;
}

/** Python `isinstance(value, int)` (bool included) and its integer. */
function pyInt(value: unknown): bigint | null {
  if (typeof value === "boolean") return value ? 1n : 0n;
  if (typeof value === "bigint") return value;
  if (typeof value === "number" && Number.isInteger(value)) return BigInt(value);
  return null;
}

function codePointLength(text: string): number {
  let count = 0;
  for (const _ of text) count += 1;
  return count;
}

/** `json.loads` raising for anything Python would refuse (NaN etc. are refused here too). */
function loads(text: string): any {
  return parsePy(text);
}

/**
 * `base64.b64decode(data, altchars, validate=True)` succeeds: Python 3.11's
 * `binascii.a2b_base64(strict_mode=True)` after the altchars translation.
 */
function validBase64(data: string, altchars: string | null = null): boolean {
  let text = data;
  if (altchars !== null) text = text.replaceAll(altchars[0], "+").replaceAll(altchars[1], "/");
  if (!/^[A-Za-z0-9+/=]*$/.test(text)) return false; // non-alphabet character
  if (text.startsWith("=")) return false; // leading padding
  const firstPad = text.indexOf("=");
  const dataPart = firstPad === -1 ? text : text.slice(0, firstPad);
  const padPart = firstPad === -1 ? "" : text.slice(firstPad);
  if (/[^=]/.test(padPart)) return false; // excess data after / discontinuous padding
  const remainder = dataPart.length % 4;
  if (remainder === 1) return false;
  if (remainder === 2) return padPart.length >= 2;
  if (remainder === 3) return padPart.length >= 1;
  return true;
}

// -- Request translation -------------------------------------------------------------------

export function anthropicToolAlias(name: string): string {
  return toolAlias(name);
}

/** `_anthropic_tool_names`: alias -> canonical name over the request's tools. */
export function anthropicToolNames(request: Json): Map<string, string> {
  const aliases = new Map<string, string>();
  const rawTools = get(request, "tools", []);
  if (!Array.isArray(rawTools)) return aliases;
  for (const item of rawTools) {
    const fn = isMapping(item) ? get(item, "function") : null;
    const name = isMapping(fn) ? get(fn, "name") : null;
    if (typeof name !== "string") continue;
    const alias = anthropicToolAlias(name);
    const previous = aliases.get(alias);
    if (previous === undefined) aliases.set(alias, name);
    else if (previous !== name) throw new PolicySessionError("Anthropic tool-name alias collision");
  }
  return aliases;
}

/** `_anthropic_content`: canonical text/image content as Anthropic blocks. */
function anthropicContent(content: unknown): Json[] {
  if (typeof content === "string") return content ? [{ type: "text", text: content }] : [];
  if (!Array.isArray(content)) throw new PolicySessionError("canonical message content must be text or a block list");
  const blocks: Json[] = [];
  for (const item of content) {
    if (!isMapping(item)) throw new PolicySessionError("canonical message content block must be an object");
    const blockType = get(item, "type");
    if (blockType === "text" && typeof get(item, "text") === "string") {
      blocks.push({ type: "text", text: item.text });
      continue;
    }
    if (blockType === "image_url") {
      const imageUrl = get(item, "image_url");
      const url = isMapping(imageUrl) ? get(imageUrl, "url") : imageUrl;
      if (typeof url !== "string") throw new PolicySessionError("canonical image_url block lacks a URL");
      const match = DATA_URL.exec(url);
      if (match === null) throw new PolicySessionError("Anthropic adapter accepts only inline base64 image data URLs");
      const [, mediaType, data] = match;
      if (!validBase64(data)) throw new PolicySessionError("canonical image data URL is invalid base64");
      blocks.push({ type: "image", source: { type: "base64", media_type: mediaType, data } });
      continue;
    }
    throw new PolicySessionError(`unsupported canonical content block: ${pyRepr(blockType)}`);
  }
  return blocks;
}

/** Python `repr()` of a JSON value, for the few messages that quote one. */
function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "string") {
    const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
    const body = value.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
    return quote + (quote === "'" ? body.replace(/'/g, "\\'") : body) + quote;
  }
  if (value instanceof PyFloat) return dumpsPy(value);
  if (typeof value === "number" || typeof value === "bigint") return String(value);
  return dumpsPy(value);
}

/** `_anthropic_tool_input`: (id, name, arguments) of one canonical tool call. */
function anthropicToolInput(call: Json): [string, string, Json] {
  const fn = get(call, "function");
  if (!isMapping(fn) || typeof get(fn, "name") !== "string") {
    throw new PolicySessionError("canonical tool call lacks a function name");
  }
  const raw = get(fn, "arguments", "{}");
  let args: any;
  if (isMapping(raw)) args = clonePy(raw);
  else if (typeof raw === "string") {
    try {
      args = loads(raw);
    } catch {
      throw new PolicySessionError("canonical tool arguments are invalid JSON");
    }
  } else throw new PolicySessionError("canonical tool arguments must be JSON text or an object");
  if (!isMapping(args)) throw new PolicySessionError("canonical tool arguments must decode to an object");
  const callId = get(call, "id");
  if (typeof callId !== "string" || !callId) throw new PolicySessionError("canonical tool call lacks an ID");
  return [callId, fn.name, args];
}

/** `_normalize_anthropic_blocks`: finalized assistant blocks as the canonical message. */
export function normalizeAnthropicBlocks(
  blocks: Json[], options: { modelCallId: string; toolNamesByAlias: Map<string, string> | null },
): Json {
  const textParts: string[] = [];
  const thinkingParts: string[] = [];
  const calls: Json[] = [];
  blocks.forEach((block, index) => {
    const blockType = get(block, "type");
    if (blockType === "text") {
      const text = get(block, "text");
      if (typeof text !== "string") throw new PolicySessionError("Anthropic text block lacks text");
      textParts.push(text);
      return;
    }
    if (blockType === "thinking") {
      const thinking = get(block, "thinking");
      const signature = get(block, "signature");
      if (typeof thinking !== "string") throw new PolicySessionError("Anthropic thinking block lacks thinking text");
      if (typeof signature !== "string" || !signature) {
        throw new PolicySessionError("Anthropic thinking block lacks a replayable signature");
      }
      thinkingParts.push(thinking);
      return;
    }
    if (blockType === "redacted_thinking") {
      const data = get(block, "data");
      if (typeof data !== "string" || !data) throw new PolicySessionError("Anthropic redacted_thinking block lacks opaque data");
      return;
    }
    if (blockType === "tool_use") {
      const args = get(block, "input");
      if (!isMapping(args)) throw new PolicySessionError("Anthropic tool input must be an object");
      let name = get(block, "name");
      if (typeof name !== "string" || !name) throw new PolicySessionError("Anthropic tool_use lacks a name");
      if (options.toolNamesByAlias !== null) name = options.toolNamesByAlias.get(name) ?? name;
      let callId = get(block, "id");
      if (typeof callId !== "string" || !callId) callId = `${options.modelCallId}-tool-${String(index).padStart(2, "0")}`;
      calls.push({
        id: callId, type: "function",
        function: { name, arguments: dumpsPy(args, { ensureAscii: false, sortKeys: true, separators: [",", ":"] }) },
      });
      return;
    }
    throw new PolicySessionError(`unsupported Anthropic assistant content block: ${pyRepr(blockType)}`);
  });
  const message: Json = { role: "assistant", content: textParts.join("") };
  if (thinkingParts.length) message.reasoning_content = thinkingParts.join("");
  if (calls.length) message.tool_calls = calls;
  return message;
}

/** `_anthropic_continuation_blocks`: the signed blocks to replay for one assistant, or null. */
function continuationBlocks(message: Json, model: string, toolNamesByAlias: Map<string, string>): Json[] | null {
  if (!has(message, PROVIDER_CONTINUATION_MESSAGE_KEY)) return null;
  const rawState = message[PROVIDER_CONTINUATION_MESSAGE_KEY];
  if (rawState === null) return null;
  if (!isMapping(rawState)) throw new PolicySessionError("provider continuation state must be an object");
  const state = ProviderContinuationState.fromMapping(rawState);
  state.validateBinding({ message, providerProtocol: ANTHROPIC_PROTOCOL, model });
  if (state.nativeKind !== ANTHROPIC_CONTINUATION_KIND) {
    throw new PolicySessionError("unsupported Anthropic continuation native kind");
  }
  const rawBlocks = get(state.nativePayload, "content");
  if (!Array.isArray(rawBlocks) || !rawBlocks.length) {
    throw new PolicySessionError("Anthropic continuation lacks assistant content blocks");
  }
  if (!rawBlocks.every(isMapping)) throw new PolicySessionError("Anthropic continuation content block is invalid");
  const blocks = rawBlocks.map((block) => clonePy(block));
  const replayToolNames = new Map(toolNamesByAlias);
  const canonicalCalls = Array.isArray(get(message, "tool_calls")) ? message.tool_calls : [];
  const canonicalNamesById = new Map<string, string>();
  for (const rawCall of canonicalCalls) {
    if (!isMapping(rawCall)) continue;
    const callId = get(rawCall, "id");
    const fn = get(rawCall, "function");
    const name = isMapping(fn) ? get(fn, "name") : null;
    if (typeof callId === "string" && callId && typeof name === "string" && name) canonicalNamesById.set(callId, name);
  }
  for (const block of blocks) {
    if (get(block, "type") !== "tool_use") continue;
    const callId = get(block, "id");
    const wireName = get(block, "name");
    const canonicalName = typeof callId === "string" ? canonicalNamesById.get(callId) : undefined;
    if (!(typeof wireName === "string" && wireName && typeof canonicalName === "string" && canonicalName)) continue;
    const previous = replayToolNames.get(wireName);
    if (previous === undefined) replayToolNames.set(wireName, canonicalName);
    else if (previous !== canonicalName) {
      throw new PolicySessionError("Anthropic continuation tool-name binding conflicts with the request");
    }
  }
  const normalized = normalizeAnthropicBlocks(blocks, {
    modelCallId: state.responseId || "provider-continuation", toolNamesByAlias: replayToolNames,
  });
  if (canonicalAssistantSha256(normalized) !== state.canonicalMessageSha256) {
    throw new PolicySessionError("Anthropic continuation payload does not match its canonical assistant");
  }
  return blocks;
}

/** `translate_anthropic_request`: the canonical native request as an Anthropic Messages body. */
export function translateAnthropicRequest(request: Json): Json {
  const forbidden = FORBIDDEN_FIELDS.filter((key) => has(request, key)).sort();
  if (forbidden.length) {
    throw new PolicySessionError("Anthropic canonical request contains wire/Responses fields: " + forbidden.join(", "));
  }
  const model = get(request, "model");
  const maxTokens = get(request, "max_tokens");
  const messages = get(request, "messages");
  if (typeof model !== "string" || !model) throw new PolicySessionError("canonical provider request lacks a model");
  const maxInt = typeof maxTokens === "boolean" ? null : pyInt(maxTokens);
  if (maxInt === null || maxInt <= 0n) throw new PolicySessionError("canonical provider request has invalid max_tokens");
  if (!Array.isArray(messages)) throw new PolicySessionError("canonical provider request messages must be a list");

  const systemBlocks: Json[] = [];
  const translated: Json[] = [];
  const toolNamesByAlias = anthropicToolNames(request);
  const appendMessage = (role: string, blocks: Json[]) => {
    if (!blocks.length) return;
    const last = translated[translated.length - 1];
    if (last !== undefined && last.role === role) {
      const existing: Json[] = last.content;
      if (get(blocks[0], "type") === "tool_result" && existing.some((item) => get(item, "type") !== "tool_result")) {
        throw new PolicySessionError("Anthropic tool results must immediately follow tool_use blocks");
      }
      existing.push(...blocks);
    } else {
      translated.push({ role, content: blocks });
    }
  };

  for (const rawMessage of messages) {
    if (!isMapping(rawMessage)) throw new PolicySessionError("canonical provider message must be an object");
    const role = get(rawMessage, "role");
    if (role === "system") {
      systemBlocks.push(...anthropicContent(get(rawMessage, "content", "")));
      continue;
    }
    if (role === "user") {
      appendMessage("user", anthropicContent(get(rawMessage, "content", "")));
      continue;
    }
    if (role === "assistant") {
      let blocks = continuationBlocks(rawMessage, model, toolNamesByAlias);
      if (blocks === null) {
        blocks = anthropicContent(get(rawMessage, "content", ""));
        const calls = get(rawMessage, "tool_calls");
        if (Array.isArray(calls)) {
          for (const rawCall of calls) {
            if (!isMapping(rawCall)) throw new PolicySessionError("canonical tool call must be an object");
            const [callId, name, args] = anthropicToolInput(rawCall);
            blocks.push({ type: "tool_use", id: callId, name: anthropicToolAlias(name), input: args });
          }
        }
      }
      appendMessage("assistant", blocks);
      continue;
    }
    if (role === "tool") {
      const callId = get(rawMessage, "tool_call_id");
      if (typeof callId !== "string" || !callId) throw new PolicySessionError("canonical tool result lacks tool_call_id");
      let content = get(rawMessage, "content", "");
      if (typeof content !== "string") content = dumpsPy(content, { ensureAscii: false, sortKeys: true });
      let isError = false;
      try {
        const decoded = loads(content);
        isError = isMapping(decoded) && get(decoded, "ok") === false;
      } catch {
        // not JSON: an ordinary text result
      }
      const block: Json = { type: "tool_result", tool_use_id: callId, content };
      if (isError) block.is_error = true;
      appendMessage("user", [block]);
      continue;
    }
    throw new PolicySessionError(`unsupported canonical message role: ${pyRepr(role)}`);
  }

  const renamed: Json = {};
  for (const [alias, canonical] of toolNamesByAlias) if (alias !== canonical) renamed[canonical] = alias;
  if (Object.keys(renamed).length) {
    systemBlocks.push({
      type: "text",
      text: "<wire_tool_aliases>\n" + dumpsPy(renamed, { ensureAscii: false, sortKeys: true }) + "\n</wire_tool_aliases>",
    });
  }

  const tools: Json[] = [];
  const rawTools = get(request, "tools", []);
  if (!Array.isArray(rawTools)) throw new PolicySessionError("canonical provider tools must be a list");
  for (const item of rawTools) {
    const fn = isMapping(item) ? get(item, "function") : null;
    if (!isMapping(fn)) throw new PolicySessionError("canonical tool definition lacks function");
    const name = get(fn, "name");
    const schema = get(fn, "parameters");
    if (typeof name !== "string" || !isMapping(schema)) throw new PolicySessionError("canonical tool definition is incomplete");
    let wireSchema: Json = clonePy(schema);
    if (!has(wireSchema, "type") && has(wireSchema, "oneOf")) wireSchema = { type: "object", ...wireSchema };
    tools.push({ name: anthropicToolAlias(name), description: get(fn, "description", ""), input_schema: wireSchema });
  }

  const body: Json = { model, messages: translated, max_tokens: maxTokens, stream: Boolean(pyTruthy(get(request, "stream", false))) };
  if (systemBlocks.length) body.system = systemBlocks;
  if (tools.length) body.tools = tools;
  const toolChoice = get(request, "tool_choice");
  if (toolChoice === "auto") body.tool_choice = { type: "auto" };
  else if (toolChoice === "none") body.tool_choice = { type: "none" };
  else if (isMapping(toolChoice)) {
    const fn = get(toolChoice, "function");
    const name = isMapping(fn) ? get(fn, "name") : null;
    if (typeof name === "string") body.tool_choice = { type: "tool", name: anthropicToolAlias(name) };
  }
  for (const field of ["temperature", "top_p"]) if (has(request, field)) body[field] = request[field];
  const effort = get(request, "reasoning_effort");
  if (effort !== null && effort !== "provider_default") {
    if (!["low", "medium", "high", "xhigh", "max"].includes(effort)) {
      throw new PolicySessionError("Anthropic reasoning_effort cannot be projected to output_config");
    }
    body.thinking = { type: "adaptive" };
    body.output_config = { effort };
  }
  const stop = get(request, "stop");
  if (typeof stop === "string") body.stop_sequences = [stop];
  else if (Array.isArray(stop)) body.stop_sequences = stop;
  return body;
}

/** Python truthiness of a JSON value (`bool(value)`). */
function pyTruthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false) return false;
  if (value === 0 || value === 0n || value === "") return false;
  if (value instanceof PyFloat) return value.value !== 0;
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  return true;
}

/** `_has_complete_anthropic_thinking_signature`: the wire replays one decodable signature. */
export function hasCompleteThinkingSignature(body: Json): boolean {
  const messages = get(body, "messages");
  if (!Array.isArray(messages)) return false;
  for (const message of messages) {
    if (!isMapping(message) || get(message, "role") !== "assistant") continue;
    const content = get(message, "content");
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isMapping(block) || get(block, "type") !== "thinking") continue;
      const signature = get(block, "signature");
      if (typeof signature !== "string" || !signature) continue;
      if (!/^[\x00-\x7f]*$/.test(signature)) continue;
      const padded = signature + "=".repeat((4 - (signature.length % 4)) % 4);
      if (validBase64(padded) || validBase64(padded, "-_")) return true;
    }
  }
  return false;
}

// -- Event accumulation (anthropic_events.py) -------------------------------------------------

export interface AnthropicResponseEvents {
  blocksByIndex: Map<bigint, Json>;
  toolFragments: Map<bigint, string[]>;
  usageValues: Json;
  stopReason: string | null;
  responseId: string | null;
  streaming: boolean;
  stoppedBlocks: Set<bigint>;
  messageStartSeen: boolean;
  messageStopSeen: boolean;
  nonStreamMessageSeen: boolean;
}

function mergeUsage(state: AnthropicResponseEvents, value: unknown): void {
  if (!isMapping(value)) return;
  for (const [key, amount] of Object.entries(value)) {
    if (nonnegativeCount(amount) !== null) state.usageValues[key] = amount;
  }
}

function sameKeys(stopped: Set<bigint>, blocks: Map<bigint, Json>): boolean {
  if (stopped.size !== blocks.size) return false;
  for (const key of blocks.keys()) if (!stopped.has(key)) return false;
  return true;
}

function streamError(event: Json): PolicySessionError {
  const error = get(event, "error");
  const errorType = isMapping(error) ? pyStr(get(error, "type")) : "provider_error";
  const failure = new PolicySessionError(`Anthropic provider stream returned ${errorType}`);
  failure.retryable = errorType === "overloaded_error";
  return failure;
}

/** Python `str(value)` for a JSON value. */
function pyStr(value: unknown): string {
  return typeof value === "string" ? value : pyRepr(value);
}

function applyEvent(state: AnthropicResponseEvents, event: Json): void {
  const eventType = get(event, "type");
  if (state.messageStopSeen) throw new PolicySessionError("Anthropic provider returned content after message_stop");
  if (eventType === "ping") return;
  if (eventType === "error") throw streamError(event);
  if (eventType === "message_start") {
    if (state.messageStartSeen || state.streaming) throw new PolicySessionError("Anthropic message_start was duplicated");
    state.streaming = true;
    state.messageStartSeen = true;
    const message = get(event, "message");
    if (isMapping(message)) {
      const id = get(message, "id");
      if (typeof id === "string" && id) state.responseId = id;
      mergeUsage(state, get(message, "usage"));
      const stop = get(message, "stop_reason");
      if (typeof stop === "string") state.stopReason = stop;
    }
    return;
  }
  if (eventType === "content_block_start") {
    if (!state.messageStartSeen) throw new PolicySessionError("Anthropic content block precedes message_start");
    state.streaming = true;
    const index = pyInt(get(event, "index"));
    const block = get(event, "content_block");
    if (index === null || !isMapping(block)) throw new PolicySessionError("Anthropic content_block_start is incomplete");
    const blockType = get(block, "type");
    if (!["text", "thinking", "redacted_thinking", "tool_use"].includes(blockType)) {
      throw new PolicySessionError(`unsupported Anthropic assistant content block: ${pyRepr(blockType)}`);
    }
    if (state.blocksByIndex.has(index)) throw new PolicySessionError("Anthropic assistant content block index was reused");
    state.blocksByIndex.set(index, clonePy(block));
    if (blockType === "tool_use") state.toolFragments.set(index, []);
    return;
  }
  if (eventType === "content_block_delta") {
    state.streaming = true;
    const index = pyInt(get(event, "index"));
    const delta = get(event, "delta");
    if (index === null || !isMapping(delta) || !state.blocksByIndex.has(index) || state.stoppedBlocks.has(index)) {
      throw new PolicySessionError("Anthropic content block delta lacks its block start");
    }
    const block = state.blocksByIndex.get(index)!;
    const deltaType = get(delta, "type");
    if (deltaType === "text_delta" && typeof get(delta, "text") === "string") {
      if (get(block, "type") !== "text") throw new PolicySessionError("Anthropic text delta targets a non-text block");
      block.text = pyStr(get(block, "text", "")) + delta.text;
    } else if (deltaType === "thinking_delta" && typeof get(delta, "thinking") === "string") {
      if (get(block, "type") !== "thinking") throw new PolicySessionError("Anthropic thinking delta targets a non-thinking block");
      block.thinking = pyStr(get(block, "thinking", "")) + delta.thinking;
    } else if (deltaType === "signature_delta" && typeof get(delta, "signature") === "string") {
      if (get(block, "type") !== "thinking") throw new PolicySessionError("Anthropic signature delta targets a non-thinking block");
      block.signature = pyStr(get(block, "signature", "")) + delta.signature;
    } else if (deltaType === "input_json_delta" && typeof get(delta, "partial_json") === "string") {
      if (get(block, "type") !== "tool_use") throw new PolicySessionError("Anthropic tool-input delta targets a non-tool block");
      if (!state.toolFragments.has(index)) state.toolFragments.set(index, []);
      state.toolFragments.get(index)!.push(delta.partial_json);
    } else {
      throw new PolicySessionError(`unsupported Anthropic content block delta: ${pyRepr(deltaType)}`);
    }
    return;
  }
  if (eventType === "content_block_stop") {
    if (!state.messageStartSeen) throw new PolicySessionError("Anthropic content block stop precedes message_start");
    state.streaming = true;
    const index = pyInt(get(event, "index"));
    if (index === null || !state.blocksByIndex.has(index) || state.stoppedBlocks.has(index)) {
      throw new PolicySessionError("Anthropic content_block_stop lacks one active block");
    }
    state.stoppedBlocks.add(index);
    return;
  }
  if (eventType === "message_delta") {
    if (!state.messageStartSeen || !sameKeys(state.stoppedBlocks, state.blocksByIndex)) {
      throw new PolicySessionError("Anthropic message_delta precedes completed content blocks");
    }
    state.streaming = true;
    const delta = get(event, "delta");
    if (isMapping(delta) && typeof get(delta, "stop_reason") === "string") state.stopReason = delta.stop_reason;
    mergeUsage(state, get(event, "usage"));
    return;
  }
  if (eventType === "message_stop") {
    state.streaming = true;
    if (!state.messageStartSeen || !sameKeys(state.stoppedBlocks, state.blocksByIndex)) {
      throw new PolicySessionError("Anthropic message_stop precedes completed content blocks");
    }
    state.messageStopSeen = true;
    return;
  }
  // Non-streaming Anthropic response.
  if (Array.isArray(get(event, "content"))) {
    if (state.streaming || state.nonStreamMessageSeen) {
      throw new PolicySessionError("Anthropic response mixed streaming and non-streaming content");
    }
    state.nonStreamMessageSeen = true;
    const id = get(event, "id");
    if (typeof id === "string" && id) state.responseId = id;
    const stop = get(event, "stop_reason");
    if (typeof stop === "string") state.stopReason = stop;
    mergeUsage(state, get(event, "usage"));
    (event.content as any[]).forEach((block, index) => {
      if (!isMapping(block)) throw new PolicySessionError("Anthropic assistant content block must be an object");
      state.blocksByIndex.set(BigInt(index), clonePy(block));
    });
    return;
  }
  throw new PolicySessionError(`unsupported Anthropic response event: ${pyRepr(eventType)}`);
}

/** `accumulate_anthropic_events`. */
export function accumulateAnthropicEvents(rawEvents: Json[]): AnthropicResponseEvents {
  const state: AnthropicResponseEvents = {
    blocksByIndex: new Map(), toolFragments: new Map(), usageValues: {}, stopReason: null, responseId: null,
    streaming: false, stoppedBlocks: new Set(), messageStartSeen: false, messageStopSeen: false, nonStreamMessageSeen: false,
  };
  for (const event of rawEvents) applyEvent(state, isMapping(event) ? event : {});
  if (state.streaming && (!state.messageStartSeen || !state.messageStopSeen
      || !sameKeys(state.stoppedBlocks, state.blocksByIndex))) {
    throw new PolicySessionError("Anthropic streaming response ended before all blocks completed");
  }
  return state;
}

/** `normalize_anthropic_usage`. */
export function normalizeAnthropicUsage(value: unknown): Json | null {
  if (!isMapping(value)) return null;
  const uncached = nonnegativeCount(get(value, "input_tokens"));
  const cacheRead = nonnegativeCount(get(value, "cache_read_input_tokens"));
  const cacheWrite = nonnegativeCount(get(value, "cache_creation_input_tokens"));
  const output = nonnegativeCount(get(value, "output_tokens"));
  if ([uncached, cacheRead, cacheWrite, output].every((item) => item === null)) return null;
  const num = (item: number | bigint | null) => (item === null ? 0 : Number(item));
  const totalInput = num(uncached) + num(cacheRead) + num(cacheWrite);
  const outputTokens = num(output);
  return {
    input_tokens: totalInput,
    cached_input_tokens: num(cacheRead),
    cache_read_input_tokens: cacheRead === null ? null : Number(cacheRead),
    cache_creation_input_tokens: cacheWrite === null ? null : Number(cacheWrite),
    cache_read_status: has(value, "cache_read_input_tokens") ? "reported" : "missing",
    cache_creation_status: has(value, "cache_creation_input_tokens") ? "reported" : "missing",
    output_tokens: outputTokens,
    reasoning_output_tokens: 0,
    total_tokens: totalInput + outputTokens,
  };
}

// -- Reply construction --------------------------------------------------------------------

/** `_property_types`: top-level property types, through oneOf/anyOf/allOf; ambiguous ones dropped. */
function propertyTypes(schema: Json): Map<string, string> {
  const collected = new Map<string, string>();
  const ambiguous = new Set<string>();
  const absorb = (node: unknown) => {
    if (!isMapping(node)) return;
    const properties = get(node, "properties");
    if (isMapping(properties)) {
      for (const [key, value] of Object.entries(properties)) {
        if (!isMapping(value)) continue;
        const kind = get(value, "type");
        if (typeof kind !== "string") continue;
        if (collected.has(key) && collected.get(key) !== kind) ambiguous.add(key);
        collected.set(key, kind);
      }
    }
    for (const branchKey of ["oneOf", "anyOf", "allOf"]) {
      const branches = get(node, branchKey);
      if (Array.isArray(branches)) for (const branch of branches) absorb(branch);
    }
  };
  absorb(schema);
  for (const key of ambiguous) collected.delete(key);
  return collected;
}

/** `_declared_property_types` over the canonical request's tools (OpenAI-shaped). */
export function declaredPropertyTypes(tools: unknown): Map<string, Map<string, string>> {
  const declared = new Map<string, Map<string, string>>();
  if (!Array.isArray(tools)) return declared;
  for (const tool of tools) {
    if (!isMapping(tool)) continue;
    const name = get(tool, "name");
    const schema = get(tool, "input_schema");
    if (typeof name !== "string" || !isMapping(schema)) continue;
    const types = propertyTypes(schema);
    if (types.size) declared.set(name, types);
  }
  return declared;
}

/** Python `str.isspace()` code points, as `str.lstrip()` removes them. */
const PY_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

function pyLstrip(text: string): string {
  let index = 0;
  while (index < text.length && PY_SPACE.has(text.charCodeAt(index))) index += 1;
  return text.slice(index);
}

/** `_decode_stringified_object_arguments`: recover object/array arguments sent as JSON text. */
function decodeStringifiedArguments(args: Json, declared: Map<string, string> | null): Json {
  const repaired: Json = {};
  for (const [name, value] of Object.entries(args)) {
    let eligible: boolean;
    const wanted = declared?.get(name);
    if (declared !== null) {
      const opener = wanted === "object" ? "{" : wanted === "array" ? "[" : null;
      eligible = typeof value === "string" && (wanted === "object" || wanted === "array")
        && pyLstrip(value).slice(0, 1) === opener;
    } else {
      eligible = typeof value === "string"
        && pyLstrip(value).slice(0, 1) === "{";
    }
    if (!eligible) {
      repaired[name] = value;
      continue;
    }
    let decoded: any;
    try {
      decoded = loads(value as string);
    } catch {
      repaired[name] = value;
      continue;
    }
    const wantObject = declared !== null ? wanted === "object" : true;
    repaired[name] = (wantObject ? isMapping(decoded) : Array.isArray(decoded)) ? decoded : value;
  }
  return repaired;
}

/** `_anthropic_blocks_with_tool_inputs`: blocks in index order with decoded inputs, and the rejected inputs. */
function blocksWithToolInputs(
  response: AnthropicResponseEvents, declaredByTool: Map<string, Map<string, string>> | null,
): [Json[], Json[]] {
  const blocks: Json[] = [];
  const rejected: Json[] = [];
  const indices = [...response.blocksByIndex.keys()].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  for (const index of indices) {
    const block = response.blocksByIndex.get(index)!;
    const fragments = (response.toolFragments.get(index) ?? []).join("");
    if (fragments) {
      const reject = (reason: string) => rejected.push({
        index: Number(index), tool_call_id: get(block, "id"), tool_name: get(block, "name"),
        fragment_chars: codePointLength(fragments), fragment_sha256: sha256Hex(utf8Strict(fragments)), reason,
      });
      let args: any;
      let parsed = true;
      try {
        args = loads(fragments);
      } catch {
        parsed = false;
        reject("invalid_json");
      }
      if (parsed) {
        if (isMapping(args)) {
          args = decodeStringifiedArguments(args, declaredByTool?.get(pyStr(get(block, "name") || "")) ?? null);
          block.input = clonePy(args);
        } else {
          reject("not_an_object");
        }
      }
    }
    if (get(block, "type") === "tool_use") {
      if (!isMapping(get(block, "input")) && !rejected.some((item) => item.index === Number(index))) {
        throw new PolicySessionError("Anthropic tool use lacks complete input");
      }
      const id = get(block, "id");
      if (typeof id !== "string" || !id) throw new PolicySessionError("Anthropic tool use lacks a provider call id");
    }
    blocks.push(block);
  }
  return [blocks, rejected];
}

export interface AnthropicReply {
  message: Json;
  usage: Json | null;
  finishReason: string | null;
  outputRejection: Json | null;
}

/** `AnthropicMessagesProvider._parse_events`: accumulate, decode tool inputs, build the reply. */
export function parseAnthropicEvents(rawEvents: Json[], options: {
  modelCallId: string; model: string; toolNamesByAlias: Map<string, string> | null;
  declaredTypesByTool: Map<string, Map<string, string>> | null;
}): AnthropicReply {
  const response = accumulateAnthropicEvents(rawEvents);
  const [blocks, rejected] = blocksWithToolInputs(response, options.declaredTypesByTool);
  if (rejected.length) {
    const safe = blocks.filter((block) => get(block, "type") !== "tool_use");
    const message = normalizeAnthropicBlocks(safe, { modelCallId: options.modelCallId, toolNamesByAlias: options.toolNamesByAlias });
    return {
      message, usage: normalizeAnthropicUsage(response.usageValues), finishReason: "invalid_tool_input",
      outputRejection: {
        code: "invalid_tool_input_json", provider_stop_reason: response.stopReason, rejected_tool_blocks: rejected,
        discarded_tool_block_count: blocks.filter((block) => get(block, "type") === "tool_use").length,
        policy_outcome: "continue_same_observation", world_state_changed: false,
      },
    };
  }
  const stop = response.stopReason;
  const finishMap: Json = { tool_use: "tool_calls", end_turn: "stop", stop_sequence: "stop", max_tokens: "length" };
  const finishReason: string | null = stop !== null && has(finishMap, stop) ? finishMap[stop] : stop;
  let message = normalizeAnthropicBlocks(blocks, { modelCallId: options.modelCallId, toolNamesByAlias: options.toolNamesByAlias });
  if (finishReason === "length" && message.tool_calls) {
    const [rejectedMessage, rejection] = rejectTruncatedTools(message, stop as string);
    return { message: rejectedMessage, usage: normalizeAnthropicUsage(response.usageValues), finishReason: "invalid_tool_input", outputRejection: rejection };
  }
  if (blocks.some((block) => ["thinking", "redacted_thinking"].includes(get(block, "type")))) {
    const continuation = ProviderContinuationState.create({
      providerProtocol: ANTHROPIC_PROTOCOL, model: options.model, nativeKind: ANTHROPIC_CONTINUATION_KIND,
      nativePayload: { content: blocks }, canonicalMessage: message, responseId: response.responseId,
    });
    message = { ...message, [PROVIDER_CONTINUATION_MESSAGE_KEY]: continuation.toDict() };
  }
  return { message, usage: normalizeAnthropicUsage(response.usageValues), finishReason, outputRejection: null };
}

// -- The provider ------------------------------------------------------------------------

export interface AnthropicMessagesOptions {
  baseUrl: string;
  apiKey: string;
  timeoutSeconds?: number;
  maxRetries?: number;
  preflightProbe?: boolean;
  recoveryPolicy?: string;
  recoveryWindowS?: number | null;
  /** first_token_deadline_v1's stall limit (STALL_S); only tests shorten it. */
  stallS?: number | null;
  anthropicVersion?: string;
  authScheme?: string;
  continuationReplay?: string;
  sessionAffinityId?: string | null;
  standbyApiKeys?: string[];
  credentialStartIndex?: number;
  fetchImpl?: typeof fetch;
  clock?: Clock;
}

export interface AnthropicCompleteOptions {
  modelCallId: string;
  deadlineMonotonicMs?: number | null;
  signal?: AbortSignal;
  onAttemptFailure?: (record: Json) => void;
  onRequestBytes?: (bytes: Buffer, sha256: string) => void;
}

const F = (value: number): PyNum => ({ value, isFloat: true });
const I = (value: number): PyNum => ({ value, isFloat: false });

/** Python `min(a, b)` keeping the type of the smaller operand (the first on a tie). */
function pyMinNum(a: PyNum, b: PyNum): PyNum {
  return b.value < a.value ? b : a;
}

export class AnthropicMessagesProvider {
  readonly protocol = ANTHROPIC_PROTOCOL;
  readonly baseUrl: string;
  apiKey: string;
  readonly timeoutSeconds: number;
  readonly maxRetries: number;
  readonly preflightProbe: boolean;
  readonly recoveryPolicy: string;
  readonly recoveryWindowS: number | null;
  readonly stallS: number | null;
  readonly anthropicVersion: string;
  readonly authScheme: string;
  readonly continuationReplay: string;
  readonly sessionAffinityId: string | null;
  readonly credentialStartIndex: number;
  readonly fetchImpl: typeof fetch;
  readonly clock: Clock;
  private readonly ring: string[];
  private credentialIndex = 0;
  private readonly rotations: Json[] = [];
  closed = false;

  constructor(options: AnthropicMessagesOptions) {
    this.baseUrl = (options.baseUrl ?? "").replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? "";
    this.timeoutSeconds = options.timeoutSeconds ?? 900.0;
    this.maxRetries = options.maxRetries ?? 2;
    this.preflightProbe = options.preflightProbe ?? false;
    this.recoveryPolicy = options.recoveryPolicy ?? LEGACY_RECOVERY_POLICY;
    this.recoveryWindowS = options.recoveryWindowS ?? null;
    this.stallS = options.stallS ?? null;
    this.anthropicVersion = options.anthropicVersion ?? "2023-06-01";
    this.authScheme = options.authScheme ?? "x_api_key";
    this.continuationReplay = options.continuationReplay ?? "exact_v1";
    this.sessionAffinityId = options.sessionAffinityId ?? null;
    this.credentialStartIndex = options.credentialStartIndex ?? 0;
    this.fetchImpl = options.fetchImpl ?? providerFetch(this.timeoutSeconds);
    this.clock = options.clock ?? SYSTEM_CLOCK;
    const standby = options.standbyApiKeys ?? [];
    if (!this.baseUrl || !this.apiKey) throw new PolicySessionError("native provider base URL and API key are required");
    if (standby.some((key) => !key)) throw new PolicySessionError("standby provider credentials must be non-empty");
    this.ring = [this.apiKey, ...standby];
    if (this.authScheme !== "bearer" && this.authScheme !== "x_api_key") {
      throw new PolicySessionError("unsupported Anthropic provider auth scheme");
    }
    if (!CONTINUATION_REPLAY_MODES.has(this.continuationReplay)) {
      throw new PolicySessionError("unsupported Anthropic continuation replay mode");
    }
    if (this.sessionAffinityId !== null && !AFFINITY_ID.test(this.sessionAffinityId)) {
      throw new PolicySessionError("invalid provider session-affinity identity");
    }
  }

  credentialRotations(): Json[] {
    return this.rotations.map((item) => ({ ...item }));
  }

  prepareRequest(request: Json): Json {
    return translateAnthropicRequest(request);
  }

  messagesUrl(): string {
    if (this.baseUrl.endsWith("/v1/messages")) return this.baseUrl;
    if (this.baseUrl.endsWith("/v1")) return `${this.baseUrl}/messages`;
    return `${this.baseUrl}/v1/messages`;
  }

  /** `_request_once` headers (the legacy path), in Python's insertion order. */
  requestHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "anthropic-version": this.anthropicVersion, "content-type": "application/json" };
    if (this.authScheme === "bearer") headers.authorization = `Bearer ${this.apiKey}`;
    else headers["x-api-key"] = this.apiKey;
    if (this.sessionAffinityId !== null) headers["x-session-id"] = this.sessionAffinityId;
    return headers;
  }

  /** `_bounded_request_in_process` headers (the deadline worker). */
  workerHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "content-type": "application/json", "anthropic-version": this.anthropicVersion };
    if (this.sessionAffinityId !== null) headers["x-session-id"] = this.sessionAffinityId;
    if (this.authScheme === "x_api_key") headers["x-api-key"] = this.apiKey;
    else headers.authorization = "Bearer " + this.apiKey;
    return headers;
  }

  private binding(): TransportBinding {
    return {
      baseUrl: this.baseUrl, apiKey: this.apiKey, timeoutSeconds: this.timeoutSeconds,
      sessionAffinityId: this.sessionAffinityId, fetchImpl: this.fetchImpl, clock: this.clock,
      anthropic: { url: this.messagesUrl(), headers: () => this.workerHeaders() },
    };
  }

  private redact(text: string): string {
    for (const key of [...this.ring].sort((a, b) => b.length - a.length)) {
      if (key) text = text.split(key).join("[REDACTED]");
    }
    return text;
  }

  private rotateCredential(statusCode: number): boolean {
    if (this.ring.length < 2) return false;
    this.credentialIndex += 1;
    this.apiKey = this.ring[this.credentialIndex % this.ring.length];
    this.rotations.push({
      schema: "ora.provider-credential-rotation.v1",
      credential_index: this.credentialIndex,
      credential_start_index: this.credentialStartIndex,
      completed_laps: Math.floor(this.credentialIndex / this.ring.length),
      status_code: statusCode,
      unix_s: pyFloat(this.clock.time()),
    });
    return true;
  }

  /** `_request_once`: one legacy_v1 attempt. */
  private async requestOnce(bytes: Uint8Array, streaming: boolean, signal?: AbortSignal): Promise<AttemptResult> {
    const control = new AttemptControl(signal);
    control.flavor = "sync";
    const events: Json[] = [];
    let first: number | null = null;
    const binding: TransportBinding = {
      ...this.binding(), anthropic: { url: this.messagesUrl(), headers: () => this.requestHeaders() },
    };
    try {
      control.readTimeout(this.timeoutSeconds);
      const response = await post(binding, bytes, control);
      control.touch();
      if (response.status >= 400) {
        const status = response.status;
        const detail = this.redact(pyTail(decodeReplace(await readAll(response, control)), 4000));
        const rotated = [401, 402, 403, 429].includes(status) && !this.preflightProbe && this.maxRetries > 0
          ? this.rotateCredential(status) : false;
        const error = httpStatusFailure(status, detail, this.baseUrl) as any;
        const kimiFlake = status === 400 && KIMI_CODING_BASE_URLS.has(this.baseUrl)
          && detail.includes(KIMI_TRANSIENT_SIGNATURE_REJECTION)
          && hasCompleteThinkingSignature(parsePy(Buffer.from(bytes).toString("utf8")));
        error.retryable = Boolean(error.retryable || kimiFlake || rotated);
        if (rotated) error.credentialRotated = true;
        if (kimiFlake) error.retryDelayS = 30.0;
        if (status === 429) {
          error.retryDelayS = 20.0;
          error.retryBudget = Math.max(this.maxRetries, 6);
        }
        if (rotated && this.credentialIndex >= this.ring.length) {
          error.retryDelayS = 90.0;
          error.retryBudget = Math.max(this.maxRetries, 8);
        }
        throw error;
      }
      if (!streaming) {
        const value = loadsBytes(await readAll(response, control));
        if (!isMapping(value)) throw new PolicySessionError("native provider returned non-object JSON");
        return { events: [value], first: 0.0 };
      }
      try {
        let announced: string | null = null;
        for await (const line of lines(response, control)) {
          if (!line) continue;
          if (line.startsWith(":")) continue;
          if (line.startsWith("event:")) {
            if (announced !== null) throw new PolicySessionError("Anthropic SSE event lacks its data payload");
            announced = pyStrip(line.slice(6));
            if (!announced) throw new PolicySessionError("Anthropic SSE event name is empty");
            continue;
          }
          if (!line.startsWith("data:")) throw new PolicySessionError("Anthropic provider returned malformed SSE framing");
          const data = pyStrip(line.slice(5));
          if (data === "[DONE]") {
            announced = null;
            break;
          }
          let parsed: any;
          try {
            parsed = loadsText(data, false);
          } catch (error) {
            throw new PolicySessionError("native provider returned malformed SSE JSON", { cause: error });
          }
          if (!isMapping(parsed)) throw new PolicySessionError("native provider SSE item must be an object");
          if (announced !== null && get(parsed, "type") !== announced) {
            throw new PolicySessionError("Anthropic SSE event name does not match its payload");
          }
          announced = null;
          if (first === null) first = this.clock.monotonic();
          events.push(parsed);
          if (get(parsed, "type") === "message_stop") break;
        }
        if (announced !== null) throw new PolicySessionError("Anthropic SSE ended before an announced event payload");
        if (!events.some((event) => ["message_stop", "error"].includes(get(event, "type")))) {
          throw new PyEOFError("Anthropic stream ended before message_stop");
        }
      } catch (error) {
        if (control.kind === "caller") throw error;
        throw new StreamInterrupted(events, transientTransport(error) || error instanceof PyEOFError, error);
      }
    } catch (error) {
      if (control.kind === "caller") throw new ProviderAborted(events);
      throw error;
    } finally {
      control.dispose();
    }
    return { events, first };
  }

  async complete(request: Json, options: AnthropicCompleteOptions): Promise<Json> {
    const clock = this.clock;
    const { modelCallId, signal, onAttemptFailure, onRequestBytes } = options;
    const startedUnix = clock.time();
    const startedMono = clock.monotonic();
    const deadline = options.deadlineMonotonicMs;
    const window = RecoveryWindow.forProvider({
      recoveryPolicy: this.recoveryPolicy, recoveryWindowS: this.recoveryWindowS,
      requestDeadlineMonotonic: deadline === null || deadline === undefined ? null : deadline / 1000,
      preflightProbe: this.preflightProbe, maxRetries: this.maxRetries, stallS: this.stallS,
    }, clock);
    const toolNamesByAlias = anthropicToolNames(request);
    const body = this.prepareRequest(request);
    const bytes = canonicalBytesPy(body);
    const requestSha256 = sha256Hex(bytes);
    const streaming = body.stream !== false;
    const ringSize = this.ring.length;
    const record = (exc: unknown, fields: {
      attempt: number; credentialIndex: number; nextIndex: number | null; started: number; startedMono: number;
      retryDelay: PyNum | 0; events: Json[]; phase: "transport" | "parse";
    }) => {
      if (!onAttemptFailure) return;
      onAttemptFailure(attemptFailureRecord(exc, {
        recoveryPolicy: this.recoveryPolicy, keys: this.ring, requestSha256, modelCallId,
        attempt: fields.attempt, credentialIndex: fields.credentialIndex, nextIndex: fields.nextIndex,
        startedUnix: fields.started, startedMonotonic: fields.startedMono,
        nowUnix: clock.time(), nowMonotonic: clock.monotonic(), retryDelay: fields.retryDelay,
        events: fields.events, phase: fields.phase,
      }));
    };
    const pause = async (delay: PyNum, attempts: number, cause: unknown) => {
      try {
        if (window !== null) await window.pause(delay, signal);
        else await clock.sleep(delay.value, signal);
      } catch (pauseError) {
        if (pauseError instanceof ProviderRecoveryDeadline) {
          recordProviderFailure(pauseError, attempts);
          throw pauseError;
        }
        const aborted = new ProviderAborted();
        recordProviderFailure(aborted, attempts);
        void cause;
        throw aborted;
      }
    };
    let rawEvents: Json[] = [];
    let firstEventAt: number | null = null;
    let attempts = 0;
    let incompleteBlockRetries = 0;
    let reply: AnthropicReply;
    for (;;) {
      if (this.closed) {
        const closed = new ProviderClosed();
        recordProviderFailure(closed, attempts);
        throw closed;
      }
      attempts += 1;
      const attemptStarted = clock.time();
      const attemptMono = clock.monotonic();
      const credentialIndex = (this.credentialStartIndex + this.credentialIndex) % ringSize;
      try {
        onRequestBytes?.(bytes, requestSha256);
        const result = window !== null
          ? await boundedRequest(this.binding(), bytes, streaming, window, signal)
          : await this.requestOnce(bytes, streaming, signal);
        rawEvents = result.events;
        firstEventAt = result.first;
      } catch (exc) {
        recordProviderFailure(exc, attempts);
        const aborted = exc instanceof ProviderAborted || Boolean(signal?.aborted);
        let retryable = Boolean(attr(exc, "retryable", false));
        if (!retryable) retryable = transientTransport(exc);
        const budgetAttr = attr(exc, "retryBudget", null);
        const retryBudget = typeof budgetAttr === "number" && Number.isInteger(budgetAttr) ? budgetAttr : this.maxRetries;
        let willRetry = !this.preflightProbe && this.maxRetries > 0 && retryable && attempts < retryBudget + 1;
        const partialRaw = attr(exc, "partialEvents", []);
        const partial: Json[] = Array.isArray(partialRaw) ? partialRaw : [];
        const echoed = partialCredentialEcho(exc, this.ring);
        willRetry = willRetry && !echoed;
        // Only the legacy path reads the hint, and every hint it sets is a float (20.0, 30.0, 90.0).
        const hint = attr(exc, "retryDelayS", null);
        const hintNum: PyNum | null = typeof hint === "number" ? F(hint) : hint instanceof PyFloat ? F(hint.value) : null;
        let delay: PyNum;
        if (hintNum !== null && hintNum.value > 0) {
          const scaled = { value: hintNum.value * 2 ** (attempts - 1), isFloat: hintNum.isFloat };
          delay = pyMinNum(scaled, F(60.0));
        } else {
          delay = pyMinNum(I(2 ** (attempts - 1)), I(4));
        }
        if (window !== null) {
          const bounded = window.transportDelay(exc, attempts, attemptMono);
          willRetry = bounded !== null && !echoed;
          delay = bounded === null || bounded.value === 0 ? F(0.0) : bounded;
        }
        if (this.closed || aborted) willRetry = false;
        const doSwitch = window === null && willRetry && exc instanceof StreamInterrupted && ringSize > 1;
        const nextIndex = (this.credentialStartIndex + this.credentialIndex + (doSwitch ? 1 : 0)) % ringSize;
        record(exc, {
          attempt: attempts, credentialIndex, nextIndex: willRetry ? nextIndex : null, started: attemptStarted,
          startedMono: attemptMono, retryDelay: willRetry ? delay : F(0.0), events: partial, phase: "transport",
        });
        if (echoed) throw echoFailure(exc, attempts);
        if (!willRetry) throw exc;
        if (doSwitch) {
          this.credentialIndex += 1;
          this.apiKey = this.ring[this.credentialIndex % ringSize];
        }
        await pause(delay, attempts, exc);
        continue;
      }
      try {
        const serialized = dumpsPy(rawEvents, { ensureAscii: false });
        if (this.ring.some((key) => serialized.includes(key))) throw new PolicySessionError(CREDENTIAL_ECHO_MESSAGE);
        reply = parseAnthropicEvents(rawEvents, {
          modelCallId, model: String(get(request, "model", "")), toolNamesByAlias,
          // Python passes the canonical (function-shaped) tools, whose top-level
          // name/input_schema are absent, so no declared types are ever found.
          declaredTypesByTool: declaredPropertyTypes(get(request, "tools", null)),
        });
        if (reply.finishReason === null) {
          throw new PolicySessionError("native provider response has no completion status; request was not retried");
        }
      } catch (exc) {
        if (!(exc instanceof PolicySessionError)) throw exc;
        exc.responseStarted = rawEvents.length > 0;
        recordProviderFailure(exc, attempts);
        const safeIncomplete = exc.message === "Anthropic message_stop precedes completed content blocks";
        let willRetry = !this.preflightProbe && safeIncomplete && incompleteBlockRetries < 1 && attempts < this.maxRetries + 1;
        let delay: PyNum = willRetry ? pyMinNum(I(2 ** (attempts - 1)), I(4)) : F(0.0);
        if (window !== null) {
          if (safeIncomplete && incompleteBlockRetries < 1) exc.retryable = true;
          const bounded = window.delay(exc, attempts);
          const serialized = dumpsPy(rawEvents, { ensureAscii: false });
          willRetry = bounded !== null && !this.ring.some((key) => serialized.includes(key));
          delay = bounded === null || bounded.value === 0 ? F(0.0) : bounded;
        }
        if (this.closed || signal?.aborted) willRetry = false;
        record(exc, {
          attempt: attempts, credentialIndex, nextIndex: willRetry ? credentialIndex : null, started: attemptStarted,
          startedMono: attemptMono, retryDelay: delay, events: rawEvents, phase: "parse",
        });
        if (!willRetry) throw exc;
        if (safeIncomplete) incompleteBlockRetries += 1;
        await pause(delay, attempts, exc);
        continue;
      }
      break;
    }
    if (window !== null && window.remaining() <= 0) {
      const error = new ProviderRecoveryDeadline(rawEvents);
      recordProviderFailure(error, attempts);
      throw error;
    }
    let responseId: string | null = null;
    for (const event of rawEvents) {
      const candidate = get(event, "type") === "message_start" ? get(event, "message") : event;
      if (isMapping(candidate) && typeof get(candidate, "id") === "string" && candidate.id) {
        responseId = candidate.id;
        break;
      }
    }
    return {
      message: reply.message,
      usage: reply.usage,
      finish_reason: reply.finishReason,
      ...(reply.outputRejection !== null ? { output_rejection: reply.outputRejection } : {}),
      response_id: responseId,
      raw_events: rawEvents,
      wire_request: body,
      model_call_id: modelCallId,
      started_unix_s: pyFloat(startedUnix),
      finished_unix_s: pyFloat(clock.time()),
      ttft_s: firstEventAt === null ? null : pyFloat(pyRound6(firstEventAt - startedMono)),
      attempts,
    };
  }

  close(): void {
    this.closed = true;
  }
}

/** `build_provider` for the anthropic_messages protocol. */
export function buildAnthropicMessagesProvider(options: AnthropicMessagesOptions & {
  outputTokenParameter?: string | null; toolStrictness?: string | null; reasoningField?: string | null;
  requestOptions?: Record<string, unknown> | null;
}): AnthropicMessagesProvider {
  const recoveryPolicy = options.recoveryPolicy ?? LEGACY_RECOVERY_POLICY;
  if (recoveryPolicy !== LEGACY_RECOVERY_POLICY && !RECOVERY_POLICIES.has(recoveryPolicy)) {
    throw new PolicySessionError("unsupported native provider recovery policy");
  }
  if (options.recoveryWindowS !== null && options.recoveryWindowS !== undefined && recoveryPolicy === LEGACY_RECOVERY_POLICY) {
    throw new PolicySessionError("recovery_window_s requires the same-credential deadline recovery policy");
  }
  if ((options.outputTokenParameter ?? null) !== null || (options.toolStrictness ?? null) !== null
      || (options.reasoningField ?? null) !== null || (options.requestOptions ?? null) !== null) {
    throw new PolicySessionError(
      "output_token_parameter, tool_strictness, reasoning_field and request_options "
      + "apply only to OpenAI-compatible providers",
    );
  }
  const continuationReplay = options.continuationReplay ?? "exact_v1";
  if (!CONTINUATION_REPLAY_MODES.has(continuationReplay)) {
    throw new PolicySessionError("unsupported Anthropic continuation replay mode");
  }
  return new AnthropicMessagesProvider({ ...options, recoveryPolicy, continuationReplay });
}
