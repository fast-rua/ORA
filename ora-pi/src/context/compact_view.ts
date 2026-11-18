// The compact history view of a context pack (`ora.compact-view.v1`).
//
// A context pack (`ora/agent/composition/prompt/context_pack.py`) may ask for a
// compact history. `compactView` builds it from the messages the Context
// pipeline projected, after the pack's static replacements
// (`context_pack.ts`):
//
// 1. The prefix -- everything before the first assistant message (system,
//    task, initial observation, round static facts) -- stays verbatim.
// 2. A turn is one assistant message and the tool and user messages after it,
//    up to the next assistant message.
// 3. The last `recent_turns` turns stay, with the assistant's text stripped,
//    its provider continuation state dropped (the history carries no
//    reasoning), and a tool message's text cut to `tool_text_cap_chars`
//    code points plus `texts.tool_text_omission` ({omitted} = the cut count).
// 4. Every older turn becomes one line of one user message that starts with
//    `texts.earlier_steps_header` (fold line grammar `turn_summary_v1`):
//      `{n}. {text} -> {tool}({arguments}) -> {status}; ...; effect: {summary}`
//    text: the assistant's stripped text, cut to `rationale_cap_chars` plus
//    "…" (omitted with its arrow when empty); arguments: compact JSON
//    (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))`), cut to
//    `arguments_cap_chars` plus "…" (null keeps them whole); status: the call's
//    tool result -- `error {code or type}`, `task_success {value}`,
//    `status {value}`, `success {value}` or `ok` for a JSON object,
//    `returned a value` for other JSON, `returned text` otherwise, `no result`
//    without one; "(no tool call)" for a turn without calls; effect: the last
//    `last_action_effect.action_summary` among the turn's user messages.
// 5. Images: the newest `image_window` image parts of the whole request stay
//    (the caller downscales them to `image_long_edge_px`); every earlier one
//    becomes the text `texts.image_placeholder` with {label} = the first 100
//    code points of the text part before it in the same message (a
//    `<recoverable_artifact>` reference is not a label).
//
// Python's `str()` and `len()` semantics are kept (code points, `True`,
// `None`, float repr), so the view is byte-identical to the Python reference
// the pack's training data was built with. Nothing here depends on the
// request that will follow: the view never keeps an older turn because a
// later call might need it.

import {
  clonePy, cpLen, cpPrefix, dumpsPy, hasKey, isMapping, PolicySessionError, PyFloat, pyRepr, pyStrip, tryLoads,
} from "./pyutil.ts";
import { PROVIDER_CONTINUATION_MESSAGE_KEY } from "./projection.ts";

type Json = any;
export type Message = Record<string, Json>;

export const COMPACT_VIEW_SCHEMA = "ora.compact-view.v1";
export const FOLD_LINE_FORMATS = new Set(["turn_summary_v1"]);
/** Characters of the text part before an omitted image that name it. */
export const IMAGE_LABEL_CHARS = 100;
/** A lossless Context retention reference: it never labels an image. */
const RECOVERABLE_ARTIFACT_PREFIX = "<recoverable_artifact>";
const ELLIPSIS = "…";

export interface CompactViewConfig {
  schema: string;
  recent_turns: number;
  tool_text_cap_chars: number | null;
  image_window: number | null;
  image_long_edge_px: number | null;
  fold: {
    line_format: string;
    rationale_cap_chars: number | null;
    arguments_cap_chars: number | null;
    latest_result_tools?: string[];
  };
  texts: { earlier_steps_header: string; image_placeholder: string; tool_text_omission: string };
}

/** Where a kept image sits in the view's messages, and its data URL. */
export interface KeptImage {
  message: number;
  part: number;
  url: string;
}

export interface CompactView {
  messages: Message[];
  keptImages: KeptImage[];
  stats: { prefix_messages: number; folded_turns: number; kept_turns: number; kept_images: number; omitted_images: number };
}

/** Refuse a configuration this view version does not implement (Python validated it at plan time). */
export function checkCompactViewConfig(config: CompactViewConfig): void {
  if (!config || config.schema !== COMPACT_VIEW_SCHEMA) {
    throw new PolicySessionError(`compact view schema must be ${COMPACT_VIEW_SCHEMA}`);
  }
  if (!FOLD_LINE_FORMATS.has(config.fold?.line_format)) {
    throw new PolicySessionError(`unsupported compact view fold line format: ${config.fold?.line_format}`);
  }
  if ((config.fold.latest_result_tools ?? []).length) {
    throw new PolicySessionError("compact view latest_result_tools is reserved and must be empty in v1");
  }
}

/** Python ``str(value)`` of a decoded JSON value. */
function pyStr(value: unknown): string {
  return typeof value === "string" ? value : pyRepr(value === undefined ? null : value);
}

/** Python truthiness of a decoded JSON value. */
function pyTruthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === "") return false;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "bigint") return value !== 0n;
  if (value instanceof PyFloat) return value.value !== 0;
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  return true;
}

/** ``text[:cap] + suffix`` when ``len(text) > cap`` (code points); unchanged otherwise or with no cap. */
function capped(text: string, cap: number | null, suffix: (omitted: number) => string): string {
  if (cap === null) return text;
  const length = cpLen(text);
  return length <= cap ? text : cpPrefix(text, cap) + suffix(length - cap);
}

/** ``template`` with its one ``{name}`` field replaced. */
function fill(template: string, name: string, value: string): string {
  return template.replace(`{${name}}`, () => value);
}

/** The prefix before the first assistant message and the turns that start at each assistant. */
export function splitTurns(messages: Message[]): [Message[], Message[][]] {
  let first = messages.findIndex((message) => message?.role === "assistant");
  if (first < 0) first = messages.length;
  const turns: Message[][] = [];
  for (const message of messages.slice(first)) {
    if (message.role === "assistant" || !turns.length) turns.push([]);
    turns[turns.length - 1].push(message);
  }
  return [messages.slice(0, first), turns];
}

/** One tool result's status word (see the header). */
export function resultStatus(text: unknown): string {
  if (typeof text !== "string") return "returned text";
  const parsed = tryLoads(text);
  if (!parsed.ok) return "returned text";
  const value = parsed.value;
  if (!isMapping(value)) return "returned a value";
  const error = value.error;
  if (isMapping(error)) {
    const code = pyTruthy(error.code) ? error.code : pyTruthy(error.type) ? error.type : "";
    return pyStrip(`error ${pyStr(code)}`);
  }
  if (hasKey(value, "task_success")) return `task_success ${pyStr(value.task_success)}`;
  for (const key of ["status", "success"]) {
    if (hasKey(value, key)) return `${key} ${pyStr(value[key])}`;
  }
  return "ok";
}

/** The ``last_action_effect.action_summary`` a user message's first text part carries, if any. */
export function actionEffect(message: Message): unknown {
  const content = message.content;
  const first = Array.isArray(content) && content.length ? (isMapping(content[0]) ? content[0].text : undefined) : content;
  if (typeof first !== "string") return null;
  const parsed = tryLoads(first);
  if (!parsed.ok || !isMapping(parsed.value)) return null;
  const effect = parsed.value.last_action_effect;
  return isMapping(effect) ? (effect.action_summary ?? null) : null;
}

function parseArguments(raw: unknown): unknown {
  if (typeof raw !== "string") return raw === undefined ? null : raw;
  const parsed = tryLoads(raw);
  return parsed.ok ? parsed.value : raw;
}

/** One older turn as its fold line (``turn_summary_v1``). */
export function foldTurn(number: number, turn: Message[], fold: CompactViewConfig["fold"]): string {
  const assistant = turn[0];
  const results = new Map<unknown, string>();
  for (const message of turn) {
    if (message.role === "tool") results.set(message.tool_call_id ?? null, resultStatus(message.content));
  }
  const calls: string[] = [];
  for (const call of Array.isArray(assistant.tool_calls) ? assistant.tool_calls : []) {
    const fn = isMapping(call?.function) ? call.function : {};
    const argumentsText = dumpsPy(parseArguments(fn.arguments), { ensureAscii: false, separators: [",", ":"] });
    const shown = capped(argumentsText, fold.arguments_cap_chars, () => ELLIPSIS);
    const status = results.has(call?.id ?? null) ? results.get(call?.id ?? null)! : "no result";
    calls.push(`${pyStr(fn.name ?? null)}(${shown}) -> ${status}`);
  }
  const effects = turn.filter((message) => message.role === "user").map(actionEffect).filter(pyTruthy);
  const raw = typeof assistant.content === "string" ? pyStrip(assistant.content) : "";
  const said = capped(raw, fold.rationale_cap_chars, () => ELLIPSIS);
  let line = `${number}. ` + (said ? `${said} -> ` : "") + (calls.length ? calls.join("; ") : "(no tool call)");
  if (effects.length) line += `; effect: ${pyStr(effects[effects.length - 1])}`;
  return line;
}

/** A recent turn's message as the view keeps it. */
function keptMessage(message: Message, config: CompactViewConfig): Message {
  const kept = clonePy(message);
  if (kept.role === "tool" && typeof kept.content === "string") {
    kept.content = capped(kept.content, config.tool_text_cap_chars,
      (omitted) => fill(config.texts.tool_text_omission, "omitted", String(omitted)));
  }
  if (kept.role === "assistant") {
    if (typeof kept.content === "string") kept.content = pyStrip(kept.content);
    delete kept[PROVIDER_CONTINUATION_MESSAGE_KEY];
    delete kept.reasoning_content;
    delete kept.reasoning;
  }
  return kept;
}

/** The compact view of ``messages`` (see the header). Kept images are left as they are, listed for the caller. */
export function compactView(messages: Message[], config: CompactViewConfig): CompactView {
  checkCompactViewConfig(config);
  const [prefix, turns] = splitTurns(messages);
  const recent = Math.min(config.recent_turns, turns.length);
  const older = turns.slice(0, turns.length - recent);
  const keep = turns.slice(turns.length - recent);
  const view: Message[] = prefix.map((message) => clonePy(message));
  if (older.length) {
    const lines = older.map((turn, index) => foldTurn(index + 1, turn, config.fold));
    view.push({ role: "user", content: `${config.texts.earlier_steps_header}\n${lines.join("\n")}` });
  }
  for (const turn of keep) for (const message of turn) view.push(keptMessage(message, config));

  const slots: Array<[number, number]> = [];
  view.forEach((message, index) => {
    if (!Array.isArray(message.content)) return;
    message.content.forEach((part: Json, partIndex: number) => {
      if (isMapping(part) && part.type === "image_url") slots.push([index, partIndex]);
    });
  });
  const dropCount = config.image_window === null ? 0 : Math.max(0, slots.length - config.image_window);
  const dropped = new Set(slots.slice(0, dropCount).map(([index, part]) => `${index}:${part}`));
  const keptImages: KeptImage[] = [];
  view.forEach((message, index) => {
    if (!Array.isArray(message.content)) return;
    let label = "";
    message.content = message.content.map((part: Json, partIndex: number) => {
      if (!isMapping(part)) return part;
      if (part.type === "text" && typeof part.text === "string") {
        if (!part.text.startsWith(RECOVERABLE_ARTIFACT_PREFIX)) label = cpPrefix(part.text, IMAGE_LABEL_CHARS);
        return part;
      }
      if (part.type !== "image_url") return part;
      if (dropped.has(`${index}:${partIndex}`)) {
        return { type: "text", text: fill(config.texts.image_placeholder, "label", label) };
      }
      const url = part.image_url?.url;
      if (typeof url !== "string") throw new PolicySessionError("a kept image part has no data URL");
      keptImages.push({ message: index, part: partIndex, url });
      return part;
    });
  });
  return {
    messages: view,
    keptImages,
    stats: {
      prefix_messages: prefix.length, folded_turns: older.length, kept_turns: keep.length,
      kept_images: keptImages.length, omitted_images: dropCount,
    },
  };
}

/** ``view`` with each kept image's URL replaced (same order as ``view.keptImages``). */
export function withImageUrls(view: CompactView, urls: string[]): Message[] {
  if (urls.length !== view.keptImages.length) throw new PolicySessionError("downscaled image count differs");
  const messages = view.messages;
  view.keptImages.forEach((slot, index) => {
    const part = messages[slot.message].content[slot.part];
    messages[slot.message].content[slot.part] = { ...part, image_url: { ...part.image_url, url: urls[index] } };
  });
  return messages;
}
