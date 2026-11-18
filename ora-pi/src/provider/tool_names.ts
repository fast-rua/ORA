// Port of ora/agent/providers/tool_names.py: deterministic strict-provider
// tool names with canonical transcript identities.

import { dumpsPy, utf8Strict } from "../pyjson.ts";
import { PolicySessionError, PyError, sha256Hex } from "./py.ts";

const STRICT_NAME = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;

function isMapping(value: unknown): value is { [key: string]: any } {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Python iteration over `value` (a list, or the keys of a dict); anything else raises. */
function pyIter(value: unknown, what: string): any[] {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return Array.from(value);
  if (isMapping(value)) return Object.keys(value);
  throw new PyError("TypeError", `'${value === null ? "NoneType" : typeof value}' object is not iterable (${what})`);
}

/** Python `obj.get(...)` on a value that may not be a dict. */
function pyGet(value: unknown, key: string, fallback: unknown = undefined): any {
  if (!isMapping(value)) {
    throw new PyError("AttributeError", `object has no attribute 'get' (${key})`);
  }
  return Object.prototype.hasOwnProperty.call(value, key) ? value[key] : fallback;
}

export function toolAlias(name: string): string {
  if (STRICT_NAME.test(name)) return name;
  let slug = name.replace(/[^A-Za-z0-9_-]/gu, "_");
  if (!slug || !/^[A-Za-z]/.test(slug)) slug = "tool_" + slug;
  return slug.slice(0, 55) + "_" + sha256Hex(utf8Strict(name)).slice(0, 8);
}

/** alias -> canonical name; includes retained tool turns after a tool change. */
export function openaiToolNames(request: { [key: string]: any }): Map<string, string> {
  const names: unknown[] = [];
  for (const tool of pyIter(pyGet(request, "tools", []), "tools")) {
    if (isMapping(tool) && isMapping(tool.function)) names.push(tool.function.name);
  }
  for (const message of pyIter(pyGet(request, "messages", []), "messages")) {
    if (!isMapping(message)) continue;
    if (message.role === "tool") names.push(message.name);
    for (const call of pyIter(pyGet(message, "tool_calls", []), "tool_calls")) {
      if (isMapping(call) && isMapping(call.function)) names.push(call.function.name);
    }
  }
  const choice = request.tool_choice;
  if (isMapping(choice) && isMapping(choice.function)) names.push(choice.function.name);
  const aliases = new Map<string, string>();
  for (const name of names) {
    if (typeof name !== "string" || !name) continue;
    const alias = toolAlias(name);
    if (!aliases.has(alias)) aliases.set(alias, name);
    if (aliases.get(alias) !== name) throw new PolicySessionError("OpenAI tool-name alias collision");
  }
  return aliases;
}

/**
 * The canonical name of a tool call the provider returned. A wire alias maps
 * back and a name the request already carries stays. Any other name that is
 * not strict (a provider tool-call parse defect, such as a parameter tag read
 * as the function name) takes its strict alias: kept verbatim it would add an
 * entry to the <wire_tool_aliases> disclosure on the next request and rewrite
 * the cached system prefix. The Runtime still refuses it as an unavailable
 * tool; the provider's raw events keep the name it sent.
 */
export function responseToolName(name: string, aliases: Map<string, string>): string {
  const canonical = aliases.get(name);
  if (canonical !== undefined) return canonical;
  if (!name || STRICT_NAME.test(name)) return name;
  for (const known of aliases.values()) if (known === name) return name;
  return toolAlias(name);
}

/** Project a copied, continuation-validated wire body in place. */
export function projectOpenaiToolNames(body: { [key: string]: any }, aliases: Map<string, string>): void {
  for (const tool of pyIter(pyGet(body, "tools", []), "tools")) {
    const fn = pyGet(tool, "function");
    if (isMapping(fn) && typeof fn.name === "string" && fn.name) fn.name = toolAlias(fn.name);
  }
  // A nameless rejected call stays nameless.
  for (const message of pyIter(body.messages, "messages")) {
    if (pyGet(message, "role") === "tool" && typeof message.name === "string") {
      message.name = toolAlias(message.name);
    }
    for (const call of pyIter(pyGet(message, "tool_calls", []), "tool_calls")) {
      const fn = pyGet(call, "function");
      if (isMapping(fn) && typeof fn.name === "string" && fn.name) fn.name = toolAlias(fn.name);
    }
  }
  const choice = body.tool_choice;
  if (isMapping(choice) && isMapping(choice.function)) {
    const name = choice.function.name;
    if (typeof name === "string") choice.function.name = toolAlias(name);
  }
  const renamed: { [key: string]: string } = {};
  let any = false;
  for (const [alias, name] of aliases) {
    if (alias !== name) {
      Object.defineProperty(renamed, name, { value: alias, enumerable: true, writable: true, configurable: true });
      any = true;
    }
  }
  if (!any) return;
  // Disclose the mapping in the provider prefix, as the Anthropic projection
  // does. Qwen3.x chat templates reject any system message not at index 0, so
  // the disclosure joins the last leading system message instead of adding one.
  const disclosure = "<wire_tool_aliases>\n" + dumpsPy(renamed, { ensureAscii: false, sortKeys: true })
    + "\n</wire_tool_aliases>";
  const messages: any[] = body.messages;
  let leading = 0;
  while (leading < messages.length && pyGet(messages[leading], "role") === "system") leading += 1;
  if (!leading) {
    messages.splice(0, 0, { role: "system", content: disclosure });
    return;
  }
  const system = messages[leading - 1];
  const content = pyGet(system, "content", null);
  if (Array.isArray(content)) {
    system.content = [...content, { type: "text", text: disclosure }];
  } else if (typeof content === "string" && content) {
    system.content = content + "\n\n" + disclosure;
  } else if (content === null || content === "") {
    system.content = disclosure;
  } else {
    throw new PolicySessionError("OpenAI system message content is not text");
  }
}
