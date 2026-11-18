// Strict JSONL framing of Pi's RPC protocol (as Pi's own dist/modes/rpc/jsonl.js):
// one JSON value per line, split on LF only, a trailing CR dropped. Payload
// strings may hold U+2028/U+2029, which Node's readline would split on.

import type { Socket } from "node:net";
import { StringDecoder } from "node:string_decoder";

/** Python's integers beyond 2**53 arrive as bigint (pyjson.ts): written as the exact JSON number. */
const exact = (_key: string, item: unknown) => (typeof item === "bigint" ? (JSON as any).rawJSON(item.toString()) : item);

export function serializeJsonLine(value: unknown): string {
  return `${JSON.stringify(value, exact)}\n`;
}

/** Call `onLine` for every LF-terminated line of `stream` (and a final unterminated one). */
export function attachJsonlLineReader(
  stream: NodeJS.ReadableStream | Socket, onLine: (line: string) => void,
): () => void {
  const decoder = new StringDecoder("utf8");
  let buffer = "";
  const emit = (line: string) => onLine(line.endsWith("\r") ? line.slice(0, -1) : line);
  const onData = (chunk: Buffer | string) => {
    buffer += typeof chunk === "string" ? chunk : decoder.write(chunk);
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      emit(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
    }
  };
  const onEnd = () => {
    buffer += decoder.end();
    if (buffer.length > 0) emit(buffer);
    buffer = "";
  };
  stream.on("data", onData);
  stream.on("end", onEnd);
  return () => {
    stream.off("data", onData);
    stream.off("end", onEnd);
  };
}
