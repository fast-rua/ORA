// Thin pi-ai adapter: wrap ORA's `complete()` as a streamSimple-style function.
//
// ORA's provider accumulates the whole reply before it returns (retry and
// credential-echo rules need the complete stream), so the Pi events are
// emitted after the call: start -> thinking -> text -> toolcalls -> done/error.
// The Pi AssistantMessage carries the ORA canonical assistant message as `ora`
// (and the full complete() dict as `oraResponse`); every ToolCall block keeps
// its exact argument string as `rawArguments`. The integration owner supplies
// `buildRequest`, which maps the Pi call onto ORA's execution request.

import {
  createAssistantMessageEventStream,
  type AssistantMessage, type AssistantMessageEventStream, type Model, type SimpleStreamOptions,
  type StopReason, type ToolCall, type TranscriptContext, type Usage,
} from "@earendil-works/pi-ai";
import type { CompleteOptions } from "./openai.ts";
import { ProviderAborted } from "./transport.ts";

type Json = { [key: string]: any };

export interface OraCompletion {
  complete(request: Json, options: CompleteOptions): Promise<Json>;
}

export interface OraStreamBinding {
  provider: OraCompletion;
  buildRequest(model: Model<any>, context: TranscriptContext, options?: SimpleStreamOptions): Json;
  nextModelCallId(): string;
  deadlineMonotonicMs?: () => number | null;
  onAttemptFailure?: CompleteOptions["onAttemptFailure"];
  onRequestBytes?: CompleteOptions["onRequestBytes"];
}

const count = (value: unknown): number => (typeof value === "number" ? value : Number(value ?? 0) || 0);

function piUsage(usage: Json | null): Usage {
  const cached = count(usage?.cached_input_tokens);
  return {
    input: Math.max(0, count(usage?.input_tokens) - cached), output: count(usage?.output_tokens),
    cacheRead: cached, cacheWrite: 0, reasoning: count(usage?.reasoning_output_tokens),
    totalTokens: count(usage?.total_tokens),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function stopReason(finish: string): Extract<StopReason, "stop" | "length" | "toolUse"> {
  if (finish === "tool_calls") return "toolUse";
  if (finish === "length" || finish === "invalid_tool_input") return "length";
  return "stop";
}

function parseArguments(raw: string): Json {
  try {
    const value = JSON.parse(raw);
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

export function oraStreamSimple(binding: OraStreamBinding) {
  return (model: Model<any>, context: TranscriptContext, options?: SimpleStreamOptions): AssistantMessageEventStream => {
    const stream = createAssistantMessageEventStream();
    const output: AssistantMessage & Json = {
      role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
      usage: piUsage(null), stopReason: "pending", timestamp: Date.now(),
    };
    void (async () => {
      stream.push({ type: "start", partial: output });
      try {
        const response = await binding.provider.complete(binding.buildRequest(model, context, options), {
          modelCallId: binding.nextModelCallId(), deadlineMonotonicMs: binding.deadlineMonotonicMs?.() ?? null,
          signal: options?.signal, onAttemptFailure: binding.onAttemptFailure, onRequestBytes: binding.onRequestBytes,
        });
        const message = response.message as Json;
        const blocks: Array<[string, string]> = [];
        if (typeof message.reasoning_content === "string") blocks.push(["thinking", message.reasoning_content]);
        if (typeof message.content === "string" && message.content) blocks.push(["text", message.content]);
        for (const [kind, text] of blocks) {
          const contentIndex = output.content.length;
          if (kind === "thinking") {
            output.content.push({ type: "thinking", thinking: "" });
            stream.push({ type: "thinking_start", contentIndex, partial: output });
            (output.content[contentIndex] as any).thinking = text;
            stream.push({ type: "thinking_delta", contentIndex, delta: text, partial: output });
            stream.push({ type: "thinking_end", contentIndex, content: text, partial: output });
          } else {
            output.content.push({ type: "text", text: "" });
            stream.push({ type: "text_start", contentIndex, partial: output });
            (output.content[contentIndex] as any).text = text;
            stream.push({ type: "text_delta", contentIndex, delta: text, partial: output });
            stream.push({ type: "text_end", contentIndex, content: text, partial: output });
          }
        }
        for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
          const raw = String(call.function?.arguments ?? "");
          const toolCall: ToolCall & { rawArguments: string } = {
            type: "toolCall", id: String(call.id), name: String(call.function?.name ?? ""),
            arguments: parseArguments(raw), rawArguments: raw,
          };
          const contentIndex = output.content.length;
          output.content.push(toolCall);
          stream.push({ type: "toolcall_start", contentIndex, partial: output });
          stream.push({ type: "toolcall_delta", contentIndex, delta: raw, partial: output });
          stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
        }
        const reason = stopReason(String(response.finish_reason));
        output.usage = piUsage(response.usage);
        output.stopReason = reason;
        output.rawStopReason = String(response.finish_reason);
        output.ora = message;
        output.oraResponse = response;
        stream.push({ type: "done", reason, message: output });
      } catch (error: any) {
        const aborted = error instanceof ProviderAborted || Boolean(options?.signal?.aborted);
        output.stopReason = aborted ? "aborted" : "error";
        output.errorMessage = error?.message ?? String(error);
        output.oraError = { type: error?.pyType ?? error?.name, provider_failure: error?.providerFailure ?? null };
        stream.push({ type: "error", reason: aborted ? "aborted" : "error", error: output });
      }
      stream.end();
    })();
    return stream;
  };
}
