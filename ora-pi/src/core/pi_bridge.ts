// The ORA policy as a Pi extension on a Pi AgentSession.
//
// Pi owns the loop: it sequences provider requests and tool calls, queues
// steering and follow-ups, aborts, persists the session tree and draws the
// TUI. ORA plugs in through public hooks only:
//   context       → ORA context policy (projection, L1–L6) before every request
//   provider      → `ora` provider whose streamSimple runs the ORA exchange
//   tool_call     → recording, one-mutation rule, gates and refusals
//   tool execute  → Python tools over RPC
//   turn_end      → continuations ("no action", truncated, rejected) and budgets
//   before_agent_start → ORA's system text
//   session_before_compact → cancelled (ORA compacts in its own policy)
// Tool-result `terminate` ends a Pi run where ORA's turn ends (action
// submitted, episode finished, call budget spent).

import { join } from "node:path";

import {
  createAssistantMessageEventStream, InMemoryCredentialStore, Type,
  type AssistantMessage, type Model,
} from "@earendil-works/pi-ai";
import {
  createAgentSessionFromServices, createAgentSessionRuntime, createAgentSessionServices, ModelRuntime,
  SessionManager, SettingsManager, type AgentSession, type AgentSessionRuntime, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

import { dumpsPy, parsePy } from "../pyjson.ts";
import { UserCredentialStore } from "./credentials.ts";
import type { OraCore, OraMessage } from "./session.ts";
import { timed } from "./profile.ts";
import { dehydrate, hydrate } from "./persistence.ts";

type Json = any;

export const UNKNOWN_TOOL = "ora_unadvertised_tool";

function zeroUsage() {
  return {
    input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function piUsage(usage: Json) {
  const value = zeroUsage();
  if (usage && typeof usage === "object") {
    value.input = Number(usage.input_tokens ?? 0) || 0;
    value.output = Number(usage.output_tokens ?? 0) || 0;
    value.cacheRead = Number(usage.cached_input_tokens ?? 0) || 0;
    value.totalTokens = Number(usage.total_tokens ?? value.input + value.output) || 0;
  }
  return value;
}

function textOf(content: Json): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((block: Json) => (block?.type === "text" ? block.text : block?.type === "image_url" ? "[image]" : "")).join("\n");
}

/** Pi-typed rendering of the projected ORA request messages (for Pi's own accounting). */
export function toPiMessages(messages: Json[], model: Model<any>): Json[] {
  const result: Json[] = [];
  const now = Date.now();
  for (const message of messages) {
    if (message.role === "system") continue;
    if (message.role === "user") {
      const content = typeof message.content === "string"
        ? [{ type: "text", text: message.content }]
        : (message.content as Json[]).map((block) => {
          if (block.type === "image_url") {
            const url = String(block.image_url.url);
            const comma = url.indexOf(",");
            return { type: "image", mimeType: url.slice(5, url.indexOf(";")), data: url.slice(comma + 1) };
          }
          return { type: "text", text: String(block.text ?? "") };
        });
      result.push({ role: "user", content, timestamp: now });
    } else if (message.role === "assistant") {
      const content: Json[] = [];
      if (typeof message.content === "string" && message.content) content.push({ type: "text", text: message.content });
      for (const call of message.tool_calls ?? []) {
        let args: Json = {};
        try {
          args = JSON.parse(String(call.function?.arguments ?? "{}"));
        } catch {
          args = {};
        }
        content.push({ type: "toolCall", id: String(call.id), name: String(call.function?.name), arguments: args });
      }
      result.push({
        role: "assistant", content, api: model.api, provider: model.provider, model: model.id,
        usage: zeroUsage(), stopReason: content.some((block) => block.type === "toolCall") ? "toolUse" : "stop",
        timestamp: now,
      });
    } else if (message.role === "tool") {
      result.push({
        role: "toolResult", toolCallId: String(message.tool_call_id), toolName: String(message.name),
        content: [{ type: "text", text: String(message.content) }], isError: false, timestamp: now,
      });
    }
  }
  return result;
}

function reasoningOf(assistant: OraMessage): string | null {
  const state = assistant._provider_continuation_state;
  const payload = state && typeof state === "object" ? state.payload : null;
  return payload && typeof payload.reasoning_content === "string" ? payload.reasoning_content : null;
}

/** The Pi assistant message for one ORA reply (ORA's own message rides along as `ora`). */
function piAssistant(model: Model<any>, reply: Json | null, core: OraCore, registered: Set<string>): AssistantMessage {
  const base = {
    role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
    usage: zeroUsage(), timestamp: Date.now(),
  };
  if (reply === null) {
    const turn = core.turn;
    const fatal = turn?.fatal ?? null;
    return {
      ...base, content: [],
      stopReason: fatal ? "error" : "aborted",
      errorMessage: fatal ? String(fatal.message) : `ORA turn ended: ${turn?.outcome?.termination ?? "stopped"}`,
    } as AssistantMessage;
  }
  const content: Json[] = [];
  const reasoning = reasoningOf(reply.assistant);
  if (reasoning) content.push({ type: "thinking", thinking: reasoning });
  if (typeof reply.assistant.content === "string" && reply.assistant.content) {
    content.push({ type: "text", text: reply.assistant.content });
  }
  for (const call of reply.calls as Json[]) {
    const rawArguments = String(call?.function?.arguments ?? "");
    let args: Json = {};
    try {
      args = JSON.parse(rawArguments || "{}");
      if (typeof args !== "object" || args === null || Array.isArray(args)) args = {};
    } catch {
      args = {};
    }
    const name = String(call?.function?.name ?? "");
    content.push({
      type: "toolCall", id: String(call.id), name: registered.has(name) ? name : UNKNOWN_TOOL,
      arguments: args, rawArguments,
    });
  }
  return {
    ...base, content, usage: piUsage(reply.usage),
    stopReason: reply.calls.length ? "toolUse" : reply.finishReason === "length" ? "length" : "stop",
    ora: { model_call_id: reply.modelCallId, finish_reason: reply.finishReason, accepted: reply.outputRejection === null },
  } as AssistantMessage;
}

export interface BridgeOptions {
  agentDir: string;
  sessionDir: string | null;
  sessionFile?: string | null;
  /** Further extensions after ORA's own (the operator TUI: ora-pi/src/tui). */
  extensions?: Array<(pi: ExtensionAPI) => void>;
  /** The user-level credential store /login writes (none: credentials stay in memory). */
  authPath?: string | null;
}

export function oraExtension(core: OraCore) {
  return (pi: ExtensionAPI) => {
    const policy = core.config.policy;
    // ORA history and state live in the Pi session as custom entries (persisted, never sent).
    // Pi persists entries with JSON.stringify, which loses Python's int/float
    // distinction (80.0 -> 80); ORA records travel as lossless JSON text.
    core.historySink = {
      appended: (messages: OraMessage[]) => pi.appendEntry("ora.history", {
        pyjson: dumpsPy({ messages: messages.map((m) => dehydrate(m, core.runRoot)) }, { ensureAscii: false }),
      }),
      state: (snapshot: Json) => pi.appendEntry("ora.state", { pyjson: dumpsPy(snapshot, { ensureAscii: false }) }),
    };
    const registered = new Set<string>();
    pi.registerProvider("ora", {
      name: "ORA",
      baseUrl: String(core.config.provider.endpoint ?? "http://ora.invalid"),
      apiKey: "ora-provider-owns-credentials",
      api: "ora-native",
      models: [{
        id: String(policy.model), name: String(policy.model), reasoning: true,
        input: policy.supports_images ? ["text", "image"] : ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: Number(policy.context_window_tokens), maxTokens: Number(policy.max_output_tokens),
      }],
      streamSimple: (model: Model<any>, _context: Json, options?: Json) => {
        const stream = createAssistantMessageEventStream();
        // A Pi abort (Esc in the TUI) is an operator stop: the Supervisor owns it.
        options?.signal?.addEventListener("abort", () => core.operatorAbort(), { once: true });
        void (async () => {
          let message: AssistantMessage;
          try {
            const reply = await core.exchange(options?.signal);
            message = piAssistant(model, reply, core, registered);
          } catch (error) {
            const turn = core.turn;
            if (turn) turn.fatal = error as Error;
            message = piAssistant(model, null, core, registered);
          }
          stream.push({ type: "start", partial: message });
          if (message.stopReason === "error" || message.stopReason === "aborted") {
            stream.push({ type: "error", reason: message.stopReason, error: message });
          } else {
            stream.push({ type: "done", reason: message.stopReason as any, message });
          }
          stream.end(message);
        })();
        return stream;
      },
    } as Json);

    const definitions: Json[] = core.config.tools.definitions;
    for (const definition of [...definitions.map((item) => item.function), { name: UNKNOWN_TOOL, description: "(no such tool)" }]) {
      registered.add(String(definition.name));
      pi.registerTool({
        name: String(definition.name),
        label: String(definition.name),
        description: String(definition.description ?? ""),
        // ORA validates arguments in Python and refuses with its own errors, so
        // Pi's schema coercion must not touch them.
        parameters: Type.Object({}, { additionalProperties: true }),
        executionMode: "sequential",
        async execute(toolCallId: string) {
          const result = await core.executeTool(toolCallId);
          return { content: [{ type: "text", text: result.text }], details: result.details, terminate: result.terminate };
        },
      } as Json);
    }

    pi.on("before_agent_start", async () => (core.systemText !== null ? { systemPrompt: core.systemText } : undefined));

    pi.on("context", async () => {
      const projected = await timed("pi.context(prepareRequest)", () => core.prepareRequest());
      if (projected === null) return undefined;
      return { messages: await timed("pi.context(toPiMessages)", () => toPiMessages(projected, { api: "ora-native", provider: "ora", id: String(policy.model) } as Json)) };
    });

    pi.on("tool_call", async (event: Json) => {
      const decision = await core.beforeTool(String(event.toolCallId));
      if (!decision.block) return undefined;
      return { block: true, reason: decision.reason, terminate: decision.terminate };
    });

    pi.on("turn_end", async () => {
      const decision = core.endOfReply();
      if (!decision.continueTurn) return undefined;
      return {
        entries: [{
          type: "custom_message", customType: "ora.continuation", display: true,
          content: textOf(decision.continuation?.content), details: { ora_continuation: true },
        }],
        continue: true,
      };
    });

    // ORA's own policy compacts (L5/L6); Pi's compaction must never rewrite the request.
    pi.on("session_before_compact", async () => ({ cancel: true }));

    // Typed input is operator guidance (in a free-play session, the next
    // instruction). The Supervisor admits it (plan-bound operator control) and
    // delivers it before the next model call, so Pi's own steering queue never
    // reaches the model directly.
    pi.on("input", async (event: Json, ctx: Json) => {
      if (event.source === "extension") return { action: "continue" };
      const text = String(event.text ?? "").trim();
      if (!text) return { action: "handled" };
      const play = core.playState !== null;
      if (event.streamingBehavior === "followUp" && !play) {
        ctx?.ui?.notify?.("ORA follow-ups must be bound in the plan; send it as a steer (Enter).", "warning");
        return { action: "handled" };
      }
      try {
        await core.submitOperatorMessage(text, play ? "instruction" : "steer");
        ctx?.ui?.notify?.(!play ? "Steer queued; the Supervisor delivers it before the next model call."
          : core.playState?.running ? "Message queued; the agent reads it before its next model call (Esc stops the instruction)."
            : "Instruction sent.", "info");
      } catch (error) {
        ctx?.ui?.notify?.(`${play ? "Instruction" : "Steer"} refused: ${(error as Error).message}`, "error");
      }
      return { action: "handled" };
    });

    // The world cannot be rewound: a branch of a live episode would show the
    // model a history the robot did not live. Forks are for closed sessions.
    const liveGuard = async (_event: Json, ctx: Json) => {
      if (core.closed) return undefined;
      ctx?.ui?.notify?.("This session drives a live episode; fork or navigate it after the run lands.", "warning");
      return { cancel: true };
    };
    pi.on("session_before_fork", liveGuard);
    pi.on("session_before_tree", liveGuard);
    pi.on("session_before_switch", liveGuard);
  };
}

export async function createPiRuntime(core: OraCore, options: BridgeOptions): Promise<AgentSessionRuntime> {
  // An untrusted "project" keeps Pi from scanning ancestor directories for
  // .git and .agents/skills: the agent process may read only its own files.
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false },
    retry: { enabled: false },
  } as Json, { projectTrusted: false } as Json);
  const cwd = options.agentDir;
  const createRuntime = async ({ sessionManager, sessionStartEvent }: Json) => {
    // Credentials stay with the ORA provider: Pi gets an in-memory store, so it
    // writes no auth.json into the run (the seal refuses credential-like files).
    // /login and /logout keep provider credentials in the user-level ORA store (outside the run);
    // the run's own model and key stay bound by the RunPlan.
    const modelRuntime = await ModelRuntime.create({
      credentials: options.authPath ? new UserCredentialStore(options.authPath) : new InMemoryCredentialStore(),
      authPath: join(options.agentDir, "pi-unused-store.json"),
      modelsPath: null, modelsStorePath: join(options.agentDir, "models-store.json"), refreshOnCreate: false,
    } as Json);
    const services = await createAgentSessionServices({
      cwd, agentDir: options.agentDir, settingsManager, modelRuntime,
      resourceLoaderOptions: {
        extensionFactories: [oraExtension(core), ...(options.extensions ?? [])],
        noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
        systemPrompt: "",
      },
    } as Json);
    const result = await createAgentSessionFromServices({
      services, sessionManager, sessionStartEvent, noTools: "builtin",
    } as Json);
    return { ...result, services, diagnostics: services.diagnostics };
  };
  const sessionManager = options.sessionDir === null
    ? SessionManager.inMemory(cwd)
    : options.sessionFile
      ? SessionManager.open(options.sessionFile, options.sessionDir)
      : SessionManager.create(cwd, options.sessionDir);
  const runtime = await createAgentSessionRuntime(createRuntime as Json, {
    cwd, agentDir: options.agentDir, sessionManager,
  });
  const session = runtime.session;
  await session.bindExtensions({} as Json);
  durableFromFirstFlush(core, sessionManager);
  core.hostMemoryDetail = () => {
    const entries: Json[] = (sessionManager as Json).fileEntries ?? [];
    const byKind: Record<string, number> = {};
    for (const entry of entries) {
      const kind = entry.type === "custom" ? `custom:${entry.customType}` : String(entry.type);
      byKind[kind] = (byKind[kind] ?? 0) + JSON.stringify(entry).length;
    }
    return {
      pi_session_entries: entries.length, pi_session_bytes: byKind,
      pi_agent_messages: (session as Json).agent?.state?.messages?.length ?? null,
      pi_agent_messages_bytes: JSON.stringify((session as Json).agent?.state?.messages ?? []).length,
    };
  };
  core.hostMemoryHolders = () => {
    const entries: Json[] = (sessionManager as Json).fileEntries ?? [];
    const dropData = (customType: string) => () => {
      for (const entry of entries) if (entry.type === "custom" && entry.customType === customType) entry.data = null;
    };
    return [
      ["pi_session_ora_state", dropData("ora.state")],
      ["pi_session_ora_history", dropData("ora.history")],
      ["pi_session_other_entries", () => { (sessionManager as Json).fileEntries = []; (sessionManager as Json).byId = new Map(); }],
      ["pi_agent_messages", () => { const state = (session as Json).agent?.state; if (state) state.messages = []; }],
    ];
  };
  const model = session.modelRuntime.getModel("ora", String(core.config.policy.model));
  if (!model) throw new Error("the ora provider model did not register");
  await session.setModel(model as Json);
  return runtime;
}

/** Run one ORA turn on the Pi session: the observation starts a Pi run; ORA decides when it ends. */
export async function runTurn(session: AgentSession, core: OraCore, display: string, details: Json): Promise<void> {
  await session.sendCustomMessage(
    { customType: "ora.observation", content: display, display: true, details },
    { triggerTurn: true },
  );
  await session.waitForIdle();
}

/**
 * Pi creates a session file only once the conversation holds a user or
 * assistant message, but ORA's turn opens with a custom message and its
 * ora.* entries must be durable from the first flush: a restart inside the
 * first turn resumes from them. The first ORA flush therefore writes the
 * file as Pi's own first write would (every entry so far, then appends).
 * This uses Pi 1.3.0's SessionManager internals (`flushed`, `_rewriteFile`),
 * checked here so a Pi upgrade that moves them fails at startup, not silently.
 */
function durableFromFirstFlush(core: OraCore, manager: Json): void {
  const sink = core.historySink;
  if (sink === null) throw new Error("the ORA extension did not bind its history sink");
  if (!manager.isPersisted() || !manager.getSessionFile()) return;
  if (typeof manager.flushed !== "boolean" || typeof manager._rewriteFile !== "function") {
    throw new Error("Pi's SessionManager no longer exposes the first session-file write ORA relies on");
  }
  const persist = () => {
    if (manager.flushed) return;
    manager._rewriteFile();
    manager.flushed = true;
  };
  core.historySink = {
    appended: (messages) => {
      sink.appended(messages);
      persist();
    },
    state: (snapshot) => {
      sink.state(snapshot);
      persist();
    },
  };
}

/** The ora.history messages and ora.state snapshots on the active branch, in order. */
export function persistedOraEntries(session: AgentSession, runRoot: string): { history: OraMessage[]; states: Json[] } {
  const history: OraMessage[] = [];
  const states: Json[] = [];
  for (const entry of (session.sessionManager as Json).getBranch()) {
    if (entry.type !== "custom") continue;
    if (entry.customType === "ora.history") history.push(...parsePy(entry.data.pyjson).messages.map((m: Json) => hydrate(m, runRoot)));
    if (entry.customType === "ora.state") states.push(parsePy(entry.data.pyjson));
  }
  return { history, states };
}

export function observationDisplay(message: OraMessage): string {
  return textOf(message.content).slice(0, 4000);
}

export { join, parsePy };
