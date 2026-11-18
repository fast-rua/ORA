// The Pi runtime of a free-play run: Pi as the user runs it, plus the robot.
//
// Unlike a benchmark run's runtime (core/pi_bridge.ts: in-memory settings, no
// skills, templates, themes, context files or built-in tools, compaction and
// retries off, ORA's own context and provider), this one loads the user's Pi:
// their settings, skills, prompt templates, themes, AGENTS.md context files
// and extensions from their agent dir, Pi's coding tools (read, write, edit,
// bash) in the run's workspace, Pi's compaction and retries, and Pi's model
// registry. Credentials from /login go to a user-level ORA store outside the
// run. Pi's codemode tool is on: the model may write a JavaScript script that
// calls the tools from Pi's sandbox; a script's robot calls go through the
// run's robot socket, as `ora-robot` from the shell does. ORA adds
// (oraPlayExtension): its provider profile as the `ora` provider, the robot
// tools, the system layers, the operator's inbox before every model call, and
// the evidence of everything that happens. The profile's request_options are
// part of the `ora` model itself, so every request Pi builds for it carries
// them, and no other provider's model does.

import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";

import {
  createAgentSessionFromServices, createAgentSessionRuntime, createAgentSessionServices, createCodemodeExtension,
  createLocalBashOperations, getAgentDir, ModelRuntime, SessionManager, SettingsManager, type AgentSessionRuntime,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { Type } from "@earendil-works/pi-ai";
import { streamSimple as openaiCompletionsStream } from "@earendil-works/pi-ai/api/openai-completions";

import type { OraCore } from "../core/session.ts";
import { t } from "../tui/i18n.ts";
import { PolicySessionError, PyError } from "../provider/py.ts";
import { validateRequestOptions } from "../provider/wire_options.ts";
import { PyFloat } from "../pyjson.ts";

import { boundContextImages, contextImageWindow } from "./context_images.ts";
import { PipelinePredictor } from "./pipeline.ts";
import { wireName, type PlayDriver } from "./driver.ts";

type Json = any;

/** Credential-shaped environment variable names: never visible to the agent's shell. */
const CREDENTIAL = /(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|AUTH)/i;

/**
 * Remove credentials from this process's environment (Pi's shell and tools inherit it).
 * Returns the ORA provider key, read before it is removed.
 */
export function scrubCredentials(env: NodeJS.ProcessEnv = process.env): { apiKey: string; removed: string[] } {
  const apiKey = String(env.ORA_PI_API_KEY ?? "");
  const removed: string[] = [];
  for (const key of Object.keys(env)) {
    if (key === "ORA_ROBOT_SOCKET") continue;
    if (CREDENTIAL.test(key) || key === "ORA_PI_STANDBY_API_KEYS") {
      delete env[key];
      removed.push(key);
    }
  }
  return { apiKey, removed: removed.sort() };
}

/** A validated option value as Pi sends it: plain JSON (a Python float crosses the RPC as a PyFloat). */
function plainJson(value: Json, path: string): Json {
  if (value instanceof PyFloat) return value.value;
  if (typeof value === "bigint") {
    throw new PolicySessionError(`request_options.${path} is an integer free play cannot send exactly`);
  }
  if (Array.isArray(value)) return value.map((item, index) => plainJson(item, `${path}[${index}]`));
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, plainJson(item, `${path}.${key}`)]));
  }
  return value;
}

/**
 * The profile's `request_options` (the provider spec's, as an episode's ORA provider gets them,
 * core/deps.ts), validated as that provider validates them when it is built; null for none.
 */
export function playRequestOptions(spec: Json): Json | null {
  const raw = spec?.request_options ?? null;
  if (raw === null) return null;
  if (spec.kind === "anthropic_messages") {
    throw new PolicySessionError("output_token_parameter, tool_strictness, reasoning_field and request_options "
      + "apply only to OpenAI-compatible providers");
  }
  let options: Json;
  try {
    options = validateRequestOptions(raw);
  } catch (error) {
    if (error instanceof PyError && error.pyType === "ValueError") {
      throw new PolicySessionError(`unsupported OpenAI request_options: ${error.message}`, { cause: error });
    }
    throw error;
  }
  return Object.fromEntries(Object.entries(options).map(([key, value]) => [key, plainJson(value, key)]));
}

/**
 * The ORA provider profile as a Pi provider (Pi's own request code, ORA's endpoint and key).
 *
 * `requestOptions` are the model's `samplingParams`: Pi's OpenAI-compatible request code adds them
 * to the body of every request for this model (a turn, compaction, a branch summary), as ORA's
 * provider adds them to every call of an episode, and a model of another provider never carries
 * them. Pi itself sends none of their fields for this model, except the `reasoning` object it uses
 * for the reasoning effort on OpenRouter-style endpoints: a profile that names `reasoning` itself
 * has Pi state the effort as `reasoning_effort`, the field ORA's provider sends.
 */
export function oraProvider(core: OraCore, apiKey: string, requestOptions: Json | null = null): Json {
  const spec = core.config.provider;
  const policy = core.config.policy;
  const api = spec.kind === "anthropic_messages" ? "anthropic-messages" : "openai-completions";
  return {
    name: `ORA${typeof policy.provider_profile === "string" ? ` (${policy.provider_profile})` : ""}`,
    baseUrl: String(spec.endpoint ?? "http://ora.invalid"),
    apiKey: apiKey || "ora-no-key",
    api,
    models: [{
      id: String(policy.model), name: String(policy.model), reasoning: reasoningLevel(policy) !== null,
      input: policy.supports_images ? ["text", "image"] : ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: Number(policy.context_window_tokens), maxTokens: Number(policy.max_output_tokens),
      // The request shape ORA's own provider uses for this profile (provider/openai.ts): a system
      // message rather than "developer", no `store`, the profile's output-token field, its request_options.
      ...(api === "openai-completions" ? {
        compat: {
          supportsDeveloperRole: false, supportsStore: false,
          maxTokensField: spec.output_token_parameter === "max_completion_tokens" ? "max_completion_tokens" : "max_tokens",
          supportsReasoningEffort: reasoningLevel(policy) !== null,
          ...(requestOptions !== null && "reasoning" in requestOptions ? { thinkingFormat: "openai" } : {}),
        },
        ...(requestOptions !== null ? { samplingParams: requestOptions } : {}),
      } : {}),
    }],
  };
}

const THINKING_LEVELS = new Set(["minimal", "low", "medium", "high", "xhigh"]);

/** The profile's reasoning effort as a Pi thinking level, or null when the profile leaves it to the provider. */
export function reasoningLevel(policy: Json): string | null {
  const effort = String(policy.reasoning_effort ?? "provider_default");
  return THINKING_LEVELS.has(effort) ? effort : null;
}

function firstLine(text: string): string {
  return (text.split("\n").find((line) => line.trim()) ?? "").trim();
}

/**
 * An operator message for the Supervisor's inbox: an `instruction` (it opens one, or joins the running
 * one), or with `queued` a `queued_instruction` (held until the running instruction ends, then the next
 * one; with none running it opens one at once). `notify` says at once where it goes (a terminal).
 */
export async function submitPlayMessage(core: OraCore, driver: PlayDriver, text: string, images: Json[] = [],
  options: { queued?: boolean; notify?: ((notice: string) => void) | null } = {}): Promise<Json> {
  const message = text.trim() || "(see the attached image)";
  const play = core.playState;
  const running = play?.running ? Number(play.instructions) || null : null;
  const queued = options.queued === true;
  options.notify?.(running === null ? t("play.instructionQueued")
    : queued ? t("play.queuedNext", { n: running }) : t("play.messageQueued", { n: running }));
  const record = await core.submitOperatorMessage(message, queued ? "queued_instruction" : "instruction");
  if (images.length && typeof record?.id === "string") driver.attachments.set(record.id, images);
  if (typeof record?.id === "string") driver.messages.submitted(record.id, message, running, queued);
  return record;
}

/** ORA's part of a free-play runtime. */
export function oraPlayExtension(core: OraCore, driver: PlayDriver, apiKey: string) {
  return (pi: ExtensionAPI) => {
    driver.pi = pi;
    const provider = oraProvider(core, apiKey, playRequestOptions(core.config.provider));
    if (driver.init.pipeline_predict === true && provider.api === "openai-completions") {
      // The ORA model's requests go through the predictor (play/pipeline.ts); every other model's do not.
      driver.predictor = new PipelinePredictor(openaiCompletionsStream as Json, {
        now: () => Date.now(),
        record: (fields: Json, artifact: Json) => core.append("pipeline_prediction", {
          visibility: "exact", payload: { ...fields, play: true },
          artifact_ref: core.artifacts.putJson(artifact, { kind: "play_pipeline_prediction" }),
        }),
      });
      const predictor = driver.predictor;
      provider.streamSimple = (model: Json, context: Json, options?: Json) => predictor.stream(model, context, options);
    }
    pi.registerProvider("ora", provider);
    const robotTools = new Set<string>();
    for (const definition of core.config.tools.definitions) {
      const fn = definition.function ?? {};
      const name = wireName(String(fn.name));
      robotTools.add(name);
      pi.registerTool({
        name, label: String(fn.name), description: String(fn.description ?? ""),
        promptSnippet: `${String(fn.name)}: ${firstLine(String(fn.description ?? "robot tool"))}`,
        parameters: Type.Unsafe(fn.parameters ?? { type: "object", properties: {} }),
        executionMode: "sequential",
        async execute(toolCallId: string, params: Json, _signal: unknown, onUpdate?: (partial: Json) => void) {
          // onUpdate: a code-mode command's robot calls as they happen (PlayDriver.noteRecord).
          return driver.robotTool(name, params, toolCallId, onUpdate);
        },
      } as Json);
    }

    // ORA's layers (planner, robot project, embodiment, the play episode task) follow Pi's own prompt.
    pi.on("before_agent_start", async (event: Json) =>
      driver.systemAppend ? { systemPrompt: `${event.systemPrompt}\n\n${driver.systemAppend}` } : undefined);

    // The operator's inbox and the act's budget, before every model call; a stop ends the run here.
    // The call then carries the images the plan's window sends (context_images.ts): pruned in batches
    // where an instruction begins, so a long session's request stays under the provider's body limit
    // and the calls of one instruction share a prefix.
    pi.on("context", async (event: Json, ctx: Json) => {
      if (await driver.boundary() !== null) ctx.abort();
      driver.noteContext(event.messages ?? []);
      const bounded = boundContextImages(event.messages ?? [], contextImageWindow(driver.init.context_images));
      return bounded.omitted ? { messages: bounded.messages } : undefined;
    });

    // A plan's request_priority adds `priority` to the ORA model's OpenAI-compatible requests (vLLM's and
    // SGLang's priority scheduling); the payload recorded is the payload sent.
    pi.on("before_provider_request", async (event: Json, ctx: Json) => {
      const priority = ctx.model?.provider === "ora" && ctx.model?.api === "openai-completions"
        && event.payload !== null && typeof event.payload === "object" ? driver.requestPriority() : null;
      const payload = priority === null ? event.payload : { ...event.payload, priority };
      driver.providerRequest(payload, ctx.model);
      return priority === null ? undefined : payload;
    });
    // The reply's first content times the call's first token (play/perf.ts; display only).
    pi.on("message_update", async (event: Json) => {
      if (event.message?.role === "assistant") driver.perf.streamed(event.assistantMessageEvent);
    });
    pi.on("message_end", async (event: Json) => {
      if (event.message?.role === "assistant") driver.assistantMessage(event.message);
    });
    // A reply the output limit cut off before any tool call: ORA's continuation and one more call
    // (the next call's boundary still checks the inbox and the budget).
    pi.on("turn_end", async (event: Json) => driver.afterTurn(event.message));
    // A call another tool made (a codemode script's) names the call that made it.
    const codemodeCalls = new Set<string>();
    const origin = (event: Json) => {
      const parent = event.parentToolCallId ? String(event.parentToolCallId) : null;
      return parent === null ? {}
        : { origin: { kind: codemodeCalls.has(parent) ? "codemode" : "nested", parent_tool_call_id: parent } };
    };
    pi.on("tool_call", async (event: Json) => {
      const id = String(event.toolCallId);
      if (event.toolName === "codemode" && !event.parentToolCallId) codemodeCalls.add(id);
      if (event.parentToolCallId && codemodeCalls.has(String(event.parentToolCallId))) {
        driver.codemodeCalls.set(id, String(event.parentToolCallId));
      }
      // A robot call is recorded by the robot tool itself, or by the robot socket for a script's.
      if (robotTools.has(String(event.toolName))) return undefined;
      driver.record("tool_call_proposed", { name: event.toolName, arguments: event.input ?? null, ...origin(event) },
        undefined, id);
      return undefined;
    });
    pi.on("tool_result", async (event: Json) => {
      if (robotTools.has(String(event.toolName))) return undefined;
      driver.record("tool_result", { name: event.toolName, ok: !event.isError, ...origin(event) },
        { content: event.content ?? [], details: event.details ?? null }, String(event.toolCallId));
      return undefined;
    });
    // The operator's `!` lines: Pi's own local shell, with the command and its output recorded.
    pi.on("user_bash", async (event: Json) => {
      driver.record("operator_shell_command", {
        command: event.command, cwd: event.cwd, exclude_from_context: Boolean(event.excludeFromContext),
      });
      const local = createLocalBashOperations();
      return {
        operations: {
          exec: async (command: string, cwd: string, options: Json) => {
            const chunks: Buffer[] = [];
            const result = await local.exec(command, cwd, {
              ...options, onData: (data: Buffer) => { chunks.push(Buffer.from(data)); options.onData(data); },
            });
            driver.record("operator_shell_result", { command, exit_code: result.exitCode },
              { output: Buffer.concat(chunks).toString("utf8") });
            return result;
          },
        },
      };
    });
    pi.on("model_select", async (event: Json) => {
      const describe = (model: Json) => (model ? { provider: model.provider, model: model.id, api: model.api } : null);
      driver.record("model_selected", { from: describe(event.previousModel), to: describe(event.model), source: event.source });
    });
    pi.on("thinking_level_select", async (event: Json) => {
      driver.record("thinking_level_selected", { from: event.previousLevel, to: event.level });
    });
    pi.on("session_compact", async (event: Json) => {
      driver.record("conversation_compacted", { reason: event.reason, entry_id: event.compactionEntry?.id ?? null },
        { summary: event.compactionEntry?.summary ?? null });
    });
    pi.on("session_tree", async (event: Json) => {
      driver.record("conversation_navigated", { old_leaf: event.oldLeafId, new_leaf: event.newLeafId });
    });
    pi.on("session_start", async (event: Json, ctx: Json) => {
      if (["new", "resume", "fork"].includes(event.reason)) {
        driver.record("conversation_switched", {
          reason: event.reason, previous_session_file: event.previousSessionFile ?? null,
          session_file: ctx.sessionManager?.getSessionFile?.() ?? null,
        });
      }
    });
    // Typed text is the operator's message: the Supervisor admits it and delivers it before the next
    // model call (pasted images ride along). Between instructions it starts the next one; while one
    // runs it joins it, and the agent decides whether it is a question or a new objective. A queued
    // one (Pi's follow-up, Alt+Enter: `queuedMessage`) waits for the running instruction to end and
    // then opens the next. `!` lines and Pi's commands never reach this hook.
    pi.on("input", async (event: Json, ctx: Json) => {
      if (event.source === "extension") return { action: "continue" };
      const queued = driver.takeQueued() || event.streamingBehavior === "followUp";
      const text = String(event.text ?? "").trim();
      const images: Json[] = Array.isArray(event.images) ? event.images : [];
      if (!text && !images.length) return { action: "handled" };
      try {
        // The terminal says where it goes at once: while the robot moves, the Supervisor admits it only
        // once the move ends. A page (Pi RPC) shows the message's own state in its language instead.
        await submitPlayMessage(core, driver, text, images, {
          queued, notify: event.source === "rpc" ? null : (notice) =>
            (ctx?.ui?.notifyLocal ?? ctx?.ui?.notify)?.call(ctx?.ui, notice, "info"),
        });
      } catch (error) {
        ctx?.ui?.notify?.(t("play.refused", { reason: (error as Error).message }), "error");
      }
      return { action: "handled" };
    });

    // The world does not branch: a conversation may move only between instructions.
    const between = async (_event: Json, ctx: Json) => {
      if (driver.turn === null) return undefined;
      ctx?.ui?.notify?.(t("play.waitToMove"), "warning");
      return { cancel: true };
    };
    pi.on("session_before_fork", between);
    pi.on("session_before_tree", between);
    pi.on("session_before_switch", between);
  };
}

export interface PlayRuntimeOptions {
  sessionDir: string;
  sessionFile: string | null;
  /**
   * A real cell (play/cell.ts): no coding tools and no codemode. Bash, file writes or a
   * codemode script on the cell's computer could reach its buses and drivers directly,
   * around the Supervisor, its admission and the operator's approval; the robot tools,
   * which go through them, are all the model gets.
   */
  robotToolsOnly?: boolean;
  /**
   * ORA's code mode (plan play.code_mode): ORA's tools are all the model gets. Its read, write, edit and bash
   * (scripts in ORA's sandbox, their robot calls served by the Supervisor) replace Pi's built-in tools and its
   * codemode, whose same-named tools would otherwise be replaced silently.
   */
  codeMode?: boolean;
  extensions: Array<(pi: ExtensionAPI) => void>;
  apiKey: string;
}

/** Pi's own runtime for the free-play run (the user's Pi, the run's workspace). */
export async function createPlayRuntime(core: OraCore, driver: PlayDriver, options: PlayRuntimeOptions): Promise<AgentSessionRuntime> {
  const agentDir = driver.init.agent_dir ?? getAgentDir();
  const cwd = driver.init.workspace;
  // Refused here with its own message: Pi reports an extension's failure only as a diagnostic.
  const requestOptions = playRequestOptions(core.config.provider);
  if (options.codeMode && options.robotToolsOnly) {
    // A real cell's Esc stops only the current motion; a script's later motions would still be asked for.
    throw new PolicySessionError("code mode in free play is not available on a real cell yet");
  }
  // Pi's codemode and built-in tools, unless ORA's own tools are all the model gets.
  const piTools = !options.robotToolsOnly && !options.codeMode;
  const createRuntime = async ({ cwd: runtimeCwd, agentDir: runtimeAgentDir, sessionManager, sessionStartEvent }: Json) => {
    const settingsManager = SettingsManager.create(runtimeCwd, runtimeAgentDir);
    const modelRuntime = await ModelRuntime.create({
      // Pi's catalogs, changelog and package installs reach the network unless the user set PI_OFFLINE.
      authPath: driver.init.auth_path, modelsPath: join(runtimeAgentDir, "models.json"),
      allowModelNetwork: !process.env.PI_OFFLINE, modelRefreshTimeoutMs: 5000,
      signal: AbortSignal.timeout(15_000),
    } as Json);
    const services = await createAgentSessionServices({
      cwd: runtimeCwd, agentDir: runtimeAgentDir, settingsManager, modelRuntime,
      resourceLoaderOptions: { extensionFactories: [
        // Pi's codemode, without its `models` global (the user's models, called with their credentials).
        ...(piTools ? [
          { name: "codemode", factory: createCodemodeExtension({ models: false }), replaceable: true }] : []),
        oraPlayExtension(core, driver, options.apiKey), ...options.extensions,
      ] },
    } as Json);
    const created = await createAgentSessionFromServices({
      services, sessionManager, sessionStartEvent,
      ...(options.codeMode ? { noTools: "builtin" }
        : options.robotToolsOnly ? { noTools: "builtin", excludeTools: ["bash", "codemode", "write", "edit"] }
        : { tools: ["+codemode"] }),
    } as Json);
    driver.session = created.session;
    return { ...created, services, diagnostics: services.diagnostics };
  };
  const sessionManager = options.sessionFile
    ? SessionManager.open(options.sessionFile, options.sessionDir)
    : SessionManager.create(cwd, options.sessionDir);
  const runtime = await createAgentSessionRuntime(createRuntime as Json, { cwd, agentDir, sessionManager });
  await runtime.session.bindExtensions({} as Json);
  driver.session = runtime.session;
  const model = runtime.session.modelRuntime.getModel("ora", String(core.config.policy.model));
  if (!model) throw new Error("the ORA provider model did not register");
  // The user's Pi models.json may override a model's sampling (providers.ora.modelOverrides); the
  // profile's request_options decide the ORA model's.
  if (!isDeepStrictEqual((model as Json).samplingParams ?? null, requestOptions)
      || (model as Json).samplingParamsByThinkingLevel !== undefined) {
    throw new PolicySessionError("the ORA provider's model sends the profile's request_options and nothing else; "
      + "remove the sampling overrides for provider `ora` from your Pi models.json");
  }
  if (runtime.session.model?.provider !== "ora" || runtime.session.model?.id !== model.id) {
    await runtime.session.setModel(model as Json);
  }
  // The profile's reasoning effort is where a session starts; /thinking changes it.
  const level = reasoningLevel(core.config.policy);
  if (level !== null && runtime.session.thinkingLevel !== level) runtime.session.setThinkingLevel(level as Json);
  return runtime;
}
