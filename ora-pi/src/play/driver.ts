// Free play (`ora play`, plan episode_mode "play") on Pi's own agent.
//
// A benchmark turn is ORA's: the `context` hook replaces Pi's messages with
// ORA's projection and the `ora` provider runs ORA's exchange. Free play runs
// Pi as a coding agent instead: Pi's own messages, compaction, model registry
// (/model, /login) and coding tools, with the robot as a device:
//   - the Supervisor still drives one act per observation; an act is one Pi
//     run, started by the scene's observation (a custom message the model
//     reads) and, at an instruction's start, the operator's instruction as a
//     user message. A new instruction's observation is a fresh look the
//     Supervisor took for it (episode/play.py `_observe_for_instruction`); when
//     that look failed, the delivery's `observation_note` follows the
//     observation it describes;
//   - ORA's robot tools are Pi tools that run in Python (tools.invoke); a
//     world-changing call ends the run, and the Supervisor executes it;
//   - before every model call the run checks the operator's inbox (Esc,
//     /reset, /quit) and the act's budget, and stops there; a message the
//     operator sends meanwhile joins the running instruction: it reaches the
//     model as an `ora.play_message` (a user message on the wire, with the
//     instruction it joined and the step it was read at for the terminal and
//     the page), steered into the run or sent with the next observation
//     (`deliver`);
//   - every model call, tool call, shell command, model or thinking change,
//     compaction and conversation branch is recorded in agent-events.jsonl.

import type { AgentSession, ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { dumpsSorted } from "../pyjson.ts";
import { executionEvidence } from "../core/execution.ts";
import { applyHints, StaticFactsState, type Hint } from "../core/hints.ts";
import type { ActParams, OraCore, OraMessage, Outcome } from "../core/session.ts";
import type { AgentEvent } from "../evidence/recorder.ts";
import { PlayMessages } from "./messages.ts";
import { PlayPerf } from "./perf.ts";
import type { PipelinePredictor } from "./pipeline.ts";
import { robotSocketCall } from "./robot_socket.ts";

/** A message the operator sent while an instruction ran, as Pi keeps it (play/messages.ts). */
export const PLAY_MESSAGE = "ora.play_message";

/**
 * What the model reads after a reply the provider's output limit cut off before any tool call (an
 * episode's `afterTruncatedOutput`, core/continuations.ts, for free play: a reply in words is a valid
 * end here, so the model may also answer).
 */
export const PLAY_TRUNCATED_CONTINUATION = "Your previous response reached the Provider output limit before it was "
  + "complete. No tool ran and the world did not change. Continue working on this same observation, more briefly: "
  + "use the available tools, or reply in words.";

type Json = any;

/** What Python's PiPlanner sends for a free-play run (init params `play`). */
export interface PlayInit {
  /** The run's workspace: the cwd of Pi's coding tools and shell. */
  workspace: string;
  /** User-level credential store for /login (outside the run). */
  auth_path: string;
  /** The user's Pi agent dir (skills, extensions, settings); null for Pi's default. */
  agent_dir: string | null;
  /** How many of the latest observations and tool results keep their images in a request (plan.play). */
  context_images?: { observations: number; tool_results: number; prune_at?: number } | null;
  /**
   * The `priority` each ORA model request carries (plan play.request_priority; none when absent):
   * `correction` for the call that first reads an operator message joined to the running instruction,
   * `default` for every other call. The server's convention decides which value runs first.
   */
  request_priority?: { correction: number; default: number } | null;
  /** Predict the next step while a motion runs (plan play.pipeline_predict; play/pipeline.ts). */
  pipeline_predict?: boolean;
  /**
   * ORA's code mode (plan play.code_mode): ORA's read, write, edit and bash (scripts in ORA's sandbox) are the
   * model's tools, in place of Pi's own and its codemode (play/runtime.ts).
   */
  code_mode?: boolean;
}

interface PlayTurn {
  step: number;
  observationEvent: AgentEvent;
  calls: number;
  callLimit: number;
  deadlineMs: number | null;
  outcome: Outcome | null;
  submitted: { action_event_id: string; tool_name: string } | null;
  modelCallId: string | null;
  lastStopReason: string | null;
  lastError: string | null;
}

/** The sequence number of a play model call id (`play-call-000012` -> 12). */
function callNumber(modelCallId: string | null): number | null {
  const match = /^play-call-(\d+)$/.exec(modelCallId ?? "");
  return match ? Number(match[1]) : null;
}

/** An operator message delivered for the next run (or, mid-run, steered in). */
interface Delivered {
  id: string;
  message: string;
  images: Json[];
  /** It joined the running instruction (episode_play.py), rather than opening one. */
  joined: boolean;
  instructionId: string | null;
  /** The instruction's number (1-based) as the Supervisor last announced it. */
  instruction: number | null;
  step: number | null;
  /** An opening instruction whose fresh observation failed: what the model reads with the observation. */
  observationNote: string | null;
}

function ended(termination: string, attempts: number): Outcome {
  return { submitted: false, termination, attempts };
}

/** A tool name the provider wire accepts (OpenAI-style names allow [A-Za-z0-9_-]). */
export function wireName(name: string): string {
  return name.replace(/[^A-Za-z0-9_-]/g, "_");
}

/** ORA content blocks (text, image_url data URLs) as Pi content (text, image). */
export function piContent(content: Json): Json[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  const blocks: Json[] = [];
  for (const block of Array.isArray(content) ? content : []) {
    if (block?.type === "image_url") {
      const url = String(block.image_url?.url ?? "");
      const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
      if (match) blocks.push({ type: "image", mimeType: match[1], data: match[2] });
    } else if (block?.type === "text") {
      blocks.push({ type: "text", text: String(block.text ?? "") });
    }
  }
  return blocks;
}

function textOf(message: OraMessage): string {
  const content = message.content;
  if (typeof content === "string") return content;
  return (Array.isArray(content) ? content : []).map((block: Json) => block?.text ?? "").filter(Boolean).join("\n");
}

export class PlayDriver {
  readonly core: OraCore;
  readonly init: PlayInit;
  session: AgentSession | null = null;
  pi: ExtensionAPI | null = null;
  turn: PlayTurn | null = null;
  /** ORA's composed layers (planner, project, episode task), appended to Pi's system prompt. */
  systemAppend: string | null = null;
  private readonly delivered: Delivered[] = [];
  private readonly staticFacts = new StaticFactsState();
  private callSequence = 0;
  readonly canonicalByWire = new Map<string, string>();
  /** Images an instruction typed here carried (pasted), by operator command id. */
  readonly attachments = new Map<string, Json[]>();
  /** Calls a codemode script made (Pi's nested tool calls), mapped to the codemode call that made them. */
  readonly codemodeCalls = new Map<string, string>();
  /** The operator's messages as the terminal and the page show them (display only). */
  readonly messages = new PlayMessages();
  /** Each model call's and robot action's timing, for the operator (play/perf.ts; display only). */
  readonly perf = new PlayPerf();
  /**
   * The model call that first read each joined operator message (by its context: Pi steers a message
   * in after the current tool batch, so the call that reads it is known only there).
   */
  readonly firstRead = new Map<string, number>();
  /** When the operator last asked for the next typed message to be queued (Alt+Enter in the terminal). */
  private queueAskedAt: number | null = null;
  /**
   * Tool calls the Supervisor ran through its action executor whose results Pi has since recorded: settled
   * before the next model call (`tools.settled`, as an episode's core does after each reply).
   */
  private pendingToolDelivery: { model_call_id: string; tool_call_id: string }[] = [];
  /**
   * The code-mode command running now (ORA's bash): the robot calls its program made so far and the images they
   * returned, shown as the command's partial result (Pi's onUpdate) while it runs; the final result replaces it.
   */
  private liveScript: { parent: string; update: (partial: Json) => void; calls: Json[]; images: Json[] } | null = null;

  constructor(core: OraCore, init: PlayInit) {
    this.core = core;
    this.init = init;
    for (const definition of core.config.tools.definitions) {
      const name = String(definition.function?.name);
      this.canonicalByWire.set(wireName(name), name);
    }
  }

  /**
   * The terminal's Alt+Enter: the message Pi submits next (Pi's own handler, a moment later) is a
   * `queued_instruction`. Pi submits it as a follow-up only while its agent streams; while the robot
   * moves Alt+Enter is Pi's Enter, so the terminal says it here first.
   */
  queueNext(now = Date.now()): void {
    this.queueAskedAt = now;
  }

  /** Whether the message being submitted was asked to be queued (taken once, within a second). */
  takeQueued(now = Date.now()): boolean {
    const asked = this.queueAskedAt;
    this.queueAskedAt = null;
    return asked !== null && now - asked <= 1000;
  }

  // -- One act: an observation, maybe an instruction, one Pi run ------------------------

  async act(params: ActParams): Promise<Outcome> {
    const session = this.session;
    if (session === null) throw new Error("the free-play session is not ready");
    if (this.turn !== null) throw new Error("a free-play act is already running");
    if (params.static !== null) {
      // The planner layer is the system text; project and episode-task layers follow it (no duplicate system).
      this.systemAppend = [params.static.system_text,
        ...params.static.messages.filter((item) => item.role !== "system").map(textOf)]
        .filter((text) => text && text.trim()).join("\n\n");
    } else if (this.systemAppend === null) {
      throw new Error("the first free-play act must carry the static context");
    }
    const { message } = this.core.resolveDraft(params.draft);
    const meta = message._observation ?? {};
    this.perf.observed(params.step);
    const observationEvent = this.core.append("observation_received", {
      observation_step: params.step,
      payload: { observation_id: meta.observation_id ?? null, state_ref: meta.state_ref ?? null, play: true },
    });
    const remainingCalls = Number(params.remaining_budget?.model_calls);
    const wall = params.remaining_budget?.wall_time_s;
    const wallSeconds = typeof wall === "object" && wall !== null && "value" in wall ? Number(wall.value) : Number(wall);
    this.turn = {
      step: params.step, observationEvent, calls: 0,
      callLimit: Number.isFinite(remainingCalls) ? remainingCalls : Number.POSITIVE_INFINITY,
      deadlineMs: Number.isFinite(wallSeconds) ? performance.now() + wallSeconds * 1000 : null,
      outcome: null, submitted: null, modelCallId: null, lastStopReason: null, lastError: null,
    };
    // The operator's instruction (or a stop) reaches the run before it starts.
    const stop = await this.boundary();
    if (stop !== null) return this.finish();
    const content = piContent(message.content);
    const delivered = this.delivered.splice(0);
    const opening = delivered.filter((item) => !item.joined);
    const joined = delivered.filter((item) => item.joined);
    for (const item of opening) if (item.observationNote) content.push({ type: "text", text: item.observationNote });
    await session.sendCustomMessage({
      customType: "ora.play_observation", content, display: true,
      details: { ora_observation_step: params.step, ora_observation_id: meta.observation_id ?? null },
    }, { triggerTurn: delivered.length === 0 });
    if (opening.length) {
      // The instruction is the user's own message (Pi's prompt templates and skills expand);
      // messages that joined it in the same delivery follow it into the same run.
      const text = opening.map((item) => item.message).join("\n\n");
      const images = opening.flatMap((item) => item.images);
      await session.sendUserMessage(images.length ? [{ type: "text", text }, ...images] : text,
        { expandPromptTemplates: true });
      for (const item of joined) await session.sendCustomMessage(this.playMessage(item), { deliverAs: "steer" });
    } else {
      // Messages that joined the running instruction while the robot moved: with this observation.
      for (const [index, item] of joined.entries()) {
        await session.sendCustomMessage(this.playMessage(item), { triggerTurn: index === joined.length - 1 });
      }
    }
    await session.waitForIdle();
    return this.finish();
  }

  /** A joined message as Pi keeps it: the operator's words (a user message to the model), and where it belongs. */
  private playMessage(item: Delivered): Json {
    return {
      customType: PLAY_MESSAGE, display: true,
      content: item.images.length ? [{ type: "text", text: item.message }, ...item.images] : item.message,
      details: { ora_message_id: item.id, ora_instruction_id: item.instructionId, ora_instruction: item.instruction,
        ora_observation_step: item.step, ora_joined: true },
    };
  }

  private finish(): Outcome {
    const turn = this.turn;
    if (turn === null) throw new Error("no free-play act to finish");
    this.turn = null;
    // An act that ended without submitting a motion leaves no prediction for the next one (play/pipeline.ts).
    if (turn.submitted === null) this.predictor?.cancel("act_ended");
    this.core.flush();
    if (turn.outcome !== null) return turn.outcome;
    if (turn.submitted !== null) {
      return { submitted: true, attempts: turn.calls, action_event_id: turn.submitted.action_event_id,
        tool_name: turn.submitted.tool_name };
    }
    if (this.core.interrupted) return ended("operator_interrupt", turn.calls);
    // A run that changed nothing ended on the agent's reply: its report, or a provider failure.
    if (turn.lastStopReason === "error") return ended("provider_error", turn.calls);
    if (turn.lastStopReason === "aborted") return ended("operator_interrupt", turn.calls);
    // Cut off by the output limit and not continued (afterTurn): no report.
    if (turn.lastStopReason === "length") return ended("output_truncated", turn.calls);
    // A text-only reply: the joined messages this reply's call was the first to read get it as their
    // answer (the Supervisor keeps the instruction running, episode_play.py); otherwise it is the report.
    const last = callNumber(turn.modelCallId);
    const answers = [...this.firstRead].filter(([, call]) => call === last).map(([id]) => id);
    return answers.length ? { ...ended("policy_no_action", turn.calls), answers } : ended("policy_no_action", turn.calls);
  }

  /**
   * After each model call (Pi's `turn_end`): a reply the provider's output limit cut off before any tool
   * call is no report. As in an episode (core/session.ts `endOfReplyInner`), the model goes on with the
   * same observation; the act's spent budget instead ends it as `output_truncated`. Returns Pi's
   * boundary result: ORA's continuation as the next message, and one more model call.
   */
  afterTurn(message: Json): Json | undefined {
    const turn = this.turn;
    if (turn === null || turn.outcome !== null || turn.submitted !== null || this.core.interrupted) return undefined;
    const content: Json[] = Array.isArray(message?.content) ? message.content : [];
    if (message?.role !== "assistant" || message.stopReason !== "length") return undefined;
    if (content.some((block) => block?.type === "toolCall")) return undefined;  // Pi fails those calls and goes on
    const spent = turn.calls >= turn.callLimit || (turn.deadlineMs !== null && performance.now() >= turn.deadlineMs);
    const text = content.filter((block) => block?.type === "text").map((block) => String(block.text ?? "")).join("");
    const event = this.core.append("provider_output_truncated", {
      visibility: "derived", observation_step: turn.step, model_call_id: turn.modelCallId,
      payload: {
        tool_call_count: 0, text_chars: text.length, play: true,
        policy_outcome: spent ? "output_truncated" : "continue_same_observation",
      },
    });
    if (spent) {
      turn.outcome = ended("output_truncated", turn.calls);
      return undefined;
    }
    // A joined message the cut-off call read first is answered by the call that continues it.
    const cut = callNumber(turn.modelCallId);
    for (const [id, call] of this.firstRead) if (call === cut) this.firstRead.set(id, this.callSequence);
    return {
      entries: [{
        type: "custom_message", customType: "ora.continuation", display: true, content: PLAY_TRUNCATED_CONTINUATION,
        details: { ora_continuation: true, ora_reason: "output_truncated", ora_event_id: event.event_id },
      }],
      continue: true,
    };
  }

  /**
   * Before a model call (and before the run starts): the act's submitted action, the operator's inbox
   * and the act's budget. Returns the stop that ends this act, or null to go on.
   */
  async boundary(): Promise<string | null> {
    const turn = this.turn;
    if (turn === null) return null;
    if (turn.outcome !== null) return turn.outcome.termination ?? "stopped";
    if (turn.submitted !== null) {
      // The act's one world-changing action is submitted: the act ends before any further model call, and
      // finish() hands that action to the Supervisor. Pi would otherwise go on past the terminating tool
      // batch when an operator message was steered in meanwhile; that message is already in Pi's
      // conversation, so the next act's first model call reads it.
      return "action_submitted";
    }
    if (this.core.interrupted) {
      turn.outcome = ended("operator_interrupt", turn.calls);
      return "operator_interrupt";
    }
    if (this.pendingToolDelivery.length) {
      // The results Pi recorded since the last call (its tool batch ended): their executions are delivered.
      await this.core.rpc.request("tools.settled", { calls: this.pendingToolDelivery.splice(0) });
    }
    const answer = await this.core.rpc.request("operator.poll", {
      observation_step: turn.step,
      before_model_call_id: `play-call-${String(this.callSequence).padStart(6, "0")}`,
      calls_in_act: turn.calls,
    });
    let stop: string | null = null;
    if (answer && typeof answer === "object" && typeof answer.stop_turn === "string") stop = answer.stop_turn;
    else if (turn.calls >= turn.callLimit) stop = "model_call_limit";
    else if (turn.deadlineMs !== null && performance.now() >= turn.deadlineMs) stop = "wall_time_limit";
    if (stop !== null) turn.outcome = ended(stop, turn.calls);
    return stop;
  }

  /**
   * Before a model call (Pi's `context` hook): the joined messages its context carries for the first
   * time are read by this call.
   */
  noteContext(messages: Json[]): void {
    for (const message of messages) {
      const id = message?.role === "custom" && message.customType === PLAY_MESSAGE ? message.details?.ora_message_id : null;
      if (typeof id === "string" && !this.firstRead.has(id)) this.firstRead.set(id, this.callSequence);
    }
  }

  /**
   * The `priority` of the model call about to be sent (plan play.request_priority), or null to send
   * none: `correction` when this call is the first to read a joined operator message (noteContext ran
   * in the call's `context` hook), else `default`.
   */
  requestPriority(): number | null {
    const priority = this.init.request_priority;
    if (!priority || !Number.isInteger(priority.correction) || !Number.isInteger(priority.default)) return null;
    const correcting = [...this.firstRead.values()].includes(this.callSequence);
    return correcting ? priority.correction : priority.default;
  }

  /** An operator message the Supervisor delivered (before the run, or steered into it). */
  deliver(payload: Record<string, Json>): void {
    const instruction = Number(this.core.playState?.instructions);
    const item: Delivered = {
      id: String(payload.id), message: String(payload.message), images: this.attachments.get(String(payload.id)) ?? [],
      joined: payload.joined === true,
      instructionId: typeof payload.instruction_id === "string" ? payload.instruction_id : null,
      instruction: Number.isInteger(instruction) && instruction > 0 ? instruction : null,
      step: Number.isInteger(payload.observation_step) ? Number(payload.observation_step) : null,
      observationNote: typeof payload.observation_note === "string" && payload.observation_note
        ? payload.observation_note : null,
    };
    this.attachments.delete(item.id);
    const session = this.session;
    if (this.turn !== null && session !== null && session.isStreaming) {
      // Mid-run (a steer): Pi delivers it after the current tool batch, before the next model call.
      if (item.joined) void session.sendCustomMessage(this.playMessage(item), { deliverAs: "steer" });
      else {
        const note = item.observationNote ? [{ type: "text", text: item.observationNote }] : [];
        void session.sendUserMessage(item.images.length || note.length
          ? [...note, { type: "text", text: item.message }, ...item.images] : item.message, { deliverAs: "steer" });
      }
      return;
    }
    this.delivered.push(item);
  }

  // -- Model calls (recorded from Pi's provider events) ---------------------------------

  providerRequest(payload: Json, model: Json): void {
    const turn = this.turn;
    const modelCallId = `play-call-${String(this.callSequence).padStart(6, "0")}`;
    this.callSequence += 1;
    if (turn !== null) {
      turn.calls += 1;
      turn.modelCallId = modelCallId;
    }
    this.core.append("provider_request", {
      visibility: "exact", observation_step: turn?.step ?? null, model_call_id: modelCallId,
      payload: { provider: model?.provider ?? null, model: model?.id ?? null, api: model?.api ?? null, play: true },
      artifact_ref: this.core.artifacts.putJson(payload ?? null, { kind: "play_provider_request" }),
    });
    this.pendingCall = modelCallId;
    this.perf.requested(modelCallId, turn?.step ?? null);
  }

  private pendingCall: string | null = null;

  /** The ORA model's request path with predictions, when the plan asks for them (runtime.ts installs it). */
  predictor: PipelinePredictor | null = null;

  assistantMessage(message: Json): void {
    if (message?.provider === "ora") this.predictor?.noteReply(message);
    const turn = this.turn;
    if (turn !== null) {
      turn.lastStopReason = String(message?.stopReason ?? "");
      turn.lastError = message?.errorMessage ? String(message.errorMessage) : null;
    }
    const modelCallId = this.pendingCall;
    this.pendingCall = null;
    this.perf.replied(message);
    if (modelCallId === null) {
      // A run stopped (or failed) before Pi sent anything: no request to answer.
      this.core.append("play_model_call_skipped", {
        visibility: "derived", observation_step: turn?.step ?? null,
        payload: { stop_reason: message?.stopReason ?? null, error: message?.errorMessage ?? null },
      });
      return;
    }
    const usage = message?.usage ?? {};
    this.core.append("provider_response", {
      visibility: "exact", observation_step: turn?.step ?? null, model_call_id: modelCallId,
      payload: {
        provider: message?.provider ?? null, model: message?.model ?? null, api: message?.api ?? null,
        stop_reason: message?.stopReason ?? null, error: message?.errorMessage ?? null,
        usage: { input_tokens: usage.input ?? 0, output_tokens: usage.output ?? 0,
          cached_input_tokens: usage.cacheRead ?? 0, total_tokens: usage.totalTokens ?? 0 },
        play: true,
      },
      artifact_ref: this.core.artifacts.putJson({ content: message?.content ?? [] }, { kind: "play_provider_response" }),
    });
  }

  // -- Robot tools (Python) ------------------------------------------------------------

  async robotTool(wire: string, args: Json, toolCallId: string,
    onUpdate?: (partial: Json) => void): Promise<{ content: Json[]; details: Json; terminate: boolean }> {
    const name = this.canonicalByWire.get(wire) ?? wire;
    const parent = this.codemodeCalls.get(toolCallId);
    if (parent !== undefined) return this.codemodeRobotCall(name, args, toolCallId, parent);
    const turn = this.turn;
    const core = this.core;
    const refuse = (code: string, message: string) => {
      const result = { ok: false, error: { code, message } };
      core.append("tool_result", {
        visibility: "exact", observation_step: turn?.step ?? null, model_call_id: turn?.modelCallId ?? null,
        tool_call_id: toolCallId, payload: { name, ok: false, play: true },
        artifact_ref: core.artifacts.putJson(result, { kind: "native_tool_result" }),
      });
      return { content: [{ type: "text", text: dumpsSorted(result) }], details: { ora_tool_call: toolCallId }, terminate: false };
    };
    const proposal = core.append("tool_call_proposed", {
      visibility: "exact", observation_step: turn?.step ?? null, model_call_id: turn?.modelCallId ?? null,
      tool_call_id: toolCallId, payload: { name, arguments: args ?? {}, play: true },
    });
    if (turn === null) return refuse("no_instruction_running", "robot tools act only while an instruction runs");
    if (turn.outcome !== null) {
      // An earlier call of this reply ended the act (its program executed actions, or the world changed under
      // it): the next act brings the new observation first.
      return refuse("turn_ended", "an earlier call in this reply ended the turn: wait for the next observation");
    }
    const [classified] = await core.rpc.request("tools.classify", { calls: [{ name, arguments: args ?? {} }] });
    if (classified?.error) {
      const error = classified.error;
      return refuse(String(error.code ?? "invalid_arguments"), String(error.message ?? "the call was refused"));
    }
    const mutation = core.config.tools.typed_effects
      ? classified?.effect === "mutation" : core.config.tools.mutation_tools.includes(name);
    if (mutation && turn.submitted !== null) {
      return refuse("one_action_per_observation",
        "one world-changing action per observation: wait for the result of the action already submitted");
    }
    const modelCallId = turn.modelCallId ?? `play-act-${turn.step}`;
    if (this.init.code_mode === true && name === "bash" && typeof onUpdate === "function") {
      this.liveScript = { parent: toolCallId, update: onUpdate, calls: [], images: [] };
    }
    let projected: Json;
    try {
      projected = await core.rpc.request("tools.invoke", {
        name, arguments: args ?? {}, effect_is_mutation: mutation, model_calls_in_turn: turn.calls,
        call_context: {
          policy_session_id: core.config.policy_session_id, model_call_id: modelCallId,
          tool_call_id: toolCallId, proposal_event_id: proposal.event_id, observation_step: turn.step,
          current_observation_event_id: turn.observationEvent.event_id, interaction_phase: "robot_rollout",
        },
      });
    } finally {
      this.liveScript = null;
    }
    if (projected.execution_delivery_guard === true) {
      this.pendingToolDelivery.push({ model_call_id: modelCallId, tool_call_id: toolCallId });
    }
    for (const record of projected.runtime_events ?? []) core.append(String(record.type), record.fields ?? {});
    // The Supervisor's ledger of what the call executed, checked against the receipts recorded meanwhile; the
    // result records it as an episode's does (core/session.ts recordToolResult).
    const execution = projected.execution ? core.acceptExecution(projected.execution, toolCallId, modelCallId) : null;
    const state = new StaticFactsState();
    const canonical = applyHints(projected.canonical, (projected.canonical_hints ?? []) as Hint[], state).delivered;
    const visible = applyHints(projected.visible, (projected.visible_hints ?? []) as Hint[], this.staticFacts).delivered;
    const ok = typeof visible.ok === "boolean" ? visible.ok
      : !("error" in visible) && !["error", "failed", "invalid", "rejected"].includes(visible.status);
    core.append("tool_result", {
      causal_parent_id: proposal.event_id, visibility: "exact", observation_step: turn.step,
      model_call_id: turn.modelCallId, tool_call_id: toolCallId,
      payload: { name, ok, play: true, ...(execution !== null ? { execution } : {}) },
      artifact_ref: core.artifacts.putJson(canonical, { kind: "native_tool_result" }),
    });
    const images: Json[] = projected.images ?? [];
    images.forEach((ref: Json, index: number) => {
      core.append("tool_image", {
        visibility: "exact", observation_step: turn.step, model_call_id: turn.modelCallId, tool_call_id: toolCallId,
        artifact_ref: ref, payload: { image_index: index },
      });
    });
    const content: Json[] = [{ type: "text", text: dumpsSorted(visible) }, ...PlayDriver.toolImageContent(
      images.map((ref: Json) => ({ type: "image_url", image_url: { url: core.dataUrl(ref) } })),
      projected.image_labels ?? null)];
    // The turn outcome (core/session.ts executeTool): actions the Supervisor executed inside the call (a
    // code-mode script's, a synchronous toolkit tool's) end the act, and the next act brings the new
    // observation; an execution that left the world unknown, or reached an environment boundary, ends it too.
    let terminate = false;
    const directive = projected.directive;
    if (directive === "world_state_unknown" || directive === "tool_execution_interrupted") {
      turn.outcome = ended(directive, turn.calls);
      terminate = true;
    } else if (directive === "execution_boundary") {
      turn.outcome = ended("tool_execution_boundary", turn.calls);
      terminate = true;
    } else if (directive === "actions_executed") {
      const executed = executionEvidence(projected.execution, toolCallId);
      if (executed.native_actions_used <= 0) throw new Error("executed directive requires positive consumption");
      turn.outcome = { submitted: false, actions_executed: true, execution: executed, attempts: turn.calls };
      terminate = true;
    } else if (mutation && projected.raw_ok !== false && directive === "action_submitted") {
      const action = projected.action;
      const actionEvent = core.append("action_proposed", {
        observation_step: turn.step, model_call_id: turn.modelCallId, tool_call_id: toolCallId,
        payload: { action_id: action.action_id, observation_id: action.observation_id }, artifact_ref: action.artifact_ref,
      });
      turn.submitted = { action_event_id: actionEvent.event_id, tool_name: name };
      terminate = true;  // the Supervisor executes it; the next act brings its result
      this.predictor?.start(toolCallId, wire, content);  // think about the next step while it runs
    }
    return { content, details: { ora_tool_call: toolCallId, ora_robot_tool: name }, terminate };
  }

  /**
   * A tool result's images as Pi content, each after its label when Python labelled them one to one
   * (tool_rpc.py `image_labels`: an observe result names each image's camera and the marks drawn on it).
   */
  static toolImageContent(images: Json[], labels: Json): Json[] {
    const named = Array.isArray(labels) && labels.length === images.length
      && labels.every((label) => typeof label === "string" && label);
    return images.flatMap((image, index) => [
      ...(named ? [{ type: "text", text: String(labels[index]) }] : []), ...piContent([image])]);
  }

  /**
   * A robot call from a codemode script runs as a script's call: through the run's robot socket,
   * which the Supervisor serves inside this turn (admission, execution, receipt and the new
   * observation, recorded with origin `codemode` and the codemode call that made it).
   */
  private async codemodeRobotCall(name: string, args: Json, toolCallId: string, parent: string) {
    const socket = process.env.ORA_ROBOT_SOCKET;
    const answer = socket
      ? await robotSocketCall(socket, "tools.call", { name, arguments: args ?? {} },
        { kind: "codemode", client: "codemode", parent_tool_call_id: parent, nested_call_id: toolCallId })
      : { ok: false, error: { code: "robot_socket_unavailable", message: "this run has no robot socket" } };
    return {
      content: [{ type: "text", text: dumpsSorted(answer) }],
      details: { ora_tool_call: toolCallId, ora_robot_tool: name, origin: "codemode" }, terminate: false,
    };
  }

  /**
   * An agent event Python recorded (the core's `record` RPC). A call the running code-mode command's program made
   * (origin `code_mode` under that command's call), its result or its image updates the command's partial result:
   * `{status: "running", calls, output: ""}` in the shape of its final one, with the latest images (a dry run's
   * plan picture among them), so the terminal and the page show each call as it happens.
   */
  noteRecord(type: string, fields: Json): void {
    const live = this.liveScript;
    const origin = fields?.payload?.origin;
    if (live === null || origin?.kind !== "code_mode" || origin.parent_tool_call_id !== live.parent) return;
    const seq = Number(origin.seq);
    if (type === "tool_call_proposed") {
      live.calls.push({ call: seq, tool: String(fields.payload.name ?? "?"), ok: true, running: true,
        ...(origin.dry_run === true ? { dry_run: true } : {}) });
    } else if (type === "tool_result") {
      const entry = live.calls.find((item) => item.call === seq);
      if (entry === undefined) return;
      entry.ok = fields.payload.ok !== false;
      delete entry.running;
    } else if (type === "tool_image" && fields.artifact_ref) {
      live.images.push(fields.artifact_ref);
      const entry = live.calls.find((item) => item.call === seq);
      if (entry !== undefined) entry.images = Number(entry.images ?? 0) + 1;
    } else {
      return;
    }
    const images = live.images.slice(-4).map((ref: Json) => ({ type: "image_url", image_url: { url: this.core.dataUrl(ref) } }));
    try {
      live.update({
        content: [{ type: "text", text: JSON.stringify({ status: "running", calls: live.calls, output: "" }) },
          ...piContent(images)],
        details: { ora_tool_call: live.parent, ora_partial: true },
      });
    } catch {
      // Display only: a partial the terminal cannot take changes nothing the command does.
    }
  }

  // -- Everything else the agent and the operator do -----------------------------------

  record(type: string, payload: Json, artifact: Json = undefined, toolCallId: string | null = null): void {
    this.core.append(type, {
      visibility: "exact", observation_step: this.turn?.step ?? null, model_call_id: this.turn?.modelCallId ?? null,
      ...(toolCallId === null ? {} : { tool_call_id: toolCallId }),
      payload: { ...payload, play: true },
      ...(artifact === undefined ? {} : { artifact_ref: this.core.artifacts.putJson(artifact, { kind: `play_${type}` }) }),
    });
  }

}
