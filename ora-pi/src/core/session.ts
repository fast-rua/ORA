// ORA's robot-agnostic agent core: one policy session, driven turn by turn.
//
// This is the TypeScript counterpart of the retired Python Native core's session
// and tool loop (ora/agent/native/{session,tool_loop}.py, now only in the
// robot-harness repository) minus everything that knows robots. The Pi agent loop sequences requests and
// tool calls (see pi_bridge.ts); this class owns the canonical ORA history, the
// per-turn budget, request/reply evidence and the dispatch protocol, and calls
// back into Python for tools and result projection.

import { appendFileSync, existsSync, readFileSync, truncateSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

import { canonicalBytesPy, clonePy, dumpsCompactSorted, dumpsPy, dumpsSorted, parsePy, PyFloat } from "../pyjson.ts";
import type { RpcPeer } from "../rpc.ts";
import { DeferredDurability } from "../evidence/durability.ts";
import { TrajectoryRecorder, type AgentEvent, type AppendOptions } from "../evidence/recorder.ts";
import { ArtifactStore } from "../evidence/artifacts.ts";
import { SessionManifestWriter } from "../evidence/manifest.ts";
import { applyHints, StaticFactsState, type Hint } from "./hints.ts";
import {
  admitToolBatch, batchRejection, isMutation, mutationProtocolError, normalizeArguments, parseToolCalls,
  RepeatedFailureTracker, toolErrorResult, ToolCallError, type BatchAdmission, type DomainCodes,
  type ParsedCall, type TerminalRefusal,
} from "./dispatch.ts";
import * as continuations from "./continuations.ts";
import { executionEvidence } from "./execution.ts";
import { crashPoint } from "./crash_points.ts";
import { ablateMemory, sampleMemory, timed, writeAllocationProfile } from "./profile.ts";

type Json = any;
export type OraMessage = Record<string, Json>;

/** What the context/provider modules must provide (implemented by context/ and provider/). */
export interface ContextDeps {
  pipeline: {
    project(args: { messages: OraMessage[]; tools: Json[]; observationStep: number }): Promise<{ messages: Json[]; tools: Json[]; manifest: Record<string, Json> }>;
    observeProviderUsage(args: { estimatedTokens: number; inputTokens: Json; projectionEpoch: Json; providerWireRequestSha256: Json }): void;
    compactorCalls(): Json[];
    snapshot?(): Json;
    restore?(state: Json): void;
    resumeScripts?(compactorResponses: number): void;
  };
  assembler: {
    prepare(projected: Json): { canonical: Json; wire: Json; execution: Json };
    noteArtifacts?(canonicalRef: Json, wireRef: Json): void;
  };
  annotateContextSources(projected: Json, args: { composition: string | null; observation: OraMessage; toolSchemas: Json[]; contextSources: Json[] }): void;
  provider: {
    protocol: string;
    complete(execution: Json, options: {
      modelCallId: string; deadlineMonotonicMs: number | null; signal?: AbortSignal;
      onAttemptFailure: (record: Record<string, Json>) => void;
    }): Promise<Record<string, Json>>;
    close(): void;
  };
  /** Closes a bound compactor's own provider (policy.context_compactor). */
  closeCompactor?(): void;
}

export interface InitParams {
  /** A free-play run (plan episode_mode "play"): Pi's own agent, see play/driver.ts. */
  play?: { workspace: string; auth_path: string; agent_dir: string | null } | null;
  /** Where /login keeps provider credentials (user-level ORA config, never the run). */
  credentials?: { auth_path: string } | null;
  run_root: string;
  run_id: string;
  policy_session_id: string;
  event_schema: string;
  session_started_payload: Record<string, Json>;
  policy: Record<string, Json>;
  provider: Record<string, Json>;
  /** A compactor bound to its own profile (policy.context_compactor); its key is ORA_PI_COMPACTOR_API_KEY. */
  compactor_provider?: Record<string, Json>;
  tools: {
    definitions: Json[]; delivered: Json[]; mutation_tools: string[]; lifecycle_tools: string[];
    capabilities: string[]; typed_effects: boolean; null_as_omitted: boolean;
  };
  domain: {
    argument_refusal_codes: string[]; infrastructure_codes: string[]; turn_instruction: string;
    composition_context_capabilities: string[]; dotted_tool_names: Record<string, string>;
    /** Code mode: each script tool's `episode_terminal` refusal, naming the direct tools left (empty otherwise). */
    terminal_refusals?: Record<string, TerminalRefusal>;
    /** Each context stage's parameter schema (ora/policy/context_stages); the context stages validate with them. */
    context_stage_schemas?: Record<string, Json>;
  };
}

export interface ActParams {
  static: { system_text: string; task_sha256: string | null; messages: OraMessage[]; sources: Json[] } | null;
  step: number;
  draft: { message: OraMessage };
  receipt_event_id: string | null;
  last_fresh_observation_event_id: string | null;
  remaining_budget: Record<string, Json>;
  terminal_lifecycle: boolean;
  allow_reminder?: boolean;
  operator_poll?: boolean;
}

export interface Outcome {
  submitted: boolean;
  actions_executed?: boolean;
  execution?: Json;
  termination?: string;
  attempts: number;
  action_event_id?: string;
  tool_name?: string | null;
  /** Free play: the joined operator messages this act's text-only last reply was the first to read. */
  answers?: string[];
}

interface ReplyState {
  modelCallId: string;
  assistant: OraMessage;
  responseEvent: AgentEvent;
  usage: Json;
  finishReason: Json;
  outputRejection: Record<string, Json> | null;
  calls: Json[];
}

interface BatchState {
  parsed: ParsedCall[];
  admission: BatchAdmission;
  byCallId: Map<string, number>;
  done: number;
  toolImages: OraMessage[];
  /** The runtime aggregates this same-tool read-only batch into one feedback image. */
  aggregated: boolean;
  /** Python's projected results of an aggregated batch, in call order, once it ran. */
  aggregatedResults: Json[] | null;
}

interface Turn {
  step: number;
  observation: OraMessage;
  observationEvent: AgentEvent;
  requestDeadlineMs: number | null;
  callLimit: number;
  allowReminder: boolean;
  terminalLifecycle: boolean;
  operatorPoll: boolean;
  attempts: number;
  outcome: Outcome | null;
  fatal: Error | null;
  pending: { projected: Json; modelCallId: string; retry?: LostRequest; replay?: ReplayedReply } | { stop: true } | null;
  /** A request the previous agent process sent but never recorded an answer for. */
  lostRequest: LostRequest | null;
  /** A reply the previous agent process recorded but whose tools it may not have finished. */
  replay: ReplayedReply | null;
  reply: ReplyState | null;
  batch: BatchState | null;
}

/**
 * The previous agent process recorded this reply and died before its tool
 * batch reached the persisted history. The resumed turn rebuilds the reply from
 * its provider_response instead of asking the model again, and runs the batch
 * once more: every event already recorded for the reply is matched and reused
 * (the replay must reproduce it field for field), and Python answers a tool call
 * it already executed from its per-act cache, so each call runs exactly once.
 */
interface ReplayedReply {
  modelCallId: string;
  requestEvent: AgentEvent;
  responseEvent: AgentEvent;
}

/** Event types the core records for a reply's batch, which a replay reproduces. */
const REPLAYED_EVENT_TYPES = new Set([
  "tool_call_proposed", "tool_result", "tool_image", "action_proposed",
  "provider_output_rejected", "provider_output_truncated", "provider_no_action_continuation",
]);

/** A batch event a replay reproduces: the core's own, or one a tool recorded for its call. */
function replayedType(type: string, toolCallId: Json): boolean {
  return REPLAYED_EVENT_TYPES.has(type) || typeof toolCallId === "string";
}

function replayKey(type: string, modelCallId: Json, toolCallId: Json, payload: Json): string {
  const image = type === "tool_image" && payload && typeof payload === "object" ? String(payload.image_index) : "";
  const action = ["action_proposed", "environment_receipt"].includes(type)
    && payload && typeof payload === "object" ? String(payload.action_id ?? payload.receipt_id ?? "") : "";
  const publication = type === "fresh_observation_published" && payload && typeof payload === "object"
    ? String(payload.observation_id ?? "") : "";
  return `${type}|${modelCallId ?? ""}|${toolCallId ?? ""}|${image}|${action}|${publication}`;
}

/** The fields of an event the core decides (not its position, hash or clock). */
function decidedFields(type: string, fields: AppendOptions): Json {
  return {
    type, observation_step: fields.observation_step ?? null, model_call_id: fields.model_call_id ?? null,
    tool_call_id: fields.tool_call_id ?? null, causal_parent_id: fields.causal_parent_id ?? null,
    visibility: fields.visibility ?? "exact", payload: fields.payload === undefined ? null : fields.payload,
    artifact_ref: fields.artifact_ref ?? null,
  };
}

interface LostRequest {
  modelCallId: string;
  requestEvent: AgentEvent;
  requestRef: Json;
  wireRef: Json;
  failedAttempts: number;
}

const CAUSAL_SLOTS = new Set(["native_slots_v3", "native_slots_v4"]);
const LOSSLESS = "action_step_lossless_v4";

/**
 * The session manifest's call records, one JSON line per call, beside the Pi
 * session. Each record carries its context projection, which grows with the
 * transcript, and Pi keeps every session entry in memory: inside ora.state they
 * made the process grow quadratically. ora.state holds only how many records
 * (and bytes) are persisted.
 */
const CALLS_FILE = "native/pi-calls.jsonl";

/** How an operator instruction of a free-play session reads. */
export const PLAY_INSTRUCTION_GUIDANCE = "An operator instruction in a free-play session, with no benchmark task. "
  + "Carry it out from current evidence with the advertised tools. When it is done, or cannot be done, reply "
  + "with a short report of what you did and observed and call no tool; the operator then gives the next "
  + "instruction. A message the operator sends while you work joins the current instruction: answer a "
  + "question and keep working, or take a changed request as the new objective.";

function ended(termination: string, attempts: number): Outcome {
  return { submitted: false, termination, attempts };
}

function sortedKeysEqual(left: Json, right: Json): boolean {
  return dumpsSorted(left) === dumpsSorted(right);
}

/** Hooks the Pi bridge installs so history changes reach the Pi session durably. */
export interface HistorySink {
  appended(messages: OraMessage[]): void;
  state(snapshot: Record<string, Json>): void;
}

export class OraCore {
  readonly config: InitParams;
  readonly runRoot: string;
  readonly durability = new DeferredDurability();
  readonly recorder: TrajectoryRecorder;
  readonly artifacts: ArtifactStore;
  readonly manifest: SessionManifestWriter;
  deps!: ContextDeps;
  readonly codes: DomainCodes;
  readonly history: OraMessage[] = [];
  readonly contextSources: Json[] = [];
  readonly calls: Json[] = [];
  readonly usageEvents: Json[] = [];
  readonly reportedTotals: Record<string, number> = {
    input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, reasoning_output_tokens: 0, total_tokens: 0,
  };
  readonly toolSchemas: Json[];
  readonly staticFacts = new StaticFactsState();
  readonly repeatedFailures = new RepeatedFailureTracker();
  callSequence = 0;
  lastEventId: string | null = null;
  systemText: string | null = null;
  taskSha256: string | null = null;
  bound = false;
  closed = false;
  interrupted = false;
  turn: Turn | null = null;
  historySink: HistorySink | null = null;
  /** A free-play session's state as the Supervisor announces it (display only; null otherwise). */
  playState: Json | null = null;
  /** Instructions typed here, by command id (display only: the Supervisor delivers what the model reads). */
  readonly playInstructions = new Map<string, string>();
  /** Free play delivers operator messages into Pi's own conversation (play/driver.ts) instead of ORA's history. */
  operatorDelivery: ((payload: Record<string, Json>) => void) | null = null;
  readonly composition: string | null;
  readonly retention: string;

  readonly rpc: RpcPeer;

  constructor(config: InitParams, rpc: RpcPeer, options: { resume?: boolean } = {}) {
    this.rpc = rpc;
    this.config = config;
    this.runRoot = config.run_root;
    this.composition = config.policy.context_composition ?? null;
    this.retention = config.policy.context_retention ?? "rewrite_compact_v1";
    this.recorder = new TrajectoryRecorder(
      this.runRoot, config.run_id, config.policy_session_id, this.durability, config.event_schema,
      { resume: options.resume },
    );
    this.artifacts = new ArtifactStore(this.runRoot, { durability: this.durability });
    this.manifest = new SessionManifestWriter(join(this.runRoot, "native/agent-session.json"), this.durability);
    this.codes = {
      argumentRefusalCodes: new Set(config.domain.argument_refusal_codes),
      infrastructureCodes: new Set(config.domain.infrastructure_codes),
    };
    // NativeToolGateway.definitions(): the wire tool list Python resolved.
    this.toolSchemas = config.tools.definitions;
    if (options.resume) {
      this.lastEventId = null;
    }
  }

  // -- Evidence -------------------------------------------------------------------

  append(type: string, options: AppendOptions = {}): AgentEvent {
    this.resentCandidate = null;
    const fields = { ...options };
    if (!Object.prototype.hasOwnProperty.call(fields, "causal_parent_id")) fields.causal_parent_id = this.lastEventId;
    const recorded = this.replayedEvent(type, fields);
    if (recorded !== null) {
      this.lastEventId = recorded.event_id;
      return recorded;
    }
    const event = this.recorder.append(type, fields);
    this.lastEventId = event.event_id;
    if (["action_proposed", "environment_receipt", "fresh_observation_published"].includes(type)) {
      this.executionEvents.set(event.event_id, event);
    }
    return event;
  }

  /** Recorded batch events of a replayed reply that the replay has not reproduced yet. */
  private replayLedger: Map<string, AgentEvent> | null = null;
  private executionEvents = new Map<string, AgentEvent>();

  /** The recorded event this append reproduces, or null to record a new one. */
  private replayedEvent(type: string, fields: AppendOptions): AgentEvent | null {
    const ledger = this.replayLedger;
    if (ledger === null || !replayedType(type, fields.tool_call_id)) return null;
    const key = replayKey(type, fields.model_call_id, fields.tool_call_id, fields.payload);
    const recorded = ledger.get(key);
    if (recorded === undefined) return null;
    const expected = canonicalBytesPy(decidedFields(type, fields));
    const actual = canonicalBytesPy(decidedFields(recorded.type, recorded as AppendOptions));
    if (!expected.equals(actual)) {
      throw new Error(`the resumed reply diverged from its recorded ${type} (${recorded.event_id}): `
        + `${expected.toString("utf8").slice(0, 300)} != ${actual.toString("utf8").slice(0, 300)}`);
    }
    ledger.delete(key);
    return recorded;
  }

  /** The replayed reply's batch is over: everything recorded for it must have been reproduced. */
  private closeReplay(): void {
    const ledger = this.replayLedger;
    this.replayLedger = null;
    if (ledger !== null && ledger.size) {
      throw new Error(`the resumed reply did not reproduce its recorded events: ${[...ledger.values()].map((event) => event.event_id).join(", ")}`);
    }
  }

  start(): void {
    this.append("session_started", { payload: this.config.session_started_payload });
    this.writeManifest(null);
  }

  writeManifest(exitReason: string | null): void {
    const provider = this.deps?.provider;
    const protocol = provider?.protocol ?? "canonical_unknown";
    const sortedSources = this.contextSources.slice().sort((a, b) => Number(a.order) - Number(b.order));
    const body = {
      schema: "ora.native-policy-session.v1",
      backend: `native_${protocol}_v1`,
      agent_core: "pi",
      provider_protocol: protocol,
      session_id: this.config.policy_session_id,
      model: this.config.policy.model,
      reasoning_effort: this.config.policy.reasoning_effort,
      initial_observation: this.config.policy.initial_observation ?? "automatic",
      context_composition: this.composition,
      context_retention: this.retention,
      ...(this.config.policy.context_token_estimator !== undefined
        ? { context_token_estimator: this.config.policy.context_token_estimator } : {}),
      context_management: this.config.policy.context_management ?? null,
      max_inline_images_per_request: this.config.policy.max_inline_images_per_request ?? null,
      automatic_observation_image: Boolean(this.config.policy.supports_images)
        && (this.config.policy.initial_observation ?? "automatic") === "automatic",
      model_context_window: this.config.policy.context_window_tokens,
      context_sources: sortedSources,
      calls: this.calls,
      context_compactor_calls: this.deps ? clonePy(this.deps.pipeline.compactorCalls()) : [],
      usage_events: this.usageEvents,
      reported_totals: this.reportedTotals,
      process_pid: process.pid,
      process_invocations: 1,
      process_restarts: 0,
      thread_id: null,
      exit_reason: exitReason,
    };
    this.manifest.write(body, exitReason !== null);
  }

  private pushHistory(...messages: OraMessage[]): void {
    this.history.push(...messages);
  }

  /** Messages already written to the Pi session (ora.history entries). */
  private pendingToolDelivery: { model_call_id: string; tool_call_id: string }[] = [];

  persistedCount = 0;
  private persistedCalls = 0;
  private persistedCallBytes = 0;
  private persistedUsage = 0;

  /**
   * Write new history and the current state into the Pi session. Called at
   * points where every message is final (its event ids are bound): before a
   * request, at the end of a turn and after runtime events.
   */
  flush(): void {
    const sink = this.historySink;
    if (sink === null) return;
    if (this.history.length > this.persistedCount) {
      sink.appended(this.history.slice(this.persistedCount));
      this.persistedCount = this.history.length;
    }
    if (this.calls.length > this.persistedCalls) {
      const lines = this.calls.slice(this.persistedCalls).map((call) => dumpsPy(call, { ensureAscii: false }) + "\n").join("");
      appendFileSync(join(this.runRoot, CALLS_FILE), lines, "utf8");
      this.persistedCallBytes += Buffer.byteLength(lines, "utf8");
    }
    sink.state({
      core: {
        call_sequence: this.callSequence, bound: this.bound, system_text: this.systemText,
        task_sha256: this.taskSha256, context_sources: this.contextSources,
        reported_totals: this.reportedTotals, history_length: this.history.length,
        static_facts: { scope: this.staticFacts.scope, fields: this.staticFacts.fields },
        repeated_failures: this.repeatedFailures.snapshot(),
        last_turn_outcome: this.lastTurnOutcome,
      },
      calls: { records: this.calls.length, bytes: this.persistedCallBytes },
      new_usage_events: this.usageEvents.slice(this.persistedUsage),
      context: this.deps?.pipeline.snapshot?.() ?? null,
    });
    this.persistedCalls = this.calls.length;
    this.persistedUsage = this.usageEvents.length;
  }

  // -- Observations -------------------------------------------------------------------

  /** Resolve a Python observation draft: hints applied, image refs as data URLs. */
  resolveDraft(draft: { message: OraMessage }): { message: OraMessage; slot: OraMessage | null } {
    const message = clonePy(draft.message);
    let slot: OraMessage | null = null;
    if (Array.isArray(message.content)) {
      const blocks: Json[] = [];
      for (const block of message.content) {
        if (block?.type === "ora_json") {
          const result = applyHints(block.value, (block.hints ?? []) as Hint[], this.staticFacts);
          slot = result.slot;
          blocks.push({ type: "text", text: dumpsSorted(result.delivered) });
        } else if (block?.type === "image_ref") {
          blocks.push({ type: "image_url", image_url: { url: this.dataUrl(block.ref) } });
        } else {
          blocks.push(block);
        }
      }
      message.content = blocks;
    }
    return { message, slot };
  }

  dataUrl(ref: Record<string, Json>): string {
    const bytes = readFileSync(join(this.runRoot, String(ref.path)));
    return `data:${ref.media_type};base64,${bytes.toString("base64")}`;
  }

  bindStatic(staticContext: ActParams["static"]): void {
    if (staticContext === null) {
      if (!this.bound) throw new Error("the first turn must carry the static context");
      return;
    }
    if (this.bound) throw new Error("static context bound twice");
    this.systemText = staticContext.system_text;
    this.taskSha256 = staticContext.task_sha256;
    this.pushHistory(...clonePy(staticContext.messages));
    this.contextSources.push(...clonePy(staticContext.sources));
    this.bound = true;
  }

  /** `_open_turn` + `_append_observation_event` + `_append_round_static_facts`. */
  openTurn(params: ActParams): Turn {
    if (this.closed || this.interrupted) throw new Error("native policy session is not active");
    const wallTime = params.remaining_budget?.wall_time_s;
    const wall = wallTime instanceof PyFloat ? wallTime.value : wallTime;
    if (wall !== undefined && wall !== null && (typeof wall !== "number" || !Number.isFinite(wall) || wall < 0)) {
      throw new Error("remaining provider wall-time budget is invalid");
    }
    const requestDeadlineMs = wall === undefined || wall === null ? null : performance.now() + wall * 1000;
    this.bindStatic(params.static);
    const { message: observation, slot } = this.resolveDraft(params.draft);
    this.pushHistory(observation);
    const meta = observation._observation;
    const payload: Record<string, Json> = { observation_id: meta.observation_id, state_ref: meta.state_ref };
    void timed;
    if (this.composition !== null && CAUSAL_SLOTS.has(this.composition)) {
      payload.causal_pair = Boolean(observation._causal_observation?.paired);
      payload.causal_cameras = [...(observation._causal_observation?.cameras ?? [])];
    }
    const observationEvent = this.append("observation_received", { observation_step: params.step, payload });
    const sourceIds = [observationEvent.event_id];
    if (typeof params.receipt_event_id === "string") sourceIds.push(params.receipt_event_id);
    if (params.last_fresh_observation_event_id !== null) sourceIds.push(params.last_fresh_observation_event_id);
    observation._source_event_ids = sourceIds;
    if (slot !== null) {
      slot._source_event_ids = [observationEvent.event_id];
      this.pushHistory(slot);
    }
    const maxCalls = Number(this.config.policy.max_model_calls_per_observation ?? 12);
    const remainingCalls = params.remaining_budget?.model_calls;
    const callLimit = Math.min(
      maxCalls,
      typeof remainingCalls === "number" && Number.isInteger(remainingCalls) ? remainingCalls : maxCalls,
    );
    const turn: Turn = {
      step: params.step, observation, observationEvent, requestDeadlineMs, callLimit,
      allowReminder: params.allow_reminder ?? true, terminalLifecycle: params.terminal_lifecycle,
      operatorPoll: Boolean(params.operator_poll), attempts: 0, outcome: null, fatal: null,
      pending: null, reply: null, batch: null, lostRequest: null, replay: null,
    };
    this.turn = turn;
    if (turn.callLimit <= 0) turn.outcome = ended("model_call_limit", 0);
    return turn;
  }

  // -- One provider exchange ---------------------------------------------------------

  /** Before each request: operator poll, projection, budget. Runs in Pi's `context` hook. */
  async prepareRequest(): Promise<Json[] | null> {
    const turn = this.requireTurn();
    turn.pending = null;
    turn.reply = null;
    turn.batch = null;
    this.flush();
    if (turn.outcome !== null || turn.fatal !== null) {
      turn.pending = { stop: true };
      return null;
    }
    // Ended parents are not retroactively interrupted by a later provider failure.
    // Flush above persisted the entire reply, including its final tool images.
    if (this.pendingToolDelivery?.length) {
      await this.rpc.request("tools.settled", { calls: this.pendingToolDelivery });
      this.pendingToolDelivery = [];
    }
    // A resumed request (lost or replayed) was admitted and polled for before the restart.
    const resumed = turn.lostRequest !== null || turn.replay !== null;
    if (turn.attempts >= turn.callLimit && !resumed) {
      turn.outcome = ended("model_call_limit", turn.attempts);
      turn.pending = { stop: true };
      return null;
    }
    if (this.interrupted) return this.stopInterrupted(turn);
    if (turn.operatorPoll && !resumed) {
      const answer = await this.rpc.request("operator.poll", {
        observation_step: turn.step,
        before_model_call_id: `model-call-${String(this.callSequence).padStart(6, "0")}`,
      });
      // A free-play session's operator stopped this instruction (or reset or quit):
      // the turn ends here, before another model call; the session stays open.
      if (answer && typeof answer === "object" && typeof answer.stop_turn === "string") {
        turn.outcome = ended(answer.stop_turn, turn.attempts);
        turn.pending = { stop: true };
        return null;
      }
    }
    if (this.interrupted) return this.stopInterrupted(turn);
    let projected: Json;
    try {
      projected = await timed("core.project", () => this.deps.pipeline.project({ messages: this.history, tools: this.toolSchemas, observationStep: turn.step }));
    } catch (error) {
      if (!String((error as Error).message).startsWith("context_exhausted")) {
        turn.fatal = error as Error;
        turn.pending = { stop: true };
        return null;
      }
      this.append("context_exhausted", { visibility: "derived", observation_step: turn.step, payload: { error: String((error as Error).message) } });
      turn.outcome = ended("context_exhausted", turn.attempts);
      turn.pending = { stop: true };
      return null;
    }
    if (this.interrupted) return this.stopInterrupted(turn);
    if (turn.lostRequest !== null) {
      // Re-sent under its own model call id: the projection is deterministic
      // from the restored state, and exchange() checks the bytes match.
      const retry = turn.lostRequest;
      turn.lostRequest = null;
      turn.pending = { projected, modelCallId: retry.modelCallId, retry };
      return projected.messages;
    }
    if (turn.replay !== null) {
      const replay = turn.replay;
      turn.replay = null;
      turn.pending = { projected, modelCallId: replay.modelCallId, replay };
      return projected.messages;
    }
    turn.attempts += 1;
    const modelCallId = `model-call-${String(this.callSequence).padStart(6, "0")}`;
    this.callSequence += 1;
    try {
      this.deps.annotateContextSources(projected, {
        composition: this.composition, observation: turn.observation, toolSchemas: this.toolSchemas,
        contextSources: this.contextSources,
      });
    } catch (error) {
      if (!String((error as Error).message).startsWith("context_exhausted")) throw error;
      this.append("context_exhausted", {
        visibility: "derived", observation_step: turn.step, model_call_id: modelCallId,
        payload: { error: String((error as Error).message) },
      });
      turn.outcome = ended("context_exhausted", turn.attempts - 1);
      turn.pending = { stop: true };
      return null;
    }
    turn.pending = { projected, modelCallId };
    return projected.messages;
  }

  /**
   * The previous agent process sent this request and died before recording
   * its answer. The lost attempt is evidenced as a failed transport attempt
   * (ora.provider-attempt-failure.v1, retry scheduled) and the identical
   * request is sent again under the same model call id, which is exactly
   * what ORA's same-credential recovery records for a dropped connection.
   */
  private async retryLostRequest(turn: Turn, prepared: Json, projected: Json, retry: LostRequest, wireRef: Json,
                                 signal?: AbortSignal): Promise<ReplyState | null> {
    if (wireRef.sha256 !== retry.wireRef.sha256) {
      throw new Error(`resumed request differs from the lost one (${wireRef.sha256} != ${retry.wireRef.sha256})`);
    }
    const nowS = (performance.timeOrigin + performance.now()) / 1000;
    const startedS = Number(retry.requestEvent.unix_s);
    const failure: Record<string, Json> = {
      schema: "ora.provider-attempt-failure.v1",
      model_call_id: retry.modelCallId,
      attempt: retry.failedAttempts + 1,
      request_sha256: retry.wireRef.sha256,
      credential_index: 0,
      next_credential_index: 0,
      http_status: null,
      phase: "transport",
      started_unix_s: startedS,
      finished_unix_s: nowS,
      duration_s: Math.round((nowS - startedS) * 1e6) / 1e6,
      retry_scheduled: true,
      retry_delay_s: 0,
      partial_event_count: 0,
      tool_execution_started: false,
      error_chain: [{ type: "AgentCoreRestarted", message: "the agent process ended while this request was in flight" }],
      partial_events: [],
      credentials_redacted: true,
      recovery_policy: "agent_core_restart_v1",
    };
    const artifact = this.artifacts.putJson(failure, { kind: "native_provider_failed_attempt" });
    const payload: Record<string, Json> = {};
    for (const [key, value] of Object.entries(failure)) if (key !== "partial_events" && key !== "error_chain") payload[key] = value;
    this.append("provider_attempt_failed", {
      causal_parent_id: retry.requestEvent.event_id, observation_step: turn.step, model_call_id: retry.modelCallId,
      visibility: "derived", artifact_ref: artifact, payload,
    });
    let response: Record<string, Json>;
    try {
      response = await this.deps.provider.complete(prepared.execution, {
        modelCallId: retry.modelCallId, deadlineMonotonicMs: turn.requestDeadlineMs, signal,
        onAttemptFailure: () => {
          throw new Error("a resumed request is sent once");
        },
      });
    } catch (error) {
      const stopped = this.interrupted;
      this.append("provider_error", {
        causal_parent_id: retry.requestEvent.event_id, observation_step: turn.step, model_call_id: retry.modelCallId,
        payload: {
          exception_type: (error as Error)?.name ?? "Error", cause_type: null, attempts: retry.failedAttempts + 2,
          response_started: null, accepted: false, ...(stopped ? { interrupted: true } : {}),
        },
      });
      if (stopped) {
        turn.outcome = ended("operator_interrupt", turn.attempts);
        return null;
      }
      turn.fatal = error as Error;
      return null;
    }
    response = { ...response, attempts: retry.failedAttempts + 1 + Number(response.attempts ?? 1) };
    const reply = this.recordReply(turn, prepared, response, retry.modelCallId, retry.requestEvent);
    this.accountCall(projected, prepared, response, reply);
    turn.reply = reply;
    return reply;
  }

  /**
   * The previous agent process recorded this reply. Its request is projected
   * again from the restored state (it must reproduce the recorded wire bytes),
   * and the reply is rebuilt from the recorded provider_response, so the model
   * is not asked twice and the trace gains no second request or response.
   */
  private replayRecordedReply(turn: Turn, prepared: Json, projected: Json, replay: ReplayedReply, wireRef: Json): ReplyState {
    const recordedWire = replay.requestEvent.artifact_ref?.sha256;
    if (wireRef.sha256 !== recordedWire) {
      throw new Error(`resumed request differs from the recorded one (${wireRef.sha256} != ${recordedWire})`);
    }
    const payload = replay.responseEvent.payload ?? {};
    const response: Record<string, Json> = {
      message: clonePy(payload.message), usage: payload.usage ?? null, finish_reason: payload.finish_reason ?? null,
      attempts: payload.attempts ?? 1, output_rejection: payload.output_rejection ?? null,
      // The provider's own clock readings were not traced; the request and response events bound them.
      started_unix_s: replay.requestEvent.unix_s, finished_unix_s: replay.responseEvent.unix_s,
    };
    const reply = this.recordReply(turn, prepared, response, replay.modelCallId, replay.requestEvent, replay.responseEvent);
    this.accountCall(projected, prepared, response, reply);
    turn.reply = reply;
    return reply;
  }

  // -- Resume after an agent-process restart ------------------------------------------

  /** Rebuild from the Pi session's ora.* entries and the trace (the recorder reopened it). */
  restore(history: OraMessage[], states: Record<string, Json>[], events: Json[]): void {
    this.history.length = 0;
    this.history.push(...history);
    this.persistedCount = history.length;
    const latest = states[states.length - 1];
    if (latest) {
      const core = latest.core;
      this.callSequence = core.call_sequence;
      this.bound = core.bound;
      this.systemText = core.system_text;
      this.taskSha256 = core.task_sha256;
      this.contextSources.length = 0;
      this.contextSources.push(...clonePy(core.context_sources));
      Object.assign(this.reportedTotals, core.reported_totals);
      this.staticFacts.scope = core.static_facts.scope;
      this.staticFacts.fields = core.static_facts.fields;
      this.repeatedFailures.restore(core.repeated_failures);
      this.lastTurnOutcome = core.last_turn_outcome ?? null;
      if (latest.context) this.deps.pipeline.restore?.(latest.context);
    }
    for (const state of states) this.usageEvents.push(...clonePy(state.new_usage_events ?? []));
    this.restoreCalls(latest?.calls ?? { records: 0, bytes: 0 });
    this.persistedCalls = this.calls.length;
    this.persistedUsage = this.usageEvents.length;
    // A request recorded after the last flush still counts.
    const requested = events.filter((event) => event.type === "provider_request").length;
    this.callSequence = Math.max(this.callSequence, requested);
    // Assistant source ids gained proposals and actions after the message was flushed.
    const byId = new Map(events.map((event) => [event.event_id, event]));
    for (const message of this.history) {
      if (message.role !== "assistant" || !Array.isArray(message._source_event_ids)) continue;
      const response = byId.get(message._source_event_ids[0]);
      if (!response) continue;
      message._source_event_ids = [response.event_id, ...events
        .filter((event) => event.model_call_id === response.model_call_id
          && (event.type === "tool_call_proposed" || event.type === "action_proposed"))
        .map((event) => event.event_id)];
    }
    this.lastEventId = events.length ? events[events.length - 1].event_id : null;
    this.resentCandidate = events.length
      ? { event: events[events.length - 1], parent: events.length > 1 ? events[events.length - 2].event_id : null }
      : null;
    this.resumeEvents = events;
    for (const event of events) {
      if (["action_proposed", "environment_receipt", "fresh_observation_published"].includes(event.type)) {
        this.executionEvents.set(event.event_id, event);
      }
    }
    // Scripted providers (tests, replays) continue where the lost process stopped.
    const scripted = this.deps.provider as Json;
    if (typeof scripted.index === "number") {
      scripted.index = events.filter((event) => event.type === "provider_response").length;
    }
    this.deps.pipeline.resumeScripts?.(events.filter((event) => event.type === "context_compactor_response").length);
  }

  resumeEvents: Json[] = [];

  /** The persisted call records; lines written after the last ora.state are cut off. */
  private restoreCalls(persisted: { records: number; bytes: number }): void {
    const path = join(this.runRoot, CALLS_FILE);
    const data = existsSync(path) ? readFileSync(path) : Buffer.alloc(0);
    if (data.length < persisted.bytes) throw new Error(`${CALLS_FILE} lost call records the session persisted`);
    if (data.length > persisted.bytes) truncateSync(path, persisted.bytes);
    const lines = data.subarray(0, persisted.bytes).toString("utf8").split("\n").filter((line) => line);
    if (lines.length !== persisted.records) throw new Error(`${CALLS_FILE} holds ${lines.length} call records, the session ${persisted.records}`);
    this.calls.push(...lines.map((line) => parsePy(line)));
    this.persistedCallBytes = persisted.bytes;
  }

  /** The open turn of a resumed act (same params the Supervisor sent before the restart). */
  resumeTurn(params: ActParams): Turn | null {
    const events = this.resumeEvents;
    let observationIndex = -1;
    for (let i = events.length - 1; i >= 0; i -= 1) {
      if (events[i].type === "observation_received" && events[i].observation_step === params.step) {
        observationIndex = i;
        break;
      }
    }
    if (observationIndex < 0) return null;  // the restart happened before this turn opened
    const observationEvent = events[observationIndex];
    const after = events.slice(observationIndex + 1);
    const finished = this.lastTurnOutcome !== null && this.lastTurnOutcome.step === params.step
      ? clonePy(this.lastTurnOutcome.outcome) : null;
    const observation = [...this.history].reverse().find(
      (message) => message._observation?.step === params.step && Array.isArray(message._source_event_ids)
        && message._source_event_ids[0] === observationEvent.event_id);
    if (!observation) throw new Error("the resumed turn's observation is not in the persisted history");
    const requests = after.filter((event) => event.type === "provider_request");
    const answered = new Set(after.filter((event) => event.type === "provider_response" || event.type === "provider_error")
      .map((event) => event.model_call_id));
    const last = requests[requests.length - 1];
    const lastResponse = [...after].reverse().find((event) => event.type === "provider_response");
    const replay = finished === null && lastResponse && !this.replyPersisted(lastResponse, after) ? lastResponse : null;
    const actions = after.filter((event) => event.type === "action_proposed");
    if (finished === null && actions.some((event) => (replay === null || event.model_call_id !== replay.model_call_id)
      && !this.settledZeroSpendAction(event, after))) {
      throw new Error("the resumed turn submitted an action outside the reply being resumed");
    }
    const wall = params.remaining_budget?.wall_time_s;
    const wallValue = wall instanceof PyFloat ? wall.value : wall;
    const maxCalls = Number(this.config.policy.max_model_calls_per_observation ?? 12);
    const remainingCalls = params.remaining_budget?.model_calls;
    const turn: Turn = {
      step: params.step, observation, observationEvent,
      requestDeadlineMs: typeof wallValue === "number" ? performance.now() + wallValue * 1000 : null,
      callLimit: Math.min(maxCalls, typeof remainingCalls === "number" && Number.isInteger(remainingCalls) ? remainingCalls : maxCalls),
      allowReminder: params.allow_reminder ?? true, terminalLifecycle: params.terminal_lifecycle,
      operatorPoll: Boolean(params.operator_poll), attempts: requests.length, outcome: null, fatal: null,
      pending: null, reply: null, batch: null, lostRequest: null, replay: null,
    };
    if (replay !== null) {
      // Its tools may have run; their results never reached the persisted history.
      const requestEvent = requests.find((event) => event.model_call_id === replay.model_call_id);
      if (!requestEvent) throw new Error("the resumed reply has no recorded request");
      turn.replay = { modelCallId: replay.model_call_id, requestEvent, responseEvent: replay };
      this.replayLedger = new Map();
      for (const event of after) {
        if (event.model_call_id !== replay.model_call_id || !replayedType(event.type, event.tool_call_id)) continue;
        const key = replayKey(event.type, event.model_call_id, event.tool_call_id, event.payload);
        if (this.replayLedger.has(key)) throw new Error(`the resumed reply recorded ${key} twice`);
        this.replayLedger.set(key, event);
      }
    } else if (finished === null && last && !answered.has(last.model_call_id)) {
      turn.lostRequest = {
        modelCallId: last.model_call_id, requestEvent: last, requestRef: last.payload.canonical_request_ref,
        wireRef: last.artifact_ref,
        failedAttempts: after.filter((event) => event.type === "provider_attempt_failed" && event.model_call_id === last.model_call_id).length,
      };
    }
    // The act already finished; only its answer was lost with the process.
    if (finished !== null) turn.outcome = finished;
    const recordedResults = [...(this.replayLedger?.values() ?? [])].filter((event) => event.type === "tool_result");
    this.append("agent_core_resumed", {
      visibility: "derived", observation_step: params.step,
      model_call_id: turn.replay?.modelCallId ?? turn.lostRequest?.modelCallId ?? null,
      payload: {
        mode: finished !== null ? "turn_finished" : turn.replay !== null ? "reply_replayed"
          : turn.lostRequest !== null ? "request_lost" : "between_requests",
        recorded_tool_results: recordedResults.map((event) => event.tool_call_id),
        recorded_batch_events: this.replayLedger?.size ?? 0,
      },
    });
    this.turn = turn;
    return turn;
  }

  /** Earlier replies may have completed zero-spend primitives and continued this turn. */
  private settledZeroSpendAction(action: AgentEvent, events: AgentEvent[]): boolean {
    const result = events.find((event) => event.type === "tool_result"
      && event.tool_call_id === action.tool_call_id && event.model_call_id === action.model_call_id
      && event.payload?.execution);
    if (!result) return false;
    try {
      const execution = executionEvidence(result.payload.execution, action.tool_call_id!, action.model_call_id!);
      if (execution.native_actions_used !== 0
        || !execution.actions.some((item: Json) => item.action_id === action.payload?.action_id)) return false;
      this.acceptExecution(execution, action.tool_call_id!, action.model_call_id!);
      return true;
    } catch {
      return false; // Missing or corrupt completion evidence retains the strict refusal.
    }
  }

  /** Whether the persisted history holds this reply (and so everything its batch added). */
  private replyPersisted(responseEvent: Json, after: Json[]): boolean {
    let sourceId = responseEvent.event_id;
    if (responseEvent.payload?.accepted === false) {
      // A rejected reply never enters the history; its continuation message does.
      const continuation = after.find((event) => event.type === "provider_output_rejected"
        && event.model_call_id === responseEvent.model_call_id);
      if (!continuation) return false;
      sourceId = continuation.event_id;
    }
    return this.history.some((message) => Array.isArray(message._source_event_ids) && message._source_event_ids[0] === sourceId);
  }

  private stopInterrupted(turn: Turn): null {
    turn.outcome = ended("operator_interrupt", turn.attempts);
    turn.pending = { stop: true };
    return null;
  }

  /** The provider exchange of the pending request. Runs in the Pi provider's streamSimple. */
  async exchange(signal?: AbortSignal): Promise<ReplyState | null> {
    const turn = this.requireTurn();
    const pending = turn.pending;
    turn.pending = null;
    if (pending === null || "stop" in pending) return null;
    const { projected, modelCallId, retry, replay } = pending;
    const prepared = await timed("core.assemble", () => this.deps.assembler.prepare(projected));
    // Serialized once: stored here, and measured by accountCall.
    prepared.canonicalBytes = canonicalBytesPy(prepared.canonical);
    const requestRef = this.artifacts.putBytes(prepared.canonicalBytes, {
      kind: "native_canonical_provider_request", suffix: ".json", mediaType: "application/json",
    });
    const wireRef = this.artifacts.putJson(prepared.wire, { kind: "native_provider_wire_request" });
    this.deps.assembler.noteArtifacts?.(requestRef, wireRef);
    if (retry) return this.retryLostRequest(turn, prepared, projected, retry, wireRef, signal);
    if (replay) return this.replayRecordedReply(turn, prepared, projected, replay, wireRef);
    const condensation = projected.manifest?.context_condensation;
    if (condensation && typeof condensation === "object") {
      this.append("context_condensation", {
        observation_step: turn.step, model_call_id: modelCallId, visibility: "derived",
        payload: { ...clonePy(condensation), next_main_canonical_request_sha256: requestRef.sha256, next_main_wire_request_sha256: wireRef.sha256 },
      });
    }
    const projection = this.append("context_projected", {
      causal_parent_id: turn.observationEvent.event_id, observation_step: turn.step, model_call_id: modelCallId,
      visibility: "derived",
      payload: { ...projected.manifest, exact_request_sha256: requestRef.sha256, wire_request_sha256: wireRef.sha256 },
    });
    const requestEvent = this.append("provider_request", {
      causal_parent_id: projection.event_id, observation_step: turn.step, model_call_id: modelCallId,
      artifact_ref: wireRef,
      payload: { model: this.config.policy.model, provider_protocol: this.deps.provider.protocol, canonical_request_ref: requestRef },
    });
    crashPoint("after_request", this.runRoot);
    let response: Record<string, Json>;
    try {
      response = await timed("core.provider", () => this.deps.provider.complete(prepared.execution, {
        modelCallId,
        deadlineMonotonicMs: turn.requestDeadlineMs,
        signal,
        onAttemptFailure: (failure) => {
          const artifact = this.artifacts.putJson(failure, { kind: "native_provider_failed_attempt" });
          const payload: Record<string, Json> = {};
          for (const [key, value] of Object.entries(failure)) if (key !== "partial_events" && key !== "error_chain") payload[key] = value;
          this.append("provider_attempt_failed", {
            causal_parent_id: requestEvent.event_id, observation_step: turn.step, model_call_id: modelCallId,
            visibility: "derived", artifact_ref: artifact, payload,
          });
        },
      }));
    } catch (error) {
      // An operator stop aborted the request: the error is the stop's answer
      // to it (marked, as Native's error_payload marks it), not a fault.
      const stopped = this.interrupted;
      const failure = (error as any)?.providerFailure ?? {};
      this.append("provider_error", {
        causal_parent_id: requestEvent.event_id, observation_step: turn.step, model_call_id: modelCallId,
        payload: {
          exception_type: (error as any)?.pythonType ?? (error as Error)?.name ?? "Error",
          cause_type: (error as any)?.causeType ?? null,
          attempts: failure.attempts ?? null,
          response_started: failure.response_started ?? null,
          accepted: false,
          ...(stopped ? { interrupted: true } : {}),
        },
      });
      if (stopped) {
        turn.outcome = ended("operator_interrupt", turn.attempts);
        return null;
      }
      turn.fatal = error as Error;
      return null;
    }
    const reply = await timed("core.record_reply", () => this.recordReply(turn, prepared, response, modelCallId, requestEvent));
    crashPoint("after_response", this.runRoot);
    await timed("core.account", () => this.accountCall(projected, prepared, response, reply));
    sampleMemory(this.runRoot, modelCallId, () => this.memoryDetail(this.callSequence % 20 === 0));
    if (this.interrupted) {
      turn.outcome = ended("operator_interrupt", turn.attempts);
      return null;
    }
    turn.reply = reply;
    return reply;
  }

  private recordReply(turn: Turn, prepared: Json, response: Record<string, Json>, modelCallId: string, requestEvent: AgentEvent,
                      recorded: AgentEvent | null = null): ReplyState {
    if (response.wire_request !== undefined && response.wire_request !== null && !sortedKeysEqual(response.wire_request, prepared.wire)) {
      throw new Error("native provider sent a wire request different from its traced projection");
    }
    const message = response.message;
    if (typeof message !== "object" || message === null || message.role !== "assistant") {
      throw new Error("native provider response lacks an assistant message");
    }
    const assistant: OraMessage = clonePy(message);
    const outputRejection = response.output_rejection ?? null;
    const usage = response.usage ?? null;
    const payload: Record<string, Json> = {
      message: assistant, usage, finish_reason: response.finish_reason ?? null,
      attempts: response.attempts ?? 1, accepted: outputRejection === null,
    };
    if (outputRejection !== null) payload.output_rejection = clonePy(outputRejection);
    const responseEvent = recorded ?? this.append("provider_response", {
      causal_parent_id: requestEvent.event_id, observation_step: turn.step, model_call_id: modelCallId,
      payload, artifact_ref: this.artifacts.putJson(response.raw_events ?? [], { kind: "native_provider_wire_response" }),
    });
    const finishReason = response.finish_reason ?? null;
    const calls = Array.isArray(assistant.tool_calls) ? clonePy(assistant.tool_calls) : [];
    if (outputRejection !== null && calls.length) throw new Error("rejected native provider output cannot expose tool calls");
    if (finishReason === "invalid_tool_input" && outputRejection === null) throw new Error("invalid_tool_input completion lacks rejection evidence");
    if (finishReason === "length" && calls.length) throw new Error("native provider response contains a truncated tool call");
    if (calls.length && finishReason !== "tool_calls") throw new Error("native provider returned tool calls without a tool_calls completion");
    if (finishReason === "tool_calls" && !calls.length) throw new Error("native provider declared tool_calls without a complete tool call");
    assistant._source_event_ids = [responseEvent.event_id];
    if (outputRejection === null) this.pushHistory(assistant);
    return { modelCallId, assistant, responseEvent, usage, finishReason, outputRejection, calls };
  }

  private accountCall(projected: Json, prepared: Json, response: Record<string, Json>, reply: ReplyState): void {
    // canonical_bytes(request).decode("utf-8"): Python len() counts code points,
    // one per UTF-8 lead byte (the bytes are well-formed).
    const canonical: Buffer = prepared.canonicalBytes ?? canonicalBytesPy(prepared.canonical);
    let codePoints = 0;
    for (let index = 0; index < canonical.length; index += 1) if ((canonical[index] & 0xc0) !== 0x80) codePoints += 1;
    const started = response.started_unix_s;
    const finished = response.finished_unix_s;
    const startedValue = started instanceof PyFloat ? started.value : started;
    const finishedValue = finished instanceof PyFloat ? finished.value : finished;
    this.calls.push({
      turn_id: reply.modelCallId,
      message_kind: "native_projected_request",
      project_message_chars: codePoints,
      project_message_bytes: canonical.length,
      started_unix_s: started ?? null,
      finished_unix_s: finished ?? null,
      latency_s: typeof startedValue === "number" && typeof finishedValue === "number"
        ? Math.round((finishedValue - startedValue) * 1e6) / 1e6 : null,
      usage: reply.usage,
      context_projection: projected.manifest,
      provider_output_status: reply.outputRejection === null ? "accepted" : "rejected",
    });
    const usage = reply.usage;
    if (usage && typeof usage === "object") {
      for (const key of Object.keys(this.reportedTotals)) {
        const value = usage[key];
        if (typeof value === "number" && Number.isInteger(value)) this.reportedTotals[key] += value;
      }
      this.usageEvents.push({
        sequence: this.usageEvents.length, turn_id: reply.modelCallId, last: { ...usage },
        total: { ...this.reportedTotals }, model_context_window: this.config.policy.context_window_tokens,
      });
    }
    const manifest = projected.manifest;
    let estimate = manifest.pressure_estimated_input_tokens;
    if (typeof estimate !== "number" || !Number.isInteger(estimate)) estimate = manifest.raw_estimated_input_tokens;
    if (typeof estimate !== "number" || !Number.isInteger(estimate) || estimate < 1) {
      throw new Error("native Context projection lacks a positive input-token estimate");
    }
    this.deps.pipeline.observeProviderUsage({
      estimatedTokens: estimate,
      inputTokens: usage && typeof usage === "object" ? usage.input_tokens : null,
      projectionEpoch: manifest.projection_epoch ?? null,
      providerWireRequestSha256: manifest.provider_wire_request_sha256 ?? null,
    });
    this.writeManifest(null);
  }

  // -- Tool batch ------------------------------------------------------------------------

  private async batchFor(turn: Turn): Promise<BatchState> {
    if (turn.batch !== null) return turn.batch;
    const reply = turn.reply;
    if (reply === null) throw new Error("tool call without a provider reply");
    const typed = this.composition !== null && CAUSAL_SLOTS.has(this.composition);
    const canonical = (name: string) => this.config.domain.dotted_tool_names[name] ?? name;
    const { parsed, names } = await parseToolCalls(
      reply.calls, reply.modelCallId,
      (index, callId) => this.adoptSynthesizedToolCallId(reply.assistant, index, callId),
      typed ? (items) => this.rpc.request("tools.classify", {
        calls: items.map((item) => ({
          name: item.name,
          arguments: normalizeArguments(item.name, item.arguments, this.config.tools.delivered, this.config.tools.null_as_omitted, canonical),
        })),
      }) : null,
    );
    const admission = admitToolBatch(parsed, names, reply.calls.length, typed, new Set(this.config.tools.mutation_tools));
    const byCallId = new Map<string, number>();
    parsed.forEach((item, index) => byCallId.set(String(item.call.id), index));
    const distinct = new Set(parsed.map((item) => item.name));
    let aggregated = false;
    if (parsed.length >= 2 && distinct.size === 1 && !distinct.has(null)
        && parsed.every((item) => item.args !== null && item.effect === "read_only" && item.error === null)) {
      const name = parsed[0].name!;
      const begun = await this.rpc.request("tools.read_only_batch", {
        name, model_call_id: reply.modelCallId, tool_call_ids: parsed.map((item) => String(item.call.id)),
        calls: parsed.map((item) => normalizeArguments(name, item.args!, this.config.tools.delivered, this.config.tools.null_as_omitted, canonical)),
      });
      aggregated = begun?.aggregated === true;
    }
    turn.batch = { parsed, admission, byCallId, done: 0, toolImages: [], aggregated, aggregatedResults: null };
    return turn.batch;
  }

  private adoptSynthesizedToolCallId(assistant: OraMessage, index: number, callId: string): void {
    if (Object.prototype.hasOwnProperty.call(assistant, "_provider_continuation_state")) {
      throw new Error("provider continuation is bound to a tool call without a stable id");
    }
    const calls = assistant.tool_calls;
    if (!Array.isArray(calls) || index >= calls.length) return;
    if (typeof calls[index] === "object" && calls[index] !== null) calls[index].id = callId;
  }

  /**
   * Pi's `tool_call` hook: record the proposal and decide whether the call runs.
   * Returns null to execute, or the block reason (the refusal's tool message).
   */
  async beforeTool(toolCallId: string): Promise<{ block: boolean; reason?: string; terminate?: boolean }> {
    const turn = this.requireTurn();
    if (turn.outcome !== null || turn.fatal !== null || this.interrupted || turn.reply === null) {
      if (this.interrupted && turn.outcome === null) turn.outcome = ended("operator_interrupt", turn.attempts);
      return { block: true, reason: "the turn already ended; this call did not run", terminate: true };
    }
    const batch = await this.batchFor(turn);
    const index = batch.byCallId.get(toolCallId);
    if (index === undefined) throw new Error(`unknown tool call ${toolCallId}`);
    if (batch.aggregated) {
      await this.runAggregatedBatch(turn, batch);
      return { block: false };
    }
    const item = batch.parsed[index];
    // `name or "invalid_tool_call"`: a call whose function name is empty is nameless too.
    const reported = item.name || "invalid_tool_call";
    const proposal = this.append("tool_call_proposed", {
      causal_parent_id: turn.reply.responseEvent.event_id, observation_step: turn.step,
      model_call_id: turn.reply.modelCallId, tool_call_id: toolCallId,
      payload: { name: reported, arguments: item.args, effect: item.effect, batch_valid: !batch.admission.refusedMutations.has(index) },
    });
    turn.reply.assistant._source_event_ids.push(proposal.event_id);
    (item as any).proposal = proposal;
    let refusal: Record<string, Json> | null = null;
    if (batch.admission.refusedMutations.has(index)) {
      refusal = item.error !== null ? this.refusalFor(item.error) : toolErrorResult(new ToolCallError(batchRejection(batch.admission.readsRun)), this.codes);
    } else if (item.error !== null || item.name === null || item.args === null) {
      refusal = this.refusalFor(item.error ?? new ToolCallError({ message: "invalid tool call" }));
    } else {
      const gate = mutationProtocolError(this.composition, item.effect, turn.terminalLifecycle,
        this.config.domain.terminal_refusals?.[item.name]);
      if (gate !== null) refusal = toolErrorResult(new ToolCallError(gate), this.codes);
    }
    if (refusal === null) return { block: false };
    const projected = await this.rpc.request("tools.project", {
      name: reported, result: refusal, tool_call_id: toolCallId, observation_step: turn.step,
    });
    const text = this.recordToolResult(turn, batch, index, reported, projected, refusal);
    return { block: true, reason: text, terminate: this.batchTerminates(turn) };
  }

  /**
   * `_execute_aggregated_read_only_batch`: every proposal first (marked combined),
   * then one Python call runs the whole batch and returns each projected result;
   * the results are recorded in call order by `executeTool`.
   */
  private async runAggregatedBatch(turn: Turn, batch: BatchState): Promise<void> {
    if (batch.aggregatedResults !== null) return;
    const reply = turn.reply!;
    const name = batch.parsed[0].name!;
    for (const item of batch.parsed) {
      const proposal = this.append("tool_call_proposed", {
        causal_parent_id: reply.responseEvent.event_id, observation_step: turn.step,
        model_call_id: reply.modelCallId, tool_call_id: String(item.call.id),
        payload: { name, arguments: item.args, effect: item.effect, batch_valid: true, visual_feedback: "combined" },
      });
      reply.assistant._source_event_ids.push(proposal.event_id);
      (item as any).proposal = proposal;
    }
    const ran = await this.rpc.request("tools.read_only_batch_run", {
      name, observation_step: turn.step, model_call_id: reply.modelCallId,
      calls: batch.parsed.map((item) => ({ tool_call_id: String(item.call.id), arguments: item.args })),
    });
    crashPoint("after_batch_run", this.runRoot);
    const results = ran?.results;
    if (!Array.isArray(results) || results.length !== batch.parsed.length) {
      throw new Error("aggregated read-only batch returned a result count drift");
    }
    batch.aggregatedResults = results;
  }

  private refusalFor(error: unknown): Record<string, Json> {
    if (error && typeof error === "object" && "classified" in (error as any)) return (error as any).classified;
    return toolErrorResult(error, this.codes);
  }

  /** Pi tool `execute()`: run the admitted call through Python and record its result. */
  async executeTool(toolCallId: string): Promise<{ text: string; terminate: boolean; details: Json }> {
    const turn = this.requireTurn();
    const batch = turn.batch!;
    const index = batch.byCallId.get(toolCallId)!;
    const item = batch.parsed[index];
    if (batch.aggregated) {
      const text = this.recordToolResult(turn, batch, index, item.name!, batch.aggregatedResults![index], null);
      if (batch.toolImages.length > 1) throw new Error("read-only batch emitted duplicate feedback images");
      return { text, terminate: this.batchTerminates(turn), details: { ora_tool_call: toolCallId } };
    }
    const proposal = (item as any).proposal as AgentEvent;
    const canonical = (name: string) => this.config.domain.dotted_tool_names[name] ?? name;
    const args = normalizeArguments(item.name!, item.args!, this.config.tools.delivered, this.config.tools.null_as_omitted, canonical);
    const mutation = isMutation(batch.admission, item.effect, item.name);
    const projected = await timed("rpc.tools.invoke", () => {
      const pending = this.rpc.request("tools.invoke", {
        name: item.name, arguments: args, effect_is_mutation: mutation,
        model_calls_in_turn: turn.attempts,
        call_context: {
          policy_session_id: this.config.policy_session_id, model_call_id: turn.reply!.modelCallId,
          tool_call_id: toolCallId, proposal_event_id: proposal.event_id, observation_step: turn.step,
          current_observation_event_id: turn.observationEvent.event_id, interaction_phase: "robot_rollout",
        },
      });
      crashPoint("in_invoke", this.runRoot);
      return pending;
    });
    if (projected.execution_delivery_guard === true) {
      this.pendingToolDelivery.push({ model_call_id: turn.reply!.modelCallId, tool_call_id: toolCallId });
    }
    crashPoint("after_invoke", this.runRoot);
    // Events the tool recorded while it ran (the artifact reader's retrievals),
    // in order, before its result: where the retired Native session appended them.
    const runtimeEvents: Json[] = projected.runtime_events ?? [];
    for (const record of runtimeEvents) this.append(String(record.type), record.fields ?? {});
    if (runtimeEvents.length) crashPoint("after_runtime_events", this.runRoot);
    if (projected.execution) this.acceptExecution(projected.execution, toolCallId, turn.reply!.modelCallId);
    if (projected.directive === "world_state_unknown" && this.replayLedger) {
      for (const [key, event] of this.replayLedger) {
        if (event.type === "action_proposed" && event.tool_call_id === toolCallId) this.replayLedger.delete(key);
      }
    }
    const text = await timed("core.record_tool", () => this.recordToolResult(turn, batch, index, item.name!, projected, null));
    // _turn_outcome
    const directive = projected.directive;
    if (directive === "world_state_unknown" || directive === "tool_execution_interrupted") {
      turn.outcome = ended(directive, turn.attempts);
    } else if (directive === "execution_boundary") {
      turn.outcome = ended("tool_execution_boundary", turn.attempts);
    } else if (directive === "actions_executed") {
      const execution = executionEvidence(projected.execution, toolCallId);
      if (execution.native_actions_used <= 0) throw new Error("executed directive requires positive consumption");
      turn.outcome = { submitted: false, actions_executed: true, execution, attempts: turn.attempts };
    } else if (mutation && projected.raw_ok !== false && directive === "action_submitted") {
      const action = projected.action;
      const actionEvent = this.append("action_proposed", {
        observation_step: turn.step, model_call_id: turn.reply!.modelCallId, tool_call_id: toolCallId,
        payload: { action_id: action.action_id, observation_id: action.observation_id }, artifact_ref: action.artifact_ref,
      });
      turn.reply!.assistant._source_event_ids.push(actionEvent.event_id);
      crashPoint("after_action", this.runRoot);
      turn.outcome = { submitted: true, attempts: turn.attempts, action_event_id: actionEvent.event_id, tool_name: item.name };
    } else if (directive === "episode_finished") {
      turn.outcome = ended("policy_finished_episode", turn.attempts);
    }
    return { text, terminate: this.batchTerminates(turn), details: { ora_tool_call: toolCallId } };
  }

  /** Supervisor execution evidence, checked against the receipts and publications this core recorded (free play too). */
  acceptExecution(value: Json, callId: string, modelCallId?: string): Json {
    const execution = executionEvidence(value, callId, modelCallId);
    for (const action of execution.actions) {
      const receipt = this.executionEvents.get(action.receipt_event_id);
      const proposed = receipt ? this.executionEvents.get(receipt.causal_parent_id) : undefined;
      if (!receipt || receipt.type !== "environment_receipt" || receipt.payload?.action_id !== action.action_id
        || !canonicalBytesPy(receipt.artifact_ref).equals(canonicalBytesPy(action.receipt_ref)) || !proposed
        || proposed.type !== "action_proposed" || proposed.payload?.action_id !== action.action_id
        || proposed.model_call_id !== execution.model_call_id
        || proposed.tool_call_id !== callId
        || proposed.payload?.observation_id !== action.before_observation.observation_id) {
        throw new Error("execution receipt does not resolve to trusted action evidence");
      }
      if (action.after_observation.step !== action.before_observation.step
        || action.after_observation.observation_id !== action.before_observation.observation_id) {
        const published = [...this.executionEvents.values()].some((event) =>
          event.type === "fresh_observation_published"
          && event.observation_step === action.after_observation.step
          && event.payload?.observation_id === action.after_observation.observation_id
          && event.payload?.after_receipt?.event_id === action.receipt_event_id);
        if (!published) throw new Error("execution lacks its trusted completed publication");
      }
      // Validate historical evidence without granting permission to recover a parent.
      // The synchronous host rejects any current-parent cached result request.
      const key = replayKey(proposed.type, proposed.model_call_id, proposed.tool_call_id, proposed.payload);
      this.replayLedger?.delete(key);
    }
    return execution;
  }

  /** `_record_tool_result`, with Python's projection and hints applied here. */
  private recordToolResult(turn: Turn, batch: BatchState, index: number, name: string, projected: Json, refusal: Json): string {
    const item = batch.parsed[index];
    const proposal = (item as any).proposal as AgentEvent;
    const toolCallId = String(item.call.id);
    const state = new StaticFactsState();
    const canonical = applyHints(projected.canonical, projected.canonical_hints ?? [], state).delivered;
    const visible = applyHints(projected.visible, projected.visible_hints ?? [], state).delivered;
    // session_core/repeated_failures.py: annotate the new result only (after delivery; the rule commutes with it).
    const repetition = this.repeatedFailures.record(name, item.args, refusal ?? canonical);
    if (repetition !== null) {
      if (canonical.error && typeof canonical.error === "object") canonical.error = { ...canonical.error, repetition };
      if (visible.error && typeof visible.error === "object") visible.error = { ...visible.error, repetition };
    }
    const text = dumpsSorted(visible);
    const toolMessage: OraMessage = { role: "tool", tool_call_id: toolCallId, name, content: text };
    if (projected.execution) toolMessage._execution = clonePy(executionEvidence(projected.execution, toolCallId));
    this.pushHistory(toolMessage);
    const ref = this.artifacts.putJson(canonical, { kind: "native_tool_result" });
    const ok = typeof visible.ok === "boolean" ? visible.ok
      : !("error" in visible) && !["error", "failed", "invalid", "rejected"].includes(visible.status);
    const resultEvent = this.append("tool_result", {
      causal_parent_id: proposal.event_id, observation_step: turn.step, model_call_id: turn.reply!.modelCallId,
      tool_call_id: toolCallId, payload: { name, ok, ...(toolMessage._execution ? { execution: toolMessage._execution } : {}) }, artifact_ref: ref,
    });
    toolMessage._source_event_ids = [resultEvent.event_id];
    toolMessage._artifact_ref = ref;
    crashPoint("after_tool_result", this.runRoot);
    const images: Json[] = projected.images ?? [];
    if (images.length) {
      const imageEvents = images.map((imageRef: Json, imageIndex: number) => this.append("tool_image", {
        visibility: "exact", observation_step: turn.step, model_call_id: turn.reply!.modelCallId,
        tool_call_id: toolCallId, artifact_ref: imageRef, payload: { image_index: imageIndex },
      }));
      const contextRefs = this.registerToolImages(images, turn.step);
      if (projected.image_text !== null && projected.image_text !== undefined) {
        const message: OraMessage = {
          role: "user",
          content: [
            { type: "text", text: projected.image_text },
            ...images.map((imageRef: Json) => ({ type: "image_url", image_url: { url: this.dataUrl(imageRef) } })),
          ],
          _tool_image_step: turn.step,
          _tool_image_name: name,
          _tool_image_call_id: toolCallId,
          _source_event_ids: imageEvents.map((event) => event.event_id),
        };
        if (contextRefs.length) message._context_image_refs = contextRefs;
        batch.toolImages.push(message);
      }
    }
    batch.done += 1;
    return text;
  }

  /** Sizes of what this process holds, for the opt-in memory samples (ORA_PI_MEMORY_LOG). */
  memoryDetail(sizes: boolean): Record<string, Json> {
    const detail: Record<string, Json> = { history_messages: this.history.length, calls: this.calls.length };
    if (!sizes) return detail;
    const bytes = (value: Json) => JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item))?.length ?? 0;
    detail.history_bytes = bytes(this.history);
    detail.calls_bytes = bytes(this.calls);
    detail.usage_events_bytes = bytes(this.usageEvents);
    detail.context_sources_bytes = bytes(this.contextSources);
    detail.provider_requests = (this.deps?.provider as Json)?.requests?.length ?? null;
    Object.assign(detail, this.hostMemoryDetail?.() ?? {});
    return detail;
  }

  /** The Pi bridge's own sizes (session entries), for memoryDetail. */
  hostMemoryDetail: (() => Record<string, Json>) | null = null;

  /** Lossless retention registers each tool image as a recoverable Context Artifact. */
  registerToolImages: (images: Json[], step: number) => Json[] = () => [];

  private batchTerminates(turn: Turn): boolean {
    if (turn.outcome !== null || turn.fatal !== null) return true;
    return turn.attempts >= turn.callLimit || this.interrupted;
  }

  /** Pi's `turn_end`: flush tool images, then decide whether the turn continues. */
  endOfReply(): { continueTurn: boolean; continuation: OraMessage | null } {
    const result = this.endOfReplyInner();
    this.closeReplay();
    return result;
  }

  private endOfReplyInner(): { continueTurn: boolean; continuation: OraMessage | null } {
    const turn = this.requireTurn();
    const reply = turn.reply;
    const batch = turn.batch;
    turn.reply = null;
    turn.batch = null;
    if (turn.outcome !== null || turn.fatal !== null) return { continueTurn: false, continuation: null };
    if (reply === null) return { continueTurn: false, continuation: null };
    if (batch !== null) {
      // _run_tool_batch: images of the batch follow all its tool messages.
      if (batch.toolImages.length) this.pushHistory(...batch.toolImages);
      if (this.interrupted) {
        turn.outcome = ended("operator_interrupt", turn.attempts);
        return { continueTurn: false, continuation: null };
      }
      if (turn.attempts >= turn.callLimit) turn.outcome = ended("model_call_limit", turn.attempts);
      return { continueTurn: false, continuation: null };
    }
    let continuation: continuations.Continuation;
    if (reply.outputRejection !== null) {
      continuation = continuations.afterRejectedOutput(reply.outputRejection);
    } else if (reply.finishReason === "length") {
      continuation = continuations.afterTruncatedOutput();
    } else {
      if (turn.terminalLifecycle || !turn.allowReminder) {
        turn.outcome = ended("policy_no_action", turn.attempts);
        return { continueTurn: false, continuation: null };
      }
      continuation = continuations.afterNoAction(this.config.tools.mutation_tools.length ? this.config.tools.mutation_tools : ["robot.submit_action"]);
    }
    const event = this.append(continuation.eventType, {
      causal_parent_id: reply.responseEvent.event_id, observation_step: turn.step,
      model_call_id: reply.modelCallId, visibility: "derived", payload: continuation.payload,
    });
    const message: OraMessage = { role: "user", content: continuation.content, _source_event_ids: [event.event_id] };
    this.pushHistory(message);
    if (this.interrupted) {
      turn.outcome = ended("operator_interrupt", turn.attempts);
      return { continueTurn: false, continuation: message };
    }
    if (turn.attempts >= turn.callLimit) {
      turn.outcome = ended("model_call_limit", turn.attempts);
      return { continueTurn: false, continuation: message };
    }
    return { continueTurn: true, continuation: message };
  }

  /** The outcome of the last finished act, persisted so a restart after it returns the same. */
  lastTurnOutcome: { step: number; outcome: Outcome } | null = null;

  finishTurn(): Outcome {
    const turn = this.requireTurn();
    this.turn = null;
    this.replayLedger = null;
    let outcome: Outcome | null = null;
    let failure: Error | null = null;
    if (turn.fatal !== null) failure = turn.fatal;
    else if (turn.outcome !== null) outcome = turn.outcome;
    else if (this.interrupted) outcome = ended("operator_interrupt", turn.attempts);
    else failure = new Error("the Pi run ended without an ORA turn outcome");
    if (outcome !== null) this.lastTurnOutcome = { step: turn.step, outcome: clonePy(outcome) };
    this.flush();
    if (failure !== null) throw failure;
    crashPoint("after_finish", this.runRoot);
    return outcome!;
  }

  requireTurn(): Turn {
    if (this.turn === null) throw new Error("no active ORA turn");
    return this.turn;
  }

  // -- Runtime events ----------------------------------------------------------------

  record(type: string, fields: AppendOptions): AgentEvent {
    const event = this.resentRecord(type, fields) ?? this.append(type, fields);
    crashPoint(`after_record.${type}`, this.runRoot);
    return event;
  }

  /**
   * The last event the previous process recorded, when the restarted core's
   * first append is Python re-sending that very record (the process died after
   * writing it, before answering): reused, never written twice.
   */
  private resentCandidate: { event: AgentEvent; parent: string | null } | null = null;

  private resentRecord(type: string, fields: AppendOptions): AgentEvent | null {
    const candidate = this.resentCandidate;
    this.resentCandidate = null;
    if (candidate === null || candidate.event.type !== type) return null;
    const expected = { ...fields };
    if (!Object.prototype.hasOwnProperty.call(expected, "causal_parent_id")) expected.causal_parent_id = candidate.parent;
    const recorded = canonicalBytesPy(decidedFields(type, candidate.event as AppendOptions));
    if (!canonicalBytesPy(decidedFields(type, expected)).equals(recorded)) return null;
    this.lastEventId = candidate.event.event_id;
    return candidate.event;
  }

  async officialVerdict(params: { artifact_ref: Json; official_success: Json; has_verdict: boolean }): Promise<AgentEvent> {
    const fields: AppendOptions = {
      visibility: params.has_verdict ? "exact" : "opaque",
      payload: params.has_verdict ? { official_success: params.official_success } : null,
      artifact_ref: params.artifact_ref,
    };
    // A re-sent verdict reuses its event; its history message was never persisted, so it is added.
    const event = this.resentRecord("official_verdict", fields) ?? this.append("official_verdict", fields);
    crashPoint("after_record.official_verdict", this.runRoot);
    await this.durability.barrier();
    if (this.retention === LOSSLESS && typeof params.official_success === "boolean") {
      let roundNumber: Json = null;
      for (let i = this.history.length - 1; i >= 0; i -= 1) {
        const value = this.history[i]._session_round;
        if (typeof value === "number" && Number.isInteger(value)) {
          roundNumber = value;
          break;
        }
      }
      this.pushHistory({
        role: "user",
        content: "<official_round_result>\n" + dumpsCompactSorted({
          round: roundNumber, passed: params.official_success,
          instruction: "This completed-round result is exact. Reflect on durable task-local lessons.",
        }) + "\n</official_round_result>",
        _protected_context: true,
        _context_source: `official_round_result_${roundNumber}`,
        _source_event_ids: [event.event_id],
      });
    }
    return event;
  }

  async operatorMessage(payload: Record<string, Json>): Promise<AgentEvent> {
    if (this.closed || this.interrupted) throw new Error("Native policy session is not active");
    const event = this.append("operator_message", { observation_step: payload.observation_step, visibility: "exact", payload });
    crashPoint("after_operator_message", this.runRoot);
    if (this.operatorDelivery !== null) {
      this.operatorDelivery(payload);
      await this.durability.barrier();
      return event;
    }
    this.pushHistory({
      role: "user",
      content: "<operator_message>\n" + dumpsCompactSorted({
        id: payload.id, message: payload.message,
        instruction: payload.type === "instruction" ? PLAY_INSTRUCTION_GUIDANCE
          : "Human operator guidance for the current benchmark task. The original task and official "
            + "evaluator remain unchanged; all robot actions still use the advertised tools.",
      }) + "\n</operator_message>",
      _protected_context: true,
      _context_source: `operator_message_${payload.id}`,
      _source_event_ids: [event.event_id],
    });
    await this.durability.barrier();
    return event;
  }

  preserveObservation(params: ActParams): void {
    if (this.closed || this.interrupted) throw new Error("native policy session is not active");
    this.bindStatic(params.static);
    const { message: observation, slot } = this.resolveDraft(params.draft);
    this.pushHistory(observation);
    const meta = observation._observation;
    const payload: Record<string, Json> = { observation_id: meta.observation_id, state_ref: meta.state_ref, provider_turn: false };
    if (this.composition !== null && CAUSAL_SLOTS.has(this.composition)) {
      payload.causal_pair = Boolean(observation._causal_observation?.paired);
      payload.causal_cameras = [...(observation._causal_observation?.cameras ?? [])];
    }
    const observationEvent = this.append("observation_received", { observation_step: params.step, payload });
    const sourceIds = [observationEvent.event_id];
    if (typeof params.receipt_event_id === "string") sourceIds.push(params.receipt_event_id);
    if (params.last_fresh_observation_event_id !== null) sourceIds.push(params.last_fresh_observation_event_id);
    observation._source_event_ids = sourceIds;
    if (slot !== null) {
      slot._source_event_ids = [observationEvent.event_id];
      this.pushHistory(slot);
    }
    this.append("observation_preserved_without_policy_turn", {
      causal_parent_id: observationEvent.event_id, observation_step: params.step, visibility: "derived",
      payload: { reason: "terminal_round_boundary", terminal_lifecycle: params.terminal_lifecycle },
    });
    this.writeManifest(null);
  }

  interrupt(reason: string): void {
    if (this.interrupted) return;
    this.interrupted = true;
    this.append("session_interrupted", { visibility: "derived", payload: { reason } });
  }

  /** A Pi-side abort (Esc): stop here and hand the stop to the Supervisor's operator path. */
  operatorAbort(): void {
    if (this.interrupted || this.closed) return;
    this.interrupt("operator_interrupt");
    this.rpc.notify("operator.stop", { reason: "operator_interrupt" });
  }

  /**
   * Typed operator input: the Supervisor admits and later delivers it. A steer by default; a
   * free-play terminal sends instructions, queued instructions (held until the running one ends)
   * and its controls (stop_instruction, reset, quit).
   */
  async submitOperatorMessage(message: string, type = "steer"): Promise<Json> {
    if (this.closed || this.interrupted) throw new Error("the policy session is not active");
    const record = await this.rpc.request("operator.submit", { message, type });
    if ((type === "instruction" || type === "queued_instruction") && typeof record?.id === "string") {
      this.playInstructions.set(record.id, message);
    }
    return record;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    try {
      this.append("session_closed", { visibility: "derived", payload: { reason: "closed" } });
      this.writeManifest("closed");
      await this.durability.barrier();
    } finally {
      this.closed = true;
      this.recorder.close();
      this.deps?.provider.close();
      this.deps?.closeCompactor?.();
    }
    writeAllocationProfile(this.runRoot);
    ablateMemory(this.runRoot, [
      ["provider_requests", () => { const provider = this.deps?.provider as Json; if (provider?.requests) provider.requests = []; }],
      ...(this.hostMemoryHolders?.() ?? []),
      ["calls", () => { this.calls.length = 0; }],
      ["history", () => { this.history.length = 0; }],
      ["usage_events", () => { this.usageEvents.length = 0; }],
      ["context_pipeline", () => { (this as Json).deps = null; }],
    ]);
  }

  /** The Pi bridge's holders for the opt-in ablation at close (ORA_PI_MEMORY_LOG=ablate). */
  hostMemoryHolders: (() => Array<[string, () => void]>) | null = null;
}

