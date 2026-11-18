// Context-level replay of a sealed run: every model call through the core's Context pipeline.
//
//   node ora-pi/test/support/context_replay.ts REPLAY_DIR --out RESULT.json
//        [--work DIR] [--calls N] [--domain JSON|@FILE]
//        [--scheme FILE | --management JSON|@FILE] [--policy JSON|@FILE] [--usage recorded|scaled|none]
//        [--context-stages JSON|@FILE]
//        [--inflate F [--inflate-from-call K]] [--round-split K] [--match-compactor]
//        [--manifests-out FILE.jsonl.gz] [--diff-requests] [--stop-on-mismatch]
//
// REPLAY_DIR comes from tests/support/agent_core/context_replay/extract.py. The
// pipeline, compactor and request assembler are built by the core's own
// construction code (core/context_wiring.ts `contextDeps`) from the recorded
// plan; only the core around them is replaced: a run root under --work holding
// the replay's image blobs, an event sink answering the compactor's events with
// the recorded event ids, and the recorded compactor responses in place of the
// compactor's Provider (config.provider.kind "replay": a scripted compactor
// whose wire projection is the plan's Provider adapter, built offline).
//
// Per model call it then does what the core does around one request
// (core/session.ts prepareRequest, exchange, accountCall): register the tool
// images that arrived since the previous call, extend the history, project,
// annotate the context sources, assemble, and feed the recorded Provider usage
// back with observeProviderUsage. Each result carries the canonical request
// sha256 and size, the wire sha256, the manifest sha256 and its projection
// transitions, compared with what the run recorded; the pipeline snapshot
// before each projection is compared with the run's ora.state.
//
// A compactor request that differs from the recorded one stops a faithful
// replay. With --scheme, --management or --policy the context settings are
// substituted: a compactor request is answered by the recorded response whose
// request had the same messages, otherwise by a fixed, numbered stub checkpoint
// (flagged). --match-compactor answers the same way without substituting
// anything: a run replayed on another tree, recorded as a reference. Two deterministic variants that are not the run either, answered
// the same way:
//   --inflate F [--inflate-from-call K]  every recorded Provider input-token
//       count from call K on is multiplied by F (the agent-core goldens'
//       `inflate_usage`), so the usage-anchored layers (L5, L6) cross their
//       watermarks earlier;
//   --round-split K  the session becomes two rounds: the observation that opens
//       a turn in call K's history starts round 2 (every later _session_round
//       + 1) and round 1 closes with an official round result (passed: false)
//       just before it, as the core appends one (session.ts officialVerdict);
//       with a closed Episode, L5 can summarize it. No sealed run has two rounds.

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { performance } from "node:perf_hooks";
import { execFileSync } from "node:child_process";
import { gunzipSync, gzipSync } from "node:zlib";

import { contextDeps } from "../../src/core/context_wiring.ts";
import { providerFromSpec } from "../../src/core/deps.ts";
import { hydrate } from "../../src/core/persistence.ts";
import { STAGE_SCHEMAS } from "./stage_schemas.ts";
import { ArtifactStore, NO_DURABILITY } from "../../src/evidence/artifacts.ts";
import { CHECKPOINT_FIELDS } from "../../src/context/projection.ts";
import { canonicalBytesPy, clonePy, dumpsCompactSorted, dumpsPy, parsePy, PyFloat } from "../../src/pyjson.ts";

type Json = any;

export const RESULT_SCHEMA = "ora.context-replay-result.v1";

function sha256(data: Buffer): string {
  return createHash("sha256").update(data).digest("hex");
}

function canonicalSha(value: unknown): string {
  return sha256(canonicalBytesPy(value));
}

function intValue(value: unknown): number | null {
  if (value instanceof PyFloat) return value.value;
  if (typeof value === "bigint") return Number(value);
  return typeof value === "number" ? value : null;
}

/** Large data URLs replaced by their sha256 and length (extract.py `skeleton`). */
function skeleton(value: unknown): unknown {
  if (typeof value === "string") {
    return value.length >= 1024 && value.startsWith("data:")
      ? `<data ${createHash("sha256").update(value, "utf8").digest("hex")} ${value.length}>` : value;
  }
  if (Array.isArray(value)) return value.map(skeleton);
  if (value !== null && typeof value === "object" && !(value instanceof PyFloat)) {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value)) out[key] = skeleton((value as Json)[key]);
    return out;
  }
  return value;
}

/**
 * Compactor bookkeeping a replay cannot reproduce: the call's wall-clock latency
 * and the raw Provider events (the scripted compactor stores a placeholder).
 */
function withoutCompactorClock(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(withoutCompactorClock);
  if (value !== null && typeof value === "object" && !(value instanceof PyFloat)) {
    const record = value as Record<string, unknown>;
    const compactorRecord = "attempt_event_id" in record && "raw_response_ref" in record;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(record)) {
      out[key] = compactorRecord && (key === "latency_s" || key === "raw_response_ref")
        ? "<compactor clock>" : withoutCompactorClock(record[key]);
    }
    return out;
  }
  return value;
}

/** The first path at which two decoded JSON values differ (canonical equality), or null. */
export function firstDifference(left: unknown, right: unknown, at = "$"): string | null {
  if (canonicalBytesPy(left).equals(canonicalBytesPy(right))) return null;
  if (Array.isArray(left) && Array.isArray(right)) {
    for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
      const found = firstDifference(left[index], right[index], `${at}[${index}]`);
      if (found !== null) return found;
    }
    return `${at} (length ${left.length} != ${right.length})`;
  }
  const isObject = (value: unknown) => value !== null && typeof value === "object" && !Array.isArray(value)
    && !(value instanceof PyFloat);
  if (isObject(left) && isObject(right)) {
    const keys = [...new Set([...Object.keys(left as Json), ...Object.keys(right as Json)])].sort();
    for (const key of keys) {
      if (!(key in (left as Json))) return `${at}.${key} (only in recorded)`;
      if (!(key in (right as Json))) return `${at}.${key} (only in replay)`;
      const found = firstDifference((left as Json)[key], (right as Json)[key], `${at}.${key}`);
      if (found !== null) return found;
    }
  }
  const show = (value: unknown) => {
    const text = dumpsPy(skeleton(value), { sortKeys: true, ensureAscii: false });
    return text.length > 200 ? `${text.slice(0, 200)}...(${text.length})` : text;
  };
  return `${at}: replay ${show(left)} != recorded ${show(right)}`;
}

/** The JSON lines of a gzip file, decoded one at a time (a long run's manifests exceed the longest string). */
function readGzipLines(file: string): Json[] {
  if (!fs.existsSync(file)) return [];
  const data = gunzipSync(fs.readFileSync(file));
  const lines: Json[] = [];
  for (let start = 0; start < data.length;) {
    const newline = data.indexOf(10, start);
    const end = newline === -1 ? data.length : newline;
    if (end > start) lines.push(parsePy(data.toString("utf8", start, end)));
    start = end + 1;
  }
  return lines;
}

export interface Replay {
  dir: string;
  data: Json;
}

export function loadReplay(dir: string): Replay {
  const data = parsePy(fs.readFileSync(path.join(dir, "replay.json"), "utf8"));
  if (data.schema !== "ora.context-replay.v1") throw new Error(`unsupported replay schema ${data.schema}`);
  return { dir, data };
}

export interface ReplayOptions {
  work?: string;
  /** `domain.context` in place of the replay's (what Python on the current tree sends at init). */
  domain?: Record<string, Json> | null;
  /** Multiply recorded Provider input tokens from call `fromCall` on by `factor`. */
  inflate?: { factor: number; fromCall: number } | null;
  /** Split the session into two rounds at the turn opened in this call's history (see the header). */
  roundSplit?: number | null;
  /**
   * Answer compactor requests by matching recorded ones (else the stub) instead of stopping at
   * the first that differs: a run replayed on another tree, recorded as a reference.
   */
  matchCompactor?: boolean;
  /** Policy fields replacing the recorded ones (a scheme's `policy`, or context_management alone). */
  policyOverrides?: Record<string, Json> | null;
  /**
   * A stage list (`policy.context_stages`, as Python resolves an ora.context-scheme.v2 scheme) for the
   * core to run instead of translating the recorded management object. The replay stays faithful: the
   * core refuses a list that is not that object's stage list.
   */
  contextStages?: Record<string, Json> | null;
  /** How recorded Provider usage feeds observeProviderUsage: as recorded, scaled to the replay's estimate, or not at all. */
  usage?: "recorded" | "scaled" | "none";
  manifestsOut?: string | null;
  diffRequests?: boolean;
  stopOnMismatch?: boolean;
  limit?: number | null;
  log?: (line: string) => void;
}

interface CompactorScript {
  recorded: Json[];
  used: Set<number>;
  current: { call: Json; next: number } | null;
  lastRequest: Json | null;
  divergence: string | null;
  results: Json[];
}

/** The scripted compactor reply for a request no recorded call answers: numbered, fixed usage. */
function stubResponse(revision: number): Json {
  const body: Record<string, string[]> = {};
  for (const field of CHECKPOINT_FIELDS) body[field] = [];
  body.task_progress = [`Checkpoint revision ${revision}: earlier exchanges are summarized by a fixed placeholder.`];
  body.re_ground_next = ["Re-ground the task objects from the current images first."];
  const content = JSON.stringify(body);
  const outputTokens = Math.max(1, Math.floor(Buffer.byteLength(content, "utf8") / 3));
  return {
    message: { role: "assistant", content },
    usage: {
      input_tokens: 40000, cached_input_tokens: 0, output_tokens: outputTokens, reasoning_output_tokens: 0,
      total_tokens: 40000 + outputTokens, cache_read_input_tokens: null, cache_creation_input_tokens: null,
      cache_read_status: "missing", cache_creation_status: "missing",
    },
    finish_reason: "stop",
  };
}

/** Round 1's official result, as OraCore.officialVerdict appends it (passed: false). */
function officialRoundResult(round: number): Json {
  return {
    role: "user",
    content: "<official_round_result>\n" + dumpsCompactSorted({
      round, passed: false,
      instruction: "This completed-round result is exact. Reflect on durable task-local lessons.",
    }) + "\n</official_round_result>",
    _protected_context: true,
    _context_source: `official_round_result_${round}`,
    _source_event_ids: ["agent-event-999999"],
  };
}

function treeCommit(): string | null {
  try {
    return execFileSync("git", ["-C", path.dirname(new URL(import.meta.url).pathname), "rev-parse", "HEAD"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

export async function runReplay(replay: Replay, options: ReplayOptions = {}): Promise<Json> {
  const log = options.log ?? (() => {});
  const started = performance.now();
  const data = replay.data;
  const settings = data.settings;
  const substituted = options.policyOverrides !== null && options.policyOverrides !== undefined
    && Object.keys(options.policyOverrides).length > 0;
  const inflate = options.inflate ?? null;
  const roundSplit = options.roundSplit ?? null;
  // Anything but the run itself answers compactor requests by matching or by stub, never by position.
  const scripted = substituted || inflate !== null || roundSplit !== null || options.matchCompactor === true;
  const mode = substituted ? "substituted" : inflate !== null || roundSplit !== null ? "variant"
    : scripted ? "matched" : "faithful";
  const usageMode = options.usage ?? "recorded";
  const policy = clonePy(settings.policy);
  if (substituted) Object.assign(policy, clonePy(options.policyOverrides));
  if (options.contextStages) policy.context_stages = clonePy(options.contextStages);
  if (policy.context_pack) throw new Error("context packs are not replayed (their view needs the Python RPC peer)");

  // -- Run root: the replay's images at the paths the history and registrations name.
  const work = options.work ?? fs.mkdtempSync(path.join(os.tmpdir(), "context-replay-"));
  const runRoot = path.join(work, "run");
  fs.rmSync(runRoot, { recursive: true, force: true });
  fs.mkdirSync(path.join(runRoot, "native/agent-artifacts"), { recursive: true });
  for (const entry of Object.values(data.blobs) as Json[]) {
    for (const relative of entry.paths) {
      // A real file, never a symlink: the Artifact catalog refuses links.
      fs.copyFileSync(path.join(replay.dir, entry.file), path.join(runRoot, relative));
    }
  }

  // -- Compactor: recorded responses, chosen when the compactor asks (see `responses`).
  const script: CompactorScript = {
    recorded: data.calls.flatMap((call: Json) => call.compactor.map((item: Json) => ({ ...item, model_call_id: call.model_call_id }))),
    used: new Set(), current: null, lastRequest: null, divergence: null, results: [],
  };
  let stubs = 0;
  const nextResponse = (): Json => {
    const request = script.lastRequest;
    script.lastRequest = null;
    if (request === null) throw new Error("the compactor asked for a response before preparing its request");
    const messagesSha = canonicalSha(request.messages.slice(1));
    const requestSha = sha256(canonicalBytesPy(request));
    const current = script.current;
    const expected = current !== null ? current.call.compactor[current.next] ?? null : null;
    const outcome: Json = {
      model_call_id: current?.call.model_call_id ?? null, request_sha256: requestSha, messages_sha256: messagesSha,
      recorded_call_id: null, request_match: null, source: null,
    };
    script.results.push(outcome);
    if (current !== null) current.next += 1;
    let chosen: Json | null = null;
    if (!scripted) {
      if (expected === null || expected.messages_sha256 !== messagesSha) {
        script.divergence = expected === null
          ? `${outcome.model_call_id}: the replay made a compactor call the run did not make`
          : `${outcome.model_call_id}: compactor request ${expected.call_id} diverged (messages ${messagesSha} != ${expected.messages_sha256})`;
        outcome.source = "diverged";
        throw new Error(script.divergence);
      }
      chosen = expected;
    } else {
      const index = script.recorded.findIndex((item, position) => !script.used.has(position)
        && item.messages_sha256 === messagesSha && item.response !== null);
      if (index >= 0) {
        script.used.add(index);
        chosen = script.recorded[index];
      }
    }
    if (chosen !== null) {
      outcome.recorded_call_id = chosen.call_id;
      outcome.request_match = chosen.request_sha256 === requestSha;
      outcome.source = "recorded";
      if (chosen.response === null) {
        // The recorded call failed without a response; fail it the same way.
        return { message: "recorded compactor call failed without a response" };
      }
      return {
        message: clonePy(chosen.response.message), usage: clonePy(chosen.response.usage),
        finish_reason: chosen.response.finish_reason,
      };
    }
    outcome.source = "stub";
    stubs += 1;
    return stubResponse(stubs);
  };
  const responses = new Proxy([] as Json[], {
    get(target, property) {
      if (property === "length") return Number.MAX_SAFE_INTEGER;
      if (typeof property === "string" && /^[0-9]+$/.test(property)) return nextResponse();
      return Reflect.get(target, property);
    },
  });

  // -- The plan's Provider adapter, offline: only its wire projection runs.
  const provider: Json = providerFromSpec(settings.provider, { apiKey: "context-replay-offline", standbyApiKeys: [] }, { maxRetries: 0 });
  const prepareWire = provider.prepareRequest.bind(provider);
  provider.prepareRequest = (request: Json) => {
    // The compactor's requests carry no tools; the policy's always do.
    if (!Object.prototype.hasOwnProperty.call(request, "tools")) script.lastRequest = request;
    return prepareWire(request);
  };
  provider.complete = async () => {
    throw new Error("the context replay never calls the policy Provider");
  };

  // -- The core around the context deps.
  let syntheticEvent = 0;
  const core: Json = {
    runRoot,
    artifacts: new ArtifactStore(runRoot, { durability: NO_DURABILITY }),
    rpc: { request: async () => { throw new Error("the context replay has no RPC peer"); } },
    registerToolImages: () => [],
    append: (type: string, fields: Json) => {
      // The attempt is appended before the compactor asks for its response, the rest after.
      const current = script.current;
      const position = current === null ? -1 : type === "context_compactor_attempt" ? current.next : current.next - 1;
      const recorded = current !== null ? current.call.compactor[position] ?? null : null;
      const fromRecord = !scripted && recorded !== null ? {
        context_compactor_attempt: recorded.attempt_event_id,
        context_compactor_response: recorded.response_event_id,
        context_compactor_failed: recorded.failed_event_id,
      }[type] : null;
      syntheticEvent += 1;
      return { event_id: fromRecord ?? `replay-event-${String(syntheticEvent).padStart(6, "0")}`, type, ...fields };
    },
  };
  const config: Json = {
    run_root: runRoot,
    run_id: settings.run_id,
    policy_session_id: settings.policy_session_id,
    event_schema: "ora.agent-event.v1",
    session_started_payload: {},
    policy,
    provider: { kind: "replay", recordings: [], compactor_responses: responses },
    tools: {
      definitions: settings.tools.definitions, delivered: [], mutation_tools: settings.tools.mutation_tools,
      lifecycle_tools: [], capabilities: [], typed_effects: false, null_as_omitted: false,
    },
    domain: {
      argument_refusal_codes: [], infrastructure_codes: [], turn_instruction: "", composition_context_capabilities: [],
      dotted_tool_names: {}, context: options.domain ?? settings.domain.context,
      artifact_reader_name: settings.domain.artifact_reader_name,
      // What Python on this tree sends at init (ora/agent/pi/domain.py context_stage_schemas).
      context_stage_schemas: STAGE_SCHEMAS,
    },
  };
  const deps = contextDeps(core, config, provider);
  const tools = settings.tools.definitions;
  const composition = policy.context_composition ?? null;

  const recordedManifests = readGzipLines(path.join(replay.dir, "recorded-manifests.jsonl.gz"));
  const recordedStates = readGzipLines(path.join(replay.dir, "recorded-states.jsonl.gz"));
  const recordedRequests = options.diffRequests
    ? new Map(readGzipLines(path.join(replay.dir, "recorded-requests.jsonl.gz")).map((item) => [item.model_call_id, item]))
    : new Map();
  const manifestsOut: string[] = [];
  const history: Json[] = [];
  const results: Json[] = [];
  let stopped: string | null = null;
  const limit = options.limit ?? data.calls.length;

  let inserted = 0;  // messages the round split added before this point of the history
  let splitAt: number | null = null;  // the history index of round 2's first observation
  for (const call of data.calls.slice(0, limit)) {
    const callStarted = performance.now();
    for (const registration of call.tool_image_registrations) core.registerToolImages(registration.images, registration.step);
    for (const message of call.history_append) {
      const hydrated = hydrate(message, runRoot);
      if (roundSplit !== null && call.index >= roundSplit) {
        if (splitAt === null && call.index === roundSplit && hydrated._observation !== undefined) {
          const round = intValue(hydrated._session_round) ?? 1;
          history.push(officialRoundResult(round));
          inserted += 1;
          splitAt = history.length;
        }
        if (splitAt !== null && Object.prototype.hasOwnProperty.call(hydrated, "_session_round")) {
          hydrated._session_round = (intValue(hydrated._session_round) ?? 1) + 1;
        }
      }
      history.push(hydrated);
    }
    if (roundSplit !== null && call.index === roundSplit && splitAt === null) {
      throw new Error(`--round-split ${roundSplit}: that call's history opens no turn`);
    }
    if (history.length !== call.history_length + inserted) throw new Error(`call ${call.index}: history length ${history.length} != ${call.history_length}`);
    const observationIndex = call.observation_index === null ? null
      : call.observation_index + (splitAt !== null && call.observation_index + 1 >= splitAt ? 1 : 0);
    const result: Json = {
      index: call.index, kind: call.kind, model_call_id: call.model_call_id, observation_step: call.observation_step,
      history_messages: history.length,
    };
    results.push(result);
    // The pipeline state this projection starts from, against the run's ora.state flushed before it.
    const before = deps.pipeline.snapshot();
    const recordedState = recordedStates[call.index]?.context ?? null;
    result.state_sha256 = canonicalSha(withoutCompactorClock(before));
    if (recordedState !== null && !scripted) {
      const replayState = withoutCompactorClock(before);
      const expectedState = withoutCompactorClock(recordedState);
      result.state_match = canonicalBytesPy(replayState).equals(canonicalBytesPy(expectedState));
      if (!result.state_match) result.state_difference = firstDifference(replayState, expectedState);
    }
    script.current = { call, next: 0 };
    let projected: Json;
    try {
      projected = await deps.pipeline.project({ messages: history, tools, observationStep: call.observation_step });
    } catch (error) {
      const message = String((error as Error)?.message ?? error);
      result.error = message;
      result.context_exhausted = message.startsWith("context_exhausted");
      result.match = { canonical: call.kind === "context_exhausted" && result.context_exhausted };
      if (script.divergence !== null) {
        stopped = script.divergence;
        break;
      }
      if (call.kind === "context_exhausted") continue;
      stopped = `${call.model_call_id}: projection failed: ${message}`;
      break;
    } finally {
      result.compactor = script.results.splice(0);
      const unused = script.current !== null ? call.compactor.length - script.current.next : 0;
      if (!scripted && unused > 0) {
        stopped = stopped ?? `${call.model_call_id}: ${unused} recorded compactor call(s) were not made by the replay`;
      }
      script.current = null;
    }
    if (stopped !== null) break;
    if (script.divergence !== null) {
      stopped = script.divergence;
      break;
    }
    if (call.kind === "context_exhausted") {
      result.match = { canonical: false };
      result.error = "the run's projection ended in context_exhausted; the replay's did not";
      // What the replay would have sent instead (a substituted scheme's request where the run's ended).
      if (options.manifestsOut) {
        manifestsOut.push(dumpsPy({ model_call_id: call.model_call_id, manifest: projected.manifest }, { ensureAscii: false }));
      }
      if (options.stopOnMismatch) break;
      continue;
    }
    deps.annotateContextSources(projected, {
      composition, observation: (observationIndex === null ? null : history[observationIndex]) ?? {}, toolSchemas: tools,
      contextSources: settings.context_sources,
    });
    const prepared = deps.assembler.prepare(projected);
    const canonicalBytes = canonicalBytesPy(prepared.canonical);
    const wireBytes = canonicalBytesPy(prepared.wire);
    const canonicalHash = sha256(canonicalBytes);
    const wireHash = sha256(wireBytes);
    deps.assembler.noteArtifacts?.(
      { kind: "native_canonical_provider_request", media_type: "application/json", path: `native/agent-artifacts/${canonicalHash}.json`, sha256: canonicalHash, bytes: canonicalBytes.length },
      { kind: "native_provider_wire_request", media_type: "application/json", path: `native/agent-artifacts/${wireHash}.json`, sha256: wireHash, bytes: wireBytes.length },
    );
    const manifest = projected.manifest;
    if (options.manifestsOut) manifestsOut.push(dumpsPy({ model_call_id: call.model_call_id, manifest }, { ensureAscii: false }));
    const recorded = call.recorded;
    Object.assign(result, {
      canonical_request_sha256: canonicalHash,
      canonical_request_utf8_bytes: canonicalBytes.length,
      wire_request_sha256: wireHash,
      manifest_sha256: canonicalSha(manifest),
      manifest_normalized_sha256: canonicalSha(withoutCompactorClock(manifest)),
      transition_kinds: Array.isArray(manifest.projection_transitions)
        ? manifest.projection_transitions.map((item: Json) => item?.kind ?? null) : [],
      projection_transitions: clonePy(manifest.projection_transitions ?? null),
      projection_epoch: manifest.projection_epoch ?? null,
      cache_epoch: manifest.cache_epoch ?? null,
      request_relation: manifest.request_relation ?? null,
      pressure_estimated_input_tokens: manifest.pressure_estimated_input_tokens ?? null,
      raw_estimated_input_tokens: manifest.raw_estimated_input_tokens ?? null,
      recorded: {
        canonical_request_sha256: recorded.exact_request_sha256,
        canonical_request_utf8_bytes: recorded.canonical_request_utf8_bytes,
        wire_request_sha256: recorded.wire_request_sha256,
        manifest_sha256: recorded.manifest_sha256,
        projection_transitions_count: Array.isArray(recorded.projection_transitions) ? recorded.projection_transitions.length : null,
        pressure_estimated_input_tokens: recorded.pressure_estimated_input_tokens,
      },
    });
    const recordedManifest = recordedManifests[call.index]?.manifest ?? null;
    const match: Json = {
      canonical: canonicalHash === recorded.exact_request_sha256,
      wire: wireHash === recorded.wire_request_sha256,
      transitions: canonicalSha(manifest.projection_transitions ?? null) === canonicalSha(recorded.projection_transitions ?? null),
      manifest: result.manifest_sha256 === recorded.manifest_sha256,
    };
    if (recordedManifest !== null) {
      const expected = { ...recordedManifest };
      delete expected.exact_request_sha256;
      delete expected.wire_request_sha256;
      match.manifest_without_compactor_clock = canonicalSha(withoutCompactorClock(manifest)) === canonicalSha(withoutCompactorClock(expected));
      if (!match.manifest_without_compactor_clock) result.manifest_difference = firstDifference(withoutCompactorClock(manifest), withoutCompactorClock(expected));
    }
    result.match = match;
    if (!match.canonical && options.diffRequests) {
      const recordedRequest = recordedRequests.get(call.model_call_id);
      if (recordedRequest) result.request_difference = firstDifference(skeleton(prepared.canonical), recordedRequest.skeleton);
    }
    // accountCall: the Provider usage of this response anchors the next projection.
    if (call.response !== null) {
      let estimate = intValue(manifest.pressure_estimated_input_tokens);
      if (estimate === null || !Number.isInteger(estimate)) estimate = intValue(manifest.raw_estimated_input_tokens);
      const usage = call.observed_usage;
      let inputTokens: Json = usage !== null && typeof usage === "object" ? usage.input_tokens ?? null : null;
      if (usageMode === "none") inputTokens = null;
      if (inflate !== null && inputTokens !== null && call.index >= inflate.fromCall) {
        inputTokens = Math.round(intValue(inputTokens)! * inflate.factor);
      }
      if (usageMode === "scaled" && inputTokens !== null) {
        const recordedEstimate = intValue(recorded.pressure_estimated_input_tokens) ?? intValue(recorded.raw_estimated_input_tokens);
        if (recordedEstimate) inputTokens = Math.round(intValue(inputTokens)! * (estimate! / recordedEstimate));
      }
      deps.pipeline.observeProviderUsage({
        estimatedTokens: estimate!, inputTokens,
        projectionEpoch: manifest.projection_epoch ?? null,
        providerWireRequestSha256: manifest.provider_wire_request_sha256 ?? null,
      });
    }
    result.seconds = Math.round(performance.now() - callStarted) / 1000;
    log(`${call.model_call_id ?? call.kind} step ${call.observation_step}: canonical ${match.canonical ? "same" : "DIFFERENT"}`
      + ` (${canonicalBytes.length} B, ${Array.isArray(manifest.projection_transitions) ? manifest.projection_transitions.length : 0} transitions)`
      + (result.state_match === false ? ` state differs at ${result.state_difference}` : "")
      + (match.manifest_without_compactor_clock === false ? ` manifest differs at ${result.manifest_difference}` : "")
      + ` ${result.seconds}s`);
    if (options.stopOnMismatch && !match.canonical) {
      stopped = `${call.model_call_id}: canonical request differs`;
      break;
    }
  }
  provider.close?.();
  if (options.manifestsOut) {
    fs.writeFileSync(options.manifestsOut, gzipSync(Buffer.concat(manifestsOut.map((line) => Buffer.from(`${line}\n`, "utf8")))));
  }
  const modelCalls = results.filter((item) => item.kind === "model_call");
  const count = (predicate: (item: Json) => boolean) => modelCalls.filter(predicate).length;
  const compactorOutcomes = results.flatMap((item) => item.compactor ?? []);
  return {
    schema: RESULT_SCHEMA,
    replay: replay.dir,
    source: data.source,
    tree_commit: treeCommit(),
    mode,
    policy_overrides: substituted ? options.policyOverrides : null,
    inflate,
    round_split: roundSplit,
    domain_override: options.domain ? canonicalSha(options.domain) : null,
    usage: usageMode,
    stopped,
    summary: {
      recorded_model_calls: data.calls.filter((call: Json) => call.kind === "model_call").length,
      replayed_model_calls: modelCalls.length,
      canonical_match: count((item) => item.match?.canonical === true),
      wire_match: count((item) => item.match?.wire === true),
      transitions_match: count((item) => item.match?.transitions === true),
      manifest_match: count((item) => item.match?.manifest === true),
      manifest_match_without_compactor_clock: count((item) => item.match?.manifest_without_compactor_clock === true),
      state_match: count((item) => item.state_match === true),
      projection_transitions: modelCalls.reduce((sum, item) => sum + (Array.isArray(item.projection_transitions) ? item.projection_transitions.length : 0), 0),
      compactor_calls: compactorOutcomes.length,
      compactor_recorded: compactorOutcomes.filter((item: Json) => item.source === "recorded").length,
      compactor_request_match: compactorOutcomes.filter((item: Json) => item.request_match === true).length,
      compactor_stubbed: compactorOutcomes.filter((item: Json) => item.source === "stub").length,
      seconds: Math.round(performance.now() - started) / 1000,
    },
    calls: results,
  };
}

function argument(value: string): Json {
  return parsePy(value.startsWith("@") ? fs.readFileSync(value.slice(1), "utf8") : value);
}

/**
 * A scheme file (ora.context-scheme.v1: its `policy`) or a bare context_management object. A v2
 * scheme lists stages that Python resolves into its policy: pass that policy with --policy.
 */
export function schemeOverrides(file: string): Record<string, Json> {
  const scheme = parsePy(fs.readFileSync(file, "utf8"));
  if (scheme && scheme.schema === "ora.context-scheme.v2") {
    throw new Error(`${file} is an ora.context-scheme.v2 scheme: resolve it with Python `
      + "(ora.agent.composition.context_scheme.load_context_scheme) and pass its policy with --policy");
  }
  if (scheme && typeof scheme.policy === "object" && scheme.policy !== null) return scheme.policy;
  if (scheme && typeof scheme.schema === "string" && scheme.schema.startsWith("ora.native-context-management.")) {
    return { context_management: scheme };
  }
  throw new Error(`${file} is neither a context scheme nor a context_management object`);
}

async function main(argv: string[]): Promise<number> {
  const positional: string[] = [];
  const options: ReplayOptions = { log: (line) => process.stderr.write(line + "\n") };
  let out: string | null = null;
  let overrides: Record<string, Json> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`${flag} needs a value`);
      return argv[index];
    };
    if (flag === "--out") out = next();
    else if (flag === "--work") options.work = next();
    else if (flag === "--scheme") overrides = { ...overrides, ...schemeOverrides(next()) };
    else if (flag === "--management") overrides = { ...overrides, context_management: argument(next()) };
    else if (flag === "--policy") overrides = { ...overrides, ...argument(next()) };
    else if (flag === "--context-stages") options.contextStages = argument(next());
    else if (flag === "--usage") options.usage = next() as ReplayOptions["usage"];
    else if (flag === "--manifests-out") options.manifestsOut = next();
    else if (flag === "--diff-requests") options.diffRequests = true;
    else if (flag === "--stop-on-mismatch") options.stopOnMismatch = true;
    else if (flag === "--limit" || flag === "--calls") options.limit = Number(next());
    else if (flag === "--domain") options.domain = argument(next());
    else if (flag === "--inflate") options.inflate = { factor: Number(next()), fromCall: options.inflate?.fromCall ?? 0 };
    else if (flag === "--inflate-from-call") {
      const fromCall = Number(next());
      options.inflate = { factor: options.inflate?.factor ?? 1, fromCall };
    } else if (flag === "--round-split") options.roundSplit = Number(next());
    else if (flag === "--match-compactor") options.matchCompactor = true;
    else if (flag === "--quiet") options.log = () => {};
    else if (flag.startsWith("--")) throw new Error(`unknown option ${flag}`);
    else positional.push(flag);
  }
  if (positional.length !== 1 || out === null) {
    process.stderr.write("usage: node ora-pi/test/support/context_replay.ts REPLAY_DIR --out RESULT.json [options]\n");
    return 2;
  }
  options.policyOverrides = Object.keys(overrides).length ? overrides : null;
  const result = await runReplay(loadReplay(positional[0]), options);
  fs.writeFileSync(out, dumpsPy(result, { ensureAscii: false, indent: 1 }) + "\n");
  process.stdout.write(dumpsPy({ out, mode: result.mode, stopped: result.stopped, ...result.summary }, { indent: 1 }) + "\n");
  return result.stopped === null ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("context_replay.ts")) {
  main(process.argv.slice(2)).then((code) => process.exit(code), (error) => {
    process.stderr.write(`${(error as Error)?.stack ?? error}\n`);
    process.exit(1);
  });
}
