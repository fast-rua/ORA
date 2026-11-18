// agent-events.jsonl writer, the TypeScript twin of
// tests/support/agent/agent_events.py::TrajectoryRecorder (the Python writer tests use).
//
// v1 events hash with ORA's Python canonical_bytes (emulated byte for byte by
// pyjson), so the unchanged Python verifier accepts them. v2 events hash with
// RFC 8785 JCS and carry monotonic_ns as a decimal string, because JCS cannot
// represent integers beyond 2**53 exactly (monotonic nanoseconds pass 2**53
// after ~104 days of host uptime).
//
// Unlike the Python recorder, this one can reopen an existing trace and
// continue its chain: a restarted agent process resumes the same run.

import { createHash } from "node:crypto";
import { closeSync, existsSync, openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";

import { jcsBytes } from "../jcs.ts";
import { canonicalBytesPy, dumpsPy, parsePy } from "../pyjson.ts";
import type { Durability } from "./durability.ts";

export const AGENT_EVENT_SCHEMA_V1 = "ora.agent-event.v1";
export const AGENT_EVENT_SCHEMA_V2 = "ora.agent-event.v2";
const VISIBILITY = new Set(["exact", "derived", "opaque"]);

export class TrajectoryError extends Error {}

export interface AppendOptions {
  visibility?: "exact" | "derived" | "opaque";
  causal_parent_id?: string | null;
  observation_step?: number | null;
  model_call_id?: string | null;
  tool_call_id?: string | null;
  payload?: unknown;
  artifact_ref?: Record<string, unknown> | null;
}

export type AgentEvent = Record<string, any> & { event_id: string; event_sha256: string; seq: number };

function sha256(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Line separators that Python's `str.splitlines()` would split inside a JSON line. */
const LINE_BREAKERS = /[\u0085\u2028\u2029]/g;

export class TrajectoryRecorder {
  readonly path: string;
  seq = 0;
  previousEventSha256: string | null = null;
  private descriptor: number;

  readonly runRoot: string;
  readonly runId: string;
  readonly policySessionId: string;
  readonly durability: Durability | null;
  readonly schema: string;

  constructor(
    runRoot: string,
    runId: string,
    policySessionId: string,
    durability: Durability | null = null,
    schema: string = AGENT_EVENT_SCHEMA_V1,
    options: { resume?: boolean } = {},
  ) {
    this.runRoot = runRoot;
    this.runId = runId;
    this.policySessionId = policySessionId;
    this.durability = durability;
    this.schema = schema;
    if (schema !== AGENT_EVENT_SCHEMA_V1 && schema !== AGENT_EVENT_SCHEMA_V2) {
      throw new TrajectoryError(`unsupported agent event schema ${schema}`);
    }
    this.path = join(runRoot, "agent-events.jsonl");
    if (existsSync(this.path)) {
      if (!options.resume) {
        throw new TrajectoryError("refusing to append to an existing agent-events.jsonl");
      }
      this.reopen();
    }
    // The run root exists (the Supervisor created it); the agent may write only this file there.
    this.descriptor = openSync(this.path, "a");
  }

  /** Continue an existing chain after verifying its tail. */
  private reopen(): void {
    const lines = readFileSync(this.path, "utf8").split("\n").filter((line) => line.trim());
    let previous: string | null = null;
    for (const [index, line] of lines.entries()) {
      const event = parsePy(line);
      if (event.seq !== index || event.previous_event_sha256 !== previous) {
        throw new TrajectoryError(`agent trace chain is broken at line ${index + 1}`);
      }
      if (event.policy_session_id !== this.policySessionId) {
        throw new TrajectoryError("agent trace belongs to another policy session");
      }
      const { event_sha256: recorded, ...body } = event;
      if (eventHash(body) !== recorded) {
        throw new TrajectoryError(`agent trace hash mismatch at line ${index + 1}`);
      }
      previous = recorded;
    }
    this.seq = lines.length;
    this.previousEventSha256 = previous;
  }

  append(eventType: string, options: AppendOptions = {}): AgentEvent {
    const visibility = options.visibility ?? "exact";
    if (!VISIBILITY.has(visibility)) throw new TrajectoryError(`invalid event visibility: ${visibility}`);
    const monotonic = process.hrtime.bigint();
    const body: Record<string, unknown> = {
      schema: this.schema,
      seq: this.seq,
      event_id: `agent-event-${String(this.seq).padStart(6, "0")}`,
      type: eventType,
      run_id: this.runId,
      policy_session_id: this.policySessionId,
      observation_step: options.observation_step ?? null,
      model_call_id: options.model_call_id ?? null,
      tool_call_id: options.tool_call_id ?? null,
      causal_parent_id: options.causal_parent_id ?? null,
      visibility,
      unix_s: (performance.timeOrigin + performance.now()) / 1000,
      monotonic_ns: this.schema === AGENT_EVENT_SCHEMA_V2 ? monotonic.toString() : monotonic,
      previous_event_sha256: this.previousEventSha256,
      payload: options.payload === undefined ? null : options.payload,
      artifact_ref: options.artifact_ref ?? null,
    };
    const event = { ...body, event_sha256: eventHash(body) } as AgentEvent;
    const line = dumpsPy(event, { sortKeys: true, ensureAscii: false })
      .replace(LINE_BREAKERS, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`);
    writeSync(this.descriptor, line + "\n");
    this.durability?.submit(this.path);
    this.previousEventSha256 = event.event_sha256;
    this.seq += 1;
    return event;
  }

  close(): void {
    closeSync(this.descriptor);
  }
}

/** The hash rule of an event body, selected by its schema. */
export function eventHash(body: Record<string, unknown>): string {
  if (body.schema === AGENT_EVENT_SCHEMA_V2) return sha256(jcsBytes(body));
  return sha256(canonicalBytesPy(body));
}
