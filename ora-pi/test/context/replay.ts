// Replays one Python Context fixture (tests/data/context_differential/, recorded from
// the retired Python Context pipeline)
// through the TypeScript pipeline in a fresh run root and reports every byte
// difference: projected messages, manifests, request assembly, compactor
// requests and events, artifact reads, the run-root file set and index.json.

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { gunzipSync } from "node:zlib";

import { ArtifactStore, NO_DURABILITY } from "../../src/evidence/artifacts.ts";
import { ContextArtifactCatalog, invokeArtifactTool } from "../../src/context/memory/artifacts.ts";
import { ToolError } from "../../src/context/memory/tool_errors.ts";
import {
  NativeContextPipeline, pipelineState, providerEstimatedInputTokens,
} from "../../src/context/native.ts";
import { NativeRequestAssembler } from "../../src/context/request_assembly.ts";
import { annotateContextSources } from "../../src/context/context_sources.ts";
import { RolloutCheckpointCompactor } from "../../src/context/rollout_compactor.ts";
import { canonicalBytesPy, clonePy, isMapping, loadsPy } from "../../src/context/pyutil.ts";
import { legacyStageScheme } from "../support/legacy_context.ts";
import { STAGE_SCHEMAS } from "../support/stage_schemas.ts";
import { PyFloat, dumpsPy } from "../../src/pyjson.ts";

/** ``recorder.skeleton``: large data URLs replaced by their sha256 and length. */
export function skeleton(value: unknown): unknown {
  if (typeof value === "string") {
    if (value.length >= 1024 && value.startsWith("data:")) {
      return `<data ${createHash("sha256").update(value, "utf8").digest("hex")} ${value.length}>`;
    }
    return value;
  }
  if (Array.isArray(value)) return value.map(skeleton);
  if (isMapping(value)) {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value)) out[key] = skeleton(value[key]);
    return out;
  }
  return value;
}

export function loadFixture(file: string): any {
  return loadsPy(gunzipSync(fs.readFileSync(file)).toString("utf8"));
}

function sha(value: unknown): string {
  return createHash("sha256").update(canonicalBytesPy(value)).digest("hex");
}

function short(value: unknown): string {
  let text: string;
  try {
    text = dumpsPy(value, { sortKeys: true, ensureAscii: false });
  } catch {
    text = String(value);
  }
  return text.length > 300 ? text.slice(0, 300) + `...(${text.length} chars)` : text;
}

/** The first differing path of two decoded JSON values (canonical equality). */
export function firstDiff(left: unknown, right: unknown, where = "$"): string | null {
  const l = left === undefined ? null : left;
  const r = right === undefined ? null : right;
  if (Array.isArray(l) && Array.isArray(r)) {
    for (let index = 0; index < Math.max(l.length, r.length); index += 1) {
      if (index >= l.length || index >= r.length) {
        return `${where}: length ${l.length} vs ${r.length}`;
      }
      const diff = firstDiff(l[index], r[index], `${where}[${index}]`);
      if (diff) return diff;
    }
    return null;
  }
  if (isMapping(l) && isMapping(r)) {
    const keys = [...new Set([...Object.keys(l), ...Object.keys(r)])].sort();
    for (const key of keys) {
      if (!(key in l) || !(key in r)) {
        return `${where}.${key}: ${key in l ? "only python" : "only typescript"} `
          + `(python=${short(l[key])}, typescript=${short(r[key])})`;
      }
      const diff = firstDiff(l[key], r[key], `${where}.${key}`);
      if (diff) return diff;
    }
    return null;
  }
  let same: boolean;
  try {
    same = canonicalBytesPy(l).equals(canonicalBytesPy(r));
  } catch {
    same = dumpsPy(l) === dumpsPy(r);
  }
  return same ? null : `${where}: python=${short(l)} typescript=${short(r)}`;
}

const hashCache = new Map<string, [string, string]>();

function filesUnder(root: string): Map<string, string> {
  const files = new Map<string, string>();
  const walk = (directory: string) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) {
        const stat = fs.statSync(full, { bigint: true });
        const identity = `${stat.ino}:${stat.size}:${stat.mtimeNs}`;
        const cached = hashCache.get(full);
        let digest: string;
        if (cached && cached[0] === identity) digest = cached[1];
        else {
          digest = createHash("sha256").update(fs.readFileSync(full)).digest("hex");
          hashCache.set(full, [identity, digest]);
        }
        files.set(path.relative(root, full).split(path.sep).join("/"), digest);
      }
    }
  };
  walk(root);
  return new Map([...files.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)));
}

export interface ReplayReport {
  name: string;
  ops: number;
  differences: string[];
}

const ERROR_NAMES: Record<string, string> = {
  PolicySessionError: "PolicySessionError",
  ContextCompactionRejected: "ContextCompactionRejected",
  ToolError: "ToolError",
  ValueError: "ValueError",
  KeyError: "KeyError",
};

export async function replayFixture(fixture: any, options: { keepRoot?: boolean } = {}): Promise<ReplayReport> {
  const differences: string[] = [];
  const name: string = fixture.name;
  const diff = (label: string, python: unknown, typescript: unknown): void => {
    const found = firstDiff(python, typescript);
    if (found) differences.push(`${label}: ${found}`);
  };
  const root = fs.mkdtempSync(path.join(os.tmpdir(), `ctx-ts-${name}-`));
  try {
    const config = fixture.config;
    const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
    const lossless = config.pipeline.retention === "action_step_lossless_v4";
    const catalog = lossless
      ? new ContextArtifactCatalog(root, config.policy_session_id, artifacts, config.reader_name) : null;
    let pendingCompactorCalls: any[] = [];
    let compactorLabel = "";
    const events: Record<string, any>[] = [];
    const eventSink = (type: string, fields: Record<string, any>) => {
      const eventId = `agent-event-${String(500000 + events.length).padStart(6, "0")}`;
      events.push({ type, ...clonePy(fields), event_id: eventId });
      return { event_id: eventId };
    };
    const compactor = config.compactor ? new RolloutCheckpointCompactor({
      provider: {
        protocol: config.compactor.protocol,
        async complete(request, { modelCallId }) {
          const recorded = pendingCompactorCalls.shift();
          if (recorded === undefined) {
            differences.push(`${compactorLabel}: TypeScript made an extra compactor call ${modelCallId}`);
            throw new Error("no recorded compactor response");
          }
          if (recorded.model_call_id !== modelCallId) {
            differences.push(`${compactorLabel}: compactor call id ${recorded.model_call_id} vs ${modelCallId}`);
          }
          if (sha(request) !== recorded.request_sha256) {
            differences.push(`${compactorLabel}: compactor request sha differs`);
          }
          diff(`${compactorLabel} compactor request`, recorded.request, skeleton(request));
          return clonePy(recorded.response);
        },
      },
      model: config.compactor.model,
      reasoningEffort: config.compactor.reasoning_effort,
      stream: config.compactor.stream,
      maxOutputTokens: config.compactor.max_output_tokens,
      artifacts,
      eventSink,
    }) : null;
    const pipelineConfig = config.pipeline;
    // v4 and v5 run as stage schemes (the core refuses them as plans; test/support/legacy_context.ts).
    const legacyScheme = legacyStageScheme(pipelineConfig.context_management);
    const pipeline = new NativeContextPipeline({
      profile: pipelineConfig.profile,
      artifacts,
      contextWindowTokens: pipelineConfig.context_window_tokens,
      maxOutputTokens: pipelineConfig.max_output_tokens,
      retention: pipelineConfig.retention,
      summaryPolicy: pipelineConfig.summary_policy,
      summaryProvider: compactor,
      contextManagement: legacyScheme === null ? pipelineConfig.context_management : null,
      stageScheme: legacyScheme,
      stageSchemas: STAGE_SCHEMAS,
      contextArtifacts: catalog,
      mutationTools: pipelineConfig.mutation_tools,
      spillThresholdBytes: pipelineConfig.spill_threshold_bytes,
      previewBytes: pipelineConfig.preview_bytes,
    });
    let recordedWire: { execution_sha256: string; wire: any } | null = null;
    let wireLabel = "";
    const assembler = config.assembler ? new NativeRequestAssembler({
      model: config.assembler.model,
      reasoningEffort: config.assembler.reasoning_effort,
      maxOutputTokens: config.assembler.max_output_tokens,
      stream: config.assembler.stream,
      prepareRequest: config.assembler.wire === "recorded" ? (execution) => {
        if (recordedWire === null) throw new Error("no recorded wire request");
        if (sha(execution) !== recordedWire.execution_sha256) {
          differences.push(`${wireLabel}: execution request sha differs`);
        }
        return clonePy(recordedWire.wire);
      } : null,
    }) : null;

    let previousMessages: any[] = [];
    let previousTools: any[] = [];
    let lastManifest: Record<string, any> | null = null;
    const pythonFiles = new Map<string, string>();

    for (const [index, op] of fixture.ops.entries()) {
      const label = `${name} op#${index} ${op.op}`;
      compactorLabel = label;
      wireLabel = label;
      const eventsBefore = events.length;
      pendingCompactorCalls = [...(op.compactor_calls ?? [])];
      let error: unknown = null;
      if (op.op === "project") {
        const messages = [...previousMessages.slice(0, op.messages.base), ...op.messages.append];
        previousMessages = messages;
        const tools = op.tools === "same" ? previousTools : op.tools;
        previousTools = tools;
        let projected: any = null;
        try {
          projected = op.allow_transition
            ? await pipeline.project({ messages, tools, observationStep: op.observation_step })
            : await pipeline.projectActionStepLosslessV4({
              messages: clonePy(messages), tools, observationStep: op.observation_step, allowTransition: false,
            });
        } catch (caught) {
          error = caught;
        }
        lastManifest = projected ? projected.manifest : null;
        if (projected && !op.error) {
          diff(`${label} messages`, op.result.messages, skeleton(projected.messages));
          diff(`${label} manifest`, op.result.manifest, projected.manifest);
          if (sha(projected.messages) !== op.result.messages_sha256) {
            differences.push(`${label}: projected messages sha differs`);
          }
          if (op.assembler) {
            recordedWire = op.assembler.error ? null : op.assembler;
            let prepared: any = null;
            let assemblyError: unknown = null;
            try {
              prepared = assembler!.prepare(projected);
            } catch (caught) {
              assemblyError = caught;
            }
            if (op.assembler.error) {
              const message = assemblyError instanceof Error ? assemblyError.message : null;
              if (message !== op.assembler.error.message) {
                differences.push(`${label} assembler error: python=${op.assembler.error.message} typescript=${message}`);
              }
            } else if (assemblyError) {
              differences.push(`${label} assembler raised ${(assemblyError as Error).message}`);
            } else {
              if (sha(prepared.execution) !== op.assembler.execution_sha256) {
                differences.push(`${label}: execution request sha differs`);
              }
              if (sha(prepared.canonical) !== op.assembler.canonical_sha256) {
                differences.push(`${label}: canonical request sha differs`);
              }
              if (sha(prepared.wire) !== op.assembler.wire_sha256) differences.push(`${label}: wire sha differs`);
              diff(`${label} wire`, op.assembler.wire, config.assembler.wire === "recorded" ? prepared.wire : skeleton(prepared.wire));
              diff(`${label} tool_choice_evidence`, op.assembler.tool_choice_evidence, prepared.toolChoiceEvidence);
              diff(`${label} manifest_after`, op.assembler.manifest_after, projected.manifest);
            }
          }
        }
      } else if (op.op === "observe_usage") {
        const args = op.args;
        if (op.derived && lastManifest !== null) {
          const derived: Record<string, unknown> = {
            estimated_tokens: providerEstimatedInputTokens(lastManifest),
            projection_epoch: lastManifest.projection_epoch ?? null,
            provider_wire_request_sha256: lastManifest.provider_wire_request_sha256 ?? null,
          };
          if (!("projection_epoch" in args) && derived.projection_epoch === null) delete derived.projection_epoch;
          if (!("provider_wire_request_sha256" in args) && derived.provider_wire_request_sha256 === null) {
            delete derived.provider_wire_request_sha256;
          }
          const expected = { ...args };
          delete expected.input_tokens;
          diff(`${label} derived usage args`, expected, derived);
        }
        try {
          pipeline.observeProviderUsage({
            estimatedTokens: args.estimated_tokens,
            inputTokens: args.input_tokens,
            projectionEpoch: args.projection_epoch,
            providerWireRequestSha256: args.provider_wire_request_sha256,
          });
        } catch (caught) {
          error = caught;
        }
      } else if (op.op === "artifact_put") {
        try {
          const result = artifacts.putBytes(Buffer.from(op.data_b64, "base64"), {
            kind: op.kind, suffix: op.suffix, mediaType: op.media_type,
          });
          diff(`${label} result`, op.result, result);
        } catch (caught) {
          error = caught;
        }
      } else if (op.op === "register_storage_ref") {
        try {
          const result = catalog!.registerStorageRef(op.storage_ref, {
            kind: op.kind, roundIndex: op.round_index, actionStep: op.action_step,
          });
          diff(`${label} result`, op.result, result);
        } catch (caught) {
          error = caught;
        }
      } else if (op.op === "catalog_put") {
        try {
          const result = catalog!.putBytes(Buffer.from(op.data_b64, "base64"), {
            kind: op.kind, suffix: op.suffix, mediaType: op.media_type,
            roundIndex: op.round_index, actionStep: op.action_step,
          });
          diff(`${label} result`, op.result, result);
        } catch (caught) {
          error = caught;
        }
      } else if (op.op === "artifact_tool") {
        const records: any[] = [];
        try {
          const result = invokeArtifactTool(catalog!, op.name, op.arguments, (payload) => records.push(payload));
          if (op.result) diff(`${label} result`, op.result, result);
        } catch (caught) {
          error = caught;
          if (op.error?.as_dict && caught instanceof ToolError) {
            diff(`${label} tool error`, op.error.as_dict, caught.asDict());
          }
        }
        diff(`${label} retrieval records`, op.retrieval,
          records.map((record) => {
            const copy: Record<string, unknown> = { ...record };
            delete copy.duration_ms;
            return copy;
          }));
        for (const record of records) {
          if (!(record.duration_ms instanceof PyFloat) && typeof record.duration_ms !== "number") {
            differences.push(`${label}: duration_ms is not a float`);
          }
        }
      } else if (op.op === "annotate_context_sources") {
        const projected = { messages: op.messages, manifest: clonePy(op.manifest_before) };
        try {
          annotateContextSources(projected, {
            composition: op.composition ?? null, observation: op.observation,
            toolSchemas: op.tool_schemas, contextSources: op.context_sources,
          });
          diff(`${label} manifest`, op.manifest_after, projected.manifest);
        } catch (caught) {
          error = caught;
        }
      } else if (op.op === "tamper") {
        fs.writeFileSync(path.join(root, op.path), Buffer.from(op.data_b64, "base64"));
      } else if (op.op === "unlink") {
        fs.unlinkSync(path.join(root, op.path));
      } else {
        differences.push(`${label}: unknown op`);
      }

      const pythonError = op.error ?? null;
      const tsMessage = error === null ? null : error instanceof Error ? error.message : String(error);
      if ((pythonError === null) !== (error === null) || (pythonError && pythonError.message !== tsMessage)) {
        differences.push(`${label} error: python=${pythonError ? `${pythonError.type}: ${pythonError.message}` : "none"} `
          + `typescript=${error === null ? "none" : `${(error as Error).name}: ${tsMessage}`}`
          + (error instanceof Error && !pythonError ? `\n${error.stack}` : ""));
      } else if (pythonError && error instanceof Error && ERROR_NAMES[pythonError.type]
        && ERROR_NAMES[pythonError.type] !== error.name) {
        differences.push(`${label} error type: python=${pythonError.type} typescript=${error.name}`);
      }
      if (pendingCompactorCalls.length) {
        differences.push(`${label}: TypeScript skipped ${pendingCompactorCalls.length} compactor call(s)`);
      }
      diff(`${label} events`, op.events ?? [], events.slice(eventsBefore));

      // Run-root files after the op.
      for (const [file, digest] of Object.entries(op.files.changed as Record<string, string>)) {
        pythonFiles.set(file, digest);
      }
      for (const file of op.files.removed as string[]) pythonFiles.delete(file);
      const tsFiles = filesUnder(root);
      const pythonDigest = op.files.digest;
      const tsDigest = sha([...tsFiles.entries()]);
      if (pythonDigest !== tsDigest) {
        const missing = [...pythonFiles.keys()].filter((file) => !tsFiles.has(file));
        const extra = [...tsFiles.keys()].filter((file) => !pythonFiles.has(file));
        const changed = [...pythonFiles.keys()].filter((file) => tsFiles.has(file) && tsFiles.get(file) !== pythonFiles.get(file));
        differences.push(`${label} files: missing=${short(missing)} extra=${short(extra)} changed=${short(changed)}`);
      }
      if (differences.length > 40) break;
    }

    const final = fixture.final;
    const tsFiles = Object.fromEntries(filesUnder(root));
    diff(`${name} final files`, final.files, tsFiles);
    if (final.index_text !== null) {
      const index = fs.readFileSync(path.join(root, "native/context-artifacts/index.json"), "utf8");
      if (index !== final.index_text) {
        differences.push(`${name} index.json bytes differ: ${firstDiff(loadsPy(final.index_text), loadsPy(index))}`);
      }
    }
    diff(`${name} final events`, final.events, events);
    diff(`${name} pipeline state`, final.pipeline_state, pipelineState(pipeline));
    return { name, ops: fixture.ops.length, differences };
  } finally {
    if (!options.keepRoot) fs.rmSync(root, { recursive: true, force: true });
  }
}
