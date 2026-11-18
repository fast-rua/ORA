// Bind the ported ORA context policy (src/context/) to one OraCore.
//
// The pipeline, the request assembler, the compactor and the Context Artifact
// catalog are the TypeScript ports; this module only constructs them from the
// resolved plan Python handed over and routes their evidence through the core's
// recorder. Robot-specific defaults come from the plan, never from here.

import { clonePy } from "../pyjson.ts";
import { NativeContextPipeline } from "../context/native.ts";
import { NativeRequestAssembler } from "../context/request_assembly.ts";
import { annotateContextSources } from "../context/context_sources.ts";
import { ContextPackView, packContextManagement } from "../context/context_pack.ts";
import { RolloutCheckpointCompactor } from "../context/rollout_compactor.ts";
import { ContextArtifactCatalog, LOSSLESS_CONTEXT_RETENTION } from "../context/memory/artifacts.ts";
import { RefError } from "../context/memory/artifact_reader.ts";
import { FakeProvider } from "../provider/index.ts";
import type { ContextDeps, InitParams, OraCore } from "./session.ts";
import { restorePipeline, snapshotPipeline } from "./persistence.ts";
import { createProvider, providerFromSpec } from "./deps.ts";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parsePy } from "../pyjson.ts";

/** The request assembler's cache-stability state: identities only; bodies stay in artifacts. */
function assemblerState(assembler: Json): Json {
  return {
    previous_canonical_sha256: assembler.previousCanonical ? assembler.lastCanonicalRef ?? null : null,
    previous_canonical_ref: assembler.lastCanonicalRef ?? null,
    previous_wire_ref: assembler.lastWireRef ?? null,
    previous_epoch: assembler.previousEpoch,
    previous_protected_prefix_sha256: assembler.previousProtectedPrefixSha256,
    previous_provider_image_window_epoch: assembler.previousProviderImageWindowEpoch,
    ...(assembler.previousCameraTail ? { previous_camera_tail: true } : {}),
  };
}

function restoreAssembler(assembler: Json, state: Json, runRoot: string): void {
  const load = (ref: Json) => (ref ? parsePy(readFileSync(join(runRoot, String(ref.path)), "utf8")) : null);
  assembler.previousCanonical = load(state.previous_canonical_ref);
  assembler.previousWire = load(state.previous_wire_ref);
  assembler.lastCanonicalRef = state.previous_canonical_ref;
  assembler.lastWireRef = state.previous_wire_ref;
  assembler.previousEpoch = state.previous_epoch;
  assembler.previousProtectedPrefixSha256 = state.previous_protected_prefix_sha256;
  assembler.previousProviderImageWindowEpoch = state.previous_provider_image_window_epoch;
  assembler.previousCameraTail = state.previous_camera_tail === true;
}

type Json = any;

const FILE_NOT_FOUND_KINDS = new Set(["unknown", "storage"]);

export interface ProviderLike {
  protocol: string;
  complete(execution: Json, options: Json): Promise<Record<string, Json>>;
  prepareRequest(execution: Json): Json;
  close(): void;
}

/**
 * The rollout compactor's provider: its own instance of the plan's provider with
 * max_retries 0 (Python copies the main provider and sets max_retries=0), so a
 * failed checkpoint call is not retried while the policy's provider keeps its
 * retries. Same model, endpoint and wire options as the policy's; the HTTP
 * client (fetchImpl) is shared, as the Python copy shares its client.
 *
 * A compactor bound to its own profile (policy.context_compactor) gets the
 * provider of `config.compactor_provider` instead: its own key
 * (ORA_PI_COMPACTOR_API_KEY), its own HTTP client, no standby keys, no retries.
 */
export function compactorProvider(config: InitParams, provider: ProviderLike): ProviderLike {
  const spec = config.provider;
  if (spec.kind === "fake" || spec.kind === "replay") {
    const responses = spec.compactor_responses ?? [];
    const fake = new FakeProvider(responses);
    return Object.assign(fake as Json, { prepareRequest: provider.prepareRequest.bind(provider) });
  }
  if (config.compactor_provider) {
    return providerFromSpec(config.compactor_provider, {
      apiKey: String(process.env.ORA_PI_COMPACTOR_API_KEY ?? ""),
    }, { maxRetries: 0 }) as Json;
  }
  const fetchImpl = (provider as Json).fetchImpl as typeof fetch | undefined;
  return createProvider(config, { maxRetries: 0, ...(fetchImpl ? { fetchImpl } : {}) }) as Json;
}

export function contextDeps(core: OraCore, config: InitParams, provider: ProviderLike): ContextDeps {
  const policy = config.policy;
  const retention = policy.context_retention ?? "rewrite_compact_v1";
  const domain = config.domain as Json;
  let catalog: ContextArtifactCatalog | null = null;
  let compactor: RolloutCheckpointCompactor | null = null;
  if (retention === LOSSLESS_CONTEXT_RETENTION) {
    const readerName = domain.artifact_reader_name ?? "artifact_read";
    const indexPath = join(core.runRoot, "native/context-artifacts/index.json");
    // The catalog rewrites its index on construction; a resumed process first reads the old one.
    const previous = (config as Json).resume && existsSync(indexPath) ? parsePy(readFileSync(indexPath, "utf8")) : null;
    catalog = new ContextArtifactCatalog(core.runRoot, config.policy_session_id, core.artifacts as Json, readerName);
    if (previous !== null) {
      if (previous.policy_session_id !== config.policy_session_id) throw new Error("Context Artifact index belongs to another session");
      (catalog as Json).entries = new Map(Object.entries(previous.artifacts ?? {}));
      (catalog as Json).persist();
    }
    // policy.context_compactor: the compactor's own model, effort and stream, named in its events.
    const bound = policy.context_compactor ?? null;
    compactor = new RolloutCheckpointCompactor({
      provider: compactorProvider(config, provider) as Json,
      model: String((bound ?? policy).model),
      reasoningEffort: String((bound ?? policy).reasoning_effort),
      stream: Boolean((bound ?? policy).stream),
      artifacts: core.artifacts as Json,
      eventSink: async (type: string, fields: Json) => core.append(type, fields),
      ...(domain.context?.compactorSystemPrompt ? { systemPrompt: domain.context.compactorSystemPrompt } : {}),
      ...(bound ? { providerBinding: { provider_profile: String(bound.provider_profile), model: String(bound.model),
        endpoint: String(bound.endpoint) } } : {}),
    } as Json);
  }
  const pipeline = new NativeContextPipeline({
    profile: String(policy.context_profile),
    artifacts: core.artifacts as Json,
    contextWindowTokens: Number(policy.context_window_tokens),
    maxOutputTokens: Number(policy.max_output_tokens),
    summaryPolicy: policy.summary_policy ?? "disabled",
    summaryProvider: compactor as Json,
    retention,
    tokenEstimator: policy.context_token_estimator,
    contextManagement: policy.context_management ?? null,
    // An ora.context-scheme.v2 scheme's stage list; the pipeline checks it is the management object's.
    contextStages: policy.context_stages ?? null,
    stageSchemas: domain.context_stage_schemas ?? null,
    contextArtifacts: catalog,
    mutationTools: [...config.tools.mutation_tools],
    ...(domain.context ? { domain: domain.context } : {}),
    // policy.context_pack may turn off the layers that act on token usage.
    ...packContextManagement(policy.context_pack),
  });
  // policy.context_pack: what each projected request delivers (context/context_pack.ts).
  const pack = policy.context_pack ? new ContextPackView(policy.context_pack, core.rpc) : null;
  const assembler = new NativeRequestAssembler({
    model: String(policy.model),
    reasoningEffort: String(policy.reasoning_effort),
    maxOutputTokens: Number(policy.max_output_tokens),
    stream: Boolean(policy.stream),
    prepareRequest: config.provider.kind === "fake" ? null : (execution: Json) => provider.prepareRequest(execution),
    ...(pack !== null ? { contextPackView: true } : {}),
  });
  if (catalog !== null) {
    const registered = catalog;
    core.registerToolImages = (images: Json[], step: number) => images.map((imageRef: Json) => {
      const logical = `artifact://${config.policy_session_id}/${imageRef.sha256}`;
      let resolved: [Json, Buffer];
      try {
        resolved = registered.resolve(logical);
      } catch (error) {
        // Python: `except FileNotFoundError` (unknown refs and missing storage).
        if (!(error instanceof RefError) || !FILE_NOT_FOUND_KINDS.has((error as Json).kind)) throw error;
        return registered.registerStorageRef(imageRef, { kind: "context_image", roundIndex: null, actionStep: step });
      }
      const [entry] = resolved;
      if (entry.media_type !== imageRef.media_type) {
        throw new Error("historical tool image differs from its existing Artifact identity");
      }
      return registered.publicRef(entry, null, step);
    });
  }
  return {
    pipeline: {
      project: (args) => (pack === null ? pipeline.project(args as Json)
        : pipeline.project(args as Json).then((projected) => pack.apply(projected as Json))) as Json,
      observeProviderUsage: (args) => pipeline.observeProviderUsage(args as Json),
      compactorCalls: () => (compactor === null ? [] : clonePy(compactor.calls)),
      snapshot: () => ({
        pipeline: snapshotPipeline(pipeline),
        compactor: compactor === null ? null : { call_sequence: compactor.callSequence, calls: clonePy(compactor.calls) },
        assembler: assemblerState(assembler),
      }),
      resumeScripts: (count: number) => {
        const scripted = (compactor as Json)?.provider;
        if (scripted && typeof scripted.index === "number") scripted.index = count;
      },
      restore: (state: Json) => {
        restorePipeline(pipeline, state.pipeline);
        if (compactor !== null && state.compactor) {
          compactor.callSequence = state.compactor.call_sequence;
          compactor.calls = clonePy(state.compactor.calls);
        }
        if (state.assembler) restoreAssembler(assembler, state.assembler, core.runRoot);
      },
    },
    assembler: {
      prepare: (projected: Json) => assembler.prepare(projected) as Json,
      noteArtifacts: (canonicalRef: Json, wireRef: Json) => {
        (assembler as Json).lastCanonicalRef = canonicalRef;
        (assembler as Json).lastWireRef = wireRef;
      },
    },
    annotateContextSources: (projected, args) => annotateContextSources(projected, args as Json),
    provider,
    ...(config.compactor_provider && compactor !== null
      ? { closeCompactor: () => (compactor as Json).provider.close() } : {}),
  };
}
