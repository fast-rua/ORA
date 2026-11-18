// Ported from `ora/agent/context/native.py` of the retired Python Native core
// (robot-harness repository): the native_api Context pipeline of one policy
// session.
//
// Lossless retention (`action_step_lossless_v4`) runs on the stage driver
// (stages/driver.ts): management v6 and v8 are translated into stages
// (stages/scheme.ts); the pipeline validates the configuration and delegates
// projection, Provider usage and its state record. Management v4, v5 and v7
// are refused here (sealed runs that used them still evaluate in Python); a
// caller may hand over a stage scheme of its own (`stageScheme`). The legacy
// `rewrite_compact_v1` projection, the default of a plan that names no
// retention, stays below.
//
// `project()` is async because L5/L6 may call the compactor (a provider call).

import type { ArtifactStore } from "../evidence/artifacts.ts";
import {
  type ContextDomain, type ContextDomainSpec, DEFAULT_CONTEXT_DOMAIN, resolveContextDomain,
} from "./domain_defaults.ts";
import { InlineMediaCache } from "./inline_media.ts";
import { ContextManagementConfigError, validateContextManagement } from "./management.ts";
import { ContextArtifactCatalog, LOSSLESS_CONTEXT_RETENTION } from "./memory/artifacts.ts";
import {
  CONTEXT_PROFILES, CONTEXT_RETENTIONS, type Message, SUMMARY_POLICIES, causalMessageViolations,
  compactOldCausalObservations, estimatedTokens, excludeReasoning, jsonBytesOf,
  messageEventIds, providerMessage, spillLargeToolResults, stripHistoricalImages, pressureEstimatedTokens,
  inlineImageCount, IMAGE_PRESSURE_ESTIMATE_TOKENS,
} from "./projection.ts";
import {
  PolicySessionError, canonicalBytesPy, canonicalLength, canonicalSha256, clonePy,
  eq, get, hasKey, isInt, isMapping, num, sortedStrings,
} from "./pyutil.ts";
import { passRecord } from "./request_size.ts";
import { StageDriver } from "./stages/driver.ts";
import { EMERGENCY_CHECKPOINT, EPISODE_CAPSULE } from "./stages/names.ts";
import { sourceSha256 } from "./stages/request_identity.ts";
import { type Limits, RESERVE_TOKENS, resolveLimits } from "./stages/services/limits.ts";
import { type StageScheme, schemeFromManagement, schemeFromStageList } from "./stages/scheme.ts";
import { checkUnownedFields, withInitialFields } from "./stages/state_fields.ts";

export { ESTIMATE_PRESSURE_WINDOW_SHARE } from "./stages/pressure_externalize/trigger.ts";

/** ``ora.agent.contracts.SummaryResult``. */
export interface SummaryResult {
  content: string;
  sourceRefs: Record<string, any>[];
  sourceSetSha256: string;
  metadata: Record<string, any>;
}

/** ``ora.agent.contracts.SummaryProvider`` (the rollout compactor). */
export interface SummaryProvider {
  summarize(args: { messages: Message[]; sourceRefs: Record<string, any>[] }): Promise<SummaryResult>;
  checkpointTargetEstimatedTokens?: number;
}

export interface ProjectedRequest {
  messages: Message[];
  tools: Record<string, any>[];
  manifest: Record<string, any>;
}

export interface NativeContextPipelineOptions {
  profile: string;
  artifacts: ArtifactStore;
  contextWindowTokens: number;
  maxOutputTokens: number;
  summaryPolicy?: string;
  summaryProvider?: SummaryProvider | null;
  spillThresholdBytes?: number;
  previewBytes?: number;
  retention?: string;
  /** Opt-in image-aware pressure for rewrite retention; legacy estimates stay frozen. */
  tokenEstimator?: string;
  contextManagement?: Record<string, any> | null;
  /**
   * Lossless retention: run this stage scheme instead of translating
   * ``contextManagement`` (which must then be absent). The manifest echoes the
   * scheme's ``source``.
   */
  stageScheme?: StageScheme | null;
  /**
   * The plan's stage list (`policy.context_stages`, from an ora.context-scheme.v2
   * scheme): run it, after checking it is the stage list of ``contextManagement``.
   */
  contextStages?: Record<string, any> | null;
  /**
   * Each stage's parameter schema and the scheme settings' (ora/policy/context_stages,
   * sent by Python at session start): every stage is validated with them. Lossless
   * retention needs them; the core keeps no copy of the rules.
   */
  stageSchemas?: Record<string, any> | null;
  contextArtifacts?: ContextArtifactCatalog | null;
  /** Defaults to ``domain.mutationTools``. */
  mutationTools?: string[];
  /** Robot rules (see domain_defaults.ts); defaults are the Python values. */
  domain?: ContextDomain | Partial<ContextDomainSpec>;
  /**
   * Lossless retention's layers that act on token usage, both on by default.
   * A context pack (``policy.context_pack``, context_pack.ts) may turn off L4
   * pressure externalization and L5/L6 rollout compaction, so that only the
   * history's own steps shape what its compact view delivers. They become the
   * `enabled` parameters of those stages.
   */
  pressureExternalization?: boolean;
  rolloutCompaction?: boolean;
}

/** The pipeline-state fields `rewrite_compact_v1` keeps: its usage anchor and epoch. */
const REWRITE_FIELDS = [
  "lastEstimatedTokens", "lastProviderInputTokens", "lastProviderProjectionEpoch",
  "lastProviderWireRequestSha256", "cacheEpoch",
];

/** The compactor's checkpoint target: the summarize stages' parameter. */
function checkpointTarget(scheme: StageScheme): number | null {
  const spec = scheme.stages.find((item) => item.stage === EPISODE_CAPSULE || item.stage === EMERGENCY_CHECKPOINT);
  return spec === undefined ? null : spec.params.checkpoint_target_tokens;
}

export class NativeContextPipeline {
  profile: string;
  artifacts: ArtifactStore;
  contextWindowTokens: number;
  maxOutputTokens: number;
  summaryPolicy: string;
  summaryProvider: SummaryProvider | null;
  spillThresholdBytes: number;
  previewBytes: number;
  retention: string;
  contextManagement: Record<string, any> | null;
  contextArtifacts: ContextArtifactCatalog | null;
  mutationTools: string[];
  domain: ContextDomain;
  /** The window's limits (stages/services/limits.ts); the reserve is the scheme's, or the default. */
  readonly limits: Limits;

  // rewrite_compact_v1's usage anchor; the stage driver keeps its own.
  lastEstimatedTokens: number | null = null;
  lastProviderInputTokens: number | null = null;
  lastProviderProjectionEpoch: number | null = null;
  lastProviderWireRequestSha256: string | null = null;
  cacheEpoch = 0;
  /** Verified payload admissions, lent to the stage driver; never evidence state. */
  inlineMedia = new InlineMediaCache();
  /** L4, and L5/L6 (see ``NativeContextPipelineOptions``). */
  readonly pressureExternalization: boolean;
  readonly rolloutCompaction: boolean;
  readonly tokenEstimator: string | undefined;
  /** The stage driver of lossless retention; null for `rewrite_compact_v1`. */
  readonly driver: StageDriver | null = null;

  constructor(options: NativeContextPipelineOptions) {
    this.profile = options.profile;
    this.artifacts = options.artifacts;
    this.contextWindowTokens = options.contextWindowTokens;
    this.maxOutputTokens = options.maxOutputTokens;
    this.summaryPolicy = options.summaryPolicy ?? "disabled";
    this.summaryProvider = options.summaryProvider ?? null;
    this.spillThresholdBytes = options.spillThresholdBytes ?? 16 * 1024;
    this.previewBytes = options.previewBytes ?? 4 * 1024;
    this.retention = options.retention ?? "rewrite_compact_v1";
    this.tokenEstimator = options.tokenEstimator;
    if (this.tokenEstimator !== undefined && (this.tokenEstimator !== "image_aware_v1"
      || this.retention !== "rewrite_compact_v1")) {
      throw new PolicySessionError("context_token_estimator=image_aware_v1 requires rewrite_compact_v1");
    }
    this.contextManagement = options.contextManagement ?? null;
    this.contextArtifacts = options.contextArtifacts ?? null;
    this.pressureExternalization = options.pressureExternalization ?? true;
    this.rolloutCompaction = options.rolloutCompaction ?? true;
    const domain = options.domain;
    this.domain = domain === undefined ? DEFAULT_CONTEXT_DOMAIN
      : "checkpointForbidden" in domain ? domain as ContextDomain : resolveContextDomain(domain);
    this.mutationTools = [...(options.mutationTools ?? this.domain.mutationTools)];

    if (!CONTEXT_PROFILES.has(this.profile)) {
      throw new PolicySessionError(`unsupported native context profile: ${this.profile}`);
    }
    if (!CONTEXT_RETENTIONS.has(this.retention)) {
      throw new PolicySessionError(`unsupported native context retention: ${this.retention}`);
    }
    if (!SUMMARY_POLICIES.has(this.summaryPolicy)) {
      throw new PolicySessionError(`unsupported summary policy: ${this.summaryPolicy}`);
    }
    let scheme: StageScheme | null = options.stageScheme ?? null;
    if (this.retention === LOSSLESS_CONTEXT_RETENTION) {
      if (this.summaryPolicy !== "rollout_checkpoint_v1") {
        throw new PolicySessionError("action_step_lossless_v4 requires summary_policy=rollout_checkpoint_v1");
      }
      if (scheme !== null) {
        if (this.contextManagement !== null) {
          throw new PolicySessionError("a stage scheme replaces context_management; give one of them");
        }
        this.contextManagement = scheme.source;
      } else {
        let raw: Record<string, any>;
        try {
          raw = validateContextManagement(this.contextManagement);
        } catch (error) {
          if (error instanceof ContextManagementConfigError) throw new PolicySessionError(error.message);
          throw error;
        }
        this.contextManagement = raw;
        const switches = {
          pressureExternalization: this.pressureExternalization, rolloutCompaction: this.rolloutCompaction,
        };
        scheme = schemeFromManagement(raw, switches);
        const listed = options.contextStages ?? null;
        if (listed !== null) {
          // The plan's stage list runs; the evidence reads the management object, so they must agree.
          const given = schemeFromStageList(listed, raw, switches);
          if (!canonicalBytesPy({ ...given, source: null }).equals(canonicalBytesPy({ ...scheme, source: null }))) {
            throw new PolicySessionError("policy.context_stages is not the stage list of policy.context_management");
          }
          scheme = given;
        }
      }
      if (options.stageSchemas === undefined || options.stageSchemas === null) {
        throw new PolicySessionError("lossless Context needs the context stage schemas (domain.context_stage_schemas)");
      }
      if (this.summaryProvider === null) throw new PolicySessionError("lossless Context fallback lacks a compactor");
      if (this.contextArtifacts === null) throw new PolicySessionError("lossless Context lacks an Artifact catalog");
      const target = checkpointTarget(scheme);
      if (target !== null && "checkpointTargetEstimatedTokens" in this.summaryProvider) {
        this.summaryProvider.checkpointTargetEstimatedTokens = target;
      }
    } else if (this.contextManagement !== null || scheme !== null) {
      throw new PolicySessionError("context_management is supported only by action_step_lossless_v4");
    }
    this.limits = resolveLimits(this.contextWindowTokens, this.maxOutputTokens,
      scheme === null ? RESERVE_TOKENS : scheme.limits.reserve_tokens);
    if (scheme !== null) {
      this.driver = new StageDriver(scheme, {
        artifacts: this.artifacts, catalog: this.contextArtifacts!, summaryProvider: this.summaryProvider!,
        domain: this.domain, mutationTools: this.mutationTools, inlineMedia: this.inlineMedia,
      }, this.limits, { profile: this.profile, retention: this.retention, summaryPolicy: this.summaryPolicy },
      options.stageSchemas!);
    }
  }

  get targetInputTokens(): number {
    return this.limits.targetInputTokens;
  }

  get hardInputTokens(): number {
    return this.limits.hardInputTokens;
  }

  async project(args: { messages: Message[]; tools: Record<string, any>[]; observationStep: number }): Promise<ProjectedRequest> {
    const working = args.messages.map((message) => clonePy({ ...message }));
    if (this.driver !== null) {
      return this.driver.project({
        messages: working, tools: args.tools, observationStep: args.observationStep, allowTransition: true,
      });
    }
    return this.projectRewriteCompact(working, args.tools, args.observationStep);
  }

  /** One lossless projection; ``allowTransition: false`` records no transition (differential replays). */
  async projectActionStepLosslessV4(args: {
    messages: Message[]; tools: Record<string, any>[]; observationStep: number; allowTransition?: boolean;
  }): Promise<ProjectedRequest> {
    if (this.driver === null) throw new PolicySessionError("projectActionStepLosslessV4 requires action_step_lossless_v4");
    return this.driver.project({ ...args, allowTransition: args.allowTransition ?? true });
  }

  // -- Legacy rewrite_compact_v1 ---------------------------------------------------

  private projectRewriteCompact(
    messages: Message[], tools: Record<string, any>[], observationStep: number,
  ): ProjectedRequest {
    let working = messages;
    const passes: Record<string, any>[] = [];
    const causal = this.profile === "native_causal_context_v1";

    let before = clonePy(working);
    let imageChanges: number[];
    [working, imageChanges] = stripHistoricalImages(
      working, observationStep, causal ? this.domain.legacyCausalImageText : null, this.domain,
    );
    passes.push(passRecord("historical_image_index", before, working, {
      changed_messages: imageChanges, replaced_event_ids: messageEventIds(working, imageChanges),
    }));

    if (causal) {
      before = clonePy(working);
      let compacted: number[];
      [working, compacted] = compactOldCausalObservations(working, observationStep, this.domain);
      passes.push(passRecord("causal_observation_allowlist", before, working, {
        changed_messages: compacted, replaced_event_ids: messageEventIds(working, compacted),
      }));
    }

    before = clonePy(working);
    let spillRefs: Record<string, any>[];
    [working, spillRefs] = spillLargeToolResults(working, {
      artifacts: this.artifacts, thresholdBytes: this.spillThresholdBytes, previewBytes: this.previewBytes,
      domain: this.domain,
    });
    const spillIndexes: number[] = [];
    working.forEach((message, index) => {
      const ref = get(message, "_artifact_ref");
      if (isMapping(ref) && spillRefs.some((item) => eq(item, ref))) spillIndexes.push(index);
    });
    passes.push(passRecord("l1_spill", before, working, {
      artifact_refs: spillRefs, externalized_event_ids: messageEventIds(working, spillIndexes),
    }));

    before = clonePy(working);
    let reasoningChanges: number[];
    [working, reasoningChanges] = excludeReasoning(working);
    passes.push(passRecord("reasoning_exclusion", before, working, {
      changed_messages: reasoningChanges, replaced_event_ids: messageEventIds(working, reasoningChanges),
    }));
    const providerMessages = working.map(providerMessage);
    if (causal) {
      const violations = causalMessageViolations(providerMessages, this.domain);
      if (violations.length) {
        throw new PolicySessionError("causal_context_forbidden_fields: " + violations.slice(0, 12).join(", "));
      }
    }
    const requestView = { messages: providerMessages, tools: [...tools] };
    const rawEstimated = estimatedTokens(requestView);
    const pressure = this.tokenEstimator === "image_aware_v1"
      ? pressureEstimatedTokens(requestView) : rawEstimated;
    const calibrationDelta = this.lastProviderInputTokens !== null && this.lastEstimatedTokens !== null
      ? this.lastProviderInputTokens - this.lastEstimatedTokens : 0;
    const estimated = Math.max(1, pressure + calibrationDelta);
    if (estimated > this.hardInputTokens) {
      throw new PolicySessionError(
        `context_exhausted: estimated ${estimated} tokens, hard limit ${this.hardInputTokens}`,
      );
    }
    const stablePrefix: Message[] = [];
    for (const message of providerMessages) {
      if (get(message, "role") !== "system") break;
      stablePrefix.push(message);
    }
    const stablePrefixHash = canonicalSha256({ messages: stablePrefix, tools: [...tools] });
    const sourceMessages: Record<string, Message> = {};
    for (const message of working) {
      const source = get(message, "_context_source");
      if (typeof source === "string") sourceMessages[source] = providerMessage(message);
    }
    const protectedIndexes = new Set<number>();
    working.forEach((message, index) => {
      if (get(message, "_protected_context") === true) protectedIndexes.add(index);
    });
    const historyMessages = providerMessages.filter((_message, index) => !protectedIndexes.has(index));
    return {
      messages: providerMessages,
      tools: [...tools],
      manifest: {
        schema: "ora.native-context-projection.v1",
        profile: this.profile,
        summary_policy: this.summaryPolicy,
        observation_step: observationStep,
        target_input_tokens: this.targetInputTokens,
        hard_input_tokens: this.hardInputTokens,
        estimated_input_tokens: estimated,
        raw_estimated_input_tokens: rawEstimated,
        ...(this.tokenEstimator === "image_aware_v1" ? {
          context_token_estimator: this.tokenEstimator,
          pressure_estimated_input_tokens: pressure,
          estimated_image_tokens: inlineImageCount(requestView) * IMAGE_PRESSURE_ESTIMATE_TOKENS,
        } : {}),
        provider_usage_anchor_tokens: this.lastProviderInputTokens,
        estimate_calibration_delta_tokens: calibrationDelta,
        request_utf8_bytes: jsonBytesOf(requestView),
        stable_prefix_hash: stablePrefixHash,
        stable_system_sha256: hasKey(sourceMessages, "harness_system")
          ? canonicalSha256(sourceMessages.harness_system) : stablePrefixHash,
        project_context_sha256: sourceSha256(sourceMessages, "project_instructions"),
        episode_task_sha256: sourceSha256(sourceMessages, "episode_task"),
        tool_schema_sha256: canonicalSha256([...tools]),
        history_sha256: canonicalSha256(historyMessages),
        history_utf8_bytes: canonicalLength(historyMessages),
        request_sha256: canonicalSha256(requestView),
        protected_context_sources: sortedStrings(Object.keys(sourceMessages)),
        causal_envelope: causal ? "ora.causal-observation.v1" : null,
        causal_forbidden_field_scan: causal ? "passed" : null,
        cache_epoch: 0,
        passes,
      },
    };
  }

  /** Anchor estimates to exact Provider usage for the request just sent. */
  observeProviderUsage(args: {
    estimatedTokens: unknown; inputTokens: unknown; projectionEpoch?: unknown;
    providerWireRequestSha256?: unknown;
  }): void {
    if (this.driver !== null) {
      this.driver.observeProviderUsage(args);
      return;
    }
    const estimated = args.estimatedTokens;
    const inputTokens = args.inputTokens;
    let projectionEpoch = args.projectionEpoch ?? null;
    const wireSha = args.providerWireRequestSha256 ?? null;
    if (!isInt(estimated) || num(estimated) < 1) {
      throw new PolicySessionError("Provider usage estimate anchor is invalid");
    }
    if (!(isInt(inputTokens) && num(inputTokens) >= 0)) return;
    if (projectionEpoch === null) projectionEpoch = this.cacheEpoch;
    if (!isInt(projectionEpoch) || num(projectionEpoch) < 0) {
      throw new PolicySessionError("Provider usage projection epoch is invalid");
    }
    if (wireSha !== null && (typeof wireSha !== "string" || !/^[0-9a-f]{64}$/.test(wireSha))) {
      throw new PolicySessionError("Provider usage wire request identity is invalid");
    }
    this.lastEstimatedTokens = num(estimated);
    this.lastProviderInputTokens = num(inputTokens);
    this.lastProviderProjectionEpoch = num(projectionEpoch);
    this.lastProviderWireRequestSha256 = wireSha as string | null;
  }

  /** The state record in the snapshot's fields (core/persistence.ts `PIPELINE_STATE_FIELDS`). */
  pipelineFields(): Record<string, any> {
    if (this.driver !== null) return this.driver.pipelineFields();
    const owned: Record<string, unknown> = {};
    for (const field of REWRITE_FIELDS) owned[field] = (this as Record<string, any>)[field];
    return withInitialFields(owned);
  }

  /** Load the state from those fields; refuse a field this pipeline does not keep that moved. */
  loadPipelineFields(fields: Record<string, any>): void {
    if (this.driver !== null) {
      this.driver.loadPipelineFields(fields);
      return;
    }
    for (const field of REWRITE_FIELDS) (this as Record<string, any>)[field] = fields[field];
    checkUnownedFields(fields, new Set(REWRITE_FIELDS), "rewrite_compact_v1");
  }
}

export type { ContextArtifactCatalog };

/** ``tool_loop._provider_estimated_input_tokens``: the estimate a usage anchor binds. */
export function providerEstimatedInputTokens(manifest: Record<string, any>): number {
  let value = get(manifest, "pressure_estimated_input_tokens");
  if (!isInt(value)) value = get(manifest, "raw_estimated_input_tokens");
  if (!isInt(value) || num(value) < 1) {
    throw new PolicySessionError("native Context projection lacks a positive input-token estimate");
  }
  return num(value);
}

/** Every lifecycle field, Python-named, for evidence and differential checks. */
export function pipelineState(target: NativeContextPipeline): Record<string, unknown> {
  const pipeline = target.pipelineFields();
  const retry: Record<string, unknown> = {};
  for (const [key, value] of pipeline.episodeRetryState as Map<string, Record<string, unknown>>) retry[key] = { ...value };
  return clonePy({
    last_estimated_tokens: pipeline.lastEstimatedTokens,
    last_provider_input_tokens: pipeline.lastProviderInputTokens,
    last_provider_projection_epoch: pipeline.lastProviderProjectionEpoch,
    last_provider_wire_request_sha256: pipeline.lastProviderWireRequestSha256,
    pending_semantic_target_epoch: pipeline.pendingSemanticTargetEpoch,
    pending_semantic_target_tokens: pipeline.pendingSemanticTargetTokens,
    last_semantic_target_validation: pipeline.lastSemanticTargetValidation,
    cache_epoch: pipeline.cacheEpoch,
    absorbed_action_step_count: pipeline.absorbedActionStepCount,
    lifecycle_media_externalized_action_step_count: pipeline.lifecycleMediaExternalizedActionStepCount,
    lifecycle_read_only_externalized_action_step_count: pipeline.lifecycleReadOnlyExternalizedActionStepCount,
    lifecycle_superseded_tool_image_event_ids: pipeline.lifecycleSupersededToolImageEventIds,
    pressure_externalized_action_step_count: pipeline.pressureExternalizedActionStepCount,
    episode_checkpointed_action_step_count: pipeline.episodeCheckpointedActionStepCount,
    episode_checkpointed_rounds: pipeline.episodeCheckpointedRounds,
    episode_checkpoint_count: pipeline.episodeCheckpointCount,
    episode_capsules: pipeline.episodeCapsules,
    emergency_summarized_action_step_count: pipeline.emergencySummarizedActionStepCount,
    emergency_compaction_count: pipeline.emergencyCompactionCount,
    emergency_checkpoint: pipeline.emergencyCheckpoint,
    emergency_checkpoint_sha256: pipeline.emergencyCheckpointSha256,
    emergency_checkpoint_ref: pipeline.emergencyCheckpointRef,
    emergency_transcript_ref: pipeline.emergencyTranscriptRef,
    last_semantic_rejection: pipeline.lastSemanticRejection,
    semantic_rejection_count: pipeline.semanticRejectionCount,
    episode_retry_state: retry,
    ...(pipeline.overLimitRequestCount ? { over_limit_request_count: pipeline.overLimitRequestCount } : {}),
    ...(Object.prototype.hasOwnProperty.call(pipeline, "exchangeAgedCount") ? {
      exchange_aging: {
        aged_exchanges: pipeline.exchangeAgedCount,
        aged_observation_groups: pipeline.exchangeAgedObservationGroups,
        reasoning_withheld_exchanges: pipeline.reasoningWithheldExchangeCount,
        aging_count: pipeline.exchangeAgingCount,
        hard_backstop_count: pipeline.hardBackstopCount,
      },
    } : {}),
    ...(Object.prototype.hasOwnProperty.call(pipeline, "emergencySummarizedExchangeCount") ? {
      emergency_exchange_coverage: {
        summarized_exchanges: pipeline.emergencySummarizedExchangeCount,
        kept_exchange: pipeline.emergencyKeptExchange,
      },
    } : {}),
  });
}
