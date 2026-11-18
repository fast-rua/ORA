// The stage driver: runs a scheme's stages on every request in phase order
// (materialize, age, deliver, summarize, guard, view) and owns the invariants
// (README.md):
//
// * no delivered message changes except through a named transition: a render
//   changes only when a stage commits, and a commit must return a transition
//   of a kind the stage declares;
// * each stage writes only its own slice (checked around every `apply`; other
//   stages change only through a notice the driver routes);
// * the fixed prefix (system prompt, project instructions, task) never ages;
// * projection is deterministic: renders are pure functions of the request and
//   the slices, and the replay goldens hold it byte for byte;
// * the over-limit rule, taken from the guard: a request the Provider
//   reported over the hard limit is refused (`context_exhausted`) unless the
//   guard sends and records it, or sizes every request itself (hard_backstop).
//
// The driver keeps one state record: its own slice (projection epoch, absorbed
// frontier, Provider usage anchor), the summary ledger and one slice per
// stage. `pipelineFields` / `loadPipelineFields` serialize it in the legacy
// pipeline's field format, so `ora.state` (core/persistence.ts) is unchanged.

import type { ProjectedRequest } from "../native.ts";
import { type Message, causalMessageViolations, losslessInvariantView, providerMessage } from "../projection.ts";
import { PolicySessionError, canonicalBytesPy, canonicalSha256, clonePy, isInt, num, sortedStrings } from "../pyutil.ts";
import { CanonicalSizer, withProjectionSizer } from "../request_size.ts";
import { projectionManifestV6 } from "./manifest_v6.ts";
import { MATERIALIZE } from "./names.ts";
import { type DriverState, type Limits, RequestView, type Services } from "./request.ts";
import { isFixedPrefix } from "./request_identity.ts";
import { type StageScheme, buildStage } from "./scheme.ts";
import { RequestEstimator } from "./services/estimator.ts";
import { segment } from "./services/segmenter.ts";
import { SchemaViolation, ToolSurfaceError, validateSchemaInstance } from "../memory/schema_validation.ts";
import {
  type Admission, type ApplyContext, type OverLimitRule, PHASES, type Phase, type Rendering, type Stage,
  type UsageObservation,
} from "./stage.ts";
import { checkUnownedFields, serializable, withInitialFields } from "./state_fields.ts";
import { SummaryLedger } from "./summaries.ts";
import { layoutSummarySlots, mergeSlotInputs, summarizedActionSteps } from "./summary_slots.ts";

/** What the legacy manifest names the pipeline by. */
export interface PipelineIdentity {
  profile: string;
  retention: string;
  summaryPolicy: string;
}

export interface LosslessInvariants {
  canonicalCausalTail: unknown;
  beforeSha256: string;
  afterSha256: string;
}

function phaseIndex(phase: Phase): number {
  return PHASES.indexOf(phase);
}

export class StageDriver {
  readonly scheme: StageScheme;
  /** The scheme's stages in phase order (stable within a phase). */
  readonly stages: Stage[];
  readonly services: Services;
  readonly limits: Limits;
  readonly identity: PipelineIdentity;
  readonly ledger = new SummaryLedger();
  readonly state: DriverState = {
    cacheEpoch: 0,
    absorbedActionStepCount: 0,
    lastEstimatedTokens: null,
    lastProviderInputTokens: null,
    lastProviderProjectionEpoch: null,
    lastProviderWireRequestSha256: null,
  };
  private readonly slices = new Map<string, unknown>();
  /** Exact size memo for one projection at a time; never state. */
  private readonly sizer = new CanonicalSizer();

  constructor(
    scheme: StageScheme, services: Omit<Services, "references" | "estimator">, limits: Limits, identity: PipelineIdentity,
    schemas: Record<string, any>,
  ) {
    validateParams(schemas, SCHEME_SETTINGS, {
      segments: scheme.segments, limits: scheme.limits, estimator: scheme.estimator, references: scheme.references,
    });
    for (const spec of scheme.stages) validateParams(schemas, spec.stage, spec.params);
    if (limits.reserveTokens !== scheme.limits.reserve_tokens) {
      throw new PolicySessionError("context limits do not use the scheme's reserve");
    }
    this.scheme = scheme;
    this.services = {
      ...services,
      references: scheme.references,
      estimator: new RequestEstimator(scheme.estimator.image_tokens),
    };
    this.limits = limits;
    this.identity = identity;
    const stages = scheme.stages.map(buildStage);
    this.stages = stages.map((stage, order) => ({ stage, order }))
      .sort((a, b) => phaseIndex(a.stage.phase) - phaseIndex(b.stage.phase) || a.order - b.order)
      .map((item) => item.stage);
    checkComposition(this.stages);
    for (const stage of this.stages) this.slices.set(stage.id, stage.init());
  }

  stage(id: string): Stage | null {
    return this.stages.find((stage) => stage.id === id) ?? null;
  }

  stateOf(id: string): unknown {
    return this.slices.get(id) ?? null;
  }

  paramsOf(id: string): unknown {
    return this.stage(id)?.params ?? null;
  }

  private slice(stage: Stage): any {
    return this.slices.get(stage.id);
  }

  private inPhase(phase: Phase): Stage[] {
    return this.stages.filter((stage) => stage.phase === phase);
  }

  /** The guard's over-limit rule; without a guard, refuse. */
  get overLimitPolicy(): OverLimitRule {
    return this.inPhase("guard")[0]?.overLimit ?? "refuse";
  }

  // -- one request --------------------------------------------------------------

  async project(args: {
    messages: Message[]; tools: Record<string, any>[]; observationStep: number; allowTransition: boolean;
  }): Promise<ProjectedRequest> {
    // Rewrite the Session Artifact index once per projection, not per ref.
    return this.services.catalog.batchedPersistence(() => withProjectionSizer(this.sizer, async () => {
      const view = new RequestView(this, {
        raw: args.messages, tools: args.tools, observationStep: args.observationStep,
        allowTransition: args.allowTransition,
      });
      const materialize = this.inPhase("materialize")[0];
      const materialized = materialize.materialize!(view, this.slice(materialize));
      view.materialized = materialized.messages;
      view.materializedRefs = this.store(view, materialized.messages, materialized.artifactRefs);
      view.firstDeliveryAdvertisedIndexes = materialized.firstDeliveryAdvertisedIndexes;
      view.reasoningChanges = materialized.reasoningChanges;
      view.truncatedReplies = materialized.truncatedReplies ?? [];
      this.segment(view);
      for (const stage of this.stages) stage.check?.(view, this.slice(stage));
      this.adopt(view, this.render(view));
      const fixedPrefix = view.working.filter(isFixedPrefix);
      // Admission uses the usage measured on the epoch this projection started
      // from; a transition below opens a new epoch but does not undo the crossing.
      view.admissionInputTokens = view.providerAnchorApplies() ? this.state.lastProviderInputTokens : null;
      for (const phase of PHASES) {
        if (phase === "materialize") continue;
        if (phase === "guard") this.overLimitRule(view);
        for (const stage of this.inPhase(phase)) {
          if (!stage.admit) continue;
          const admission = stage.admit(view, this.slice(stage));
          view.admissions.set(stage.id, admission);
          if (admission !== null && admission.admitted) await this.apply(view, stage, admission, fixedPrefix);
        }
      }
      const invariants = this.verify(view);
      return {
        messages: view.providerMessages,
        tools: [...args.tools],
        manifest: projectionManifestV6(this, view, invariants),
      };
    }));
  }

  /**
   * The stages' `store` hooks on the materialized history; with any, the
   * references of the request are the union by ref, in ref order.
   */
  private store(view: RequestView, messages: Message[], refs: Record<string, any>[]): Record<string, any>[] {
    const storing = this.stages.filter((stage) => stage.store);
    if (!storing.length) return refs;
    const stored = storing.flatMap((stage) => stage.store!(view, this.slice(stage), messages));
    const unique = new Map<string, Record<string, any>>();
    for (const reference of [...refs, ...stored]) unique.set(String(reference.ref), reference);
    return sortedStrings(unique.keys()).map((key) => clonePy(unique.get(key)!));
  }

  /** Segment the materialized history (services/segmenter.ts). */
  private segment(view: RequestView): void {
    const segmentation = segment(view.materialized, {
      unit: this.scheme.segments.unit, exactTail: this.scheme.segments.exact_tail,
      exactTailUnit: this.scheme.segments.exact_tail_unit,
      mutationTools: this.services.mutationTools, domain: this.services.domain,
    });
    view.unit = segmentation.unit;
    view.spans = segmentation.spans;
    view.exchanges = segmentation.exchanges;
    view.inventory = segmentation.inventory;
    view.groups = segmentation.groups;
    view.completedCount = segmentation.completedCount;
    view.currentRound = segmentation.currentRound;
    view.exactTail = segmentation.exactTail;
    view.exactTailFrom = segmentation.exactTailFrom;
    view.exactTailUnit = segmentation.exactTailUnit;
    view.exchangeTail = segmentation.exchangeTail;
  }

  /**
   * Render every stage from its slice: the age phase on the materialized
   * history (a guard that renders its own frontier, hard_backstop `aging:
   * own`, last), the deliver phase on that, the summary slots laid out on the
   * delivered history, then the deliver phase's appendix.
   */
  private render(view: RequestView): Rendering {
    let history = view.materialized;
    let aged = history;
    const appendix: Message[] = [];
    const notes = new Map<string, Record<string, any>>();
    for (const phase of ["age", "deliver"] as const) {
      const rendering = phase === "age" ? [...this.inPhase("age"), ...this.inPhase("guard")] : this.inPhase(phase);
      for (const stage of rendering) {
        if (!stage.render) continue;
        const rendered = stage.render(view, this.slice(stage), history);
        history = rendered.history;
        if (rendered.appendix) appendix.push(...rendered.appendix);
        if (rendered.note) notes.set(stage.id, rendered.note);
      }
      if (phase === "age") aged = history;
    }
    const working = [...layoutSummarySlots(history, this.slotInputs()), ...appendix];
    return { aged, delivered: history, working, notes };
  }

  /** Make ``rendering`` the request. */
  private adopt(view: RequestView, rendering: Rendering): void {
    view.aged = rendering.aged;
    view.delivered = rendering.delivered;
    view.notes = rendering.notes;
    view.adopt(rendering.working);
  }

  private slotInputs() {
    return mergeSlotInputs(this.inPhase("summarize")
      .filter((stage) => stage.slots)
      .map((stage) => stage.slots!(this.slice(stage))), this.services.references.pointers);
  }

  /** ActionSteps the summary slots stand in for. */
  summarizedActionSteps(): number {
    return summarizedActionSteps(this.slotInputs());
  }

  /** Apply an admission; the stage changes the request only through a committed, named transition. */
  private async apply(view: RequestView, stage: Stage, admission: Admission, fixedPrefix: Message[]): Promise<void> {
    const others = this.otherSliceDigests(stage.id);
    const noticed = new Set<string>();
    let committed = false;
    const context: ApplyContext = {
      render: () => this.render(view),
      commit: (rendering?: Rendering) => {
        if (committed) throw new PolicySessionError(`context stage ${stage.id} committed twice`);
        committed = true;
        // The first transition of this projection opens a new epoch.
        if (this.state.cacheEpoch === view.epochBefore) this.state.cacheEpoch += 1;
        this.adopt(view, rendering ?? this.render(view));
        if (!fixedPrefixBytes(view.working).equals(fixedPrefixBytes(fixedPrefix))) {
          throw new PolicySessionError(`context stage ${stage.id} changed the fixed prefix`);
        }
        return this.state.cacheEpoch;
      },
      notify: (target, notice, detail = {}) => {
        const recipient = this.stage(target);
        if (recipient === null) return;
        if (!recipient.notice) throw new PolicySessionError(`context stage ${target} takes no notice`);
        recipient.notice(this.slice(recipient), notice, stage.id, detail);
        noticed.add(target);
      },
      summaries: this.ledger,
    };
    const transition = await stage.apply!(view, this.slice(stage), admission, context);
    if (transition === null) {
      if (committed) throw new PolicySessionError(`context stage ${stage.id} changed the request without a transition`);
    } else {
      if (!committed) throw new PolicySessionError(`context stage ${stage.id} recorded a transition it did not commit`);
      if (!stage.transitions.includes(transition.kind)) {
        throw new PolicySessionError(`context stage ${stage.id} recorded an undeclared transition ${transition.kind}`);
      }
      if (transition.absorbedActionSteps !== undefined) {
        this.state.absorbedActionStepCount = Math.max(this.state.absorbedActionStepCount, transition.absorbedActionSteps);
      }
      view.transitions.push(transition.record);
    }
    for (const [id, digest] of others) {
      if (!noticed.has(id) && this.sliceDigest(id) !== digest) {
        throw new PolicySessionError(`context stage ${stage.id} wrote the state of ${id}`);
      }
    }
  }

  private sliceDigest(id: string): string {
    const stage = this.stage(id)!;
    return canonicalSha256(serializable(stage.stateFields(this.slice(stage))));
  }

  private otherSliceDigests(id: string): Map<string, string> {
    const digests = new Map<string, string>();
    for (const stage of this.stages) if (stage.id !== id) digests.set(stage.id, this.sliceDigest(stage.id));
    return digests;
  }

  /** The guard's rule for a request the Provider reported over the hard limit. */
  private overLimitRule(view: RequestView): void {
    const authoritative = view.providerAnchorApplies() ? this.state.lastProviderInputTokens : null;
    view.overLimitInputTokens = null;
    // A guard with its own estimate (hard_backstop) has sized the request it lets through.
    if (this.overLimitPolicy === "own_estimate") return;
    if (authoritative === null || authoritative <= this.limits.hardInputTokens) return;
    if (this.overLimitPolicy !== "send_and_record") {
      throw new PolicySessionError(
        `context_exhausted: Provider reported ${authoritative} input tokens, hard limit ${this.limits.hardInputTokens}`,
      );
    }
    view.overLimitInputTokens = authoritative;
  }

  /** Refuse a request that carries privileged fields or lost lossless semantics. */
  private verify(view: RequestView): LosslessInvariants {
    const { domain } = this.services;
    const violations = causalMessageViolations(view.providerMessages, domain);
    if (violations.length) {
      throw new PolicySessionError("causal_context_forbidden_fields: " + violations.slice(0, 12).join(", "));
    }
    const { groups, materialized } = view;
    // The exact tail's messages: from its first ActionStep on, or (exchanges) the anchor and the latest exchanges.
    const tailStart = view.exactTailFrom < groups.length ? groups[view.exactTailFrom][0] : materialized.length;
    const ranges = view.exchangeTail?.ranges ?? [[tailStart, materialized.length]];
    const tailOf = (messages: Message[]) => ranges.flatMap(([start, end]) => losslessInvariantView(
      messages.slice(start, end), domain));
    const canonicalCausalTail = tailOf(materialized);
    const projectedCausalTail = tailOf(view.aged);
    if (!canonicalBytesPy(canonicalCausalTail).equals(canonicalBytesPy(projectedCausalTail))) {
      throw new PolicySessionError("lossless projector changed the exact causal ActionStep tail");
    }
    const beforeSha256 = canonicalSha256(losslessInvariantView(materialized, domain));
    const afterSha256 = canonicalSha256(losslessInvariantView(view.aged, domain));
    if (beforeSha256 !== afterSha256) {
      throw new PolicySessionError("lossless projector changed normalized policy-visible semantics");
    }
    return { canonicalCausalTail, beforeSha256, afterSha256 };
  }

  // -- Provider usage -------------------------------------------------------------

  /** Anchor estimates to exact Provider usage for the request just sent. */
  observeProviderUsage(args: {
    estimatedTokens: unknown; inputTokens: unknown; projectionEpoch?: unknown; providerWireRequestSha256?: unknown;
  }): void {
    const estimated = args.estimatedTokens;
    const inputTokens = args.inputTokens;
    const projectionEpoch = args.projectionEpoch ?? null;
    const wireSha = args.providerWireRequestSha256 ?? null;
    if (!isInt(estimated) || num(estimated) < 1) {
      throw new PolicySessionError("Provider usage estimate anchor is invalid");
    }
    if (!(isInt(inputTokens) && num(inputTokens) >= 0)) {
      this.observed({ inputTokens: null, projectionEpoch, providerWireRequestSha256: wireSha });
      return;
    }
    if (projectionEpoch === null) throw new PolicySessionError("lossless Provider usage lacks its projection epoch");
    if (!isInt(projectionEpoch) || num(projectionEpoch) < 0) {
      throw new PolicySessionError("Provider usage projection epoch is invalid");
    }
    const epoch = num(projectionEpoch);
    if (epoch !== this.state.cacheEpoch) throw new PolicySessionError("lossless Provider usage projection epoch is stale");
    if (wireSha === null) throw new PolicySessionError("lossless Provider usage lacks its wire request identity");
    if (typeof wireSha !== "string" || !/^[0-9a-f]{64}$/.test(wireSha)) {
      throw new PolicySessionError("Provider usage wire request identity is invalid");
    }
    const tokens = num(inputTokens);
    this.state.lastEstimatedTokens = num(estimated);
    this.state.lastProviderInputTokens = tokens;
    this.state.lastProviderProjectionEpoch = epoch;
    this.state.lastProviderWireRequestSha256 = wireSha;
    this.observed({ inputTokens: tokens, projectionEpoch: epoch, providerWireRequestSha256: wireSha });
  }

  private observed(usage: UsageObservation): void {
    for (const stage of this.stages) stage.observeUsage?.(this.slice(stage), usage);
  }

  // -- the state record in the legacy pipeline's fields -----------------------------

  /** The state as the legacy pipeline's fields (core/persistence.ts `PIPELINE_STATE_FIELDS`). */
  pipelineFields(): Record<string, any> {
    const fields: Record<string, any> = {
      lastEstimatedTokens: this.state.lastEstimatedTokens,
      lastProviderInputTokens: this.state.lastProviderInputTokens,
      lastProviderProjectionEpoch: this.state.lastProviderProjectionEpoch,
      lastProviderWireRequestSha256: this.state.lastProviderWireRequestSha256,
      cacheEpoch: this.state.cacheEpoch,
      absorbedActionStepCount: this.state.absorbedActionStepCount,
      lastSemanticRejection: this.ledger.lastRejection,
      semanticRejectionCount: this.ledger.rejectionCount,
    };
    for (const stage of this.stages) {
      for (const [field, value] of Object.entries(stage.stateFields(this.slice(stage)))) {
        if (field in fields) throw new PolicySessionError(`two context owners serialize ${field}`);
        fields[field] = value;
      }
    }
    // Fields of stages this scheme does not run keep their initial values.
    return withInitialFields(fields);
  }

  /** Load the state from the legacy pipeline's fields; refuse a frontier of a stage this scheme does not run. */
  loadPipelineFields(fields: Record<string, any>): void {
    this.state.lastEstimatedTokens = fields.lastEstimatedTokens;
    this.state.lastProviderInputTokens = fields.lastProviderInputTokens;
    this.state.lastProviderProjectionEpoch = fields.lastProviderProjectionEpoch;
    this.state.lastProviderWireRequestSha256 = fields.lastProviderWireRequestSha256;
    this.state.cacheEpoch = fields.cacheEpoch;
    this.state.absorbedActionStepCount = fields.absorbedActionStepCount;
    this.ledger.lastRejection = fields.lastSemanticRejection;
    this.ledger.rejectionCount = fields.semanticRejectionCount;
    const owned = new Set(Object.keys(this.pipelineFieldsOwned()));
    for (const stage of this.stages) {
      this.slices.set(stage.id, stage.restoreFields(fields));
      for (const field of Object.keys(stage.stateFields(this.slice(stage)))) owned.add(field);
    }
    checkUnownedFields(fields, owned, "this stage scheme");
  }

  /** The fields the driver itself serializes. */
  private pipelineFieldsOwned(): Record<string, unknown> {
    return {
      lastEstimatedTokens: null, lastProviderInputTokens: null, lastProviderProjectionEpoch: null,
      lastProviderWireRequestSha256: null, cacheEpoch: null, absorbedActionStepCount: null,
      lastSemanticRejection: null, semanticRejectionCount: null,
    };
  }
}

/** The provider-visible fixed prefix of a request. */
function fixedPrefixBytes(messages: Message[]): Buffer {
  return canonicalBytesPy(messages.filter(isFixedPrefix).map(providerMessage));
}

/** The scheme settings' schema among the stage schemas (ora/policy/context_stages/scheme.schema.json). */
export const SCHEME_SETTINGS = "scheme";

/**
 * Validate ``params`` with the schema of ``id`` that Python sent
 * (ora/policy/context_stages/<id>.schema.json): the rules live in those files only.
 */
function validateParams(schemas: Record<string, any>, id: string, params: unknown): void {
  const schema = schemas[id];
  if (schema === undefined || schema === null) throw new PolicySessionError(`no parameter schema for context stage ${id}`);
  try {
    validateSchemaInstance(params, schema, { rootPath: "$input" });
  } catch (error) {
    if (error instanceof SchemaViolation || error instanceof ToolSurfaceError) {
      throw new PolicySessionError(`context stage ${id}: ${error.message}`);
    }
    throw error;
  }
}

/** Refuse a scheme whose stages cannot run together. */
function checkComposition(stages: Stage[]): void {
  const ids = stages.map((stage) => stage.id);
  const duplicate = ids.find((id, index) => ids.indexOf(id) !== index);
  if (duplicate !== undefined) throw new PolicySessionError(`context scheme lists stage ${duplicate} twice`);
  if (stages.filter((stage) => stage.phase === "materialize").length !== 1 || stages[0].id !== MATERIALIZE) {
    throw new PolicySessionError("context scheme needs exactly one materialize stage, first");
  }
  if (stages.filter((stage) => stage.phase === "guard").length > 1) {
    throw new PolicySessionError("context scheme has more than one guard");
  }
  stages.forEach((stage, index) => {
    for (const needed of stage.requires) {
      if (!ids.includes(needed)) throw new PolicySessionError(`context stage ${stage.id} requires ${needed}`);
    }
    // A stage reads the admissions of the stages it runs after: when present, they come first.
    for (const earlier of stage.after ?? []) {
      if (ids.indexOf(earlier) > index) throw new PolicySessionError(`context stage ${stage.id} must run after ${earlier}`);
    }
    for (const excluded of stage.excludes) {
      if (ids.includes(excluded)) throw new PolicySessionError(`context stage ${stage.id} excludes ${excluded}`);
    }
  });
}
