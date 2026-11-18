// One request as the stages see it: its inputs, the materialized history and
// its segmentation, the current render, and read-only access to the driver's
// shared state and to other stages' slices and admissions.
//
// Only the driver writes a RequestView; stages read it.

import type { ArtifactStore } from "../../evidence/artifacts.ts";
import type { ContextDomain } from "../domain_defaults.ts";
import type { InlineMediaCache } from "../inline_media.ts";
import type { ContextArtifactCatalog } from "../memory/artifacts.ts";
import type { SummaryProvider } from "../native.ts";
import {
  type Message, inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf, providerMessage,
} from "../projection.ts";
import type { RequestEstimator } from "./services/estimator.ts";
import type { Limits } from "./services/limits.ts";
import type { ReferenceSettings } from "./services/references.ts";
import type { ExchangeTail, Exchanges, SegmentUnit } from "./services/segmenter.ts";
import type { Admission } from "./stage.ts";

export type { Limits };

/** Services the driver lends every stage; never stage-specific. */
export interface Services {
  artifacts: ArtifactStore;
  catalog: ContextArtifactCatalog;
  summaryProvider: SummaryProvider;
  domain: ContextDomain;
  mutationTools: string[];
  inlineMedia: InlineMediaCache;
  /** The scheme's references setting (services/references.ts). */
  references: ReferenceSettings;
  /** The one request-size estimate (services/estimator.ts). */
  estimator: RequestEstimator;
}

/** The driver's own slice: the projection epoch, the absorbed frontier and the Provider usage anchor. */
export interface DriverState {
  cacheEpoch: number;
  absorbedActionStepCount: number;
  lastEstimatedTokens: number | null;
  lastProviderInputTokens: number | null;
  lastProviderProjectionEpoch: number | null;
  lastProviderWireRequestSha256: string | null;
}

/** What a RequestView reads from its driver. */
export interface DriverAccess {
  readonly state: Readonly<DriverState>;
  readonly services: Services;
  readonly limits: Limits;
  stateOf(stageId: string): unknown;
  paramsOf(stageId: string): unknown;
}

export interface SizeSnapshot {
  pressure: number;
  images: number;
  media_bytes: number;
  request_bytes: number;
}

export class RequestView {
  readonly raw: Message[];
  readonly tools: Record<string, any>[];
  readonly observationStep: number;
  readonly allowTransition: boolean;
  /** The projection epoch this request started from. */
  readonly epochBefore: number;
  readonly services: Services;
  readonly limits: Limits;

  // materialize phase
  materialized: Message[] = [];
  materializedRefs: Record<string, any>[] = [];
  firstDeliveryAdvertisedIndexes: number[] = [];
  reasoningChanges: number[] = [];
  truncatedReplies: number[] = [];

  // segmentation (services/segmenter.ts, right after materialization)
  unit: SegmentUnit = "action_step";
  /** The spans of the scheme's unit; ``exchanges`` whatever the unit. */
  spans: [number, number][] = [];
  exchanges: Exchanges = { from: 0, starts: [], spans: [] };
  inventory: Record<string, any>[] = [];
  groups: [number, number][] = [];
  completedCount = 0;
  /** The Session round of the open ActionStep. */
  currentRound: unknown = null;
  /** ActionSteps every stage keeps exact, and the first of them (services/segmenter.ts). */
  exactTail = 0;
  exactTailFrom = 0;
  /** The unit the exact tail counts in; with exchanges, its exchanges, anchor and message ranges. */
  exactTailUnit: SegmentUnit = "action_step";
  exchangeTail: ExchangeTail | null = null;

  // the current render
  /** The history after the age phase: lossless, what the invariants compare with `materialized`. */
  aged: Message[] = [];
  /** The history after the deliver phase, without its appendix: what a compaction summarizes. */
  delivered: Message[] = [];
  /** The request's messages: summary slots laid out on `delivered`, then the appendix. */
  working: Message[] = [];
  providerMessages: Message[] = [];
  requestView: Record<string, any> = {};
  pressureRaw = 0;
  pressure = 0;
  calibrationDelta = 0;
  tokenEstimateStatus = "";

  /** Provider-reported input of the request before this one, when it was measured on this epoch. */
  admissionInputTokens: number | null = null;
  /** Provider-reported input over the hard limit, when the anchor applies; set before the guard phase. */
  overLimitInputTokens: number | null = null;
  readonly transitions: Record<string, any>[] = [];
  readonly admissions = new Map<string, Admission | null>();
  notes = new Map<string, Record<string, any>>();
  /** Per stage, render results by input history (identity) and key; scoped to this request. */
  private readonly memos = new Map<string, WeakMap<Message[], Map<string, unknown>>>();

  private readonly driver: DriverAccess;

  constructor(driver: DriverAccess, args: {
    raw: Message[]; tools: Record<string, any>[]; observationStep: number; allowTransition: boolean;
  }) {
    this.driver = driver;
    this.raw = args.raw;
    this.tools = args.tools;
    this.observationStep = args.observationStep;
    this.allowTransition = args.allowTransition;
    this.epochBefore = driver.state.cacheEpoch;
    this.services = driver.services;
    this.limits = driver.limits;
  }

  /** The current projection epoch. */
  get epoch(): number {
    return this.driver.state.cacheEpoch;
  }

  /** The Provider usage anchor (the driver's slice). */
  get usage(): Readonly<DriverState> {
    return this.driver.state;
  }

  /** Another stage's parameters; null when the scheme does not run it. */
  paramsOf<P>(stageId: string): Readonly<P> | null {
    return (this.driver.paramsOf(stageId) ?? null) as Readonly<P> | null;
  }

  /**
   * A stage's render of ``input`` under ``key`` (the part of its slice the
   * render reads), computed once per request: renders are pure, and the
   * passes before it return the same history for the same slices.
   */
  memoize<T>(stageId: string, input: Message[], key: string, compute: () => T): T {
    let byInput = this.memos.get(stageId);
    if (byInput === undefined) {
      byInput = new WeakMap();
      this.memos.set(stageId, byInput);
    }
    let byKey = byInput.get(input);
    if (byKey === undefined) {
      byKey = new Map();
      byInput.set(input, byKey);
    }
    if (!byKey.has(key)) byKey.set(key, compute());
    return byKey.get(key) as T;
  }

  /** Another stage's slice, read-only; null when the scheme does not run it. */
  stateOf<S>(stageId: string): Readonly<S> | null {
    return (this.driver.stateOf(stageId) ?? null) as Readonly<S> | null;
  }

  /** An earlier stage's admission on this request; null when it made none. */
  admission<A extends Admission>(stageId: string): A | null {
    return (this.admissions.get(stageId) ?? null) as A | null;
  }

  /** What a stage's latest render recorded. */
  note(stageId: string): Record<string, any> | null {
    return this.notes.get(stageId) ?? null;
  }

  /** What a compaction summarizes: the history as the model was shown it. */
  compactorInput(): Message[] {
    return this.delivered;
  }

  /** Make ``working`` the request and re-estimate it. */
  adopt(working: Message[]): void {
    this.working = working;
    this.providerMessages = working.map(providerMessage);
    this.requestView = { messages: this.providerMessages, tools: [...this.tools] };
    this.pressureRaw = this.services.estimator.pressure(this.requestView);
    const state = this.driver.state;
    const calibrated = this.services.estimator.calibrate(this.pressureRaw, {
      applies: this.providerAnchorApplies(),
      inputTokens: state.lastProviderInputTokens,
      estimatedTokens: state.lastEstimatedTokens,
    });
    this.pressure = calibrated.tokens;
    this.calibrationDelta = calibrated.delta;
    this.tokenEstimateStatus = calibrated.status;
  }

  /** Provider usage was measured on the current projection epoch. */
  providerAnchorApplies(): boolean {
    const state = this.driver.state;
    return state.lastProviderInputTokens !== null && state.lastEstimatedTokens !== null
      && state.lastProviderProjectionEpoch === state.cacheEpoch;
  }

  /** Pre-transition sizes that a transition record reports. */
  sizeSnapshot(): SizeSnapshot {
    return {
      pressure: this.pressure,
      images: inlineImageCount(this.requestView),
      media_bytes: inlineMediaBytesFromRefs(this.working),
      request_bytes: jsonBytesOf(this.requestView),
    };
  }
}
