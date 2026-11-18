// The stage interface of the context manager (README.md in this directory).
//
// A scheme is an ordered list of stages, each with its parameters. The driver
// (driver.ts) runs them on every request in phase order and owns the
// invariants. A stage owns one state slice: it changes that slice only in its
// own hooks (`apply`, `notice`, `observeUsage`), never another stage's slice
// and never the pipeline object. It reads the request, the shared services and
// other stages' slices through the RequestView, read-only.
//
// Hooks, by when the driver calls them:
// * `materialize` (phase materialize): the canonical base of the request,
//   built once from the raw history;
// * `store`: right after materialization, store as Context Artifacts what this
//   stage's renders may later leave out of the request, recorded on the
//   (private) materialized messages;
// * `check`: refuse a slice the request's segmentation cannot carry;
// * `render` (phases age and deliver): pure, the history after this stage's
//   recorded transitions, and messages appended after the history;
// * `slots` (phase summarize): pure, what this stage's summaries contribute to
//   the shared slot layout (summary_slots.ts);
// * `admit`: pure, whether this request crosses one of the stage's thresholds,
//   with the record the manifest carries;
// * `apply`: records the admitted transition in the stage's own slice, then
//   `commit()` (the driver re-renders every stage) and returns the named
//   transition; a rejected or deferred admission returns null without commit.
//   A stage may first `render()` a trial request from the slices as they are
//   now, to see whether its new frontier changes anything: when it does not,
//   the frontier stays moved and the request (and its epoch) stays as it was;
// * `notice`: another stage's transition asks this stage to update its slice,
//   with the numbers the notice names;
// * `observeUsage`: the Provider reported the input tokens of a request;
// * `manifest`: the stage's fields of the projection manifest.

import type { Message } from "../projection.ts";
import type { RequestView } from "./request.ts";
import type { SummaryLedger } from "./summaries.ts";

export const PHASES = ["materialize", "age", "deliver", "summarize", "guard", "view"] as const;
export type Phase = typeof PHASES[number];

/** A stage's decision on one request. */
export interface Admission {
  /** The manifest record of the decision; `apply` may complete it. */
  record: Record<string, any>;
  /** Whether the driver calls `apply`. */
  admitted: boolean;
}

/** A named change of the delivered request: one projection transition. */
export interface Transition {
  kind: string;
  /** The `ora.context-projection-transition.v4` record the manifest lists. */
  record: Record<string, any>;
  /** Summaries: the leading ActionSteps the request no longer carries exactly. */
  absorbedActionSteps?: number;
}

/** One render of a request from the stages' slices. */
export interface Rendering {
  /** The history after the age phase. */
  aged: Message[];
  /** The history after the deliver phase, without the appendix. */
  delivered: Message[];
  /** The request's messages: summary slots laid out on `delivered`, then the appendix. */
  working: Message[];
  /** What each stage's render recorded. */
  notes: Map<string, Record<string, any>>;
}

/** What the driver lends a stage while it applies an admission. */
export interface ApplyContext {
  /** A trial render from every slice as it is now; the request does not change. */
  render(): Rendering;
  /**
   * The stage has recorded its transition in its slice: open the projection
   * epoch (once per projection) and make ``rendering`` (a trial render made
   * after the stage's last change, or else a fresh render) the request.
   * Returns the epoch. A stage that commits must return its transition.
   */
  commit(rendering?: Rendering): number;
  /** Ask another stage to update its own slice (its `notice` hook), with the numbers the notice names. */
  notify(stage: string, notice: string, detail?: NoticeDetail): void;
  /** Rejections of lossy summaries, shared by the summarize stages. */
  summaries: SummaryLedger;
}

/** What the materialize phase builds from the raw history. */
export interface Materialized {
  messages: Message[];
  artifactRefs: Record<string, any>[];
  firstDeliveryAdvertisedIndexes: number[];
  reasoningChanges: number[];
  /** materialize `truncated_reply_reasoning: withheld`: the truncated replies sent without their reasoning. */
  truncatedReplies?: number[];
}

/** One render step of the age or deliver phase. */
export interface Rendered {
  history: Message[];
  /** Messages after the whole history (deliver phase: the camera tail). */
  appendix?: Message[];
  /** What this render recorded, for the stage's manifest (`RequestView.note`). */
  note?: Record<string, any>;
}

/** The inputs of the shared slot layout (summary_slots.ts). */
export interface SlotInputs {
  episodeCheckpointedActionSteps: number;
  episodeCapsules: Record<string, any>[];
  emergencySummarizedActionSteps: number;
  emergencyCheckpoint: Record<string, any> | null;
  emergencyTranscriptRef: Record<string, any> | null;
  /** Whether the slots carry their exact-transcript Pointer: the scheme's `references.pointers`. */
  transcriptPointers: boolean;
  /**
   * The unit the emergency checkpoint covers; with exchanges, the exchanges it
   * covers (from the first) and the one before them kept exact.
   */
  emergencyUnit: "action_step" | "exchange";
  emergencySummarizedExchanges: number;
  emergencyKeptExchange: number | null;
}

/** The slot inputs a summarize stage owns. */
export type SlotContribution = Partial<Omit<SlotInputs, "transcriptPointers">>;

/** What a notice names besides its kind (such as the frontier another stage moved to). */
export type NoticeDetail = Readonly<Record<string, number>>;

/**
 * A guard's rule for a request the Provider reported over the hard limit:
 * `refuse` it (`context_exhausted`), `send_and_record` it, or leave it to the
 * guard (`own_estimate`: the guard sizes every request with its own estimate).
 */
export type OverLimitRule = "refuse" | "send_and_record" | "own_estimate";

/** One Provider report of a request's input tokens, as the driver accepted it. */
export interface UsageObservation {
  /** null when the Provider reported no input tokens. */
  inputTokens: number | null;
  /** The projection epoch the request was sent at (null when the report named none). */
  projectionEpoch: unknown;
  providerWireRequestSha256: unknown;
}

export interface Stage<Params = any, State = any, A extends Admission = Admission> {
  readonly id: string;
  readonly phase: Phase;
  readonly params: Params;
  /** Transition kinds this stage may record; the driver refuses any other. */
  readonly transitions: readonly string[];
  /** Stage ids this stage needs in the same scheme. */
  readonly requires: readonly string[];
  /** Stage ids this stage cannot run beside. */
  readonly excludes: readonly string[];
  /** Stage ids whose admission on the same request this stage reads: when present, they run before it. */
  readonly after?: readonly string[];
  /** A guard's over-limit rule (driver.ts `overLimitRule`); only guards declare one. */
  readonly overLimit?: OverLimitRule;

  init(): State;
  /**
   * The slice as fields of the pipeline's `ora.state` snapshot
   * (core/persistence.ts); `restoreFields` reads them back. The snapshot format
   * is the pre-stage pipeline's, unchanged.
   */
  stateFields(state: State): Record<string, unknown>;
  restoreFields(fields: Record<string, any>): State;

  check?(view: RequestView, state: Readonly<State>): void;
  materialize?(view: RequestView, state: Readonly<State>): Materialized;
  /** Edits ``messages`` (the private materialized history) in place; the Context Artifact references it stored. */
  store?(view: RequestView, state: Readonly<State>, messages: Message[]): Record<string, any>[];
  render?(view: RequestView, state: Readonly<State>, history: Message[]): Rendered;
  slots?(state: Readonly<State>): SlotContribution;
  admit?(view: RequestView, state: Readonly<State>): A | null;
  apply?(view: RequestView, state: State, admission: A, context: ApplyContext): Promise<Transition | null>;
  notice?(state: State, notice: string, from: string, detail: NoticeDetail): void;
  observeUsage?(state: State, usage: UsageObservation): void;
  manifest?(view: RequestView, state: Readonly<State>, admission: A | null): Record<string, unknown>;
}

/** A scheme entry: which stage, with which parameters. */
export interface StageSpec {
  stage: string;
  params: Record<string, any>;
}
