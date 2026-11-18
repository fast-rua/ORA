# Context stages

The context manager as an ordered list of stages, each with parameters. A
stage owns one group of switches of the context-management design space
(when to compress, what ages, how images reach the model, who summarizes, what
happens at the limit); the driver runs the stages on every request and owns
the invariants.

Lossless retention (`action_step_lossless_v4`) runs here: management v6
(`ora.native-context-management.v6`, the default of LIBERO-Pro, RoboTwin,
RoboCasa365 and RoboLab for tool calls), v8 and v9 (exchange aging; their
schemes were retired on 2026-11-05, sealed runs and the test fixtures in
`tests/data/context_scheme_fixtures` keep them) and v10 (`context_keep_written`, the
default of RoboDojo and of code mode everywhere, and `context_keep_written_unaged_reasoning`). The core refuses v4, v5 and v7 (`../management.ts`); Python still
validates them and evaluates their sealed runs, and their recorded
differential fixtures run here through two stage parameters
(`stride_aging.strided`, `episode_capsule.trigger_ratio`; the test translation
is `ora-pi/test/support/legacy_context.ts`). `rewrite_compact_v1`, the context
of a plan that names no retention, is the one projection outside the stages
(`../native.ts`).

## The stage interface (`stage.ts`)

A stage has an `id`, a `phase`, its `params`, the transition kinds it may
record, the stages it `requires` and `excludes`, the stages it runs `after`
(whose admission on the same request it reads), and one state slice
(`init`). Its hooks:

| Hook | Phase | Pure | What it does |
|---|---|---|---|
| `materialize` | materialize | | builds the canonical base of the request from its raw history |
| `store` | materialize | | stores as Context Artifacts what the stage's renders may leave out, on the private materialized messages (exchange_aging: long arguments; reasoning_replay: continuation states) |
| `check` | all | yes | refuses a slice the request's segmentation cannot carry |
| `render` | age, deliver (a guard's: after the age phase) | yes | the history after the stage's recorded transitions, and messages appended after it |
| `slots` | summarize | yes | the slot inputs the stage owns in the shared summary layout (`summary_slots.ts`) |
| `admit` | all | yes | whether this request crosses one of the stage's thresholds, with the manifest record |
| `apply` | all | | records the admitted transition in the stage's own slice, `commit()`s, returns the named transition; it may `render()` a trial request first, and when its new frontier changes nothing the model sees, return null with the frontier moved |
| `notice` | | | another stage's transition asks this stage to update its own slice, with the numbers it names (an aged frontier) |
| `observeUsage` | | | the Provider reported a request's input tokens |
| `manifest` | all | yes | the stage's fields of the projection manifest |
| `stateFields`, `restoreFields` | | | the slice in the `ora.state` snapshot's fields |

A stage writes only its own slice, never another stage's and never the
pipeline. It reads the request, the shared services and other stages' slices
and admissions through the `RequestView` (`request.ts`).

## The driver (`driver.ts`)

On each request: `materialize`, then every stage's `store` (with any, the
request's Artifact references are their union in ref order); the segmenter
(ActionSteps, exchanges, the open round, the exact tail); `check`; a render; then, phase by phase, every stage's
`admit` and, when admitted, `apply`; the over-limit rule before the guard
phase; the invariant checks; the manifest (`manifest_v6.ts`).

A render runs the age phase on the materialized history (then a guard's
render, hard_backstop `aging: own`), the deliver phase on that (what a
compaction summarizes), lays the summary slots out on the delivered history,
and appends the deliver phase's appendix (the camera tail).

The driver builds the shared services from the scheme: limits, the estimator,
references and the segmenter ([services](services/README.md)).

Invariants the driver holds:

- **No change except through named transitions.** The request is re-rendered
  only when a stage calls `commit()`; a stage that commits must return a
  transition of a kind it declares, and one that returns a transition must
  have committed. The first commit of a projection opens a new projection
  epoch, which the request assembler's cache check relies on.
- **Each stage owns its slice.** Around every `apply` the driver compares
  every other stage's slice before and after; only a stage the applying stage
  sent a notice to may change.
- **The exact tail stays lossless.** The age phase may not change what the
  exact tail shows, normalized by identity: the messages from its first
  ActionStep on or, with an exact tail in exchanges, the anchor exchange and
  the latest exchanges (`services/segmenter.ts`).
- **The fixed prefix never ages.** After every commit the provider-visible
  system prompt, project instructions and task must be byte-identical to the
  request's first render.
- **Deterministic projection.** Renders, `slots`, `admit` and `manifest` are
  pure functions of the request and the slices; context-level replays of
  sealed runs (`test/support/context_replay.ts`) hold them byte for byte.
- **The over-limit rule, from the guard.** A request the Provider reported
  over the hard input limit (window − max output − 2,048) is refused with
  `context_exhausted`, unless the scheme's guard has `over_limit:
  send_and_record`, which sends it and records it, or is `hard_backstop`,
  which estimates every request before it is sent and so replaces the rule
  (`overLimit: own_estimate`). Only `hard_backstop` keeps the design's
  invariant that no request estimated over the hard limit is sent; v6 has it
  since 2026-11-14 (`aging: own`), v6 plans resolved before keep
  `over_limit_guard` (`refuse`), which refuses only after the report.

It also refuses a scheme without exactly one `materialize` stage first, with
two guards, with one stage twice, whose `requires`/`excludes` fail, or with a
stage before one it runs `after`. Renders are memoized per request by stage,
input history and the slice values they read (`RequestView.memoize`).

## State

One record: the driver's own slice (projection epoch, absorbed frontier,
Provider usage anchor), the summary ledger (rejected summaries, shared by the
summarize stages, `summaries.ts`) and one slice per stage. For now it is
serialized in the snapshot's pipeline fields (`pipelineFields`,
`loadPipelineFields`; `state_fields.ts`, `core/persistence.ts`), so `ora.state`
and the goldens' state digests are unchanged; fields of a stage the scheme
does not run keep their initial values, and restoring a state that moved one
is refused. The rewrite pipeline serializes into the same fields.
Changing the state format is a later, separately blessed step.

## Schemes (`scheme.ts`)

`schemeFromManagement` is the one place a management object becomes a stage
list; sealed plans, v1 schemes and the goldens carry management objects. A
plan resolved from an `ora.context-scheme.v2` scheme also carries its stage
list (`policy.context_stages`, `schemeFromStageList`): the pipeline runs it
after checking it equals the translation of the plan's management object,
which the manifest still echoes. A caller may also hand the pipeline a scheme
of its own (`NativeContextPipeline` `stageScheme`), as the differential replay
does for v4 and v5.

Every stage's parameters, and the scheme settings, are validated with the
schemas of `ora/policy/context_stages/` that Python sends at session start
(`domain.context_stage_schemas`; tests and replays read the same files,
`ora-pi/test/support/stage_schemas.ts`). The stages keep no parameter rules of
their own; lossless retention without the schemas is refused.

v6:

| Order | Stage | Phase | Parameters from v6 |
|---|---|---|---|
| 1 | [`materialize`](materialize/README.md) | materialize | reasoning excluded (Pointers advertised per `references`); `truncated_reply_reasoning` (optional): truncated replies sent without their reasoning |
| 2 | [`stride_aging`](stride_aging/README.md) | age | `tail_aging_stride_action_steps` (4) |
| 3 | [`eager_externalize`](eager_externalize/README.md) | age | `eager_large_result_bytes` (L1/L2) |
| 4 | [`media_window`](media_window/README.md) | age | `inline_media_tail_action_steps` (2) (L3) |
| 5 | [`pressure_externalize`](pressure_externalize/README.md) | age | `trigger_inline_images`, `trigger_estimated_tokens`, `trigger_provider_input_ratio` (L4) |
| 6 | [`episode_capsule`](episode_capsule/README.md) | summarize | `episode_checkpoint_trigger_tokens`, checkpoint target and maximum, retries 4 / 4 |
| 7 | [`emergency_checkpoint`](emergency_checkpoint/README.md) | summarize | `emergency_fallback_ratio` alone, `emergency_target_ratio`, checkpoint target and maximum |
| 8 | [`hard_backstop`](hard_backstop/README.md) | guard | `backstop_exact_tail_exchanges` (optional, 2 in `context_lossless_v6`): `aging: own`, every request estimated before it is sent, all but the latest exchanges aged from L6's watermark on (`trigger_ratio` = `emergency_fallback_ratio`); without it [`over_limit_guard`](over_limit_guard/README.md) `refuse` (v6 plans resolved before 2026-11-14) |

Scheme-level: `segments` (unit `action_step`, exact tail 3), `limits`
(reserve 2,048), `estimator.image_tokens` (4,096) and `references` (Pointers
shown, advertised over 50,000 bytes, a 2,048-byte head preview).

v8:

| Order | Stage | Phase | Parameters from v8 |
|---|---|---|---|
| 1 | [`materialize`](materialize/README.md) | materialize | reasoning excluded |
| 2 | [`camera_delivery`](camera_delivery/README.md) | deliver | `camera_images`, `camera_tail_observations` (rolling tail, 2) |
| 3 | [`episode_capsule`](episode_capsule/README.md) | summarize | `episode_checkpoint_trigger_tokens`, checkpoint target and maximum, retries 4 / 4 |
| 4 | [`emergency_checkpoint`](emergency_checkpoint/README.md) | summarize | `compaction_trigger_tokens`, `emergency_fallback_ratio`, `emergency_target_ratio`, checkpoint target and maximum |
| 5 | [`over_limit_guard`](over_limit_guard/README.md) | guard | `send_and_record` |

Scheme-level: `segments` (unit `action_step`, exact tail
`causal_exact_tail_action_steps`, 3), `limits` (reserve 2,048),
`estimator.image_tokens` (`image_estimate_tokens`, 530) and `references`
(no Pointer shown).

v9 ([exchange aging](../../../../docs/reference/exchange-aging-context.md)):

| Order | Stage | Phase | Parameters from v9 |
|---|---|---|---|
| 1 | [`materialize`](materialize/README.md) | materialize | reasoning excluded |
| 2 | [`exchange_aging`](exchange_aging/README.md) | age | `exchange_aging_trigger_tokens` and `_ratio`, `exact_tail_exchanges`, `aged_argument_bytes`, `aged_result_bytes`, `aged_preview_bytes` (head and tail halves) |
| 3 | [`reasoning_replay`](reasoning_replay/README.md) | age | `reasoning_replay`, `reasoning_tail_exchanges` |
| 4 | [`camera_delivery`](camera_delivery/README.md) | deliver | `camera_images` (`exchange_window_v1` in every v10 scheme), `camera_tail_observations` |
| 5 | [`episode_capsule`](episode_capsule/README.md) | summarize | as v8 |
| 6 | [`emergency_checkpoint`](emergency_checkpoint/README.md) | summarize | as v8 |
| 7 | [`hard_backstop`](hard_backstop/README.md) | guard | `backstop_exact_tail_exchanges` |

Scheme-level: `segments` (unit `exchange`, exact tail 3), `limits` (reserve
2,048), `estimator.image_tokens` (530) and `references` (Pointers shown,
none advertised at first delivery). The three exchange stages share one
rewrite (`exchange_aging/rewrite.ts`): exchange_aging owns the aged and
observation frontiers, reasoning_replay the reasoning frontier, and each moves
another's only by a notice. The v9 state is the snapshot's `exchangeAging`
object (`state_fields.ts` `EXCHANGE_FIELDS`), present only for a scheme that
runs them; the manifest adds v9's fields after v8's and the exchange passes
after the L1-L4 passes.

A context pack (`policy.context_pack`, `../context_pack.ts`
`packContextManagement`) that turns L4 off sets `pressure_externalize`'s
`enabled` to false; one that turns rollout compaction off sets both summary
stages' `enabled` to false. The stages stay in the scheme: they render as
before, keep their manifest fields and record why nothing ran.

## Manifest (`manifest_v6.ts`)

The `ora.native-context-projection.v6` manifest, field for field and in the
legacy order. Stages supply their fields by name; a field two stages supply,
or one the format has no place for, is refused. In a scheme without an age
phase (v8), the age fields take their "nothing aged" values and the five
L1–L4 passes are listed unchanged.
