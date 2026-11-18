# emergency_checkpoint (L6)

Phase `summarize`. Code: `stage.ts`, the summary writer in `checkpoint.ts`.

## What it does

When the Provider reported, for the request before, at least the watermark
(`trigger_tokens` capped by `trigger_ratio` of the hard input limit) on the
projection epoch this request starts from, and L5 has no priority, every
ActionStep before the exact tail is summarized by the compactor into one
bounded `ora.emergency-context-checkpoint.v1`. The packet carries the
previous checkpoint, the capsules it does not cover yet and the newly
absorbed ActionSteps as the model was shown them; the stored transcript is
cumulative. The checkpoint slot replaces the absorbed span; a later
checkpoint replaces it. A rejected summary keeps the current projection.

**In exchanges** (`unit: exchange`, with `segments.exact_tail_unit:
exchange`). In code mode an ActionStep can hold dozens of exchanges, and an
ActionStep tail keeps them all: d6f33596dfed sat `not_admitted` from call 106
to call 142 with the open ActionStep fixed at 33 while its input grew from
161k to 563k. Counted in exchanges, the checkpoint absorbs every exchange
before the latest `segments.exact_tail` exchanges, the anchor exchange left
out: the exchange of the latest robot action (its call, its receipt and the
observation after it, the exchange where the current observation arrived),
kept exact however old. The next checkpoint absorbs the exchange it kept once
a newer action has taken the anchor's place. The slot layout keeps the kept
exchange between the slots and the exchanges from the frontier on
(`../summary_slots.ts`).

After a checkpoint, the next request's Provider input is compared with
`target_ratio` of the hard limit (telemetry only). Whether the slot names its
exact transcript is the scheme's `references.pointers` (v8: no).

## Design-space switches

G1 (when), G2 (writer: the policy's compactor), G3 (target and maximum
length, validated, everything before the exact tail).

## Parameters

| Parameter | Values | v8 |
|---|---|---|
| `trigger_tokens` | Provider input tokens, or null: ratio-only, the watermark is `trigger_ratio` of the hard input limit | 150,000 (v8, v9), null (v6) |
| `trigger_ratio` | share of the hard input limit | 0.8 |
| `target_ratio` | share of the hard input limit | 0.65 |
| `checkpoint_target_tokens`, `checkpoint_max_tokens` | estimated tokens | 1,200, 1,600 |
| `enabled` | false only when a context pack turns rollout compaction off | true |
| `unit` | absent or `action_step`; `exchange` (the exact tail must count in exchanges too) | absent |

The exact tail is the scheme's (`segments.exact_tail`, 3 ActionSteps).
A ratio-only L6 (`trigger_tokens: null`, `trigger_ratio: 0.6`) compacts at
about 618k on a 1M window and 146k on a 262k one; management validation keeps
`target_ratio` below `trigger_ratio`.

## Requires, excludes

Requires `materialize`. Reads `episode_capsule`'s admission (its priority)
and slice (its coverage and capsules) when that stage runs. Takes the notice
`superseded_by_episode_capsule`.

## Evidence

Manifest: `emergency_compaction_admission`, `emergency_target_validation`,
`emergency_summarized_through_action_step`, `emergency_compaction_count`,
`emergency_checkpoint_sha256`, `emergency_exact_transcript_ref` and
`resolved_compaction_trigger_tokens`, the watermark absolute or ratio-only
(the legacy manifest carries it for the append-only versions, v8 and v9, not
for v6). Transition
`emergency_compaction` (sizes before and after, packet, checkpoint and
transcript refs, the checkpoint budget, the compactor's metadata, the target).
Artifacts: the cumulative transcript, the packet, the checkpoint. A rejection
is recorded in the summary ledger.

In exchanges the transition also records `unit: exchange`,
`emergency_summarized_exchanges_before`/`_after`, `absorbed_exchanges` and
`kept_exchange`; the manifest adds `emergency_exchange_coverage` (`unit`,
`summarized_exchanges`, `through_exchange_index`, `kept_exchange`) and the
exact tail's record (`exact_tail`, the driver's); its ActionStep fields count
the ActionSteps the checkpoint covers whole. State (in `ora.state`'s
`emergencyExchangeCoverage`): `emergencySummarizedExchangeCount`,
`emergencyKeptExchange`.

State (`ora.state` fields): `emergencySummarizedActionStepCount`,
`emergencyCompactionCount`, `emergencyCheckpoint`, `emergencyCheckpointSha256`,
`emergencyCheckpointRef`, `emergencyTranscriptRef`,
`pendingSemanticTargetEpoch`, `pendingSemanticTargetTokens`,
`lastSemanticTargetValidation`.
