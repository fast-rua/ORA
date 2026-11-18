# episode_capsule (L5)

Phase `summarize`. Code: `stage.ts`, the summary writer in `capsule.ts`.

## What it does

When the Provider reported, for the request before, at least `trigger_tokens`
input tokens on the projection epoch this request starts from, the next
closed Episode (a whole earlier Session round, complete, wholly before the
exact tail, ending in its official round result) is summarized by the
compactor into one `ora.episode-context-capsule.v1`. The capsule slot then
stands in for the Episode's ActionSteps. A rejected summary keeps the
Episode; retries of the same source are bounded (4) and spaced by completed
ActionSteps (4, 8, 16; `retry_attempts`, `retry_backoff_action_steps`). Whether
the slot names its exact transcript is the scheme's `references.pointers` (v8:
no). While an eligible Episode was not rejected, L5 has
priority over L6 on this request.

No sealed run has reached it (no run has a second Session round); a replay
variant with two rounds and inflated Provider usage exercises it on v6, where
its capsule supersedes an emergency checkpoint.

## Design-space switches

G1 (when: an absolute Provider-input watermark, or one scaled with the window
as A1's ratio mode), G2 (writer: the policy's compactor), G3 (target and
maximum length, validated). G4 (off) is a scheme
without this stage, once the manifest format no longer requires its fields.

## Parameters

| Parameter | Values | v8 |
|---|---|---|
| `trigger_tokens` | Provider input tokens, or null: the ratio alone (v10) | 150,000 (v6 and v8) |
| `trigger_ratio` | null: the watermark is `trigger_tokens`; a ratio: max(`trigger_tokens`, ratio × hard input limit), which scales with the window (v4/v5's behaviour), or int(ratio × hard input limit) when `trigger_tokens` is null (formula `int(hard_input_tokens * ratio)`) | null |
| `retry_attempts` | attempts of one Episode's capsule | 4 |
| `retry_backoff_action_steps` | completed ActionSteps a rejected attempt waits, doubling per attempt | 4 |
| `checkpoint_target_tokens`, `checkpoint_max_tokens` | estimated tokens | 1,200, 1,600 |
| `enabled` | false only when a context pack turns rollout compaction off | true |

## Requires, excludes

Requires `materialize`. Reads `emergency_checkpoint`'s slice when that stage
runs: a capsule that covers the active emergency checkpoint supersedes it and
sends `superseded_by_episode_capsule`, on which that stage clears its
checkpoint.

## Evidence

Manifest: `episode_checkpoint_admission` (authority, the anchor tokens and
epoch, the watermark, retry policy and attempts, the eligible round, status
and reason), `resolved_episode_checkpoint_trigger_tokens` and `_formula`,
`episode_checkpointed_through_action_step`, `_rounds`, `episode_checkpoint_count`;
the formula names the scaling (`absolute_tokens`, or
`max(absolute_tokens, int(hard_input_tokens * ratio))`).
Transition `episode_capsule` (sizes before and after, the packet, capsule and
transcript refs, any superseded checkpoint, the checkpoint budget, the
compactor's metadata). Artifacts: the Episode's exact transcript, the
compaction packet, the capsule. A rejection is recorded in the summary
ledger (`last_semantic_rejection`, `semantic_rejection_count`).

State (`ora.state` fields): `episodeCheckpointedActionStepCount`,
`episodeCheckpointedRounds`, `episodeCheckpointCount`, `episodeCapsules`,
`episodeRetryState`.
