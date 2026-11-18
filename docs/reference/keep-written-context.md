# Keep what the model wrote (context management v10)

`ora.native-context-management.v10` is v9 ([exchange aging](exchange-aging-context.md))
with the switches a policy needs to keep what the model wrote and age only data.
Its base scheme is `context_keep_written`, the default of RoboDojo in both modes and
of code mode on LIBERO-Pro, RoboTwin, RoboCasa365 and RoboLab since 2026-11-05
([context schemes](context-schemes.md#code-modes-scheme)). `context_keep_written_unaged_reasoning`
(arm H, the default on 2026-11-04 and 05) changes one thing and is an option a
RunSpec selects with `policy.context_scheme`. The third arm,
`context_age_scripts` (B: scripts over 1,024 bytes aged, their unsigned
reasoning rebound), was retired on 2026-11-05; its mechanisms stay for
schemes that age scripts, and its stage list is a test fixture
(`tests/data/context_scheme_fixtures/age-scripts.json`).

## Why

In code mode most of a run's history is what the model wrote. Over 72 sealed
RoboDojo code-mode runs (2026-11-03), reasoning was 64 % of the history that
v9 never ages (median about 420,000 tokens per run), and the scripts the model
wrote were the next largest part. v9 aged scripts over 1,024 bytes; each such
rewrite changed its assistant message, so the message also lost its reasoning,
while `reasoning_withheld_exchanges` reported 0. In one run
(classify_objects-l0) a measurement taken at call 4 was gone from the request
by call 24: its result had aged to a 256-byte preview that showed only the
result's bookkeeping header, and the reasoning and scripts that repeated it had
aged with their messages. And L6 counted ActionSteps, so in code mode, where
one ActionStep holds dozens of exchanges, it could not summarize while the
request grew from 161,000 to 563,000 tokens.

## What ages

At the v9 watermark (`min(100,000, 0.5 × hard input limit)` Provider-reported
input tokens), once 6 exchanges have left the exact tail since the last aging,
one rewrite ages every exchange before the exact tail. In those exchanges:

| Part | `context_keep_written` | `context_age_scripts` (retired) | v9 (`context_rolling_tail_v3`, retired) |
|---|---|---|---|
| Tool-call arguments (scripts) | never | over 1,024 bytes: a reference with their first and last 128 bytes | the same as `context_age_scripts` |
| The model's messages | never | never | never |
| Tool results | over 8,192 bytes: their structured fields and a reference | the same | over 2,048 bytes: a reference with their first and last 128 bytes |
| Tool-returned images, camera images | references | references | references |
| Observation bodies outside the latest 2 | references | references | references |
| Reasoning | kept (`all`) | kept, rebound to the rewritten message when unsigned | dropped with an aged script |

An aged result's structured fields are its `ok`, `status`, `exit_code` and
`actions_executed` (and the domain's marker status keys), and for a failed
one an `error` with a `type` and `message`: the result's own `error` field,
or the last `Type: message` line of its output. They come with `result_bytes`,
`"externalized": true` and the `recoverable_artifact` Pointer that
`artifact_read` reads. A result at or under 8,192 bytes never ages.

**The exact tail** is the latest 6 exchanges plus the exchange of the latest
robot action (its call, its result and the observation after it), however old
(`segments.exact_tail_unit: exchange`, see [context schemes](context-schemes.md#scheme-format-v2-a-scheme-lists-its-stages)).
An aging leaves that exchange as it is too (`exchange_aging.keep_anchor`), and
so does the hard backstop.

**L5 and L6 by ratio.** L5 summarizes a closed Episode at 0.15 of the hard
input limit (about 154,500 tokens on a 1M window, where v9's absolute 150,000
sat). L6 summarizes whole exchanges before the exact tail at 0.6 of the hard
limit, with no absolute cap, down to 0.5. The hard backstop keeps every
request under the hard limit, as in v9.

## Reasoning

No assistant message is rewritten under `context_keep_written`, so every one
keeps its reasoning (`reasoning_replay: all`), a message with no tool call
(reasoning cut off at the output limit, for example) included: it is the
model's writing, and nothing ages it. Reasoning leaves the request only
through:

- the hard backstop, which withholds it from every exchange it ages;
- L5 and L6, which replace whole exchanges with a summary;
- under `context_keep_written_unaged_reasoning` (`unaged_exchanges`), each
  aging, which withholds it from the exchanges it ages, the latest robot
  action's excepted.

`reasoning_binding` says what happens to the reasoning of a message whose
arguments an aging replaced. A continuation state is bound to the hash of
its message (`canonical_message_sha256`). `strict` sends the rewritten
message without it, as v9 did. `rebind_unsigned` binds an unsigned state to
the rewritten message and sends it: the OpenAI-compatible `reasoning_content`
(DeepSeek), which the Provider accepts beside any assistant message. A signed
state (Anthropic thinking blocks, which replay only with the content they
were signed with) is withheld as under `strict`. The retired
`context_age_scripts` used `rebind_unsigned`, so on DeepSeek it aged scripts
without also dropping their reasoning.

## Fields

v10 is v9's closed object (`ora/agent/context/lossless/management_fields.py`,
validated in `management.py`, mirrored in `ora-pi/src/context/management.ts`)
with these changes:

| Field | v10 | `context_keep_written` |
|---|---|---|
| `aged_argument_bytes` | may be null: arguments never age | null |
| `episode_checkpoint_trigger_tokens` | may be null, with `episode_checkpoint_trigger_ratio` set instead | null |
| `episode_checkpoint_trigger_ratio` | new: L5 at int(hard limit × ratio), below `emergency_fallback_ratio`; exactly one of it and the tokens | 0.15 |
| `aged_result_preview` | new: `head_and_tail` (the preview) or `structured` | `structured` |
| `compaction_unit` | new: `action_step` (as v9) or `exchange`: L6 and the exact tail count exchanges; the exact tail is `exact_tail_exchanges` exchanges plus the anchor, and `causal_exact_tail_action_steps` (still 3) has no effect | `exchange` |
| `reasoning_binding` | new: `strict` or `rebind_unsigned` | `strict` |

Its other values: `exchange_aging_trigger_tokens` 100,000 and `_ratio` 0.5,
`exact_tail_exchanges` 6, `aged_result_bytes` 8,192, `aged_preview_bytes` 256,
`backstop_exact_tail_exchanges` 2, `compaction_trigger_tokens` null,
`emergency_fallback_ratio` 0.6, `emergency_target_ratio` 0.5, `camera_images`
`exchange_window_v1`, `image_estimate_tokens` 530.

As stages (`ora-pi/src/context/stages/scheme.ts` `schemeV10`), v10 is v9's
stages with `exchange_aging.argument_bytes`, `result_preview` and
`keep_anchor`, `reasoning_replay.binding`, `episode_capsule.trigger_ratio`,
`emergency_checkpoint.unit` and `segments.exact_tail_unit`. A v2 scheme's
stage list is v10's when its `exchange_aging` states `result_preview`; it must
then state `reasoning_replay.binding` too, keep the anchor exactly when its
exact tail counts exchanges, and count that tail as the aging's
`exact_tail_exchanges`.

## Schemes

| Scheme | Arm | Changes against `context_keep_written` | Prompt | Selected by |
|---|---|---|---|---|
| `context_keep_written` | A (and P, T) | | `keep-written.md` | the default: `robodojo_toolkit_v7` and `v6`, and `native_full_explicit_motion_direct_v4`'s `code_mode_context_scheme` |
| `context_keep_written_unaged_reasoning` | H | `reasoning_replay` `unaged_exchanges` | `keep-written.md` | an explicit `policy.context_scheme` (the default on 2026-11-04 and 05) |

In the context experiment of 2026-11-04 (30 runs per arm), H matched A on success, 12 to 12, with a lower peak input (median 205,000 against 291,000 tokens, at most 317,000 against 598,000) and 22% less input per run; B (`context_age_scripts`, with `aged_argument_bytes` 1,024 and `reasoning_binding` `rebind_unsigned`, prompt `age-scripts.md`) saved little and was retired on 2026-11-05. The user then made `context_keep_written` the default: its arm P (preview optional) had the experiment's best success, 15 of 30. `context_keep_written_unaged_reasoning` has a retention entry (`tests/data/policy-retention.json`). The prompt
tells the model that what it wrote is never shortened, what an aged result keeps, that the
latest robot action's turn stays, and that `artifact_read` and `observe` with
a `ref` return the original.

## Evidence

Besides v9's evidence, a manifest with an exact tail in exchanges records
`exact_tail` and `emergency_exchange_coverage` (the exchange unit's), and its
`exchange_aging` record carries `kept_exchange`,
`reasoning_unbound_messages` (rewritten messages sent without their
reasoning only because their arguments aged) and
`reasoning_rebound_messages` (those whose unsigned reasoning was rebound).
The evaluation (`ora/orchestrator/evaluation/context_coverage.py`) refuses an
unbound or rebound message where arguments never age, a rebound one under
`strict`, and a near tail other than the ActionSteps the exact tail spans.

## Code and tests

- `ora-pi/src/context/management.ts`, `stages/scheme.ts`,
  `stages/exchange_aging/`, `stages/reasoning_replay/stage.ts`,
  `stages/services/references.ts` (`structuredToolMarker`),
  `provider/continuation.ts` (`reboundTo`).
- `ora/agent/context/lossless/management_fields.py`, `management.py`,
  `thresholds.py`; the stage-list bridge
  `ora/agent/composition/schemas/context_stages.py`.
- `ora-pi/test/keep_written.test.ts`;
  `tests/ora/agent/context/lossless/test_context_management_v10.py` (each
  scheme resolves, validates and differs from `context_keep_written` only in
  its own fields); `tests/ora/agent/composition/schemas/test_context_stages.py`;
  `tests/ora/orchestrator/evaluation/test_exchange_unit_evidence.py`.
