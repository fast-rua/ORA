# Exchange aging (context management v9)

`ora.native-context-management.v9` is [append-only context
(v8)](append-only-context.md) plus three things: **exchange aging**, a lossless
rewrite at a Provider-input watermark; a **hard backstop** that keeps every
request under the hard limit; and **reasoning replay**, which by default
keeps the reasoning of every assistant message an aging leaves unchanged and
can instead withhold an aged exchange's reasoning in the aging that ages it.
Every v8 field keeps its v8 meaning; L1–L4 still never run, camera images
follow `camera_images`, and L5/L6 and the checkpoint validation are those of
v8. v9 adds the camera mode `exchange_window_v1`, which every v10 scheme uses:
camera images are one more kind of aged content and leave with their exchange
([camera images](#camera-images)).

In the Pi core v9 runs as context stages (`ora-pi/src/context/stages/`):
`materialize`, `exchange_aging`, `reasoning_replay`, `camera_delivery`,
`episode_capsule` (L5), `emergency_checkpoint` (L6) and `hard_backstop`, with
its segments counted in exchanges ([stages](#stages)). The stages reproduce the
reference implementation (`1a9c1b117`) byte for byte on the context goldens'
v9 cases. No scheme selects v9 since 2026-11-05: its schemes
(`context_rolling_tail_v3` and three comparison arms, never a default) were
retired, sealed runs keep their own commits, and their stage lists are test
fixtures (`tests/data/context_scheme_fixtures`). The values below are theirs.
[Management v10](keep-written-context.md) is v9 with the switches that keep
what the model wrote: arguments that never age, aged results that keep their
structured fields, L6 and the exact tail in exchanges.

See the [context management design space](../design/context-management-design-space.md)
for what this design trades off, its switches and the open experiments (a design
record, not a contract).

## Why

v8 rewrites history only at L5 (a closed Episode) and L6 (a model-written
checkpoint of the completed ActionSteps before a 3-step exact tail). In code
mode most model calls (perception, failed scripts) complete no ActionStep, and
small self-hosted models often write a checkpoint the validator rejects, so on
a 262,144-token window v8 sent requests over the window and the Provider
refused them (11 RoboDojo code-mode runs ended `trusted_fault` this way). In
the final over-limit requests, tool-call arguments (whole scripts rewritten
with heredocs), tool results and observation text were 70–86% of the input.

## Exchanges

An exchange is one assistant message and every message after it up to the
next assistant message: its tool results, the images a tool returned, the
observations that arrived. Exchange `k` starts at the `k`-th assistant message
after the static prefix; the opening observation ages with exchange 0. Unlike
the ActionStep, every model call is an exchange, so code mode ages. In code
mode an exchange is one script written and run: code mode's action step, the
counterpart of the ActionStep that tool-call mode's ActionStep-based layers age
by.

## Exchange aging

While the same-epoch Provider-reported input is at or above the aging
watermark

```
min(exchange_aging_trigger_tokens, int(hard_input_tokens * exchange_aging_trigger_ratio))
```

and at least `exact_tail_exchanges` exchanges have left the exact tail since
the last aging (the batch), one rewrite (a new projection epoch, transition
`exchange_aging`, `semantic_loss: false`) ages every exchange older than the
latest `exact_tail_exchanges`. In an aged exchange:

- a tool call's `arguments` longer than `aged_argument_bytes` become
  `{"arguments_bytes": N, "arguments_externalized": true, "preview": "<head>\n[...
  M bytes cut ...]\n<tail>", "recoverable_artifact": {Pointer}}`;
- a tool result longer than `aged_result_bytes` becomes the v4–v7
  recoverable-artifact marker: its status fields, a preview, `externalized:
  true` and the Pointer (mutation-tool results stay exact), with
  `preview_truncated`, `preview_bytes` (the bytes of the result the preview
  keeps) and `result_bytes`;
- each preview is the first and the last half of `aged_preview_bytes` (128 and
  128 bytes of 256) in UTF-8 bytes, never splitting a character, around
  `\n[... M bytes cut ...]\n`, where M is the bytes between them; a string
  within the budget is kept whole (`headTailPreview`,
  `ora-pi/src/context/stages/services/references.ts`). A
  result whose tool declares a bounded `summary` keeps that summary instead,
  as in v4–v7. The v4–v7 layers' previews keep their beginning;
- every image a tool returned (the user message a tool's images arrive in,
  or an image an action's tool published with the observation, such as
  RoboTwin's closed-skill receipt views) and, under `camera_images:
  exchange_window_v1`, every camera image of an observation leaves the
  request: the `<recoverable_artifact>` marker materialization put before it
  stays as its Pointer, whose opener is `observe(ref=...)` (v4–v7's L3
  references, kept in the delivered history only for an image no longer
  inline); a camera image's label stays before its marker. Under
  `rolling_tail_v1` camera images keep the rolling tail;
- an observation's measured body becomes its identity fields and a
  `recoverable_observation` Pointer if, when the aging ran, the observation
  was not one of the camera tail's latest `camera_tail_observations` (the
  latest 2 under a mode without a tail). The aging fixes that set of
  observations (`aged_observation_groups`); an aged observation that a newer
  observation pushes out of the camera tail keeps its body until the next
  aging.

The originals are Context Artifacts (`context_tool_call_arguments`,
`context_tool_result`, `context_observation`, the images') that
`artifact_read` and `observe(ref=...)` return, so v9 sets `context_tools:
["artifact_read"]` (which also gives `observe` its `ref` mode); resolution
refuses v9 without it.
An assistant message whose arguments were replaced is no longer the message
its provider continuation state is bound to (`replay_policy:
required_unmodified`), so it is sent without that state, which is stored as a
`context_provider_continuation` artifact. Whether the other assistant messages
of aged exchanges keep their state is `reasoning_replay`'s (below): under the
default `all` they keep it.

What a request shows is a function of the history and of frontiers that move
only at an aging, a reasoning-window stride or the backstop, each a new
projection epoch: between them a message already delivered never changes (the
request assembler refuses a request that would change one,
`cache_prefix_rewritten`), so the history only grows and a Provider prefix
cache holds. The
batch keeps rewrites batched (at most one per `exact_tail_exchanges` calls)
while they keep happening as long as the request stays high, even when what
aging cannot touch (the fixed prefix, the exact tail, a camera tail) already
exceeds the watermark. The admission record names why it did or did not age:
`provider_usage_unavailable`, `projection_epoch_unanchored`,
`below_watermark`, `awaiting_exchange_batch` or
`new_exchanges_since_last_aging`, with the pending count. L5/L6 are admitted
as in v8, on the same report.

## Camera images

Under `camera_images: exchange_window_v1` (`camera_tail_observations` null,
v9 only) camera images are not delivered separately: an observation's camera
images stay inline, after their labels, in the history message where the
observation arrived, and leave the request only when the exchange that
observation belongs to ages: in an aging batch, in the hard backstop, or when
an L5/L6 summary absorbs its span. So below the watermark nothing is rewritten
and every camera image is inline, part of the cached prefix; above it, the
camera images of the latest 6 to 11 exchanges are inline and older ones are
their references; the opening observation ages with exchange 0, as its text
does. A delivered message never changes between agings, and the request
assembler relates consecutive requests of one epoch as `append_only`.

Every observation of an exchange keeps its images until that exchange ages;
there is no rule that keeps only an exchange's latest observation. The history
receives one observation per turn: in a RoboDojo code-mode run
(2026-11-03, 157 model calls) 145 fresh observations were published to the
scripts during their commands, but the history held 62 observation messages
and no exchange carried more than one. Two observations share an exchange only
in exchange 0 (the opening observation and the first action's) and at a round
boundary, and the opening observation is sent before exchange 0 exists, so a
latest-only rule could not be applied before it is first delivered.

The same window can age the latest observation's images when 6 exchanges
without an observation follow it (2 of the 157 requests of that run);
`observe` shows the current views. Holding that run's aging frontiers fixed
(3 cameras, about 530 tokens per image), the window delivered about 2,900
more camera-image tokens per request than the 2-observation camera tail (median
1,800, at most 10,800; 47 of 157 requests fewer), 2.4% of the run's Provider
input. `rolling_tail_v1` (the retired scheme `context_rolling_tail_v3_camera_tail`)
keeps the camera tail of [append-only context](append-only-context.md#camera-images)
under v9: aging never touches its camera images.

## Hard backstop

Before a request is sent, after every layer (a rejected or impossible L6
included), v9 estimates it: the provider-anchored suffix estimate when the
epoch has a Provider report, otherwise the Harness estimate scaled by the last
measured Provider/estimate ratio (never scaled down), otherwise the Harness
estimate. If that exceeds the hard limit (`context_window_tokens -
max_output_tokens - 2048`), every exchange but the latest
`backstop_exact_tail_exchanges` is aged, reasoning and tool-returned images
included (transition `hard_backstop`). If the request still exceeds the
limit, the turn ends with `context_exhausted`, as v4–v7 end it, before the
request is sent. Under
`exchange_window_v1` the backstop takes the camera images of every
observation outside its exact exchanges. v9 never sends
a request it estimates over the limit; v8's `context_over_limit` warning does
not exist in v9. The backstop is the one rewrite that is not batched: it runs
only for a request the Provider would otherwise refuse.

Management v6 runs the same backstop since 2026-11-14 (its optional
`backstop_exact_tail_exchanges`, [context schemes](context-schemes.md#v6s-hard-backstop-2026-11-14)).
v6 has no exchange stages, so the backstop keeps and renders its own frontier
(`hard_backstop` `aging: own`, `ora-pi/src/context/stages/hard_backstop/own_render.ts`):
reasoning, read-only tool results (to v6's head preview) and tool-returned
images of every exchange but the latest 2; observations stay whole. It ages
from L6's watermark on (`trigger_ratio`, 0.8 of the hard limit, by the
request's own estimate), not only over the hard limit, since v6 has no lossless
layer inside its exact ActionStep tail.

## Reasoning replay

`reasoning_replay` decides which assistant messages are sent with their
provider continuation state (the replayed reasoning). The withheld states are
stored as `context_provider_continuation` artifacts; messages that keep their
state are unmodified.

- `all` (`context_rolling_tail_v3`'s, and `context_keep_written`'s): every assistant message an
  aging leaves unchanged keeps it, aged exchanges included, as v8 does.
- `unaged_exchanges`: only exchanges that have not aged keep it. An aging
  withholds it from exactly the exchanges it ages, in the same rewrite: the
  reasoning frontier moves to the aged frontier, so below the watermark
  nothing is withheld, the prefix cache is untouched until an aging and the
  withholding costs no rewrite of its own. `context_keep_written_unaged_reasoning` (the default on 2026-11-04 and 05, an
  option since) selects it (v9's comparison arm
  `context_rolling_tail_v3_unaged_reasoning` did before). It exists because, measured on
  DeepSeek high-effort RoboDojo runs with `all`, a request was still about
  171,000 tokens after 24 of 30 exchanges had aged, about two thirds of it the
  replayed reasoning of aged exchanges (median 4,700 tokens per call), so
  requests climbed past L6's 150,000 and the rollout compaction still ran in
  41 of 49 runs.
- `recent_exchanges`: only the latest `reasoning_tail_exchanges` exchanges'
  assistant messages keep it. To keep the prefix cache, the window slides in
  strides: once `2N` exchanges carry their state, everything before the
  latest `N` is sent without it (transition `reasoning_replay_window`), so
  between `N` and `2N - 1` exchanges keep it and the history is rewritten once
  per `N` exchanges. An aging also moves the window. No scheme selects it
  since the retired `context_rolling_tail_v3_recent_reasoning` (N = 6).

The hard backstop withholds the reasoning of every exchange it ages under
every mode.

## Fields

v9 is v8's closed object plus these fields, all required
(`ora/agent/context/lossless/management_fields.py`, validated in `management.py`, mirrored in
`ora-pi/src/context/management.ts`):

| Field | `context_rolling_tail_v3` (retired) | Meaning and why |
|---|---|---|
| `exchange_aging_trigger_tokens` | 100000 | Absolute aging watermark; must be below `compaction_trigger_tokens`. Two thirds of L6's 150,000: on a 262,144 window with 16,384 output tokens (hard limit 243,712) aging comes 50,000 tokens before L6. Applied (by estimate) to the three measured final over-limit requests, aging would have left 42–52% of them. |
| `exchange_aging_trigger_ratio` | 0.5 | Hard-window cap; must be below `emergency_fallback_ratio`, so aging precedes L6 on every window (60,416 on a 131,072 window with 8,192 output tokens). |
| `exact_tail_exchanges` | 6 | Latest exchanges an aging keeps exact (the current write-run-read-fix loop of a script), and the batch: an aging waits for this many newly aged exchanges, so the prefix cache is rewritten at most once per 6 calls and holds between batches. A separate batch field would only let the batch differ from the tail; aging fewer exchanges than the tail per rewrite buys little, and more would let the request run further past the watermark. |
| `aged_argument_bytes` | 1024 | Short commands (`python step.py`) stay; heredoc scripts (median 3.3–4.2 KB in the measured runs) are stored. |
| `aged_result_bytes` | 2048 | Stores a full bash result (median 1.9–3.0 KB) and keeps short ones. |
| `aged_preview_bytes` | 256 | The preview budget of a stored argument or result: its first and last 128 bytes stay beside its Pointer, around a marker of the bytes cut (exchange_aging's `preview`, `{head_bytes: 128, tail_bytes: 128}`). A script prints its conclusion last, so a head alone dropped it (head only until 2026-11-03). |
| `backstop_exact_tail_exchanges` | 2 | Exchanges the backstop keeps exact; at most `exact_tail_exchanges`. |
| `reasoning_replay` | `all` | `all`, `unaged_exchanges` or `recent_exchanges` ([above](#reasoning-replay)). The default replays the reasoning of every assistant message an aging leaves unchanged; withholding aged exchanges' reasoning (about two thirds of a request after most exchanges had aged, measured with `all`) is the `unaged_exchanges` comparison arm. |
| `reasoning_tail_exchanges` | null | `N` for `recent_exchanges`; null for `all` and `unaged_exchanges`. |

## Schemes

The four v9 schemes, `context_rolling_tail_v3` (v9, `exchange_window_v1`,
`reasoning_replay: all`, prompt `rolling-tail-v3.md`), `_camera_tail`
(`rolling_tail_v1`, 2 observations, prompt `rolling-tail-v3-camera-tail.md`),
`_unaged_reasoning` and `_recent_reasoning` (N = 6), were `ora.context-scheme.v2`
schemes from 2026-11-04 to 2026-11-05, never selected by a baseline, and were
retired with their prompts once `context_keep_written_unaged_reasoning` ([keep-written
context](keep-written-context.md)) became the default. Their stage lists are
test fixtures in `tests/data/context_scheme_fixtures`.

## Evidence

Each `context_projected` record of a v9 projection carries v8's
`resolved_compaction_trigger_tokens` and `camera_delivery`, no
`context_over_limit`, and adds:

| Field | Content |
|---|---|
| `resolved_exchange_aging_trigger_tokens` | The aging watermark of this window |
| `exchange_aging` | `ora.context-exchange-aging.v1`: `exchanges`, `exact_tail_exchanges`, `aged_exchanges` (exchanges before this frontier are aged), `reasoning_replay`, `aged_observation_groups` (observations before this frontier may show a Pointer for their body), `reasoning_withheld_exchanges`, `aging_count`, this projection's `admission` (status, reason, the Provider report, the watermark, `new_exchanges_since_last_aging` and `batch_exchanges`), `delivered`: the history messages (before a compacted span is dropped) whose `arguments`, `tool_results`, `tool_images`, `camera_images` (under `exchange_window_v1` only), `observations` or `reasoning` an aging changed, `delivered_tool_images_externalized`: the tool-returned images the request itself carries as references, and under `exchange_window_v1` `delivered_camera_images_externalized`: the observations' camera images it carries as references |
| `hard_backstop` | null, or `{status: "applied", estimate_basis, pre_estimated_input_tokens, post_estimate_basis, post_estimated_input_tokens, hard_input_tokens, backstop_exact_tail_exchanges, hard_backstop_count}` |
| `request_estimate` | `{tokens, basis, hard_input_tokens}`: the estimate the backstop compared with the limit for the request sent |

The passes add `exchange_argument_reference`, `exchange_tool_result_reference`,
`exchange_tool_image_reference`, `exchange_camera_image_reference` (under
`exchange_window_v1` only), `exchange_observation_reference` and
`exchange_reasoning_withheld`, each with its changed messages and artifact
refs. Under `exchange_window_v1` `camera_delivery` names `observe.ref` as the
retrieval and has no tail. Transitions `exchange_aging`, `reasoning_replay_window` and
`hard_backstop` record the three frontiers before and after, the changed message
counts and the request size before and after; an aging also cites the
Provider report at or above the watermark and its `batch_exchanges`. The
semantic slots carry their exact-transcript Pointer, as in v4–v7. The pipeline
state (`ora.state`) adds the frontiers and the counts as one `exchangeAging`
object (`exchangeAgedCount`, `exchangeAgedObservationGroups`,
`reasoningWithheldExchangeCount`, `exchangeAgingCount`, `hardBackstopCount`).

The exchange passes share one measurement: the first,
`exchange_argument_reference`, goes from the history before the aging to the
history after it, and the others record no size change, so the whole
reduction is credited to the argument pass. The reference did the same, and
the stages keep it: splitting it per pass changes every v9 manifest, so it
belongs to the per-stage evidence format (plan unit K5a).

Evaluation (`ora/orchestrator/evaluation/context_coverage.py`) applies v8's
append-only checks, with the delivered image count less the
`delivered_tool_images_externalized` references (and under
`exchange_window_v1` less the `delivered_camera_images_externalized` ones,
which that mode must record), and adds: the resolved
watermark matches the formula; the aged span leaves at least the backstop's
exact exchanges; nothing is a reference before an exchange aged, camera images
included; under `all`
no reasoning is withheld outside the aged span, and under `unaged_exchanges`
`reasoning_withheld_exchanges` equals `aged_exchanges`; `request_estimate` and every
backstop's result are within the hard limit
(`lossless_context_request_over_hard_limit` otherwise). The projection walk
(`lossless.py`) requires the three frontiers to only grow (the observation
frontier never past the camera tail's observations) and every v9 transition
to be lossless and to move a frontier, and an aging to move the aged frontier by
at least its batch and to cite a Provider report at or above its watermark on
the epoch it left (`exchange_aging_admission_invalid` otherwise). The artifact
provenance audit treats v9 as Pointer-publishing: every Pointer in a tool
result, observation, aged tool or camera image marker, semantic slot or (where the
arguments differ from what the model proposed) an aged tool call must resolve
in the catalog with a matching identity.

## Stages

| Order | Stage | Phase | Parameters from v9 |
|---|---|---|---|
| 1 | `materialize` | materialize | reasoning excluded |
| 2 | `exchange_aging` | age | `exchange_aging_trigger_tokens`, `exchange_aging_trigger_ratio`, `exact_tail_exchanges`, `aged_argument_bytes`, `aged_result_bytes`, `aged_preview_bytes` (as `preview`, head and tail) |
| 3 | `reasoning_replay` | age | `reasoning_replay`, `reasoning_tail_exchanges` |
| 4 | `camera_delivery` | deliver | `camera_images`, `camera_tail_observations` |
| 5 | `episode_capsule` | summarize | as v8 |
| 6 | `emergency_checkpoint` | summarize | as v8 |
| 7 | `hard_backstop` | guard | `backstop_exact_tail_exchanges` |

Each stage directory (`ora-pi/src/context/stages/<stage>/README.md`) says what
the stage does, which switches it owns and the evidence it writes.

Scheme-level: `segments` (unit `exchange`, exact tail 3 ActionSteps for the
summaries), `limits` (reserve 2,048), `estimator.image_tokens`
(`image_estimate_tokens`, 530) and `references` (Pointers shown, none
advertised at first delivery). The aged arguments and continuation states are
stored at materialization by the stages that may leave them out (each
stage's `store`). The hard backstop is the guard and sizes every request
itself, so the driver's rule for a request the Provider reported over the
limit does not apply.

## Code and tests

- `ora-pi/src/context/stages/exchange_aging/`: `render.ts` (the stored
  arguments, the aged render), `rewrite.ts` (one exchange rewrite and its
  transition record, the request estimate), `stage.ts` (admission, the
  manifest fields); `reasoning_replay/`, `hard_backstop/`;
  `camera_delivery/delivery.ts`: keeps the marker of an image no longer inline
  (under `exchange_window_v1` the camera images stay inline until the exchange
  aging removes them); `services/references.ts`: the head-and-tail preview;
  `projection.ts` `losslessInvariantView`: stored arguments and withheld
  continuation states compare by identity; `stages/scheme.ts` `schemeV9`.
- Context-level replays of four sealed v9 code-mode runs (LIBERO-Pro,
  RoboTwin, two RoboDojo), each with its own settings, must equal what the
  reference tree produced, byte for byte (until 2026-11-05 also under
  `context_rolling_tail_v3`).
- `ora-pi/test/exchange_aging.test.ts`: the crossing, repeated batches above
  the watermark, aged tool images, an aged observation leaving the camera tail
  between agings (no rewrite until the next aging), each reasoning-replay mode
  through the OpenAI wire projection (`unaged_exchanges`: nothing withheld
  below the watermark, exactly the aged exchanges' reasoning withheld by an
  aging), the backstop (tool images included, under `all` and
  `unaged_exchanges`) and its refusal, v8 unchanged; under
  `exchange_window_v1` every camera image inline below the watermark with each
  request extending the last, an aging that takes out exactly the aged
  exchanges' camera images (the opening observation's with exchange 0) and the
  backstop that takes those outside its 2-exchange tail; every request passes
  the request assembler's prefix check. `ora-pi/test/context_stages.test.ts`:
  the v9 stage list, the state snapshot, the stages' requirements.
- `tests/agent_core/test_exchange_aging_context.py`: Synthetic Pi episodes
  without a completed action that age at the watermark, age again in batches
  while the request stays above it, drop aged tool images from the request,
  an action episode under `exchange_window_v1` whose observations keep their
  camera images until their exchange ages, a backstopped episode and a
  `context_exhausted` one, each completed run evaluating valid.
  Recorded LIBERO-Pro and RoboTwin sessions replay under v9 with a low
  watermark.
- `tests/ora/agent/context/lossless/test_context_management_v9.py`: the
  closed object and the watermark order;
  `tests/ora/agent/composition/schemas/test_context_stages.py`: the four
  schemes resolve to the reference line's management objects.
- `tests/ora/orchestrator/evaluation/test_context_coverage.py`: reasoning
  withheld per mode and the aged camera images under the exchange window.
