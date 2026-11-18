# Append-only context (context management v8)

Context management v4–v7 are recoverable-first: as an ActionStep ages, its
read-only results, media and observation bodies become recoverable-artifact
Pointers (layers L1–L4), and only then is anything summarized (L5/L6). Every
aging step rewrites the projected history, so a Provider prefix cache is
rewritten with it.

`ora.native-context-management.v8` is append-only. Between rollout compactions
the projected history only grows: L1–L4 never run, nothing ages into a Pointer,
and no recoverable-artifact Pointer or reader is shown to the model. Only L5
(a closed Episode) and L6 (the rollout compaction) rewrite the history, as the
`projection_epoch` they open records. Fixed camera images, which arrive with
every observation and lose their value quickly, follow `camera_images`.

v8 runs on `action_step_lossless_v4` retention with `rollout_checkpoint_v1`,
so the artifact catalog, the compactor, the lossless invariants and their
evaluation are those of v4–v7. In the Pi core it runs as stages
(`ora-pi/src/context/stages/`): the management object is translated into
`materialize`, `camera_delivery`, `episode_capsule` (L5),
`emergency_checkpoint` (L6) and `over_limit_guard` (send and record), which a
stage driver runs on every request ([Code and tests](#code-and-tests)). The
requests, the manifest and the `ora.state` snapshot are those of the earlier
pipeline, byte for byte. [Exchange aging (v9)](exchange-aging-context.md) is
v8 plus a lossless rewrite of old exchanges at a Provider-input watermark and
a hard backstop, and [management v10](keep-written-context.md), v9 that keeps
what the model wrote, is what the maintained schemes select.

No scheme selects v8 any more ([context schemes](context-schemes.md)). Its
scheme `context_rolling_tail_v2` (`rolling_tail_v1`, 2 observations,
`context_tools: []`: no `artifact_read`, no `observe(ref=...)`, and its own
prompt `rolling-tail-v2.md`) was the context of `robodojo_toolkit_v7` and
`robodojo_toolkit_v6` until `context_keep_written_unaged_reasoning` replaced it on 2026-11-04
([keep-written context](keep-written-context.md)), and was retired on
2026-11-05; sealed runs keep their own commits, and the v8 code stays for the
v9 and v10 schemes that build on it. The schemes `context_rolling_tail_v1` (the same policy
without the prompt) and `context_append_until_compact_v1`
(`inline_until_compaction_v1`) were retired on 2026-10-31; sealed runs keep
their recorded definitions, and the `inline_until_compaction_v1` camera mode
below stays implemented for them.

## Fields

The object is closed: every field is required and no other is accepted
(`ora/agent/context/lossless/management_fields.py`, validated in `management.py`, mirrored in
`ora-pi/src/context/management.ts`).

| Field | Meaning |
|---|---|
| `schema` | `ora.native-context-management.v8` |
| `causal_exact_tail_action_steps` | Completed ActionSteps a compaction always keeps exact, with their images. Must be 3. |
| `camera_images` | `rolling_tail_v1` or `inline_until_compaction_v1` (below); v9 also takes `exchange_window_v1`. |
| `camera_tail_observations` | Observations the rolling tail carries, at least 2; null for `inline_until_compaction_v1` and `exchange_window_v1`. |
| `episode_checkpoint_trigger_tokens` | L5: summarize a closed Episode once same-epoch Provider input reaches this. |
| `compaction_trigger_tokens` | L6: summarize every ActionStep before the exact tail once same-epoch Provider input reaches `min(compaction_trigger_tokens, emergency_fallback_ratio x hard limit)`. |
| `emergency_fallback_ratio` | The hard-window cap of the L6 watermark. |
| `emergency_target_ratio` | The post-L6 request-size target, telemetry only. Below the fallback ratio. |
| `checkpoint_target_estimated_tokens`, `checkpoint_max_estimated_tokens` | The checkpoint's requested size and soft maximum. |
| `image_estimate_tokens` | Tokens the request estimate charges per inline image. |

The hard limit is `context_window_tokens - max_output_tokens - 2048`; v4–v7
keep the L6 watermark at `emergency_fallback_ratio` of it.

## Camera images

The fixed camera images of an observation are the image blocks the
observation message labels `{LABEL} · sequence {n} · {camera} · latest at this
message`, for the cameras its `_causal_observation` lists. Other images,
including every tool-returned image, are never moved: they stay where the tool
delivered them until a compaction absorbs their ActionStep, and leave with it
(the checkpoint is text). v9 also ages tool-returned images with their
exchange.

| Mode | Management | Camera images in a request | Schemes |
|---|---|---|---|
| `rolling_tail_v1` | v8, v9, v10 | one reference line per observation in the history; the latest `camera_tail_observations` observations' images in one message at the end | none since 2026-11-05 (the retired `context_rolling_tail_v2` and `_v3_camera_tail`; sealed runs) |
| `inline_until_compaction_v1` | v8, v9, v10 | inline in their observation message until a compaction absorbs it | none (sealed runs) |
| `exchange_window_v1` | v9, v10 | inline in their observation message until the exchange it arrived in ages; then each image's recoverable reference | `context_keep_written`, `context_keep_written_unaged_reasoning` |

**`inline_until_compaction_v1`.** Camera images stay in their observation
message until a compaction absorbs it. Consecutive requests of one epoch are
`append_only`.

**`exchange_window_v1`** (v9 only; management v8 refuses it). The same as
`inline_until_compaction_v1` until the observation's exchange ages: an
exchange aging, the hard backstop or a compaction of its span. The aging then
removes the camera images from the observation message and leaves each
image's label and its `<recoverable_artifact>` marker, which `observe(ref=...)`
reopens, in a new projection epoch; between agings consecutive requests are
`append_only`. Images are not maintained separately: they are one more kind
of aged content under the exchange window ([exchange
aging](exchange-aging-context.md#camera-images)).

**`rolling_tail_v1`.** Every observation message replaces its camera images
with one reference that never changes afterwards:

```
Camera images of observation step 4 (head, left_wrist, right_wrist) are not repeated here: while it is one
of the latest 2 observations they are attached once at the end of the request; afterwards observe with step=4
shows them again.
```

The retrieval clause appears only when the advertised `observe` takes `step`.
Each request then ends with one user message, the camera tail: the camera
images of the latest `camera_tail_observations` observations, oldest first,
each after its label with `· observation step {step}` (and `· current` on the
newest). The tail is built from the full history, so it keeps rolling after a
compaction. Only that last message differs between consecutive requests of
one epoch; the request assembler checks that the previous request without its
tail is an exact prefix of the current one, in canonical messages and in wire
units, and records the relation `camera_tail_append`.

A rollout compaction summarizes the history as the model was shown it: the
compactor receives the references, not the camera images that left the
history.

## When the window is exceeded

v4–v7 end the turn with `context_exhausted` once the Provider reports more
input than the hard limit (v6 plans resolved since 2026-11-14 estimate every
request first instead, with [the hard backstop](context-schemes.md#v6s-hard-backstop-2026-11-14)).
v8 sends the request anyway: the compaction that
should have shrunk it ran when its watermark was crossed, and only the
Provider can tell whether the request still fits. The projection records the
warning, and a request the Provider refuses is a provider failure of the run.
v9 replaces this with its hard backstop: it never sends a request it estimates
over the limit ([exchange aging](exchange-aging-context.md#hard-backstop)).

## Evidence

Each `context_projected` record of a v8 projection adds:

| Field | Content |
|---|---|
| `resolved_media_retention_policy` | `append_only_until_compaction_v1` |
| `resolved_compaction_trigger_tokens` | The L6 watermark of this window |
| `camera_delivery` | `ora.camera-delivery.v1`: the mode and window, the retrieval it names (`observe.step` in rolling mode when observe takes a step, `observe.ref` under `exchange_window_v1`, else null), and in rolling mode the tail (`action_step_index`, `observation_step`, cameras), its image count and each observation's camera-image count |
| `context_over_limit` | null, or `{status: "warning_sent_anyway", provider_input_tokens, hard_input_tokens, estimated_input_tokens, over_limit_request_count}` |

`resolved_inline_image_target` is null and the lifecycle and pressure
watermarks stay unset. `active_inline_images` and the request estimates
describe the delivered request.

Evaluation (`ora/orchestrator/evaluation/context_coverage.py`) checks that no
layer aged, that the camera record matches the plan's management, that the
tail is the inventory's latest observations, and that the delivered image
count equals the history's images outside the compacted span, minus their
camera images in rolling mode, plus the tail's. The artifact provenance audit
refuses any recoverable-artifact Pointer in a v8 request
(`append_only_context_published_pointer`).

## Code and tests

- `ora-pi/src/context/stages/scheme.ts`: `schemeFromManagement`, the v8
  object as its five stages with their parameters.
- `ora-pi/src/context/stages/driver.ts`: runs the stages in phase order, feeds
  the summary stages the delivered history (without the camera tail), applies
  the over-limit rule of the guard's parameters, keeps the state record in the
  pipeline's `ora.state` fields; `manifest_v6.ts` writes the manifest.
- `ora-pi/src/context/stages/camera_delivery/`: the delivered history and the
  camera tail; `episode_capsule/`, `emergency_checkpoint/`, `over_limit_guard/`:
  L5, L6 and the warning. Each stage directory has a README (what the stage
  does, its switches, its evidence). Under `exchange_window_v1` the
  `exchange_aging` stage takes an aged exchange's camera images out.
- `ora-pi/src/context/request_assembly.ts`: the `camera_tail_append` relation.
- `tests/agent_core/test_append_only_context.py`: Synthetic Pi episodes in both
  modes, a compaction and an over-limit request, each evaluating valid.
