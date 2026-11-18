# hard_backstop

Phase `guard`. Code: `stage.ts` (the estimate and the exchange stages'
rewrite are `../exchange_aging/rewrite.ts`; what it shows when it ages by
itself, `own_render.ts`).

## What it does

Keeps every request under the hard input limit (window − max output −
reserve) before it is sent. After every other stage, the request is estimated:
anchored on the Provider's report of the current epoch, otherwise the raw
estimate scaled by the last measured Provider/estimate ratio (never down),
otherwise the raw estimate. Over the limit, all but the latest
`exact_tail_exchanges` exchanges are aged in one rewrite (`hard_backstop`),
reasoning included and, under camera_delivery `exchange_window_v1`, camera
images; a request still over the limit ends the turn with
`context_exhausted`. It runs only for a request the Provider would otherwise
refuse, so it is the one exchange rewrite that is not batched.

Because it sizes every request itself, the driver's rule for a request the
Provider reported over the limit does not apply under it (`overLimit:
own_estimate`); there is no `context_over_limit` record.

**Aging by itself** (`aging: own`, management v6 with
`backstop_exact_tail_exchanges`, the default scheme `context_lossless_v6`'s
since 2026-11-14). v6 has no exchange stages, so the backstop keeps its own
frontier (`aged`, exchanges from the first) and renders it; the driver runs a
guard's render at the end of the age phase, so the lossless invariants check
it. In every exchange before the frontier, the open ActionStep's and the exact
ActionStep tail's included: assistant messages are sent without their
reasoning, read-only tool results become the scheme's head preview and
Pointer (as L4 renders one; mutation results and exact causal outcomes stay),
and tool-returned images in their own message are left as their markers
(`observe(ref=...)`). Observation messages stay whole, so the latest
observation keeps its images and body.

**From a watermark** (`trigger_ratio`; v6: L6's `emergency_fallback_ratio`,
0.8). The backstop then ages as soon as the request's own estimate, after
every other stage, reaches that share of the hard limit, not only over the
limit (reason `estimated_input_at_or_above_watermark`, the record's
`watermark_tokens`). So L6's watermark holds inside an open ActionStep, where
L6 has nothing to absorb. Within the limit it ages only when L6 cannot act on
the next report: it defers while L6 has fresh ActionSteps and only waits for
a Provider report at its watermark (L6's admission `below_high_watermark` or
`projection_epoch_unanchored`), so lossless aging never keeps the reports
under L6's watermark while L6 has something to summarize; and only when its
frontier moves by at least `exact_tail_exchanges` exchanges, so a request
that stays above the watermark is not rewritten on every call. A request
whose latest exchanges alone reach the watermark is sent as it is while it
fits under the hard limit; over the limit the backstop always runs.

Without it, v6 guarded only after the
Provider's report: L6 summarizes completed ActionSteps before a 3-step exact
tail, so inside one open ActionStep (ea103b427290: 19 calls, one robot action)
nothing could shrink, and L6 is admitted on the report of the request before,
one call late, while a call can add a whole output limit of reasoning; seven
RoboLab tool-call runs (2026-11-14) sent requests estimated 9,000-36,000 tokens over the hard
limit and ended on the report.

## Design-space switches

H1 (estimate first and age down to the latest exchanges), H3.

## Parameters

| Parameter | Values | v10 default (`context_keep_written`) | v6 (`context_lossless_v6`) |
|---|---|---|---|
| `exact_tail_exchanges` | ≥ 1, at most exchange_aging's | 2 | 2 |
| `aging` | absent (the exchange stages render), `own` | absent | `own` |
| `trigger_ratio` | absent (over the hard limit only), or a share of the hard limit in (0, 1) | absent | 0.8 (L6's `emergency_fallback_ratio`) |

## Requires, excludes

Requires `exchange_aging` and `reasoning_replay` (it moves their frontiers
through notices); with `aging: own`, excludes them instead. A scheme has at
most one guard.

## Evidence

The manifest's `hard_backstop`: null, or `{status: "applied",
estimate_basis, pre_estimated_input_tokens, post_estimate_basis,
post_estimated_input_tokens, hard_input_tokens,
backstop_exact_tail_exchanges, hard_backstop_count}`; `request_estimate`
(`tokens`, `basis`, `hard_input_tokens`) of every request. Transition
`hard_backstop`. State: `hardBackstopCount`.

With `aging: own`: the transition also records `aging: own`, its aged and
reasoning frontiers (one frontier, both) and `externalized_messages`
(`tool_results`, `tool_images`, `reasoning`); the manifest adds
`hard_backstop_aging` (`aging`, `exchanges`, `exact_tail_exchanges`,
`aged_exchanges`, `delivered`, `delivered_tool_images_externalized`) and the
passes `exchange_tool_result_reference`, `exchange_tool_image_reference` and
`exchange_reasoning_withheld` (one measurement, credited to the first). Its
frontier is the `ora.state` snapshot's `exchangeAging.exchangeAgedCount`. The
evaluation (`context_coverage.py`) checks the record and the request estimate
and counts the externalized images out of the inline ActionSteps' images.
