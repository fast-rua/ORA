# pressure_externalize (L4)

Phase `age`. Code: `stage.ts`, the passes in `render.ts`, the estimate
watermark in `trigger.ts`.

## What it does

When the request is under pressure and the read-only frontier is ahead of its
own, every remaining read-only result and every older observation body before
the read-only frontier becomes a recoverable reference. Pressure is any of:
more than `inline_images` inline images; an estimate above the larger of
`estimated_tokens` and `window_share` of the hard input limit; a Provider
report on this epoch at or above `provider_input_ratio` of the hard limit. It
is not evaluated while `stride_aging` has aging pending. A frontier that
changes nothing visible stays moved without a transition.

## Design-space switches

A1 (trigger: inline images, estimate floor or window share, reported ratio),
A3 (signal: estimate and Provider report), C2 and C4 (what ages: all
remaining read-only results and observation bodies).

## Parameters

| Parameter | Values | v6 |
|---|---|---|
| `inline_images` | images | 24 |
| `estimated_tokens` | tokens | 48,000 |
| `window_share` | share of the hard input limit | 48,000 / 197,952 |
| `provider_input_ratio` | share of the hard input limit | 0.7 |
| `enabled` | false only when a context pack turns L4 off | true |

With `enabled: false` the stage stays in the scheme: it renders (its passes
are listed unchanged), keeps `resolved_inline_image_target`, and is never
admitted.

## Requires, excludes

Requires `materialize` and `stride_aging`, and runs after it.

## Evidence

Manifest passes `l4_read_only_result_reference` and
`l4_observation_reference`, `resolved_inline_image_target`, and the frontier
behind `pressure_externalized_through_action_step`. Transition
`pressure_externalization` (its reasons, frontier before and after, sizes
before and after). State (`ora.state` field):
`pressureExternalizedActionStepCount`.
