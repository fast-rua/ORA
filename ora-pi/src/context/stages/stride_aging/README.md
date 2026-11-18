# stride_aging

Phase `age`, first of the age stages. Code: `stage.ts`, the plan in `plan.ts`.

## What it does

Decides, once per request, whether the age stages move their frontiers and to
where. Read-only bodies keep one exact near window (the scheme's exact tail or
the media tail, whichever is longer); media keep the current Episode's latest
`media_window` tail. A frontier moves only when it lags its target by at least
`stride` ActionSteps, or when a new Episode releases the earlier one's media,
so a request between two aging points changes nothing. While aging is pending,
L4 pressure is not evaluated.

It has no render and no state. Its admission on each request is the plan
(`TailAging`: the targets, the frontiers to move to, the reasons, whether
anything is pending), which `eager_externalize`, `media_window` and
`pressure_externalize` read; it is never applied.

## Design-space switches

A4 (batch: the stride, or none) for the ActionStep unit; the read-only near
window is derived from the scheme's exact tail and the media tail.

## Parameters

| Parameter | Values | v6 |
|---|---|---|
| `stride` | ActionSteps | 4 |
| `strided` | `true`: the tails are minimums that age in batches of `stride` and the manifest reports the aging; `false`: every frontier moves at each fresh observation, `stride` 1, no `tail_aging` record (v4's behaviour) | `true` |

## Requires, excludes

Requires `materialize` and `media_window` (it reads the media tail). The age
stages that read its plan run after it.

## Evidence

Manifest: `resolved_near_tail_action_steps`, `resolved_media_retention_policy`
(`current_round_action_step_tail_stride_v1`, unstrided
`current_round_action_step_tail_v1`), and, when strided,
`resolved_tail_aging_stride_action_steps` and `tail_aging`
(`ora.context-tail-aging.v1`: aged, reasons, targets, pending steps, whether
pressure was evaluated or deferred). Its `check` refuses read-only coverage
inside the near window.
