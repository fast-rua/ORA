# media_window (L3)

Phase `age`. Code: `stage.ts`, the pass in `render.ts`.

## What it does

When `stride_aging` moves the media frontier, the images of the ActionSteps
before it become recoverable references (their markers stay), and an earlier
round's static-facts slot ages with its round's media; the current round's
slot never does. The completed ActionSteps of the current Episode within
`tail_action_steps` keep their media inline, and a new Episode releases the
earlier one's. A frontier that changes nothing visible stays moved without a
transition.

## Design-space switches

E (images, the ActionStep window of v6): the media tail.

## Parameters

| Parameter | Values | v6 |
|---|---|---|
| `tail_action_steps` | completed ActionSteps of the current Episode | 2 |

## Requires, excludes

Requires `materialize` and `stride_aging`, and runs after it. Excludes
`camera_delivery`: both would decide where camera images go.

## Evidence

Manifest pass `l3_lifecycle_media_reference`, and the media frontier behind
`lossless_externalized_through_action_step`,
`lifecycle_media_externalized_through_action_step` and
`protected_inline_media_action_steps`. Transition
`lifecycle_media_externalization` (frontier before and after, the media tail,
the protected current round, sizes before and after). State (`ora.state`
fields): `lifecycleMediaExternalizedActionStepCount`,
`lifecycleSupersededToolImageEventIds` (no producer adds a superseded
tool-image event yet; the field is carried for restore).
