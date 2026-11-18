# camera_delivery

Phase `deliver`. Code: `stage.ts`, rendering in `delivery.ts`.

## What it does

Decides where the fixed camera images of observations reach the model, and
strips the internal recoverable-artifact markers of inline images from the
delivered history (v8's references show no Pointer; the codec is
services/references.ts). The marker of an image an exchange aging removed has
no image after it and stays: it is the reference the model is shown.

- `rolling_tail_v1`: every observation message keeps one stable text
  reference instead of its camera images; one message after the whole history
  carries the camera images of the latest `tail_observations` observations
  (oldest first, the last marked current). Only that message changes from one
  request to the next, so the history before it stays a cacheable prefix.
- `inline_until_compaction_v1`: the images stay in their observation message
  until a summary absorbs its ActionStep.
- `exchange_window_v1` (needs `exchange_aging`): the same here; the exchange
  aging removes them from the observation message when the exchange it
  arrived in ages, leaving each image's marker (`observe(ref=...)` reopens
  it).

Tool-returned images are untouched here (exchange_aging ages them). The history it renders (without the
tail) is what the summarize stages compact.

## Design-space switches

E1 (camera image window): `rolling_tail`, `inline_until_compaction` and
`exchange_window` (management v9). Free play's latest-N window is an option
this stage is to gain.

## Parameters

| Parameter | Values | v8 |
|---|---|---|
| `mode` | `rolling_tail_v1`, `inline_until_compaction_v1`, `exchange_window_v1` | `rolling_tail_v1` (v9: `exchange_window_v1`) |
| `tail_observations` | ≥ 2 in rolling mode, null otherwise | 2 (v9: null) |

## Requires, excludes

Requires `materialize`. Excludes `media_window`: both would decide where
camera images go.

## Evidence

The manifest's `camera_delivery` (`ora.camera-delivery.v1`): mode, window,
the retrieval the references name (`observe.step` when the advertised observe
tool takes `step`; `observe.ref` for `exchange_window_v1`), and in rolling mode the tail (`action_step_index`,
`observation_step`, cameras), its image count and each observation's
camera-image count. The request assembler relates consecutive requests of one
epoch as `camera_tail_append` by the tail's header. No state, no transition.
