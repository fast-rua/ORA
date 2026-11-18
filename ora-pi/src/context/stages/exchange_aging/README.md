# exchange_aging

Phase `age`. Code: `stage.ts`; what an aged exchange shows, `render.ts`; what
the exchange stages share (the rewrite, its counts, the request estimate),
`rewrite.ts`.

## What it does

Ages exchanges (one assistant message and what follows it up to the next:
its tool results, the images they returned, the observations that arrived;
in code mode one script written and run) into recoverable references, in
batches, at a Provider-input watermark (management v9).

- **Admission.** The watermark is `min(trigger_tokens, hard limit ×
  trigger_ratio)`. While the same-epoch Provider-reported input of the request
  before is at or above it, one rewrite ages every exchange before the exact
  tail (the latest `exact_tail_exchanges`), once at least
  `exact_tail_exchanges` exchanges have left that tail since the last aging.
  Rewrites come in batches (one projection epoch each, the prefix cache holds
  between them) and keep coming while the request stays high.
- **What ages**, in every exchange before the frontier: a tool call's
  arguments longer than `argument_bytes` become a JSON reference
  (`arguments_externalized`, their size, a head-and-tail `preview`, the
  `recoverable_artifact` Pointer; reasoning_replay then sends that assistant
  message without its continuation state); a tool result longer than
  `result_bytes` becomes a head-and-tail preview and its Pointer (mutation
  tools' results stay); every image a tool returned is left as the marker
  materialization put before it, which `observe(ref=...)` reopens; under
  camera_delivery `exchange_window_v1` an observation's camera images too; an
  observation's measured body becomes a reference when its observation group
  is before the observation frontier, fixed at each aging (or backstop) to the
  groups outside the camera tail (the latest 2 without a tail), so a new
  observation never rewrites an aged one.
- **Management v10** (`keep-written`, docs/reference/keep-written-context.md):
  `argument_bytes` null leaves every tool call's arguments, so every
  assistant message, as written; `result_preview: structured` makes an aged
  result keep its status fields (`ok`, `status`, `exit_code`,
  `actions_executed` and the domain's marker status keys), a failed one's
  `error` type and message (its own `error` field, or the last `Type:
  message` line of its output), `result_bytes` and the Pointer, with no
  output text (`services/references.ts` `structuredToolMarker`);
  `keep_anchor`, with an exact tail in exchanges, leaves the anchor exchange
  (the latest robot action's call, receipt and observation) as it is at
  every aging and backstop, as the exact tail does.
- **Storage.** At materialization (`store`) every tool call's arguments
  longer than `argument_bytes` are stored as Context Artifacts
  (`context_tool_call_arguments`) and recorded on the private materialized
  message, so `artifact_read` returns the original and the lossless invariant
  view restores it by identity.

Every rewrite is lossless (`semantic_loss: false`); frontiers only grow.

## Design-space switches

A1–A5 (when to compress: Provider-reported watermark, ratio cap, batches),
B1–B2 (the unit: exchanges), C1–C4 (what ages: arguments, results, tool
images, observation text), D1 (the head-and-tail preview).

## Parameters

| Parameter | Values | v10 default (`context_keep_written`) |
|---|---|---|
| `trigger_tokens` | ≥ 1 | 100,000 |
| `trigger_ratio` | (0, 1) | 0.5 |
| `exact_tail_exchanges` | ≥ 1 (the exact tail and the batch) | 6 |
| `argument_bytes` | ≥ 1 (v9 schemes had 1,024), or null: arguments never age (v10) | null |
| `result_bytes` | ≥ 1 | 8,192 |
| `preview` | `{head_bytes, tail_bytes}` | 128 and 128 (`aged_preview_bytes` 256) |
| `result_preview` | absent or `head_and_tail` (the preview), `structured` (v10) | `structured` |
| `keep_anchor` | absent or false; true with an exact tail in exchanges (v10) | true |

The watermark stays below `emergency_checkpoint`'s, so the lossless aging
comes before the summarizing compaction (management validation).

## Requires, excludes

Requires `materialize` and `reasoning_replay`; the scheme's segments unit is
`exchange`. Reads camera_delivery's mode and tail.

## Evidence

Passes `exchange_argument_reference`, `exchange_tool_result_reference`,
`exchange_tool_image_reference`, `exchange_camera_image_reference`
(`exchange_window_v1` only) and `exchange_observation_reference`, after the
L1–L4 passes. They share one measurement with reasoning_replay's pass, as the
reference implementation did: the first pass carries the whole reduction (K5a
splits it per pass). `resolved_exchange_aging_trigger_tokens`, and
`exchange_aging` (`ora.context-exchange-aging.v1`): exchanges, exact tail,
the frontiers, reasoning mode and withheld exchanges, aging count (with
`keep_anchor`, `kept_exchange`; with an exact tail in exchanges,
`reasoning_unbound_messages` and `reasoning_rebound_messages`, reasoning_replay's), this
request's `admission` (`reason`: `provider_usage_unavailable`,
`projection_epoch_unanchored`, `below_watermark`, `awaiting_exchange_batch`
or `new_exchanges_since_last_aging`), the changed messages per pass
(`delivered`) and the images the request carries as references. Transition
`exchange_aging`. State (in `ora.state`'s `exchangeAging`):
`exchangeAgedCount`, `exchangeAgedObservationGroups`, `exchangeAgingCount`,
and with `keep_anchor` `exchangeKeptExchange`.
