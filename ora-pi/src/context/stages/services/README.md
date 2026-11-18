# Shared services

What every stage uses and no stage owns. The driver builds them from the
scheme and lends them to the stages through the `RequestView`
(`view.services`, `view.limits`).

| Service | File | Scheme setting | What it holds |
|---|---|---|---|
| limits | `limits.ts` | `limits.reserve_tokens` (2,048) | hard input limit = window − max output − reserve; the telemetry target, `TARGET_INPUT_SHARE` (0.75) of window − max output; the refusal of a window with no input budget |
| estimator | `estimator.ts` | `estimator.image_tokens` (v6 4,096, v8 and v9 530) | the one request-size estimate: text at canonical bytes / 3, each inline image at `image_tokens`, anchored on the Provider's report of the current projection epoch |
| references | `references.ts` | `references` | whether recoverable-artifact Pointers are shown (`pointers`; v8 shows none, so nothing is advertised and summary slots name no transcript), which tool results advertise their Pointer at first delivery (`advertise_over_bytes`), the preview an externalized result keeps (`preview.head_bytes`); the preview strategies, head and head-and-tail (`headTailPreview`, exchange_aging's `preview`); the marker codec (tool-result references, marker detection, stripping the markers of inline images, the artifact marker encoding) |
| segmenter | `segmenter.ts` | `segments.unit`, `segments.exact_tail`, `segments.exact_tail_unit` | ActionSteps (observation groups, their inventory, the open round, the exact tail) and exchanges (one assistant message and what followed it; exchange 0 starts at the opening observation), built once per request after materialization; the exact tail in ActionSteps (absent unit) or in exchanges: the latest `exact_tail` exchanges plus the anchor exchange, the latest robot action's (the exchange where the current observation arrived), however old, with the ActionStep that holds its first message for the ActionStep consumers |

The summarize stages and the lossless invariants count ActionSteps whatever
the unit; `segments.unit` names what the age stages count in (`action_step`
for v6 and v8, `exchange` for v9: exchange_aging requires it).

Named constants elsewhere: the compactor's output ceiling
(`../../rollout_compactor.ts` `COMPACTOR_MAX_OUTPUT_TOKENS`, 8,192), the L4
estimate share (`../pressure_externalize/trigger.ts`
`ESTIMATE_PRESSURE_WINDOW_SHARE`, 48,000 / 197,952) and the summary-preview
ceiling (`references.ts` `SUMMARY_PREVIEW_MAX_BYTES`, 16,384). The L5 retry
schedule and the checkpoint lengths are `episode_capsule` and
`emergency_checkpoint` parameters.
