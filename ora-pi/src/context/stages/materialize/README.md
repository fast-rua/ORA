# materialize

Phase `materialize`, always the first stage of a scheme.

## What it does

Builds the canonical base of a request from its raw history, once per
request: recoverable payloads are registered in the Context Artifact catalog
(images by marker and reference, large tool results by reference, inline media
through the shared media cache), and reasoning of earlier assistant messages
is excluded. Every later stage renders on this base; the driver's lossless
invariants compare the aged history with it.

## Design-space switches

None of its own. Which tool results advertise their Pointer at first delivery
is the scheme's `references.advertise_over_bytes` (services/references.ts): v8
shows no Pointer, so nothing is advertised; v6 advertises results larger than
its L1 threshold.
`reasoning` is `exclude`, the only option until `reasoning_replay` (F1)
lands as its own stage.

`truncated_reply_reasoning: withheld` (F, management v6's optional
`truncated_reply_reasoning`, the default scheme `context_lossless_v6`'s since
2026-11-14): a reply the Provider's output limit cut off before any tool call
is sent without its provider continuation state, its reasoning, from the first
request that carries it, so no delivered message changes. Such a reply is an
assistant message with no tool call followed by the core's truncation
continuation (`core/continuations.ts` `afterTruncatedOutput`: "Your previous
response reached the Provider output limit before proposing an action..."),
which stays and tells the model the reply was cut. The reply contributed no
decision; its reasoning, up to the whole output limit (32,768 tokens on the
RoboLab tool-call runs of 2026-11-14, 38 such replies in six of the seven runs that ended
`context_exhausted`), would otherwise be replayed on every later request. The
state stays on the message as private metadata and the trace keeps the reply
whole (its `provider_response` event).

## Parameters

| Parameter | Values | v8 | v6 (`context_lossless_v6`) |
|---|---|---|---|
| `reasoning` | `exclude` | `exclude` | `exclude` |
| `truncated_reply_reasoning` | absent (replayed) or `withheld` | absent | `withheld` |

## Requires, excludes

Nothing; every other stage requires it.

## Evidence

Two manifest passes, each measured from the raw to the materialized history:
`recoverable_payload_materialization` (its artifact refs and the messages
that advertise a Pointer at first delivery) and `reasoning_exclusion` (the
messages whose reasoning was excluded, truncated replies withheld included);
with `truncated_reply_reasoning`, the manifest's `truncated_reply_reasoning`
(`mode`, `withheld_messages`, `withheld_event_ids`: the truncated replies the
request carries without their reasoning). Artifacts in the Context Artifact
catalog. No state, no transition.
