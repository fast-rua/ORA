# eager_externalize (L1/L2)

Phase `age`. Code: `stage.ts`, the passes in `render.ts`, the transition
record in `transition.ts`.

## What it does

When `stride_aging` moves the read-only frontier, the read-only tool results
of the ActionSteps before it change: one larger than `large_result_bytes`
becomes a recoverable reference with the scheme's `references` preview (L1), and a
later exact duplicate becomes a reference without one (L2). Results of tools
that change the world, results with an exact causal outcome and artifact
reads stay exact. If the new frontier changes nothing the model sees, the
frontier stays moved and the request, and its epoch, stay as they were.

## Design-space switches

C2 (what ages: large and duplicate read-only results) on the stride's
schedule. The preview (D1) is the scheme's references setting.

## Parameters

| Parameter | Values | v6 |
|---|---|---|
| `large_result_bytes` | bytes | 50,000 |
| `deduplicate` | `true` (the one option) | `true` |

## Requires, excludes

Requires `materialize` and `stride_aging`, and runs after it.

## Evidence

Manifest passes `l1_large_result_reference` and
`l2_exact_duplicate_reference`. Transition
`lifecycle_read_only_externalization` (frontier before and after, the changed
events, sizes before and after). Artifacts: the externalized results. State
(`ora.state` field): `lifecycleReadOnlyExternalizedActionStepCount`.
