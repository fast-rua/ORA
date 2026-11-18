# reasoning_replay

Phase `age`, after `exchange_aging`. Code: `stage.ts`.

## What it does

Decides which assistant messages are sent with their provider continuation
state (the reasoning the Provider replays):

- `all`: every assistant message an aging leaves unchanged keeps it;
- `unaged_exchanges`: an aging also withholds it from every exchange it ages,
  in that same rewrite, so below the aging watermark nothing is withheld and
  the prefix is rewritten only where aging rewrites it anyway;
- `recent_exchanges`: only the latest `tail_exchanges` to
  2 × `tail_exchanges` − 1 exchanges keep it; once 2N exchanges carry it, one
  rewrite (`reasoning_replay_window`) withholds it back to the latest N, so the
  prefix is rewritten once per N exchanges.

Whatever the mode, an assistant message whose arguments an aging replaced is
no longer the message its state is bound to. Under `binding: strict` (or
absent) it is sent without it; under `rebind_unsigned` (v10) an unsigned
state (`provider/continuation.ts` `UNSIGNED_CONTINUATION_KINDS`: the
OpenAI-compatible `reasoning_content`) is bound to the rewritten message and
sent with it, the original kept as private metadata for the lossless
invariant view, while a signed one (Anthropic thinking blocks, which replay
only with the message they were signed with) is withheld as under strict.
The hard backstop withholds it from every exchange it ages. The exchange an
aging leaves as it is (exchange_aging `keep_anchor`) keeps it in every mode.
At materialization
(`store`) every continuation state is stored as a Context Artifact
(`context_provider_continuation`); a withheld state stays on the message as
private metadata, and the lossless invariant view compares it by identity.

## Design-space switches

F1–F3 (reasoning replay).

## Parameters

| Parameter | Values | v10 default (`context_keep_written`) |
|---|---|---|
| `mode` | `all`, `unaged_exchanges`, `recent_exchanges` | `all` (`unaged_exchanges` in `context_keep_written_unaged_reasoning`) |
| `tail_exchanges` | ≥ 1 for `recent_exchanges`, null otherwise | null |
| `binding` | absent or `strict`, `rebind_unsigned` (v10) | `strict` |

## Requires, excludes

Requires `materialize` and `exchange_aging`, and runs after it (it reads its
admission: an aging moves the reasoning frontier itself, by mode, through a
notice).

## Evidence

Pass `exchange_reasoning_withheld` (measured with exchange_aging's passes);
the reasoning fields of `exchange_aging` (with an exact tail in exchanges
also `reasoning_unbound_messages`, rewritten messages withheld only because
their arguments aged, and `reasoning_rebound_messages`, those whose unsigned
state was rebound; `reasoning_withheld_exchanges` counts the frontier
alone); transition
`reasoning_replay_window`. State: `reasoningWithheldExchangeCount`.
