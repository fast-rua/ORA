# over_limit_guard

Phase `guard`.

## What it does

Declares what happens to a request the Provider reported over the hard input
limit (window − max output − 2,048) on the projection epoch it is sent at
(`overLimit`, from its parameter). The driver applies the rule before the
guard phase: `refuse` ends the turn
with `context_exhausted`; `send_and_record` sends the request (the summaries
that should have shrunk it already ran; the Provider decides whether it
fits) and this stage records it. A scheme without a guard refuses.

It reacts to the previous Provider report; it does not estimate the request
before sending it, so one call that adds more than the margin (a reply cut at
the output limit replays up to the whole limit as reasoning) sends a request
over the limit and the run ends on its report. v9's
[`hard_backstop`](../hard_backstop/README.md) (H1, estimate first and age to
the latest exchanges) is the other guard, and v6's since 2026-11-14 (`aging:
own`, management v6's `backstop_exact_tail_exchanges`); v6 plans resolved
before then keep this guard.

## Design-space switches

H1 in its reactive form (v4–v8).

## Parameters

| Parameter | Values | v8 |
|---|---|---|
| `over_limit` | `refuse` (v4–v7), `send_and_record` | `send_and_record` (v8), `refuse` (v6 without `backstop_exact_tail_exchanges`) |

## Requires, excludes

Nothing.

## Evidence

With `send_and_record`, the manifest's `context_over_limit`: null, or
`{status: "warning_sent_anyway", provider_input_tokens, hard_input_tokens,
estimated_input_tokens, over_limit_request_count}`. State
(`ora.state` field, written only when non-zero): `overLimitRequestCount`.
