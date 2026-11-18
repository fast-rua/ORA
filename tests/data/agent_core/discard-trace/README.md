# Discarded-action traces

Frozen agent traces for `tests/ora/agent/evidence/test_discarded_action_trace.py`.
Each `<case>.jsonl` is the `agent-events.jsonl` of one session that submitted an
action the Supervisor never executed, closed (or not) as its name says;
`native/agent-artifacts/` holds every artifact they reference, by content hash.

| Case | What the session did after its proposal |
|---|---|
| `discarded` | discarded it on `model_call_budget`; the run ended on that budget |
| `unclosed` | nothing; the run ended on `model_call_budget` |
| `termination-mismatch` | discarded it on `model_call_budget`; the run ended on `wall_time_budget` |
| `reason-policy-finished` | discarded it naming `policy_finished_episode` |
| `reason-none` | discarded it naming no reason |
| `activity-after-discard` | discarded it, then ran another turn on the same observation |

They were recorded with the Python Native core, before ORA removed it, on the
legacy two-tool Synthetic boundary (`native_minimal_v1` Context, a scripted
model). The traces are evidence in the format sealed runs keep, so they are not
re-recorded: a change that makes the verifier read them differently changes
how sealed runs evaluate.
