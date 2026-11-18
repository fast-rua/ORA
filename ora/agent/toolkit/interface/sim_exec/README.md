# Actions

Stateless category assembled by `interface/assemble.py`. Inputs and responses
retain the existing semantic envelopes. Mutable state belongs to the injected
owner, never this capability. No import of orchestrator or environment.

`execute_action(action, budget)` synchronously returns a semantic envelope with
a public receipt and remaining parent budget. Admission, durable intent, worker
execution and publication belong to the injected Supervisor. Snapshot Actions
never executes: the first action raises `ActionNotExecuted` (an
`AccessDenied`) carrying the proposed action and budget. `submit_action` is the released v1 proposal API.

Robot v2 assembly uses `SynchronousActions`, exposing only `execute_action`.
It has no `submit_action` method, and trusted dispatch removes that backend
adapter too. Legacy v1 `Actions` keeps proposal submission with the single-parent
proposal guard; that guard never limits synchronous primitives.
