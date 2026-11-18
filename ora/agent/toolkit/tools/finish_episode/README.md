# finish_episode

Episode exit policy and the existing finish publication format.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool finish_episode`.

Local helpers enforce latched success first, then argument/category/reason and
abort-evidence rules, then pending-action and unconfirmed-success rules. Public
budget, infrastructure signals and offered-candidate ledger are frozen inputs.
Official judgment comes exclusively from api.check_task when enabled; the
context-pack mode without check_task makes no such query or judgment leak.

The request carries the finished record without volatile wall time. The live
adapter validates its fixed schema and stamps time while atomically publishing
policy-finish.json. Replay compares the request against the recorded cursor,
never writes control files, and rejects a changed request as Divergence. Older
records without the required boundary verdict metadata fail NotRecorded.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.
