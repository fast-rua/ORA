# check_task

The sole capability-gated data route to official task judgment.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool check_task`.

The tool accepts no arguments, preserves the open-observation precondition,
and projects the original success/task_success or unavailable result from
api.check_task. It cannot read a verdict from configuration, artifacts or files.

Replay returns only the recorded value at this cursor and boundary. Missing
records fail NotRecorded; disabled capability fails AccessDenied. No final
verdict is backfilled. Optional verdict metadata is for the finish policy and
is not added to this tool's model-visible result.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.
