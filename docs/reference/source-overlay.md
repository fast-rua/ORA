# Source overlays

A run normally starts only from a clean commit: `source` in the RunPlan is
`{commit, dirty}` and preflight, launch, the Supervisor and sealing refuse any
other checkout. A **source overlay** lets an outside author, such as an RSI
agent's evaluation, run its own tools on top of one committed base without
making a commit of it and without re-qualifying anything.

## What may be replaced

Only files under `ora/agent/toolkit/tools/`, `ora/policy/tool_surfaces/` and
`ora/policy/task_files/` (a run's own task file, `policy.task_file`),
and `policy.lock` (regenerated over them). No file a qualification record
hashes lives there: the RoboDojo worker and play kernels,
`ora/orchestrator/episode/action_executor.py`, `pyproject.toml`, `uv.lock` and
the RoboTwin seed-scan inputs stay the base commit's, so card, task and lane
qualification records of the base commit still apply.

## The overlay file

The checkout states its overlay in `.ora-source-overlay.json` at its root
(ignored by git):

```json
{
  "schema": "ora.source-overlay.v1",
  "base_commit": "<the checkout's HEAD, 40 hex>",
  "files": {"ora/agent/toolkit/tools/move_to/tool.py": "<sha256>", "...": null},
  "provenance": {"agent_repo": "...", "agent_commit": "..."}
}
```

`files` lists every path that differs from the base commit, with the sha256 of
its bytes, or `null` for a path removed from the base. `provenance` is recorded
as given. New files must be staged (`git add`) so that `./scripts/ora lock`
covers them; staging does not change the check below.

## The check

`ora.orchestrator.spec.source_overlay.source_state` accepts the checkout when
HEAD is `base_commit` and every difference from HEAD (tracked, staged or
untracked, as `git status --untracked-files=all` lists it) is a listed path
whose bytes match. Plan resolution then records `source` as the base commit,
clean, and the overlay whole, with the sha256 of its canonical JSON, as the
plan's `source_overlay` block, which the plan hash covers. Preflight
(`source_identity`), launch, the Supervisor and sealing require the same
overlay again. Without the file, nothing changes: a checkout that differs from
its commit is dirty and refused, as before.

To reproduce a run, check out its `source.commit`, restore the files of
`source_overlay.files` and write the overlay file back.
