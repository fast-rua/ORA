## What and why

<!-- One change per pull request: what it changes and why. Link the issue if there is one. -->

## How it was tested

<!-- The commands you ran and their summary lines (scripts/dev/check, or the
     pytest, node and docs commands from CONTRIBUTING.md). For a simulator,
     controller or asset change: the simulator and asset versions, tasks and
     seeds you ran on the real simulator, and what you did not run. -->

## Checklist

- [ ] `tests/repository/test_structure.py` passes: layers import downward, tests mirror the code, every package has its README.
- [ ] What the model sees is unchanged (`scripts/dev/compare-model-visible <base> HEAD` reports no difference), or every difference is explained above.
- [ ] Evidence formats and schemas are unchanged, or the change, its `docs/reference/` page and how runs sealed before it still evaluate are described above.
- [ ] The docs and package READMEs that describe the changed code are updated (`docs/reference/` for a contract; `./scripts/ora dev cli-reference --write` for a command).
- [ ] New tests sit in the mirrored tree (`tests/ora/<package path>/` for `ora/<package path>/`) and import no other test module.
- [ ] `policy.lock` is current if `ora/policy/` changed; no lockfile changed unless this pull request is about that dependency.
- [ ] No file listed under "Files that may not move" in `docs/developer/repository-structure.md` moved or changed, or the re-qualification it needs is described above.
- [ ] A moved or removed module left no compatibility shim; its callers are rewritten.
