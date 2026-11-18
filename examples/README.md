# Examples

## Responsibility

Small, runnable starting points that use ORA's current defaults: a smoke test
of the quickstart, the official-suite cell lists, and a host-profile template.
They show how to drive ORA; they do not reproduce an archived experiment's
score.

| Example | Purpose |
|---|---|
| `uv run --locked ora run --env synthetic` | Real model API, no simulator or GPU; see the [quickstart](../docs/quickstart.md). |
| `uv run --locked python -m examples.quickstart_smoke [--keep DIR]` | The same command against a deterministic local HTTP model: it resolves, preflights, leases, supervises, evaluates and seals one episode without an API key, a simulator or a GPU; CI runs it. `--keep DIR` keeps the run in the workspace `DIR` and prints its run ID and the `ora watch` command that replays it. |
| [official_suite](official_suite/README.md) | List, compose and queue the official LIBERO-Pro and RoboTwin suite cells on the default composition (`run.py`, `scheduler.py`, `experiment.yaml`). |
| [tasks/robodojo-toolkit-v6.yaml](tasks/robodojo-toolkit-v6.yaml) | A task file (`ora.task.v1`) on RoboDojo's default toolkit surface v6: `ora task resolve|run|submit` it with `--task`, `--layout` or `--set` overrides; see [task files](../docs/reference/tasks.md). |
| [hosts/example-host.yaml](hosts/example-host.yaml) | Template of an execution host profile: copy it to `ora/orchestrator/hosts/<id>.yaml` (or a directory named by `ORA_HOSTS_DIR`) and export `ORA_EXECUTION_HOST=<id>`. |
| [LIBERO recipe](../docs/environments/libero.md) | Install and compose one Panda episode. |
| [RoboTwin recipe](../docs/environments/robotwin.md) | Install and compose one dual-arm episode. |
| [RoboDojo recipe](../docs/environments/robodojo.md) | Install and compose one Isaac Sim episode. |

## What belongs here, and what does not

Here: examples that a reader copies or runs as they are, against the current
defaults. Not here: reusable behaviour (it belongs in `ora/`, and an example
calls it), new experiments with their own launchers and results, model keys
and generated runs (keep them outside Git; `outputs/` is ignored).

## Dependency direction

Examples import `ora` and call the `ora` command; `ora` never imports
`examples`. Tests may import the shipped examples (`tests/` imports `ora`,
`viewer`, `examples` and `tests.support` only).

## Entry points

- `uv run --locked python -m examples.quickstart_smoke [--keep DIR]` (from the
  checkout; a Python without ORA's environment stops and says so)
- `python examples/official_suite/run.py {cells,compose,services,queue}`: list
  the official-suite cells, write one cell's RunSpec, qualify the resident
  services and queue the cells across lanes.

## Documentation

- [Quickstart](../docs/quickstart.md) and
  [official suites on a cluster](../docs/getting-started/official-suite.md)
- [Evaluation protocol](../docs/evaluation/protocol.md)
- [Environments](../docs/environments/README.md)
