# Scripts

Repository-local commands. They add no hidden behaviour; run `--help` on any of
them for the full argument surface. Run them from the repository root.

Everything an operator or author runs is a subcommand of `ora` (`ora --help`
lists them by group, the [command reference](../docs/usage/cli-reference.md)
prints every one) or of `ora-viewer`; `scripts/` keeps only the wrappers that
select an interpreter for them and the checks that are not Python tools. A new
tool belongs in `ora/app/` as a command, not here.

`scripts/ora` selects the locked project interpreter: the checkout's `.venv`,
else an explicit `$PYTHON`, else uv's project environment
(`$UV_PROJECT_ENVIRONMENT`, used as it is and never synced), and only then
`python3` from `PATH`, which it says on stderr. The other shell wrappers source
`scripts/lib/python.sh`, which has the same order and, when none of those is
set and `uv` is installed, lets `uv run --locked python` create or sync the
checkout's `.venv` from `uv.lock` before falling back to `python3`. A set
`$UV_PROJECT_ENVIRONMENT` or `$UV_NO_SYNC` never leads to `uv run`, so a shared
environment is never re-synced. Each wrapper puts the checkout first on
`PYTHONPATH`. `scripts/ora-remote` and `scripts/setup-robodojo` need only the
standard library and take `python3` without the `uv run` step or the notice.
The Python tools `scripts/dev/compare-model-visible` and
`scripts/dev/refresh-vendored-pins` run under the interpreter that starts them
(`python3` from their `#!` line, or `"$PYTHON" scripts/dev/<tool>`).

## User commands

| Script | What it does |
|---|---|
| `scripts/ora` | Wrapper for the `ora` command (`python -m ora.app`, `ora/app/main.py`): setup, local and single-GPU runs, interactive sessions, watch, plan/launch/evaluate/seal, report, spec, qualify, suite, services, providers, ladder and the `dev` authoring tools (`ora --help` lists them by group). It keeps its own copy of the interpreter order above (without the `uv run` step), so it also works on its own. See the [command guide](../docs/usage/commands.md) and the generated [command reference](../docs/usage/cli-reference.md). |
| `scripts/install.sh` | The one-line user installer (`curl -fsSL .../scripts/install.sh \| sh`), POSIX `sh` because it runs before ORA or its interpreter exists: installs uv and Node.js 22 when missing, checks out ORA under `~/.local/share/ora/ORA`, runs `uv sync --all-extras --locked` and the agent core's `npm ci` there, and writes an `ora` command into `~/.local/bin` that runs `scripts/ora` with that Node. It changes nothing outside those directories and refuses a checkout with local changes or an `ora` command it did not write ([install](../docs/getting-started/installation.md#one-line-install)). |
| `scripts/ora-remote` | Run `ora` on a remote host over SSH and show its terminal UI and world view locally through a loopback tunnel (`python -m ora.app.remote`). |
| `scripts/setup-robodojo` | Prepare the pinned isolated RoboDojo runtime outside Git, one stage at a time; it does not launch an episode (`ora setup --env robodojo` runs every stage). It stays a wrapper because the installer it runs by path (`ora/environment/backends/robodojo/installation/setup_runtime.py`) and that installer's `INSTALL.md` are pinned qualification inputs that name it. |
| `scripts/docs` | Preview (`serve`) or strictly build (`build`) the Blume documentation site in `docs-site/`, with its own locked dependencies. |
| `scripts/yam-reset` | Kept for the YAM cell's operators: forwards to `scripts/ora cell` (`scripts/yam-reset home` is `ora cell home`; alone it runs `ora cell diagnose`). The verbs are `ora cell`'s ([real robot cells](../docs/usage/real-cells.md)). |

The commands that used to be wrappers here: `ora services` (was
`scripts/model-services`), `ora report` (`scripts/run-report`), `ora dev
prompt-bundle` and `ora dev task-contracts` (`scripts/dev/audit-*`), and
`ora-viewer --open-browser`, `ora-viewer compile videos` and `ora-viewer compile
previews` (`scripts/open-viewer`, `scripts/dev/compile-viewer-videos`,
`scripts/dev/compile-libero-previews`).

## Developer tools (`scripts/dev/`)

Checks and maintenance tools that are not `ora` commands. None calls a provider
or starts a simulator.

| Script | What it does |
|---|---|
| `scripts/dev/check` | CI's checks locally: after `npm ci` in `ora-pi` and `ora-pi/web`, five lanes run at once (A: the Python suite under pytest-xdist; B: the agent-core tests, `-m agent_core` (`tests/agent_core/`), with `ORA_TEST_REQUIRE_AGENT_CORE=1`; C: the TypeScript agent-core tests and ORA's web page; Y: the YAM station's MuJoCo tests under `$YAM_PYTHON`, reported as skipped where it is unset or lacks MuJoCo and i2rt; D: `policy.lock`, whitespace and, in the development repository, its `internal/` checks), then the documentation build and the no-write check of `ora baselines --explain`. Each step logs to its own file; a summary and a non-zero exit report failures. `--quick` installs only `ora-pi`'s packages and runs lanes A, Y and D, `--public` leaves out `internal/`, `--no-docs` the documentation build. Lane B's quickstart cases preflight a clean checkout, so commit first (lane A's run on a snapshot of the working tree). When the interpreter lacks pytest-xdist it is layered on with `uv run --no-sync`. |
| `scripts/dev/compare-model-visible` | `BASE CANDIDATE`: resolve a fixed matrix of offline `native_api` RunSpecs (the default cell of each of the five backends and the Synthetic quickstart, each of those in code mode with the base and the extended library, `ora local`'s default task and YAM's free play) at both commits (each in a scratch copy of its tree, with this script's cells) and report which tools, prompt layers or projected fields the model would see differently. Exit 0 when nothing differs. Run it before integrating a change to prompts, tools, observation projection or composition. |
| `scripts/dev/refresh-vendored-pins` | Recompute the install revision of the vendored RoboTwin planner after editing `ora/environment/backends/robotwin/vendored/` and rewrite the current pins; older pins are kept. |

## Other entry points

| Entry | Purpose |
|---|---|
| `ora/app/main.py` (`python -m ora.app`) | The `ora` command behind `scripts/ora` and the installed `ora` console script: one parser, the command table and the grouped help; each command is one module in `ora/app/`. |
| `examples/official_suite/run.py` | List, compose and queue the official LIBERO-Pro and RoboTwin suite cells. |
| `ora/environment/backends/robotwin/qualification.py` | RoboTwin expert/oracle seed qualification (installed as `robotwin-qualify`); `ora qualify robotwin qualify` starts it per lane. |
| `viewer/app.py` (`ora-viewer`) | The read-only Viewer: `ora-viewer --open-browser [--page runs\|libero\|libero-pro\|robotwin\|robodojo]` serves it, `ora-viewer compile videos\|previews` builds its videos and LIBERO gallery previews offline, `ora-viewer dashboard --outputs ROOT` serves the read-only run dashboard (`viewer/dashboard/`). |

`ora/app/README.md` lists every module started with `python -m` or by path (the
simulator worker, the supervisor, the probes and the other process entries);
none is an operator command.

## Shared shell code (`scripts/lib/`)

| File | What it does |
|---|---|
| `scripts/lib/python.sh` | Sourced by the shell wrappers and `scripts/dev/check`: `ora_python NAME` sets `PYTHON` in the order above and puts the checkout on `PYTHONPATH`; `ora_python --stdlib NAME` skips the `uv run` step and takes `python3` from `PATH` without the notice. |

## Adding a script

Add a script only when it carries behaviour that cannot live in the `ora` CLI
(a new operator or authoring tool is an `ora` command in `ora/app/`). Put user
commands in `scripts/` and maintenance tools in `scripts/dev/`, give it a
`--help` description, select its interpreter with `scripts/lib/python.sh`, and
add its row here in the same change:
`tests/ora/app/test_main.py::test_documented_script_and_test_module_inventory_is_current`
fails on any undocumented script.
