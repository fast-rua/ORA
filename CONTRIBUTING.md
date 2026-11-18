# Contributing

Thank you for helping with ORA. This page is the path from a fresh clone to a
pull request: set up, put the code where it belongs, run the checks that apply
to your change, and open the PR with what a reviewer needs.
[Start contributing](docs/developer/onboarding.md) is the short version for
your first hour; [AGENTS.md](AGENTS.md) holds the same rules for coding agents;
the [code map](docs/code-map.md) says which package owns what.

## Before you start

- Branch from `main` (in a fork, unless you have write access) and open one
  pull request per change: a fix, a feature, a move. Split a large change into
  a series of pull requests that each pass the checks on their own.
- For a bug, a small fix or a documentation change, open a pull request
  directly, or an issue first if you want to agree on the approach.
- For a new environment backend, a change to a schema, the evidence format, a
  default, or what the policy may see or do, open an issue or a draft PR that
  describes the contract before you implement it. The issue templates (bug
  report, feature request, new environment, design RFC) ask for what is needed.
  To add a simulator, follow [add an environment backend](docs/developer/add-backend.md).
- Report security problems privately as described in [SECURITY.md](SECURITY.md),
  never in a public issue.

## Set up

You need Python 3.11 with [uv](https://docs.astral.sh/uv/), Node.js 22.19 or
later with npm, and Git. Install FFmpeg (`ffmpeg` and `ffprobe` on `PATH`; on
Ubuntu, `sudo apt-get install ffmpeg`) before running the full suite: the
video export tests use it. The Synthetic quickstart does not need it.

```bash
git clone https://github.com/fast-rua/ORA.git
cd ORA
uv sync --all-extras --locked                                # Python environment from uv.lock
npm ci --prefix ora-pi --no-audit --no-fund                  # the Pi agent core (TypeScript)
npm ci --prefix ora-pi/web --no-audit --no-fund              # ORA's web page, for web changes
uv run --locked ora doctor                                   # setup discovery; starts nothing
```

No test needs a model API key, a simulator or a GPU. Simulators are installed
separately, each in its own interpreter; see [Environments](docs/environments/README.md)
when your change needs one.

## Where code goes

The [repository structure](docs/developer/repository-structure.md) states the
layout and its rules, and the [code map](docs/code-map.md) links every
package's README. A README says what belongs in its package and what does not,
what the package may import and how it is entered: read it before you add a
module.

`tests/repository/test_structure.py` checks these rules on every change,
statically (it never imports `ora`), and prints each violation as
`file:line -> target`:

- **Layers import downward only:** `ora/app` (the `ora` command) >
  `ora/orchestrator` (plans, launch, the Supervisor, evaluation, sealing) >
  `ora/agent` (the policy side) and `ora/environment` (the simulator side) >
  `ora/contracts` (the action envelope both sides share). `agent` and
  `environment` never import each other, and `ora` never imports `viewer`.
- **A simulator lives in two places:** `ora/environment/backends/<sim>/`,
  which only the isolated worker imports under the simulator's own
  interpreter, and `ora/orchestrator/backends/<kind>/`, the host side, which
  the orchestrator reaches only through `backends/registry.py`.
- **One way inside the agent:** its units import only lower units; a tool is
  one directory under `ora/agent/toolkit/tools/`, and the trusted host it runs
  under, with the tool runtime frame, is `ora/agent/toolkit/host/`.
- **Lazy packages:** a package with sub-packages imports nothing eagerly in its
  `__init__.py`.
- **Tests mirror the code** (below) and never import one another.
- **Every package has a README** in the same five sections; update it when you
  move a module, add an entry point or change what the package may import.

Two more rules hold for every change, though no test can catch every case.
A moved or removed module leaves no compatibility shim, re-export stub or
`python -m` alias behind: rewrite its callers in the same pull request. And the
files [that may not move](docs/developer/repository-structure.md#12-files-that-may-not-move)
keep their path and bytes, comments included, because simulator qualification
records hash them. When you move files, make the move its own commit
(`git mv` only) and the reference rewrites the next one, so `git log --follow`
traces every file.

### Tests mirror the code

A test of `ora/<package path>/<module>.py` goes in
`tests/ora/<package path>/test_<module or behavior>.py`; `tests/viewer/` covers
`viewer/`. A helper several test modules share lives in `tests/support/` (the
only test package another test may import), a fixture they share in the
nearest `conftest.py` above them, and fixed samples in `tests/data/`.
`tests/agent_core/` holds the TypeScript agent core's parity and sealed-replay
tests, marked `agent_core`, and `tests/repository/` the repository-wide guards.
A test drives the code against the fakes in `tests/support/`, never a network,
a model API or a simulator. [Testing](docs/developer/testing.md) covers
choosing a check, splitting a test family and the timing-sensitive tests.

### Add an environment, a tool or a command

| To add | Where it goes | Read |
|---|---|---|
| An environment backend | The worker half in `ora/environment/backends/<sim>/` (with its identities in `ora/environment/benchmarks/` and control constants in `ora/environment/controllers/`), the host half in `ora/orchestrator/backends/<kind>/` with one entry in `backends/registry.py`, a README for each half and an environment page | [Add an environment backend](docs/developer/add-backend.md), [environment compatibility](docs/developer/authoring/environment-compatibility.md) |
| A tool | The tool's directory under `ora/agent/toolkit/tools/<name>/` (its `tool.py`, description and README), what its data interface needs from the trusted side in `ora/agent/toolkit/host/`, and its manifest and schemas under `ora/policy/tool_surfaces/` (then `./scripts/ora lock`); the toolkit dispatch (`host/runtime/dispatch.py`) serves it on every surface that advertises it | [the toolkit](ora/agent/toolkit/README.md), [the host](ora/agent/toolkit/host/README.md), [tool surfaces](docs/developer/authoring/tool-surface.md) |
| An `ora` command | A module in `ora/app/` with `register(subparsers)` and `run(args)`, its name in `COMMANDS` and `GROUPS` in `ora/app/main.py`, a row in the [ora/app](ora/app/README.md) command table, then `./scripts/ora dev cli-reference --write` and the [command guide](docs/usage/commands.md). An environment's own `spec` or `qualify` tool is declared by its backend (`spec_tools`, `qualify_tools`), not added to `ora/app` | [ora/app](ora/app/README.md), [command reference](docs/usage/cli-reference.md) |

## Run the checks

Run the checks your change touches while you work, and all of them that apply
before you open the PR. [tests/README.md](tests/README.md) maps each test
directory to the code it covers, and [Testing](docs/developer/testing.md) says
which check a change needs.

| What changed | Check |
|---|---|
| Python code | `scripts/dev/check --quick`; one package: `uv run --locked python -m pytest -q tests/ora/<package path>` |
| A module moved, a package added, an import across packages | `uv run --locked python -m pytest -q tests/repository/test_structure.py` |
| The agent core (`ora-pi/`) or anything it calls in Python | `uv run --locked node --test ora-pi/test/*.test.ts ora-pi/test/*.test.mjs` and `uv run --locked python -m pytest -q -rs -m agent_core tests` |
| ORA's web page (`ora-pi/web/`) | `npm run typecheck --prefix ora-pi/web`, `npm test --prefix ora-pi/web`, `npm run build --prefix ora-pi/web` |
| Documentation, or a page published from a README | `./scripts/docs build`; `./scripts/docs serve` for a live preview |
| Prompts, tool surfaces, embodiment cards, observation projection or composition | `scripts/dev/compare-model-visible <base> HEAD` (see below) |
| Files under `ora/policy/` | `./scripts/ora lock`, then commit `policy.lock`; `./scripts/ora lock --check` must pass |
| Dependencies | `uv lock --check`; `npm ci` refuses a `package-lock.json` that disagrees with its `package.json` |

`scripts/dev/check` runs most of these at once, in parallel lanes, and prints
a summary: the Python suite without the agent core, the agent-core tests, the
TypeScript and web checks, the policy lock and whitespace, then the
documentation build (`compare-model-visible` needs a base commit and runs on
its own). `--quick` runs only the Python suite and the policy-lock and
whitespace checks, `--workers N` sets the pytest workers and `--logs DIR` keeps
each step's log outside the checkout; [scripts/README.md](scripts/README.md)
has the rest.
`uvx pre-commit run --all-files` runs the policy-lock and whitespace hooks.
CI runs the same commands on every pull request: the Python suite, the
agent-core and web checks, the policy lock and the documentation build.

**Lockfiles.** `uv.lock`, `ora-pi/package-lock.json`,
`ora-pi/web/package-lock.json` and `docs-site/package-lock.json` are
canonical. Change one only in a PR whose purpose is that dependency change,
and say why; never regenerate a lock as a side effect of another change.

**What the model sees.** Every run records the identity of what the model was
shown. If your change touches prompts, tools, observation projection or
composition, run `scripts/dev/compare-model-visible origin/main HEAD` and
explain every difference it reports in the PR. A change that should not affect
the model, such as a refactor, must report no difference.

**Simulator changes.** The suite checks contracts against fake simulators; it
does not check physics. When your change affects a simulator backend, a
controller or its assets, also run the smallest meaningful episode or
qualification on the real simulator, and say in the PR what you ran (simulator
and asset versions, tasks, seeds) and what you did not.

## Open a pull request

Keep a pull request to one change, in the style of the code around it, with
the smallest tests that exercise the new behavior. Before you request review,
check that:

- the checks for what you changed pass locally (see the table above);
- new behavior has a test, and every new test module states the contract it
  protects in its module docstring;
- a change to a public API, CLI command or option, default, prompt, tool,
  context rule, run lifecycle or evidence format updates its page under
  [docs/reference/](docs/reference/README.md) (or the [command guide](docs/usage/commands.md), and
  `./scripts/ora dev cli-reference --write` for the generated
  [command reference](docs/usage/cli-reference.md)) in the same PR;
- `policy.lock` is current when `ora/policy/` changed, and every
  model-visible difference is explained;
- no lockfile changed unless the PR is about that dependency;
- credentials, run outputs, simulator assets, model weights and large
  media stay out of Git (`outputs/` is ignored for runs);
- third-party code keeps its license notice: put the license text under
  `third_party/` and name the source in `NOTICE`;
- the description says what changed, why, how you tested it, and any
  breaking change.

The [pull-request template](.github/pull_request_template.md) asks for the same:
what changed and why, how you tested it (the commands and their result lines),
and a checklist: the structure test passes, what the model sees is unchanged or
each difference is explained, evidence formats are unchanged or the change is
described, the docs and package READMEs that describe the changed code are
updated, and new tests sit in the mirrored tree.

## Versioning

ORA follows [semantic versioning](https://semver.org/). The
version is declared in `pyproject.toml`, `ora-pi/package.json`,
`ora-pi/web/package.json` (and their lockfiles) and `CITATION.cff`;
`tests/ora/app/test_version.py` checks that they agree, and `ora --version`
prints it. Maintainers set it when a release is cut; a pull request does not
change it.

ORA is at 0.x: until 1.0.0 a minor release (0.2.0) may change a public
contract, and its release notes say how to migrate; a patch release (0.1.1)
is a backward-compatible fix or addition. From 1.0.0 on:

- **Major** (2.0.0): a breaking change to a public contract: a CLI command or
  option, the RunSpec schema, the evidence format, a documented Python API
  under `docs/reference/`, or the action contract between agent and
  environment. The release notes say how to migrate.
- **Minor** (1.1.0): a backward-compatible addition, such as a new command,
  option, environment, tool or default.
- **Patch** (1.0.1): a backward-compatible fix.

A change to what the model sees (prompts, tools, observation projection or the
default composition) changes the identity of new runs, and their results are
not directly comparable with earlier ones. Release notes name such changes
even when they are not breaking.
