# ORA: contributor and agent guide

This repository is a thin harness that runs a language-model agent as the
policy of a simulated robot. The agent sees only public observations and tools;
an isolated simulator backend executes its actions; only the benchmark's own
official boundary decides success. Every run is planned, executed, evaluated
and sealed with hash-bound evidence.

## Layout

Every package carries a README (responsibility, what belongs there and what
does not, dependency direction, entry points, documentation);
`docs/code-map.md` links them all and `docs/developer/repository-structure.md`
states the import rules and the files that may not move.

- `ora/app/`: the `ora` command: one parser, one command table
  (`ora/app/main.py`), the help topics, the terminal and web front ends. Every
  operator and authoring tool is an `ora` command; the Viewer is `ora-viewer`.
- `ora/orchestrator/`: RunSpec/RunPlan resolution (`spec/`), preflight, host
  profiles, leases and launch, the Supervisor's episode loop (`episode/`),
  evaluation and sealing, reports and the suite queue;
  `ora/orchestrator/backends/<kind>/` holds the host side of each simulator
  behind `backends/registry.py`.
- `ora/agent/`: the `native_api` policy (direct OpenAI-compatible or
  Anthropic model calls; its agent loop runs in `ora-pi/`), tools (one family
  per `tools/<family>/`), composition, context rules and evidence formats;
  `agent/pi/` is the Python half of the default Pi agent core,
  `agent/session_core/` the Python half every `native_api` session shares
  (tool gateway, static prefix, observation and tool-result projections) and
  `agent/tui/` the `ora watch` transcript; `agent/model_services/` holds the
  resident perception services.
- `ora-pi/`: the default agent core in TypeScript on the pinned Pi packages
  (provider loop, request assembly, context management, agent-side evidence);
  `ora-pi/web/` is ORA's web page for a live run (`ora web`) and its server.
- `ora/contracts/`: the versioned action envelope shared by agent
  and environment.
- `ora/robot_client/`: the standard-library robot client (`ora-robot`) a
  free-play agent's own scripts call through the Supervisor.
- `ora/environment/`: transport-neutral sessions, Synthetic, host-safe
  benchmark and controller contracts, and the isolated RoboTwin,
  LIBERO/LIBERO-Pro, RoboDojo, RoboCasa365 and RoboLab backends
  (`environment/backends/<sim>/`, loaded only by the worker process under each
  simulator's own interpreter).
- `ora/policy/`: prompt bundles, ToolSurfaces, embodiment and
  environment assets, the default baseline catalog. `policy.lock` pins them.
- `examples/`: the quickstart smoke, the official-suite launcher and the host
  profile template.
- `viewer/`: the read-only local run viewer.
- `tests/`: the deterministic CPU-only suite. Its tree mirrors the code:
  `tests/ora/<package path>/` tests `ora/<package path>/` and `tests/viewer/`
  the viewer; shared fixtures live in `tests/support/` (the only test package
  another test may import) and fixed samples in `tests/data/`;
  `tests/agent_core/` holds the TypeScript-core parity and sealed-replay tests,
  marked `agent_core`; `tests/repository/` the repository-wide guards.
  `tests/README.md` maps each directory to the code it covers.
- `scripts/`: the wrappers `ora`, `ora-remote` and `setup-robodojo`, and the
  docs build; `scripts/dev/` holds the local checks and maintenance tools.
  `scripts/README.md` lists them.

## Setup and checks

```bash
uv sync --all-extras --locked
npm ci --prefix ora-pi --prefer-offline --no-audit --no-fund   # Node.js >= 22.19
uv run --locked ora doctor
scripts/dev/check --quick            # the Python suite and the repository checks
scripts/dev/check                    # everything CI runs, in parallel lanes
uv run --locked python -m pytest -q tests/ora/agent/toolkit/host/runtime   # one package
uv run --locked python -m pytest -q -m agent_core tests            # the agent-core tests
npm ci --prefix ora-pi/web && npm test --prefix ora-pi/web   # ORA's web page (`ora web setup` builds it)
./scripts/ora lock --check           # policy.lock is current
./scripts/ora baselines --explain    # how the default composes
uv run --locked ora-viewer --open-browser
```

`scripts/dev/check` runs its lanes in parallel and prints one summary
(`--workers N`, `--logs DIR`); `scripts/README.md` describes its options.

Python 3.11 and `uv.lock` are canonical; do not regenerate the lock as a side
effect of another change. Simulators run in their own interpreters, named by
environment variables (`ROBOTWIN_PYTHON`, `LIBERO_PRO_PYTHON`,
`ROBODOJO_PYTHON`, `ROBOCASA_PYTHON`, `ROBOLAB_PYTHON`). A live run needs a
registered execution host: copy
`examples/hosts/example-host.yaml` to `ora/orchestrator/hosts/<id>.yaml`
(or a directory named by `ORA_HOSTS_DIR`) and export
`ORA_EXECUTION_HOST=<id>`. Generated runs and credentials stay
outside Git under the ignored `outputs/` workspace.

## Invariants

- The policy never receives privileged information: no object ground truth,
  segmentation, contacts, task source, expert code or success predicate.
  `observation.dump_state` is read-only; `robot.submit_action` (or a surface's
  renamed proposal tool) is the only state-changing tool and always passes the
  Supervisor.
- Success comes only from the selected benchmark's official boundary
  (RoboTwin `check_success`, LIBERO termination, RoboDojo's task success,
  RoboCasa365 `_check_success`, RoboLab's success termination).
  Synthetic runs report `official_success: null`.
- What the model sees is identity-checked. Any change to prompts, tools,
  observation projection or composition must be deliberate: run
  `scripts/dev/compare-model-visible <base> HEAD` and explain every difference.
- Policy assets are referenced by path and pinned by `policy.lock`. After
  adding or changing a file under `ora/policy/`, run
  `./scripts/ora lock` and commit the lock in the same change.
- Evidence formats and schemas change only on purpose, with the
  `docs/reference/` page that defines them; runs sealed before the change still
  evaluate and replay.
- A new feature becomes the default for the maintained consumers it applies
  to (the baseline catalog and default resolvers), with its tests and the
  matching `docs/reference/` page updated in the same change.
- Keep changes minimal and in the surrounding style; prefer the smallest test
  that exercises the changed behavior.

## Structure rules

`tests/repository/test_structure.py` checks the first four statically on every
change; `docs/developer/repository-structure.md` states them in full (rules
L1-L9 in section 2, the test tree in section 8, the READMEs in section 9).

- Layers import downward only: `ora/app` > `ora/orchestrator` >
  {`ora/agent`, `ora/environment`} > `ora/contracts`. `agent` and
  `environment` never import each other, and `ora` never imports `viewer`. The
  orchestrator core reaches a simulator only through
  `ora/orchestrator/backends/registry.py`, and only the worker entry points
  import `ora/environment/backends/`.
- Inside `ora/agent` the units and the tool families import one way (L6, L7),
  and a package with sub-packages keeps its `__init__.py` lazy (L9).
- Tests mirror the code (`tests/ora/<package path>/`) and import only `ora`,
  `viewer`, `examples` and `tests.support`, never another test module.
- Every package the structure lists has a README. Update it in the same change
  when you move a module, add an entry point or change what the package may
  import.
- A moved or removed module leaves no compatibility shim, re-export stub or
  `python -m` alias behind: rewrite its callers in the same change. Every
  operator and authoring tool is an `ora` subcommand (`ora/app/`), not a new
  script or `python -m` entry point.
- Moving files is its own commit (`git mv` only), apart from the rewrites that
  follow, so `git log --follow` traces every file.
- Pinned files keep their path and bytes, comments included: the RoboDojo
  worker and play kernels, the `ora/orchestrator/robodojo_*.py` modules,
  `ora/orchestrator/episode/action_executor.py`, the RoboTwin seed-scan inputs
  (`ora/environment/backends/robotwin/{qualification,session,common}.py`,
  `ora/environment/benchmarks/robotwin.py`), `uv.lock` and `pyproject.toml`.
  Qualification records hash them, so a change to one is a decision that comes
  with the re-qualification section 12 of the structure page names for it.

## Documentation

`CONTRIBUTING.md` is the path from a clone to a pull request and
`docs/developer/onboarding.md` the first hour for a new collaborator.
`docs/reference/README.md` describes the current APIs, defaults, authority
boundaries and evidence model; `docs/architecture.md` and `docs/code-map.md`
describe structure; `docs/developer/authoring/` holds the authoring guides
(tool surfaces, prompt bundles, task contracts, environment compatibility,
qualification, trajectory capture, frontend style) and
`docs/developer/add-backend.md` the steps for a new environment.

The documentation site is built with Blume from `docs/` (`docs-site/`). Run
`./scripts/docs serve` for a local preview and `./scripts/docs build` for strict
validation. List public pages in `docs-site/navigation.json`. See
`docs/developer/documentation.md` and `docs/developer/docs-style.md`.
Documentation dependencies are locked in `docs-site/`; do not change the
runtime's `uv.lock` for them.
