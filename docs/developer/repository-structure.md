# Repository structure

This page describes how the repository is laid out, the import rules that
keep it that way, the files that may not move, and the decisions behind the
layout. It records the tree as built by the October 2026 restructure; file
counts are tracked files, and for Python directories the number of `.py`
modules. `tests/repository/test_structure.py` checks the import rules, the
test tree and the package READMEs on every change, statically, without
importing `ora`.

## 0. Finding your way in

```text
README.md                what ORA is, and one `ora run` episode
docs/quickstart.md       install, one Synthetic episode, one simulator episode
docs/how-ora-works.md    where the policy, the tools, the environment and the evidence sit in one run
docs/architecture.md     the four layers and the two places a backend lives
docs/code-map.md         one row per package, with a link to its README
<package>/README.md      responsibility; what belongs here and what does not;
                         dependency direction; entry points; documentation
```

After these six steps a reader can answer three questions. **What is shared by
every environment:** `ora/contracts`, `ora/agent`, and the orchestrator and
environment cores. **What belongs to one simulator:** only
`ora/environment/backends/<sim>/` (the worker half, in the simulator's own
interpreter) and `ora/orchestrator/backends/<kind>/` (the host half). **Where a
tool's code is:** `ora/agent/toolkit/tools/<name>/`, one directory per tool.

A new collaborator's first hour (install, the checks, one episode with and
without a simulator, the three rules no change may break) is
[Start contributing](onboarding.md); [Contributing](../../CONTRIBUTING.md) is
the path to a pull request, which `.github/pull_request_template.md` turns into
a checklist.

## 1. Top level

```text
README.md  AGENTS.md  CLAUDE.md (imports AGENTS.md)  CONTRIBUTING.md  LICENSE  NOTICE  SECURITY.md  CITATION.cff
pyproject.toml  uv.lock  policy.lock  .python-version
ora/          1172  the Python package (707 modules; 414 files of policy data)
ora-pi/        166  the TypeScript agent core (src/ 88) and ORA's web page (web/ 48)
viewer/         42  the read-only run viewer and run dashboard, its own package (ora-viewer)
tests/         651  the CPU-only suite (section 8)
docs/           77  the documentation pages; docs-site/ builds them
docs-site/      12  the documentation site's build (Blume)
examples/        9  quickstart smoke, official-suite cells, host template
scripts/         9  ora, ora-remote, docs and setup-robodojo wrappers; dev/ checks
third_party/     3  license texts
.github/         8  CI workflows, issue templates, the pull-request template
```

## 2. Layers and import rules

```text
ora/app            the `ora` command: parse, ask, print, draw
   |
ora/orchestrator   RunSpec to RunPlan, preflight, leases, launch, the Supervisor's
   |               episode loop, evaluation, sealing, reports;
   |               backends/<kind>/ behind a string registry
   +---------------------------+
   |                           |
ora/agent                   ora/environment
the policy side             sessions, worker, Synthetic;
                            backends/<sim>/ only in the isolated interpreter
   |                           |
   +---------------------------+
   |
ora/contracts      the action envelope and the contracts both sides share

beside the layers: ora/policy (data, pinned by policy.lock), ora/robot_client
(standard library only), ora/version.py, ora/cell (the real-cell kit)
```

| Rule | Statement | Status |
|---|---|---|
| L1 | Layers import downward only: `app` > `orchestrator` > {`agent`, `environment`} > `contracts`; `agent` and `environment` import nothing of each other; `policy` and `version` import no `ora` package; `robot_client` imports none and only `orchestrator` and `app` may import it; `cell` imports none (it reaches a robot's `cell_adapter` by a module-name string) and only `environment`, `orchestrator` and `app` may import it | Holds |
| L2 | `ora` never imports `viewer` (the live world view moved into `ora/orchestrator/world_view.py`; `viewer` imports `ora`) | Holds |
| L3 | The orchestrator core reaches a backend only through `ora/orchestrator/backends/registry.py`, whose `BACKENDS` maps `runtime.kind` to `"ora.orchestrator.backends.<kind>.backend:BACKEND"`; the seven pinned `ora/orchestrator/robodojo_*.py` modules count as part of the RoboDojo backend | Holds |
| L4 | The environment core imports `ora/environment/backends/` only from the isolated-interpreter entry points `ora/environment/worker.py` and `replay_worker.py`; nothing outside `ora/environment/planners/` imports it (a planner worker is started by module name) and a planner imports no backend | Holds |
| L5 | `ora/agent` never imports `ora/environment/backends`, `ora/orchestrator` or `ora/app` | Holds |
| L6 | Inside `ora/agent`, a unit imports only lower units: `pi`, `tui` > `session_core`, `scripted` > `context` > {`memory`, `code_mode`} > `toolkit/host` > {`composition`, `toolkit`} > `providers`, `model_services` > `evidence`, `sessions` > `contracts`, `error_classes`; the toolkit's host, which holds the tool runtime frame (`host/runtime/`), counts as a unit of its own, so nothing else in `toolkit` imports it | Holds |
| L7 | Retired with the `ora/agent/tools` package (P21): it ordered that package's family sub-packages, and the tool runtime frame they held now sits in `ora/agent/toolkit/host/runtime/` | Retired |
| L8 | Tests import only `ora`, `viewer`, `examples` and `tests.support`; no test module imports another | Holds |
| L9 | A package with sub-packages imports nothing eagerly in its `__init__.py` (lazy facades); allowed exceptions: `ora/agent/__init__.py` re-exports its contracts, and `ora/environment/backends/robodojo/__init__.py` imports `session.py` because it is a pinned kernel file | Holds |

## 3. `ora/environment/` (173 modules, 50.6k lines)

```text
ora/environment/                16  base, contracts, session, definitions, factory, rpc, worker,
                                    replay_worker, synthetic, synthetic_capture, execution_capture,
                                    controller_replay, graphics_probe, workspace_profiles,
                                    robodojo_play_factory (+ lazy __init__)
├── benchmarks/                 18  host-safe identities, rosters, pins, play-scene registries (+ 4 locked JSON)
├── controllers/                26  host-safe motion-transaction, safety and per-embodiment contracts,
│                                   the planner-neutral motion planning contract and the public-depth world
├── planners/                    4  motion planner workers, each under its own interpreter: curobo_v2/, timing/
└── backends/                       imported only inside the isolated worker
    ├── libero_pro/             28  (+ installation/requirements)
    ├── robotwin/               30  play_scenes/ 4, vendored/ 2 (copied into RoboTwin, not imported)
    ├── robodojo/               30  play_scenes/ 6 (+ scenes/<set>/*.yml), installation/ 3
    ├── robocasa/                7
    └── robolab/                10  no __init__.py (a namespace package)
```

The layer kept its shape: the RoboDojo worker kernel and the RoboTwin seed-scan
inputs live here, and moving them would void existing qualification records
(section 12). The changes were the lazy `__init__.py` facade and a README per
directory. `robodojo_play_factory.py` kept its name (decision 2).

## 4. `ora/agent/` (287 modules, 53.3k lines)

```text
ora/agent/                       4  contracts, error_classes (ToolError), scripted (+ __init__ facade)
├── toolkit/host/               23  the trusted host the toolkit's tools run under: the adapters, I/O recording, observe's dual-arm image labels (+ README):
│   ├── runtime/                16  the tool runtime frame: ToolRuntime protocol, factory, RobotToolRuntime base, ManipulationToolRuntime and the host state its mixins keep, the toolkit dispatch, publisher and input freezer (lazy __init__)
│   ├── compute/                 5  the restricted geometry evaluator and its grammar
│   ├── finish/                  5  finish_episode, check_task
│   ├── grasp/                  12  grasp proposers, per-embodiment compilers and their identities, lift geometry, the Panda top-down fallback family
│   ├── perception/             12  the segmented-target store, mask localization, grounding previews, the segmentation and grounding backend protocols, the fallback segmenter
│   ├── reach/                   4  align_plan's candidates, jaw-centre offsets, probe feedback
│   ├── motion_rules/            6  move_pose's argument reading, refusals, observed clearance
│   ├── skills/                  8  grasp_execute's stage loop
│   └── shared/                  7  saved steps, target / artifact / service refusals, artifact I/O, RGB-D fusion
├── composition/                 5  baseline, ora_local, locked_assets, inspection (+ lazy facade)
│   ├── surface/                16  ToolSurface resolution, tool names, units, view config, motion modes, compute grammar, robot_runtime definitions
│   ├── schemas/                12  constants tool text shares with its runtime, the finish contract
│   ├── embodiment/              8  descriptors, jaw models, control and geometry cards, environment context, moved sources
│   ├── prompt/                  5  prompt bundles, policy layers, context packs, slots
│   └── task_contract/           5  public completion contracts (generic and RoboDojo)
├── context/                     2  causal_value (+ lazy facade)
│   ├── observation/            10  one observation's projection (projector, contact points, public facts)
│   ├── delivery/                7  capability-trimmed delivery (trims, dedup, round static facts)
│   └── lossless/                7  Context v4-v6 contracts and the rollout compactor
├── evidence/                   18  canonical, agent_events, trace_verify, trace_privacy, sft_export, wire_export, image binding, request archive, play trace
├── memory/                      9  markers, read_tool, catalog, tool_runtime
├── code_mode/                   9  code mode: contract, coding tools, runtime, sandbox, shell, workspace,
│                                   runner and robot library (run inside the sandbox) (+ README)
├── providers/                  22  openai_compatible, anthropic_*, fake, factory.build_provider, profiles, credentials, recovery
├── model_services/              9  SAM3 / Contact-GraspNet resident services and their clients
├── session_core/               16  the Python half of a native_api session
├── sessions/                    3  PolicySession facade, scripted adapter
├── pi/                         13  the Pi core's process, RPC, tool RPC, robot socket, code mode
└── tui/                         6  the ora watch transcript
```

`tests/data/published-limits.json` names its owners as
`ora.agent...:NAME`, and six embodiment JSON files name module paths; both
were rewritten with the moves, and the embodiment files re-pinned in
`policy.lock`.

## 5. `ora/orchestrator/` (187 modules, 38.7k lines)

```text
ora/orchestrator/               16  paths, leases, launch, supervise, faults, import_closure, progress,
                                    world_view (+ world_view_assets/), identity-registry.json, lazy facade,
                                    and the seven pinned robodojo_*.py (section 12)
├── spec/                       26  RunSpec to RunPlan: run_spec and its section resolvers, values, benchmark
│                                   builders, defaults, policy pins and lock, experiment identity, baselines
├── preflight/                  11  launch gates, checks, resources and host storage, the provider probe
├── episode/                    32  coordinator, loop, actions, action_executor, stage_outcome, rounds, play,
│                                   trusted stops, operator control, script actions, agent core selection
├── evaluation/                 24  evaluate / seal / export and their checks, metrics, context audit and
│                                   manifests, ownership audit, model-visible identity, revalidation
├── reports/                     6  run_report and its price table
├── suite/                       3  official-suite cells and queue
├── play/                        3  play scenes and their plan binding
├── hosts/                          local-cpu and single-gpu host profiles
└── backends/                    3  registry, synthetic (+ README)
    ├── libero_pro/              8
    ├── robotwin/               13  seed scans, seed manifests and admission, play audit
    ├── robodojo/               23  motion/ 5
    ├── robocasa/                6
    └── robolab/                11  the RoboLab-120 roster and contract generator, the contract witness
```

An `OrchestratorBackend` has 24 fields besides its `kind`: only
`resolve_runtime` and `preflight_runtime` are required, `capabilities` is
data, and every other hook is optional with a core default. Each backend package's `__init__.py` holds a docstring only, and
`registry.backends()` is the one call that imports all of them. The core
sub-packages are not layered among themselves yet: `episode` and `evaluation`
read each other's vocabularies, and `spec` and `preflight` read
`episode/agent_core.py`.

## 6. `ora/app/` and the `ora` command

```text
ora/app/                        38  main (one parser, COMMANDS, GROUPS), one module per command group:
                                    setup, doctor, run, play, replay, web, interactive, plan, runs,
                                    evidence, report, spec, qualify, suite, providers, services,
                                    catalog, ladder, dev, cli_reference; helpers: passthrough,
                                    quickstart, local, local_setup, gpu_setup, gpu_recipes, env_file,
                                    run_services, task_catalog, backend, world, web_gateway,
                                    replay_export, remote, remote_session
└── help/                        6  topic pages: runs, evidence, play, remote, safety, rl-interface
```

`ora --help` groups the commands:

```text
Run an episode                       setup doctor run play replay web interactive attach sessions local
Formal runs and evidence             plan launch retry status stop watch workspace evaluate seal
                                     revalidate export export-sft export-traces report metadata state metrics
                                     ownership trace-validate
Environment qualification and suites spec qualify suite "play audit"
Configuration                        providers services baselines lock
Developer                            ladder dev help
```

Help has three levels: `ora --help` (the groups, in English or Chinese),
`ora <command> --help` (argparse), and `ora help <topic>` (the pages in
`ora/app/help/`, package data). A topic wins over a command of the same name:
`ora help play` prints the page, `ora play --help` the options.
`docs/usage/cli-reference.md` is generated from the parser by `ora dev
cli-reference --write`, and `tests/ora/app/test_cli_reference.py` fails on a
stale page; `docs/usage/commands.md` is the hand-written guide.

Operator tools a backend owns are reached by kind, not imported by the app:
`ora spec ENV TOOL`, `ora qualify ENV TOOL` and `ora play audit --env ENV` run
the tools the backend declares (`spec_tools`, `qualify_tools`, `play_audit`).
`ora spec libero` covers LIBERO-Pro; the RoboTwin seed-scan tool is `ora
qualify robotwin qualify`. The Viewer's offline builds are `ora-viewer compile
previews|videos`.

Twenty-five modules keep a `python -m` entry point or run by path because a
process starts them (the simulator worker, the supervisor, probes, audits,
installers, the robot client); [ora/app/README.md](../../ora/app/README.md)
lists them with who starts each, and `tests/ora/app/test_main.py` checks the
list against the `__main__` guards. Folded tools left no compatibility shim at
their old `python -m` path.

`scripts/` keeps `ora`, `ora-remote`, `docs`, `setup-robodojo` (the pinned
RoboDojo installer's receipt names it), `lib/python.sh` and `dev/{check,
compare-model-visible, refresh-vendored-pins}`; the other wrappers became `ora`
or `ora-viewer` commands.

## 7. `ora-pi/`

```text
ora-pi/
├── bin/                         2  ora-rpc-attach, attach.sh
├── src/                        88  main.ts, rpc.ts, jcs.ts, pyjson.ts
│   ├── core/  context/ (+ memory/)  provider/  evidence/  play/  rpc/  tui/  watch/
├── test/                           node --test unit tests
└── web/                        48  ORA's web page: React client (src/), Node server (server/), tests
```

The directory layout did not change. `src/watch/index.mjs` now starts `python
-m ora.app.backend`.

## 8. `tests/`

The test tree mirrors the code tree: `tests/ora/<package path>/test_*.py`
covers `ora/<package path>/`, and `tests/viewer/` covers `viewer/`. Only
`tests/support/` may be imported across directories (rule L8); fixed samples
live in `tests/data/`. `tests/agent_core/` holds the TypeScript-core parity
and sealed-replay tests, selected by the `agent_core` marker, and
`tests/repository/` the repository-wide guards (canonical imports, module
size ratchet, packaging, policy lock, this structure). Lanes are selected by
marker, not by path. [`tests/README.md`](../../tests/README.md) maps each
directory to the code it covers, and [Testing](testing.md) says which check
to run for a change.

```text
tests/                           2  conftest (+ __init__); README.md
├── ora/                         1  the mirror of ora/
│   ├── contracts/               3
│   ├── agent/                   4  the ToolError envelope (error_classes)
│   │   ├── composition/         3  surface/ 19, schemas/ 3, embodiment/ 7, prompt/ 4, task_contract/ 4
│   │   ├── context/             2  observation/ 6, delivery/ 7, lossless/ 7
│   │   ├── toolkit/host/       13  runtime/ 21, grasp/ 5, perception/ 6, compute/ 2, finish/ 3, skills/ 5, shared/ 1, reach/ 7
│   │   ├── evidence/            7
│   │   ├── memory/              2
│   │   ├── code_mode/           4
│   │   ├── providers/          11
│   │   ├── model_services/      3
│   │   └── session_core/       11
│   ├── environment/             5  worker RPC, execution capture, local platform, wait
│   │   ├── controllers/         7
│   │   ├── planners/            0  curobo_v2/ 1 (the worker over a stand-in engine), timing/ (the real worker, skipped without its interpreter)
│   │   └── backends/            1  libero_pro/ 14, robotwin/ 20, robodojo/ 30, robocasa/ 2, robolab/ 9
│   ├── orchestrator/           10  leases, launch, supervise, paths and storage, world view, control plane
│   │   ├── spec/               11
│   │   ├── preflight/           4
│   │   ├── episode/            15
│   │   ├── evaluation/          8
│   │   ├── reports/             2
│   │   ├── suite/               3
│   │   ├── play/                2
│   │   └── backends/            3  libero_pro/ 2, robotwin/ 5, robodojo/ 18, robocasa/ 2, robolab/ 3
│   ├── app/                    24  one module per command module, test_main, test_help, the CLI reference
│   ├── robot_client/            2
│   └── cell/                    7  config, calibration, lock, occupancy, geometry; operator/ 3, depth/ 2
├── viewer/                      7  dashboard/ 3
├── agent_core/                 21  the Pi core: parity, faults, crash resume, stops, sealed replays
├── repository/                  9  canonical imports, module-size ratchet, packaging, policy lock,
│                                   path-only references, undefined globals, dead state, test_structure
├── support/                     1  agent/ 34, agent_core/ 11 (+ provider/ 10, replay/ 4), cell/ 2,
│                                   context_pack/ 2, libero/ 3, orchestrator/ 13, robodojo/ 17, robotwin/ 7
└── data/                      155  files: agent_core/ 24, context_differential/ 105, context_pack/ 5,
                                    delivery_trim/ 6, hosts/ 3, prompts/ 5, robodojo/ 4, 3 JSON
```

Counts are `.py` modules per directory (test modules, `__init__.py` and
`conftest.py`); `tests/data/` counts files. The suite has 328 test modules:
295 under `tests/ora/`, 19 in `tests/agent_core/`, 8 in `tests/repository/`
and 6 in `tests/viewer/`.

## 9. `docs/` and the package READMEs

```text
docs/                         README (index), quickstart, how-ora-works, architecture, code-map,
                              context-assembly, glossary, support
├── getting-started/      11  installation, local, single-gpu, model-apis, <env> x 5,
│                             official-suite, troubleshooting
├── usage/                10  runs, play, real-cells, interactive, remote, safety, rl-interface,
│                             compute-usage, commands (hand-written), cli-reference (generated)
├── environments/          7  README, common, <env> x 5
├── evaluation/            1  protocol
├── reference/            12  public interfaces, configuration and evidence, runtime lifecycle,
│                             policy I/O, context packs, experiment identity, RoboTwin seed and
│                             scene qualification, RoboTwin contact recovery, RoboDojo motion
│                             qualification, RoboCasa365 and RoboLab, local CPU compatibility
├── api/                   5  actions, environment, plans, policy
├── robots/                1  frames and units (the robot and task pages are generated)
├── developer/             6  onboarding, testing, documentation, docs-style, add-backend,
│   │                         repository-structure
│   └── authoring/         7  tool-surface, prompt-bundle, qualification, task-contract,
│                             environment-compatibility, trajectory-capture, frontend-style
├── templates/             2
├── design/                3  design records, not current contracts; not published
└── assets/                5
```

`docs/guides/` was split by reader: user-facing pages went to
`getting-started/` and `usage/`, author-facing ones to `developer/authoring/`,
and the test-suite navigation page folds into the developer testing page.
`docs-site/navigation.json` lists every page; `./scripts/docs build` fails on
a page missing from it, a broken link or anchor, and syntax the conversion
does not handle, and `scripts/dev/check` runs it.

Every package below carries a README in the same five sections
(responsibility; what belongs here and what does not; dependency direction;
entry points; documentation), and `test_every_package_has_readme` fails when
one is missing: `ora/`, `ora/contracts`, `ora/environment`,
`ora/environment/backends/<sim>` x 5, `ora/environment/benchmarks`,
`ora/environment/controllers`, `ora/environment/planners`, `ora/agent`,
`ora/agent/composition`, `ora/agent/context`, `ora/agent/code_mode`, `ora/orchestrator`,
`ora/orchestrator/backends`, `ora/app`, `ora/policy`, `ora/robot_client`,
`ora/cell`, `ora-pi`, `viewer`, `tests`, `scripts`, `examples`. Every `README.md` under
`ora/` ships with the wheel as package data.

## 10. `viewer/` and `examples/`

- `viewer/` stays its own package (`ora-viewer`): it imports `ora`, `ora` never
  imports it, and it carries the HTML templates `ora` does not. The live world
  view moved into `ora/orchestrator/world_view.py`; the Viewer's offline
  builds are `ora-viewer compile previews|videos`. `ora-viewer dashboard`
  serves the run dashboard (`viewer/dashboard/`), which imports nothing from
  `ora`.
- `examples/`: `quickstart_smoke.py`, `official_suite/` (`run.py`,
  `scheduler.py`, `experiment.yaml`; the scheduler takes its leases from
  `ora/orchestrator/leases.py`) and `hosts/example-host.yaml`.

## 11. Removed and folded

| Item | Why |
|---|---|
| The RoboTwin v1 closed-skill profile `robotwin_toolchain_b2_closed_noservo_v1` and the code only it reached (`tools/placement*.py`, `tools/runtime/holding.py`, `tools/retention*.py`, the place and stop branches of the skill driver, ledger, candidates and receipts) | Only v1 advertised those tools; the default v2 excludes them. Shared components whose names say v1 stay, because v2 composes them |
| `context/projection.py`'s test-only inventory, externalize and checkpoint code (975 lines) and `context/preview_shortfall.py` | The TypeScript core is the implementation; the differential fixtures in `tests/data/context_differential` are frozen |
| `evidence/durability.py` and its always-`None` parameter | Never wired into production; the TypeScript core owns write durability |
| `pi/jcs.py` (to `tests/support/agent/jcs.py`), `tui/__main__.py`, ten unused `__main__` guards | Only tests used them, or nothing started them |
| `ora/orchestrator/cli.py`, `ora_cli.py` and the `ora_*` product modules | One parser and one command table in `ora/app/` |
| `ora/orchestrator/spec.py`'s `experiments.*` launch-gate path | It named a package that does not exist here; every plan that used it failed |
| `scripts/model-services`, `run-report`, `open-viewer`, `dev/audit-prompt-bundle`, `dev/audit-task-contracts`, `dev/compile-libero-previews`, `dev/compile-viewer-videos` | `ora services`, `ora report`, `ora dev …` and `ora-viewer` commands |
| Renamed, kept: `context/codex.py` to `context/observation/projector.py`, `delivery_trim_native.py` to `delivery/trim_envelope.py`, `providers/native.py` split into one module per wire protocol (`build_native_provider` is `factory.build_provider`) | The names were left over from the Codex and Native cores; the code serves the Pi path |

Not removed: `robotwin/vendored/{planner,robot}.py` (the payload copied into
RoboTwin), `policy/planners/native/*.md` (locked prompt layers) and
`policy/baselines/native-full-explicit-motion-direct-v4.json` (the default
baseline's id).

## 12. Files that may not move

| Files | Why | If moved or edited |
|---|---|---|
| The 82 RoboDojo worker-kernel files (`WORKER_KERNEL_PATHS` in `ora/orchestrator/robodojo_kernel.py`): 15 in `ora/contracts`, 5 in the `ora/environment` root (`base`, `execution_capture`, `factory`, `rpc`, `worker`), 2 in `environment/benchmarks` (`robodojo`, `robodojo_layouts`), 22 in `environment/controllers`, 38 in `environment/backends/robodojo` | Card and task qualification records hash them by path and bytes, comments included | Every card and task record is void until the RoboDojo lanes are re-qualified on a healthy Isaac card |
| The 9 RoboDojo play-kernel files (`PLAY_KERNEL_PATHS` in `ora/orchestrator/robodojo_play_kernel.py`): `robodojo/play_scenes/*.py`, `benchmarks/robodojo_play_layouts.py`, `benchmarks/robodojo_play_scenes.py`, `environment/robodojo_play_factory.py` | Every play-scene audit binds them | The play audit runs again |
| `ora/orchestrator/episode/action_executor.py` | The one orchestrator member of `robodojo_revalidation.DEPENDENCIES`, read by path from the plan's commit | Lane revalidation certificates made at another path or bytes do not load; the lane is re-signed (scene smoke, then revalidate) |
| `ora/orchestrator/robodojo_kernel.py`, `robodojo_qualification.py`, `robodojo_play_kernel.py`, `robodojo_play_audit.py` | Past commits are read at the first three paths for the kernel manifests (`WORKER_KERNEL_PATHS`, `PLAY_KERNEL_PATHS`) and their owner tuples; the play kernel's files name the fourth | May move only with the new path added to the owner tuples and the old one kept; `WORKER_KERNEL_PATHS` stays a frozen literal |
| `ora/orchestrator/robodojo_revalidation.py`, `robodojo_scene_smoke.py`, `robodojo_layout_smoke.py` | `CHECKER` and `CHECKER_FILES` name these paths | May move with those constants; their algorithms' output is frozen |
| Everything else under `ora/environment/backends/robodojo/` | Its installer writes `scripts/setup-robodojo` into its receipt, and its scene YAMLs are bound per scene | Re-sign the matching records |
| `ora/environment/backends/robotwin/{qualification,session,common}.py`, `ora/environment/benchmarks/robotwin.py`, `uv.lock`, `pyproject.toml` (the whole file) | The RoboTwin seed scans' `producer_dependency_hash` and `checker_hash` | New scans do not merge with older segments, and a running scan cannot resume |
| Module paths inside JSON (`tests/data/published-limits.json` owners, `policy.lock`-pinned embodiment descriptors) | Data that names code | Edit the JSON with the move, and re-pin a pinned file, in the same commit |
| What the model sees, the evidence schemas, the evaluability of sealed runs | Product invariants | `scripts/dev/compare-model-visible <base> HEAD` reports no difference; old sealed runs still evaluate and replay |

Comments in pinned files that still name pre-restructure module paths stay as
they are until those files are next re-signed; so do the worker kernel's
mentions of run forks (`execution_request.py`, `ik.py`, `ik_sampler.py`,
`kinematics_query.py`, `native_sequence.py`), a feature that has not landed;
what they say of a fork holds for a restore from a recorded boundary, the case
that exists.

## 13. The interface for training loops

ORA has no Python training code; a training loop needs three things, and each
exists:

1. **A replaceable model endpoint.** The policy is reached through a provider
   profile (`ora/agent/providers/`, `ora providers add`): a checkpoint being
   trained is one more OpenAI-compatible or Anthropic Messages endpoint.
2. **Identifiable calls.** Each episode's agent trace (`agent-events.jsonl`)
   records every model call with `run_id`, `policy_session_id`,
   `observation_step` and `model_call_id`, and the exact request bytes by
   sha256. These ids are not on the wire: only a profile that declares
   `session_affinity: x_session_id_run_v1` sends `X-Session-ID:
   ora:<run_id>:native`, so a server can group requests by run.
3. **Evidence export.** The reward is the run's `official_success`;
   `ora trace-validate RUN` checks the trace and `ora export-sft RUN` writes
   the accepted model calls as tool-call JSONL with their attribution, loss
   mask, provider usage and manifest (`evidence/trace_verify.py`,
   `evidence/sft_export.py`); `--view wire` exports every call as it was
   sent, each image once, with each call's token ids, and proves the export
   rebuilds the archived requests (`evidence/wire_export.py`,
   `evidence/token_ids.py`); `--view openai` packages it for sharing with a
   standard-library verify tool (`evidence/openai_package.py`,
   `evidence/openai_package_tool.py`).
   Concurrency comes from the resource leases and the suite queue.

[Connect a training loop](../usage/rl-interface.md) and `ora help rl-interface`
describe it for users.

## 14. Decisions taken

| # | Question | Decision |
|---|---|---|
| 1 | Name of the command-line package | `ora/app`: it holds the terminal UI, the web gateway and the backend process as well as the parser |
| 2 | Keep `action_executor.py` pinned at the orchestrator root, or move it into `episode/` and re-sign the RoboDojo lanes once; the same for renaming `robodojo_play_factory.py` | `action_executor.py` moved into `episode/` (with `stage_outcome.py`), and the RoboDojo lanes are re-signed once after the move. `robodojo_play_factory.py` kept its name and path, so the play audits stay valid |
| 3 | `context/projection.py`'s test-only code | Deleted; the differential fixtures are frozen |
| 4 | The RoboTwin v1 closed-skill profile | Deleted |
| 5 | Where the TypeScript-core parity tests live | `tests/agent_core/`, a marker-selected lane, not `tests/ora/agent/pi/` |
| 6 | The M0-M5 ladder | Stayed, as `ora ladder` in the developer group, until its retirement on 2026-11-02 ([policy README](../../ora/policy/README.md)) |
| 7 | `viewer/` as its own package or `ora/viewer/` | Its own package |
| 8 | The `scripts/ora` wrapper | Kept; only its target changed |
| 9 | Branch protection and a pull-request flow on `main` | After the restructure |

As built, a few choices differ from the first plan:

- Backend registry entries name a module per package
  (`ora.orchestrator.backends.robodojo.backend:BACKEND`), so each package's
  `__init__.py` stays empty (rule L9).
- The shared vocabularies went where rule L6 allows: tool names to
  `composition/surface/tool_names.py`, units to `composition/surface/units.py`,
  context slots to `composition/prompt/slots.py`, the SAM3 and
  Contact-GraspNet clients to `model_services/clients.py`.
- `environment_preflight.py`, which only forwarded RoboDojo, was removed, and
  `controller_preflight.py` (the RoboTwin control-binding check) became part
  of `backends/robotwin/preflight.py`.
- `evidence/trajectory.py`, `memory/artifacts.py` and `providers/__init__.py`
  stay as re-exporting facades for the orchestrator's imports.
- `scripts/setup-robodojo` stays (the pinned installer names it); the RoboTwin
  seed-scan subcommand is still `qualify`, not `scan`.
- `ora/environment/backends/robolab/` has no `__init__.py` yet.

## 15. What moved

| From | To |
|---|---|
| `ora/agent/tools/*.py` (87 flat modules) | `ora/agent/tools/<family>/` (ten families; `robot.py` became `runtime/base.py` with two mixins; `grasp/__init__.py` split into perception and grasp modules) |
| `ora/agent/composition/*.py` | `surface/`, `schemas/` (the `*_schema.py` binders, suffix dropped), `embodiment/`, `prompt/`, `task_contract/` |
| `ora/agent/context/*.py` | `observation/` (`codex.py` became `projector.py`), `delivery/`, `lossless/` |
| `ora/agent/tools/tool_names.py`, `context/policy_units.py`, `context/slots.py` | `composition/surface/tool_names.py`, `composition/surface/units.py`, `composition/prompt/slots.py` |
| `ora/agent/tools/repeated_failures.py` | `ora/agent/session_core/repeated_failures.py` |
| `ora/agent/tools/runtime/` (P21; `toolkit_dispatch`, `toolkit_tool_data` and `toolkit_inputs` dropped their prefix), `tools/grasp/panda_topdown.py`, `tools/perception/{backends,fallback_segmenter}.py`, `tools/observe/dual_arm.py` | `ora/agent/toolkit/host/runtime/`, `host/grasp/`, `host/perception/`, `host/observe_labels.py`; `set_gripper`'s advisory capability constant went into `host/runtime/core.py`, and the `ora/agent/tools` package is gone |
| `ora/agent/providers/native.py` | `openai_compatible.py`, `anthropic_*.py` and the other protocol modules; `factory.build_provider` |
| `ora/agent/evidence/trajectory.py` | `canonical`, `agent_events`, `trace_privacy`, `trace_verify`, `sft_export` (the old module re-exports them) |
| `ora/agent/memory/artifacts.py` | `markers`, `read_tool`, `catalog`, `tool_runtime` |
| `ora/orchestrator/spec_*.py`, `run_spec_inputs.py`, `yaml_cache.py`, `quickstart.py`, … | `ora/orchestrator/spec/` |
| `ora/orchestrator/resource*.py`, `provider_probe_*.py`, `host_storage.py`, … | `ora/orchestrator/preflight/` |
| `ora/orchestrator/episode_*.py`, `round_*.py`, `termination.py`, `stagnation.py`, `action_executor.py`, `stage_outcome.py`, … | `ora/orchestrator/episode/` |
| `ora/orchestrator/evaluation*.py`, `verdict_guard.py`, `metrics*.py`, `context_*.py`, `revalidation.py`, … | `ora/orchestrator/evaluation/` |
| `ora/orchestrator/run_report*.py`, `suite_*.py`, `play_scenes.py` | `reports/`, `suite/`, `play/` (`scenes.py` and `binding.py`) |
| Lease and launch code in `ora/orchestrator/cli.py`, `lease_claim.py` | `ora/orchestrator/leases.py`, `launch.py` |
| Backend-specific code in the orchestrator core | `ora/orchestrator/backends/<kind>/` behind `registry.py` |
| `ora/orchestrator/cli.py`, `ora_cli.py`, `ora_*.py`, `doctor.py`, `providers_cli.py`, `model_services_cli.py` | `ora/app/` (`ora.app.main:main`) |
| `viewer/live_world.py`, `viewer/static/live_world.js` | `ora/orchestrator/world_view.py`, `world_view_assets/` |
| `docs/guides/*.md` | `docs/usage/compute-usage.md`, `docs/getting-started/official-suite.md`, `docs/developer/add-backend.md`, `docs/developer/authoring/*.md` |
| `docs/guides/cli-reference.md` (hand-written) | `docs/usage/commands.md`; its RoboTwin qualification detail to `docs/reference/robotwin-qualification.md` |

`git log --follow` traces any file: every move was its own commit, separate
from the rewrites that followed it.
