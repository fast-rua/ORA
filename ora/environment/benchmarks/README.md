# ora/environment/benchmarks

## Responsibility

What the host needs to know about each benchmark without importing its
simulator: task identities and rosters, suite expansion, the pinned source and
asset revisions, the public task language, which episodes official evaluation
draws from, and the registries of ORA's own free-play scenes. RunSpec
builders, launch preflight, the Viewer and the isolated workers all read the
same modules, so the host and the worker agree on what a task is.

| Module | Holds |
|---|---|
| `libero.py`, `libero_pro.py`, `libero_pro_language.py` | The five original LIBERO base suites, the LIBERO-Pro suites and robustness cases, and the pinned BDDL language of the forty source-paired LIBERO-Pro tasks |
| `robotwin.py`, `robotwin_instructions.py`, `robotwin-instructions-v1.json`, `robotwin-episode-instructions-unseen-v1.json` | The pinned RoboTwin 2.0 randomized evaluation (tasks, configurations, revision) and the public instruction a RunSpec delivers for each task |
| `robodojo.py`, `robodojo_asset_revision.py`, `robodojo_physics.py` | RoboDojo identities and admission, the reviewed task-specific asset selection and each task's physics device |
| `robodojo_layouts.py`, `robodojo_layout_catalog.py`, `robodojo_layout_reach.py`, `robodojo-layout-pools-v2.json` | The per-round layout schedule of a persistent RoboDojo session, the human-reviewed layout pools and the trusted-side reach footprint of a saved layout |
| `robocasa.py`, `robolab.py` | RoboCasa365 and RoboLab pins, admitted task rosters and public context |
| `evaluation_sets.py`, `evaluation-sets-v1.json` | Which episodes official evaluation draws from, so practice can refuse them |
| `robotwin_play_scenes.py` | ORA's own RoboTwin free-play scene sets (`tabletop-v1`, `household-v1`): sets, scenes, source binding and the audit criteria |
| `robodojo_play_scenes.py`, `robodojo_play_layouts.py` | ORA's own RoboDojo free-play scene sets (`isaac-tabletop-v1`): index loading, per-scene source binding, `validate_play_runtime`, and the deterministic per-seed layout generator |

## What belongs here, and what does not

Here: facts about a benchmark that hold without its simulator: identities,
pins, rosters, public language and pure-Python generators. Not here: anything
that imports a simulator or builds a scene (`ora/environment/backends/<sim>/`),
host-side rules such as RunSpec building, seed scans and lane qualification
(`ora/orchestrator/backends/<kind>/`), and policy-visible task contracts
(`ora/policy/environments/`, resolved by `ora/agent/composition/task_contract/`).
The modules stay flat, one or a few per benchmark.

The four JSON files are policy assets read by path: `policy.lock` pins them
(`ora/orchestrator/spec/policy_lock.py` covers `ora/environment/benchmarks/*.json`),
so a changed file is re-pinned with `./scripts/ora lock` in the same commit.

**Pinned files.** These modules are hashed by path and bytes into existing
qualification records; do not rename them or change a byte, comments included,
without re-signing those records:

- `robodojo.py` and `robodojo_layouts.py`: RoboDojo worker kernel
  (`WORKER_KERNEL_PATHS`), bound by every card and task record.
- `robodojo_play_layouts.py` and `robodojo_play_scenes.py`: RoboDojo play
  kernel (`PLAY_KERNEL_PATHS`), bound by every play-scene audit.
- `robotwin.py`: a RoboTwin seed-scan producer dependency
  (`ora/orchestrator/backends/robotwin/evaluation_seeds.py`
  `source_dependencies`); a changed byte stops new scans from merging with
  older segments.

## Dependency direction

These modules import the standard library, PyYAML, each other and
`ora/environment/workspace_profiles.py`, never a simulator, an
`ora/environment/backends/` module or any `ora` package above the environment.
The orchestrator, the app, the Viewer and the isolated workers import them.
`__init__.py` holds only a docstring.

## Entry points

- Imported by module, for example `ora.environment.benchmarks.robotwin.TASKS`
  or `ora.environment.benchmarks.libero_pro`.
- `ora spec ENV tasks` prints a benchmark's roster from these modules.
- No module here has a process entry point.

## Documentation

- [Environments](../../../docs/environments/README.md) and the generated task
  pages
- [Practice initial states and evaluation sets](../../../docs/reference/configuration-and-evidence.md#practice-initial-states-and-evaluation-sets)
- [Evaluation protocol](../../../docs/evaluation/protocol.md)
- [Free play](../../../docs/usage/play.md)
