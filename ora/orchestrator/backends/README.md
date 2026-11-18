# ora/orchestrator/backends

## Responsibility

Host-side orchestration that belongs to one `runtime.kind`: the kind's own
RunSpec fields, its launch preflight and qualification records, what the
Supervisor and the evaluation take from its runs, its free-play scenes, its
`ora run` cell and its `ora doctor` probe. The orchestrator core asks
`registry.py` for the backend of a plan's kind and calls the hooks that backend
declares; it never names a kind in its own rules.

## What belongs here, and what does not

| Path | Holds |
|---|---|
| `registry.py` | `BACKENDS` (`runtime.kind` -> `"module:attribute"`), `ALIASES` (`libero` -> `libero_pro`), the `OrchestratorBackend` hooks and `BackendCapabilities`, and `backend()`, `find()`, `backends()`, `canonical()`, `manifest_admitter()` |
| `synthetic.py` | The in-process Synthetic world: no simulator, record or qualification |
| `<kind>/backend.py` | The kind's `BACKEND`: its two required hooks (`resolve_runtime` in `runtime.py`, `preflight_runtime` in `preflight.py`), its capabilities and the optional hooks it implements |
| `libero_pro/` | Driver configuration, preflight, supervision binding, round evaluation, play scenes and the `ora run` cell |
| `robotwin/` | Seed scans (`evaluation_seeds.py`, `seed_lanes.py`, `seed_segments.py`) and their manifest rules (`seed_manifest.py`, `admission.py`, `qualification_exception.py`), preflight, play scenes and their audit (`play_audit.py`), the `ora run` cell |
| `robodojo/` | The RunSpec builder (`spec.py`), preflight, card, task and lane qualification, probes, physics attestation, supervision and evaluation of persistent and scheduled sessions, play scenes, the `ora run` cell, the `ora doctor` probe, and the L0 motion journals, metrics and control replay (`motion/`) |
| `robocasa/`, `robolab/` | Runtime fields, preflight, the RunSpec builder (`spec.py`) and the scripted sparse-motion qualification; for RoboLab also the RoboLab-120 roster and contract generator (`task_source.py` reads the pinned source, `task_wording.py` the public clauses per predicate family, `task_review.py` the private review, compatibility and witness plan, `task_catalog.py` the `ora dev robolab-tasks` driver and `ora spec robolab tasks`) and the seed-0 contract witness runner (`contract_witness.py`, `ora qualify robolab contract-witness`) |
| `yam/` | Runtime fields, preflight, the scripted sparse-motion qualification and free play on the cell as it stands (`play.py`: the explicit free-play RunSpec and the rig binding of `ora play --env yam`; `switches.py`: the station's default-off switches, one row each) |

Seven RoboDojo backend modules live in the orchestrator root instead of
`robodojo/`, because qualification records and past commits name their paths:
`robodojo_kernel.py`, `robodojo_qualification.py`, `robodojo_play_kernel.py`,
`robodojo_revalidation.py`, `robodojo_scene_smoke.py`,
`robodojo_layout_smoke.py` and `robodojo_play_audit.py`. The import rules count
them as part of this backend, and `robodojo/` imports them from there. The same
holds for one environment-layer file, `ora/environment/robodojo_play_factory.py`,
which the play kernel hashes ([orchestrator README](../README.md)).

Not here: simulator code and anything that imports a simulator (that is
`ora/environment/backends/<kind>/`, run in the isolated worker; a backend here
may import that kind's host-safe modules under `ora.environment`, such as
`ora.environment.benchmarks.<kind>`), and rules every kind shares (those stay
in the core, with a capability or an optional hook where a kind differs).

## Dependency direction

The core imports `registry.py` only, and never calls `backend()` at import
time: a backend is imported on first use, from its string. A backend imports
the core freely (`spec/`, `preflight/`, `episode/`, `evaluation/`, `play/`,
the root modules) and never another backend. Every package `__init__.py` here
holds a docstring only, so importing one backend module does not load its
siblings; `registry.backends()` is the one call that imports all of them.

## Entry points

- `registry.backend(kind)` (raises `UnknownBackend`) and `registry.find(kind)`
  (`None` for an absent or foreign kind); hooks are attributes of the returned
  `OrchestratorBackend`, capabilities are `.capabilities`.
- Adding a kind: a package with a `backend.py` declaring `BACKEND`, an entry in
  `BACKENDS`, the environment definition in `ora/environment/definitions.py`,
  and capability data instead of new kind checks in the core.
- Operator tools: a backend declares them as `registry.Tool`s, and `ora`
  runs them by kind: `spec_tools` (`ora spec ENV TOOL`: `tasks`, `spec`,
  `compose`, `language`), `qualify_tools` (`ora qualify ENV TOOL`: RoboTwin's
  seed scans from `robotwin/evaluation_seeds.py`, the RoboDojo lane from
  `robodojo/lane_qualification.py` with its physics probe and `motion/`
  replay and metrics, the RoboCasa365 and RoboLab `motion` records) and
  `play_audit` (`ora play audit --env ENV`). `ora spec ENV` and `ora qualify
  ENV` list them. Only two keep a `python -m` entry point, because a run cell
  starts them as subprocesses with `Tool.command`:
  `robotwin.evaluation_seeds` (`qualify`) and `robodojo.lane_qualification`
  (`card`, `task`); `ora.orchestrator.robodojo_play_audit` keeps its own
  because its bytes are pinned.

## Documentation

- [Orchestrator README](../README.md)
- [RoboTwin](../../../docs/environments/robotwin.md),
  [LIBERO](../../../docs/environments/libero.md),
  [RoboDojo](../../../docs/environments/robodojo.md) and
  [RoboCasa365 and RoboLab](../../../docs/reference/robocasa-robolab-backends.md)
- [RoboDojo motion qualification](../../../docs/reference/robodojo-motion-qualification.md)
- [Repository structure](../../../docs/developer/repository-structure.md),
  sections 5 and 12


Toolkit live replay uses the optional `toolkit_live` registry hook. RoboDojo's
`toolkit_live.py` claims a caller-selected slot, launches the separate diagnostic
entry, verifies every control step and the boundary, then yields the session.
It always closes the process and releases its own lease/scratch. No existing
policy-worker or qualification entry point is replaced.

RoboDojo `qualification.py` is the `ora qualify robodojo motion` host. Its
numerical suite remains in the environment backend and is explicitly tested
host-safe: importing it must not import Isaac, Omni, Kit or the simulator package.
It starts the plant only through `environment.factory`, with the ordinary
host-owned execution-state callbacks, and plans every move the suite asks for
with the toolkit lib (`MatrixPlanner`), as the default tools plan an episode's
(`docs/reference/robodojo-motion-qualification.md`).
