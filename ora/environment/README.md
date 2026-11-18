# ora/environment

## Responsibility

The world side of a run. An environment session is one trusted external-world
episode: it publishes public observations, executes admitted actions with
measured receipts, and reads the benchmark's own official verdict. This
package holds the transport-neutral session contract, the registry that says
which environments exist and how each one is reached, the isolated worker
process and its local RPC, the in-process Synthetic world, and the simulator
backends.

| Path | Holds |
|---|---|
| `base.py`, `contracts.py`, `session.py` | The `EnvironmentSession` protocol and its validators, the canonical session with its policy-safe runtime descriptor, and the descriptor-validating facade the orchestrator holds |
| `definitions.py` | One `EnvironmentDefinition` per environment id (`synthetic`, `libero_pro`, `robotwin`, `robodojo`, `robocasa`, `robolab`): transports, embodiments, action schemas, capabilities, preflight id and the `worker_loader` string (`"module:attr"`) of its worker session |
| `factory.py` | `build_environment_session`: the composition root that turns a plan's runtime into a session, in process for Synthetic and over local RPC for every simulator |
| `rpc.py`, `worker.py` | The host-side `LocalRpcEnvironmentSession`, which starts `python -m ora.environment.worker` under the plan's `runtime.python_env` (the simulator's own interpreter), and that worker, which imports the session class of the configured backend (one branch per backend, the class each definition's `worker_loader` names) and serves one session |
| `synthetic.py`, `synthetic_capture.py` | Synthetic, the simulator-free world of the quickstart and the tests (`official_success: null`), and its drawn capture |
| `execution_capture.py`, `controller_replay.py`, `replay_worker.py` | Control-only recording of the original execution's pixels; compact controller traces and the worker that replays them for the Viewer |
| `robolab_replay.py` | The host side of a RoboLab replay: the job on the environment worker a run uses, and the trace's clock against the receipts |
| `scene_trace.py` | The scene trace of a re-executed run (`ora replay RUN --scene`, `ora.scene-trace.v1`): the simulator-neutral format and writer, which run in the simulator's interpreter with numpy; each backend's exporter reads its own bodies |
| `mujoco_scene_trace.py` | The scene trace's MuJoCo reader, for any scene on MuJoCo: bodies, visual geoms and meshes from the model, poses from the data, each arm's TCP as a body plus an offset. It imports no MuJoCo; a backend hands it its simulator and names its robots (`backends/libero_pro/scene_trace.py`, `backends/robocasa/scene_trace.py`) |
| `incident_replay_worker.py` | One bad action of a sealed run again: the recorded prefix on the session a live episode builds, then the action's recorded controls or its target through `session.execute` |
| `graphics_probe.py` | The offscreen graphics probe preflight runs by path in a simulator interpreter, and the choice of Mesa's software EGL device for a worker that renders on the CPU (`bind_software_egl_device`) |
| `workspace_profiles.py` | Source-owned hard action envelopes per environment |
| `robodojo_play_factory.py` | The host-side composition of a RoboDojo free-play worker (backend `robodojo_play`), which the episode coordinator swaps in for a RoboDojo play run |
| `benchmarks/` | Host-safe benchmark identities, rosters, suite expansion and source pins ([README](benchmarks/README.md)) |
| `controllers/` | Host-safe controller contracts and safety envelopes shared by preflight and the workers ([README](controllers/README.md)) |
| `planners/` | Motion planner workers behind the planner-neutral contract of `controllers/motion_planner.py`, each started by module name under its own interpreter ([README](planners/README.md)) |
| `backends/<sim>/` | The simulator adapters, imported only inside the isolated worker: [LIBERO / LIBERO-Pro](backends/libero_pro/README.md), [RoboTwin](backends/robotwin/README.md), [RoboDojo](backends/robodojo/README.md), [RoboCasa365](backends/robocasa/README.md), [RoboLab](backends/robolab/README.md) |

## What belongs here, and what does not

Here: what runs on the world's side of the boundary, and the host-safe facts
the host needs about it without importing a simulator (`benchmarks/`,
`controllers/`, `definitions.py`). Simulator imports belong only in
`backends/<sim>/`, which run in that simulator's interpreter.

Not here: deciding whether an action is admitted (the Supervisor in
`ora/orchestrator/episode/`), anything the policy sees as tools or text
(`ora/agent/`), and the host-side rules of one backend: its RunSpec fields,
preflight, seed or lane qualification and evaluation hooks are in
`ora/orchestrator/backends/<kind>/`. A new environment adds its definition
here and its backend under `backends/`, and its host side under the
orchestrator ([add an environment](../../docs/developer/add-backend.md)).
A new backend also needs a branch in `worker.py` and a builder in
`factory.py`, two pinned files (below), so adding one re-signs the RoboDojo
qualification records.

**Pinned files.** Five root modules are members of the RoboDojo worker kernel
(`WORKER_KERNEL_PATHS` in `ora/orchestrator/robodojo_kernel.py`), hashed by
path and bytes into every RoboDojo card and task qualification record:
`base.py`, `execution_capture.py`, `factory.py`, `rpc.py` and `worker.py`.
`robodojo_play_factory.py` is a member of the play kernel (`PLAY_KERNEL_PATHS`
in `ora/orchestrator/robodojo_play_kernel.py`), which every RoboDojo play-scene
audit binds; it kept its RoboDojo name for that reason, and renaming it re-runs
the play audit. Do not rename these files or change a byte of them, comments
included, without re-signing those records. The pinned files of
`benchmarks/`, `controllers/` and `backends/` are listed in their READMEs.

## Dependency direction

`ora.environment` imports `ora.contracts` and itself, never `ora.agent`,
`ora.orchestrator` or `ora.app` (rule L1). Nothing imports `planners/`: a
planner worker is started by module name, and it imports `controllers/` only
(rule L4). The core (this directory,
`benchmarks/` and `controllers/`) never imports `backends/` (rule L4), except
the two isolated-interpreter entry points `worker.py` and `replay_worker.py`:
they run inside the simulator's interpreter and are where the `worker_loader`
strings of `definitions.py` resolve. A backend imports the core, `benchmarks/`,
`controllers/` and `ora.contracts`. Two backends reuse a sibling that runs on
the same simulator stack: `robocasa/` imports `libero_pro/` helpers (both are
robosuite) and `robolab/` imports `robodojo/`'s Isaac Sim graphics and crash
reporting helpers; no other backend imports another.
`__init__.py` is a lazy facade: importing `ora.environment` loads only that
module, and each exported name resolves to its owning module on first use.

## Entry points

- `from ora.environment import build_environment_session, environment_definition, EnvironmentSession`.
- `python -m ora.environment.worker --socket S --config C`: started by
  `rpc.py` under the simulator's interpreter, never by hand.
- `python -m ora.environment.replay_worker`: started by `viewer/replay.py`.
- `python -m ora.environment.incident_replay_worker`: started by `ora replay --incident`.
- `graphics_probe.py`, run by path by the LIBERO launch preflight.
- The backends' own process entries (probes, previews, audits, installers)
  are listed in [ora/app/README.md](../app/README.md#entry-points).

## Documentation

- [Environment sessions API](../../docs/api/environment.md)
- [Environments](../../docs/environments/README.md), one page per simulator
- [Runtime lifecycle](../../docs/reference/runtime-lifecycle.md)
- [Add an environment](../../docs/developer/add-backend.md) and
  [environment compatibility](../../docs/developer/authoring/environment-compatibility.md)
- [Repository structure](../../docs/developer/repository-structure.md), the
  pinned files

RoboDojo requests also carry `ora.execution-state.v1`: the orchestrator-owned
state accessor supplies commands and explicit recovery targets and consumes
updates. RPC stores no independent command cache. The worker installs these
values in a request scope and removes them on return, including exceptions.
