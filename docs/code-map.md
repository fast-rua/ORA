# Code map

Every package carries a README that says what it is responsible for, what
belongs in it and what does not, which way its imports go, how it is entered
and which pages document it. Start from the table below and follow the
README. Code that belongs to one simulator lives in only two places:
`ora/environment/backends/<sim>/`, which runs in that simulator's own
interpreter, and `ora/orchestrator/backends/<kind>/`, the host side of the
same backend. Everything else is shared by every environment; the
[architecture](architecture.md) page shows the layers.

## Packages

| Package | What it owns | README |
|---|---|---|
| `ora/` | The Python package and its import rules | [ora](../ora/README.md) |
| `ora/contracts/` | The versioned action envelope and the pose, motion, refusal and receipt contracts the agent and the environment share | [contracts](../ora/contracts/README.md) |
| `ora/environment/` | Environment sessions, the registry of environments, the isolated worker and its local RPC, Synthetic | [environment](../ora/environment/README.md) |
| `ora/environment/benchmarks/` | Host-safe benchmark identities, rosters, pins and play-scene registries | [benchmarks](../ora/environment/benchmarks/README.md) |
| `ora/environment/controllers/` | Host-safe motion-transaction, safety and per-embodiment control contracts | [controllers](../ora/environment/controllers/README.md) |
| `ora/environment/planners/` | Motion planner workers behind the planner-neutral contract (`curobo_v2/`) and the path timing worker (`timing/`), each under its own interpreter | [planners](../ora/environment/planners/README.md) |
| `ora/environment/backends/<sim>/` | The simulator adapters, imported only by the isolated worker | [LIBERO](../ora/environment/backends/libero_pro/README.md), [RoboTwin](../ora/environment/backends/robotwin/README.md), [RoboDojo](../ora/environment/backends/robodojo/README.md), [RoboCasa365](../ora/environment/backends/robocasa/README.md), [RoboLab](../ora/environment/backends/robolab/README.md) |
| `ora/agent/` | The policy side: the Python half of the Pi core, providers, model services, the agent trace and its export | [agent](../ora/agent/README.md) |
| `ora/agent/code_mode/` | Code mode (`policy.code_mode`): the script contract, the code tools, the bubblewrap sandbox, the `robot` library and the extended library (`ora_lib`, the observation mount), the persistent workspace | [code_mode](../ora/agent/code_mode/README.md) |
| `ora/agent/toolkit/` | Runtime/replay public data interface and tool authoring capabilities | [toolkit](../ora/agent/toolkit/README.md) |
| `ora/agent/toolkit/interface/` | Classified data capabilities and runtime/replay wire adapters | [interface](../ora/agent/toolkit/interface/README.md) |
| `ora/agent/toolkit/interface/perception/` | Explicit external model service calls | [interface/perception](../ora/agent/toolkit/interface/perception/README.md) |
| `ora/agent/toolkit/interface/state_access/` | Current state reads and tool namespace writes | [interface/state_access](../ora/agent/toolkit/interface/state_access/README.md) |
| `ora/agent/toolkit/interface/sim_query/` | Worker planner requests and snapshot refusals | [interface/sim_query](../ora/agent/toolkit/interface/sim_query/README.md) |
| `ora/agent/toolkit/interface/sim_exec/` | Action submission and snapshot refusals | [interface/sim_exec](../ora/agent/toolkit/interface/sim_exec/README.md) |
| `ora/agent/toolkit/interface/lifecycle/` | Episode control and replay output intents | [interface/lifecycle](../ora/agent/toolkit/interface/lifecycle/README.md) |
| `ora/agent/toolkit/interface/privileged/` | RSI-only privileged state reads | [interface/privileged](../ora/agent/toolkit/interface/privileged/README.md) |
| `ora/agent/toolkit/interface/artifacts/` | Granted blob and tool artifact access | [interface/artifacts](../ora/agent/toolkit/interface/artifacts/README.md) |
| `ora/agent/toolkit/interface/compute/` | Pure numeric calculation | [interface/compute](../ora/agent/toolkit/interface/compute/README.md) |
| `ora/agent/toolkit/interface/context/` | Context attachment declarations | [interface/context](../ora/agent/toolkit/interface/context/README.md) |
| `ora/agent/toolkit/state/` | Supervisor-owned namespaces, permissions and publication snapshots | [state](../ora/agent/toolkit/state/README.md) |
| `ora/agent/toolkit/lib/` | Immutable typed capabilities and pure geometry, accessing data only through the interface | [lib](../ora/agent/toolkit/lib/README.md) |
| `ora/agent/toolkit/tools/` | Atomic tool migration and local policy helpers; public data interface only | [tools](../ora/agent/toolkit/tools/README.md) |
| `ora/agent/toolkit/tools/grasp_plan/` | Public grasp candidate policy and review | [grasp_plan](../ora/agent/toolkit/tools/grasp_plan/README.md) |
| `ora/agent/toolkit/tools/grasp_execute/` | Prepared commitment and one action proposal | [grasp_execute](../ora/agent/toolkit/tools/grasp_execute/README.md) |
| `ora/agent/toolkit/tools/move_pose/` | Measured-frame pose compilation, IK references and one action proposal. | [move_pose](../ora/agent/toolkit/tools/move_pose/README.md) |
| `ora/agent/toolkit/tools/move_to/` | Translation-only motion with the measured orientation held. | [move_to](../ora/agent/toolkit/tools/move_to/README.md) |
| `ora/agent/toolkit/tools/compute/` | Restricted numeric expressions over caller-supplied bindings. | [compute](../ora/agent/toolkit/tools/compute/README.md) |
| `ora/agent/toolkit/tools/check_task/` | The sole capability-gated data route to official task judgment. | [check_task](../ora/agent/toolkit/tools/check_task/README.md) |
| `ora/agent/toolkit/tools/finish_episode/` | Episode exit policy and the existing finish publication format. | [finish_episode](../ora/agent/toolkit/tools/finish_episode/README.md) |
| `ora/agent/toolkit/tools/observe/` | Current annotated cameras and retained raw public camera images. | [observe](../ora/agent/toolkit/tools/observe/README.md) |
| `ora/agent/toolkit/examples/` | Editable replay scripts outside the immutable lib | [examples](../ora/agent/toolkit/examples/README.md) |
| `ora/agent/toolkit/host/` | The trusted host the toolkit's tools run under: their data interfaces, I/O recording, and the tool runtime frame each episode builds (`runtime/`) | [host](../ora/agent/toolkit/host/README.md) |
| `ora/agent/composition/` | ToolSurface, embodiment, prompt-layer and task-contract resolution | [composition](../ora/agent/composition/README.md) |
| `ora/agent/context/` | What an observation and a tool result deliver to the policy | [context](../ora/agent/context/README.md) |
| `ora/orchestrator/` | RunSpec to RunPlan (`spec/`), preflight, leases and launch, the Supervisor's episode loop (`episode/`), evaluation and sealing, reports, the suite queue, play scenes | [orchestrator](../ora/orchestrator/README.md) |
| `ora/orchestrator/backends/` | The host side of each `runtime.kind`: RunSpec fields, preflight, qualification, seed scans, evaluation hooks, run cells, behind one registry | [backends](../ora/orchestrator/backends/README.md) |
| `ora/orchestrator/toolkit/` | Recorded-call selection, perception binding and script evaluation (`ora toolkit replay`) | [toolkit replay](../ora/orchestrator/toolkit/README.md) |
| `ora/app/` | The `ora` command: one parser, one command table, the help topics, the terminal and web front ends | [app](../ora/app/README.md) |
| `ora/policy/` | Prompt layers, ToolSurfaces, embodiment and environment assets and the default baseline, pinned by `policy.lock` | [policy](../ora/policy/README.md) |
| `ora/robot_client/` | The standard-library `ora-robot` client of a free-play run | [robot_client](../ora/robot_client/README.md) |
| `ora/cell/` | The real-cell kit (`ora cell`): the cell config and calibration, the cell lock, bus and camera checks, camera calibration, the depth service, the cell board and the operator verbs, written against a robot's `CellAdapter` | [cell](../ora/cell/README.md) |
| `ora-pi/` | The TypeScript agent core (provider loop, request assembly, context, agent evidence) and ORA's web page | [ora-pi](../ora-pi/README.md) |
| `ora-pi/src/context/stages/` | The context stage driver and the stages a scheme lists (v6 and v8) | [stages](../ora-pi/src/context/stages/README.md) |
| `viewer/` | The read-only run viewer (`ora-viewer`) and run dashboard (`ora-viewer dashboard`, `viewer/dashboard/`), its own package | [viewer](../viewer/README.md) |
| `tests/` | The CPU-only suite, a tree that mirrors the code | [tests](../tests/README.md) |
| `scripts/` | The `ora`, `ora-remote` and docs wrappers and the development checks | [scripts](../scripts/README.md) |
| `examples/` | The quickstart smoke, the official-suite cells and the host template | [examples](../examples/README.md) |
| `docs-site/` | The documentation site's build | [docs-site](../docs-site/README.md) |

## Where a feature lives

| Feature | Location |
|---|---|
| Plan resolution, defaults and source identity | `ora/orchestrator/spec/` (`run_spec.py` and its section modules, `defaults.py`, `source.py`, `source_overlay.py` for runs on a base commit with replaced tool files); the compactor's own Provider binding in `spec/context_compactor.py` |
| The `ora` command | `ora/app/` ([README](../ora/app/README.md)): `main.py` (one parser, the command table and the grouped help) and one module per command group (`plan.py`, `runs.py`, `evidence.py`, `run.py`, `setup.py`, `doctor.py`, `catalog.py`, `providers.py`, `services.py`, and `report.py`, `spec.py`, `qualify.py`, `suite.py`, `dev.py` over the operator tools' own parsers) |
| Admission, leases and supervision | `preflight/` (`gates.py`, resources and host storage), `leases.py`, `launch.py` (launch, supervisor and landing seal), `supervise.py` (the supervisor process); `ora/version.py` (the version `ora --version` and the transcript headers show) |
| Terminal watch, interactive sessions and local CPU entry | `ora/agent/tui/` and `ora-pi/src/watch/` (transcript and its pi-tui front end); `ora/app/interactive.py`, `backend.py`, `task_catalog.py`, `local.py`, `local_setup.py`, `world.py`, `remote.py`, `remote_session.py`, `web.py`, `web_gateway.py`, `web_cell.py` (a real cell's controls on the page), `replay.py` (a finished run's replay), `replay_export.py` (its video); `ora/orchestrator/episode/operator_control.py`; `ora-pi/src/tui/`, `ora-pi/src/rpc/` (Pi RPC host, `ora-rpc-attach`), `ora-pi/web/` (ORA's web page and its server, seeded from Watercol/pi-web-ui); `ora/orchestrator/world_view.py`; `ora/environment/synthetic_capture.py`; `ora/agent/session_core/operator_messages.py` |
| Episode coordination and official evaluation | `ora/orchestrator/episode/` (`coordinator.py`, `loop.py`, `action_executor.py`), `ora/orchestrator/evaluation/` (`evaluate.py` and its checks); the run report in `ora/orchestrator/reports/`, the official-suite queue in `ora/orchestrator/suite/`; task files (`ora task`) in `ora/orchestrator/task/` |
| Free play (`ora play`): scenes, instructions in turn, Pi as a coding agent, its evidence | `ora/app/play.py`, `ora/orchestrator/spec/play.py`, `episode/play.py`, `evaluation/play.py`, `play/scenes.py`; `ora-pi/src/play/`; `ora/agent/evidence/play_trace.py`; `ora/policy/prompt_bundles/templates/episode-task-play-v2.md` |
| `native_api` policy lifecycle and provider/tool loop (the Pi agent core) | `ora-pi/src/core/session.ts`, `dispatch.ts`; its Python half in `ora/agent/pi/` |
| Context packs (`policy.context_pack`) | `ora/agent/composition/prompt/context_pack.py` (pack, profile, texts, overlay, images), `ora/orchestrator/spec/context_pack.py` (plan binding), `ora/agent/pi/context_pack_rpc.py` (session RPC), `ora-pi/src/context/context_pack.ts` and `compact_view.ts` (delivery) |
| Provider protocols and credentials | `ora/agent/providers/`; a bound rollout compactor's own provider in `compactor_binding.py` (Pi: `ora/agent/pi/factory.py`, `ora-pi/src/core/context_wiring.ts`) |
| Policy composition, embodiment and tool contracts | `ora/agent/composition/` |
| The Python half of a `native_api` session on the Pi core: tool gateway and provider tool definitions, static prefix, observation and tool-result projections, receipt projection, the `PolicySession` wrapper | `ora/agent/session_core/` |
| Context projection, compaction and delivery | `ora/agent/context/` |
| Recoverable context artifact storage | `ora/agent/memory/` (historical package name; no Task Memory) |
| Policy-visible tools and closed skills | `ora/agent/toolkit/tools/` (the tools), `ora/agent/toolkit/host/` (their host, with the runtime frame in `host/runtime/`) |
| Resident perception services | `ora/agent/model_services/` |
| Trace and policy-state evidence | `ora/agent/evidence/` |
| Action envelopes and shared validation | `ora/contracts/` |
| Free-play robot scripting (`ora-robot`) | `ora/robot_client/` (standard-library client), `ora/agent/pi/robot_socket.py` (socket and script calls; `script_stop.py`, a real cell's stop ending a script), `ora/orchestrator/episode/script_actions.py` (Supervisor admission and execution) |
| Transport-neutral sessions and workers | `ora/environment/` |
| Host-safe benchmark identities and task catalogs | `ora/environment/benchmarks/` |
| Host-safe controller contracts | `ora/environment/controllers/` |
| Motion planning: the planner-neutral contract, the public-depth world, the FK parity check, the host's copy of the planner's Panda spheres, the base-placement search and the base move's check (footprint and carried arm), the planner workers | `ora/environment/controllers/motion_planner.py`, `depth_world.py`, `depth_world_inputs.py`, `depth_esdf.py`, `fk_parity.py`, `robot_spheres.py`, `base_placement.py`, `held_depth.py`; `ora/environment/planners/` |
| Simulator implementations | `ora/environment/backends/{robotwin,libero_pro,robodojo,robocasa,robolab}/` |
| Host-side orchestration one backend owns: RunSpec builders, preflight, qualification, seed scans, evaluation, probes | `ora/orchestrator/backends/{robodojo,robotwin,libero_pro,robocasa,robolab}/`; RoboDojo motion journals, metrics and control replay in `backends/robodojo/motion/`. The RoboDojo worker and play kernels, the scene and layout smokes, the lane revalidation checker and the play-scene audit stay at `ora/orchestrator/robodojo_*.py`: kernel history, checker identity and the play kernel's files name those paths |
| Versioned prompt/tool/environment assets | `ora/policy/`; integrity in `policy.lock` |
| One-cell and official-suite examples | `examples/`, `ora/orchestrator/spec/benchmark.py`, `ora/orchestrator/backends/{robodojo,robocasa,robolab}/spec.py` |
| Play scenes (a qualified scene without its task) | `ora/orchestrator/play/scenes.py` (scenes, `build_play_spec`), `ora/orchestrator/play/binding.py` (`bind_play_runtime`) and each backend's `play` hook in `ora/orchestrator/backends/<backend>/play.py`; the backends' play bindings in `backends/libero_pro/direct.py` and `backends/robotwin/session.py` |
| ORA's own RoboTwin play scenes (`--scene-set tabletop-v1`) and their audit | `ora/environment/benchmarks/robotwin_play_scenes.py` (host-safe registry: sets, scenes, source binding); `ora/environment/backends/robotwin/play_scenes/` (worker-only `base.py`, `tabletop_v1.py`); `backends/robotwin/play_scene_audit.py` (the worker's build, settle and render checks); `ora/orchestrator/backends/robotwin/play_audit.py` (the audit command, its record and `play_scene_admission`) |
| ORA's own RoboDojo play scenes (`ora play --env robodojo`, `isaac-tabletop-v1`) and their audit | `ora/environment/benchmarks/robodojo_play_scenes.py` (host-safe registry code: sets, index loading, per-scene source binding, `validate_play_runtime`, the audit's criteria), `robodojo_play_layouts.py` (host-safe per-seed layout generator); `ora/environment/backends/robodojo/play_scenes/` (worker-only: scene YAMLs and each set's `index.yml` of scene entries, `lifecycle.py` reward-free TaskEnv, `operator_view.py` the operator's third-person camera, `bootstrap.py` Kit boot, `session.py` worker backend `robodojo_play`, `audit.py` checks); `ora/environment/robodojo_play_factory.py` (worker config); `ora/orchestrator/robodojo_play_kernel.py` (code-only play kernel identity), `robodojo_play_audit.py` (the audit command, its record and `play_scene_admission`), `backends/robodojo/play_preflight.py` (launch checks) |
| Read-only run registry and media inspection | `viewer/` |

Use the explicit canonical subpackages for imports. The package root does not re-export implementation classes. Add shared behavior to its owner rather than to a launcher or benchmark-specific example. [Contributor instructions](../AGENTS.md) describe the required checks, and the [repository structure](developer/repository-structure.md) states the import rules and the files that may not move.
