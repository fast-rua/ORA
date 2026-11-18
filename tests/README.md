# tests

## Responsibility

The deterministic test suite. Every test runs on a CPU-only machine and starts
no provider, simulator, GPU, resident model service or physical robot: backend
tests drive the real session code against fake plants, fake SDKs and recorded
numeric witnesses, and the agent-core tests drive the Pi core (`ora-pi/`)
against a scripted model server. The suite checks contracts, not simulator
physics or live service readiness; those are qualified on a simulator host
([`docs/developer/authoring/qualification.md`](../docs/developer/authoring/qualification.md)).

Each test module's docstring states the contract it protects. Read it before
changing an expected value. `tests/ora/app/test_main.py` fails on a module
without one and on a module whose directory this README does not name.

## What belongs here, and what does not

The tree mirrors the code: `tests/ora/<package path>/test_*.py` covers
`ora/<package path>/`, and `tests/viewer/` covers `viewer/`. A module that
covers more than one package lives with the package it protects most; a module
named after a backend (`test_robodojo_*`, `test_robotwin_*`, `test_libero_*`)
lives in that backend's directory, under `ora/environment/backends/` when it
drives the backend's session or worker code and under
`ora/orchestrator/backends/` when it drives the host side (admission,
qualification, specs, evaluation). Beside the mirror sit `tests/agent_core/`
(the Pi core), `tests/repository/` (repository guards), `tests/support/`
(shared helpers) and `tests/data/` (fixed inputs). Nothing else belongs at the
top of `tests/`; `tests/repository/test_structure.py` fails on another entry
there and on a directory under `tests/ora/` with no code directory to mirror.

| Directory | What it covers |
|---|---|
| `tests/agent_core/` | The agent core run by Pi (`ora-pi/`, TypeScript), held to goldens recorded from the retired Python Native core: Synthetic episodes whose requests and events equal their goldens under every context retention, the Anthropic wire, a rollout compactor bound to its own Provider endpoint, model and provider faults end to end, crash resume of the Pi core, an operator stop during a request, a read-only tool call, an action's execution or a mutation's dispatch, a world mutation refused beside other calls (`mutation_batch_rejected`), a core lost inside a nested call, the free-play robot socket (`ora-robot`) and its RPC multiplexing, the CLI quickstart on Pi, JSON serialization parity, opt-in replays of the sealed acceptance runs, context-pack delivery on the Pi core, and code mode's scripts (one parent invocation of the Supervisor's executor, a script's refusals, the trusted terminal boundary's refusal in code and tool-call mode; with `library: extended` the `query` and `explain` requests on the record as nested calls, each call's explain inputs, the run verifying and every explanation rendered again from the record, and a `base` script writing the events it wrote before). Every test here carries the `agent_core` marker. |
| `tests/ora/contracts/` | The shared action envelope (`ora/contracts/`): canonical Pose/Frame compilation and the committed skill-sequence action. |
| `tests/ora/agent/composition/` | How a run's policy is composed: the Native default baseline and the delivery guards that hold every enforced limit to what the policy is told. |
| `tests/ora/agent/composition/surface/` | ToolSurface resolution: backend and result-contract parity, published limits and help, canonical tool names, handler joins, routing, the closed-skill and correctness surfaces, argument envelopes and null optional arguments, policy units in degrees, derived grasp review, motion parameter delivery and the public-data successor default. |
| `tests/ora/agent/composition/schemas/` | Tool schemas: grasp rejection evidence in the current result schemas, the RoboLab grasp_plan compiler identity and the reach reference of the pose primitives. |
| `tests/ora/agent/composition/embodiment/` | Embodiment descriptors, the ARX cards, descriptor pins, environment prompt profiles and public geometry delivery. |
| `tests/ora/agent/composition/prompt/` | Prompt bundle authoring, the default bundle templates and the bundle quantity ratchet. |
| `tests/ora/agent/composition/task_contract/` | Task-contract activation, delivery and review. |
| `tests/ora/agent/context/` | Complete public RGB-D delivered to the policy's observation and both provider wires. |
| `tests/ora/agent/context/observation/` | Observation projection: the historical near window, the controlled point in each camera, public robot facts, skill observation views and the privileged-field denylist. |
| `tests/ora/agent/context/delivery/` | Delivery trimming and its prose, payload de-duplication and round static facts. |
| `tests/ora/agent/context/lossless/` | The lossless Context: management v5 stride and v6 watermark, canonical sizing, the rollout compactor and its verdict masking, render reuse. |
| `tests/ora/agent/code_mode/` | Code mode (`policy.code_mode`): the contract, its tools and prompt layers, the code tools' result schemas, the persistent workspace (edits, reads, snapshots), the `robot` library's typed results and filled parameters (`library: extended`, on recorded answers) and the bubblewrap sandbox with the `robot` library (no network, hidden host paths, time and memory limits, the workspace mount, a workspace file named after a module breaking no import with `library: extended` and the model's own modules still importing, `base` keeping Python's own path; skipped where bubblewrap is unavailable); the extended library's mounts (`library.py`, the masks of the segmented targets included); `pose_after` read from the mounted observation, on recorded action answers; `prompt: v3` (`test_planner.py`): refused without `library: extended`, `v2`/`base` composing byte for byte as before it (pinned digests), only the planner changing, the planner's trigger lines naming only what exists, its worked script run in the sandbox against recorded answers, and the `interaction` label per kind of surface (the note after `move_pose`'s index line in each arm's or the flat spelling, nothing where the schema has none, the label reaching the tool); `robot.explain` over the RPC (sandbox, `Result.call_id`); `robot.query` (`test_query.py`): each query equal to the toolkit lib function it wraps on the recorded RoboDojo observation (degrees in and out, to 1e-6 degrees), recording nothing, and its refusals (arguments, `query_unavailable`), the query documentation, and the allowance as a contract limit of `library: extended` only; a dry run's typed result (`Result.feasible`, `predicted_pose`, `route`, `screening`) and the `dry_run` argument and part of the arm-moving tools' documentation (`test_robot_library.py`, `test_docs.py`); the sandbox interpreter held to its lock (`test_sandbox_interpreter.py`: every package pinned with hashes, NumPy 1.26.4 and SciPy 1.13.1 kept, an interpreter that differs refused with the lock, the package, both versions and the build command, the interpreter recorded in the preflight check, Pillow, scikit-image and Shapely working inside the sandbox; the sandbox runs skipped where it is unavailable or the interpreter differs from the lock); the sandbox's host prerequisites (`test_sandbox_prerequisites.py`: the interpreter setup built found without `ORA_CODE_MODE_PYTHON`, an explicit value winning, and the preflight and doctor refusals saying what to install or set for `bwrap`, user namespaces and the interpreter, on fake `bwrap` scripts). |
| `tests/ora/agent/code_mode/ora_lib/` | The extended library's sandbox-side package (`ora_lib`, `library: extended`): its frames layer against `geometry_query`, its geometry and perception fragments (`depth_inliers`, the mask measurements, `object_geometry`) against the toolkit lib on the recorded RoboDojo frame, the SciPy equivalents of the lib's OpenCV steps against OpenCV, and its frames card against the frames layer. |
| `tests/ora/agent/memory/` | Recoverable Context artifacts of the lossless Context. |
| `tests/ora/agent/evidence/` | Agent-side evidence: inline fsync of evidence writes, public receipt binding, the trace verifier on discarded actions, the free-play trace check, the archive of a sealed run's provider requests and the SFT export. |
| `tests/ora/agent/providers/` | The provider wires and their bindings: OpenAI-compatible and Anthropic translation, streams and recovery deadlines, request options and tool strictness, credential preflight, the loopback gateway client, the compactor's own provider and an operator stop during provider work. |
| `tests/ora/agent/model_services/` | The resident SAM3 / Contact-GraspNet services: the operator and the Contact-GraspNet input policy (`ora services` is in `tests/ora/app/`). |
| `tests/ora/agent/session_core/` | The Python half every `native_api` session shares: the action summary, receipt projection and its feedback (planner failures, rejected candidates, perception), legacy delivery bytes, image call labels, tool-boundary recovery and the atomic target-state runtime. |
| `tests/ora/agent/toolkit/interface/` | Runtime/replay I/O contracts, strict cursors, public blobs, fake service recomputation and legacy snapshot bytes. |
| `tests/ora/agent/toolkit/interface/state_access/` | Category assembly, dual privilege grants, current-state reads and simulator-free snapshot capabilities. |
| `tests/ora/agent/toolkit/state/` | Namespace ownership, finite JSON, immutable publication chains and native-byte snapshot loading |
| `tests/ora/agent/toolkit/lib/` | Immutable typed capabilities, full candidate pools, pure metric geometry, replay classification, public-data access and versioned library identity. |
| `tests/ora/agent/toolkit/tools/` | Atomic tool fixed-data equivalence and static import/I/O boundary. |
| `tests/ora/agent/toolkit/` (`test_explain.py`, `test_stages.py`) | `explain`: the perception and arm-moving tools' stage templates held to their schemas (a dry run's record included), to the stage events the tools record and to recorded answers (golden renderings), the template language, the renderings of dry runs; `stage()` a no-op unless collected, tool results unchanged with and without collection. |
| `tests/ora/agent/toolkit/` (`test_stack_parity.py`) | One call on the toolkit, surface by surface: each row runs a call on a stand-in host and checks what the model gets back and `submitted-action.json` against the digest recorded while the retired handlers were its oracle (`tests/data/toolkit-goldens`). Stand-in hosts: no simulator behaviour. |
| `tests/ora/agent/toolkit/tools/vision_ground/` | Grounding result, query artifacts and target artifacts for 0/1/3 candidates, against goldens recorded beside the handlers. |
| `tests/ora/agent/toolkit/tools/grasp_plan/` | Full hybrid candidate result, review artifacts and service request for both X5 arms and fake raw pools (`test_tool.py`); `test_grip_site.py`: one call on LIBERO, RoboCasa, RoboLab and RoboTwin, a learned pool, an empty one and a target the fallback family cannot use; both against goldens. |
| `tests/ora/agent/toolkit/tools/grasp_execute/` | Prepared skill result, ledger, commitment and single action, and the legacy-visible submission, against goldens (`test_tool.py`); the stage loop and RoboTwin's precommit probe. |
| `tests/ora/agent/toolkit/tools/move_pose/` | Fixed-data move_pose/move_to equivalence, public errors and action bytes; move_pose's speeds, slow approach, screens and budget fit (`test_speed.py`); `dry_run` on move_pose and move_to (`test_dry_run.py`): the dry run is the plan the real call runs (predicted pose, IK solution and native actions equal the executed action's), it executes nothing whatever the tool tries, a plan the real call would refuse is its answer, a malformed request is still a refusal, no advertised schema carries the flag, and recorded runs' predictions against their executed end poses. |
| `tests/ora/agent/toolkit/tools/recover_pose/` | recover_pose's retrace of the latest recorded move (restated stack top, cut and continue, refusals naming return_home and move_pose, no fallback, not_executed replay) and the released targets route; `dry_run` of the retrace and of return_home (`test_dry_run.py`). |
| `tests/ora/agent/toolkit/tools/return_home/` | return_home's lift and screened unwind as one trajectory, with and without the lift, refusals, not_executed replay. |
| `tests/ora/agent/toolkit/tools/move_base/` | move_base's absolute and relative targets as one proposed world-frame action, its refusals in the handler's order, the travel bound the base servo shares, and its description held to the RoboCasa surface's definition. |
| `tests/ora/agent/toolkit/tools/compute/` | Fixed-data old/new equivalence, public errors and side effects. |
| `tests/ora/agent/toolkit/tools/check_task/` | Fixed-data old/new equivalence, public errors and side effects. |
| `tests/ora/agent/toolkit/tools/finish_episode/` | Fixed-data old/new equivalence, public errors and side effects. |
| `tests/ora/agent/toolkit/tools/observe/` | Fixed-data old/new equivalence, public errors and side effects. |
| `tests/ora/agent/toolkit/host/runtime/` | The tool runtime frame: the toolkit dispatch, recording and publication on a runtime, mutation compensation, the runtime package's lazy facade. |
| `tests/ora/agent/toolkit/host/`, `tests/ora/agent/toolkit/host/perception/`, `tests/ora/agent/toolkit/host/grasp/` | The host the frame runs tools under: batched `mark_point`, saved-step refusals and dual-arm image labels; `move_pose` argument shapes, refusals, observed clearance and the target-state v4 proposal of `robot_runtime_v2`; dry runs; segmentation, grounding and target localization; grasp services and the Panda and ARX X5 top-down families. |
| `tests/ora/agent/toolkit/host/reach/` | The reach checks, the IK reference and nearest reachable orientation, and grasp_plan's jaw choice, on the toolkit's reach code in the host's call shape (`tests/support/agent/reach_stacks.py`). |
| `tests/ora/agent/toolkit/host/compute/`, `tests/ora/agent/toolkit/host/finish/`, `tests/ora/agent/toolkit/host/skills/`, `tests/ora/agent/toolkit/host/shared/` | The toolkit's trusted host: the restricted `compute` evaluator (grammar, isolation, the evaluator worker); finish outcomes, the untried-candidate ledger and abort evidence; the closed grasp skill and its receipts; target and grasp refusals. |
| `tests/ora/agent/` (`test_infrastructure_classification.py`, `test_schema_refusals.py`, `test_tool_error_feedback.py`) | The `ToolError` envelope of `error_classes.py`: infrastructure classification, schema refusals and the argument rules' feedback. |
| `tests/ora/agent/tui/` | The terminal view's condensed receipt lines: which arm an action moved. |
| `tests/ora/environment/` | Backend-neutral environment code: the isolated worker RPC, execution capture, local CPU platform selection, the choice of Mesa's software EGL device on fake device lists (`test_software_egl_device.py`) and the shared `wait` action. Synthetic is exercised through the session, orchestrator and quickstart tests rather than its own directory.; `test_replay_program.py` every backend's replay program (RoboLab's steps, unavailable marker and worker on stand-ins), `test_scene_trace.py` the scene trace's format and `test_mujoco_scene_trace.py` its MuJoCo reader over stand-ins holding a compiled model's arrays, with one test on the real engine that needs `mujoco` (skipped in the project environment; run it with the LIBERO interpreter) |
| `tests/ora/environment/planners/curobo_v2/` | The cuRobo V2 planner worker over a CPU stand-in engine, and ORA's changes to the robots' V2 files (finger placement, sphere hash, the X5's finger spheres and locked jaws). |
| `tests/ora/environment/planners/timing/` | The TOPP-RA + Ruckig timing worker end to end under `ORA_TIMING_PYTHON` (skipped without it, as in CI, where `test_motion_planner.py` and `test_motion_timing.py` hold the contract, the delegation and the fallback), and the guard that keeps Ruckig's waypoint fields unset. |
| `tests/ora/environment/controllers/` | Controller contracts: control-profile capabilities, the motion transaction, the CPU CuRobo hybrid-motion runtime, the alignment acceptance band, the two-arm depth world and FK parity. |
| `tests/ora/environment/backends/libero_pro/` | The direct LIBERO/LIBERO-Pro driver: reset and termination (and the software EGL device a Mesa lane binds once and its resets record, `test_libero_egl_device.py`), camera resolution, CuRobo planning diagnostics and reach checks, gripper feedback, wait, robustness variants, practice initial states and play scenes., and which simulator, robot and TCP a scene trace reads |
| `tests/ora/environment/backends/robotwin/` | The RoboTwin backend: dual-arm motion v2 and the vendored planner and its install pin, end-effector pacing, the shared planner's frames, plans and reach rows, IK feedback, contact recovery, observation and play scenes. |
| `tests/ora/environment/backends/robocasa/`, `tests/ora/environment/backends/robolab/` | The RoboCasa365 and RoboLab single-arm executors, host-safe: the end-effector speed, public stop and jaw codes, free-space waypoints, receipt keys the policy may see, no control step or latched success past the RoboCasa horizon, the admitted task rosters, RoboCasa's `observe` result carrying the mobile base and RoboLab's carrying the Robotiq end effector, both through the real dispatcher. `test_robocasa_replay.py` and `test_robocasa_scene_trace.py`: on a stand-in kitchen, what a run records and that recording touches no random state, the program built from it with the base and scene hashes, its re-execution (a nudged object fails it, a new round rebuilds from its recorded seed), a kitchen rebuilt in the region order its run kept, a run with no controls getting a marker and no program, the row per control step with no step tap, and what is backdrop in a kitchen. `test_robocasa_egl_device.py`: the reset's record of the EGL device the kitchen renders on. `test_robolab_control_record.py` RoboLab's record of what reached the simulator on a stand-in articulation (neutral while recording, a replay's rows reach the write, a failed recording stops the record and not the run) and `test_robolab_scene_trace.py` its scene trace (the cubes and cylinders of its tables, the TCP on the flange, one row per native action). |
| `tests/ora/environment/backends/yam/` | The real YAM rig's questions to the operator through the approval broker, with no bus, camera or i2rt: enabling, each motion, a home that voids a waiting plan, closing at rest without a question, and a blocked way home that asks, holds on a decline and releases in place only when approved or at the bound; and a real worker's start setting the broker's auto-approve switch for itself only when its run asks for it, else clearing it. The simulated cell (`test_sim_cell.py`, no MuJoCo): a `robot.sim` config keeps its lock away from the real cell's, its buses count every process's handles and frames, held arms stop following once another process disables them, and the simulated operator answers each kind of question; `test_sim_cell_flow.py` (MuJoCo, mink and i2rt: the YAM interpreter; skipped elsewhere) runs `ora cell setup --ask broker --json` on a fresh simulated cell end to end and checks the calibration found the injected error, and the board's live view on it. |
| `tests/ora/orchestrator/backends/yam/` | `ora play --env yam`: the station's play RunSpec, the rig choices (`--rig-mode`, `--operator-present`, `--cell`, `--auto-approve`) bound into its runtime and the unsafe ones refused. |
| `tests/ora/environment/backends/robodojo/` | The RoboDojo backend and its ARX X5 embodiment: session and SDK boundary, the Isaac scene-trace reader on stand-ins for USD and PhysX, IK continuity, inter-arm admission and clearance, recovery, motion stall and qualification, layouts and rounds, the closed-skill default surface, perception, the play worker and the start gate. |
| `tests/ora/orchestrator/` | The orchestrator's root modules: the resource leases (`leases.py`), launch and its known failures (`launch.py`), the supervisor process entry (`supervise.py`), the workspace and shared output storage (`paths.py`), the read-only world view (`world_view.py`) and the control plane end to end. |
| `tests/ora/orchestrator/spec/` | RunSpec to RunPlan: the benchmark builders, context packs and the compactor binding, frozen compositions, model-visible identity, experiment identity, the practice partition, result tiers, free-play RunSpecs, the YAML parse cache, and a baseline's default code mode (`test_code_mode_default.py`: RoboDojo's, `null` keeping tool calls, free play and context packs getting none, no look at the host). |
| `tests/ora/orchestrator/preflight/` | Launch preflight: the gates, host storage, and resource admission under concurrent launches. |
| `tests/ora/orchestrator/episode/` | One episode: the agent core selection, the runtime end to end on fake environments (`test_runtime_v1*.py`), multi-round sessions, the native-action allowance, finish rounds, the skill-sequence executor, operator messages and the operator view, round wait accounting, stagnation records and the termination authority; a script's dry run on the real toolkit dispatch and the Supervisor's executor (`test_code_mode_toolkit_binding.py`): no action, receipt or ledger entry, the observation stays open, a tool that tries to act cannot, and the released primitives refuse; a real toolkit call's explanation rendered again from what its result recorded. |
| `tests/ora/orchestrator/evaluation/` | Evaluation and its evidence checks: decision accounting, the official verdict guard, ownership audit, artifact provenance, runtime and visual usage metrics and the observer-first context audit. |
| `tests/ora/orchestrator/reports/` | The run report. |
| `tests/ora/orchestrator/toolkit/` | Offline replay CLI, exact/alternative tools, structured missing requests, capability overrides, progress, runtime projection parity, legacy public pixels, source immutability and fake services. |
| `tests/ora/orchestrator/suite/` | The suite cells and queue, and the official-suite example composed offline. |
| `tests/ora/orchestrator/task/` | Task files, run bindings and scheduler requests (`ora task`). |
| `tests/ora/orchestrator/play/` | Play scenes. |
| `tests/ora/orchestrator/backends/` | The backend registry (`registry.py`): every `runtime.kind` loads its backend with the two required hooks, every optional hook is absent or callable as the core calls it, aliases and unknown kinds, the registry's own imports and the capabilities that replaced the core's kind lists. |
| `tests/ora/orchestrator/backends/libero_pro/` | Software-rendered (Mesa) LIBERO lanes on the cluster fixture host profile. |
| `tests/ora/orchestrator/backends/robotwin/` | The RoboTwin orchestrator backend: the official evaluation setting, seed-scan lanes and segments, and ORA's own play scene sets. |
| `tests/ora/orchestrator/backends/robodojo/` | The RoboDojo orchestrator backend and the RoboDojo files pinned in the orchestrator root: host admission, the CPU-physics lane and physics probe, the operator and installer, the worker and play kernel path sets, layered and lane qualification and revalidation, scheduled-layout evaluation and single layouts, transport and sealing, motion metrics and replay, the RoboDojo RunSpec builder, ORA's own play scenes and the launch preflight of a play plan. |
| `tests/ora/orchestrator/backends/robocasa/`, `tests/ora/orchestrator/backends/robolab/` | The RoboCasa365 and RoboLab orchestrator backends, host-safe: the runtime fields, a composed cell resolved into a plan, the preflight probes and the sparse-motion qualification script on a scripted worker session. |
| `tests/ora/app/` | The `ora` command (`ora/app/`): one parser and its command table, grouped help and the CLI reference (`test_main.py`), every subcommand's and backend tool's `--help` (`test_help.py`), the import budget, and one module per command module: `ora local`/`setup`/`run` on one GPU, `ora interactive` and its JSON-lines backend, `ora watch`, `ora web`/`replay`, `ora providers`/`services`, `ora-remote`, the quickstart and `baselines --explain`. |
| `tests/ora/robot_client/` | The free-play robot client (`ora.robot_client`) and `ora-robot` in process against a scripted socket: its protocol (the same as the Pi-side server's), errors, image choice, origin and exit statuses. |
| `tests/ora/cell/`, `tests/ora/cell/operator/`, `tests/ora/cell/depth/`, `tests/ora/cell/cameras/` | The real-cell kit (`ora/cell/`) on an in-memory robot (`tests/support/cell/fake.py`), never a bus, camera or the host's lock: cell config validation and `check_against`, calibration loading (legacy files included) and saving, the cell lock across processes, bus receivers and arm-driving processes from fake `/proc` trees, the pure geometry, a plain checkerboard's corners named by its colours, the operator verbs (`stop`, the `Holder`, `home` holding rather than dropping), the approval broker (answers by id, across processes, stale askers and timeouts) and the board's answers to it, the live view's flying-pixel filter and the depth fusion. `ora cell --json` is in `tests/ora/app/test_cell.py`. Every test here runs with the cell lock in its own `tmp_path` (`tests/ora/cell/conftest.py`). |
| `tests/viewer/` | The read-only Viewer: run discovery, controller replay and caches, process timelines and videos, pricing and visual-fidelity labels; `tests/viewer/dashboard/` the run dashboard over synthetic RoboDojo-, RoboTwin-, LIBERO-Pro-, RoboCasa365- and RoboLab-like runs (cohorts and conditions, run detail, path safety, read-only serving; the 3D replay's bundle for every backend: clock, commands, observations, the aloha-agilex description, a Synthetic run's refusal; a run's scene trace: the bundle, the objects layer's reason without one, the robot's links and TCP where no URDF poses the robot, `api/scene-trace` and its path safety; a RoboDojo trace on the native-action clock, its rounds and spans for a run without ground truth; RoboCasa365 and RoboLab traces on their receipts' clocks, and the note of a run sealed without its controls). |
| `tests/repository/` | Repository guards: `policy.lock`, path-only asset references, packaging, the user installer (`test_install_script.py`: `scripts/install.sh` against a local repository with stub uv and npm: the checkout at the requested commit, the sync and `npm ci`, the `ora` command it writes, an update on rerun, and its refusals), the module-size ratchet, undefined globals, dead tool-runtime state, retired import paths and the target structure (`test_structure.py`: the import rules, the test tree and the package READMEs of [`docs/developer/repository-structure.md`](../docs/developer/repository-structure.md)). |
| `tests/support/` | Shared helpers, not tests, in one package per area: `agent/`, `cell/`, `robodojo/`, `robotwin/`, `libero/` and `orchestrator/` (for example `cell/fake.py`'s in-memory `CellAdapter` and cell config, `robodojo/backend.py`'s fake RoboDojo plant and SDK boundary, `orchestrator/benchmark_specs.py`'s spec builders, `orchestrator/runtime_v1.py`'s fake LIBERO environments, `orchestrator/run_specs.py`'s Native RunSpecs, `orchestrator/sparse_motion.py`'s scripted qualification session, `agent/surface_copies.py`'s test-local ToolSurface copies), `context_pack/` and `agent_core/`: the agent-core drivers, Synthetic episodes on the Pi core and their goldens (`golden.py`), an operator stop during a hanging request, a tool call or an action's execution (`stops.py`), a free-play session driven through its inbox (`play_session.py`: its operator reads only complete event lines, a failing operator quits the session, and a session past its deadline is quit with every thread's stack), sealed-run extraction and replay, and the provider and Context helpers the TypeScript tests call. |

`tests/data/` holds the fixed inputs the tests read:

- `hosts/`: the `test-host`, `test-cluster` and `test-arm` execution host
  profiles;
- `prompts/` and `robodojo/*.md`: small prompt and manual layers the runtime
  tests compose instead of the shipped bundles;
- `delivery_trim/`: per-backend observation and delivered-text goldens for the
  delivery trim;
- `robodojo/*-receipts.json`: recorded RoboDojo receipts the per-arm
  translation and retract-diagnostic projections are checked against;
- `robotwin/grasp-execute-precommit-probe.json`: a recorded RoboTwin `grasp_plan` candidate, the
  `grasp_execute` call that chose it and the motion probe its precommit check sent, which both stacks
  must read alike once the worker's own probe answers it;
- `code_mode/script-answers.json`: what code-mode scripts read on recorded
  runs of the five benchmarks (rebuilt with ORA's receipt projection, pruned),
  which the `robot` library's typed results are checked against;
- `code_mode/motion-answers.json`, `motion-explanations.json`: what scripts read from `move_pose`, `move_to`,
  `rotate_wrist`, `return_home` and `recover_pose` on recorded RoboDojo toolkit-v5 runs (stored results with ORA's
  degree delivery and causal filter, pruned), and the golden `explain` rendering of each;
- `code_mode/motion-predictions.json`: for each move of those runs that reached its target, the end pose the plan
  expected (the value a dry run returns as `predicted_pose`) beside the pose the receipt measured;
- `code_mode/perception-answers.json`, `perception-explanations.json`: what
  scripts read from `vision_ground`, `vision_segment`, `geometry_query` and
  `grasp_plan` on recorded runs (one call per kind of outcome, pruned), and the
  golden `explain` rendering of each;
- `default-expectations.json`: the expected default baseline and per-backend
  instances the default-activation checks compare against;
- `backend-parity.json`: one row per capability and tool a maintained default
  declares, with its class and per-backend exceptions, for the cross-backend
  parity check (`tests/ora/agent/composition/surface/test_backend_parity.py`);
- `identity-registry-fixture.json`: a synthetic experiment lineage for the
  run-identity tests;
- `fk-parity/<robot>.json`: a robot's kinematic chain and eight or more joint samples with
  the tool link as the simulator recorded it and as the planner computes it
  (`tests/ora/environment/controllers/test_fk_parity.py`); a robot joins the shared
  planner by adding one;
- `motion-pace-paths.json`: three recorded LIBERO global-planner joint paths on
  their pacing grid, which the shared end-effector timing and the RoboDojo
  toolkit's must time identically
  (`tests/ora/environment/controllers/test_motion_pace_parity.py`);
- `context_pack/`: a synthetic context pack and one recorded history, for the
  context-pack tests of `tests/ora/orchestrator/spec/` and `tests/agent_core/`;
- `agent_core/golden/`: what the retired Python Native core sent and recorded
  on the scripted agent-core scenarios, which the Pi core must reproduce;
  `agent_core/discard-trace/`: frozen agent traces of discarded actions for the
  trace verifier;
- `context_differential/`: the Context fixtures the TypeScript pipeline
  replays (see [The agent core](#the-agent-core)).

Not here: anything that starts a simulator, a GPU process, a provider or a
model service (that is qualification); files written outside `tmp_path` or the
temporary workspace the root `conftest.py` binds; helper modules outside
`tests/support/`; and test-only code in `ora/` (it belongs in
`tests/support/`).

To add a test, put the module in the directory that mirrors the code it
protects (`tests/ora/<package path>/`, adding the directory and its
`__init__.py` if it is new, and its row above) and open it with a docstring
that states the contract it protects in one or two sentences. An agent-core
test goes in `tests/agent_core/`, which marks it. Prefer a behavioural
assertion over a source-layout one, and the smallest check that exercises the
change.

## Dependency direction

A test module imports `ora`, `viewer`, `examples` and `tests.support`, never
another test module or a `conftest.py` (rule L8 of
`tests/repository/test_structure.py`). The suite is collected from `tests/`
with the repository root on `sys.path` (`[tool.pytest.ini_options]` in
`pyproject.toml`), so a helper is imported as `tests.support.<area>.<topic>`.

`tests/support/` holds plain helpers and constants and defines no fixture. A
helper or constant that more than one module needs goes in
`tests/support/<area>/<topic>.py`, named after the thing it holds. A fixture
that more than one module needs is registered in the narrowest `conftest.py`
above them all, and a module whose every test needs a fixture that is not
autouse names it in `pytestmark = pytest.mark.usefixtures(...)`.
`test_structure.py` fails on a cross-test import and on a fixture defined in
`tests/support/`.

The conftests and what they provide:

- `tests/conftest.py`, before any `ora` module is imported: points
  `ORA_WORKSPACE_ROOT` and `ORA_CONFIG_HOME` at temporary directories, binds
  the fixture execution hosts through `ORA_HOSTS_DIR=tests/data/hosts`, runs
  BLAS and OpenMP with one thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`) so parallel workers do not oversubscribe the host, and
  rewrites the asserts of `tests/support/`. Its autouse fixtures restore
  `os.environ` after every test (code under test sets variables such as
  `ORA_BASE_URL` or `MUJOCO_GL` directly) and stub the host-storage probes (opt
  out with the `host_storage_real` marker). It also provides
  `robodojo_activation`, `ready_model_services` and
  `fake_sdk_candidate_boundary`.
- `tests/agent_core/conftest.py`: marks every test there `agent_core` and
  provides the agent-core preconditions (`node_runtime`, `pi_core`,
  `clean_source_tree`).
- `tests/ora/orchestrator/conftest.py`: `workspace`, a private workspace for
  composed RunSpecs; `tests/ora/orchestrator/backends/conftest.py`:
  `simulator_host`, a host profile with RoboCasa365 and RoboLab slots.
- `tests/ora/environment/backends/robodojo/conftest.py`: `sparse`, a
  sparse-motion session on the fake plant; it and
  `tests/ora/orchestrator/backends/robodojo/conftest.py` bind the named
  RoboDojo runtime (see [What a run needs](#what-a-run-needs)).
- `tests/ora/agent/toolkit/host/reach/conftest.py`: `dojo`, a fake RoboDojo
  session for the reach check.
- `tests/ora/cell/conftest.py`: `cell_lock_path` (autouse: the cell lock in
  the test's `tmp_path`), `quiet_buses`, `adapter` (a `FakeAdapter`) and
  `cell`, a cell on it with its rest pose recorded.

## Entry points

### Markers and lanes

Lanes are selected by marker, not by path. `tests/agent_core/conftest.py`
marks every test there `agent_core` (registered in `pyproject.toml`);
`host_storage_real` (registered by the root `conftest.py`) lets a test run
`ora.orchestrator.preflight.host_storage` against its own fake probes.

| Lane | Selection | CI job | `scripts/dev/check` |
|---|---|---|---|
| The suite | `-m "not agent_core"`, in parallel, without the no-write check | `tests` | lane A |
| The agent core | `-m agent_core`, in one process | `agent-core` | lane B |
| TypeScript agent core, ORA's web page | `node --test ora-pi/test/*.test.ts ora-pi/test/*.test.mjs`; typecheck, test and build in `ora-pi/web` | `agent-core` | lane C |
| Repository checks | `./scripts/ora lock --check` and whitespace in local changes | `tests` (the lock) | lane D |
| No-write check | `tests/ora/app/test_baselines.py::test_the_command_writes_no_files`, alone | `tests`, last | last |

Lanes A and B and both CI jobs set `ORA_TEST_REQUIRE_AGENT_CORE=1`, so a test
that needs the Pi core fails instead of skipping when the core cannot start.
CI and `scripts/dev/check` run the suite under pytest-xdist with a 300-second
per-test timeout (pytest-timeout), so a module must not depend on another
module's side effects or on running alone. The one exception is
`tests/ora/app/test_baselines.py::test_the_command_writes_no_files`: it
compares the checkout's ignored files before and after, so both deselect it
from the parallel run and run it by itself at the end.

### Commands

```bash
scripts/dev/check                                           # what CI runs, in four parallel lanes
scripts/dev/check --quick                                   # lanes A and D and the no-write check
uv run --locked pytest -q -n auto -m "not agent_core"       # the suite in parallel (pytest-xdist, a dev extra)
uv run --locked pytest -q -m agent_core                     # the agent-core tests
uv run --locked pytest -q                                   # everything, in one process
uv run --locked pytest -q tests/ora/environment/backends/robodojo/        # one package
uv run --locked pytest -q tests/ora/agent/toolkit/host/reach/test_reach_check.py # one module
uv run --locked node --test ora-pi/test/*.test.ts ora-pi/test/*.test.mjs  # the TypeScript agent-core tests
npm test --prefix ora-pi/web                                # ORA's web page and its server (vitest; npm ci there first)
```

`scripts/dev/check --workers N` sets lane A's pytest-xdist workers (default:
the CPUs, at most 8) and `--logs DIR` keeps each step's log outside the
checkout; `scripts/dev/check --help` lists the rest. CI runs
`./scripts/ora lock --check` before the suite; a stale `policy.lock` also
fails `tests/repository/test_policy_lock.py`.

### The agent core

The agent core (provider loop, request assembly, Context management and
agent-side evidence) runs on Pi (`ora-pi/`). The Python Native core it was
ported from left ORA (it remains in the robot-harness repository); what that
core sent and recorded on the scripted scenarios of `tests/agent_core/`, when
the Pi core matched it byte for byte, is frozen as goldens in
`tests/data/agent_core/golden/` (written and read by
`tests/support/agent_core/golden.py`; the directory's README says how a
reviewed change re-records one from Pi). Each run names its core explicitly
(`ORA_AGENT_CORE=pi`) and checks the core that ran: only the Pi core writes
`native/pi-core.log`.

`tests/agent_core/test_quickstart_cores.py` drives the CLI, whose preflight
admits only a clean source tree; with local changes it skips with that reason
(and fails under `ORA_TEST_REQUIRE_AGENT_CORE=1`), so commit before running
lane B. `tests/ora/app/test_quickstart.py` drives the same CLI under a Git
environment that shows the working tree committed as is
(`tests/support/orchestrator/source_snapshot.py`), so local changes do not stop
it: it runs the checkout's own code, changes included.

`ora-pi/test/*.test.ts` (`node --test`) compare the TypeScript provider and
Context ports with the Python implementations through the helpers in
`tests/support/agent_core/`. They run those helpers under `ORA_TEST_PYTHON`,
else the active virtual environment (`uv run --locked node --test ...`
activates the project's), else `uv run --locked python`, and keep scratch files
under `ORA_TEST_SCRATCH`, else the system temporary directory.
`context_differential` reads the fixtures in `tests/data/context_differential/`,
recorded from the Python Context pipeline before ORA removed it with the Native
core: the scenarios built from the L1-L6, stride, cost and round-slot tests,
and the pipelines the Native loop and the pipeline's own tests built
(`captured_*`). Their recorder needed that pipeline, so they are not
re-recorded; `ORA_CONTEXT_FIXTURES=<dir>` reads another recorded directory
instead.

A sealed run's Context can also be replayed call by call.
`python -m tests.support.agent_core.context_replay.extract RUN_DIR --out DIR`
writes what the Context manager received at every model call (the history
appended since the previous call, tool images registered, compactor calls with
their recorded responses, the Provider usage) and what the run recorded;
`node ora-pi/test/support/context_replay.ts DIR --out RESULT.json` builds the
pipeline and request assembler with the core's own construction code
(`ora-pi/src/core/context_wiring.ts`), steps through the calls and reports per
call the canonical and wire request sha256, the pipeline state, the manifest
and its transitions, compared with the run. Run on the run's own commit, it
reproduces the recorded requests byte for byte; `--scheme`, `--management` and
`--policy` substitute the Context settings, `--inflate` and `--round-split`
make deterministic variants (the header of the runner explains them).

## What a run needs

Nothing in the suite needs a simulator or a GPU. What it does need from its
environment, as measured on 2026-10-22 when the tree was restructured:

- **A Git checkout.** About 260 tests run `git` against the checkout, nearly
  all through RunPlan resolution, which records the source commit and whether
  the tree is dirty (`ora/orchestrator/spec/run_spec.py`), and through
  evaluation, `policy.lock` (tracked files only) and the RoboDojo kernel
  manifests (`git show` of committed files). From an export without `.git`
  (a tarball or sdist) about 220 of them fail, so run the suite from a clone.
  A dirty tree changes one outcome: the CLI quickstart of
  `tests/agent_core/test_quickstart_cores.py` (see
  [The agent core](#the-agent-core)).
- **Node 22.19 or newer and `npm ci` in `ora-pi`** for every test that starts
  the Pi core: the Pi-core cases of `tests/agent_core/` and about 40 more
  outside it, the episodes that run a policy session (multi-round sessions,
  finish rounds, operator control, the official verdict guard, the SFT export)
  and the doctor and default-core quickstart cases of
  `tests/ora/app/test_quickstart.py`. Without them those cases skip with the
  reason (`tests/support/agent_core/preconditions.py`);
  `ORA_TEST_REQUIRE_AGENT_CORE=1` turns the skip into a failure. CI sets it in
  both its jobs, `scripts/dev/check` in lanes A and B. Product code finds node on
  `PATH`, then under `~/.local/bin` and `~/.nvm`; with node on `PATH` no test
  depends on `HOME`.
- **The pinned RoboDojo runtime**, for 137 layout, X5 IK and CPU-physics cases
  under `tests/ora/environment/backends/robodojo/` and
  `tests/ora/orchestrator/backends/robodojo/`: its `source/RoboDojo` and
  `assets/Assets` under `ORA_TEST_ROBODOJO_RUNTIME`, else `~/robodojo-runtime`
  (`ora setup --env robodojo`'s default; a set `ROBODOJO_ROOT` or
  `ROBODOJO_ASSET_ROOT` still overrides the source or Assets root). Every task,
  swap_T included, reads that one Assets checkout, as the runtime does; an
  installation of the earlier asset pin `a14409d7` holds swap_T's layouts as
  LFS objects upstream no longer serves, so those cases skip there. Without
  it those cases skip with the reason, so which of them run depends on the
  developer's `HOME`. With the variable set, the tests also point product code
  at that runtime (the `conftest.py` of each directory). A layout that is
  still a Git LFS pointer skips its case under either runtime, the reason
  naming each pointer file; the runtime's owner pulls it, never the suite.
- **Real time.** About 70 tests wait on real time, from 0.1 to 10 seconds:
  recovery deadlines of a real provider worker
  (`tests/ora/agent/providers/test_recovery_deadline.py`), provider faults
  and crash resume of the Pi core (`tests/agent_core/`), worker start and
  close deadlines (`tests/ora/orchestrator/backends/robodojo/test_robodojo_runtime.py`
  and `test_robodojo_motion_replay.py`, the runtime tests in
  `tests/ora/orchestrator/episode/`), seed watchdogs and polling loops.
  Multi-round sessions also run under a wall-time budget
  (`tests/ora/orchestrator/episode/test_multi_round_session.py`,
  `test_round_wait_accounting.py`, `test_policy_finish_rounds.py`). On a
  loaded host or a stalling file system these are the tests that fail first
  (a timeout, or a session ending in `wall_time_budget`); rerun them on a
  quiet host before treating such a failure as a regression.
- **`ffmpeg` and `ffprobe` on `PATH`** for the Viewer video export tests (CI
  installs them; on Ubuntu, `sudo apt-get install ffmpeg`).
- **`tar` and `zstd`** for `tests/ora/agent/evidence/test_request_archive.py`,
  which skips without them.
- **The sealed acceptance runs** of 2026-10-13, which live outside the
  repository, for the opt-in replays: `ORA_TEST_SEALED_LIBERO`,
  `ORA_TEST_SEALED_ROBOTWIN` and `ORA_TEST_SEALED_ROBODOJO` name the sealed
  run or a replay fixture extracted from it (`python -m
  tests.support.agent_core.replay.extract --source RUN --out DIR`; extraction
  needs `zstd`), and `ORA_TEST_SEALED_CACHE` keeps extractions between runs.
  Unset, each such check is reported as skipped with the reason. They cover
  `tests/agent_core/test_sealed_replay.py`, the sealed-run checks of
  `provider_accumulate` and `provider_anthropic` and the LIBERO replay Context
  fixtures.

## Documentation

- [Testing](../docs/developer/testing.md): choosing a check, splitting a test
  family, timing and parallel runs, ratchets, and the simulator-only code the
  suite does not reach.
- [Repository structure](../docs/developer/repository-structure.md), section 8:
  the target test tree that `tests/repository/test_structure.py` holds.
- [Qualification authoring](../docs/developer/authoring/qualification.md): the
  live simulator, provider and GPU checks that are not part of this suite.
- [Contributing](../CONTRIBUTING.md): what to run before a pull request;
  [`scripts/README.md`](../scripts/README.md): `scripts/dev/check`.

- `tests/ora/agent/pi/`: trusted tool-execution RPC metadata and receipt projection; a code-mode script's
  legacy action (a committed skill sequence's staged receipt, post-close views and uncounted time, a refusal
  that releases its skill ledger, a failure after dispatch that is not reported as not started); a script's
  reused perception results (the same call on the same observation, a new observation, the call limit); an extended
  script call's `call_id`, the explanation a failed call carries and the `explain` op (`test_code_mode_explain.py`); a script's
  queries and its query allowance (`test_code_mode_query.py`): charged, refused past the count and the seconds, recorded
  as a nested request but never a tool call, served to a command's programs in the sandbox, absent from a `base` contract;
  what a script asked of the library on the record (`test_code_mode_record.py`): the requests as nested calls numbered
  among the calls, each call's explain inputs, `explain` from the record past the in-memory window and after a restart, the
  trace verifying and every explanation rendered again from it, and a `base` contract recording what it recorded; a
  script's `dry_run=True` (`test_code_mode_dry_run.py`): the flag reaches the tool out of band, the call is a read-only nested
  call charged to the script's queries, the answer is the plan in degrees with its explanation, and a refused dry run, a
  `base` contract and a tool without a dry run leave the argument alone.
