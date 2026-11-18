# ora/environment/backends/robodojo

## Responsibility

The RoboDojo simulator adapter (environment id `robodojo`): the two-arm ARX X5
in Isaac Sim 5.1 on the pinned RoboDojo checkouts. Everything here runs inside
the isolated RoboDojo interpreter (`ROBODOJO_PYTHON`, the plan's
`runtime.python_env`), started by `ora/environment/rpc.py` as the environment
worker, except the host-side installer under `installation/`. It boots Isaac,
builds the official task at a saved layout, publishes public RGB-D
observations, executes the joint trajectories the toolkit plans with measured
receipts, and reads RoboDojo's task verdict as the official one. A second
worker backend, `robodojo_play`, runs ORA's own free-play scenes.

| Module | Holds |
|---|---|
| `session.py`, `bootstrap.py`, `bridge.py`, `start_gate.py`, `graphics_compat.py` | `RobodojoSession`, the worker session `definitions.py` names; Isaac application boot and exact environment construction; the policy-neutral EvalEnv lifecycle; the node-level Vulkan start gate; the process-local Vulkan allocation-limit compatibility layer |
| `reset_diagnostics.py` | Evidence-recorded reset timer control and optional scoped fatal traceback; [operator instructions](../../../../docs/reference/robodojo-reset-diagnostics.md) |
| `observation.py`, `camera_capture.py`, `feedback.py`, `execution_capture.py`, `robot_asset_geometry.py` | Calibration and the public observation, exact RGB-D from the tiled annotators, controller feedback for observations and receipts, original pixels after existing physics steps, the loaded finger meshes and joint frames |
| `sparse_motion.py`, `ik.py`, `motion_adapter.py`, `recovery.py`, `interaction_control.py`, `sequence_admission.py`, `wait.py` | Sparse motion v3: the receipt every executed action shares, the open/close `set_gripper` and the read-only `probe_motion` (`move_pose`, `move_free_space` and `recover_motion` are refused before any write); bounded SDK IK selection (the kinematics query's `canonical_first` and runtime action v3); the read-only `align_plan` bridge; the recovery route a runtime may name; support-arm evidence; admission of a trusted sequence's children; the shared `wait` |
| `trajectory.py`, `gripper_ramp.py`, `native_sequence.py`, `tick_sequence.py` | The per-physics-step executors: `follow_joint_trajectory` (knot interpolation with velocity feed-forward, tracking and progress guards evaluated per physics step, hold-at-stop, explicit settle) and the rate- and lead-limited `set_gripper` (an optional hold lead once the first finger stops, an optional joint-velocity first reading for its arm settle); their journaled tick runner (journal and trace v2: requested, resolved and applied position and velocity targets per step) and the bridge's tick with caller-supplied targets. Restore sends a v2 tick's recorded requested steps again (`control_replay`) |
| `kinematics_query.py`, `ik_sampler.py` | The read-only kinematics query (`ora.kinematics-request.v1`: IK chains seeded from the previous solution with explicit CuRobo seeds, FK, lockstep sphere screens answered by the SDK's own planner; `plan_path`, `pace_path` and `contact_path` by the shared base); the CuRobo seed generators as episode state (`exec.ik_sampler_state`) |
| `shared_motion.py` | All that RoboDojo adds to the shared motion base: its planner instance on the SDK's robot file (the SDK's planner cannot update a world), the FK parity check at session start, the depth-world input of the last observation with the other arm in it, and the three answers with their evidence (`motion-planner/`) |
| `reach_check.py`, `reachability.py` | The read-only reach check on the shared planner's IK; link6 kinematics from the SDK's robot model and the batch IK answer of `align_plan` and `probe_motion` (`no_planner` when the runtime names none) |
| `round_lifecycle.py`, `layout_rounds.py`, `common.py` | Measured reset and round isolation, the per-round layout schedule of a persistent session, shared measurement helpers |
| `motion_checks.py`, `motion_qualification.py` | The scripted sparse-motion qualification cases and their judgements |
| `control_replay.py`, `physics_scene_probe.py` | Control replay of a sealed run and the physics probe of one scene, both inside the RoboDojo interpreter |
| `isaac_scene.py`, `scene_trace.py` | The scene trace's exporter (`ora/environment/scene_trace.py`), run by `control_replay --scene-trace`: `isaac_scene.py` reads any Isaac stage's bodies (rigid prims through one PhysX view, the meshes the renderer draws) and names nothing of RoboDojo's, and RoboLab imports it; `scene_trace.py` is RoboDojo's use of it. Read-only, outside the worker kernel |
| `play_scenes/` | The `robodojo_play` worker: Kit boot for a play scene, the reward-free TaskEnv lifecycle, the operator's third-person view, the play session and the play-scene audit worker; the scene YAMLs and each set's `index.yml` under `scenes/<set>/` |
| `installation/` | The host-side installer of the pinned runtime (`setup_runtime.py`, standard library only), the SDK wheel rebuild, the Vulkan profile patch and compatibility manifest, and `INSTALL.md` |

## What belongs here, and what does not

Here: code that imports Isaac Sim or the RoboDojo SDK, or runs in its
interpreter, and the installer of that interpreter. Not here: the host-safe
identities, layouts, physics devices and play-scene registry
(`ora/environment/benchmarks/robodojo*.py`), the ARX X5 control contract
(`ora/environment/controllers/robodojo_arx.py`, `ora/contracts/robodojo_arx.py`),
the play worker's host-side factory (`ora/environment/robodojo_play_factory.py`),
and the host side: RunSpec builder, preflight, card / task / lane
qualification, supervision, evaluation and the motion journals
(`ora/orchestrator/backends/robodojo/`, plus the seven pinned
`ora/orchestrator/robodojo_*.py` modules).

**Pinned: the whole directory.** Its modules are hashed by path and bytes:
28 of them are members of the RoboDojo worker kernel (`WORKER_KERNEL_PATHS` in
`ora/orchestrator/robodojo_kernel.py`; every module in this directory's root
except `control_replay.py`, `physics_scene_probe.py` and the scene trace's
`isaac_scene.py` and `scene_trace.py`), bound by every card
and task qualification record and every lane revalidation; the six
`play_scenes/*.py` modules are members of the play kernel (`PLAY_KERNEL_PATHS`
in `ora/orchestrator/robodojo_play_kernel.py`) and each scene's YAML and index
entry are bound per scene, so every play-scene audit binds them;
`installation/setup_runtime.py` writes `scripts/setup-robodojo` into the
graphics-compatibility receipt it records. Do not rename, move or edit a file
here, comments included, as part of an unrelated change: any byte change
re-runs the RoboDojo lane qualification (card, task, scene smoke, revalidate)
or the play audit on a healthy Isaac card. Comments in these files that still
name pre-restructure module paths stay until such a re-sign.

## Dependency direction

Imports `ora.environment` (core, `benchmarks`, `controllers`) and
`ora.contracts`, plus Isaac Sim and the SDK, never the agent or the
orchestrator. `ora/environment/backends/robolab/` reuses `graphics_compat.py`,
the crash-reporting and renderer helpers of `bridge.py` and `isaac_scene.py`
(both run Isaac Sim 5.1). The environment core reaches this package only through
`worker.py`, `replay_worker.py` and the `worker_loader` string; the package
`__init__.py` imports `session.py` eagerly, an allowed exception to rule L9
because it is a kernel member.

## Entry points

- `session:RobodojoSession` and `play_scenes.session:RobodojoPlaySession`:
  loaded by `ora/environment/worker.py` (backends `robodojo` and
  `robodojo_play`).
- `python -m ora.environment.backends.robodojo.control_replay`: `ora qualify
  robodojo motion-replay`, and `ora replay RUN --scene` with `--scene-trace`.
- `python -m ora.environment.backends.robodojo.physics_scene_probe`: `ora
  qualify robodojo physics-probe`.
- `python -m ora.environment.backends.robodojo.play_scenes.audit`: the worker
  of `ora play audit --env robodojo`.
- By path: `graphics_compat.py` (the installer and `ora doctor --env
  robodojo`), `installation/setup_runtime.py` (`scripts/setup-robodojo`,
  `ora setup --env robodojo`) and `installation/rebuild_sdk_wheel.py` (the
  installation guide, under the Isaac interpreter).
- `motion_qualification.py` keeps the numerical gate suite and record schema.
  Its legacy standalone host `main` refuses before worker boot; the state-owner
  host migration is deferred to architecture v1 step 2. Supported qualification
  entry points are listed by `ora qualify robodojo --help`.

## Documentation

- [RoboDojo](../../../../docs/environments/robodojo.md), the
  [quick run](../../../../docs/getting-started/robodojo.md) and the
  [installation reference](installation/INSTALL.md)
- [RoboDojo motion qualification](../../../../docs/reference/robodojo-motion-qualification.md)
- [Free play](../../../../docs/usage/play.md) and the
  [play-episode reference](../../../../docs/reference/configuration-and-evidence.md#play-episodes)
- [Host-side backend](../../../orchestrator/backends/README.md) and the
  [orchestrator's pinned files](../../../orchestrator/README.md)


### Toolkit live diagnostic

`toolkit_worker.py` is a separate private entry for `ora toolkit replay --live`.
`toolkit_restore.py` reuses control_replay and the unchanged RobodojoSession.
`toolkit_render.py` replays recorded render synchronizations at their original
physics clocks, including execution captures between native ticks. It uses the
existing capture callback without advancing physics. Each restored round is
also rendered once at its start when the history does not render there (an
unavailable history, e.g. a source run without `native/render-synchronization.jsonl`):
the first render after reset fixes the
wrist cameras' mount on link6, so it must not first happen at a moved boundary
(`round_start_renders` in `render-history-replay.json`). The host verifies the
selected control prefix and physical state exactly before regular RPC is
authorized; fresh sensor differences are nonblocking diagnostics. Tools consume
the original published sensor bytes. An additional launch flag gates action execution.
It is not the policy worker or part of its qualification hash. Diagnostic
source hashes are recorded separately. No agent/orchestrator imports enter
this worker, and private pose checks are verifier-only.

`execution_request.py` borrows command/recovery state from each host request,
returns updates and clears all borrowed fields on exit. Recovery receives
explicit joints; its existing safety route is unchanged. The regular and
toolkit diagnostic transports both use it. Standalone replay/probe/audit jobs
are each one scoped diagnostic request. See [state](../../../../docs/reference/toolkit-state.md).


For the phase-one `--native-action` snapshot selector, the same diagnostic
transport restores a truncated tick prefix. `toolkit_native_state.py` restores
request bookkeeping from the pre-action owner state plus tick directives and
current recovery anchors; it never writes robot or object poses. Source version
differences are warnings on this new route; exact physical checks remain on the
host. Fresh sensors become the detached snapshot observation, and the worker
closes before scripts run. No new physical tool execution is granted. The older
plain `--at --live` route retains its original published-pixel semantics.

Replay-only source compatibility is documented in
[toolkit-protocol](../../../../docs/reference/toolkit-protocol.md).
`source_identity.py` keeps production source-pin admission strict and emits
recorded/current warnings only for diagnostic configs; it is bound in both
worker kernel manifests. Physical/control replay remains independently checked.
