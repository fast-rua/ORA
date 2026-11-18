# ora/environment/backends/robotwin

## Responsibility

The RoboTwin 2.0 simulator adapter (environment id `robotwin`): the dual-arm
ALOHA-AgileX in SAPIEN on the pinned RoboTwin checkout (`ROBOTWIN_ROOT`).
Everything here runs inside the isolated RoboTwin interpreter
(`ROBOTWIN_PYTHON`, the plan's `runtime.python_env`), started by
`ora/environment/rpc.py` as the environment worker. It loads the official task
at an admitted seed, publishes public RGB-D observations, plans and executes
admitted motion with the vendored planner chain, and reads RoboTwin's own
`check_success` as the official verdict.

| Module | Holds |
|---|---|
| `session.py`, `common.py`, `round_lifecycle.py` | `RoboTwinLiveSession`, the worker session `definitions.py` names (direct and model-free over the pinned RoboTwin API, with the evaluator-state helper), its shared constants and errors, episode binding and round restart |
| `action_plan.py` | The read-only answer to a dry run: the sparse route, planned, validated and timed, stopped before any controller write |
| `observation.py`, `gripper_state.py`, `kinematics.py`, `robot_asset_geometry.py` | Public RGB-D, calibration and proprioception; gripper command and encoder evidence; public and planner pose frames with the Pinocchio arm models; the loaded robot URDF's collision geometry |
| `sparse_admission.py`, `sparse_planning.py`, `sparse_execution.py`, `sparse_settlement.py`, `wait.py` | Sparse motion v3: admission and routing, planning with blocked-motion recovery, execution of validated plans and gripper targets, settling and the official terminal latch, the shared `wait` |
| `pacing.py` | End-effector pacing of sparse moves (`ora/environment/controllers/motion_pace.py`): every validated global plan is timed at the public TCP and wrist-checked before any controller write, and a requested speed spreads each local servo step over physics steps; `round_lifecycle.py` binds the run's timing worker at the first round (`motion_timing.bind_preflight`) |
| `motion_adapter.py`, `motion_transaction.py`, `dense_motion.py`, `local_cartesian.py`, `local_cartesian_tracking.py`, `planning.py` | The plant adapter for the backend-neutral motion transaction, dense execution, the bounded local Cartesian servo, plan validation helpers |
| `probe.py`, `probe_feedback.py` | Read-only reachability probes and, from the vendored planner, IK candidate counts |
| `shared_motion.py` | RoboTwin's place on the shared motion base: the preflight's planner bound once per process, the frames read off the vendored transform, forward-kinematics parity on both arms, one public-depth world per physical state and arm with the other arm's spheres in it, and the plans and IK `motion_adapter.py` answers `move_pose`, the reach check and `align_plan` from |
| `reach_inter_arm.py` | The refusal of the retired other-arm reach screen, which `session.py` still imports |
| `control_profile.py`, `asset_manifest.py` | Control-profile source checks and the capabilities derived from it; hash-bound manifests of the scene assets a qualified episode loads |
| `execution_capture.py`, `operator_capture.py` | Original-execution media; the operator-only world and execution video |
| `scene_trace.py` | The scene trace's exporter (`ora/environment/scene_trace.py`): every actor's and link's visual mesh and pose read from the SAPIEN scene during a replay, on the same step tap as the execution capture; read-only, worker-only |
| `qualification.py` | The trusted expert / oracle seed qualification runner behind `ora qualify robotwin` |
| `play_scenes/`, `play_settle.py`, `play_scene_audit.py` | ORA's own free-play scene sets (`tabletop-v1`, `household-v1`; worker-only `Base_Task` subclasses), their post-action settle and the worker side of their audit |
| `vendored/`, `vendored_install.py` | The planner and robot modules ORA installs onto a vanilla RoboTwin checkout (copied there, never imported from here; MIT attribution beside them; the expert, the seed scans and a session with no bound planner plan on them) and their installer |

## What belongs here, and what does not

Here: code that imports RoboTwin, SAPIEN or the planner stack, or runs in that
interpreter. Not here: the host-safe task roster, revision and instructions
(`ora/environment/benchmarks/robotwin.py`, `robotwin_instructions.py`) and
the play-scene registry (`robotwin_play_scenes.py`); the seed-scan scheduler,
seed manifests and their admission, preflight, play scenes and the play-audit
command (`ora/orchestrator/backends/robotwin/`).

**Pinned files.** `qualification.py`, `session.py` and `common.py` are inputs
of the RoboTwin seed scans' `producer_dependency_hash` and `checker_hash`
(`source_dependencies` in `ora/orchestrator/backends/robotwin/evaluation_seeds.py`,
with `ora/environment/benchmarks/robotwin.py`, `uv.lock` and `pyproject.toml`).
A changed byte in any of them stops new scans from merging with older
segments and stops a running scan from resuming; do not rename them or edit
them as part of an unrelated change. `vendored/` is content-addressed by the
`source_binding` of control profile v10
(`ora/policy/embodiments/aloha-agilex/control-profile-v10.json`): after
editing it, run `./scripts/dev/refresh-vendored-pins`.

## Dependency direction

Imports `ora.environment` (core, `benchmarks`, `controllers`) and
`ora.contracts`, plus RoboTwin and its stack, never another backend, the agent
or the orchestrator. The environment core reaches it only through
`worker.py`, `replay_worker.py` and the `worker_loader` string.

## Entry points

- `session:RoboTwinLiveSession`: loaded by `ora/environment/worker.py`.
- `python -m ora.environment.backends.robotwin.qualification` (console
  `robotwin-qualify`): one seed-scan lane, started by `ora qualify robotwin
  qualify`.
- `python -m ora.environment.backends.robotwin.play_scene_audit`: the worker of
  `ora play audit --env robotwin`.
- `python -m ora.environment.backends.robotwin.vendored_install`: run by
  `ora setup --env robotwin`.

## Documentation

- [RoboTwin](../../../../docs/environments/robotwin.md) and the
  [quick run](../../../../docs/getting-started/robotwin.md)
- [RoboTwin seed and scene qualification](../../../../docs/reference/robotwin-qualification.md)
- [RoboTwin contact recovery](../../../../docs/reference/robotwin-contact-recovery.md)
- [Free play: RoboTwin scene sets](../../../../docs/usage/play.md#tables-with-many-objects-robotwin-scene-sets)
- [Host-side backend](../../../orchestrator/backends/README.md)
