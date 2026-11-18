# ora/environment/controllers

## Responsibility

The controller rules that both the host and the isolated worker must apply the
same way: how a motion transaction is planned, admitted, executed and judged by
measured convergence; the safety envelopes and clearance checks; each
embodiment's control constants; and the wire contracts of the planner workers.
Launch preflight reads them to check a plan before any simulator starts, and
the simulator backends execute with them.

| Module | Holds |
|---|---|
| `motion_transaction.py`, `measured_convergence.py`, `convergence_request.py` | The backend-neutral motion transaction (intent, plan artifact, execution, status) and the stage loop and request admission of `execute_measured_convergence` |
| `plan_validation.py`, `motion_progress.py`, `command_drift.py` | Joint-plan normalization and admission, measured progress for executors and receipts, drift from the last issued pose target |
| `reach_check.py`, `reachability.py`, `alignment.py` | Exact-pose reach checks with the backend's read-only inverse kinematics, Cartesian reachability from forward kinematics, and IK-ranked approach poses for `align_plan` |
| `joint_path_clearance.py`, `inter_arm_clearance.py` | Straight joint-path self-collision screening with a solver's sphere model; the action-start clearance policy between two arms |
| `gripper_stall.py`, `force_feedback.py` | The jaw-stall rule every gripper loop shares; sensor-frame force projection, the bounded precision force stop and a contact move's two stops |
| `embodiment_control.py`, `profile_capabilities.py`, `hybrid_motion.py` | Hash-bound execution semantics kept apart from embodiment facts, capability derivation from a control profile, the hybrid Panda motion profile |
| `kept_plans.py` | The planner a session asks: what a dry run planned is what the same request later runs, keyed by the request, world hash and start joints included |
| `motion_planner.py` | The planner-neutral motion planning contract (plan and IK requests and answers, the neutral failure reasons), the implementation registry and `MotionPlannerClient`, which starts and talks to a planner worker of `ora/environment/planners/` |
| `motion_pace.py` | Planner-neutral end-effector pacing: any planner's joint path timed so the controlled point keeps the requested speed (joints as the ceiling, a slow final segment), the planned peaks, the wrist report and the refusal of an unrequested wrist turn, and the budget fit that never cuts a large wrist turn; delegated to the timing worker when the process has one, otherwise a port of the RoboDojo toolkit's timing, held to it by a parity test |
| `motion_timing.py` | Which timing a process's paced moves use (`configure`, `ORA_TIMING_PYTHON`, the fallback), the hardware jerk limits per robot and the `timing` evidence of a receipt |
| `motion_retime.py` | The `retime` contract (request, answer, the timing refusals) and `TimingClient`, which starts the timing worker under `ORA_TIMING_PYTHON` |
| `motion_worker.py` | What the planner and timing clients share: the JSON-lines worker process, the protocol id, the contract's value checks |
| `pace_refusal.py` | The refusals of pacing a path, in the same words whichever timing measured it |
| `jaw_flip_retry.py` | The one retry of a pose move the pacing refused for its wrist (`paced_path_wrist_unwind`): with the jaws open, the target's jaw twin (a half-turn about the approach axis), and the record a receipt carries (`motion.wrist.jaw_flip_retry`); LIBERO, RoboCasa365 and RoboLab call it from the real call and the dry run alike |
| `depth_world_inputs.py` | The world-input file of `public_depth_voxel_v1`: the cameras' depth, the held object and, on a two-arm robot, the other arm's joints and base |
| `fk_parity.py` | The forward-kinematics check a robot passes before a planner speaks for it: `ROBOT_TOLERANCES`, and `parity`, the planner's link poses against the simulator's |
| `depth_world.py` | The planners' public-depth collision world (`public_depth_voxel_v1`), one parameter set per robot (`ROBOT_WORLDS`): the voxel ESDF built from the named cameras' depth, the start and goal carves and the held-object spheres, and its export to cuRobo V2's voxel layout |
| `depth_load.py` | The spheres a held load is attached to the planner with: fitted to its voxels, and cut at the floor of one goal. |
| `depth_esdf.py` | The depth world's clamped signed distance field (`signed_distance`): capped integer transforms over the occupied box, bit for bit the whole grid's |
| `base_placement.py` | Where a mobile manipulator could stand to reach a target (the base-placement search: stands around the target (at the measured torso height where no action sets the lift), the base's footprint against the public depth on floor a camera saw, the shared IK blank and across the servo's arrival band, the refused move's own admission from the stand, ranked by the base's move with nudges in place last) and `move_base`'s check along the base servo's path, the footprint and the robot's body it carries; a backend supplies its `MobileBase`, the measured `Here`, its `OwnBody` (what of the depth is its own) and the planner's answers |
| `held_depth.py` | Which public depth is the robot's own beyond its body: the depth a held load's chain of voxels links to (split where it runs on past the load's reach, fragments near it joined) and the mixed pixels at the robot's edge; `base_placement.public_points` takes both out of what a base move meets and carries the load's |
| `robot_spheres.py` | The shared planner's collision spheres of the Panda on the host: cuRobo V2's `franka.yml` with ORA's changes, on the URDF chain, so a host knows where the planner has the robot at measured joints (`SPHERE_SET_SHA256`, the set the planner names, checked at session start) |
| `libero_osc.py`, `libero_wait.py` | The source-bound RoboSuite `OSC_POSE` contract and the LIBERO `wait` hold |
| `robodojo_arx.py`, `robocasa_omron.py`, `robolab_droid.py` | One owner per embodiment for its control constants: the ARX X5 (RoboDojo), the PandaOmron (RoboCasa365) and the DROID Franka with Robotiq 2F-85 (RoboLab) |

## What belongs here, and what does not

Here: deterministic, host-safe controller and safety rules, one file per rule
or per embodiment's constants. Not here: simulator calls and the executors
that step a scene (`ora/environment/backends/<sim>/`), the action envelope
itself (`ora/contracts/`), and whether a proposal is admitted at all (the
Supervisor, `ora/orchestrator/episode/`). The modules stay flat.

**Pinned files.** Most modules here are members of the RoboDojo worker kernel
(`WORKER_KERNEL_PATHS` in `ora/orchestrator/robodojo_kernel.py`, which lists them),
hashed by path and bytes into every RoboDojo card and task qualification record:
the shared motion base as well as `alignment.py`, `convergence_request.py`,
`gripper_stall.py`, `joint_path_clearance.py`, `measured_convergence.py`,
`motion_progress.py`, `motion_transaction.py`, `plan_validation.py`,
`reach_check.py`, `reachability.py` and `robodojo_arx.py`. Do not rename them
or change a byte, comments included, without re-qualifying the RoboDojo lanes.

## Dependency direction

These modules import the standard library, numpy, `ora.contracts` and each
other, never a simulator, an `ora/environment/backends/` module or a package
above the environment. The backends import them; so do the orchestrator's
preflight, evaluation and backend modules.

## Entry points

- `from ora.environment.controllers import execute_measured_convergence, MotionBackendAdapter, ...`:
  the package re-exports the motion-transaction names; import the other
  modules directly.
- `motion_planner.py` starts `python -m <IMPLEMENTATIONS[name].worker_module>`
  (`ora.environment.planners.curobo_v2.worker`); `motion_planner.planner_client`
  builds the client a `runtime.motion_planner` configuration selects. No module
  here is started as a process.

## Documentation

- [Runtime lifecycle](../../../docs/reference/runtime-lifecycle.md)
- [RoboDojo motion qualification](../../../docs/reference/robodojo-motion-qualification.md)
- [RoboTwin contact recovery](../../../docs/reference/robotwin-contact-recovery.md)
- [Frames and units](../../../docs/robots/frames-and-units.md)
