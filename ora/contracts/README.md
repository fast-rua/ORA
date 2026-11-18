# ora/contracts

## Responsibility

The versioned contracts that cross the trust boundary between the agent and
the environment: what a proposed action is, how a policy pose request compiles
into a controller target, and what a refusal, a reach check or a receipt
carries. Each module is a schema identifier plus the pure validation and
compilation behind it; both sides of the boundary, the orchestrator's
Supervisor and the isolated simulator worker import the same code, so they
cannot disagree about a field.

| Module | Contract |
|---|---|
| `toolkit_protocol.py` | Manually numbered data-interface protocol and advisory replay compatibility evidence |
| `execution_state.py` | Private `ora.execution-state.v1` request/reply: host-owned holds, measured anchors and explicit recovery joint targets |
| `action.py` | The action envelopes (`ora-action.v3` runtime action, `ora-action.v4` target state, `ora-motion-action.v1`/`v3`) and `validate_runtime_action` |
| `joint_trajectory.py` | `follow_joint_trajectory` and the rate-limited `set_gripper` target of `ora-motion-action.v3`, and the per-action native-action cap (20 legacy, 250 for these) |
| `motion_speed.py` | The end-effector speed a move asks for (the optional `speed` of an `ora-motion-action.v3` `move_pose`): the presets fast, normal and careful, `speed_m_s`, `slow_approaching`, their caps, the policy-argument rules and the wire form |
| `kinematics.py` | The read-only kinematics query (`ora.kinematics-request.v1`: `ik_chain`, `fk`, `screen_path`) carried over the align bridge |
| `pose.py`, `motion.py` | Backend-neutral pose compilation (`ora.pose-frame-contract.v2`, `ora.compiled-pose.v1`) and the shared motion thresholds and servo compilation |
| `skill_sequence.py` | One committed manipulation sequence submitted as a single policy decision (`ora-skill-sequence-action.v1`) |
| `wait.py` | The shared `wait` action and its receipt |
| `gripper_feedback.py` | Public gripper intent, drive target and encoder evidence, kept apart |
| `acceptance.py` | Closed numeric motion evidence |
| `planning_diagnostics.py` | Bounded planning-failure feedback and the motion-probe request |
| `alignment.py`, `reach_check.py` | The read-only `align_plan` and reach-check request and result envelopes |
| `action_plan.py` | The read-only plan of one action a dry run asks for (`ora.action-plan.v1`): its segments, the tools that move the robot, and an arm's chain of joints from forward kinematics |
| `robotwin_reach.py` | RoboTwin's reach-check models and refusal reasons |
| `refusal.py`, `workspace.py` | Public pre-execution refusals; action-workspace validation and recovery |
| `policy_receipt.py` | A receipt's copy for the policy: without the evidence that names an implementation (a paced move's timing, the planner's own record), which stays in the sealed receipt |
| `robot_geometry.py` | Host-safe comparison of public robot-only geometry |
| `robodojo_arx.py` | The source-bound ARX X5 facts shared by policy composition and the RoboDojo controller |

## What belongs here, and what does not

Here: a contract both the agent side and the environment side must read the
same way, with its schema id and its validation. Not here: any session,
transport or simulator code (`ora/environment/`), tool behaviour
(`ora/agent/toolkit/`), or the Supervisor's admission rules
(`ora/orchestrator/episode/`). A contract changes by a new schema version, not
by editing an old one in place.

**Pinned files.** Fourteen modules are members of the RoboDojo worker kernel
(`WORKER_KERNEL_PATHS` in `ora/orchestrator/robodojo_kernel.py`), whose card
and task qualification records hash them by path and bytes: `acceptance.py`,
`action.py`, `execution_state.py`, `gripper_feedback.py`, `joint_trajectory.py`,
`kinematics.py`, `motion.py`, `planning_diagnostics.py`,
`pose.py`, `reach_check.py`, `robodojo_arx.py`, `robot_geometry.py`,
`skill_sequence.py` and `wait.py`. Renaming one, or changing a byte, a comment
included, changes the kernel identity. Historical verification stays strict; live
admission records a warning for a missing qualification or changed kernel
(see [state and admission](../../docs/reference/toolkit-state.md)). `alignment.py`, `refusal.py` and `workspace.py` are not kernel
members.

## Dependency direction

The bottom layer: these modules import only the standard library and each
other, never another `ora` package or a third-party one. Every layer above may import them
(`ora.agent`, `ora.environment`, `ora.orchestrator`, `ora.app`).

## Entry points

- `from ora.contracts import validate_runtime_action, compile_pose_request, ...`:
  the package re-exports the action, motion and pose names; import the other
  modules directly (`ora.contracts.wait`, `ora.contracts.reach_check`).
- No module here has a process entry point.

## Documentation

- [Action contracts API](../../docs/api/actions.md)
- [Policy I/O](../../docs/reference/policy-io.md) and
  [runtime lifecycle](../../docs/reference/runtime-lifecycle.md)
- [Frames and units](../../docs/robots/frames-and-units.md)
- [Repository structure](../../docs/developer/repository-structure.md),
  the pinned files
