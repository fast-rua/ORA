# Policy I/O schema

A policy turn receives one fresh observation, may call read-only tools, and
ends with at most one world-changing tool call or a lifecycle call. The
Supervisor validates that call against a versioned action contract, executes
it, and publishes the receipt with the next observation. This page lists the
fields on each side, for the default baseline
`native_full_explicit_motion_direct_v4` unless stated. Frames and units per
robot are on [Frames and units](../robots/frames-and-units.md); the loop itself
is described in [How ORA works](../how-ora-works.md) and
[Runtime lifecycle](runtime-lifecycle.md). In the trees, `?` marks an
optional or nullable field and `[n]` a fixed-length array.

## Observation delivered each step

For every fresh observation the Supervisor publishes a context record. The agent
core renders it, with every camera's RGB image attached, into the model's
observation message. None of this needs a tool call.

```text
ora.context.v1
├─ main_camera             str                   agentview | head_camera | head
├─ proprioception          object                raw measured state (manifest below)
├─ previous_execution      object?               receipt of the last action; null before the first
├─ remaining_budget        object                see "Budgets"
├─ runtime_notice?, policy_state {step, path, sha256, state_sha256}, observation_fresh
├─ forbidden_modalities_absent  [str]            object GT, segmentation, contact, predicate, ...
├─ current_visual_observation                    all declared cameras, native RGB
│  ├─ images[]             {camera, modality "rgb", width, height, sha256, path}
│  ├─ camera_geometry.<camera>
│  │  ├─ intrinsic_cv float[3][3]; extrinsic_cv float[4][4] world->camera; cam2world_cv (inverse)
│  │  ├─ camera_origin_world_m float[3] (m); camera_axes_world {right, down, forward}
│  │  │  (the three derived values come from plain Python arithmetic: the same digits on every host)
│  │  └─ depth_unit "metres"; depth_semantics "optical-axis z distance in metres"
│  └─ depth_delivery       "on_demand"           tools read depth; it is not inlined
├─ skill_observations?     [image ref]           views captured inside a committed skill
└─ public_robot_facts      ora.current-public-robot-facts.v1
   ├─ controller_feedback? RoboDojo only: arrival tolerances, native_actions_per_command 20,
   │                       gripper_drive_limit
   ├─ cameras.<camera>     {width, height, pixel_coordinates "opencv_top_left_xy",
   │                        calibration {status "reported_in_camera_geometry"},
   │                        controlled_point_pixels.<arm> {pixel_xy?, inside_image,
   │                                                       in_front_of_camera, depth_m}}
   └─ end_effectors.<arm>  "right" (Panda, Robotiq) or "left" and "right"
      ├─ pose_xyz_xyzw     float[7]              world metres, quaternion xyzw
      ├─ controlled_frame  panda_grip_site | aloha_grip_center_tcp | arx_x5_link6 |
      │                    robotiq_2f85_tcp
      ├─ axes              {approach_world, closing_world: float[3]}
      └─ gripper           per robot, below
```

`gripper` by robot; every variant also states
`grasp_evidence: not_established_by_joint_measurement`:

| Field | LIBERO (Panda) | RoboTwin (ALOHA) | RoboDojo (ARX X5) | RoboLab (Robotiq 2F-85) |
|---|---|---|---|---|
| `measured_finger_displacements_m` | float[2], m | — | float[2], m | — (`measured_finger_joint_deg`, the finger joint, deg) |
| `measured_aperture_mm` | float, mm | — (raw `gripper_opening_m`, m) | `null`, uncalibrated | float, mm, inner fingertip-pad faces from measured link poses |
| `measured_opening_normalized` | 0 closed to 1 open | 0 closed to 1 open, or `null` | encoder scale, about 0.185 empty close to 1 open | aperture over 85 mm, 0 closed to 1 open |
| `commanded_opening_normalized` | float? | float? | float | float? |
| `command_target` | `open`, `close`, `unknown` | same | same | same |

**Observation manifest.** Each backend writes an immutable `ora-observation.v1`:

```text
ora-observation.v1
├─ observation_id str; sequence int; encoding "metric_rgbd_v1"; main_camera str
├─ cameras.<camera>  {width, height, source_camera?, mount?, calibration_dynamic?,
│                     files: {rgb_lossless_png, depth_numpy_m_f32, calibration}: {path, sha256, bytes}}
├─ proprioception.end_effectors.<arm>  {pose_xyz_xyzw, gripper?, gripper_finger_qpos?,
│                     gripper_command?, gripper_command_target?, gripper_opening_m?, gripper_state?}
├─ controller_feedback?           RoboDojo
├─ robot_geometry_verification?   descriptor hash check; public facts require it
└─ previous_execution_receipt?    {path, sha256, receipt}
```

**Budgets.** `remaining_budget` holds `robot_decisions` (int, world-changing calls left this round),
`model_calls` (int), `wall_time_s` (float, seconds), on RoboDojo and LIBERO
`native_actions_limit` and `native_actions_remaining` (on LIBERO the plan's
`budgets.max_sim_steps`, one native action per 50 ms control tick, and every
receipt's `native_actions`), and `session_round` /
`session_rounds` under lossless context or multi-round sessions. The values are
those of the moment the observation was published; a further turn on the same
observation shows them unchanged, while the agent core caps that turn's model
calls by what the round has left when the turn starts.

**Receipt of the last action.** `previous_execution` is the verified record
`{path, sha256, receipt}`. `receipt.schema` selects how the backend's receipt is shown, so fields vary: the validated
action, measured before/after state, requested versus measured motion, a
planner summary, a motion status (`planner_failed`, `plan_rejected`,
`not_reached`, `execution_fault`, `reached` on RoboTwin), and for a wait an
`ora.wait-receipt.v1` block (`ticks_requested`, `ticks_used`, `physics_steps`,
`physics_steps_per_tick`, `tick_period_s`, `terminated_early`). A receipt never
claims contact, grasp or task success.

## Action contracts

Tools compile into an action envelope before the Supervisor sees it. Every envelope has exactly the fields `schema`,
`observation_id`, `action_id`, `action`, `arguments` and
`bound_observation_sha256`; the two ids and the hash are non-empty strings.
Any key containing `actor`, `object_gt`, `ground_truth`, `segmentation`,
`contact`, `success_predicate`, `play_once`, `task_source` or `env` is rejected
anywhere in the payload.

| Codec | `schema` | Used by |
|---|---|---|
| `runtime_action_v3` | `ora-action.v3` | `ora local` (`local_atomic_cpu_v1`) and the Synthetic Native test surface; RoboDojo also accepts it |
| `target_state_v4` | `ora-action.v4` | Synthetic (`robot_runtime_v2`): the quickstart, `ora interactive`, `ora play --env synthetic` |
| `motion_action_v3` | `ora-motion-action.v3` | default LIBERO, RoboTwin and RoboDojo surfaces |
| — | `ora-skill-sequence-action.v1` | `grasp_execute` committed sequences |

**Action v3 (`ora-action.v3`).** `move_pose` and `set_gripper` are mutually
exclusive: `move_pose` holds the current gripper directive and `set_gripper`
holds the current pose target. Measured state may still change under contact.
`robot.submit_action` no longer takes this form: `robot_runtime_v1`, which did, was
retired on 2026-11-02, and sealed runs keep what they recorded
([policy README](../../ora/policy/README.md)).

```text
move_pose    {arm: left|right, position_m: float[3] (world m), orientation_xyzw: float[4]}
set_gripper  {arm: left|right, gripper: float in [0, 1]}
```

**Target-state v4 (`ora-action.v4`).** `action: "target_state"`; pose and
gripper are independent and at least one is required. `target_state_phases`
stages a combined request as a v3 `move_pose` phase (`<action_id>:pose`), then a
v3 `set_gripper` phase (`<action_id>:gripper`) that is skipped if the pose did
not arrive. One robot decision, one receipt.

```text
{arm: left|right, pose?: {position_m: float[3], orientation_xyzw: float[4]} | null,
 gripper?: float in [0, 1] | null}
```

**Motion v3 (`ora-motion-action.v3`).** Sparse per-arm targets; an omitted arm
is not commanded.

```text
move_pose        {targets: {<arm>: {position_m, orientation_xyzw}},
                  interactions: {<arm>: free_space|precision|contact_approach}}
                 or {plan_ref: "plan-<32 hex>"}     invalidated by any world change
move_free_space  as move_pose, interaction must be free_space
set_gripper      {targets: {<arm>: open|close}}
                 or {targets: {<arm>: {opening: open|close|float 0..1, rate_per_s?, max_lead_m? <= 0.0108,
                                       hold_lead_m? <= max_lead_m,
                                       end?: {tolerance, stall_window_native_actions, stall_still_step},
                                       arm_settle?: {joint_speed_rad_s, max_native_actions,
                                                     first_reading?: native_action|joint_velocity}}}}
recover_motion   {targets: {<arm>: backtrack_last_safe|home}}
wait             {ticks: int 1..20}
follow_joint_trajectory
                 {arms: {<arm>: {knots_rad: [[6] x N], gripper_knots?: [N], expected_link6_xyz_xyzw?: [[7] x N]}},
                  interpolation?: knots_linear_ff,
                  guards?: {tracking_rad?: float | [N], tracking_window_steps?, progress?: {window_steps,
                            min_ratio, min_commanded_rad} | null, tracking_growth?: {min_error_rad,
                            min_rise_rad, over_steps, sustained_steps} | null, on_stop?: hold_measured},
                  gripper?: {max_lead_m}, settle?: {native_actions, joint_tolerance_rad, joint_speed_rad_s},
                  admission?: {max_joint_step_rad <= 0.2},
                  part_of_move?: {fraction, goals_xyz_xyzw: {<arm>: [7]}},
                  recovery?: {targets: {<arm>: backtrack_last_safe|home},
                              recovery_targets: {<arm>: {strategy, arm_joint_qpos, pose_xyz_xyzw}}}
                           | {targets: {<arm>: retrace}, retrace: {<arm>: {moves: 1..5, arm_joint_qpos}}}}
```

`follow_joint_trajectory` and the object form of `set_gripper` run only on
RoboDojo (every other backend refuses them as `invalid_arguments` naming
`action`). In the object form, `hold_lead_m` is the lead from the native action
after the first one in which the first finger moved less than
`end.stall_still_step`: a close may approach at one lead and keep a smaller
squeeze once its finger stops (the stall and arrival rules do not change).
`arm_settle.first_reading: joint_velocity` reads the arms' measured joint
velocities before the first settle tick, so an arm already slower than
`joint_speed_rad_s` costs no native action; `native_action`, the default, looks
once after one held native action. A target without them normalizes, runs and
records exactly as before. One such action may use up to 250 native actions
(`joint_trajectory.native_action_cap`; 20 for every other action). Knot `k` is
the joint target at the end of native action `k`; the environment drives every
physics step linearly between consecutive knots (knot 0 is the measured joints)
with the knot slope as velocity feed-forward, and the last executed step with
velocity 0. Admission refuses knots outside the live joint limits or a step
above `admission.max_joint_step_rad` (`trajectory_joint_limit`,
`trajectory_start_discontinuous`, `trajectory_step_exceeds_limit`) before any
write. Guards are evaluated after every physics step: tracking trips after
`tracking_window_steps` steps with the commanded-minus-measured joint error
above the knot's bound (default a quarter of the knot step, at least
0.012 rad; `stop.rule` `bound`), or when the error keeps growing (default:
above 0.004 rad and up by more than 0.0015 rad over 10 physics steps, for 10
consecutive steps; `stop.rule` `growing_error`); progress
trips when the measured joints advance along the commanded direction by less
than `min_ratio` of the commanded advance over `window_steps` steps. A trip holds
the measured joints with velocity 0 for the rest of the tick and ends the
action (receipt `trajectory.stop`). The receipt (`ora.robodojo-sparse-motion-receipt.v1`
with `motion_strategy: joint_trajectory`, `command_invariant` v3 and a
`trajectory` block, `ora.robodojo-trajectory-receipt.v1`) reports raw final
joint and link6 residuals. `part_of_move` marks knots that end short of the
caller's goal (a move cut to fit its budget): when every knot runs, the action
still ends `not_reached` with `native_action_budget_exhausted`, and
`trajectory.part_of_move` and each arm's `motion_progress` (outcome
`cut_short`) give the measured distance left to the goal; the residuals and the
held pose target are the cut point's. The `action_summary` line states that
distance left, never the cut point's residual, and names
`native_action_budget_exhausted` (`stage_evidence.failure_code`) rather than the
public `failure_code` `controller_timeout`. A recovery its budget cut short
(`recovery_progress.outcome` `cut_short`) states the joint travel left the same way, having
no goal pose. A trajectory stopped by its tracking or
progress guard carries `arms.<arm>.stop_evidence` (`session_core/stop_evidence.py`):
`path_fraction` of its knots, the `lagging_joint` (1 is the base joint) and `lag_rad`,
the commanded joints at the stop against the measured joints. Every backend's
`action_summary` adds one clause for a move stopped under way
(`controller_tracking_error`, `controller_stalled`, `blocked`, `contact_stop`,
`tracking_diverged`): how far along it stopped, from `path_fraction` or else the
distance moved against the distance left, which way it was moving, the joint that fell
behind where the receipt can tell, and for the first three, when `physical_cause` is
`unknown` and no measured contact ended it, that it probably touched something, worded as inference.
A RoboCasa arm stop on a joint held back records that joint itself (`endpoint.lagging_joint`,
`lag_rad`, and `path_fraction` for a stop before the last knot), and the same `stop_evidence` carries it.
Where the receipt names the stop that ended a move (`detail.endpoint.stop_reason` for an arm on
RoboCasa and RoboLab, `detail.base.stop_reason` for RoboCasa's base), the line follows the outcome
with that stop when the outcome does not say it: `ended by the gripper contact stop` for
`contact_detected` (a `reached` move wherever it stopped), and, for a move that did not reach,
`ended by joint lag` (`joint_lag`), `ended with the joints at rest short of the path's end`
(`settle_stalled`), `ended still settling after the last target` (`settle_timeout`) and
`ended by base contact` (`base_contact`), never with a force value. A move the planner refused (`planner_failed`), or whose plan's validation refused it before it moved (`failure_stage` `plan_validation`, RoboCasa's admission), ends its `action_summary` with the planner's own words, the first `recovery` sentence of its planning diagnostic (LIBERO's and RoboCasa's `detail.planning_diagnostics`, a RoboDojo or RoboTwin arm's `planning_diagnostics` or `planner.planning_diagnostics`), naming the arm when two arms moved: why it refused and what near the goal plans. The fixed hints a planner gives when it found no cause (the shared planner's and LIBERO's "Try a different target position or orientation, or an intermediate pose.", RoboTwin's vendored and RoboDojo's SDK no-endpoint texts) are left to the detail. An effect with `detail.admission_refusal` then restates it (`ora/agent/toolkit/tools/_refusal_words.py`, the words a refused dry run's `summary` uses too): where a line's IK stopped and why (`cause`, `at_pose`, the joint's jump), whether the target itself is reachable from here (`endpoint`), which robot part enters what the depth shows and how far along, a target's distance from the arm mount against the reach, where a base move's footprint or the arm it carries meets the depth (the arm: retract it first), and the first of `base_suggestions`, a base pose from which the same move is admitted (or that none was found, and how many the carried arm blocks). A move that lifted RoboCasa's torso first says so after the motion clause, from `detail.torso`. An explicit settle that does not converge
while the link6 pose is inside the endpoint acceptance (10 mm, 0.1 rad) is
`reached` with `failure_stage` `settle` and `failure_code` `arm_not_settled`
(`motion_progress.outcome` `arrived_not_settled`, `settle_joint_error_rad`);
only a pose outside it is `endpoint_residual_exceeded`. A trajectory moves the previous recovery anchor
(`exec.recovery.<arm>.previous`) to its start and records with it `measured_path_rad`, the joints measured
at the end of every native action it ran; a released `move_pose` drops that path. `recovery` marks a
recovery the toolkit planned (recover_pose and return_home on the trajectory surfaces): it names every arm
of the trajectory and restates either its anchors as `recover_motion` does (return_home: `home`) or, for a
retrace, each arm's retrace stack depth and its top move's start joints; the host refuses it with
`stale_or_mismatched_recovery_target` unless they are current, and the trajectory keeps the anchor and its
path (receipt `trajectory.recovery`). Every other admitted trajectory also records its move on the arm's undo
stack `exec.recovery.<arm>.undo` (`moves`, oldest first, the last 5 arm-moving tool calls): a new entry for a
new tool call (its id, tool and ordinal, start joints and pose, measured path), or its path appended to the
entry of the same call. A retrace that runs every knot pops the entry it named; a cut or stopped one leaves it
on top with `undo_progress` (`retrace_calls`, `fraction_undone`). The receipt names the operation
(`trajectory.undo`) and each arm's `motion_progress.undo_depth`. The host lends the tool call identity with
each execute request (`tool_call`). Per-step evidence is in the controller journal and
trace v2 (`requested`, `command`, applied position and velocity targets,
measured state; samples carry the 10 `control_steps`), and restore sends those
recorded steps again.

## Tool calls on the default surfaces

The three robot surfaces advertise, in order: `observe`, `mark_point`,
`geometry_query`, `compute`, `vision_segment`, `vision_ground`, `grasp_plan`,
`grasp_execute` and `align_plan` (dual-arm only), `set_gripper`, `move_to`,
`move_pose`, `rotate_wrist`, `rotate_pitch`, `recover_pose`, `wait_ticks`,
`report_issue`, `check_task`, `finish_episode`. A world-changing tool must be
the only tool call in its assistant response. It returns `status: "submitted"`;
its effect arrives as the next receipt.

```text
move_pose  targets.<arm>   (LIBERO: the same fields flat, no targets, no arm)
├─ position_mode       hold_current | absolute | delta              required
├─ position_m?         float[3]  world, metres                      with absolute
├─ translation_m?, position_frame?   float[3] m; world | end_effector   with delta
├─ orientation_mode    hold_current | absolute_quaternion | axis_angle_delta   required
├─ orientation_xyzw?   float[4]  world                              with absolute_quaternion
├─ rotation_axis?, rotation_angle_deg?, orientation_frame?          with axis_angle_delta
│                      x|y|z; number in DEGREES; world | end_effector
└─ interaction?        free_space | precision | contact_approach    default free_space
```

Other mutation inputs (dual-arm / LIBERO): `set_gripper {targets: {<arm>: open|close}, target_ref?}`
/ `{action, target_ref?}`; `move_to {targets: {<arm>: {position_m}}}` / `{position_m}`;
`rotate_wrist {targets: {<arm>: deg in [-180, 180]}}` / `{angle_deg}`;
`recover_pose {targets: {<arm>: previous|home}}` / `{strategy}`; `wait_ticks {ticks: 1..20}`. The toolkit
trajectory surfaces (robodojo_placement_toolkit_v5, v6) take `recover_pose {arms: [left|right, ...]}` (undo
each arm's latest move along its measured path, one per call) and `return_home {arms: [...]}` (lift, then a
screened joint unwind to the reset joints).

**Refusal envelope:**

```text
{ ok: false,
  error: { code, message, retryable: bool, retry_requires_changed_arguments?: bool,
           field?, received?, constraints: object, infrastructure: bool,
           repetition?: {identical_failures_in_a_row: int, note} } }
```

`retryable` means a later call of this tool can still succeed; it is not an
instruction to resend. `repetition` is added when the same failing call repeats.
Arguments that are not one JSON object are refused before any tool reads them,
on every tool the toolkit serves: `invalid_arguments`, `$input has the wrong
input type; expected object`, `constraints: {violated: "type", type: ["object"]}`.

| Family | Codes |
|---|---|
| Argument (same request is refused again) | `invalid_arguments`, `invalid_motion_arguments`, `invalid_servo_delta`, `invalid_tool_call`, `out_of_bounds`, `target_ref_unavailable`, `unknown_section`, `workspace_bounds` |
| Turn and authority | `action_pending_fresh_observation`, `duplicate_action`, `mutation_batch_rejected`, `episode_terminal`, `stale_observation` |
| Lifecycle | `task_success_not_confirmed`, `task_success_latched`, `finish_reason_insufficient`, `finish_category_required`, `abort_without_infrastructure_evidence` |
| Infrastructure (`infrastructure: true`) | `internal_tool_error`, `environment_contract`, `environment_unavailable`, `artifact_hash_mismatch`, `artifact_integrity_fault`, `invalid_camera_calibration`, `pose_state_unavailable`, `embodiment_binding`, `grounding_unavailable`, `tool_capability_unavailable`, `tool_surface_capability_mismatch` |

Backend input refusals arrive as
`{kind: "refusal", type: "EnvironmentRefusal", code: "invalid_arguments", message,
retryable: true, field, received, constraints: {allowed}}`.

## Lifecycle

| Call | Input | Effect |
|---|---|---|
| `check_task` | `{}` | Read-only; `task_success` from the official checker, or retryable `task_checker_unavailable` |
| `finish_episode` | `outcome` (`success`, `give_up`, `abort`), `reason`, `category?` | Ends the policy session. `success` needs `check_task=true` at the same boundary, and a surface without `check_task` (`robodojo_placement_toolkit_v6`) has no `success` outcome, since the official predicate ends the episode by itself; `give_up` needs `category` (`unreachable`, `object_lost`, `scene_damaged`, `repeated_failure`, `other`; the file-backed toolkit surfaces also declare `insufficient_budget`, for when what `remaining_budget` has left cannot finish the remaining steps, and the runtime accepts it on every surface); `abort` needs an infrastructure failure the Harness recorded this round. `reason`: at least 24 characters and 4 words; a success reason names its evidence source |
| `report_issue` | `{message, kind?}`; `kind` is `defect` (default) or `confusion` | Appends a diagnostic record; no robot action, episode continues. The file-backed toolkit surfaces declare `kind`; released surfaces declare only `message`. Each `openeta-reported-issues.jsonl` row is `{message, kind, time}`, `kind` defaulting to `defect` |

Only the environment's official boundary sets the verdict; `finish_episode`
never does. Synthetic exposes no lifecycle tool and reports
`official_success: null`.

Underneath, the Supervisor drives a `PolicySession` (`start`, `act`,
`accept_runtime_event`, `interrupt`, `close`); `act`
returns `ActionProposal`, `NoAction` or `LifecycleRequest`, and runtime events
are `fresh_observation`, `environment_receipt`, `action_discarded` (a round
budget ended the round before the proposal was executed), `official_verdict`,
`run_termination` and `operator_message`. `PolicyTurnInput.budget` (the round
budget left when the turn starts) reaches the core as the `turn_budget` keyword
of the delegate's `act`.

## Known quirks

- "v3" names two contracts: `ora-action.v3` and `ora-motion-action.v3` (the default).
- Recovery is `previous` in tool schemas and `backtrack_last_safe` in the motion contract.
- Base tool definitions use dotted names (`vision.segment`); default surfaces advertise `vision_segment`.
- Base `finish_episode` definitions list `failure`; default surfaces replace it with `give_up` and `category`.
- One matrix, two names: `extrinsic_cv` in `camera_geometry`, `world_to_camera_cv` in public facts.
- Measured pose: `pose_xyz_xyzw` (observation), `eef_pose_xyz_xyzw` (descriptors),
  `measured_eef_pose_xyz_xyzw` (ARX controller contract).
- RoboDojo's controller: `CUROBO_IK_JOINT_INTERPOLATION` (prompt profile),
  `SDK_IK_JOINT_INTERPOLATION+LOCAL_CARTESIAN_STEP` (workspace profile); codec
  `runtime_action_v3` (controller contract) versus `motion_action_v3` (baseline).
- Camera records differ: LIBERO omits `mount`; RoboTwin omits `source_camera`; RoboTwin
  and Synthetic calibration files carry `cam2world_gl`, LIBERO and RoboDojo `cam2world_cv`.
- The Synthetic quickstart runs `target_state_v4`, but its manual (`aloha-agilex/manual-v1.md`)
  describes exclusive v3 actions, and its task moves `left`, which the target-state
  manual calls non-actionable.

## Related pages

[Frames and units](../robots/frames-and-units.md),
[action contracts API](../api/actions.md), [public interfaces](public-interfaces.md),
[configuration and evidence](configuration-and-evidence.md),
[evaluation protocol](../evaluation/protocol.md),
[ToolSurface authoring](../developer/authoring/tool-surface.md).
