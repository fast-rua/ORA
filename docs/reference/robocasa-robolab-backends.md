# RoboCasa365 and RoboLab backends

Two single-arm backends join RoboTwin, LIBERO/LIBERO-Pro and RoboDojo on the
default baseline `native_full_explicit_motion_direct_v4` (instances `robocasa`
and `robolab`). Both drive `motion_action_v3` through the Supervisor, run in an
isolated `local_rpc` worker under their own interpreter and take their verdict
only from the benchmark's own success check.

## Identity and admitted tasks

| | RoboCasa365 | RoboLab |
|---|---|---|
| Source pins | robocasa `a9e152e2` (1.0.1), robosuite `5ce6643f` (1.5.2), MuJoCo 3.3.1, target split | RoboLab `ad45d4f9`, Isaac Sim 5.1.0.0, IsaacLab 2.3.2.post1, warp 1.11.0 |
| Host-safe identity | `ora/environment/benchmarks/robocasa.py` | `ora/environment/benchmarks/robolab.py` |
| Embodiment | PandaOmron: Panda arm on a holonomic base with torso | DROID Franka Panda with a Robotiq 2F-85 |
| Controlled point | robosuite `grip_site` (`panda_grip_site`) | Robotiq TCP 0.131 m along the flange +x (`robotiq_2f85_tcp`: local z approach, local x jaw line) |
| Control tick | 20 Hz (arm joint-position targets, sent as deltas from the measured joints; base velocity) | 15 Hz (absolute differential-IK target, 8 physics steps of 1/120 s) |
| Cameras (`agentview`, `wrist`) | `robot0_agentview_left` (base-mounted, moves with the base), `robot0_eye_in_hand` | `over_shoulder_left_camera` (fixed), `wrist_cam` |
| Tasks | all 50 of `target50` (18 atomic seen, 16 composite seen, 16 composite unseen); `OpenDrawer`, `PickPlaceCounterToCabinet` and `NavigateKitchen` with a reviewed completion contract | all 120 tasks of RoboLab-120 ([RoboLab tasks](robolab-tasks.md)), each with a reviewed completion contract |
| Episode boundary | each task's evaluation horizon from robocasa `dataset_registry.py` (450 to 4350 control steps), passed to `robosuite.make`, not the Kitchen constructor's 1000-step default | the upstream `episode_length_s` at 15 Hz (300 to 4500 native actions; 750 and 900 for the three reviewed tasks) |
| Verdict | `_check_success`, latched at the first true step within the horizon | RoboLab's success termination (the episode freezes at success) |

Task language comes from the episode itself: RoboCasa samples the scene per
seed, so `backends.robocasa.spec` reads the language from a reset of that seed before
composing; RoboLab uses the task's upstream default instruction. A RoboLab seed does
not change the scene: the environment's only reset event is `reset_scene_to_default`
and no RoboLab-120 task adds one, so every seed starts from the same authored layout.

## Motion

Both backends drive their arms in `ora/environment/backends/{robocasa,robolab}/motion.py`;
they differ in what runs. RoboCasa plans on the shared motion base, RoboLab has no
trajectory optimizer and no global route.

**RoboCasa: admitted joint paths.** The PandaOmron's arm part of the
HYBRID_MOBILE_BASE controller is robosuite JOINT_POSITION
(`robocasa_omron.ARM_CONTROLLER`, kp 400) instead of the shipped OSC_POSE
delta; base, torso and gripper parts stay as shipped. Its input is a delta
from the measured joints, so the IO layer sends the remaining difference to the
held target and RoboCasa's all-zero settle action at reset still holds the
arm. Admission (`session.arm_path_plan`, `backends/robocasa/shared_motion.py`)
turns a target into the joint path that will run, on the shared motion base. The
PandaOmron's arm is the Panda the shared planner already holds (robosuite's arm and
`PandaGripper`), so it plans as `franka_panda`, from `robot0_link0`, the frame that
moves with the base and the lift; the session starts by checking the planner's
forward kinematics against the simulator's hand (`fk_parity`, 0.5 mm and 0.0005 rad
on 22 joint rows, 2026-11-01) and refuses a planner that disagrees. free_space takes
the planner's collision-free route; precision follows the straight grip-site line,
solved by the shared IK every 20 mm with each solution seeded from the previous and
the joints interpolated between them, unless it turns the hand more than 15 degrees,
when it takes the route too; contact_approach always follows the line.

The collision world is `public_depth_voxel_v1`, rebuilt for each request from both
public cameras at that moment, so after `move_base` or a lift the next plan is in the
new frame and the new view: no simulator state enters it, the MuJoCo contacts the
previous check read are gone. A line solves its IK against a blank world, for speed,
and is then checked in the real one: the arm's links along the whole line, in the
world with no goal clearing, and on precision the whole robot, in the world a plan
would see; 1 cm of depth is tolerated, and contact_approach may press with the hand,
fingers and a held load. What the cameras cannot see is free. That covers the Omron
base and torso unless they are in view, and neither the arm's own base nor anything
else unseen is checked: the measured stops below end such a move. The grid is the
Panda's tabletop box, so nothing is modelled more than 0.25 m below or 0.75 m above the
arm mount, or more than 0.6 m to either side. The worker runs in its `kitchen` scene
(`ORA_PLANNER_SCENE`): a goal below a seen surface goes to the solver instead of the
tabletop's `goal_below_support`, because the counter above a drawer is no floor for a
hand in front of that drawer. The surface's slab stays under the goal clearing, so a
hand driven into a counter is still refused.

The public receipt carries `admission_refusal`
(`ora/agent/session_core/refusal_detail.py`: path kind and, for a line, `path_fraction`,
`penetration_m` and `robot_part` of arm or gripper), never scene identities. A line whose
inverse kinematics stops says why: `cause` `ik_unsolved` (no solution at that cut, with its
`ik_residual` where the solver gave one) or `joint_branch_jump` (a joint, numbered from 1,
jumping `jump_rad` past `bound_rad`, `LINE_MAX_JOINT_JUMP_RAD`, between cuts: the IK changed
branch), the cut's grip-site pose `at_pose`, and `endpoint`, the target asked on its own from
the measured base pose and torso height (`reachable`, `reachable_only_through_the_world` or
`unreachable`): `reachable` means the line, not the target, is what fails. A target more than 1.05 m from the arm
mount, at the nearest mount height the torso lift allows, is refused first
(`target_out_of_reach_from_base`, with `distance_from_mount_m`, `reach_m` and
`nearest_mount_height_m`); failed paths are refused with
`target_unreachable_at_base_pose`, `target_in_collision_at_base_pose` or
`path_in_collision_at_base_pose` and zero native actions (`target_in_collision_at_base_pose`
when the arm reaches the pose in free space but not in the world;
`motion_planner_unavailable` when the planner itself failed, and nothing moves). The admitted path is timed at the grip site for the move's speed and
checked for an unrequested wrist turn before anything moves
([motion speed](motion-speed.md)), and the executor (`MotionRun.follow_path`)
streams its knots, one per tick, each tick advancing at most 0.05, 0.025 or
0.012 rad per joint and 30, 12 or 8 mm of grip-site travel by interaction; it has arrived once the measured joints stay
within 0.005 rad of the path's end for 3 ticks and the grip site is within
5 mm and 0.05 rad of the target (else `endpoint_residual_exceeded`). Three
stops mean the arm is held back by something and end the action as `blocked`
with the arm held where it stopped; the receipt's `endpoint.stop_reason` names
which: joints trailing their moving target by more than 0.2 rad for 3 ticks
(`joint_lag`); once the last target is sent, joints at rest short of it, 3
ticks each moving every joint less than 0.001 rad while one is still more than
0.005 rad from it (`settle_stalled`, the rule of
`controllers/motion_progress.py`'s `JointBlockedMonitor`); or joints still
unsettled 20 ticks after it (`settle_timeout`). After either settle stop the
grip-site check decides: on target is `reached`, else `blocked`. Each of the
three also records the joint then farthest from its target (`lagging_joint`,
1 is the base joint, and `lag_rad`) and, for `joint_lag`, how far along its
knots the target was (`path_fraction`). Before 2026-11-06 the arm waited the
full 20 ticks however early it came to rest (in 111 audited runs, 581 of 669 waits
that ran out had been at rest, then waited a median 12 ticks more), and the lag
stop was named `blocked`. contact_approach stops on a gripper contact-force
threshold (`contact_detected`, `reached` wherever it stopped; the force is never
published); 150 ticks is the action budget. Measured 2026-11-01: a 4 cm precision move takes
12 ticks, a 60-degree reorientation 36, with endpoint errors under 1 mm; a plan takes
about 0.45 s, a 20-cut line about 0.8 s of IK plus 0.5 s per check.

When the arm alone cannot reach at the measured torso height (`target_unreachable_at_base_pose`),
the lift is admitted on the whole move (`session._lift_plan`). The shared IK is asked about the
target every 0.04 m of the lift's 0.34 m travel, in a blank world, at each height and 10 mm
(`TORSO_TOLERANCE_M`, where the lift may stop) either side (`shared_motion.torso_heights`); from
the nearest of those heights, at most `TORSO_ADMIT_HEIGHTS` of them, the path is admitted as at
the measured height, but from the lifted arm mount and in the world seen from there (a stand,
below): the line's chained IK, joint jumps and world check, or the global route, and then the
path is paced at the call's speed with the wrist check and its jaw twin, as the real call paces
it. The first height that passes is offered, and a dry run shows that path (`lift_then_arm`,
the torso a servo, the arm planned). When none passes, the refusal (the arm's own, from the
measured height) adds `torso_admission` to the receipt's `reach_check`: the measured height
(`measured_m`), the heights the arm reaches the target from (`reaching_m`) and, for each height
tried, its code with where and why it failed (`cause`, `joint`, `robot_part`, `path_fraction`,
`penetration_m`, ...) or the pacing's code (`stage` `pacing`). It is evidence: the policy's
projection and a dry run's answer leave it out, and the sparse-motion qualification record keeps
each check's `reach_check` and `torso`. A line whose target the arm reaches from here (`endpoint`
`reachable`) is refused without a lift: the lift moves the line with the arm and does not mend
it. The torso moves first with the arm joints held (at most 0.05 m per tick, 30 ticks), stays
there, and the real call then plans the path again from where the lift stopped, in the world
seen from it; a target with no path from there ends as `not_reached` with the torso at its new
height. The lift itself is not checked against the world. An arm that reaches the target in
free space but is blocked by the world does not lift. The receipt carries a `torso` block
(`from_m`, `to_m`, `target_m`, `stop_reason`, `arm_path_replanned`), which the public effect
carries as `detail.torso` and its summary line as "torso moved from ... to ... m first", and the
height is public as `mobile_base.torso_height_m`. A torso-only command lifts the hand about 25 mm
per tick and settles about 3 mm short of its target.

**Stands.** A lift or a base move is asked about before it runs in the frame the arm mount would
have there: a `Stand` (`ora/environment/controllers/base_placement.py`), whose mount is the
measured one carried by the base's move and the lift. Its world input
(`shared_motion.stand_world`, `world-inputs/round-R-step-S-stand-NNN.npz`) holds the same public
depth with the robot's own pixels taken out, because the robot is no longer where the cameras saw
it; the planner then removes and carves the arm at the stand as it does at any start. The robot's
own pixels are the ones the planner's own world would take for the robot at the measured state
(`shared_motion.own_body`, a `base_placement.OwnBody`): within the planner's collision spheres at
the measured joints plus the world's 3 cm robot margin (`robot_point_margin_m`), inside the hand's
box (the world's `hand_box_m`, ending 1 cm past the fingertips at `carve_box_tip_m`) or the base
link's (`robot_spheres.BASE_LINK_BOX_M`, `panda_link0`'s collision mesh, which its two small
spheres leave mostly out), inside the spheres the planner would attach for a load the jaws hold
(`shared_motion.held_spheres`: `depth_world.Scene` of this moment's world input) or those of the
depth linked to that load (`base_placement.linked_load`, below), or within 3 cm of the base where it
stands. The spheres are the host's copy of the planner's
(`ora/environment/controllers/robot_spheres.py`: cuRobo V2's `franka.yml` with ORA's finger shift
and 4 mm buffer, placed by the URDF chain from `panda_link0`, the fingers locked open); at session
start the set's hash must be the one the planner names in its ready answer
(`collision_spheres_sha256`) and the host's chain must land on the planner's hand at the FK parity
rows (`shared_motion.own_spheres_check`, in `motion-planner/binding.json` as `own_spheres`), or the
session refuses the planner. On the 2026-11-06 qualification's world inputs (control steps 0 and
159, where nothing but the robot lay within 0.20 m of the arm) this takes out every point within
0.20 m of the arm, the shell that 0.10 m capsules left (in which the lifted arm met its own
picture, `panda_link3` 11 to 15 mm deep, and the torso raise was refused) included. Until
2026-11-06 the own pixels were capsules along the link origins, 0.15 m about the arm and 0.12 m
about the hand, which took the scene next to the hand out too: in the RoboCasa smoke
`rc-ab-952eeed23` all 2,970 wrist points within 5 cm of a refused strawberry target (GarnishPancake,
step 3422), which now stay, so a stand next to the hand was called clear of what lay there.

**Base placement.** A refusal with `target_out_of_reach_from_base`,
`target_unreachable_at_base_pose`, `target_in_collision_at_base_pose` or
`path_in_collision_at_base_pose` (after the lift, where one was tried) carries
`admission_refusal.base_suggestions` and `base_search` (`session._suggested`,
`shared_motion.base_search`): the base poses from which this same move is admitted. A line a base
move does not mend gets none: one whose target the arm reaches from here (`endpoint` `reachable`),
and one whose gripper enters the world (`robot_part` `gripper`: what lies around the target stays
where it is). The search is the shared motion base's (`controllers/base_placement.search`);
RoboCasa supplies its robot (`shared_motion.MOBILE_BASE`: the footprint `BASE_FOOTPRINT_BOXES_M`
from robosuite's `omron_mobile_base.xml`, the column's top rising with the lift; the base servo's
gains, caps and tolerances; the 1.05 m reach; the shoulder 0.333 m above the mount), its own body
(above) and its planner. The lift is no stand's to choose (`torso_range_m` None): `move_base` does
not move it and no action sets it, so every stand keeps the measured torso height and a suggestion
names none (until 2026-11-06 a stand chose a height around the one that levels the shoulder with
the target; 18 of 107 first suggestions in the smoke named a height no action then executed).
Stands are arm-mount positions 0.40 to 0.70 m (and the current distance) from the target on
bearings within 90 degrees of the current one, each with the base's current yaw, the yaw facing
the target and the one between, within 1.5 m of the base, the target within the reach. A stand is
kept when its footprint, grown by 5 cm, holds no public depth point (the floor, the robot's own
body and what lies within 3 cm of the base where it stands are left out; unseen space is free) and
at least a tenth of the floor under its pedestal (5 cm cells) was seen, by a camera ray that ended
on or passed over it below the pedestal's top, or lies under the base now (the stands behind
DeliverStraw's cabinet face had at most 6 % seen, those in front 10 to 81 %). Stands are ranked by
the base's travel plus 0.3 m per radian of turn, plus 0.5 m for a nudge in place (less than 5 cm
and 15 degrees; 25 of the smoke's 107 first suggestions were turns of 5 to 15 degrees in place);
the cheapest 48 are asked of the shared IK in one blank-world request. Then, in that order, a stand
at least 0.15 m or 20 degrees from those kept must pass `move_base`'s own check on the way there
(its footprint and its carried arm, below; `base_search.arm_blocked` counts those the carried arm
fails) 1 cm stricter than the check itself (`Search.suggestion_slack_m`: the footprint grown 2 cm,
the carried body holding no depth point at all), so a move to the printed, rounded stand, or one a
few millimetres off it, is admitted too (in smoke `rc-ab-6864e57de` a move 3 mm off its suggestion
was refused, its load 1.6 mm deeper than the tolerance), the shared IK must reach the target from each pose the servo may stop at around it (its
0.03 m position tolerance either way along x and y and its 2.3-degree yaw tolerance either way, one
blank request of six queries; the base stopped 1.2 to 1.5 cm off in the smoke, and a target at the
edge of reach was then out of it), and the move itself must be admitted from there as its real call
admits it (`session._admitted_from`): the same path, the straight line for a contact approach or a
precision move that turns the hand at most 15 degrees and the global route otherwise, in the
stand's world, then its pacing at the call's speed with the wrist check and its jaw twin. Up to four
stands are admitted, until two are kept; a stand whose admission the planner could not answer is
not kept. Each suggestion is a ready `move_base` target (`x_m`, `y_m`, `yaw_deg`) with
`distance_m`, `turn_deg` and `arm_path` `admitted`; `base_search` counts the stands, those whose
footprint is clear and floor seen, those asked, those the IK reached (`ik_reached`), those the
carried arm would block (`arm_blocked`), those admitted-checked (`checked_in_world`) and the
search's `max_travel_m`; the receipt's record also keeps `floor_unseen` and `arrival_unreached`.
Until 2026-11-06 a stand was checked by the target's IK alone in the stand's world and a joint-space
plan the real call does not run (`arm_path` `planned`/`not_planned`, which a not-planned stand
could rank first with): a contact line refused for an IK branch jump was refused the same way at
the suggested stand. The search's questions bypass the plans a dry run keeps, its answer is kept
for the moment (round and control step) and the move (path kind, interaction and speed), so a dry
run and the real call that follows say the same, and its record is
`motion-planner/base-placement.jsonl`. It costs the public depth's screens on the host (about
0.35 s), one blank IK request of up to 48 queries, a six-query band request per stand that passes
the move's check, and up to four admissions (a stand world of about 0.5 MB each, and the
planner's world built from it). Only public inputs enter it. `recover_motion` returns to arm joints: previous to those measured at the
start of the last admitted move_pose, home to those at episode start, along a planned
route with the torso held.

**RoboLab: the shared planner, then the Cartesian servo.** A `free_space` or
`precision` move of more than 50 mm or 15 degrees plans
(`backends/robolab/shared_motion.py`): cuRobo V2 for the robot `franka_robotiq`
(V2's Franka arm, the Robotiq 2F-85's collision spheres fitted to RoboLab's USD on
`panda_hand` and the finger links, `planners/curobo_v2/robots.py`), in
`public_depth_voxel_v1` built from the last observation's two cameras. The path is
timed and wrist-checked, then streamed as joint targets, one row per native action
(`bridge.step_joints` has the registered IK term apply them for that step). A planner
refusal moves nothing; a planner that cannot answer (no current observation, an error)
leaves the move to the servo, and `motion.planner` says why. At session start the
planner's forward kinematics must land on the simulator's TCP within 2 mm and 0.02 rad
(`fk_parity`; the arm's URDF chain matches 328 recorded poses to 0.005 mm,
`tests/data/fk-parity/franka_robotiq.json`).

The servo ends every planned path and takes contact approaches and short moves. Each
tick moves RoboLab's absolute differential-IK target a bounded step, which a requested
end-effector speed lowers ([motion speed](motion-speed.md)), toward the next waypoint
on the measured Robotiq TCP; a free-space target the planner could not answer, beyond
a short horizontal radius, is reached by lift, traverse and descend. A per-waypoint
stall window ends a leg as `controller_stalled`, contact_approach stops on a
contact-force threshold, and a target the shared IK does not solve is refused
(`target_out_of_reach`; with no IK answer, a target more than 0.95 m from the robot
root).

Both executors keep the name of the stop that ended an arm move in the
receipt's `endpoint.stop_reason` (RoboLab: `reached`, `contact_detected`,
`no_progress`, `budget_exhausted`, `episode_ended`; RoboCasa as above, with
`settled` for an arrival). Since 2026-11-06 the shared projection
(`ora/agent/session_core/stop_detail.py`) delivers it as
`detail.endpoint.stop_reason` and a RoboCasa base move's as
`detail.base.stop_reason`; a RoboCasa stop's `lagging_joint`, `lag_rad` and
`path_fraction` arrive as `detail.arms.right.stop_evidence` (`lag_deg` on a
degree surface). The knot counts and ticks stay in the receipt, and no force
crosses. Before, a contact stop read only as `reached` (in 111 audited runs: 502 contact
stops, 397 of them more than 10 mm short), and the policy could not tell the
three `blocked` stops apart. The `action_summary` line names the stop (see
[policy I/O](policy-io.md)).

Every constant is single-owner in
`ora/environment/controllers/robocasa_omron.py` and `robolab_droid.py`, and
every number the tool help states is registered against it in
`tests/data/published-limits.json`.

The jaw loops use the shared stall rule (`ora/environment/controllers/gripper_stall.py`):
a close that stops on an object reports `not_reached` with
`gripper_stalled_before_target` and `gripper_outcome.measured_aperture_mm`, as
on LIBERO and RoboDojo. On RoboLab `recover_motion` returns to a measured
pose (the start of the last arm motion, or the episode start); on RoboCasa to
measured joints, as above. With no recovery state it is `plan_rejected` /
`recovery_state_unavailable`.

RoboCasa adds `move_base` (motion v3 action `{"target": {"x_m", "y_m",
"yaw_rad"}}` in the world frame). The policy tool `move_base`
(`ora/agent/toolkit/tools/move_base/`, capability `mobile_base_navigation_v1`,
after ASPIRE's `behavior.navigate_to_pose`) takes an absolute world pose or a
relative forward/left/turn move; a target more than 6.0 m from the current base
pose is refused before submission. The servo drives the yaw-hinge anchor at up
to 0.6 m/s and 57.3 deg/s (about 44 control ticks per metre and 39 per quarter
turn, measured 2026-10-18), adds the measured slide-joint deadband and
stops on arrival after 4 settled ticks with the arm joints held, on a sustained base
contact (`failure_code blocked`, `base.stop_reason` `base_contact`: a
bumper-equivalent outcome with no force value) or on a stall
(`controller_stalled`, `base.stop_reason` `no_progress`): every tick outside the
arrival tolerance, the servo's progress measure (anchor distance plus 0.3 m per
radian of heading error) gained less than 0.005 m over the last 10 ticks, so a
pinned base stops after 10 ticks and 3 braking ticks. Before 2026-11-06 the
check ran once per 30-tick window at 0.01 m, and a stall took 33 to 153 ticks
(in 111 audited runs: 80 stalls, typically 63). Before anything moves, the base's footprint along the path the servo would
drive (`base_placement.servo_path`, checked every 5 cm and 5 degrees), grown by 1 cm, is checked
against the public depth of that moment (`session.base_path_refusal`, `base_placement.sweep`): a
footprint that would hold a depth point is refused with `plan_rejected`, stage `admission`,
`base_path_in_collision` and no native action, its `admission_refusal` naming `path_kind` `base`,
`robot_part` `base`, `path_fraction`, `penetration_m` and the point's `obstacle_height_m` and
`obstacle_xy_m`. The robot's body the base carries is checked along the same path: the planner's
spheres of the arm at its measured joints and of a held load (above), moved with the base, may hold
no depth point more than 1 cm inside them (`Search.carried_tolerance_m`, the depth world's voxel,
as deep as a straight line's links may lie), else the move is refused the same way with
`robot_part` `arm` or `held_object`, and the summary says to retract the arm first. A held load is the
planner's and the public depth linked to it (`base_placement.linked_load`, `held_depth.load_voxels`): the 1 cm
voxels within 0.30 m of the load's spheres' centre (`Search.load_reach_m`) that a chain of voxels, each within two
voxels of the last, links to the spheres. Where the chain runs on past the reach, the load touches the scene (a wall
it hangs beside, the door a handle belongs to): each voxel within the reach goes to whichever it is fewer links from,
the load's spheres or the depth past the reach, and the load keeps its own; the part that stays beside the load
refuses a move that would drag it. A fragment within three voxels of the load's depth whose own chain stays within
the reach joins it. That depth is the robot's own (no obstacle, and dark in a stand's world), and 32 spheres fitted
to it as the planner fits its load (`depth_load.fit_held_spheres`) are carried as the load. So is a mixed pixel at the
robot's or its load's silhouette (`held_depth.edge_pixels`): next to an own pixel more than 5 cm nearer and to a pixel
more than 5 cm farther (`Search.edge_step_m`). The planner's world bounds its load 0.15 m past the jaws and leaves one
that stands on a support to the scene, so in smoke `rc-ab-6864e57de` two bowls held by the rim and lifted kept only the
jaws' slice as their load, and the rest of them, left in the depth, refused 67 base moves as the load meeting itself;
in `rc-ab-6d8eb1acf` a bowl within 2 cm of a wall was all left to the scene with the wall, a 43-voxel piece of a bowl
lay 2.4 cm from the rest, and two mixed pixels at a bowl's edge sat between it and the scene: 6 more such refusals.
Replayed on those runs' world inputs, StoreLeftoversInBowl's 82 refusals by the carried arm or load in
`rc-ab-6864e57de` become 61 admitted moves and 21 refusals (13 by the footprint, 7 by the load, 1 by the arm), all at
the room's walls and cabinets (y -4.51 to -4.54 m and -3.63 to -3.76 m; the ones checked stay put when the hand
moves), and SteamInMicrowave [59] is admitted; of the 7 load refusals of `rc-ab-6d8eb1acf`, 3 are admitted, 3 meet
the room's wall or a step with the arm or the footprint instead, and a static object 0.16 m from a held squash is
still refused. Depth within 3 cm of the load's spheres is its own, as it is for the arm, so a load touching a wall
can be driven that far into it before the check meets the wall. In the 2026-11-06 smoke three base
moves stalled with the hand in a fridge; of the 25 smoke moves a world input replays, the check
refuses three of the four that stalled (the carried arm 19 to 51 mm deep) and one of the 19 that
arrived (24 mm). The floor, the robot's own body and what lies within 3 cm of the base where it
stands are left out, and unseen space is free. A dry run answers with the same check
(`segments[0].footprint` says it passed). The base pose is
public proprioception:
`public_robot_facts.mobile_base` carries `x_m`, `y_m`, `yaw_deg` and
`frame: "world"` (the frame of `move_base`'s absolute target), plus
`torso_height_m` when the backend reports it, in every observation and every
`observe` result. RoboCasa's `observe` result
(`robocasa_mobile_v1/observe.json`, surface revision 2) is
the fixed-base LIBERO one with this optional field added; revision 1 bound the
LIBERO result unchanged and refused every live `observe` as an
`internal_tool_error` (`unknown result fields ['mobile_base']`).

## Episode budget

Both backends report their official step limit as the native-action
allowance (`orchestrator/episode/native_allowance.py`): the RunPlan carries
`runtime.native_action_limit` (bound by the runtime resolver to the task's
official value, never changed by a spec), every receipt counts its
`native_actions` (one per control tick, base moves included), and every turn's
remaining budget shows `native_actions_limit` and `native_actions_remaining`
beside robot decisions, model calls and wall time. Those two count down to
`runtime.official_limit_multiple` times the official limit (a fresh
RoboCasa365 or RoboLab spec runs at 1000, where no limit binds and the two
fields are not shown, as on RoboDojo; `ora spec robolab compose
--official-limit-multiple 1`, like RoboDojo's composer, stops the episode at
the official limit, as a benchmark that counts only successes within it does):
see [the extended budget](configuration-and-evidence.md#official-limit-and-the-extended-budget).
The Harness `max_sim_steps` cap is not folded into these limits; the extended
limit binds.

Neither backend reports the official limit itself to the policy (user decision
2026-11-06, reversing RO-23's delivery: told that the official verdict was
decided, a policy gives up, and the late successes are lost; 57 of the 92
earlier RoboCasa365 successes came late). So at the multiple a fresh spec runs
at, no remaining-budget field counts native actions, and no tool text names one:
`move_pose` and `move_base` say only that every control tick spends one native
action (until 2026-11-06 they named `native_actions_remaining`, which such a plan
never delivers), and `finish_episode` says a round ends at official success or
when an allowance in `remaining_budget` is spent, and that the task's official
step limit does not end it. `test_parameter_delivery_guards.py` checks every
surface's tool texts against the budget fields its backend's plan delivers.

RoboCasa's environment is built with `ignore_done=True`, so the worker holds
the stop itself (`MotionRun._step`): once success is latched, the extended
horizon is reached or the run is aborted, no further control step runs. An action cut
off by that stop ends there as `not_reached` with
`episode_terminal_during_execution` and the episode ends as
`robocasa_horizon_exhausted`; a `move_base` cut off this way skips its braking
ticks, so its `native_actions` never exceed the steps that remained. A success
`_check_success` reports after the official horizon is never latched: it ends
the episode the same way and is recorded as `success_after_official_limit`.
RoboLab's Isaac Lab environment truncates at the extended length; a success past
the official length is recorded the same way, and `first_success` stays null on both.

A RoboLab `official-verdict.json` also carries `robolab_score`
(`ora.robolab-subtask-score.v1`), RoboLab's graded Score as its leaderboard
computes it. The worker reads RoboLab's own `SubtaskCompletionRecorder`
(`get_current_subtask_info`) after every native action and keeps:
- `final`: the last read;
- `at_official_limit`: the read on the action that reached the official limit;
- `official_score`: 1.0 on official success, as RoboLab's results code counts
  one, else the score at the official limit, or the last score when the episode
  ended before it.

A task RoboLab tracks no subtasks for scores null. The score is evaluation-only:
no observation, receipt or tool result carries it.

## Policy surfaces

| Surface | Derived from | Differences |
|---|---|---|
| `robocasa_mobile_direct_v1` | `libero_placement_direct_overhead_v1` | adds `move_base`, `robocasa_rgbd_v1`, `mobile_base_navigation_v1` and `public_robot_facts.mobile_base` in the `observe` result; drops the IK reach annotations; `toolkit_only_v1` |
| `robolab_direct_v1` | `robocasa_mobile_direct_v1` | no base; `robolab_rgbd_v1`, `robotiq_2f85_tcp_axes_v1`, `contact_graspnet_robotiq_2f85_tcp_compiler_v1`; its own `observe` result with the Robotiq end effector; `toolkit_only_v1` |

Both surfaces declare `toolkit_only_v1`: their tools, `move_base` included, run on the
toolkit (`ora/agent/toolkit/tools/`) instead of the runtime handlers, with the same text,
and hand over the action the handlers handed over, which the environment plans as it
executes it ([toolkit execution](toolkit-execution.md#a-surface-whose-environment-plans-its-moves)).
Such a run also writes `native/toolkit-io` and `state/tool-deltas.jsonl`, so
`ora toolkit replay RUN --at N --tool T` can rerun one of its calls.

On RoboLab, `contact_graspnet_robotiq_2f85_tcp_compiler_v1` selects the
compiler identity `contact_graspnet_to_robolab_robotiq_2f85_tcp_v1` (the Panda
grip-site compilation bound to the Droid Franka Robotiq descriptor), which
`grasp_plan` publishes whole as `compiler`. RoboLab's `grasp_plan.json` has
that result's `compiler.controlled_point` as `robotiq_2f85_tcp` in place of the
inherited Panda `robosuite_grip_site` (a binder rebound it when the surface was
layered); every other compiler field keeps the LIBERO-family schema, and the
LIBERO and RoboCasa surfaces keep the Panda constant. A RoboLab surface without
that constant refused every `grasp_plan` result with `internal_tool_error`
(`compiler.controlled_point differs from result const`).

RoboCasa and RoboLab declare `grasp_pose_world_axes_v1` (RO-27), as every
default with `grasp_plan` does (user decision 2026-11-06): every candidate's
`pregrasp_pose` and `contact_pose` also carry `approach_world` and
`jaw_line_world`, the pose's rotation applied to the controlled point's local
approach and closing axes (RoboCasa's grip site: +Z and +X), rounded to 6
decimals. They are the two fields of `move_pose`'s `world_axes` mode, so a pose
copies into such a target as it is. Each `grasp_plan.json` result schema
requires both; the compiler and the stored candidate records are unchanged
([toolkit execution](toolkit-execution.md)). The trace-privacy
`contact_pose` exemption (`agent/evidence/trace_privacy.py`) admits the pose with
exactly this pair.

RoboLab's `observe` result
(`robolab_direct_v1/observe.json`, surface revision 2) is
the fixed-base LIBERO one with the end effector the public-facts projection
publishes for the Robotiq TCP: `controlled_frame: "robotiq_2f85_tcp"` and a
`gripper` holding `measured_finger_joint_rad` (delivered in degrees as
`measured_finger_joint_deg`), `measured_aperture_mm` (the separation of the
inner fingertip-pad faces, from the measured link poses),
`measured_opening_normalized` (that separation over the 85 mm maximum),
`measurement_semantics`, `grasp_evidence`, and `commanded_opening_normalized`
and `command_target` beside them. Revision 1 bound the LIBERO result
unchanged, whose end effector is the Panda grip site or the ALOHA TCP, and
refused every live `observe` as an `internal_tool_error`
(`controlled_frame is outside the result enum`); the LIBERO surface keeps that
result.

Both surfaces are file-backed, written out from their layered manifests with
the same text ([flattened defaults](toolkit-surface.md#surfaces-written-out-from-layered-manifests)):
each tool's text is the `<tool>.json` beside the manifest, edited in place, or
the tool's own `description.json` where they are equal. That text replaces the
inherited LIBERO claims
(CuRobo routes, IK reach checks, the LIBERO tick, joint-space recovery) with
what each backend's handler does.

RoboLab declares the IK reach capabilities (`grasp_plan_ik_reachability_annotation_v1`,
`move_pose_reach_along_request_v2`, `pose_primitive_reach_along_request_v2`,
`reach_nearest_orientation_v2`): its reach check is the shared IK in the planner's
world (`backends/robolab/reach_check.py`, `ik_model robolab_reach_ik_depth_world_v1`,
`matches_move_pose_admission` false, as on LIBERO), over the `align_plan` bridge. It does
not declare `read_only_ik_reachability_v1` or the `align_plan` tool. RoboCasa declares
none of these: it solves IK inside its own move_pose admission. Both answer a dry run on the same bridge
(`action_plan.py`, `ora.action-plan-request.v1`): what the move would run, stopped before the first step, and the plan is
kept for the real call. RoboCasa's `move_base` and any move RoboLab leaves to the servo have no path until they run, so
the answer draws the commanded line, unplanned; RoboCasa's carries its footprint check. A refusal the environment
answers without a message is said by the shared dry-run record in the receipt's words (the planner's first sentence and
the refusal detail's, `ora/agent/toolkit/tools/_refusal_words.py`), in its `summary` and the picture's legend.
`tests/data/backend-parity.json` records what is left as `backend_missing` exceptions
`P0090-B01` to `P0090-B06`, confirmed by the operator on 2026-10-18.

Embodiment cards: `ora/policy/embodiments/panda-omron/` and
`droid-franka-robotiq/` (descriptor, compiled card, wording review). The RoboCasa
surface binds the PandaOmron descriptor v2 and its card
`manual-public-geometry-v2.md`; v1 keeps its bytes for the runs sealed with it
(`moved_sources.MOVED`: the same robot geometry). v2 adds three facts the card
states (RO-15, RO-26): the arm mount (`topology.manipulators[].mount`: directly
above the published base `(x_m, y_m)` at world z = 0.7 m +
`mobile_base.torso_height_m`, the lift's 0–0.34 m; robosuite
`omron_mobile_base.xml`, `fixed_support` 0.50 m plus `support` 0.20 m), the
Runtime's reach admission (`geometry.reach_admission`: 1.05 m from that mount at
the nearest height the lift allows, `ARM_MAX_REACH_FROM_SHOULDER_M` and
`TORSO_RANGE_M` in `environment/controllers/robocasa_omron.py`, applied by
`session.workspace_refusal`) and the wrist camera's fixed mount
(`sensing.cameras[].mount_in_controlled_frame`: optical centre
`[0, 0.05, -0.097]` m in `right_grip_site` axes, looking along +approach, image
right -closing, image down local -Y; robosuite `eye_in_hand` through the
PandaGripper frame, and equal to the recorded wrist calibrations).
`tests/ora/agent/composition/embodiment/test_public_geometry_delivery.py` holds
the descriptor's reach and lift range to those constants. Every compiled card
(all backends) also says that local +approach points out of the gripper, from the
palm toward the fingertips, names the fingertip end of the finger extent and the
pads' span along +approach instead of a signed interval, and gives each pad's
thickness along the jaw-closing line and width across it (RO-25, RO-26). The RoboLab
descriptor's finger and pad extents were measured from the loaded Isaac stage;
both workers measure the loaded robot geometry on every reset and compare it
with the descriptor (`loaded_robot_geometry_before_policy_v1`) before the policy
starts, writing `native/robot-geometry/round-N.json`.

## Task contracts

`ora/policy/environments/{robocasa,robolab}/task-contracts/registry-v1.json`
bind one reviewed public completion contract per admitted task, scoped to the
source pin (RoboCasa contracts by the robocasa pin). RoboLab's contracts were
generated per predicate family by `ora dev robolab-tasks generate` (2026-11-05)
and each entered `tasks` through `ora dev robolab-tasks promote` once its
review's nine checks passed; a generated contract still under `pending_review`
is withheld as `contract_review_pending`, its review keeps `status:
needs_review`, and launch preflight fails `robolab_task_contract` for the task
([RoboLab tasks](robolab-tasks.md)). Each review binds the
predicate sources and a private seed-0 witness: direct state edits or scripted
moves checked by the official predicate, with counterexamples. Notable gaps the
contracts state: a RoboCasa drawer counts as open at about half its depth; the
pick-and-place object must be wholly inside the cabinet with the gripper well
away; navigation ends at the spot in front of the fixture, facing it; the
RoboLab bowl tasks require releasing the objects. The RoboLab stack predicate
does not require release; that scoring gap stays in the private review.

## Qualification

`./scripts/ora qualify robocasa motion` and
`backends.robolab.qualification` run the scripted motion record through the real
worker (motion, gripper, wait, recovery, base moves on RoboCasa, verdict and
close). `ora/environment/definitions.py` admits `motion_action_v3` on each
backend by citing its passing record and sha256; a changed executor needs a
new record. Launch preflight (`backends/robocasa/preflight.py`, `backends/robolab/preflight.py`)
checks the interpreter, source pins, for RoboCasa the timing worker and the shared
planner (it starts one, and the worker reads the interpreter and source root from the
check's record, `shared_motion_planner`) and, for RoboLab, the graphics
compatibility runtime, the renderer device and the task's contract
(`robolab_task_contract`: delivered, or deliberately replaced by a task file).
`ora qualify robolab contract-witness` runs a task's seed-0 contract witness in
the same worker ([RoboLab tasks](robolab-tasks.md#witness-and-promotion)).

## Replay

A RoboCasa365 run records every action vector it sends and seals a replay program;
`ora replay RUN --check OUT` re-executes it and `--scene` keeps the scene of that
re-execution for the run dashboard ([replay program](configuration-and-evidence.md#replay-program)).
The vectors are recorded at `OmronIO.step`, below every motion module, so a paced arm
path, the torso, the gripper and the `move_base` servo, which is steered tick by tick
from the measured base and follows no path, replay the same way and without a planner.
The session the replay rebuilds runs the same reset, the same control step
(`OmronIO.apply`) and the same success latch.

**The seed does not fix a kitchen.** `Counter.get_reset_regions` dedupes its top geoms with
`list(set(valid_geoms))`, and an XML element hashes by its address, so a counter lists
its regions in an order that changes between processes. The seeded generator then picks by
index, and the same draw lands on another region: on `NavigateKitchen` seed 5 a stand
mixer stands 1.65 m along the island from where it stands in other processes, about half
the time, whatever the hash seed or BLAS threads. `OpenDrawer` seed 6 lists no such
counter and came out the same in 8 of 8. So each round keeps the order of every answer in
`native/controller-replay/region-order-round-NNNN.json` (`ora.robocasa-region-order.v1`;
`construction.region_order`). A run only reads the answers: RoboCasa decides as before,
so two runs of one cell may still differ. A replay puts the same regions in the run's order
and refuses a counter that offers others. A run recorded before this has no order: it
replays only if its kitchen comes out the same, which the state hashes decide. RoboCasa
also draws site colours from the global `numpy.random` state, unrecorded, which no
physics reads.

A RoboLab run records every native action's tensor and each physics step's
joint targets as they entered PhysX, and seals a replay program from them;
`ora replay RUN --check OUT` re-executes it in the environment worker on the
caller's card and `--scene` exports its scene trace, both with the worker
inputs under [Running](#running) (see [Replay program](configuration-and-evidence.md#replay-program)).

Bit-exact replay across CPU families is not claimed. A run of either backend sealed
before its per-tick controls were recorded has no program.

## Running

```bash
./scripts/ora spec robocasa compose OpenDrawer --seed 0 --experiment EXP --resource-profile LANE --provider-profile PROFILE --service-report REPORT
./scripts/ora spec robolab compose BananaInBowlTask --seed 0 --experiment EXP --resource-profile LANE --provider-profile PROFILE --service-report REPORT
```

Worker inputs: `ROBOCASA_PYTHON`, `CUROBO_V2_PYTHON`, `CUROBO_V2_ROOT` and
`ORA_TIMING_PYTHON`; `ROBOLAB_ROOT`, `ROBOLAB_PYTHON`,
`ROBOLAB_GRAPHICS_RUNTIME` (the RoboDojo Vulkan compatibility runtime) and
`ROBOLAB_LD_LIBRARY_PATH` (system libraries such as libGLU). A RoboCasa lane is
a Mesa EGL CPU slot (`resource_class robocasa_sim`) whose first `CUDA_VISIBLE_DEVICES`
entry the planner runs on (about 0.7 GiB); a RoboLab lane binds one
CUDA device that must equal `runtime.renderer_gpu`.

A RoboCasa process renders on Mesa's software EGL device whenever software
rendering is asked for (`MUJOCO_GL=egl` with `LIBGL_ALWAYS_SOFTWARE` set or Mesa's
vendor file as the only `__EGL_VENDOR_LIBRARY_FILENAMES` entry). Mesa lists every
DRM render node before its software device, so on a host with `/dev/dri` render
nodes the software device is not device 0 (on an eight-GPU H200 node it is device
8). `construction.build_environment`, which the worker and the language probe of
`backends.robocasa.spec` both call, enumerates the EGL devices
(`graphics_probe.bind_software_egl_device`), takes the first that reports
`EGL_MESA_device_software`, names llvmpipe, softpipe or swrast, or has no DRM device
file and no hardware device extension, makes an OpenGL context current on it and
sets `MUJOCO_EGL_DEVICE_ID` to its index. It does so after robosuite is imported,
whose import asserts the slot's value is one of `CUDA_VISIBLE_DEVICES`, and before
robosuite creates its first context, which reads it again. No such device stops the
process with an error naming every device seen. Each round reset
(`round-resets/round-NNNN.json`, `ora.robocasa-reset.v1`, and the Supervisor's `reset`
event) records the choice as `egl_device` (`ora.software-egl-device.v1`: `index`,
`vendor` from EGL, `driver` the OpenGL renderer, `egl_version`, `gl_version`, the
device's `extensions`, the slot's `configured_device_id` and every device seen), or
`null` when software rendering is not asked for.
