# RoboDojo motion and physics qualification

Three qualification-only operators measure RoboDojo motion and scene physics.
None starts a policy model, writes inside a sealed run or is
policy-visible: the motion metrics checker and `control` replay measure
recorded motion and replay recorded controls exactly, the physics scene
probe measures whether each scene behaves under the selected physics device,
and the scripted motion gate drives a fixed case table through the route
every episode runs.
Card and scene-smoke qualification, keyed on the device, are owned by the
[qualification authoring contract](../developer/authoring/qualification.md) and
the [runtime lifecycle](runtime-lifecycle.md#robodojo-physics-device).

## RoboDojo motion metrics and control replay

Layer L0 of the RoboDojo motion v2 design
measures v1 motion under fixed definitions and replays recorded controls
exactly. It changes no executor, starts no policy and never writes inside a
sealed run. Six modules implement it:

| Module | Runs in | Role |
|---|---|---|
| `ora/orchestrator/backends/robodojo/motion/journal.py` | Harness interpreter, no Isaac | Joins `receipts/action-NNNN.json` with `controller/action-NNNN.json{,l}` into immutable tick and per-arm segment records; refuses a trace whose `action_id` differs from its receipt, a native-action count with a gap, and a truncated or half-written line. |
| `ora/orchestrator/backends/robodojo/motion/measures.py` | Harness interpreter, no Isaac | The pinned definitions (`robodojo-motion-metrics.v2`) and their per-run counts. |
| `ora/orchestrator/backends/robodojo/motion/metrics.py` | Harness interpreter, no Isaac | The report and its CLI. |
| `ora/orchestrator/backends/robodojo/motion/replay.py` | Harness interpreter | Replay launcher (run scratch, slot lease, worker configuration, worker process group) and the replay report. |
| `ora/orchestrator/backends/robodojo/replay_kernel.py` | Harness interpreter | Kernel admission: whether this checkout's worker kernel replays the run's physics. |
| `ora/environment/backends/robodojo/control_replay.py` | RoboDojo interpreter | The replay worker: an in-process `RobodojoSession` fed the recorded controls. With `--scene-trace DIR` (the launcher's flag of the same name, one pass only; `ora replay RUN --scene`) it also writes the scene that replay passes through, reading state only. |

### Metric checker

```bash
./scripts/ora qualify robodojo motion-metrics RUN [RUN ...] \
    [--journal-root DIR] [--upto N] [--planned-waypoints FILE] [--object-poses FILE] [--v-cmd RAD_S] [--output FILE]
```

`RUN` is a run directory or its `native/` root. The report
(`ora.robodojo-motion-metrics-report.v1`) carries the definition
version, the definitions, the `v_cmd` used, one metric object per run, the
runs it could not read (`errors`, exit code 1; the others are still measured)
and the pooled metrics. Every ratio is a numerator, a denominator and their
share. `--output` must lie outside every source run; without it the report
goes to stdout. `--journal-root` reads a control replay's traces against the
original receipts. `--planned-waypoints` (`ora.robodojo-planned-waypoints.v1`:
per action and arm, one joint waypoint per native action and optionally
`predicted_in_tick_speed_rad_s`) and `--object-poses` apply to one run.

An *arm-tick* is one native action of an arm whose `<arm>_arm_joint_state` is
in the recorded control. A *speed sample* is the change between consecutive
journaled physics steps over 0.004 s; a tick's first step is taken against the
previous tick's last step when the physics clock is contiguous, never across a
round start. A *segment* is a run of consecutive ticks of one action with one
segment name commanding one arm. The target is that of its first tick; a
segment whose target changes (a planned trajectory or a stall hold) is not a v1
segment. *Global* segments are `move`, `plan_ref` (a probe-planned move),
`precontact`, `retract` and every `recover*`. `local` is its own
class; `gripper`, `wait` and `stall_hold` are not motion segments. Two cohorts
split the arm-ticks and the jaw episodes: *closed_command*, where the tick-end
recorded gripper directive is `close`, and *holding_mode*, v1's form of the v2
holding rule: after a `set_gripper` close that ended `stalled_before_target`,
until the arm's next `set_gripper`, the round end, or the jaw reaching its
floor. Percentiles are the value at index floor(q·n) of the ascending sort.

| Metric | Denominator | Numerator | Constants and source |
|---|---|---|---|
| `v1_baseline_diagnostics.saturation_physics_step` (v1 only) | arm-ticks, with `closed_command` and `holding_mode` subsets | a joint speed sample ≥ 4.75 rad/s | 0.95 × the X5 actuator's 5 rad/s (upstream `robot_config/x5.py:37-45`) |
| `v1_baseline_diagnostics.saturation_tick_displacement` (v1 only) | moving arm-ticks: largest joint change over the tick ≥ 0.005 rad; a round's first tick has no known start and is excluded | change ≥ 0.19 rad | 0.95 × 5 rad/s × 40 ms |
| `v1_baseline_diagnostics.out_of_step.spread_above_0.3` / `_0.5` (v1 only) | constant-target global and local segments with a known start and at least two joints whose target is more than 0.05 rad away | normalized progress spread of those joints above the threshold at any physics step; `changing_target_segments` counts the segments left out | 0.3 from the design, 0.5 from the bottleneck review; not applicable to planned trajectories, whose joint paths are not straight |
| `commanded_speed_violation` | arm-ticks whose ten steps recorded the arm's applied targets, with cohorts | an applied-target step between two of the tick's own physics steps, over 0.004 s, above `v_cmd`; the step from the previous tick's last target is not counted | `v_cmd` 4.0 rad/s, the design's candidate (`--v-cmd` overrides); `commanded_peak_rad_s` gives the distribution |
| `tracking_error` | arm-ticks (v1) or the ticks of planned `(action, arm)` pairs | largest joint \|tick-end measured − waypoint\| > 0.03 rad | v1 has no waypoint, so the recorded endpoint target stands in (`waypoint_source: recorded_control_target`); 0.03 is the proposed ARX `joint_convergence_tolerance_rad` |
| `peak_ratio` | planned `(action, arm)` segments | measured peak joint speed over the planned peak (waypoint steps over 40 ms) above 1.3 | `not_applicable` to v1; 1.3 is the L2 proposed bound |
| `impulse_saturation` | planned ticks whose `predicted_in_tick_speed_rad_s` is below `v_cmd` | a measured joint speed sample ≥ 4.75 rad/s | `not_applicable` to v1 or without a predicted in-tick command |
| `link6_peak_speed_m_s` | arm-ticks, by cohort | p50, p90 and max of the largest link6 displacement per physics step over 0.004 s | — |
| `sag.free_space`, `sag.holding` | `move`, `plan_ref` and `precontact` segments with a known start and a receipt segment record (recovery segments are not transport); `unmatched` counts those without a record | link6 z below min(z start, receipt goal z) − 0.02 m | goal z from `arms.<arm>.segments[].target_pose_xyz_xyzw`; holding when the segment starts in the holding_mode cohort |
| `held_then_lost.holding_mode`, `.closed_command` | closes that ended above the jaw floor + 0.03 (closed_command; the stalled ones also holding_mode) | the jaw at floor + 0.03 at a tick end before the arm's next jaw command (`jaw_floor`), or, with a replay's private object poses, the object nearest the jaw contact centre (link6 +x 0.1501 m) at the close drifting more than 20 mm in the link6 frame (`object_displacement`); `objects` lists each selected object, its distance to the jaw centre and its drift | ARX gripper tolerance; `ARX_X5_GRIPPER_GEOMETRY.contact_baseline_link6_x_m`; `object_frame_check` is `consistent` when the median selected-object distance is at most 0.05 m, and when it is `inconsistent` the object counts are `null` and the numerator is the jaw proxy alone (`object_evidence: withheld`) |
| `arrival` | every pose command (`move_pose`, `move_free_space`); refusals count as not reached | every targeted arm ends within 0.01 m and 0.1 rad of its final commanded pose (arguments, or the last receipt segment of a `plan_ref` move); `executed` gives the same over commands that advanced a native action | ARX arrival tolerances |
| `native_actions.per_command` | commands that advanced a native action, by action kind | native actions per command (distribution) | — |
| `native_actions.per_segment` | arm-independent segments (a synchronized pair counts once): `global`, `local` and `by_segment` | native actions per segment (distribution) | a mean over global and local together mixes two routes and is not reported |
| `native_actions.per_round` | rounds × the task's official native-action limit | native actions used | `NATIVE_ACTION_LIMITS` |
| `timeout_stillness` | pose commands ending `native_action_budget_exhausted` | every targeted arm moved no joint ≥ 0.005 rad in each of its last three ticks | the SDK executor's `controller_stalled` rule, restated in `motion/measures.py` since its `motion_stall.py` was deleted with it |

Changes from the first cut (`robodojo-motion-metrics.v1`, never used as a
baseline): segments split by name only, not by name and
target; out-of-step counts constant-target segments only; sag takes its goal
height from the receipt and covers pose segments only; arrival's denominator is
every pose command; `closed_jaw` became the `closed_command` cohort beside a
new `holding_mode` cohort; held-then-lost gained the closed-command cohort and
selects its object by the jaw contact centre; commanded-speed violation, v1
tracking error, peak ratio and impulse saturation were added. Commanded speed
counts only the steps inside each tick (an early v2 report also counted the
step from the previous tick), and a stalled close that ended at the
jaw floor opens no holding-mode episode.

### Control replay

The replay contract is `control`: recorded controls in, physics out; no IK,
planner or executor decision is re-run.

- Inputs, read-only from the sealed native root: `source-manifest.json` and
  `reset.json` (source revisions, layout and `actual_source_fingerprint`),
  `receipts/action-NNNN.json` (round index, status),
  `controller/action-NNNN.json` (per tick: segment, recorded `control`,
  tick-end measured state including `gripper_feedback`), the physics journal
  `controller/action-NNNN.jsonl` for the host comparison (one line per physics
  step, written through one line-buffered handle kept open for the action,
  `common.JournalLines`, and closed before the receipt hashes it: the same bytes
  the per-step open, append and close wrote; a line's applied position and
  velocity targets and measured state come from one device-to-host copy of
  every arm's articulation buffers, `common.arm_readings`, which holds the
  values the SDK's separate joint, finger and link-pose reads return, bit for
  bit), and `plan.json`
  (runtime, `budgets.max_sim_steps`, source commit). The worker refuses a plant
  whose source revisions, layout or actual source fingerprint (loaded files,
  installed packages, resolved configuration; the device binding is not part
  of it) differ from the run's.
- Round 0 starts from `session.start()`; every later round from
  `session.restart_round` (the session-home joint write and upstream reset the
  run used). Before each tick the session's gripper directives are restored
  from the tick's recorded `gripper_feedback.requested_opening_normalized`,
  because the published `gripper_command_target` derives from them and not from
  physics. Each recorded `control` dict then goes, in order, through the
  unchanged `native_tick` (`session._conformance_tick`) with a physics-step
  recorder in the session's `record_physics` fields.
- The SDK executor of runs recorded on it could rewrite a retained SDK target
  outside any tick: a `motion_stall` hold with no native action left wrote
  `prev_control` directly. To reproduce that, before each tick every target-arm dimension the
  tick does not command (`<arm>_arm_joint_state`, `<arm>_ee_joint_state`) is
  compared with the recorded first physics line's resolved `command`, which is
  that dimension's exact tick-start value (`_fixed_meta_control` copies
  unrequested dimensions unchanged), and seeded from it when they differ.
  Seeds are listed in `held_target_seeds`; support-arm keys are never seeded.
  The host's first divergence carries a `cause_hint`: a held dimension that
  differs at a tick start names an unreproduced out-of-tick rewrite, a
  support-arm key names the support queue.
- The replay counts applied targets that differ from the resolved command
  instead of raising, and omits the original session's second check, that held
  dimensions kept their pre-action targets. The host compensates: it compares
  every applied target of every physics step, held dimensions and support arms
  included, with the original journal line.
- Actions with no native tick (refusals) are skipped and listed. A `plan_ref`
  move replays its recorded controls and is not re-resolved, so the probe that
  minted it is not needed. Read-only probes (`probe_motion`, `reach_check`,
  `align_plan`, grasp IK annotations) and observations advance no physics and
  are not replayed. A round that turns terminal before its recorded end stops
  and records the unreplayed ticks.
- Private object poses are read after every tick through the evaluator's own
  CPU adapter, env-relative. Their frame is recorded as `env_relative (assumed
  = episode_local), unverified`; the worker also records the layout's env
  origin (`frame_probe`), and the host's `object_frame_check` confirms the frame
  on the first replay that selects held objects.
- The launcher refuses a dirty or non-Git checkout and a worker kernel that is
  not admitted for the run. `backends.robodojo.replay_kernel.kernel_admission` compares
  every file of either commit's worker-kernel manifest. A file in the reviewed
  executor-only table (`sparse_motion.py`, `recovery.py`, `motion_adapter.py`,
  `wait.py`, `ik.py`, `contracts/planning_diagnostics.py`, and `motion_stall.py`,
  `staged_contact.py` and `inter_arm.py`, deleted with the SDK executor, each
  with its reason) may change, or be gone, except for the names the replay reaches
  (session constants, `recovery.configured_route`,
  `sparse_motion.gripper_closed_floor`); every other file may only add: each
  top-level name the run's version binds must bind identically (every binding,
  AST without line numbers), `__all__` may only grow and new names may appear.
  In both, module-level statements that bind no name must be unchanged, and a
  reached name bound at neither commit refuses. `session`, `bridge`,
  `round_lifecycle`, `feedback` and `observation` reference the table's modules
  only through reached names or reviewed callers the replay never runs
  (`execute_sparse_motion_v3`, `solve_ik`, `probe_motion`,
  `RobodojoMotionBackendAdapter`); a test holds the
  scan to the table. The status is `identical_kernel`,
  `executor_drift_admitted` (the report then carries a `kernel_notice` that
  the kernel is not the run's and was admitted by AST equivalence) or
  `control_path_changed`.
- Each pass stages its `TMPDIR`/`TMP`/`TEMP` through `host_storage.stage_launch`
  and releases it afterwards, claims its slot lease (a conflict refuses; retry
  when the slot frees), and runs the worker in its own process group, killed as
  a group on timeout or interrupt. The worker's pid and process group are
  written to `launch.json` as soon as it starts. In parallel `determinism`, the
  main thread's SIGINT/SIGTERM handler kills every registered group, so the
  pool threads blocked in `wait` return; the serial path is unchanged.
- Outputs, per pass under `--output`: `worker-config.json`, `worker.log`,
  `launch.json` (replay id, profile, command, run scratch, and the commit,
  worker-kernel digest, kernel admission and replay-worker digest at launch;
  then the exit code and seconds), `environment/` (the session root, with
  replayed `controller/action-NNNN.json{,l}` in the original layout),
  `private/object-poses.jsonl` (qualification evidence, never policy-visible)
  and `replay-worker.json`; then `robodojo-motion-replay-report.json`
  (`ora.robodojo-motion-replay-report.v1`) with the run and Harness
  identities, the metric-definition version, per-pass physics-step
  divergence, the metrics of the original and the replay, and the failures.
  `--report-only` rebuilds it on any machine; outside a Git checkout the
  Harness identity fields are `null`, `rebuilt_outside_git` is true and the
  launched-at-this-commit check is skipped.

| Mode | Passes | Passes when |
|---|---|---|
| `replay` | one | every selected tick replays without a worker error or applied-target mismatch; arm joints, finger joints, the normalized jaw, link6 position and every applied target equal the original's on every physics step; the pass ran at this commit with an admitted kernel |
| `metrics-after-replay` (default) | one | as `replay`, and the replayed journals' metrics equal the original's |
| `determinism` | two, in parallel with `--second-resource-profile` | as `metrics-after-replay` for each, and the two passes agree on every physics step. Private object poses get their own status (`within_bound`, `exceeds_bound`, `unmatched` or `unavailable`) against 1 mm and do not decide the result; if they exceed it, drop metrics use the replay-to-replay spread as their noise floor |

Operator command, on the qualified host from the clean committed checkout with
`.env` sourced for the `ROBODOJO_*` paths (the runtime's `python_env`, normally
`ROBODOJO_PYTHON`, selects the worker interpreter):

```bash
export ORA_EXECUTION_HOST=<host-id>
export TMPDIR=<writable scratch on a partition with free space>   # the launcher's own TMPDIR
./scripts/ora qualify robodojo motion-replay \
    --run outputs/<experiment>/runs/<run_id> \
    --output outputs/<experiment>/diagnostics/motion-replay/<run_id>-<stamp> \
    --resource-profile robodojo-gpu4-d1 --mode metrics-after-replay [--upto N]
```

The worker's own `TMPDIR` is the per-pass run scratch, not the launcher's.
Acceptance replays a few recorded runs in `determinism` mode; a pass costs the
run's native actions in stepping plus one simulator start of about 90 s. No
replay has run on this implementation yet. The design
places the `intent` mode in L0 as well; it is not implemented here (it needs a
candidate executor and re-issues the probes that minted consumed `plan_ref`s),
and neither are the ARX control profile and its validator.

Known limits of a `control` replay:

- The replay turns execution capture off, also for a run recorded with it on.
  Capture renders after physics steps; that it leaves the plant unchanged is
  expected but not yet shown.
- Policy observations are not replayed. Runs of imitate_sorting_sequence and
  make_kong journal a `support_arm0` key; an observation
  may consume the support-arm queue, and the probe clock guard does not see
  that. When the first divergence falls on a support key (`cause_hint:
  support_arm_target`), check this first.

## RoboDojo physics scene probe

The CPU-physics lane ([lifecycle](runtime-lifecycle.md#robodojo-physics-device))
runs nine scenes on IsaacLab `cpu` physics. Whether each scene actually
behaves there (the belt moves objects, articulations initialize and hold, the
liquid exists) is a physics question no card or scene smoke answers, so a
bounded qualification-only probe measures it. It is development evidence for
the lane's design: it admits nothing, binds no card or gate and is never
policy-visible.

| Module | Runs in | Role |
|---|---|---|
| `ora/orchestrator/backends/robodojo/physics_probe.py` | Harness interpreter | Launcher (default runtime of the task on the slot's GPU, slot lease, worker configuration, per-cell scratch, process group), the judge and the report. |
| `ora/environment/backends/robodojo/physics_scene_probe.py` | RoboDojo interpreter | The worker: an in-process `RobodojoSession` reset on layout 0, held with `session._conformance_tick({})`. |

No worker entry point imports the worker module, so it is outside the worker
kernel and its edits stale no card or gate. Per cell the worker:

- resets the task (`session.start()`) and reads the physics mode back once:
  IsaacLab's simulation device, `use_fabric`, `/physics/suppressReadback`,
  `/physics/fabricEnabled`, PhysX `get_overwrite_gpu_setting()` (upstream
  forces `1`, GPU), each physics scene's GPU-dynamics and broadphase
  attributes, and `SimulationManager.get_physics_sim_device()`;
- reads every `PhysxSurfaceVelocityAPI` belt with its surface velocity and
  world bounds;
- holds `--ticks` native actions (default 100 for a conveyor, 50 otherwise),
  stopping early only on the official terminal, and every `--sample-every`
  ticks (and at 0 and the last tick) records every layout instance's world
  position, articulation joint positions, and fluid particle counts, finiteness
  and bounds;
- in a scene with a liquid, records at tick 0 and at the end of the hold
  (`fluid_evaluator_inputs`) the raw inputs of the official fluid predicate
  `is_A_fluid_in_B`: each labelled liquid's particle positions as upstream
  returns them with its layout init pose, and each labelled rigid or geometry
  instance's env-relative pose and local oriented-bbox vertices. They support
  a read-only offline replay of the predicate; the judge does not read them;
- with `--in-place-restart`, also restarts round 1 in place and reads each
  rigid object's `physxRigidBody:disableGravity` and `physics:collisionEnabled`
  before and after (the CPU in-place restart hazard; report only).

The two fluid tasks used `--probe-only-native-action-limit` (their upstream
`step_lim`, 400 and 800) until their admission; the probe-only table
is now empty, and the flag is refused for an admitted task.

A cell passes when the worker held every requested tick (or the official
terminal ended the hold); the IsaacLab simulation device is the declared one
(on CPU without Fabric or suppressed readback); no object fell more than
0.03 m below its reset height, went below the floor, became non-finite or
vanished; and, by scene kind, every rigid object resting on a belt stays within
0.03 m of the belt top and advances at least half the belt speed per native
action (0.1 m/s x 0.04 s = 4 mm), every articulation's joints stay finite and
within 0.05 of reset, and every fluid object has a positive, finite and
constant particle count.

```bash
# From a clean committed checkout on a registered host; one cell per task x slot, all cells in parallel.
ORA_EXECUTION_HOST=<host-id> ./scripts/ora qualify robodojo physics-probe \
  --output <dir> --cell press_by_number:robodojo-gpu4-d1 --cell pick_from_conveyor_by_image:robodojo-gpu5-d1
# rejudge existing cells without a simulator
./scripts/ora qualify robodojo physics-probe --output <dir> --report-only
```

Each cell writes `<dir>/<task>--<profile>/` (`launch.json`, `worker-config.json`,
`worker.log`, `probe-worker.json`, the worker's `environment/`), and the run
writes `<dir>/robodojo-physics-scene-probe-report.json`
(`ora.robodojo-physics-scene-probe.v1`) with per-cell checks and
failures. `--physics-device` overrides every cell's device for a comparison and
is recorded per cell.

The CPU lane's other qualification is the usual one, keyed on the device: a
CPU terminal card (`run.py card --gpu N --physics-device cpu`, measured on a
CPU-physics task and shareable to another GPU with `shared_from`), one CPU
scene smoke and revalidation per lane (`gates/<task>-cpu-gpu<N>.json`), and,
before a `motion_action_v3` CPU cell can resolve, the scripted sparse-motion
gate run on a CPU runtime with its record cited beside
`definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES`. No CUDA card or gate is
re-measured. That CPU record passed 24/24 cases of the v1 table (`swap_blocks`),
so CPU cells now resolve. On the same
host and source all nine scenes passed the physics probe (both conveyors carry
objects at the belt's 4 mm per native action), the CPU card passed on
`pour_liquid_into_cup`, and seven CPU scene smokes passed; `fill_egg_holder` and
`pour_liquid_into_cup` failed the smoke's in-place round restart ("official
layout did not settle stably"), the documented CPU restart hazard. Since a CPU
lane runs one round (rule `robodojo_cpu_in_place_restart`), an unscheduled CPU
smoke no longer restarts in place: it records `checks.round_reset =
not_applicable_cpu_single_round` and `restart: null`, and revalidation accepts
that for an unscheduled CPU runtime only (coverage drops `round_reset` from
`fresh`; the in-place boundary stays under `not_established`). A
layout-scheduled CPU smoke and every CUDA smoke still require the reset.

For intentional control-path changes, add `--allow-kernel-drift` to
`ora qualify robodojo motion-replay`. The default remains strict. This flag
permits `control_path_changed` at launch and report evaluation; it does not
permit a dirty checkout, an unresolvable kernel identity, or a different commit
between launch and report. Both launch identities and the report retain
`kernel_admission.status`, `differing`, and `refused_changes`; the report adds a
structured `warnings` entry and `kernel_notice`. Rebuilding with `--report-only`
also requires the explicit flag. All tick, target, metric and determinism checks
remain unchanged: a physical difference still fails. This is a diagnostic
validation override, not a claim that the kernels are equivalent.

## RoboDojo scripted motion gate

```bash
ORA_EXECUTION_HOST=<host-id> ./scripts/ora qualify robodojo motion \
    --runtime-config CONFIG --output DIR [--gate FILE]
```

The host is `orchestrator/backends/robodojo/qualification.py`; the cases and
their judges are `environment/backends/robodojo/motion_checks.py` and the runner
`motion_qualification.py`. It starts one worker through the production
factory/RPC bootstrap with a host-owned execution state, runs no policy, and
writes `DIR/robodojo-sparse-motion-qualification.json`
(`ora.robodojo-sparse-motion-qualification.v2`); only a passing record is copied
to `--gate`. The former `python -m` host refuses with this command in its error
message.

**Route.** From v2 every arm motion is the one an episode runs. The host plans
each move with the toolkit lib (`MatrixPlanner`) over the worker's read-only
kinematics query and binds a `follow_joint_trajectory`; the worker executes it,
lent `native_action_cap` native actions as a tool lends them. A move is planned
as move_pose plans a compiled request at its default speed: the fast pace and
its settle, the interaction label choosing the route (`lib.motion_plan.route`,
with the default surfaces' staged contact approach), the orientation used
exactly (no jaw-twin choice). The home recovery is planned as return_home plans
it and the retrace as recover_pose does. Jaw commands are the rate-limited
`set_gripper`, holds are `wait`. No case sends `move_pose`, `move_free_space`,
`recover_motion` or a string `set_gripper` target; the worker refuses the first
three before any write (`unsupported_action_for_backend`) since the SDK executor
that ran them was removed.

**Planner, timing and motion profile.** The runtime binds `motion_planner`,
`motion_timing` and `motion_profile`; a configuration without them gets those of
the default baseline (`robodojo_toolkit_v7`; motion profile v2 since 2026-11-15),
and `session.motion_stack_source` records which. Every move is planned at the
bound profile's `fast` (`MatrixPlanner.config`) and `session.motion_profile` names
the profile (`profile_id`, `path`, `sha256`). A
field present but not an object is refused. Before the worker starts, the host
starts the timing worker and writes its passing check to `DIR/preflight.json`,
which the worker binds (`motion_timing.bind_preflight`); without it the worker
would time with `motion_pace`'s fallback. A timing worker that does not start
stops the gate before the simulator. `session.free_route` is `global`: a
free-space move must take the shared planner's path, so a silent fall back to
the joint line fails its case.

Motion matrix, round 0, each case starting where the previous one ended:

| Case | Request | Passes when |
|---|---|---|
| `free_space_translation_8cm` | free space, (-8, +6, +4) cm | reached on the free route |
| `combined_translation_rotation_45deg` | free space, 6 cm and 45 degrees about the approach axis | reached on the free route |
| `orientation_only_small_12deg` | precision, 12 degrees | reached on `line` |
| `orientation_only_90deg` | free space, 90 degrees | reached, stopped, or refused before any action |
| `orientation_only_near_180deg` | free space, 179 degrees | as the 90-degree case |
| `precision_small_local_6mm` | precision, +6 mm z | reached on `line` |
| `contact_approach_small_8mm` | contact approach, -8 mm z | reached or stopped, on `line` (with the contact-path check where the planner runs) |
| `contact_approach_staged_80mm` | contact approach, -80 mm z | reached or stopped, on `staged` |
| `planner_refusal_unreachable` | free space to `UNREACHABLE_POSITION` | refused before any action with `goal_unreachable` (`paced_path_unreachable` on the joint route) |
| `below_tolerance_noop` | precision, the current pose | reached on `line` |
| `dual_arm_synchronized_lift` | free space, both arms +4 cm | both arms reached in one trajectory |
| `trajectory_step_exceeds_limit` | knots: a hold, then joint 1 by 1.5 x 0.2 rad | `plan_rejected`, `trajectory_step_exceeds_limit`, no native action |
| `trajectory_joint_limit` | knots: joint 1 at 4 pi rad | `plan_rejected`, `trajectory_joint_limit`, no native action |

A reached case also needs the `joint_trajectory` strategy and the
`knots_linear_ff` executor, and each targeted arm within the ARX arrival
tolerance both of its last knot's forward kinematics (the receipt's
`pose_error`) and of the requested target. Every executed case needs the
untargeted arm to drift less than 5 mm and a verified command invariant; a
rotation-only case needs link6 to move less than 30 mm, whatever its outcome.

Sequence cases:

| Case | What runs | Passes when |
|---|---|---|
| `recover_home_after_matrix` | home recovery of both arms | reached, each arm within tolerance of its reset pose |
| `gripper_close_holds_pose`, `gripper_open_holds_pose` | rate-limited `set_gripper` | reached by the ramp executor within the gripper tolerance, pose targets unchanged |
| `recover_retrace_after_move` | a 5 cm planned lift, then a retrace | the lift reached; the retrace reached the pre-lift pose |
| `jaw_flip_variants_probed` | `probe_motion` with `parallel_jaw_equivalents` | two variants |
| `probe_consistency_unreachable` | a probe of, then a move to, `UNREACHABLE_POSITION` | the probe says unreachable and the move is refused for reach; nothing runs |
| `official_preterminal_boundary` | the verdict before the limit | no verdict, not terminal |
| `official_terminal_boundary_intact` | 20-tick `wait`s to the official limit | a terminal official Boolean at the limit; a later action refused; the verdict stable on reads |
| `round_reset_isolation` | restart round 1, then a 4 cm planned lift | reset record, terminal revoked, fresh observation, arm home, round-1 counters restarted |

**Removed in v2**, because nothing an episode runs can reach them:
`probe_then_plan_ref_execution`, `plan_ref_survives_gripper_only`,
`plan_ref_expires_after_motion` and the stale-reference check of
`round_reset_isolation` (a probe mints no plan reference since it answers from
the shared planner's IK, and the toolkit refuses `plan_ref`);
`free_space_lift_translate_descend` (the SDK executor's lift, translate and
descend segments: a planned path has none); `contact_approach_large_refused`
(the SDK executor's route refusal: the toolkit chooses the route now, and the
default surfaces stage such a move, `contact_approach_staged_80mm`). The
2-pi-alias monitor is gone with the SDK IK endpoint: a rotation is a planned
joint path whose wrist turn the timing checks (`paced_path_wrist_unwind`), and
the 30 mm bound is all that remains of it. `recover_backtrack_after_move` became
`recover_retrace_after_move`, the recovery the trajectory surfaces run.

**Not measured.** The matrix runs in free air, so the tracking and
growing-error stops, the progress guard and the settle failure, the stops
recorded episodes trip most, never fire; no jaw closes on an object. The card
and task probes and the scene smoke still drive `ora-action.v3` through the
SDK's IK, so their evidence stays on a route no episode runs. The unit tests
drive the whole gate over the fake plant, which has no shared planner: there
free-space moves take the joint route, and only a run on a card exercises
`plan_path`, the timing worker and `contact_path`.
