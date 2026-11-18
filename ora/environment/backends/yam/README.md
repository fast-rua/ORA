# ora/environment/backends/yam

## Responsibility

The bimanual I2RT YAM station adapter (environment id `yam`): two YAM v1 arms with
linear_4310 grippers and wrist RealSense D405s, plus two fixed D435i, on i2rt v1.4.0.
Everything here runs inside the YAM interpreter (`YAM_PYTHON`, the plan's
`runtime.python_env`), started by `ora/environment/rpc.py` as the environment worker,
except `cell_adapter.py`, which the real-cell kit (`ora/cell`, `ora cell`) loads in the
same interpreter on the cell's computer, and `sim_cell_adapter.py`, which it loads instead
for a simulated cell (`robot.sim`: no hardware). One planner, one set of checks and one receipt
format serve three rig modes: `sim` (the station model, kinematic), `read` (the real
cameras and encoders; motors never enabled; every motion answered with its plan) and
`on` (the real arms move, each motion after the operator approves it).

| Module | Holds |
|---|---|
| `session.py` | `YamSession`, the worker session `definitions.py` names: reset (with the loaded finger geometry verified against `runtime.public_geometry_descriptor` when the surface delivers public robot facts), observation, execution, status and the null official verdict |
| `action_plan.py` | The read-only answer to a dry run: the kinematic schedule, kept for the real call |
| `motion_exception.py` | Why the station still plans its own motions instead of the shared base, its owner and its expiry (`tests/repository/test_motion_exceptions.py`) |
| `motion.py` | Admission, IK path planning and its fixed-speed timing, the per-tick checks on the ticks as timed (joint limits, table clearance, inter-arm distance; a motion that starts inside a clearance may only leave it), execution and receipts of `motion_action_v3` (a gripper command is ramped and held where the jaws stall); an unreachable target's refusal names the highest height that solves and, on a session with `runtime.reach_query` (`ora play --reach-query`), the nearest position at its height that solves (`nearest_reachable_position`), an inter-arm refusal the two nearest parts |
| `shared_motion.py` | With `runtime.motion_timing` (`ora play --motion-timing`, off by default): the same IK samples timed by ORA's shared TOPP-RA + Ruckig worker under each interaction's caps, with acceleration limits and a jerk-limited start and stop, each straight piece rest to rest; the worker bound from the run's preflight (`motion_timing.bind_preflight`), the station's own timing without one; the receipt's `motion` ([motion speed](../../../../docs/reference/motion-speed.md#yam)) |
| `shared_planner.py` | With `runtime.motion_planner` (`ora play --motion-planner`, off by default): the arm paths and the admission's reach question from ORA's shared cuRobo V2 planner (`yam_linear_4310`), as the simulator backends plan. A free-space target is one collision-free plan in the public-depth world (every calibrated camera's depth, the other arm at its measured joints); a precision or contact target the station's straight line, by the shared IK every 1 cm and checked in that world with the gripper allowed to touch. The cell's calibration goes on every request (the corrected base as the arm's frame, the encoder zero offsets on every joint sent and off every row answered); a start up to 2 degrees past a limit plans from just inside it. The station's timing, per-tick checks and refusal words stay; a planner error leaves the arm to `motion.py`, and the receipt's `motion.planned_by` (evidence only) says which planned it. Bound from the run's preflight (`shared_motion_planner`), checked by FK parity at start |
| `kinematics.py`, `scene.py` | FK, IK and clearance on the station MJCF, and each gripper's finger and pad boxes in its TCP frame (`contact_geometry`, the public descriptor's measurement); the cell model (table, objects, cameras, and a placeable checkerboard for the simulated cell) |
| `rig.py` | The `Rig` protocol and `SimRig` |
| `real_rig.py` | `RealRig` for `read` and `on`: start-up checks (on `on` also the grippers' readings against their recorded limits), guards, the operator's approvals through the cell's broker (`ora/cell/operator/approval.py`), the operator's `home` on SIGUSR1, shutdown; its cameras and buses come from the cell config (`runtime.cell_config`) |
| `devices.py` | The DM motor frames: the passive DISABLE-frame reader (`0xFD`), feedback parsing, the forbidden-frame list and `disable_all` |
| `arms.py` | The one way to take a real arm over through i2rt (`connect`: the gripper's force limit lowered to `GRIPPER_FORCE_LIMIT_N`, the gripper relaxed, motor temperatures recorded) and to let the arms go (`release`); `real_rig.py` and `cell_adapter.py` both use it |
| `operator_words.py` | The worker's questions for the person at the cell, in Chinese (enabling the arms, a motion plan, releasing the arms where they are), as the cell board shows them; `real_rig.py` asks them |
| `cell_adapter.py` | The YAM `CellAdapter` of `ora/cell`: passive reads through `devices.py`, take-over through i2rt (with the gripper model when the gripper limits are known), station kinematics, calibration poses and the facts the cell board shows |
| `sim_cell_adapter.py` | The simulated station's `CellAdapter`: kinematic arms over the shared state, buses that count handles and frames, cameras MuJoCo renders from the truth's model of the cell (with the checkerboard), the station's model and planning for the kit; refuses the real cell's lock |
| `sim_cell.py` | A simulated cell's shared state directory: the truth (`world.json`: the injected base, joint-zero and wrist-mount errors, fixed cameras, table, board, gripper stops), the arms under `flock`, the board, the bus handles; `SimCell`, the simulated person (`ora cell sim operator`), and `recovery`, a calibration against the truth. No MuJoCo |
| `external_load.py` | What an arm is pushed on with: measured joint torque minus i2rt's own model torque (gravity, friction); `LoadMonitor` stops a motion (`external_load_stop`) when a joint stays over its limit, and keeps each motion's peaks (`rig-loads/`) |
| `selfcheck.py` | The start-up self-check `RealRig` runs on `read` and `on` before anything is enabled (motors, temperatures, joints, gripper readings, the table through the wrists, every camera's depth, the calibration's age), each check with its fix; on `on` one refusal lists every failed check |
| `grippers.py` | A gripper gives torque only while it works: `relax` (motor enabled, no torque) when the arms are taken over and at rest, `engage` for a gripper command; a gripper command runs in force control (`take_force_control`, `apply_force`: position gain off, a torque capped at `GRIPPER_FORCE_LIMIT_N` every tick); `limit_force` lowers i2rt's limiter for position-controlled grips |
| `health.py` | `arm_alerts`: a held arm's alerts (stopped control loop, motor temperature, a gripper that keeps pushing), shared by the worker and the cell kit's holders |
| `capture.py` | The cameras' original pixels for the world view and `ora web` (`native/execution-capture`, Control-only): `sim` renders every boundary and its motions; `read` and `on` record what each observation read, with no extra camera read while the arms can move |
| `common.py` | The worker's schemas, rig modes and artifact helpers |

The robot-agnostic parts that used to live here (the calibration file, the cell lock,
occupancy, RealSense capture, camera registration, the depth service, the cell board,
the reset verbs and the calibration motions) are the real-cell kit,
[`ora/cell`](../../../cell/README.md).

## Safety rules the code keeps

- Nothing in this backend can build DM frame `0xFE` (write the encoder zero). The passive
  reader sends `0xFD` (DISABLE, which returns the motor state) and nothing else.
- Every program that touches the buses or the cameras takes the cell lock
  (`ora/cell/lock.py`) first, and refuses a CAN bus that has another reader or carries
  frames. A DISABLE frame to an arm another program holds drops it.
- `on` needs `runtime.operator_present: true` in the RunSpec, refuses to start while a
  process that drives arms is running, and enables i2rt only after the operator approves.
  Every motion is approved too unless `auto_approve`. The worker has no terminal: it asks
  through the cell's approval broker, answered on the cell board (等你批准), with `ora cell
  approve|decline --id ID` or with play's `/approve`, `/decline [reason]`; no answer in
  300 s (enable) or 180 s (a motion) is a decline. A home while a motion waits voids it.
  The broker's auto-approve switch answers motion questions only (enabling and releasing
  are still asked): `session.py` sets it at a real worker's start for that worker when
  `runtime.auto_approve_motions` (`ora play --auto-approve`), and clears it otherwise;
  `ora cell auto-approve on` and play's `/yolo on` set it during a run.
- A stop (tracking error, joint torque, motor temperature, a dead control loop, Ctrl-C)
  holds the arms where they are measured. Closing drives the arms home (rest, else the
  start pose) and releases them there without a question. A blocked way home holds and
  asks whether to release in place; declined or unanswered (60 s), home is tried again,
  up to three times within the close budget (`close_timeout_s` less 15 s). At that bound
  the arms are released in place, as the last question says, because the worker is
  ending and the motors let go without it anyway; the board's 回到起始位 takes them over
  to keep them held. `auto_approve` asks nothing at close either.
- The worker's lock label says it homes on SIGUSR1; on that signal it drives the arms to
  rest and keeps holding them (the episode goes on; the motion in progress reports
  `operator_home`).

## The cell's computer

The cell config names this robot (`robot.environment: yam`, `robot.model:
yam_station_linear_4310_d405`), its buses (`left: can_left`, `right: can_right`) and its
four cameras (`overhead`, `front`: `fixed_world`; `wrist_left`, `wrist_right`:
`end_effector`). Every tool is `ora cell` ([real robot cells](../../../../docs/usage/real-cells.md)):

```
ora cell init --example ~/.config/ora/cell.yaml     # then fill in the serials and robot.python
ora cell init | check | read | record-rest | grippers
ora cell calibrate --method charuco --square-mm N | wrists --role front | arm-poses
ora cell board [--depth-service http://localhost:8770]                     # http://localhost:8765
ora cell depth-service --model auto|promptda|lingbot|inpaint [--max-side N]
ora cell home | hold | float | release | estop | diagnose | stop | kill | cameras | can
scripts/yam-reset [VERB]                             # the same as ora cell VERB; alone: diagnose
```

The cell config is `--cell`, else `$ORA_CELL`, else `~/.config/ora/cell.yaml`; the
calibration file is the config's `calibration` path. A calibration file from before the
split (`~/.config/ora/yam-cell.json`, `ora.yam-cell-calibration.v1`) still loads: point
`calibration` at it; its serials are ignored, since they now live in the config.

## Reset

What the motors do without a controller, from a read-only register survey (2026-10-28):
all fourteen have `TIMEOUT` 8000 and `CTRL_MODE` 1 (MIT). A motor that hears no frame for
its timeout disables itself, so an arm whose controller died falls within that time
(i2rt documents the unit as ms; it may be 50 µs). No state leaves an arm holding without
a live controller, and no reset can catch an arm after its timeout: support the arms, or
keep early motions low over the table.

| What happened | What the arms do | Do |
|---|---|---|
| A motion looks wrong during an `on` episode | moving | Ctrl-C in the worker's terminal (the arms hold), `ora cell home` (or the board's 回到起始位, play's `/home`: the worker drives them to rest and holds), or the arm power switch (the arms drop) |
| The episode should end now, from another terminal | holding | `ora cell stop`: the worker returns the arms to rest |
| The worker hangs and does not answer `stop` | holding while its i2rt thread lives | `ora cell home`: after 20 s it ends the worker and takes the arms over at once; or support the arms and `ora cell kill` |
| A guard stopped a motion | holding the measured pose | read the receipt; `recover_motion home`, or end the episode |
| A motor faulted (over-current, overload, over-temperature) | that arm's loop stops; it falls after the timeout | `ora cell diagnose` to read the faults; they clear on the next enable or an arm power cycle |
| The arms are limp somewhere else than rest | limp | `ora cell float` (guide them by hand; needs a running board) or `ora cell home` (moves them) |
| A bus is down, `ERROR-PASSIVE` or `BUS-OFF`, or frames are stuck | — | `ora cell can` (sudo) |
| A camera streams no frames | — | `ora cell cameras` |
| Not sure | — | `ora cell diagnose`: who holds the cell, bus states, arm faults, distance from rest |

`diagnose`, `stop`, `cameras` and `can` never move an arm. `release` and `estop` disable
motors. `hold`, `float` and `home` take the arm motors over with i2rt; with the
grippers' limits recorded (`ora cell grippers`) the gripper model is used and the jaws
are held where they are, without them only the arm motors are enabled, and no i2rt
gripper calibration runs. `home` moves in a straight joint path at 0.25 rad/s, refuses
when that path meets the table or the other arm, and then holds where the arms are.

The board's reset panel runs the same verbs with a confirmation dialog for every one that
can move or drop an arm; the verbs that need the cell end live view first.

## Depth completion

`ora cell depth-service` (`ora/cell/depth/service.py`) takes RGB, raw metric depth and
the intrinsic matrix and returns completed metric depth (`.npz` over HTTP), so the model
runs wherever a GPU is. The board sends every camera's newest frame and keeps raw depth
whenever the service is slow or down. Models, lightest first: `inpaint` (OpenCV, CPU),
`promptda` (Prompt Depth Anything ViT-S by default, the sensor depth as a metric prompt;
218 MB of GPU memory and 32 to 42 ms a frame at 848x480 on the cell's RTX 3070 Ti,
2026-10-28), `lingbot` (LingBot-Depth, ViT-L). `auto` takes the largest the free GPU
memory allows. Each checkpoint's licence is on its model card.
