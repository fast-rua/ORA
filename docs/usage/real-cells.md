# Run a real robot cell

A **cell** is one physical robot installation: the arms, the buses they hang on,
the cameras, the table, and the computer beside them. A benchmark fixes the
scene, the task and the verdict; a cell fixes none of them. Someone owns it,
brings it up, types instructions to the agent one after another in free play,
judges the outcome themselves (`official_success` is null) and must be able to
stop and recover the arms at any moment. This page is for that person.

ORA's real-cell kit (`ora/cell/`, [README](../../ora/cell/README.md)) is the
part every cell needs whatever the robot: the cell config, the calibration, one
holder of the cell at a time, bus and camera checks, camera calibration, the
depth service, the cell board, and the operator's verbs. A robot family plugs
into it with a `CellAdapter`. Today one family has one: the bimanual I2RT YAM
station (environment `yam`, model `yam_station_linear_4310_d405`,
[backend README](../../ora/environment/backends/yam/README.md)). It also has a
simulated cell, which runs the whole kit with no hardware
([below](#a-simulated-cell)).

Everything below is one command, `ora cell`, run on the cell's computer.

## Two files

| File | Who writes it | Schema | Where |
|---|---|---|---|
| Cell config | you, by hand, once per installation | `ora.cell-config.v1` | `--cell PATH`, else `$ORA_CELL`, else `~/.config/ora/cell.yaml` |
| Cell calibration | the `ora cell` tools only | `ora.cell-calibration.v1` | the config's `calibration` path |

The config says what the cell is: which robot, which bus each arm is on, which
camera device plays which role, and the limits you want. The calibration holds
what the tools measured: fixed-camera extrinsics, the table, gripper limits and
the rest pose, and where the installation departs from the robot model: each
arm's base pose (`arm_base_corrections`, a correction in the cell's world),
each joint's encoder zero error (`joint_zero_offsets`, model angle = reading +
offset) and each wrist camera's mount (`wrist_camera_corrections`). Every tool
and the worker apply these to the robot model, so planning, clearance checks
and camera poses use the measured robot; joint readings, the rest pose and the
robot's own encoder zeros stay as the motors report them. `ora cell calibrate
--method hand-eye` measures all of them at once ([below](#calibrate-a-new-cell))
and keeps its fit's report in `bundle_fit`. A measured value never goes into the config, and a secret never
goes into either (model keys stay in `ora providers`). A calibration file from
the earlier YAM tools (`ora.yam-cell-calibration.v1`) still loads; its camera
serials are ignored, since they now live in the config.

## Write the cell config

`ora cell init --example PATH` writes an example to fill in. A complete config,
with what each field does:

```yaml
schema: ora.cell-config.v1               # required, exactly this
cell_id: my-cell-01                      # required: names the cell on the board and to `ora cell`
robot:
  environment: yam                       # required: the robot family's backend (ora/environment/backends/yam)
  model: yam_station_linear_4310_d405    # required: the robot model (geometry, policy composition)
  python: ~/envs/yam/bin/python          # optional: the interpreter with the robot's drivers (i2rt,
                                         #   pyrealsense2, python-can); `ora cell` re-runs itself there
  buses: {left: can_left, right: can_right}   # required: arm -> bus interface, exactly the robot's arms
cameras:                                 # role -> device; roles are the robot model's public cameras
  overhead:    {serial: "123456789012", mount: fixed_world}     # mount: fixed_world or end_effector,
  front:       {serial: "123456789013", mount: fixed_world}     #   as the robot model mounts that role
  wrist_left:  {serial: "123456789014", mount: end_effector}
  wrist_right: {serial: "123456789015", mount: end_effector}
calibration: ~/.config/ora/cells/my-cell-01.calibration.json  # required: where the tools write
safety:
  max_joint_speed_rad_s: 0.6             # optional, a positive number; may only lower the model's limit
  operator_present_required: true        # default true
  foreign_controllers:
    ignore: []                           # command-line fragments of programs you vouch for: they run
                                         #   on this machine but never open the cell's buses
perception:
  depth_completion: http://localhost:8770   # the board's depth service (`ora cell depth-service`); null: none
policy:
  provider_profile: deepseek_flash_openai_low_16k_v1   # recorded with the cell; `ora cell` does not read it
```

`ora cell init` refuses a config that misses a required field, names an unknown
mount or gives a non-positive speed, and one that disagrees with its robot:
`buses` must name exactly the robot's arms, and every camera role must be one
of the robot model's cameras with the mount the model gives it. Today
`max_joint_speed_rad_s` and `operator_present_required` are validated but not
yet applied by any tool; the robot model's own limits hold, and a run on the
real arms needs `runtime.operator_present: true` in its RunSpec.

## From a powered-off cell to a play session

`ora cell setup` walks a new cell through steps 1 to 6 below in one command:
`devices`, `rest`, `grippers` (limits and which way each opens), `fixed_cameras`,
`handeye`, `table` (the table's height from the calibrated fixed cameras) and
`verify` (the table each wrist camera sees through its arm is level and at that
height). It does only what is not done yet, asks before each step that moves
anything (on the terminal, or through the approval broker with `--ask broker`),
and stops at the first step that fails; run it again to go on. `--status` says
what each step's state is (`done`, `missing`, or `stale`: a gripper whose
reading left its limits after a power cycle, a table measured before the
calibration) and what comes next, without moving anything; `--redo STEP` repeats
a step. With `--json` the result is the last line of stdout and stderr carries
one JSON line per event: `{"progress": ...}` and `{"step", "state": "running" |
"done" | "failed"}`. The checkerboard is asked once and kept in the calibration
(`board`).

Each step is one command. It refuses with a reason, and leaves the cell as safe
as it found it. Add `--cell PATH` to any of them, or set `ORA_CELL` once.

| Step | Command | Arms | Writes |
|---|---|---|---|
| 0 | `ora cell init` | not touched | nothing: validates the config against its robot, shows each bus's state and whether each camera is connected |
| 1 | `ora cell check` | unpowered or limp | nothing: no other reader or traffic on the buses, every camera free, one frame each with its share of valid depth; running programs that drive arms are listed |
| 2 | `ora cell read` | read passively (YAM: DISABLE frames) | nothing: joints, faults, temperatures, tool positions, joints against the model's limits and the rest pose |
| 3 | `ora cell record-rest` | read passively; put them at rest first | the rest pose; refused when a joint is outside the model's limits (an encoder zero has moved) |
| 4 | `ora cell grippers` | each gripper opens and closes once, its arm held; then you say for each whether it is open or closed | the grippers' [closed, open] limits, the way you saw them (`--orient` alone checks them again, nothing moving) |
| 5 | `ora cell calibrate --method charuco --square-mm N` | read passively | every fixed camera from one ChArUco board on the table, placed through the wrist cameras; N is one black square as shown, measured |
| | `ora cell calibrate --method wrists --role ROLE` | read passively | one fixed camera from what the wrist cameras see of the table, marked provisional |
| | `ora cell calibrate --method arm-poses` | move together through a few poses near rest, then back to where they started and released | each fixed camera that already has a pose (from `charuco` or `wrists`), registered again on the moving arms; the captures stay under `~/.cache/ora/cell-calibration/` |
| 6 | `ora cell calibrate --method hand-eye` | one at a time, around a checkerboard you place a few times, each motion after you approve it | after you approve the result: the fixed cameras, the wrist cameras' mounts, the arms' bases and their joint zeros, from one fit ([Calibrate a new cell](#calibrate-a-new-cell)) |
| 7 | `ora cell board [--port 8765] [--depth-service URL]` | held only when you ask | nothing: the operator's page at `http://localhost:8765` (forward the port over ssh from elsewhere) |
| 8 | a free-play run on rig mode `read` | never enabled | a play run: every motion is answered with its plan |
| 9 | a free-play run on rig mode `on` | enabled and moved after you approve (board, `ora cell approve`, `/approve`) | a play run |

`calibrate`, `setup` and `grippers` stop on SIGINT (Ctrl-C, the board's 中止, a
supervisor's stop) even when a shell started them in the background, which
leaves SIGINT ignored: they take it back at start and, with `--json`, say so on
stderr (`{"interruptible": true}`). `grippers` and `calibrate` ask for confirmation first (`--yes` skips it) and take
`--no-save` to report without writing. The board's page shows the cell's state,
the live view (the camera streams and one point cloud of the cell, with
completed depth when a depth service runs), the calibration checklist and the
verbs below. The depth service takes RGB and raw depth and returns completed
depth; start it where a GPU is:

```bash
ora cell depth-service --model auto|promptda|lingbot|inpaint [--port 8770] [--host 0.0.0.0] [--max-side N]
```

or run `python -m ora.cell.depth.service ...` under the interpreter that holds
the model. The board keeps the sensor's depth wherever it measured and uses the
model's only in holes near a measurement.

Steps 8 and 9 run the robot's worker with the cell config: the YAM runtime takes
`runtime.rig_mode` (`sim`, `read`, `on`) and `runtime.cell_config` (the config
path) in a RunSpec, and its preflight loads the config. A one-line
`ora play --env yam` entry is not wired yet.

## Calibrate a new cell

The robot model says where the arms, their joints and their wrist cameras are on
paper; a real installation differs. The agent sees through the cameras and moves
the arms in one world, so every link between a pixel and a joint angle must
agree: where each fixed camera stands, how each wrist camera sits on its flange,
where each arm's base really is, and where each joint's encoder reads zero. A
wrist camera sees through its whole arm, so these errors show up mixed; only
views from many poses and several board positions, fitted together, tell them
apart. `ora cell calibrate --method hand-eye` does that.

You need a **plain checkerboard**, printed or bought, glued flat on something
rigid (acrylic, aluminium, foam board) and matte; **a ruler**, to measure it;
and tape or a weight to keep it still. The cell needs its rest pose (step 3),
its grippers' limits (step 4) and a rough pose for at least one fixed camera
(step 5): the board is located with it before any arm moves.

```bash
ora cell calibrate --method hand-eye [--square-mm 20 --pattern 11x8] [--placements 5] [--arms left,right]
```

It asks, in this order (on the terminal, or through the approval broker when
started without one or with `--ask broker`):

1. **The board** (`value`), unless given: one square's side in millimetres (measure
   ten squares and divide: a 2 % print scale is a 2 % error in every distance), and
   the inner corners as `COLSxROWS`.
2. **Each placement** (`task`): lay the board flat where the named arms reach and
   the fixed cameras see it, at a new position or angle each time. The first
   placement is for every arm (it ties them into one world); the others go to one
   arm in turn, so five give each of two arms three. The fixed cameras locate the
   board, and up to eight wrist views per arm are planned around it, each pose and
   every segment between poses checked against the table (or the board's plane),
   the other arm and the arm's own links.
3. **Each motion** (`motion`): one arm moves at a time, the other at rest; the
   views are taken, and the arms return to where they started and are released.
   The operator's home (board, `ora cell home`, `/home`) stops it. A board no
   camera sees, too few safe views, a declined or stopped motion: the placement is
   asked again with the reason. Decline a placement to end the captures.
4. **The result** (`write`): the changes in millimetres and degrees, the fit's error
   per camera, and the **held-out** error: each placement's wrist views predicted
   by a fit of all the others, the number that says what the calibration is worth
   where it was not fitted. It also names what should make you hesitate (a camera
   that fits badly, a placement predicted poorly, an arm seen in fewer than three
   placements). Approve, and the old calibration is copied beside itself
   (`<name>.before-handeye-<time>.json`) and the new one written; decline, and
   nothing changes.

On the cell board, the **手眼标定** panel starts the same command (`--ask
broker`): the three board fields are optional (empty ones are asked), 开始
confirms first, and every question then appears in the board's **等你批准**
card: 做好了 / 结束拍摄 for a placement, a value box for the board's size, 写入 /
不写入 for the result. The panel shows the progress lines and, at the end,
whether the calibration was written, its backup and what to hesitate about. The
board refuses to start while anything holds the cell (a play run: `/quit` it
first) or while the board itself holds the arms, and turns its live view off
first, since the calibration needs the cameras. 中止标定 sends the command
SIGINT: during a motion that is the operator's home (the arms stop, go back to
where they started, and the placement is asked again; 结束拍摄 then fits what
was captured), otherwise the calibration ends and nothing is written. A stop
asked before the command has taken SIGINT back (its first second or so) is held
and sent the moment it says it can take it, rather than lost. The play
terminal and `ora web` have no `/calibrate`: they exist only while a play run
holds the cell, which the calibration needs.

The fit (`ora/cell/bundle.py`) takes the first arm's base as the world, fits a
base correction for every other arm, the zero errors of each arm's inner joints
(the first joint's error is the base's turn, the last one's the camera mount's),
each wrist camera's mount and each fixed camera, and one pose per placement, all
against the board's corners in pixels. The cell lock is held from the first
placement to the last capture, so nothing else takes the arms in between; while
you place the board the arms are released at rest and the lock's label says so.
The captures stay under `~/.cache/ora/cell-calibration/handeye-<time>/`, one
folder per placement; `--records DIR ...` fits them again without moving
anything, and `--no-save` reports without asking to write.

Calibrate again after a camera or an arm is moved, bumped or remounted, and
whenever `ora cell read` finds the rest pose away from the recorded one.

`ora` usually runs in the project's environment, which has none of the robot's
drivers. When the config names `robot.python`, every `ora cell` command re-runs
itself in that interpreter, with this checkout on `PYTHONPATH`.

## The operator's verbs

The verbs act on the arms whoever holds them, and act **now**, also while a play
session executes a motion. They are implemented once
(`ora/cell/operator/actions.py`, and the answers to the worker's questions in
`approval.py`); the board's buttons, `ora cell <verb>` and the play terminal's
slash commands call the same functions, and `ora web`'s page runs the play
terminal's own command lines ([the web page from your laptop](#the-web-page-from-your-laptop)).

| Verb | Board | Shell | Play terminal | Web page (`ora web`) | Effect |
|---|---|---|---|---|---|
| home | 回到起始位 | `ora cell home` | `/home` | *Home* | the arms go back to the recorded rest pose and only then are released; nothing drops |
| hold | 保持不动 | `ora cell hold` | `/hold` | *Hold* | take the arms and hold the measured pose |
| float | 浮动（手扶） | `ora cell float` | `/float` | *Float…* | gravity compensation for guiding by hand; confirmed first |
| release | 释放（会软掉） | `ora cell release` | `/release` | *Release…* | the arms go limp; confirmed first |
| estop | 紧急失能 | `ora cell estop` | `/estop` | *E-stop…* | DISABLE to every motor, whoever holds the cell; confirmed first |
| diagnose | 诊断 | `ora cell diagnose` | `/diagnose` | *Diagnose* | who holds the cell, bus states, faults, distance from rest, and what to do |
| stop | | `ora cell stop` | | | ask the program holding the cell to home and release |
| kill | 结束占用进程 | `ora cell kill` | | | end the program holding the cell; enabled arms fall within the motor timeout; confirmed first |
| cameras | 相机复位 | `ora cell cameras` | | | hardware-reset the cell's cameras |
| can | CAN 重置 | `ora cell can` | | | re-initialise the bus interfaces (sudo; prints the command when it needs your password) |
| approve | 等你批准: 批准 | `ora cell approve --id ID [--value V]` | `/approve [value]` | the question's card: *Approve…* | yes to the worker's waiting question (see The robot's questions) |
| decline | 等你批准: 拒绝 | `ora cell decline --id ID [--reason TEXT]` | `/decline [reason]` | the question's card: *Decline…* | no to it; a declined motion's reason reaches the agent |
| mode | 动作批准: 每个动作都问 / 小动作自动 / 全部自动 | `ora cell mode ask\|safe\|auto\|status` | `/mode [ask\|safe\|auto]`, Shift+Tab | the footer's mode menu | which motions wait for a person (see The robot's questions); `auto` confirmed first |

The page also takes the play terminal's commands typed into its box.

How `home` gets the arms back:

- The board holds them (after `hold` or `float`): it drives them home itself.
- A program holds the cell and says it homes on a signal (an ORA worker, a
  calibration motion): it is sent SIGUSR1 and drives its own arms home. An ORA
  worker keeps holding them at rest and its episode goes on; the motion in
  progress reports `operator_home`. A program that does not answer within 20 s
  is ended and the arms are taken over at once, inside the motors' timeout.
- Nobody holds them and they are already at rest: nothing moves.
- Otherwise the arms are taken over and driven home in a straight joint path at
  0.25 rad/s. A path that meets the table or the other arm, or an arm that does
  not follow, stops there and holds; float the arms closer to rest by hand and
  try again.

`hold` and `float` keep the arms held, which needs a long-lived holder: the
cell board. `ora cell hold|float` therefore needs a running board, and while one
runs every `ora cell <verb>` goes through it (`--board-port`, else
`$ORA_CELL_BOARD_PORT`, else 8765). The play terminal's commands run
`ora cell <verb> --json`; they exist only when the run drives a real cell
(rig mode `read` or `on`). With `--json`, `ora cell` prints one object,
`{"ok": true, "result": ...}`, or `{"ok": false, "error": ...}` with exit 1;
`--yes` skips the confirmations. Episode-level controls stay in play's own
commands: `/reset` (a new round, which on a real cell begins at rest), `/stop`
and `/quit` ([free play](play.md)).

## The robot's questions

A play session's worker runs in its own session with no terminal, so it asks
the person at the cell through the cell's approval broker
(`ora/cell/operator/approval.py`). On rig mode `on` it asks:

| Question | When | No answer in | Declined |
|---|---|---|---|
| `enable` (要给两条手臂通电吗？) | once, after the start-up checks, before i2rt takes the motors | 300 s | the run stops; nothing was enabled |
| `motion` (左臂动作等待批准, 夹爪动作等待批准, …) | each motion plan, unless auto-approve is on (below) or `runtime.auto_approve` | 180 s | the motion is refused as `operator_declined`, with the reason |
| `release` (手臂没能回到起始位) | at the end, only when the way home is blocked | 60 s, within the close budget | the arms stay held and home is tried again |

**Stop here** (`ora cell stop-motion`, `approval.request_stop_motion`) ends the motion
the worker is executing at its next control tick, where the arms are, and keeps
them held there; the motion reports `operator_hold` and the agent gets a fresh
observation. A motion the agent is still planning when the stop arrives is
refused at its approval, before anything moves (`operator_declined`, telling the
agent to read the operator's latest message). It writes `stop-motion.json`
beside the questions for the program that holds the cell; nothing else changes,
a request older than 10 s is ignored, and `ora cell stop-motion --withdraw`
takes back one nobody read (the agent answered in words, no motion came). A front end's *stop and say* sends it, then the message: the agent
reads the message with the new observation. `home` is different: it drives
the arms back to rest.

Each motion's receipt keeps when its question was asked and answered and by
whom (`timestamps.approval`: `request_id`, `created_unix_s`, `answered_unix_s`,
`by`, `approved`; also for an auto-approved or stopped motion). The context
projection drops `timestamps`, so the agent does not read it; a free-play
session announces the wait as `approval_wait_s` with each `action_finished`.

The cell's **permission mode** decides which motions wait for a person (`ora cell
mode ask|safe|auto|status`, `approval.mode_state` / `switch_mode`): `ask`, the
default, asks every motion; `safe` lets a motion plan answer itself when it
lasts at most 10 s, moves no joint more than 0.6 rad and stays at least 3 cm
above the table and 15 cm from the other arm (a plan that does not state one of
these is asked); `auto` lets every motion of the worker that drives the arms
now answer itself (the auto-approve switch: `ora play --auto-approve`, `ora cell
auto-approve on`). `ask` and `safe` belong to the cell (`mode.json` beside the
questions): they can be chosen before a worker starts and they stay across runs.
`auto` belongs to one worker and falls back to the cell's mode when it ends.
Enabling the arms and releasing them are asked in every mode, and every
self-answered motion is recorded as an answer (`by: "safe mode (...)"`).

Every front end shows the mode and switches it; only `auto` is confirmed first.
The play terminal's footer starts with it (*▸▸ ask · every motion asked*,
*safe · small motions run*, *auto · every motion runs*, the last in bold red),
read from the broker's files at most once a second; `/mode` says it in full,
`/mode ask|safe|auto` sets it (`ora cell mode … --by "play terminal"`) and
Shift+Tab cycles ask → safe → auto → ask (auto only on rig mode `on`).
`ora web`'s footer shows the same words and opens a menu on a click, with
`safe`'s limits; typed `/mode …` works as in the terminal. The board's 动作批准
panel is a three-way selector (每个动作都问 / 小动作自动 / 全部自动) that says who
set the mode and `safe`'s limits. Each front end also shows the motions that
answered themselves, with the numbers safe mode checked (*Ran without asking:
左臂动作 · safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the
arms)*): the terminal as a notice, the page in its cell bar, the board under its
selector.

No answer in time counts as a decline. A guided setup step
(`ora cell calibrate --method hand-eye`) asks three more kinds: `task` (do
something by hand, such as placing the board, then approve; decline to stop),
`value` (approve with a value: `ora cell approve --id ID --value 20`, the
board's JSON `{"value": "20"}`; its `details` may name a `unit` and an
`example`) and `write` (save what was measured). Only a `value` question takes a
value, and its approval needs one. Each question says in plain words what
will happen (which arms, how long, how close to the table and to each other;
the plan itself is attached), and the board shows it as a card at the top
(**等你批准**, with 批准 and 拒绝 and an optional reason; a `task` card says 做好了
and 结束拍摄, a `write` card 写入 and 不写入, and a `value` card has a box for
the value). The shell answers
with `ora cell pending`, then `ora cell approve --id ID` or `ora cell decline
--id ID --reason TEXT`; without `--id`, on a terminal, the one waiting question
is shown and answered after a yes. The play terminal's `/approve` and
`/decline [reason]` show the question with its id and send the answer after a
`y`; a `value` question is approved with `/approve <value>` (its unit and
example are named), and `/approve` without one, or a value for any other
question, is refused before anything is sent; during an `on` run it also says *… 输入 /approve 或 /decline* as soon as a
question appears. `ora web`'s page shows each waiting question as a card above
its input box, with the same words, the id, the seconds left and a reason field
(a value field for a `value` question; *Done…* and *End captures…* for a
`task`, *Save…* and *Don't save…* for a `write`), and answers that id after a
confirmation; typed `/approve <value>` works as in the terminal. These answers touch only the broker's
files: never the cell lock, the board or the hardware.

An answer names the question's id, and an id that is not waiting (answered,
expired, or asked by a worker that has ended) is refused: nobody approves a plan
they did not see. A `home` sent while a motion waits for approval voids it,
since the plan starts where the arms were; the motion is declined with that
reason.

Closing an episode drives the arms home (the recorded rest pose, else the start
pose) and releases them there without a question: at rest nothing drops. The
episode's home is the operator tools' drive (`ora/cell/homing.py`): the
straight joint path is checked against the table and the other arm before
anything moves. If the way home is blocked (that check, or an arm that does not
follow), the arms hold where they are and `release` is asked:
approve to release them in place (support them first); decline, or leave it
unanswered, and they stay held while home is tried again, up to three times
within the worker's close budget (`runtime.close_timeout_s`, default 120 s,
less 15 s). At that bound the arms are released in place anyway, and the last
question says so: the worker is ending, and without its control loop the
motors let go after their timeout regardless. To keep them held past it, use
the board's 回到起始位 (`ora cell home`, `/home`): the board takes the arms over
and holds them for as long as needed.

### Auto-approve

Approving every motion is unworkable for an operator working alone at the cell,
so the broker has an auto-approve switch (`YOLO`), for motion questions only.
Enabling the arms (where they start holding) and releasing them in place (where
they go limp) are always asked.

```bash
./scripts/ora play --env yam --rig-mode on --operator-present --auto-approve ...   # on from the start
./scripts/ora cell auto-approve on        # on for the worker driving the arms now
./scripts/ora cell auto-approve off
./scripts/ora cell auto-approve           # status
```

In the play terminal, the same commands are `/yolo on` (asks y/N once), `/yolo off`
and `/yolo`; the switch is the permission mode `auto`, so `/mode auto`, the
board's 全部自动 and the page's footer menu turn it on too, and `/mode ask` or
`/mode safe` turn it off. While the switch is on, a motion question answers itself at once.
It still writes the normal answer file, with `by: "auto-approve (<who turned it
on>)"` and the question's `kind` and `title`, so the record lists every motion
and shows that nobody looked at it. RealRig receives the same `(approved)`
outcome as for a person's approval, so the agent's receipt and the trace do not
change. A `home` sent while a plan is pending still voids it.

The switch is `auto-approve.json` in the broker's folder
(`ora.cell-auto-approve.v1`: `kinds` (only `motion`), `by`, `since_unix_s`,
`pid`, `run`). It answers only the questions of the one process named by `pid`,
so these rules apply:

- A real worker's start clears it, then sets it for itself only when its run was
  started with `--auto-approve` (`runtime.auto_approve_motions`). A switch never
  carries into another run.
- `ora cell auto-approve on` names the worker that holds the cell lock in rig
  mode `on`, and is refused when there is none. Its `run` is that lock label.
- A switch whose worker has ended is stale: it answers nothing, and the state
  reports it as off, with the reason.

`ora cell auto-approve` reads and writes only this file. It takes no lock and
starts no board.

`runtime.auto_approve` (RealRig's own flag, set only in a hand-written RunSpec)
is broader: the worker asks nothing at all, not even before enabling the arms or
releasing them at the end. `ora play` never sets it.

The broker's folder is `$ORA_CELL_APPROVALS`, else `ora-cell-approvals` beside
the cell lock (`/tmp/ora-cell-approvals` by default); the worker receives both
variables. Each question is `<id>.request.json` (`ora.cell-approval-request.v1`:
`id`, `kind`, `title`, `text`, `details`, `created_unix_s`, `expires_unix_s`,
`timeout_s`, the asker's `pid` and label) and each answer `<id>.answer.json`
(`ora.cell-approval-answer.v1`: `id`, `approved`, `reason`, `by`,
`answered_unix_s`, the question's `kind` and `created_unix_s`, and `value` for an
approved `value` question), created only once, so the first answer counts. The worker
removes its question once decided; answers stay for an hour as the record.

## The web page from your laptop

`ora web` serves a live run's page on the computer the run runs on: the
conversation on one side and the cell's four cameras on the other, with the
cell's controls (the robot's questions, the permission mode and auto-approve,
the verbs, the held arms' alerts; [free play](play.md#on-the-web-page)). On a real cell that is the
cell's computer; from a laptop you reach the page through SSH. The examples
call the cell's computer `yam` (an alias in `~/.ssh/config`) and the checkout
there `/path/to/ORA` (an absolute path).

Build the page once on the cell's computer, and again after an update that
changes it (`ora web` says so):

```bash
ssh yam 'cd /path/to/ORA && ./scripts/ora web setup'
```

Then, from the laptop's checkout, either start the play session with its
terminal here and its page in the laptop's browser:

```bash
./scripts/ora-remote yam --repo /path/to/ORA --web play --env yam --rig-mode on --operator-present \
  --cell ~/.config/ora/cell.yaml --provider PROFILE --confirm-launch
```

or open the page of a play session that already runs on the cell (started in a
`tmux` there, say, so that it outlives your SSH session):

```bash
./scripts/ora-remote yam --repo /path/to/ORA web --last
```

`ora-remote` hands a one-time launch token to the cell's computer in a 0600
file, starts `ora web --socket` there on a 0600 unix socket in your runtime
directory, forwards a local port to that socket (`ssh -L`) and opens
`http://127.0.0.1:PORT/?ora_token=...` here ([web view](remote.md#web-view)).
Nothing listens beyond loopback on either side.

Without `ora-remote`, do the same by hand. On the cell's computer, next to the
running session:

```bash
ssh yam
cd /path/to/ORA
d=${XDG_RUNTIME_DIR:-$HOME/.cache}/ora-web; mkdir -p "$d"; chmod 700 "$d"
(umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$d/token"); cat "$d/token"   # copy it
./scripts/ora web --last --socket "$d/web.sock" --token-file "$d/token" --no-browser
```

and on the laptop, with the directory and the token that printed:

```bash
ssh -N -L 127.0.0.1:8790:/run/user/1000/ora-web/web.sock yam     # the socket's absolute path
# open http://127.0.0.1:8790/?ora_token=TOKEN
```

The token works once: the browser exchanges it for a cookie, and a second
browser needs a new `ora web`. `ora web` reads and removes the token file.

Run `ora web` from the checkout the session runs from, with the same
`ORA_CELL_LOCK`, `ORA_CELL_APPROVALS` and `ORA_CELL_ALERTS` as the session
(normally none of them is set): its gateway reads the questions, the switch,
the alerts and the cell lock from those files, and its buttons run that
checkout's `scripts/ora cell`. When the session ends, the page shows what it
recorded and its cell controls keep working until you stop `ora web`
(Ctrl+C); the board (`ora cell board`) and the shell do too.

## A simulated cell

The kit names no robot, and the YAM station has a simulated `CellAdapter` too: a
cell config with `robot.sim: true` runs every `ora cell` command, the board and
`ora cell setup` against kinematic arms and cameras MuJoCo renders, with nothing
plugged in. Use it to learn the setup and the board, to try a change to the
calibration on a cell whose truth you know, or while someone else has the arms.

```bash
ora cell sim init ~/.config/ora/sim-cell.yaml --python ~/envs/yam/bin/python
ora cell board --cell ~/.config/ora/sim-cell.yaml --port 8766       # one terminal
ora cell sim operator --cell ~/.config/ora/sim-cell.yaml            # another: the person at the cell
ora cell setup --cell ~/.config/ora/sim-cell.yaml --ask broker      # a third, or start it from the board
ora cell sim status --cell ~/.config/ora/sim-cell.yaml              # the calibration against the truth
```

`sim init` writes the config (environment `yam`, buses `sim_left` and
`sim_right`, cameras named `sim-...`); `--python` names an interpreter with
MuJoCo, mink and i2rt, where every `ora cell` command then runs. The cell keeps
everything in its state directory, `<config>.sim/` beside the config (or
`robot.sim: {state: DIR}`):

| File | Holds |
|---|---|
| `world.json` | the truth, written by the first command: where the right arm's base really is, each inner joint's encoder zero error, each wrist camera's mount error, where the fixed cameras stand, the table, the checkerboard (20 mm, 11x8) and each gripper's motor readings at its stops. Edit it, with the cell idle, to inject another error |
| `arms.json` | each arm's encoder readings, its gripper's reading and which process holds its motors |
| `board.json` | where the checkerboard lies on the table, absent when it is off |
| `bus/`, `cell.lock`, `approvals/`, `alerts.json` | the simulated buses, and the cell's lock, questions and alerts |

The simulated cell keeps its own lock, approvals and alerts there unless
`ORA_CELL_LOCK`, `ORA_CELL_APPROVALS` or `ORA_CELL_ALERTS` name others, so it
never shares them with a real cell on the same computer, and its adapter
refuses the real cell's `/tmp/ora-cell.lock`. Pass `--cell` (or set `ORA_CELL`)
to `ora cell pending` and `approve` so they find its questions; run the board on
another port than a real cell's.

What is simulated, and how:

- **Arms.** Held, they follow every command at once; released, they stay where
  they are. Their encoders read the true angle minus the injected zero error.
  `estop` and `release` from another process disable them, and a calibration
  motion then stops as on a real cell. Their speed is the kit's: a calibration
  takes as long as at the real cell.
- **Buses.** Files: every reader or held arm counts as a receiver while open
  and every read or command as a frame, so `ora cell check`, `diagnose`, the
  board's bus chips and the motions' guards see other programs as they would on
  socketcan.
- **Cameras.** The two fixed cameras and the two wrist cameras are rendered at
  848x480 with their intrinsics and metric depth, from the truth's model: the
  right base moved, the joints and wrist mounts off. The kit plans and fits on
  the plain robot model, as at a real cell. Rendering needs a headless OpenGL
  (`MUJOCO_GL=egl`, the default here, or `osmesa`).
- **Grippers.** Their stops come back from the gripper calibration named by the
  motor's direction, so the right gripper's are swapped until you, or the
  simulated operator, say it is open.
- **The person.** `ora cell sim place X Y [--yaw DEG]` lays the board with its
  middle at (X, Y) on the table and `sim clear` takes it off. `sim operator`
  answers the cell's questions as a person would until you stop it
  (`--seconds S`): it folds the arms to rest, says the board's size and which
  way each gripper opens from the truth, lays the board where each placement
  asks (and nearer the arm when asked again, declining after four tries),
  approves every motion and the write. Answer some yourself from the board and
  stop it to try a decline.

`ora cell sim status` prints the injected errors, the board, the arms and
`recovery`: for each arm, the calibrated base and wrist mount against the truth
(mm, degrees), the joint zeros injected, found and their difference, each fixed
camera's distance from where it stands, and the table height's. A fresh setup
finds the default errors (a 7.5 mm, 0.7 degree base and zeros up to 1.4
degrees) to within a millimetre and a tenth of a degree. The truth's left arm base is the
calibration's world, and the first and last joints' zeros belong to the base and
the wrist mount: injected there, the calibration cannot tell them apart and
moves them elsewhere. `tests/ora/environment/backends/yam/test_sim_cell_flow.py`
runs the whole setup this way in about a minute (in the YAM interpreter;
`test_sim_cell.py` checks the arms, buses and operator without MuJoCo).

A simulated cell drives no robot: the YAM worker refuses it for rig modes
`read` and `on` (its `sim` rig mode is the episode simulation).

## The safety contract

Every real backend keeps these properties. The kit's tests (`tests/ora/cell/`)
check its half of them against a fake robot: one holder, a `home` that holds
instead of dropping, and the refusals.

1. **Read before write.** Nothing is enabled before an occupancy check and a
   passive read: no other reader or traffic on the buses, joints inside the
   model's limits.
2. **One holder.** Every program that touches the buses or the cameras takes
   the cell lock (`$ORA_CELL_LOCK`, else `/tmp/ora-cell.lock`) and says in it
   what it does; `ora cell diagnose` reads it. Programs outside ORA do not know
   the lock, which is why the occupancy check still listens to the buses.
3. **Modes.** `sim`, `read` (motors never enabled; every motion answered with
   its plan) and `on` (requires `operator_present: true`; enabling the arms
   and each motion approved, the motions unless configured otherwise).
4. **Return to rest never drops.** From any state, `home` drives the arms to the
   recorded rest pose and only then releases; a blocked path holds in place.
   Disabling motors (the arms fall) is separate and confirmed: `release`,
   `estop`. A worker that ends releases its arms at rest without asking and
   asks before releasing them anywhere else.
5. **Guards stop and hold.** Tracking error, torque, temperature, a dead
   control loop or a foreign controller stop the motion where the arms are
   measured. On YAM a joint pushed on from outside stops it long before the
   motor-torque guard: each tick the measured joint torque minus i2rt's own
   model torque (its gravity and friction terms) is the external load, and a
   load over the joint's limit (6 N·m shoulder and elbow, 2.5 N·m wrist;
   `yam_station.EXTERNAL_LOAD_STOP_NM`) for 5 ticks is `external_load_stop`,
   naming the arm, the joint and the load. Each motion's peak loads go to the
   worker's `rig-loads/<action_id>.json`, the data the limits are tuned on
   (`ora/environment/backends/yam/external_load.py`). With `ora play
   --smooth-commands` (`runtime.smooth_arm_commands`, off by default) each
   control tick's arm target is reached along the line from the last one,
   streamed at 250 Hz with that line's velocity as i2rt's feed-forward, instead
   of one position step per tick with a zero velocity target; a hold is still
   commanded at once.
6. **A gripper works only when told to.** Every program that takes the arms
   over (the play worker, the board's holder, a calibration motion) relaxes the
   grippers at once: the motor stays enabled but gives no torque. A gripper
   command engages it for that move; the operator tools relax it again once it
   arrives, and the worker relaxes every gripper when the arms reach rest. A
   gripper held by position control pushes whenever its target is out of reach
   (a stop, an object, limits named the wrong way round) and overheats within
   minutes, which stops the arm's control loop. Which way a gripper opens is
   what the person saw (`ora cell grippers`), not the motor's nominal direction.
   On YAM a gripper command runs in force control: the motor's own position
   gain is off, its damping stays, and each tick the worker sends a torque
   capped at 15 N of grip from the first contact on, friction compensation
   included (`grippers.force_torque`; the acef teleoperation follower's law,
   soft gain 1 N·m/rad, friction 0.12 N·m past a 0.02 rad deadband). The
   command is ramped, never stepped (40 mm/s of aperture closing, 80 mm/s
   opening; it leads the measured jaw by at most 3 mm), and jaws that stall on
   an object are held 1 mm past where they stopped instead of being driven on
   to closed (`yam_station.GRIPPER_*`). A close stepped to the end under
   position control crossed the whole aperture in 0.26 s and snapped a
   fingertip (2026-11-14); i2rt's force limiter acts only once it has seen the
   jaws blocked. The operator tools' grip (the board calibration closes and
   opens the jaws once) moves the same way, in force control and ramped, and
   relaxes each gripper as soon as it arrives or stalls. `ora cell grippers`
   still drives the jaws to their stops under position control: that is how
   it finds them.
7. **Alerts reach the person.** Whoever holds the arms publishes their alerts
   (`ora/cell/alerts.py`: a motor warming up or near its protection, a gripper
   pushing, a stopped control loop) every half second; the board shows them as a
   banner. A critical one stops a calibration motion like the operator's home.
8. **A start-up self-check before anything is enabled.** The YAM worker
   checks, from what it measured at start: no motor fault; every motor below
   50 °C; the joints inside the model's limits; each gripper motor's reading
   within its recorded [closed, open] limits (0.1 rad of margin: a motor zero
   that moved over a power cycle makes the opening wrong by the whole travel);
   the table the wrist cameras see where the calibration puts it; one frame
   with enough valid depth from every camera; and the calibration's age (named
   past 14 days, never refused). On `on` a failed check refuses the start
   before the arms are enabled, and the refusal lists every failed check with
   its fix; a check that could not be made is named and does not refuse. On
   `read` the same checks run with nothing enabled, so a `read` session is the
   dry run of a real start. The result is the worker's `rig-startup.json`
   (`self_check`); `ora cell setup --status` reports the gripper check as
   `stale` (`ora/environment/backends/yam/selfcheck.py`).
9. **Forbidden frames.** Writes that change the robot's own calibration (on
   YAM's DM motors, frame `0xFE`, the encoder zero) cannot be built.

## Add a new robot

A new installation of a supported robot model needs only its `cell.yaml` and
steps 0 to 9. A new robot family adds a `CellAdapter`, the seam between the kit
and the robot's drivers (`ora/cell/adapter.py`; provisional, drawn from one
family). Put it in `ora/environment/backends/<environment>/cell_adapter.py` as
`make_adapter(config)`:

- **Facts the kit reads**: `environment`, `model`, `arms`, `buses` (arm to bus
  interface, from the config), `camera_mounts` (each public camera role to
  `fixed_world` or `end_effector`), `wrist_cameras` (wrist camera role to arm),
  `joint_names`, `control_hz`, `tracking_stop_rad` (the operator motions'
  tracking stop), `finger_open_m`, `foreign_controller_markers`
  (command-line fragments of programs that drive these arms), and for the
  hand-eye planner `reach_seed_offset` (the joint offset from rest that brings
  the wrist out over the table, its inverse kinematics' seed),
  `reach_offset_limit_rad` (no joint is planned further from rest) and
  `wrist_image_occluded` (pixel boxes of the wrist image the jaws hide).
- **`open_reader(gripper_limits)`** returns a `Reader` whose `read()` returns
  one `ArmReading` per arm (joints, gripper, faults, temperatures, torques)
  **without enabling any motor**, and `close()`.
- **`take_over(arms, mode=, gripper_limits=)`** returns a `Held`: refuse a bus
  another program uses first, release whatever was enabled if taking over fails
  partway, then `joints()`, `command(targets)` (joint targets; grippers held),
  `alive()` (each arm's control loop), `set_mode("hold"|"float")` without
  letting go, and `release()`, the only call that makes the arms limp.
- **`disable_all()`**: the last resort; it must work whoever holds the buses.
- **`calibrate_grippers(arms, should_stop)`**: each gripper's [closed, open]
  limits.
- **`kinematics(table)`**: the robot model's `Kinematics` (MuJoCo model and
  data, `world_offset`, `arm_limits` and `set_state` in encoder readings,
  `tcp`, `chain`, `wrist_camera`, `inter_arm_distance`, `table_distance`,
  `self_distances` (an arm's wrist against its own lower links), `ik`,
  `within_limits`, and the calibration's corrections: `camera_corrections`,
  `joint_offsets`, `set_base_corrections`).
- **`calibration_offsets()`**: joint offsets from rest for the camera
  calibration poses; the kit checks every path for table and inter-arm
  clearance.
- **`facts()`**: what the operator should know about the robot, shown on the
  board.

The robot's worker keeps the same contract: it takes the cell lock for its
whole episode; if it returns its arms to rest on SIGUSR1 it puts `[homes on
SIGUSR1]` in its lock label, and `[at rest, holding]` once there
(`ora/cell/lock.py`); its drivers cannot build a frame that rewrites the
robot's calibration. Then check the adapter without hardware: a fake one in the
shape of `tests/support/cell/fake.py` exercises the kit, and
`ora cell init` checks a config against the real one. A family may also ship a
simulated adapter, `sim_cell_adapter.py` with the same `make_adapter(config)`,
which a config with `robot.sim` loads instead; it gives the kit its own `bus`
(`Bus`: `receivers`, `traffic`, `interface` per channel) and `camera_driver`
(`CameraDriver`: `open`, `connected`, `busy`, `reset`), which a hardware
adapter leaves out to get socketcan and RealSense. The robot model also
needs its geometry, controller constants, embodiment and tool surface, and the
family a `Rig` qualified in `sim` before `read` and `on`
([design note](../design/real-robot-cells.md), section 8).
