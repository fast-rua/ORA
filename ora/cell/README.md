# ora/cell

## Responsibility

The real-cell kit: what every real robot cell needs, whatever the robot. It
reads the cell config the cell's owner writes (`ora.cell-config.v1`) and the
calibration the tools write (`ora.cell-calibration.v1`), keeps one holder of
the cell at a time, checks that no other program reads or drives the buses,
captures and registers the cameras, runs the depth completion service, and
gives the operator one set of verbs (`home`, `hold`, `float`, `release`,
`estop`, `diagnose`, `stop`, `kill`, `cameras`, `can`) behind the cell board,
`ora cell` and the play terminal's slash commands, and one place where a
robot's worker asks the operator (enable, each motion, release in place) and
those three front ends answer (`approve`, `decline`). A robot family plugs in
through one `CellAdapter` (`adapter.py`); the kit is written once against it.
The protocol is provisional: it is drawn from one robot family (I2RT YAM on
DM motors over socketcan). The buses and cameras reach the kit through the
adapter's `bus` and `camera_driver` (socketcan and RealSense unless it names
others), so a simulated cell (`robot.sim`, a robot's `sim_cell_adapter.py`) runs
the whole kit with no hardware.

| Module | Holds |
|---|---|
| `config.py` | The cell config: load and validate `ora.cell-config.v1` (`--cell`, else `$ORA_CELL`, else `~/.config/ora/cell.yaml`); `check_against` an adapter; a simulated cell's state directory (`robot.sim`) and `isolate`, which keeps its lock, approvals and alerts there |
| `calibration.py` | The calibration file: fixed-camera extrinsics, table, gripper limits, rest pose, notes, and the measured corrections of the robot model (arm bases, joint encoder zeros, wrist-camera mounts) that `apply` hands to a kinematics; legacy `ora.yam-cell-calibration.v1` files load as the current schema |
| `adapter.py` | The `CellAdapter`, `Reader`, `Held`, `Kinematics`, `Bus`, `Camera` and `CameraDriver` protocols, `ArmReading`, `load_adapter` (`ora.environment.backends.<environment>.cell_adapter.make_adapter`, or `sim_cell_adapter` for a simulated cell) and `bus_of` / `camera_driver_of` (the adapter's, else socketcan and RealSense) |
| `lock.py` | `CellLock` (an exclusive `flock` on `$ORA_CELL_LOCK`, else `/tmp/ora-cell.lock`, labelled with what the holder does), `holder()`, and the `HOMES_ON_SIGNAL` / `AT_REST` label markers |
| `occupancy.py` | Read-only checks: CAN receivers from `/proc/net/can`, traffic heard on a bus, processes that drive arms, busy RealSense cameras, interface state from `ip`; `SOCKETCAN`, the default `Bus` |
| `homing.py` | The one drive home every program holding the real arms uses (`drive`): the straight joint-space line checked against the table and between the arms first (`path_problems`), then a ramped drive that ends at rest, blocked, on a stopped control loop, short of rest or out of time; each caller passes its own `HomeSettings` |
| `geometry.py` | Table planes, joint sanity against limits and rest, fixed-camera registration on the arms (static and moving) |
| `bundle.py` | The joint fit of a cell's geometry from checkerboard views: fixed cameras, wrist-camera mounts, arm bases and joint zeros against the board's corners in pixels (`solve`), its leave-one-placement-out check (`validate`), and the calibration it writes (`apply_fit`); no hardware |
| `errors.py`, `io.py` | `CellError`, the one refusal type; atomic JSON writes |
| `cameras/` | `realsense.py` (aligned RGB-D with a depth validity check; `REALSENSE`, the default `CameraDriver`), `charuco.py` (every camera from one ChArUco board; a plain checkerboard's pose, its half turn and `by_colour`, which names its corners by its colours), `wrist_registration.py` (a provisional fixed-camera pose from the wrist views of the table) |
| `depth/` | `service.py` (the depth completion service: RGB + raw depth in, completed depth out, over HTTP) and `client.py` (the board's per-camera completer and `fuse`) |
| `operator/context.py` | `Cell`: one cell's config, adapter and calibration, its `bus` and `camera_driver` (`open_camera`), and the passive arm read |
| `operator/actions.py` | The operator verbs, implemented once, and `Holder`, the arms one process holds for as long as the person needs |
| `operator/approval.py` | The approval broker: a worker's questions (`ask`) and their answers by id (`pending`, `answer`), as request and answer files in `$ORA_CELL_APPROVALS`, else `ora-cell-approvals` beside the cell lock; a dead asker's question is ignored, an answer for an id not pending is refused, no answer in time is a decline; the auto-approve switch (`auto-approve.json`: motion questions of one worker pid answer themselves, still recorded) |
| `operator/board.py`, `board.html`, `live.py` | The cell board (status page, verbs over HTTP, the worker's questions with 批准 / 拒绝, the long-lived holder for `hold` and `float`) and its live view (streams, fused point cloud, passive joint reads) |
| `operator/motion.py`, `operator/tools.py` | Supervised calibration motions (grippers, cameras on moving arms; the `Mover` publishes the held arms' alerts and stops on a critical one) and the setup tools that enable nothing (read, record rest, wrist check, ChArUco, fixed camera from the wrists, which way each gripper opens) |
| `operator/handeye.py` | The guided hand-eye calibration: asks the person for the board and each placement, plans safe wrist views around the board, moves and records them, fits (`bundle`) and writes after a yes |
| `operator/setup.py` | `ora cell setup`: a new cell step by step (devices, rest, grippers, fixed cameras, hand-eye, table, verify); `status` reads what is done, missing or stale, `run` does the rest in order |
| `operator/asking.py` | How a setup step asks the person: `BrokerAsker` (the approval broker) or `TerminalAsker` |

## What belongs here, and what does not

Here: code every real cell needs and that names no robot: the config and
calibration files, the lock, occupancy, cameras, depth, and operator tools
written against the `CellAdapter`. Not here: anything that speaks to one robot
family's drivers or knows its geometry. That is the robot's backend: its
`CellAdapter` (`ora/environment/backends/<environment>/cell_adapter.py`, and
`sim_cell_adapter.py` for its simulated cell),
device drivers, kinematics, the `Rig` and its episode session
(`ora/environment/backends/yam/` for the YAM station). The `ora cell` command
itself is `ora/app/cell.py`. A measured value never goes into the cell config,
and a secret never goes into either file.

## Dependency direction

`ora.cell` imports no other `ora` package (rule L1 places it beside the
layers): it reaches a robot only through the `"ora.environment.backends.<environment>.cell_adapter"`
string in `adapter.load_adapter`. `ora/environment/backends/<robot>/`,
`ora/orchestrator/backends/<robot>/` and `ora/app/cell.py` import it; the
policy side (`ora/agent`) never does. Inside the package, `operator/` imports
the top-level modules, `cameras/` and `depth/`, and no other module of the kit
imports `operator/`. Device libraries (`pyrealsense2`, `python-can`, `cv2`) are imported
inside the functions that need them, so the package loads in `ora`'s own
environment; the hardware paths run in the robot's interpreter (the config's
`robot.python`), where `ora cell` re-executes itself. The package
`__init__.py` imports nothing.

## Entry points

- `ora cell ...` (`ora/app/cell.py`): `init`, `check`, `read`, `record-rest`,
  `grippers [--orient]`, `calibrate --method arm-poses|charuco|wrists|hand-eye`, `setup`, `board`,
  `depth-service`, the operator verbs and `pending`, `approve`, `decline`,
  `auto-approve on|off|status`, and `sim init|status|place|clear|operator` for a
  simulated cell (through the adapter's `sim`).
- `python -m ora.cell.depth.service`: the depth service under whatever
  interpreter holds the model, on any machine the cell reaches (also
  `ora cell depth-service`).
- `operator.actions`: called by the board, by `ora cell <verb>` and, through
  `ora cell <verb> --json`, by the play terminal's slash commands.
- `operator.approval`: `ask` from a robot's worker (`ora/environment/backends/yam/real_rig.py`);
  `pending` and `answer` from the board (`/api/approval`; it also starts
  `ora cell calibrate --method hand-eye --ask broker`, `/api/calibrate/start|stop`), `ora cell
  pending|approve|decline` and, reading the request files itself, the play
  terminal (`ora-pi/src/play/cell.ts`). `ask` and `ask_value` also from
  `operator.asking` (the `task`, `value` and `write` questions of `handeye` and
  `tools.orient_grippers`). The auto-approve switch:
  `start_run` from a robot's worker at its start (`ora/environment/backends/yam/session.py`),
  `switch_auto_approve` from `ora cell auto-approve` (and the play terminal's `/yolo`),
  `auto_approve_state` for any front end that shows it.
- `config.load` and `calibration.load`: read by a robot's worker
  (`runtime.cell_config`) and preflight.

## Documentation

- [Real robot cells](../../docs/usage/real-cells.md): the cell config, the
  startup steps, the operator verbs, the safety contract and adding a robot
- [The YAM backend](../environment/backends/yam/README.md): the one robot
  family with a `CellAdapter` today, and a simulated one
- [A simulated cell](../../docs/usage/real-cells.md#a-simulated-cell)
