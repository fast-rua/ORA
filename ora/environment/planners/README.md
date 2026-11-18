# ora/environment/planners

## Responsibility

The motion planner implementations behind the planner-neutral contract in
`ora/environment/controllers/motion_planner.py`. Each implementation is one
sub-package whose worker runs as its own process, under its own interpreter,
started by `MotionPlannerClient`: it reads one JSON request per line (`ready`,
`plan`, `ik`, `check`, `fk`, `close`) and answers each with a planner-neutral answer. A
backend asks in robot terms (start joints, a goal tool pose or goal joints, a
world reference and its hash, a held object) and gets a timed joint
trajectory, a status and one neutral reason, so the planner can be replaced
without touching the backend. The timing worker is the same kind of process: it
plans nothing, and answers `retime`.

| Path | Holds |
|---|---|
| `curobo_v2/worker.py` | `Worker`, which validates requests, builds the world from the request's public inputs (`controllers/depth_world.py`) with the planner's own robot spheres, exports it to the planner's layout, attaches the held object, resets every seed per goal, runs the start and goal checks, answers a pose plan that turns the wrist the long way (a detour past the pacing's rule, or a goal a whole turn away on a joint with room for a nearer equivalent) as a straight joint line to the same pose where one turns less and passes its checks (`detail.wrist_detour`) and names the neutral reason; and the JSON-lines loop |
| `curobo_v2/wrist_detour.py` | A pose plan that turns the wrist the long way and the straight joint lines to the same pose that may replace it: the goal's nearest 2 pi equivalent in range, the other IK solutions the planner's attempts kept, the ranking the pacing's wrist rule implies, the line as a trajectory; pure numpy |
| `curobo_v2/engine.py` | `CuroboV2Engine`, the only module that calls cuRobo V2: one `MotionPlanner` and one `RobotCollisionChecker` sharing one scene collision checker and robot model, and the planner settings |
| `curobo_v2/path_clearance.py` | What the goal carve hides from a plan: `path_clearance`, a path's smallest clearance from any world, for this check and any reader of a planned path; `FingerScene`, the scene whose goal carve clears only what the fingers and load may touch; `what_stands`, the words a refusal uses |
| `curobo_v2/robots.py` | The robots V2 plans for: each robot's V2 file (bundled, the simulator's own with `--robot-file`, as the ARX X5's and the ALOHA's, or for a URDF robot the file ORA writes around the URDF `--robot-file` names, as the YAM's) and ORA's changes to it (the Franka's fingers where LIBERO's are, the X5's and the ALOHA's finger spheres, the Robotiq hand on the Franka, the sphere buffer), and the sphere set's hash every answer names; pure Python |
| `curobo_v2/yam.py` | One arm of the I2RT YAM station (`yam_linear_4310`) on i2rt's station URDF, for either arm: the base link and the public TCP as the tool frame, the sphere fit of every link, fingertip and the wrist camera, the self-collision pairs and the fold's padding, the V2 file around the URDF; pure Python |
| `timing/retime.py` | TOPP-RA + Ruckig timing of one joint path under a speed request: the pieces, the caps, the tick rows, Ruckig's certificate, the wrist check and the budget; the only module that calls the two libraries |
| `timing/worker.py` | The timing worker's JSON-lines loop (`ready`, `retime`, `close`) |
| `timing/requirements.py` | The exact resolved package set of the timing interpreter (`ORA_TIMING_PYTHON`) |

## What belongs here, and what does not

Here: code that imports a planner library, or runs in its interpreter, and is
not tied to one simulator. Not here: the contract and the client lifecycle
(`ora/environment/controllers/motion_planner.py`, host-safe), the world builder
and its per-planner exports (`ora/environment/controllers/depth_world.py`), a
simulator backend (`ora/environment/backends/<sim>/`). A new implementation
adds a sub-package and an entry in `IMPLEMENTATIONS`.

## Dependency direction

Imports `ora.environment.controllers` and the planner library, never a
backend, the environment core, the agent or the orchestrator. Nothing in
`ora` imports this package: the client starts a worker by module name. The
`__init__.py` files import nothing.

## Entry points

- `python -m ora.environment.planners.curobo_v2.worker --device D --robot R --world-model M [--robot-file F]`:
  started by `MotionPlannerClient` under `CUROBO_V2_PYTHON`, with the V2 source
  root (`CUROBO_V2_ROOT`, a clean checkout at the bound revision) first on
  `PYTHONPATH`. `R` is `franka_panda`, `arx_x5`, `aloha_agilex`, `franka_robotiq` or `yam_linear_4310`,
  `M` is `none` or `public_depth_voxel_v1`, and `F` is the robot file a
  backend's simulator ships, or for the YAM i2rt's `yam_station_linear_4310_d405.urdf`.
- `python -m ora.environment.planners.timing.worker`: started by `TimingClient`
  under `ORA_TIMING_PYTHON`, a small CPU interpreter holding `timing/requirements.py`
  and nothing of a simulator's.

## The YAM station

`yam_linear_4310` is the real bimanual I2RT YAM cell's arm (`ora/environment/backends/yam/`). With
`runtime.motion_planner` (`ora play --motion-planner`, off by default) the YAM backend plans on this worker
(`backends/yam/shared_planner.py`); without it, it keeps its own mink IK and planner (`backends/yam/motion.py`).
The switch becomes the default after it is qualified on the real cell.

One model, one arm at a time, the other arm in the world, as the X5 (`arx_x5`, RoboDojo's two arms) and the ALOHA
do. The two arms are the same chain on bases 0.61 m apart with one orientation, so one six-joint model plans either:
a request names the left joints and is in the moving arm's base frame (`left_base` or `right_base`), and a two-arm
world input stands the other arm in as its spheres at its measured joints. A model of both arms as one robot (V2
can plan a two-arm description; its bundled `dual_ur10e.yml` is one) would check the two moving arms against each
other inside the solver. It is not taken, for three reasons: the pose goal, the IK, the refusal and the jaw are each
one arm's in the contract every backend already speaks; the X5 and the ALOHA are planned this way, so the YAM reuses
their two-arm world inputs as they are; and a twelve-joint problem doubles the IK seeds' dimension and gives the CUDA
graphs another shape. What it costs: in a simultaneous move (`motion.py` executes one when both arms have goals)
each arm's plan sees the other standing at its start, so the inter-arm clearance over the two timed paths together
stays a check of the backend's, as `motion.py`'s per-tick `inter_arm_distance` is today.

The description is i2rt's station URDF, the source of the MJCF `StationKinematics` loads, so the planner and the
backend read one geometry; `left_base` as the base link keeps the right arm's subtree out of V2's model. The tool
frame `left_tcp` is the station's public TCP (`controllers/yam_station.py`), an extra link on `left_gripper`. The
fit and the self-collision pairs are in `yam.py`'s docstring: 190 spheres, at most 12 mm past a link's surface (6 mm
on the fingertips) before the 4 mm buffer; links that can meet within the limits are checked, the shoulder against
the wrist included. V2 at d17b54ce pads self-collision by the sphere buffer as well, so the YAM's links give it back
there (`self_collision_buffer`), and `link1` and `link2` give 6 mm more for the folded rest pose: the rest pose and
the cell's measured rest one degree past joints 2 and 3 pass V2's start check, two degrees past does not.

The cell's calibration (`ora/cell/calibration.py`, `apply`) changes the model at run time, and the backend carries
it without a per-cell robot file (`shared_planner.py`):

- The right arm's base correction is a pose, not a joint: the request is already in the moving arm's base frame, so
  the backend hands the corrected base as that frame (the world input's base pose, the other arm's base pose), and
  the robot file stays the CAD's.
- Each joint's encoder zero offset: the planner plans model angles (`reading + offset`, as `StationKinematics` does
  inside), so the backend adds the offsets to every joint it sends (the start, the world input's own and other
  arm) and takes them off every planned row. The limits stay the URDF's model limits. A measured start past a limit
  (the arm on its hard stop with the offset putting it just outside, `StationKinematics.widened_limits`) is further
  out than the worker's own 5 mrad slack, so the backend plans a start at most 2 degrees past a limit from 2 mrad
  inside it and puts the step from the measured start first in the path: inward only, which the station's widened
  per-tick limit check admits. A start further out is refused (`joint_limit`).
- The wrist cameras' corrections move only where the backend says the camera sees from; the camera's spheres stay
  on the CAD mount, a correction of a few millimetres being well inside the fit's margin.

The YAM plans in `public_depth_voxel_v1` with its own parameters (`depth_world.PUBLIC_DEPTH_VOXEL_YAM_V1`): a 1.06
x 1.56 x 0.86 m grid centred at (0.25, 0, 0.30) in the moving arm's base frame, which holds the arm's reach, the
other arm's base on either side and the table top (the base plane), and a hand box measured on the station model
(the gripper and its open fingertips span 0.07 x 0.14 x 0.16 m behind the TCP, the hand convention's origin). The
world input is the depth of every calibrated public camera (the two fixed D435i and the two wrist D405), read when
the plan is made, with the other arm at its measured joints. The session checks the planner's forward kinematics
against the station model at start, around both arms' measured joints (`shared_planner.fk_parity`). The URDF's
velocity limits are placeholders (1 rad/s): the station's own timing, or the timing worker's caps, time every path.

Measured on the simulated station (2026-11-15, the YAM workstation's RTX 3070 Ti), the station's own planner and
the shared one on the same 60 admitted targets per case, each the TCP of a joint vector drawn within the limits and
pointing within 60 degrees of down (so the arm reaches it), half per arm:

| Case | Station plans | Shared plans | Both planned: joint travel, duration (shared / station, median) |
|---|---|---|---|
| `free_space` from the ready pose | 38 | 46 | 0.86, 0.91 |
| `free_space` from the cell's rest, 0.9 and 0.6 degrees past joints 2 and 3 | 40 (17 `ik_unreachable` on the lift) | 50 | 0.64, 0.63 |
| `precision`, within 0.35 rad of the ready pose | 58 | 58 | 1.00, 1.00 (the same line) |

The shared planner refused 6 free-space targets the station planned: their solutions lie folded near the base or
reach into the other arm, where V2's spheres find 0.5 to 2 mm of self-contact; the station checks no self-collision.
A free-space plan takes 0.7 to 1.0 s at the median and 1.1 s at the 95th percentile (building the world from the four
848 x 480 depth images is most of it), the station's 0.27 and 0.40 s. On positions and tilts drawn over the whole
workspace, about four in five admitted targets are out of the arm's reach for both, and the shared planner plans as
many or more in every case. The scripted sparse-motion qualification (`orchestrator/backends/yam/qualification.py`)
passes through the real worker path with the switch on, every move planned by the shared planner.

Verified on the YAM workstation's RTX 3070 Ti (8 GB) with V2 d17b54ce, torch 2.7.0+cu128, warp 1.19.0, 2026-11-14:
the worker's engine loads the file, names `robots.YAM_SPHERE_SET_SHA256`, finds the rest pose clear and plans a pose
goal and a joint goal from rest; its FK is the fixture's `planner`. The CI tests run without V2 (`tests/data/fk-parity/
yam_linear_4310.json`), and `tests/ora/environment/planners/curobo_v2/test_yam.py` checks the description against the
station model under the YAM interpreter.

## Documentation

- [Configuration and evidence: motion planner](../../../docs/reference/configuration-and-evidence.md)
- [Motion speed and wrist checks](../../../docs/reference/motion-speed.md)
- [Repository structure](../../../docs/developer/repository-structure.md)
