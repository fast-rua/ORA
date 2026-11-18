# ora/environment/backends/robocasa

## Responsibility

The RoboCasa365 simulator adapter (environment id `robocasa`): the PandaOmron
mobile manipulator in robosuite 1.5 kitchens. Everything here runs inside the
isolated RoboCasa interpreter (`ROBOCASA_PYTHON`, the plan's
`runtime.python_env`), started by `ora/environment/rpc.py` as the environment
worker. It builds one kitchen task at a seed, publishes public RGB-D
observations and robot state, executes admitted arm and base motion with
measured receipts, and reads RoboCasa's own `_check_success` as the official
verdict.

| Module | Holds |
|---|---|
| `session.py` | `RoboCasaSession`, the worker session `definitions.py` names: the privilege-clean episode writer |
| `construction.py` | Builds one RoboCasa365 environment inside the worker, and keeps or hands back the order of counters' regions, which decides its kitchen beyond the seed |
| `robot.py` | Trusted PandaOmron state reads and one-control-step commands |
| `action_plan.py` | The read-only answer to a dry run: what `motion.py` would run, stopped before the first step, the lift's admitted path and `move_base`'s footprint check included |
| `shared_motion.py` | All RoboCasa adds to the shared motion base: the planner instance, frames, the public depth world, the FK parity check, IK, routes and lines, why a line's IK stopped, stands (a lift or a base move asked about from the mount's frame there, in a world without the robot's own pixels), the PandaOmron's `MobileBase` for the base-placement search and `move_base`'s footprint check |
| `kinematics.py` | Forward kinematics of the arm and the geometry of a line; reads the robot's joints and nothing of the scene |
| `motion.py` | `motion_action_v3` execution: joint-position execution of admitted IK paths, timed at the grip site by the run's timing worker and wrist-checked before anything moves (`ora/environment/controllers/motion_pace.py`, `motion_timing.bind_preflight`; a wrist refusal with open jaws runs the target's jaw twin, `jaw_twin_path`), the gripper, `move_base`, the shared `wait` and home recovery |
| `scene_trace.py` | RoboCasa's use of the MuJoCo scene-trace reader (`ora/environment/mujoco_scene_trace.py`): the session's simulator, robot and grip site, a row per control step, what is backdrop in a kitchen |
| `common.py` | Schemas, the backend error and shared helpers (partly re-exported from `libero_pro/common.py`) |

## What belongs here, and what does not

Here: code that imports robosuite or RoboCasa, or runs in that interpreter.
Not here: the pins, task roster and public context
(`ora/environment/benchmarks/robocasa.py`), the PandaOmron control constants
(`ora/environment/controllers/robocasa_omron.py`), and the host side: runtime
fields, preflight, the RunSpec builder and the scripted motion qualification
(`ora/orchestrator/backends/robocasa/`).

No file here is in a hash-bound qualification manifest. RoboCasa365 joined
`motion_action_v3` on a scripted motion record cited by sha256 beside
`SPARSE_MOTION_QUALIFIED_ENVIRONMENTS` in `ora/environment/definitions.py`;
a change to the motion here needs `ora qualify robocasa motion` run again and
the new record cited there.

## Dependency direction

Imports `ora.environment` (core, `benchmarks`, `controllers`),
`ora.contracts` and two helpers of the LIBERO backend (`libero_pro/common.py`
and `libero_pro/robot_asset_geometry.py`; both are robosuite), plus the
simulator stack; never the agent or the orchestrator. The environment core
reaches it only through `worker.py` and the `worker_loader` string.

## Entry points

- `session:RoboCasaSession`: loaded by `ora/environment/worker.py`.
- No module here is started as a process; `ora qualify robocasa motion`
  drives the real worker from `ora/orchestrator/backends/robocasa/qualification.py`.

## Documentation

- [RoboCasa365](../../../../docs/environments/robocasa.md) and the
  [quick run](../../../../docs/getting-started/robocasa.md)
- [RoboCasa365 and RoboLab backends](../../../../docs/reference/robocasa-robolab-backends.md)
- [Host-side backend](../../../orchestrator/backends/README.md)
