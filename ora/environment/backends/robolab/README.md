# ora/environment/backends/robolab

## Responsibility

The RoboLab simulator adapter (environment id `robolab`): the DROID Franka
with a Robotiq 2F-85 gripper in Isaac Sim 5.1 on the pinned RoboLab checkout
(`ROBOLAB_ROOT`). Everything here runs inside the isolated RoboLab interpreter
(`ROBOLAB_PYTHON`, the plan's `runtime.python_env`), started by
`ora/environment/rpc.py` as the environment worker. It starts the upstream
task, publishes metric RGB-D from the over-shoulder and wrist cameras with
calibration and TCP proprioception, executes admitted motion on the shared
planner's joint paths and an absolute-IK Cartesian servo on the Robotiq TCP,
and latches RoboLab's own success termination as the official verdict.

| Module | Holds |
|---|---|
| `session.py` | `RobolabSession`, the worker session `definitions.py` names: the privilege-clean episode writer |
| `action_plan.py` | The read-only answer to a dry run: what `motion.py` would run, stopped before the first step |
| `bridge.py` | The pinned upstream RoboLab lifecycle, with simulator imports confined to start-up |
| `motion.py` | `motion_action_v3` execution: the two routes, a planned joint path streamed one row per tick (or its target's jaw twin, when the wrist check refuses it with the jaws open: `shared_motion.pace_or_twin`) and the Cartesian servo, whose step per tick a requested end-effector speed caps (`ora/environment/controllers/motion_pace.py` `step_caps`) |
| `shared_motion.py` | RoboLab's place on the shared motion base: the planner, the depth world from the last observation, FK parity, the shared IK and timing |
| `control_record.py` | What reached the simulator per native action (`native/controller-replay/`): the tap on the articulation's write, recorded in a run and fed back by its replay; neutral, and a failure stops the record, not the run |
| `scene_trace.py` | The scene trace of a replay program's re-execution: the bodies of the stage, the cubes and cylinders RoboLab's tables are built of, the TCP on the flange, one row per native action, read by RoboDojo's `isaac_scene.py` |
| `reach_check.py` | The read-only reach check over the `align_plan` bridge: the shared IK, judged at the TCP |
| `contract_witness.py` | The seed-0 task-contract witness: a review's witness plan run from `start` (direct pose edits, a gripper closed on a held object, RoboLab's own verdict per case, and a visibility probe of the named objects at reset); private, never policy-visible |
| `common.py` | Numerical and artifact helpers of the worker |

The directory has no `__init__.py`: the worker imports its modules as a
namespace package.

## What belongs here, and what does not

Here: code that imports Isaac Sim or RoboLab, or runs in that interpreter.
Not here: the pin, task roster and public context
(`ora/environment/benchmarks/robolab.py`), the DROID control and geometry
constants (`ora/environment/controllers/robolab_droid.py`), and the host side:
runtime fields, preflight, the RunSpec builder and the scripted motion
qualification (`ora/orchestrator/backends/robolab/`).

No file here is in a hash-bound qualification manifest. RoboLab joined
`motion_action_v3` on a scripted motion record cited by sha256 beside
`SPARSE_MOTION_QUALIFIED_ENVIRONMENTS` in `ora/environment/definitions.py`;
a change to the motion here needs `ora qualify robolab motion` run again and
the new record cited there.

## Dependency direction

Imports `ora.environment` (core, `benchmarks`, `controllers`),
`ora.contracts` and three RoboDojo helpers: two make Isaac Sim 5.1 start on
these hosts (`robodojo/graphics_compat.py` and the crash-reporting and
renderer helpers of `robodojo/bridge.py`) and one reads an Isaac stage for the
scene trace (`robodojo/isaac_scene.py`), plus the simulator stack; never the
agent or the orchestrator. The environment core reaches it only through
`worker.py` and the `worker_loader` string.

## Entry points

- `session:RobolabSession`: loaded by `ora/environment/worker.py`; a replay
  program runs from its `start` when the worker's configuration names
  `replay_program_job` (`ora/environment/robolab_replay.py`), and a contract
  witness when it names `contract_witness_job`
  (`ora/orchestrator/backends/robolab/contract_witness.py`).
- No module here is started as a process; `ora qualify robolab motion` drives
  the real worker from `ora/orchestrator/backends/robolab/qualification.py`.

## Documentation

- [RoboLab](../../../../docs/environments/robolab.md) and the
  [quick run](../../../../docs/getting-started/robolab.md)
- [RoboCasa365 and RoboLab backends](../../../../docs/reference/robocasa-robolab-backends.md)
- [Host-side backend](../../../orchestrator/backends/README.md)
