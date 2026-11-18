# ora/environment/backends/libero_pro

## Responsibility

The LIBERO and LIBERO-Pro simulator adapter (environment id `libero_pro`; the
`ora` commands also call it `libero`). Everything here runs inside the
isolated LIBERO interpreter (`LIBERO_PRO_PYTHON`, the plan's
`runtime.python_env`), started by `ora/environment/rpc.py` as the environment
worker: it constructs the pinned LIBERO / LIBERO-Pro environment on the Franka
Panda, publishes public RGB-D observations, executes admitted motion with
measured receipts and reads LIBERO's own termination as the official verdict.

| Module | Holds |
|---|---|
| `session.py` | `LiberoProSession`, the worker session `definitions.py` names; its mixins split the work below |
| `action_plan.py` | The read-only answer to a dry run: the sparse route, planned and timed, stopped before the first step |
| `construction.py`, `direct.py`, `pinned_imports.py`, `registration.py`, `source_binding.py`, `robustness.py`, `worker_environment.py` | Construction of the source-bound direct driver (one LIBERO simulator owned by the worker, and its free-play binding), imports and benchmark data bound to the pinned checkouts, the LIBERO-Pro perturbation suites upstream does not register, source revisions and the qualified controller binding, the BDDL robustness cases; a Mesa lane's software EGL device (`construction.bind_egl_device`, also used by the preview worker) |
| `scene_trace.py` | The scene trace of a replay: the session's `MjSim`, Panda and grip site for `ora/environment/mujoco_scene_trace.py` |
| `observation.py`, `render_guard.py`, `execution_capture.py` | Public RGB-D, calibration and proprioception; the refusal of frames that are not a rendering; original-pixel capture |
| `sparse_motion.py`, `local_motion.py`, `local_motion_report.py`, `planned_motion.py`, `motion_settle.py`, `motion_evidence.py`, `wait.py` | Sparse motion v3: one semantic target routed over the bounded OSC local executor and the global CuRobo `JOINT_POSITION` route (its gains, reach and per-knot lead from the bound motion profile, `hybrid_motion.libero_joint_controller`; replays load the recorded gains), settling, receipts and planner evidence, and the shared `wait`; every global route is paced at the grip site and its wrist checked (`ora/environment/controllers/motion_pace.py`, by the timing worker `runtime.motion_planner.timing` selects, with grip-site kinematics from a scratch copy of the measured state; a wrist refusal of a pose move with open jaws runs the target's jaw twin instead, `controllers/jaw_flip_retry.py`), and a requested speed caps the local OSC step per tick |
| `motion_probe.py`, `probe_feedback.py`, `reach_check.py`, `planning_diagnostics.py` | Read-only reachability probes, IK candidate counts, the per-pose reach check and CuRobo IK diagnostics |
| `world_inputs.py` | The public snapshot (depth, calibration, base pose, joints, fingers) a world-mode planner builds its collision world from, written before every plan, probe and reach check |
| `planner_request.py` | The session's closed plan and IK request shapes and their planner-neutral form |
| `planner_port.py` | `NeutralPlannerPort`: a planner-neutral implementation (`runtime.motion_planner.implementation`) behind the session's planner interface, its answers turned into the session's plan and IK records |
| `common.py`, `robot_asset_geometry.py` | Constants and helpers; the robot-only MuJoCo geometry collector |
| `installation/requirements-x86_64.txt` | The pinned packages of the LIBERO interpreter |

## What belongs here, and what does not

Here: code that imports LIBERO, robosuite, MuJoCo or CuRobo, or that runs in
their interpreter. Not here: the host-safe suite identities and BDDL language
(`ora/environment/benchmarks/libero.py`, `libero_pro.py`,
`libero_pro_language.py`), the OSC and `wait` contracts the host also reads
(`ora/environment/controllers/libero_osc.py`, `libero_wait.py`), and the
host-side driver configuration, preflight,
supervision binding, round evaluation and play scenes
(`ora/orchestrator/backends/libero_pro/`).

No file here is in a hash-bound qualification manifest. LIBERO's sparse-motion
qualification is a record cited by sha256 beside
`SPARSE_MOTION_QUALIFIED_ENVIRONMENTS` in `ora/environment/definitions.py`;
a change to the motion routes here needs that scripted qualification run
again and the new record cited there.

## Dependency direction

Imports `ora.environment` (core, `benchmarks`, `controllers`) and
`ora.contracts`, plus the simulator stack, never another backend, the agent
or the orchestrator. `ora/environment/backends/robocasa/` reuses
`common.py` and `robot_asset_geometry.py` (both are robosuite). The
environment core reaches this package only through `worker.py`,
`replay_worker.py` and the `worker_loader` string.

## Entry points

- `session:LiberoProSession`: loaded by `ora/environment/worker.py`.
- `python -m ora.environment.backends.libero_pro.probe`: the metadata probe of
  the LIBERO launch preflight.
- `python -m ora.environment.backends.libero_pro.preview`: rebuilds the Viewer's
  LIBERO gallery previews (`ora-viewer compile previews`).

## Documentation

- [LIBERO / LIBERO-Pro](../../../../docs/environments/libero.md) and the
  [quick run](../../../../docs/getting-started/libero.md)
- [Local CPU compatibility](../../../../docs/reference/ora-local-cpu-compatibility.md)
- [Host-side backend](../../../orchestrator/backends/README.md)
