# RoboLab: install and run one episode

Read [shared setup](common.md) for host slots, model profiles, perception services and launch review. The default RoboLab surface drives a DROID Franka with a Robotiq 2F-85 gripper through RoboLab's absolute differential-IK target and RGB-D perception. RoboLab's own success termination supplies the verdict.

## At a glance

The DROID Franka with a Robotiq 2F-85 in Isaac Sim 5.1. All 120 RoboLab-120
tasks are admitted; a task launches once its completion contract is reviewed
(`BananaInBowlTask`, `RubiksCubeAndBananaTask` and `StackYellowOnRedTask` so
far; [RoboLab tasks](../reference/robolab-tasks.md)). Each task's instruction,
limit and completion contract is on
[RoboLab tasks](https://fastrua.ai/ora/docs/robots/tasks/robolab/). Needs one NVIDIA GPU per episode
with working Vulkan.

| | |
| --- | --- |
| Observation | `agentview` (RoboLab's fixed over-the-shoulder camera) and `wrist`, RGB-D with per-observation calibration; the Robotiq TCP pose and measured jaw opening, the last action's receipt and the remaining budget |
| Action | Sparse motion proposals through the Supervisor (`move_pose`, `move_to`, rotations, `set_gripper`, `recover_pose`, `wait_ticks`); poses name the Robotiq TCP. A free move plans on the [shared motion base](../reference/robocasa-robolab-backends.md) and follows a joint path, one 15 Hz control tick per row; a contact approach or short move runs the Cartesian servo, each tick moving RoboLab's IK target a bounded step |
| Speed | `move_pose`, `move_to` and the rotations take `speed`, `speed_m_s` and `slow_approaching`, as on every benchmark: a planned path is timed and wrist-checked before it moves, and the servo caps its step per tick ([motion speed](../reference/motion-speed.md)) |
| Success | RoboLab's success termination (the episode freezes at success). `check_task` returns that boolean, never the predicate |
| Task prompt | The task's upstream default instruction, plus its public completion contract |
| Budget | No native-action limit: a run ends at its robot-decision, model-call or wall-time budget (`official_limit_multiple` 1000, as RoboDojo). The official limit (the upstream episode length at 15 Hz, 300 to 4500 control steps) still decides `official_success`; a later success is [recorded apart](../reference/configuration-and-evidence.md#official-limit-and-the-extended-budget) as `success_after_official_limit` |

Robot facts: [DROID Franka with Robotiq](https://fastrua.ai/ora/docs/robots/embodiments/droid-franka-robotiq/) ·
[frames and units](../robots/frames-and-units.md) ·
[policy I/O](../reference/policy-io.md). How motion, admission and the budget
work: [RoboCasa365 and RoboLab backends](../reference/robocasa-robolab-backends.md).

## Source, interpreter and graphics runtime

RoboLab runs in its own Isaac Sim interpreter, built from the pinned RoboLab
checkout:

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/RoboLab "$ORA_INSTALL_ROOT/sources/RoboLab"
cd "$ORA_INSTALL_ROOT/sources/RoboLab"
git checkout --detach ad45d4f974725d020f82c2b0d77d78533aeba2b3
git lfs install --local && git lfs pull
UV_PROJECT_ENVIRONMENT="$ORA_INSTALL_ROOT/envs/robolab" uv sync --python 3.11 --extra isaac51
uv pip install --python "$ORA_INSTALL_ROOT/envs/robolab/bin/python" warp-lang==1.11.0
export ROBOLAB_ROOT="$ORA_INSTALL_ROOT/sources/RoboLab"
export ROBOLAB_PYTHON="$ORA_INSTALL_ROOT/envs/robolab/bin/python"
```

`uv sync` writes a `uv.lock` into the checkout; add it to
`.git/info/exclude` so the checkout stays clean, which launch preflight
requires. Preflight also checks Isaac Sim 5.1.0.0, IsaacLab 2.3.2.post1 and
Warp 1.11.0 in that interpreter.

The worker needs two more inputs:

- `ROBOLAB_GRAPHICS_RUNTIME`: the process-local Vulkan compatibility runtime
  RoboDojo builds, the runtime root that holds `graphics-compat/` after
  [`setup-robodojo graphics-compat`](robodojo.md); preflight checks its
  `installation.json`.
- `ROBOLAB_LD_LIBRARY_PATH`: a directory of system libraries the host lacks,
  such as `libGLU`; leave it unset when the host has them.

The shared motion base needs three more in the launch environment:

- `CUROBO_V2_PYTHON` and `CUROBO_V2_ROOT`: the cuRobo V2 interpreter and checkout at
  `d17b54ce32cba095c0b000c4c58777075d11de0e`, as on [LIBERO](libero.md). Preflight
  records them for the worker, whose own environment is allowlisted. The planner runs
  as its own process on the slot's GPU, beside Isaac.
- `ORA_TIMING_PYTHON`: the [timing worker](../reference/motion-speed.md) interpreter.

The environment start holds Kit's start, the scene and the planner's start. That took
six to ten minutes on a shared filesystem, so `runtime.request_timeout_s` is 1200.
The run's cache starts empty, so the planner compiles its kernels on every run: 50 s with
six CPUs beside Kit, more than 180 s with four. Its requests get 300 s; give a run at
least six CPUs.

## Register a slot

A RoboLab slot binds one CUDA device, which is also the Kit renderer device:

```yaml
execution_resources:
  robolab-gpu1:
    runtime_kind: robolab
    resource_class: robolab_sim
    slot_id: gpu1
    worker_environment:
      CUDA_VISIBLE_DEVICES: "1"
    minimum_free_memory_mib: 20480
```

Run one Isaac process per GPU. If a start reports `ERROR_DEVICE_LOST`, stop
and leave that GPU alone: a Kit process that crashes on that device can leave
its Vulkan state unusable for every later start until the GPU is reset. The
worker process (`ora/environment/worker.py`) therefore never closes Kit in
process: `RobolabSession.close` writes the evidence and closes the motion
planner, then the process ends with `os._exit` (at interpreter exit, or after
`CLOSE_EXIT_GRACE_S` = 10 s should Kit's threads hold it), as RoboDojo's
worker does. An in-process owner of a session keeps the full teardown.

## Compose and launch

After qualifying the perception services, compose one cell:

```bash
./scripts/ora spec robolab compose BananaInBowlTask \
  --seed 0 --experiment robolab-example --resource-profile robolab-gpu1 \
  --provider-profile PROVIDER_PROFILE --service-report PATH_TO_READINESS_REPORT
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode).
The first start of Kit on a GPU takes several minutes; later starts reuse its
caches. The result carries the official boolean verdict; completion alone
does not imply success. The composer inherits the current default tools and
prompts and an 80-decision, 200-model-call, six-hour ceiling; three times the
task's episode length binds the native actions.
