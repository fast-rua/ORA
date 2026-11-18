# RoboCasa365: install and run one episode

Read [shared setup](common.md) for host slots, model profiles, perception services and launch review. The default RoboCasa365 surface drives the PandaOmron mobile manipulator through admitted joint paths, a mobile-base tool and RGB-D perception. RoboCasa's own `_check_success` supplies the verdict.

## At a glance

The PandaOmron (a Panda arm on a holonomic base with a lifting torso) in
MuJoCo kitchens of the target split. All 50 `target50` tasks are admitted
(18 atomic seen, 16 composite seen, 16 composite unseen; 2026-11-05). Three of
them, `OpenDrawer`, `PickPlaceCounterToCabinet` and `NavigateKitchen`, carry a
reviewed completion contract ([RoboCasa365 tasks](https://fastrua.ai/ora/docs/robots/tasks/robocasa/));
the other 47 run on the episode's own instruction alone, as RoboCasa's
evaluation does.
The simulator renders on the CPU through Mesa EGL, so it needs no GPU of its
own; the perception services do.

| | |
| --- | --- |
| Observation | `agentview` (mounted on the base, it moves with the base) and `wrist`, RGB-D with per-observation calibration; the grip-site pose and measured jaw opening, the base pose and torso height (`mobile_base`), the last action's receipt and the remaining budget |
| Action | Sparse motion proposals through the Supervisor (`move_pose`, `move_to`, rotations, `set_gripper`, `recover_pose`, `wait_ticks`) and `move_base`; poses name the grip site. An arm target is admitted on the shared motion base: a planned route, or a straight line solved by the shared IK, kept out of a collision world built from the two public cameras' depth; the torso lift is used when the arm alone cannot reach |
| Speed | `move_pose`, `move_to` and the rotations take `speed`, `speed_m_s` and `slow_approaching`, as on every benchmark; every arm path is timed at the grip site by the TOPP-RA + Ruckig worker under `ORA_TIMING_PYTHON` and a hand turn more than 45 degrees past the request is refused before anything moves ([motion speed](../reference/motion-speed.md)); `move_base` is outside it |
| Success | RoboCasa's `_check_success`, latched at the first step it holds within the horizon. `check_task` returns that boolean, never the predicate |
| Task prompt | The episode's own instruction (RoboCasa samples the kitchen and target from the seed), plus the task's public completion contract where one is reviewed (three tasks) |
| Budget | No native-action limit: a run ends at its robot-decision, model-call or wall-time budget (`official_limit_multiple` 1000, as RoboDojo and RoboLab). The task's evaluation horizon (450 to 4350 control steps) still decides `official_success`; a later success is [recorded apart](../reference/configuration-and-evidence.md#official-limit-and-the-extended-budget) as `success_after_official_limit` |

Robot facts: [PandaOmron](https://fastrua.ai/ora/docs/robots/embodiments/panda-omron/) ·
[frames and units](../robots/frames-and-units.md) ·
[policy I/O](../reference/policy-io.md). How motion, admission and the budget
work: [RoboCasa365 and RoboLab backends](../reference/robocasa-robolab-backends.md).

## Source, interpreter and assets

RoboCasa365 runs in its own interpreter, built from robosuite and robocasa at
the revisions ORA pins:

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs"
git clone https://github.com/ARISE-Initiative/robosuite "$ORA_INSTALL_ROOT/sources/robosuite"
git -C "$ORA_INSTALL_ROOT/sources/robosuite" checkout --detach 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git clone https://github.com/robocasa/robocasa "$ORA_INSTALL_ROOT/sources/robocasa"
git -C "$ORA_INSTALL_ROOT/sources/robocasa" checkout --detach a9e152e2eea16d342dac40e4577fb4c68b932854
uv venv --python 3.11 "$ORA_INSTALL_ROOT/envs/robocasa"
uv pip install --python "$ORA_INSTALL_ROOT/envs/robocasa/bin/python" -e "$ORA_INSTALL_ROOT/sources/robosuite"
uv pip install --python "$ORA_INSTALL_ROOT/envs/robocasa/bin/python" -e "$ORA_INSTALL_ROOT/sources/robocasa"
"$ORA_INSTALL_ROOT/envs/robocasa/bin/python" -m robocasa.scripts.setup_macros
"$ORA_INSTALL_ROOT/envs/robocasa/bin/python" -m robocasa.scripts.download_kitchen_assets
export ROBOCASA_PYTHON="$ORA_INSTALL_ROOT/envs/robocasa/bin/python"
```

The asset download is about 10 GB and lands inside the robocasa checkout, as
does the macros file `setup_macros` writes. Launch preflight checks the
interpreter, both checkouts at their pinned revisions, and the packages
robocasa 1.0.1, robosuite 1.5.2 and MuJoCo 3.3.1.

Arm moves plan on the shared motion base, so a run also needs `CUROBO_V2_PYTHON`
and `CUROBO_V2_ROOT` (the cuRobo V2 interpreter and the checkout it imports, as for
LIBERO) and `ORA_TIMING_PYTHON` in the launch environment. Launch preflight starts
the planner once and refuses a run where it does not start.

## Register a slot

A RoboCasa365 slot is a Mesa EGL CPU slot; its first `CUDA_VISIBLE_DEVICES` entry
is the card the episode's planner runs on, about 0.7 GiB of it. Add one per
concurrent episode to your [host profile](common.md#register-the-host-and-a-slot):

```yaml
execution_resources:
  robocasa-mesa-0:
    runtime_kind: robocasa
    resource_class: robocasa_sim
    slot_id: mesa-0
    worker_environment:
      CUDA_VISIBLE_DEVICES: "0"
      MUJOCO_EGL_DEVICE_ID: "0"
      MUJOCO_GL: egl
      PYOPENGL_PLATFORM: egl
      __EGL_VENDOR_LIBRARY_FILENAMES: /usr/share/glvnd/egl_vendor.d/50_mesa.json
      LIBGL_ALWAYS_SOFTWARE: "1"
      LP_NUM_THREADS: "2"
    minimum_free_memory_mib: 2048
```

`MUJOCO_EGL_DEVICE_ID` here only satisfies robosuite's import check, which
wants it among the `CUDA_VISIBLE_DEVICES` entries. Mesa lists every DRM render
node first (NVIDIA's too, which it cannot drive) and its software device last,
so the software device is device 0 only on a host without `/dev/dri`. The
worker, and the probe that composing runs, therefore picks the device that
reports `EGL_MESA_device_software` (or names llvmpipe, softpipe or swrast, or
has no DRM device file) after robosuite is imported and before it creates a
context, and sets `MUJOCO_EGL_DEVICE_ID` to it. Where there is no such device
it stops and names every device it saw. Each round's reset records the device
as `egl_device`.

One control step costs about 65 ms of physics and a published observation
about half a second of CPU rendering, so a full episode takes tens of minutes
mostly spent in the model.

## Compose and launch

After qualifying the perception services, compose one cell. Composing resets
the seed once in the RoboCasa interpreter to read that episode's instruction:

```bash
./scripts/ora spec robocasa compose OpenDrawer \
  --seed 0 --experiment robocasa-example --resource-profile robocasa-mesa-0 \
  --provider-profile PROVIDER_PROFILE --service-report PATH_TO_READINESS_REPORT
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode).
The result carries the official boolean verdict; completion alone does not
imply success. The composer inherits the current default tools and prompts and
an 80-decision, 200-model-call, six-hour ceiling; no native-action limit
binds (`official_limit_multiple` 1000).
