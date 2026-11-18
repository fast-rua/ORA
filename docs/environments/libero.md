# LIBERO / LIBERO-Pro: install and run one episode

Read [shared setup](common.md) first. Both base LIBERO suites and the Pro perturbations use the direct LIBERO driver with the Panda robot. Official LIBERO termination supplies the verdict. The default surface uses CuRobo planning, RGB-D perception and overhead footprint localization.

## At a glance

One fixed-base Franka Panda arm (public arm `right`) in MuJoCo. Tasks are the
LIBERO Spatial, Object, Goal and LIBERO-10 suites and the LIBERO-Pro
perturbation suites; [evaluation protocol](../evaluation/protocol.md#seeds-and-case-lists)
lists the official cells. Rendering needs an NVIDIA GPU with EGL, or a Mesa
software-rendered slot (below).

| | |
| --- | --- |
| Observation | `agentview` (fixed) and `wrist` RGB-D at 256 x 256 with per-observation calibration, the measured grip-site pose and gripper, the last action's receipt and the remaining budget |
| Action | Motion proposals through the Supervisor: `move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`, `set_gripper` (`open` / `close`), `recover_pose`, `wait_ticks`; targets name the grip site |
| Success | LIBERO termination; a truncated episode is `false`. `check_task` returns that boolean, never the predicate |
| Task prompt | The suite's own task language, unchanged |

Robot facts: [Franka Panda](https://fastrua.ai/ora/docs/robots/embodiments/franka-panda/) ·
[frames and units](../robots/frames-and-units.md) ·
[policy I/O](../reference/policy-io.md).

## Source, data and interpreter

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs"
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$ORA_INSTALL_ROOT/sources/LIBERO-PRO"
git -C "$ORA_INSTALL_ROOT/sources/LIBERO-PRO" checkout --detach 2b910b5b5f53016bef9907632f6f840f1ce2229c
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro "$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data"
git -C "$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data" checkout --detach c86fc3b8293185a6f373677018ff3e37f8391602
uv venv --python 3.11 --seed "$ORA_INSTALL_ROOT/envs/libero"
uv pip install --python "$ORA_INSTALL_ROOT/envs/libero/bin/python" \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$ORA_INSTALL_ROOT/envs/libero/bin/python" \
  numpy==1.26.4 mujoco==3.8.1 robosuite==1.4.1 gym==0.26.2 gymnasium==0.29.1 \
  bddl==3.6.0 einops==0.8.1 cloudpickle==3.1.1 pyyaml==6.0.2 pillow==12.2.0 \
  imageio==2.37.4 imageio-ffmpeg==0.6.0 opencv-python==4.11.0.86 scipy==1.16.1 \
  h5py==3.14.0 termcolor==3.1.0 matplotlib==3.11.1 packaging==26.2 easydict==1.13
export LIBERO_PRO_PYTHON="$ORA_INSTALL_ROOT/envs/libero/bin/python"
export LIBERO_PRO_ROOT="$ORA_INSTALL_ROOT/sources/LIBERO-PRO"
export LIBERO_PRO_DATA_ROOT="$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The base suites read BDDL/init-state files from the code checkout; Pro reads the pinned dataset. The task definitions and init-state files are required, while demonstration downloads are skipped. Avoid global `LIBERO_TYPE`, `LIBERO_CONFIG_PATH` and `PYTHONPATH`; ORA creates its own isolated path map. The separate worker needs no training framework.

The default planner is cuRobo V2 at `d17b54ce32cba095c0b000c4c58777075d11de0e`. For now it runs under the RoboDojo runtime's interpreter, which already holds V2 ([RoboDojo](robodojo.md); `ora setup --env libero` installs that runtime): set `CUROBO_V2_PYTHON` to `<robodojo-runtime>/env/bin/python` and `CUROBO_V2_ROOT` to the cuRobo checkout it imports, `<robodojo-runtime>/source/RoboDojo/third_party/curobo`, which must be clean at that revision.

The default baseline sets `runtime.motion_planner.world_model: public_depth_voxel_v1`: the planner plans, and the reach check solves IK, in a collision world rebuilt for every request from the public depth of both cameras. The local OSC segments of a move do not check that world, and `world_model: none` plans against the robot alone; see [LIBERO planner world model](../reference/configuration-and-evidence.md#libero-planner-world-model).

The default baseline sets `runtime.motion_planner.implementation: curobo_v2`, the planner-neutral cuRobo V2 worker, with ORA's Franka model (the fingers where LIBERO's simulated gripper has them); a `motion_planner` block without the field is refused. The policy-visible text names no planner library; see [LIBERO behind a neutral planner](../reference/configuration-and-evidence.md#libero-behind-a-neutral-planner) for the comparison on recorded requests.

Every global route is timed by the [TOPP-RA + Ruckig worker](../reference/motion-speed.md#the-timing-worker) under `ORA_TIMING_PYTHON`, a small CPU interpreter `ora setup --env libero` builds. It is paced at the grip site before it runs, whichever planner answered it: `move_pose` takes an optional `speed` (`fast`, the default; `normal`; `careful`), `speed_m_s` and `slow_approaching`, with the same meaning as on RoboDojo; each joint stays the ceiling (at most 0.55 rad/s and 15 rad/s² here, what the global route's `JOINT_POSITION` controller follows under motion profile v3: kp 50, a reach of 0.2 rad and every knot aimed ahead by its own speed; its terminal settle ends once the arm rests inside the local 5 mm / 0.05 rad bound; see [the joint executor](../reference/motion-speed.md#the-joint-executor)); under motion profile v4, the default, a planned free-space move whose plan stays within 2 cm of the straight line and turns the hand at most 45 degrees runs along that line on OSC instead, the plan having checked the gripper's path there but not the arm's posture ([the straight OSC line](../reference/motion-speed.md#the-straight-osc-line-v4)); every control tick spends one native action of the round's allowance, which `remaining_budget` reports (the plan's `max_sim_steps`); a route whose hand would turn more than 45 degrees beyond the requested turn is refused before anything moves; and the bounded local OSC segments cap their commanded step per tick at the same speeds. See [Motion speed and wrist checks](../reference/motion-speed.md).

A bounded OSC segment can stop early, and the move is then `not_reached` with the arm where it stopped. `precision` stops at `contact_stop` when the measured force passes 40 N. A `contact_approach` pushes on purpose, so force alone does not end it: turning a stove knob took 133 N while the hand kept advancing. It stops at `contact_stop` when the force stays more than 40 N above the segment's lowest reading for 3 control ticks and the distance to the target falls by less than 0.5 mm over them, and at `tracking_diverged`, with or without contact, when that distance has risen for 2 ticks and stands more than 5 mm above the lowest it reached. The receipt's `contact_motion` gives the numbers. These stops do not check a path against the world, and the OSC action contract's tracking envelope stays telemetry for every other move.

A move with two segments, a free-space or out-of-envelope `precision` move (global route, then a local settle) or a staged `contact_approach` (global route to a pre-contact pose, then the local approach), runs its second segment only while the episode goes on. When the global segment leaves the episode terminal (official success, LIBERO's own horizon, the run's `max_sim_steps`, an abort or a control fault), the move returns that segment's receipt: `arms.right.settle` or `arms.right.precontact` reads `attempted: false` with `episode_terminated_before_settle` or `episode_terminated_before_contact_approach`, and `terminal_after_action` is true. The round then ends on the session's own reason with its verdict. `terminal_after_action` on every LIBERO receipt, `wait` included, is the session's terminal status after the action, not only LIBERO's `terminated` or `truncated`: the global route does not set `truncated` when it crosses `max_sim_steps`.

```bash
uv run --locked ora doctor --env libero_pro
```

`doctor` names this backend `libero_pro`; `ora setup`, `ora run` and `ora play` take `--env libero` for the same environment. It checks package discovery and directories. It does not qualify rendering. The worker's render guard rejects corrupted RGB/depth; check EGL/CUDA device correspondence before allocating concurrent slots. Start with one LIBERO renderer per GPU and keep its graphics selector explicit.

### Motion qualification

`ora qualify libero motion` runs the scripted sparse-motion cases through the real worker with no policy or provider: global, staged and local routes, large turns that must not alias, a planner refusal, and the probe-then-`plan_ref` lifecycle. It starts from a run that passed preflight on this host, whose plan and bindings it reuses; `--worker-env` replaces a value of that run's lane environment when the qualification runs on another lane.

```bash
./scripts/ora qualify libero motion --run outputs/<experiment>/runs/<run_id> \
  --output outputs/<experiment>/qualification/motion --worker-env CUDA_VISIBLE_DEVICES=0
```

The tool prints each case, the record path and its sha256; `ora/environment/definitions.py` cites the record of the current route set.

### Software-rendered (Mesa) slots

NVIDIA EGL tolerates one LIBERO renderer per GPU, and on some drivers a SAPIEN (RoboTwin) process anywhere on the same machine corrupts its frames. To pack several LIBERO episodes on a host, or to run LIBERO beside RoboTwin, render on the CPU with Mesa llvmpipe instead. Declare a slot whose `worker_environment` selects Mesa's EGL vendor:

```yaml
  libero-mesa-gpu0:
    runtime_kind: libero_pro
    resource_class: libero_pro_sim
    slot_id: lpm-gpu0-s0
    worker_environment:
      CUDA_VISIBLE_DEVICES: "0"          # CuRobo's GPU; robosuite also needs the EGL device listed here
      MUJOCO_EGL_DEVICE_ID: "0"
      MUJOCO_GL: egl
      PYOPENGL_PLATFORM: egl
      __EGL_VENDOR_LIBRARY_FILENAMES: /usr/share/glvnd/egl_vendor.d/50_mesa.json
      LIBGL_ALWAYS_SOFTWARE: "1"
      LP_NUM_THREADS: "2"
    minimum_free_memory_mib: 2048
```

Mesa lists every DRM render node first (NVIDIA's too, which it cannot drive) and its software device last, so the software device is device 0 only on a host without `/dev/dri`. The worker therefore picks it itself (`construction.bind_egl_device`, once per process; the preview worker does the same): after robosuite is imported, whose import asserts that `MUJOCO_EGL_DEVICE_ID` is one of the `CUDA_VISIBLE_DEVICES` entries, and before the first context, which reads it again, it takes the device that reports `EGL_MESA_device_software` (or names llvmpipe, softpipe or swrast, or has no DRM device file and no hardware device extension), makes an OpenGL context current on it and sets `MUJOCO_EGL_DEVICE_ID` to its index. With no such device the worker stops and names every device it saw. `reset.json` and every `round-resets/round-NNNN.json` record the choice as `egl_device` (`ora.software-egl-device.v1`: `index`, `vendor` from EGL, `driver` the OpenGL renderer, the versions, the device's extensions, the slot's `configured_device_id` and every device seen; `null` when the slot does not ask for software rendering); the policy never sees it. So a slot's `MUJOCO_EGL_DEVICE_ID` only has to pass robosuite's import check. A slot whose planner runs on GPU K may list `"K"` with `MUJOCO_EGL_DEVICE_ID: "K"`, or `"K,0"` with `"0"`. The planner and preflight's free-memory check use the first entry, while lease admission prices the slot's floor on every GPU listed, so `"K,0"` also charges GPU 0. Each lane costs CPU (about two cores at `LP_NUM_THREADS=2`) rather than GPU rendering. Mesa and NVIDIA images differ slightly (depth matches; RGB does not bit-match), so keep one renderer per scored cohort and do not pool their results.

## Compose and launch

Register a `libero-gpu0` slot and obtain a [perception readiness report](common.md#install-the-shared-perception-services). For a Pro object-swap task:

```bash
./scripts/ora spec libero compose libero_object_swap 0 \
  --seed 0 --experiment libero-example --resource-profile libero-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort libero-example-g001
```

For the original LIBERO Spatial suite:

```bash
./scripts/ora spec libero compose libero_spatial 0 \
  --seed 0 --experiment libero-base-example --resource-profile libero-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort libero-base-example-g001
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode). The source-bound task index and init-state selection are resolved by the composer. The run records the official verdict and the source/data pins. These recipes inherit the current default and an 80-decision, 200-model-call, 10,000-sim-step, six-hour ceiling; they do not promise a particular task success rate. Actual runtime and token use appear in the evidence.

If MuJoCo reports missing EGL, check driver graphics libraries and the slot's `MUJOCO_EGL_DEVICE_ID`. If the wrong LIBERO imports, clear inherited package selectors before starting the isolated worker. A data-path error should be fixed in the installation, not by weakening the source gate.
