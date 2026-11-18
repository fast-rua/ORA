# RoboTwin: install and run one episode

Read [shared setup](common.md) for host slots, model profiles, perception services and launch review. The default RoboTwin surface uses the ALOHA dual-arm robot, RGB-D perception and the shared planner; the expert and the seed scans use the vendored CuRobo planner. The simulator's `check_success` supplies the verdict.

## At a glance

The dual-arm ALOHA-AgileX (two 6-DoF arms) in SAPIEN. 50 tasks, each in clean
(`demo_clean`) and randomized (`demo_randomized`) scenes; every task's
instruction and completion contract is on [RoboTwin tasks](../robots/tasks/robotwin.md).
Needs an NVIDIA GPU, CUDA for the planner, and per-seed qualification (below).

| | |
| --- | --- |
| Observation | `head_camera` and `front_camera` (fixed), `left_camera` and `right_camera` (on the grippers), RGB-D with per-observation calibration; both arms' TCP poses and measured jaw separation, the last action's receipt and the remaining budget |
| Action | Sparse per-arm motion proposals through the Supervisor (`move_pose`, `move_to`, rotations, `set_gripper`, `recover_pose`, `wait_ticks`), plus `grasp_plan` / `grasp_execute` and `align_plan`; poses name each arm's TCP |
| Speed | `move_pose`, `move_to` and the rotations take `speed`, `speed_m_s` and `slow_approaching` beside `targets`, as on every benchmark; every global route is timed at the TCP by the TOPP-RA + Ruckig worker under `ORA_TIMING_PYTHON`, which `ora setup --env robotwin` builds, and a hand turn more than 45 degrees past the request is refused before anything moves ([motion speed](../reference/motion-speed.md)) |
| Planning | `move_pose`'s global routes, the reach check and `align_plan` plan on the shared planner under `CUROBO_V2_PYTHON` and `CUROBO_V2_ROOT`, which `ora setup --env robotwin` builds, in a collision world rebuilt per physical state from public depth with the other arm in it ([planner configuration](../reference/configuration-and-evidence.md)) |
| Success | RoboTwin's `check_success`, latched as `eval_success`, one verdict per round. `check_task` returns that boolean, never the predicate |
| Task prompt | The task's public instruction, plus its public completion contract |

Robot facts: [ALOHA-AgileX](../robots/embodiments/aloha-agilex.md) ·
[frames and units](../robots/frames-and-units.md) ·
[policy I/O](../reference/policy-io.md).

## Source, interpreter and planner

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ORA_INSTALL_ROOT/sources/RoboTwin"
git -C "$ORA_INSTALL_ROOT/sources/RoboTwin" checkout --detach c3ddfa8b97d5519efa828b075999bd0006778e5e
uv run --locked python -m ora.environment.backends.robotwin.vendored_install \
  --robotwin-root "$ORA_INSTALL_ROOT/sources/RoboTwin"
git -C "$ORA_INSTALL_ROOT/sources/RoboTwin" rev-parse HEAD
# Expected: cf4872cc90538fd4bca761c8db912d8f65e49f2b
uv venv --python 3.10 --seed "$ORA_INSTALL_ROOT/envs/robotwin"
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  -r "$ORA_INSTALL_ROOT/sources/RoboTwin/script/requirements.txt" \
  numpy==1.26.4 torchvision==0.19.1 setuptools==69.5.1
git clone https://github.com/NVlabs/curobo.git "$ORA_INSTALL_ROOT/sources/curobo"
git -C "$ORA_INSTALL_ROOT/sources/curobo" checkout --detach d64c4b005459db10c5dd867d8b30a87d5bda9bdb
# CUDA_HOME must select a toolkit compatible with this Torch environment.
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  -e "$ORA_INSTALL_ROOT/sources/curobo" --no-build-isolation
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" warp-lang==1.12.0
export ROBOTWIN_ROOT="$ORA_INSTALL_ROOT/sources/RoboTwin"
export ROBOTWIN_PYTHON="$ORA_INSTALL_ROOT/envs/robotwin/bin/python"
export CUROBO_ROOT="$ORA_INSTALL_ROOT/sources/curobo"
export CUROBO_PYTHON="$ROBOTWIN_PYTHON"
```

The installer creates a deterministic source commit. Do not edit the planner to bypass the revision gate. The original identity text retained inside the installer is part of that commit's reproducibility.

## Assets

Download `background_texture.zip`, `embodiments.zip` and `objects.zip` from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), selecting and recording an immutable dataset revision. Training checkpoints and demonstration trajectories are unnecessary. Record archive SHA-256 values, extract all three into `$ROBOTWIN_ROOT/assets`, then regenerate absolute robot paths:

```bash
(cd "$ROBOTWIN_ROOT" && "$ROBOTWIN_PYTHON" script/update_embodiment_config_path.py)
uv run --locked ora doctor --env robotwin
```

The `assets/background_texture/unseen/` pool must be complete. Moving the installation requires regenerating embodiment paths. SAPIEN requires working Vulkan libraries, not merely a visible CUDA GPU.

## Qualify one seed

On your registered `robotwin-gpu0` slot, prepare a bounded scan for one task. This runs the upstream expert, not the language model:

```bash
./scripts/ora qualify robotwin prepare \
  --robotwin-root "$ROBOTWIN_ROOT" --robotwin-python "$ROBOTWIN_PYTHON" \
  --tasks adjust_bottle --slots robotwin-gpu0 \
  --accepted-per-task 1 --candidate-count 3 --task-config demo_clean
./scripts/ora qualify robotwin qualify \
  --plan PATH_TO_PRINTED_PLAN --confirm QUALIFY-ROBOTWIN:PRINTED_HASH
```

The immutable plan gives the estimate and confirmation hash. Candidate probes can take minutes (the three-candidate `adjust_bottle` scan above finished in about 7 minutes on one data-center GPU lane, under its 28.5-minute estimate); this bounded scan is separate from a full 50-task qualification. Preparation and qualification records are written under `runtime-v1/robotwin-evaluation/` in the outputs root (`ORA_OUTPUTS_ROOT` when set, otherwise `<workspace>/outputs`); this namespace is fixed. `qualify` prints the path of the resulting `seed-manifests/adjust_bottle.json`, which names accepted seeds and binds the source/configuration. Select a seed actually present in it. Reuse still-valid evidence instead of rescanning on every launch. See [commands](../reference/robotwin-qualification.md#evaluation-seeds-ora-qualify-robotwin).

## Compose and launch

After qualifying perception services, substitute their report path, the seed manifest and an accepted seed:

```bash
./scripts/ora spec robotwin compose adjust_bottle \
  --seed ACCEPTED_SEED --task-config demo_clean \
  --qualification PATH_TO_SEED_MANIFEST \
  --experiment robotwin-example --resource-profile robotwin-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort robotwin-example-g001
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode). The result contains the official boolean verdict; completion alone does not imply success. `demo_randomized` needs qualification for that configuration. The composer inherits the current default tools/prompts and an 80-decision, 200-model-call, 10,000-sim-step, six-hour ceiling. Actual duration and tokens are reported in the episode evidence.

## Free play

`ora play --env robotwin` plays an audited seed of an official task without its
success check, or, with `--scene-set tabletop-v1` or `--scene-set household-v1`
(or `--scene` naming one of their scenes), one of ORA's own tables with many
objects: three rebuilt prototypes, or a breakfast table and a kitchen counter
whose objects have household masses. Audit a scene set on your RoboTwin slot
before the first session, and again after a change to its scene modules:

```bash
./scripts/ora play audit --env robotwin --scene-set household-v1 --seeds 5 --resource-profile robotwin-gpu0
./scripts/ora play --env robotwin --scene kitchen --provider PROFILE --service-report PATH_TO_READINESS_REPORT --confirm-launch
```

A scene of a set has its own operator camera for you (1280 × 720, from your
side of the table at three quarters, never one of the agent's cameras), which
`/view` and the web page show. Its execution capture records that camera beside
the head and front cameras at 30 fps of simulated time, and after each action
the scene keeps running, the arms holding their last command, until its objects
rest (at most 2 s), so a knocked-over object is seen falling. The wall time of
a motion grows about twofold; see [free play](../usage/play.md#tables-with-many-objects-robotwin-scene-sets).
Official tasks in free play keep RoboTwin's observer camera and the 10 fps capture.
