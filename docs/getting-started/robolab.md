# Get started with RoboLab

[RoboLab](https://github.com/NVlabs/RoboLab) is a tabletop manipulation
benchmark in Isaac Sim. ORA admits all 120 tasks of RoboLab-120 on a DROID
Franka with a Robotiq 2F-85 gripper, and every one has a reviewed completion
contract ([RoboLab tasks](../reference/robolab-tasks.md)). RoboLab's own success
termination decides success.

| | |
|---|---|
| Machine | x86_64 Linux with an NVIDIA GPU whose Vulkan works, one per episode |
| Install | the explicit steps on [RoboLab: install and run one episode](../environments/robolab.md) |
| Run | `./scripts/ora spec robolab compose`, then the shared review and launch commands |

The one-command path (`ora setup --env`, `ora run --env`) does not cover
RoboLab yet. The explicit path has four steps:

1. Clone RoboLab at its pinned revision with its Git LFS assets, build its
   Isaac Sim 5.1 interpreter, point `ROBOLAB_GRAPHICS_RUNTIME` at RoboDojo's
   Vulkan compatibility runtime and set `ROBOLAB_ROOT` and `ROBOLAB_PYTHON`.
2. Register a one-GPU slot (`runtime_kind: robolab`) in your host profile and
   qualify the perception services ([shared setup](../environments/common.md)).
3. Compose one cell, for example `BananaInBowlTask` on seed 0, with
   `./scripts/ora spec robolab compose`.
4. Review and launch it; the result carries RoboLab's verdict.

## What to expect

The first Kit start on a GPU takes several minutes. The policy sees the
task's episode length as its native-action allowance (750 control steps for
`BananaInBowlTask`; 300 to 4500 across the roster), one
per 15 Hz tick. Run one Isaac
process per GPU, and stop at the first `ERROR_DEVICE_LOST`.

## More

- [RoboLab tasks](https://fastrua.ai/ora/docs/robots/tasks/robolab/): instructions, limits and completion contracts.
- [DROID Franka with Robotiq](https://fastrua.ai/ora/docs/robots/embodiments/droid-franka-robotiq/): the robot manual the model reads.
- [RoboCasa365 and RoboLab backends](../reference/robocasa-robolab-backends.md): motion, admission, budget and qualification.
