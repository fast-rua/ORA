# Get started with RoboCasa365

[RoboCasa365](https://github.com/robocasa/robocasa) is a kitchen benchmark for a mobile
manipulator. ORA admits all 50 tasks of its target split (`target50`) on the
PandaOmron (a Panda arm on a holonomic base with a lifting torso); three of
them, `OpenDrawer`, `PickPlaceCounterToCabinet` and `NavigateKitchen`, carry a
reviewed completion contract. The robot drives its base
with `move_base`, its arm follows admitted joint paths, and RoboCasa's own
`_check_success` decides success.

| | |
|---|---|
| Machine | x86_64 Linux; the simulator renders on the CPU (Mesa EGL), the perception services need an NVIDIA GPU |
| Disk | about 10 GB of kitchen assets plus the interpreter |
| Install | the explicit steps on [RoboCasa365: install and run one episode](../environments/robocasa.md) |
| Run | `./scripts/ora spec robocasa compose`, then the shared review and launch commands |

The one-command path (`ora setup --env`, `ora run --env`) does not cover
RoboCasa365 yet. The explicit path has four steps:

1. Build the RoboCasa interpreter at the pinned robosuite and robocasa
   revisions, download the kitchen assets and set `ROBOCASA_PYTHON`.
2. Register a Mesa EGL slot (`runtime_kind: robocasa`) in your host profile
   and qualify the perception services ([shared setup](../environments/common.md)).
3. Compose one cell, for example `OpenDrawer` on seed 0, with
   `./scripts/ora spec robocasa compose`.
4. Review and launch it; the result carries RoboCasa's verdict.

## What to expect

Each seed samples its own kitchen and target, so the instruction comes from
that episode. The policy sees the task's horizon as its native-action
allowance (750 control steps for `OpenDrawer` and `PickPlaceCounterToCabinet`,
450 for `NavigateKitchen`), and every base or arm tick spends one. Arm targets
the robot cannot reach or that would run into the kitchen are refused before
any motion and cost nothing; the refusal says why.

## More

- [RoboCasa365 tasks](https://fastrua.ai/ora/docs/robots/tasks/robocasa/): horizons and completion contracts.
- [PandaOmron](https://fastrua.ai/ora/docs/robots/embodiments/panda-omron/): the robot manual the model reads.
- [RoboCasa365 and RoboLab backends](../reference/robocasa-robolab-backends.md): motion, admission, budget and qualification.
