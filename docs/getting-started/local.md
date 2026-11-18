# Run ORA on your computer (CPU)

`ora local` runs one LIBERO task on your own machine with a CPU MuJoCo
simulator: no GPU, CUDA, CuRobo or perception service. The model is called
through the DeepSeek API, so you need an API key and network access. Your
provider bills the calls.

This is a reduced composition for trying ORA, not the benchmark setup. Its
tools are bounded end-effector moves, gripper open/close, RGB-D observation and
point marking. It has no motion planner, segmentation or learned grasp service,
so do not compare its results with the full server composition.

## Requirements

- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) and
  Node.js 22.19 or newer with npm ([install](installation.md#nodejs-and-the-agent-core)).
- A Python 3.11 interpreter. `ora setup` does not download one; install it
  with `uv python install 3.11` if needed.
- macOS 13 or newer on Apple Silicon (native OpenGL), or Linux with the
  system OSMesa library for offscreen software rendering.
- About 2 GB of disk for the pinned LIBERO sources, assets and the simulator's
  own Python environment.

## Install

From a clone of this repository:

```bash
uv sync --all-extras --locked --python 3.11
./scripts/ora setup
```

`ora setup` downloads the pinned LIBERO-Pro code and task data, creates an
isolated CPU simulator environment and installs the agent core's Pi packages
(`ora-pi/`), which include the terminal UI.
It starts no simulator and calls no API. `--root PATH` (or `ORA_LOCAL_ROOT`)
chooses the installation directory; the default is your user data directory.

## Check

```bash
./scripts/ora providers key DEEPSEEK_LOW_API_KEY   # once; or export it (DEEPSEEK_API_KEY is accepted too)
./scripts/ora doctor --env local
```

`doctor --env local` checks the sources, simulator environment, graphics
binding, credential presence and terminal UI without starting the simulator or
calling the API. It prints what is missing and the next command.

The profile also reads `DEEPSEEK_BASE_URL`, which you need not set: it
defaults to the profile's endpoint, `https://api.deepseek.com`. While the key
is missing, `doctor` lists it beside the key (`required_environment`), and the
check passes once the key is stored. A `DEEPSEEK_BASE_URL` left set to any
other address fails the check (`expected_endpoint`): unset it, or set exactly
that URL.

## Run

```bash
./scripts/ora local --confirm-launch
```

The task picker opens with LIBERO `libero_goal` task 0, seed 0 ("open the
middle drawer of the cabinet"); press Enter to start it. `--suite`,
`--task-id` and `--seed` select another original LIBERO task. Without
`--confirm-launch` you can browse but not start a task.

A browser window follows the run's recorded camera frames: the scene stays
still while the model thinks and moves only during robot actions. While the
run is active you can steer it from the terminal; see
[interactive sessions](../usage/interactive.md). Runs land under
`outputs/ora-interactive/runs/`.

## Status

This path is new. Platform selection and the composition are covered by
offline tests; it has not yet been validated end to end on a Mac. Please
report the output of `./scripts/ora doctor --env local --json` with any issue.
