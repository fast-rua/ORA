# Get started with RoboDojo

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) is a dual-arm (ARX)
manipulation benchmark in NVIDIA Isaac Sim 5.1, with long-horizon tasks such
as sorting, pouring, stacking by instruction and solving equations with
blocks. ORA drives both arms through perception, grasp-planning and motion
tools; the task's official success function decides success.

| | |
|---|---|
| Machine | x86_64 Linux with one NVIDIA GPU, 32 GB or more; driver R580 recommended |
| Disk | ~105 GB (66 GB of assets through git-lfs) |
| Install | `./scripts/ora setup --env robodojo` |
| Run | `./scripts/ora run --env robodojo --task general_pickup --confirm-launch` |

You need ORA itself (`git clone https://github.com/fast-rua/ORA.git &&
cd ORA && uv sync --all-extras --locked`), a C++ compiler,
unprivileged user namespaces for bubblewrap, and a DeepSeek API key (`ora setup --env robodojo`
installs git-lfs and `bwrap` under its prefix when the machine has none).
The default policy is code mode: the model acts by writing Python that calls
its tools, and each program runs in a bubblewrap sandbox
([check and install it](single-gpu.md#2-user-space-tools);
`./scripts/ora doctor --env robodojo` reports `code_mode_bwrap`,
`code_mode_user_namespaces` and `code_mode_interpreter`).

## Install and run

```bash
./scripts/ora setup --env robodojo
./scripts/ora providers key DEEPSEEK_LOW_API_KEY      # once; or export DEEPSEEK_LOW_API_KEY=sk-...
./scripts/ora run --env robodojo --task general_pickup --confirm-launch
```

- **Install.** About 10 minutes plus downloads on a fast connection. Setup
  installs:
  - RoboDojo, its submodules and assets at pinned revisions;
  - a Python 3.11 environment with Isaac Sim 5.1 and CUDA 12.8 PyTorch;
  - a process-local Vulkan compatibility layer that every RoboDojo worker
    checks before it starts (nothing system-wide);
  - the perception services, the agent core's Pi packages and the terminal UI;
  - the code-mode sandbox interpreter (`ORA_CODE_MODE_PYTHON`, Python 3.11 with
    NumPy, SciPy, Pillow, scikit-image and Shapely from a hash-pinned lock).

  If the machine lacks `libGLU` or the Vulkan loader, setup unpacks them into
  its own directory without sudo. Check the [tools to install first](single-gpu.md#2-user-space-tools).
- **First run on a GPU.** RoboDojo qualifies the card once per GPU, driver and
  installation: two short episodes, about 9 minutes. Each new task is then
  qualified once, in about 4 minutes. `ora run` shows an estimate and asks
  before each; later runs reuse the records.
- **Run.** The perception services start if needed; then the terminal UI
  opens with the task. Press Enter in the picker. Each task runs layout 0.
- **Result** on an RTX PRO 6000: `general_pickup` reached the official success
  in 2 decisions and 8 model calls; the episode took about 2 minutes. GPU
  memory peaked at 10.4 GB for Isaac Sim plus 8–10 GB for the perception
  services.

## Choosing a task

`--task <name>`. The tasks ORA runs:

`align_blocks` `arrange_largest_number` `build_tower` `classify_objects`
`classify_objects_by_language` `cover_blocks` `deposit_coin` `fasten_screws`
`fill_egg_holder` `fill_pen_holder` `fold_clothes` `general_pickup` `hang_mugs`
`imitate_sorting_sequence` `insert_key` `insert_tubes` `make_kong` `make_toast`
`match_and_pick_from_conveyor` `organize_table` `pack_objects_into_box`
`pick_from_conveyor_by_image` `play_stacking_toy` `play_tic_tac_toe`
`play_Xylophone` `plug_in_charger` `pour_balls_into_vase` `pour_by_language`
`pour_liquid_into_cup` `press_by_number` `push_T` `put_bottles_into_dustbin`
`solve_equation` `sort_nesting_dolls_by_size` `stack_blocks`
`stack_blocks_by_language` `stack_bowls` `store_laptop_and_headphones`
`store_tools_in_toolbox` `swap_blocks` `swap_T` `sweep_blocks`, and the
`*_random` variant of each generalization task: `arrange_largest_number_random`
`fold_clothes_random` `hang_mugs_random` `make_toast_random`
`pack_objects_into_box_random` `pour_liquid_into_cup_random` `push_T_random`
`sort_nesting_dolls_by_size_random` `stack_blocks_random` `stack_bowls_random`
`store_laptop_and_headphones_random` `sweep_blocks_random`

ORA pins RoboDojo `f57e163` and one asset tree, `1127bda`, for every task
(2026-11-14): `swap_T` runs from a fresh installation again and
`swap_blocks` can now succeed. Results for `swap_blocks` and `press_by_number`
(whose buttons now reset differently) and for `organize_table` and
`stack_blocks_by_language` (whose instructions changed) are not pooled with
runs at the earlier pin. The twelve `*_random`
variants (`stack_bowls_random` and the other generalization tasks') have
reviewed completion contracts since 2026-11-14, so `ora run` offers them too
([pinned revision](../environments/robodojo.md#pinned-revision)).

`general_pickup` is the cheapest first task. RoboDojo tasks are much harder
on average than LIBERO or RoboTwin; a sealed run with an official verdict, not
task success, is what a first run shows.

## While it runs

Enter sends the agent a message (the run is then marked human-assisted), Esc
stops it, Ctrl-C leaves the UI with the run still going. Replay any run with
`./scripts/ora watch RUN_ID`. See [Watch, steer and inspect](single-gpu.md#6-watch-steer-and-inspect).

## More

- [One-GPU guide](single-gpu.md): requirements, GPU and driver notes (R595, Hopper, Blackwell), troubleshooting.
- [Explicit RoboDojo installation](../environments/robodojo.md) and the [installer reference](../../ora/environment/backends/robodojo/installation/INSTALL.md), for a custom or multi-GPU host.
