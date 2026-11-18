# Changelog

What changed in each ORA release and what an upgrade needs, newest first. Each
release's [page on GitHub](https://github.com/fast-rua/ORA/releases) adds its
highlights and results.

## 0.1.0 — 2026-11-18

The harness is rebuilt around one motion base and a new agent core. An open
model can now write programs instead of single tool calls, run five benchmarks
and be steered in plain language while the robot works.

### Upgrade

- Node.js 22.19 or newer is required: the Pi core is ORA's only agent core.
  `ora doctor` checks it.
- An installed ORA updates by running the installer again:
  `curl -fsSL https://raw.githubusercontent.com/fast-rua/ORA/main/scripts/install.sh | sh`.
  In a source checkout, `git pull`, then `uv sync --all-extras --locked` and
  `npm ci --prefix ora-pi`.
- Store provider keys once with `ora providers key NAME`; they live in
  `~/.config/ora/pi-auth.json` (mode 0600), never in a run.
- RoboDojo is pinned at `f57e163` (XPolicyLab `5d0be14`) with one Hugging Face
  asset tree, `1127bda`, for every task. Install the new pins
  (`./scripts/setup-robodojo` or `ora setup --env robodojo`) and re-qualify
  each lane: card and task records made at RoboDojo `ee67a146` and assets
  `a14409d7` do not carry over. `ROBODOJO_SWAP_T_ASSET_ROOT` and the
  `assets-35efbc7d` checkout are no longer used; `swap_T` reads the same
  assets as every other task.
- The RoboDojo worker kernel and play kernel changed with motion profile v2
  (the jaw ramp, a pace's own jerk, `runtime.motion_profile`): re-qualify each
  lane (card, task, scene smoke, revalidate) and re-run the play-scene audit
  before the first run on this version.

### Added

- Code mode: the model can write a short Python program each turn that
  measures, drives several motions and checks its own result. Programs run in a
  bubblewrap sandbox with no network and can call only the model's own tools;
  every motion still passes the Supervisor. Code mode is RoboDojo's default.
- One motion base: inverse kinematics, collision-aware planning (cuRobo in a
  collision world built from public depth), path timing (TOPP-RA and Ruckig),
  the wrist-unwind guard and the refusal vocabulary, shared by all five
  simulators. Each robot passes a forward-kinematics parity check before the
  planner speaks for it. Where moves are planned on the policy side, a dry run
  shows the planned path as a camera-projected picture, and a refused move
  returns the same picture.
- Free play: `ora play` loads a scene with no task; you give instructions one at
  a time and can talk to the agent while the robot moves. `ora play --code-mode`
  runs it in code mode. Play sessions are sealed and never scored.
- The Pi agent core: the agent loop, request assembly, context management and
  provider calls run on the pinned Pi packages (`ora-pi/`); tools, robot
  semantics and authority stay in Python. `/model`, `/resume`, `/fork`,
  `/tree`, your own skills and `AGENTS.md` work as in a coding agent.
- A web page for every run: `ora web RUN` shows the conversation, each tool call
  or program and every camera; `ora replay RUN` replays a finished run on the
  same page and `--export out.mp4` makes a video. `ora cameras` draws the
  cameras beside the terminal.
- RoboCasa365 (the 50 target tasks, PandaOmron mobile manipulator) and RoboLab
  (120 tasks, DROID Franka in Isaac Sim 5.1), each in its own interpreter with
  the benchmark's own success check.
- Replay programs in sealed runs: `ora replay --check` re-executes the recorded
  actions. Token use is counted exactly from the provider's usage.
- A one-line installer, `ora setup --env <env>` for simulators (for RoboDojo it
  also installs git-lfs and bubblewrap when missing), and `ora run --gpu N`.

### Changed

- All 120 RoboLab-120 tasks now deliver a reviewed completion contract and pass
  launch preflight; the last 20 were promoted once their seed-0 witness matched.
  A review's witness plan may now state where an object goes and how it is turned
  (`put`, `park`) and how the pick-up grasp holds it (`order`, `depth`, `flip`,
  `shift`), with a `plan_fix` note that says why. The witness keeps its evidence
  path relative to `--evidence-root` when a directory below it is a link.
- RoboDojo moves and jaw commands are shorter (ARX X5 motion profile v2, the
  default for new RoboDojo plans; `runtime.motion_profile`). `fast` runs each
  joint at up to 4.5 rad/s and 60 rad/s² (was 4 and 40) and is certified at
  5000 rad/s³ instead of the unverified 1000, which had held its acceleration to
  about half the cap. A close approaches at the SDK limiter's full drive and keeps
  the 5 mm squeeze once its finger stops, skips the settle look on an arm already
  at rest, a release opens at full drive, and a jaw command ends after 2 still
  native actions instead of 3. On the first actions of five sealed RoboDojo runs,
  re-executed both ways, 24 fast moves took 218 native actions instead of 299 and
  13 jaw commands 56 instead of 117, every one ending as before. `normal`,
  `careful` and `speed_m_s` are unchanged; plans that bind no profile keep v1.

- LIBERO free-space moves the planner finds clear and nearly straight run along
  that straight line on OSC (motion profile v4, the default for new LIBERO
  plans): within 2 cm of the line and at most 45 degrees of hand turn. The plan
  then vouches for the gripper's path, not the arm's posture; a wrist force
  past 40 N stops the line, and the receipt's `arms.right.route_choice` says
  which route ran and why. Re-executing the actions of six sealed strict-limit runs took
  94 to 137 control steps instead of 117 to 160 under v3 and 235 to 334 under
  v2. Plans bound to v2 or v3 keep their routes.
- The LIBERO policy now sees its step budget: `remaining_budget` reports the
  plan's `max_sim_steps` as `native_actions_limit` and `native_actions_remaining`,
  every LIBERO receipt counts its control ticks as `native_actions`, and the
  `move_pose` text states the measured cost of a first move at each speed.
- LIBERO's planned moves follow their own pacing (motion profile v3, the
  default for new LIBERO plans). The global route paced every joint at 1 rad/s
  but ran RoboSuite's default joint controller, which follows about 0.16 rad/s,
  so the first move of an episode took a median 164.5 control steps for 35.5
  paced ticks. v3 gives the route's controller (kp 50) a 0.2 rad reach, aims
  each knot ahead by its own speed, paces at 0.55 rad/s and 15 rad/s² and ends
  the terminal settle once the arm rests inside the local 5 mm / 0.05 rad
  bound: six recorded first moves took 24 to 101 control steps instead of 50
  to 305. The `move_pose` text on LIBERO now states 31.5 deg/s as the joint
  ceiling. Plans bound to v2 keep RoboSuite's controller and their runs replay
  unchanged; a v3 run records its gains in its controller replay. Results under
  v2 and v3 are not pooled.
- RoboDojo `swap_blocks` can now succeed: upstream restores its spring
  button's drive target on reset, which also changes `press_by_number`'s reset
  buttons. `organize_table` no
  longer asks to open the drawer, and its completion contract's fourth clause
  follows the new instruction; `stack_blocks_by_language` gains the missing
  space in its instruction. Results for these four tasks before and after the
  bump are not pooled. Every other task's contract is unchanged and re-issued
  for the new pins (task-contract registry-v9, layout-portability v4, layout
  pools v2).
- The twelve RoboDojo `*_random` variants are admitted by the runtime with
  their base task's native-action limit and physics device, and each has a
  reviewed completion contract (task-contract registry-v10, the default, and
  layout-portability v5): the task roster, the composer and `ora run` offer
  all 54 RoboDojo tasks, a variant on its own layouts 0-9 (`hang_mugs_random`
  all but layout 5). Eleven variants deliver their base task's contract text
  unchanged; `sweep_blocks_random` scores bricks, earbuds and paper balls and
  has its own. registry-v9 and layout-portability v4 stay in the tree for
  specs composed before v10.
- A RoboDojo RunSpec on the synchronous toolkit surface gets the
  `budgets.max_robot_decisions` it names; a value above 100 was silently cut
  to 100. A spec that names none still gets 100, and a free-play session still
  keeps at most 100.

### Fixed

- Tool-call runs on LIBERO-Pro, RoboTwin, RoboCasa365 and RoboLab
  (`context_lossless_v6`) no longer send a request their own estimate puts over
  the hard input limit. The scheme's guard is now the hard backstop: every
  request is estimated before it is sent, and one at or above L6's watermark
  (0.8 of the hard limit) has all but the latest two exchanges aged
  (reasoning, read-only tool results and tool-returned images), the open
  ActionStep's included, so compaction starts at that watermark even where L6
  has nothing to summarize; a request that still exceeds the hard limit ends
  the turn with `context_exhausted` before it is sent. A reply
  the output limit cut off before any tool call is sent without its reasoning.
  Seven RoboLab runs (2026-11-14) had ended `context_exhausted` this way. Runs and
  plans resolved before keep their behaviour.

### Removed

- The Python Native agent core.
- `ora-tui/` (the terminal UI is part of the Pi core), `experiments/`, `ops/`
  and `ci/`.
- The MkDocs site. The documentation is published at https://fastrua.ai/ora/docs/;
  `./scripts/docs serve` previews it locally with Blume.

### Known limitations

- RoboDojo: ORA reads the measured gripper opening where RoboDojo's standard
  observation gives the commanded one, so ORA's RoboDojo numbers are not on the
  official observation.
- In free play the agent's `bash` runs as you, with credentials removed and no
  sandbox; code mode is the sandboxed way to script.
- ORA installs as a Git checkout; there is no PyPI package yet. macOS runs the
  CPU LIBERO path only.

## Preview — 2026-10-15

The October preview, the first public tree; it carried no version tag.
Open-weight models run LIBERO, RoboTwin and RoboDojo tasks through ORA's tools,
from one command on one GPU, with a terminal UI to
watch and steer them.
