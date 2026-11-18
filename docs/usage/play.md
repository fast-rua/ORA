# Free play

`ora play` loads one random scene that this host has qualified, with no task,
and opens the run's terminal UI. You give the robot instructions one at a time;
the agent carries out each one, reports what it did and what it sees, and waits
for the next. You can also talk to it while it works: ask what it is doing, or
change your mind, and it answers or switches without stopping. The episode does
not end after an instruction.

In free play the agent is Pi itself, working as a coding agent with the robot
as a device. It has Pi's own conversation, compaction and model registry, Pi's
coding tools (read, write, edit, bash) and its `codemode` tool in the run's
workspace, and your Pi skills, prompt templates, `AGENTS.md` context files and
extensions. ORA adds the robot tools, the scene and the Supervisor, which keeps
authority over the world: one robot action at a time, each with its receipt and
evidence, whether the model calls a robot tool or a script does
([scripting the robot](#scripting-the-robot)).

```bash
./scripts/ora play --env synthetic                         # choose a provider profile, then play
./scripts/ora play --env synthetic --provider env          # Synthetic only: ORA_BASE_URL, ORA_MODEL, ORA_API_KEY
./scripts/ora play --env libero --provider PROFILE --service-report REPORT --confirm-launch
./scripts/ora play --env robotwin --prepare-only           # write the RunSpec, launch it later with --spec
./scripts/ora play --env robotwin --scene-set tabletop-v1 --provider PROFILE --service-report REPORT --confirm-launch
./scripts/ora play --env robotwin --scene shell_game_memory_3 ...   # one scene of a scene set (its seeds are drawn)
./scripts/ora play --env robodojo --provider PROFILE --service-report REPORT --confirm-launch   # an Isaac Sim tabletop
./scripts/ora play --env robodojo --scene kitchen_tidy ...           # one RoboDojo scene (its seeds are drawn)
./scripts/ora play --env robodojo --code-mode ...                    # ORA's code mode ([code mode](#code-mode))
./scripts/ora play --env yam --rig-mode sim --service-report REPORT --provider PROFILE   # the YAM station's model ([the YAM station](#the-yam-station))
./scripts/ora play --env yam --rig-mode sim --no-perception --provider PROFILE         # ... without SAM3 and Contact-GraspNet
./scripts/ora play --env yam --rig-mode read --service-report REPORT --provider PROFILE --confirm-launch    # the real cell, nothing moves
./scripts/ora play --env yam --rig-mode on --operator-present --service-report REPORT --provider PROFILE --confirm-launch
./scripts/ora play --env yam --rig-mode on --operator-present --auto-approve ...   # motions run without asking
```

## The cameras beside the terminal

Pi's terminal shows the conversation; `ora cameras` shows the run's cameras
in the pane beside it, two by two, redrawn at every captured frame (each
observation and each action's end), with how old the frame is and the
observation step the agent last read:

```bash
ora cameras --last          # in a second pane or split of the same terminal
```

It draws through the terminal's own image protocol: kitty's (kitty, Ghostty)
or iTerm2's (iTerm2, WezTerm). Another terminal, macOS Terminal among them, is
refused with that reason; over SSH, where the terminal cannot be recognised,
`--protocol kitty|iterm2` names it. Inside tmux the picture passes through to
the terminal outside: `tmux set -g allow-passthrough on` first. The command
reads the run's capture and writes nothing; `q` ends it.

## Keys and commands

| Key or command | What it does |
|---|---|
| Enter | Sends the line. Between instructions it starts the next one. While an instruction runs, the message joins it: the agent reads it before its next step, answers a question and keeps working, or takes a changed request as the new objective. |
| Alt+Enter | While an instruction runs, queues the line instead: the Supervisor holds it (a `queued_instruction`) until the running instruction ends, then it opens the next one. A notice says so, and the footer counts the queued ones (*1 queued*) until each starts. Between instructions it is Enter. |
| Esc | Stops the running instruction at its next boundary. A model reply already in flight is recorded and any motion it proposed still runs; after that, no further model call is made. The session then waits for the next instruction. On a real cell, Esc while the robot moves first ends that motion where the arms are (`/halt`), so the stop takes effect at once; Esc while the agent is still planning sends the same stop first, and the worker refuses the motion that reply proposes before anything moves (`operator_declined`, telling the agent to read your latest message). When the reply proposes no motion, the stop is withdrawn as the instruction ends (`ora cell stop-motion --withdraw`), so it never refuses a later motion. |
| `/reset` | Loads another of the scenes drawn for this session. |
| `/view` | Shows the operator view in the conversation: the environment's camera for you (a RoboTwin scene set's `operator_camera`, else RoboTwin's observer camera; a RoboDojo scene's third-person `operator_camera`; Synthetic's drawn front view), which the agent never receives. |
| `/quit` | Ends the session. ORA evaluates and seals the run. |
| `/stop` | Stops the run as an operator stop. The run stays incomplete, as any stopped run does. |
| `!command` | Runs a shell command in the run's workspace. The agent sees the command and its output (`!!` keeps both out of the conversation). |
| `@path` | Refers to a file. The agent reads it with its read tool. |
| `/model`, Ctrl+P | Chooses or cycles the model. ORA's provider profile is listed as provider `ora`; providers you have logged in to are listed beside it. Every request to the `ora` model carries the profile's `request_options` ([model APIs](../getting-started/model-apis.md)), as a benchmark run's requests do; a model of another provider gets none. |
| `/thinking` | Sets the reasoning effort. |
| `/login`, `/logout` | Manages provider credentials. They are stored in your user-level ORA config (`~/.config/ora/pi-auth.json`, or `$ORA_CONFIG_HOME`), never in the run. |
| `/compact` | Compacts the conversation with Pi's own compaction. Pi also compacts automatically, as your Pi settings say. |
| `/tree`, `/fork`, `/new`, `/resume`, `/clone`, `/import` | Move the conversation. The robot and the scene stay as they are, and the terminal says so. These are allowed only between instructions. |
| `/skill:name`, `/template args` | Your Pi skills and prompt templates, expanded when the instruction is delivered. |
| `/reload`, `/trust`, `/debug` | As in Pi. |
| `/share`, `/bug` | Upload the conversation, or a bug report about it, after a confirmation that says what is sent and where. Nothing is uploaded without that confirmation. |
| `/status`, `/help`, `/export`, Ctrl+O | As in [interactive sessions](interactive.md). |
| Ctrl+D, `/detach` | Leaves the terminal. The session waits, and `ora attach` returns to it. |

When a scene loads, its main camera image (or the operator view, when the
environment has one) appears before your first instruction. Every new
instruction starts from a fresh look at the world: before its first model
call the Supervisor takes a new observation the same way it observes a scene
it has just loaded (it reads the cameras and the arms' state and moves
nothing), and the model receives that observation, then your instruction, in
the same request. On a real cell this matters because people move things
between instructions. A message that joins a running instruction takes no new
look, since the model already receives an observation after every action. If
the look fails (a camera that does not answer, for example), the instruction
still runs on the last observation. The failure is recorded as an
`auxiliary_failure` with stage `play:instruction_observation`, and the model
reads a note after that observation: *ORA could not take a fresh observation
at the start of this instruction (…). The observation with it is the latest
one, step N, captured … ago; the scene may have changed since.* The footer shows the
scene (for example `scene 2/8`), the instruction count and what the agent does
now, with the seconds it has taken: *thinking… 4s*, *writing… 6s*, *running
vision_ground… 1s*, *moving… 2s* while the robot executes an action, and
*waiting for your instruction* between instructions. Once the model has
answered, a second line under it gives the model's speed and the loop
between robot actions: *45 tok/s · TTFT 0.8 s · loop 6.2 s (target 3 s)*,
the output tokens per second of generation and the time to the first token
over the last 10 model calls, and the median time from an observation to the
next one over the last 10 actions. The target is 3 s, or `ORA_LOOP_TARGET_S`.
`ora web`'s page breaks each loop down ([speed](remote.md#web-view)).

A message you send while an instruction runs is announced as queued (*joins
instruction 2; the agent reads it before its next step*), and appears in the
conversation once the Supervisor has delivered it, marked *you · joins
instruction 2 · delivered at step 7*; the agent reads it before its next model
call (a message that arrives while the agent proposes a robot action, or while
the robot executes one, is read with the observation after that motion), and
its reply follows. When the agent answers
your message in words only, that reply is shown as its *Answer* and the
instruction keeps running: the agent goes on with the task at its next step.
Only a reply with no new message of yours to answer is the instruction's
report. A reply the model's output limit cut off before any tool call (for
example, all of it spent on reasoning) is never the report: ORA asks the model
to continue on the same observation, as it does in a benchmark episode, and
Pi's terminal marks the reply *Response was truncated before completion*.
When the session's model calls or time leave no call to continue it, the
instruction ends *the model's reply was cut off at its output limit*. The end
of each instruction is announced, for example *done; the agent reported*, *stopped
(Esc)* or *its model calls ran out*, and marked in the conversation where it
happened (*■ instruction 2 stopped (Esc) · at step 9*).

The terminal is drawn as the project teaser's terminal pane, as `ora web`'s page
is: ORA's theme for Pi (grays, a teal accent for marks only, green for verdicts
only; truecolor where the terminal has it, the nearest 256-color palette entry
where it does not), a header with ORA's pixel-arm mark, the model, its effort and
the environment and task, the model's words after `●`, each robot tool call as
`● name(arguments)` with its result under a `⎿` corner, the status line above the
prompt (a teal spinner, *Moving on the robot… (12s · esc to interrupt)*), the
prompt after `❯` between two rules, and the footer `▸▸ mode · scene ·
instruction · queued · context` with `● effort · model` on the right (the
counts and budget give way on a narrow terminal; `/status` has them all). The
theme is set for the session only, never in your Pi settings, and your
terminal's own background stays; `ORA_TUI_THEME=pi` keeps Pi's theme and look.

ORA's own notices, footer, header and help are in Chinese when `ORA_UI_LANG=zh`
(or a `zh*` locale, `LANG=zh_CN.UTF-8`); `ORA_UI_LANG=en` keeps English. Pi's
own interface stays as Pi draws it. The web page (`ora web`) has its own
English/Chinese switch ([web view](remote.md#web-view)).

A finished session replays on the same page with `ora replay RUN_ID` (the
waiting at 16×, each robot motion in real simulated time; like `ora web`, it
needs the page built once with `ora web setup`), and `ora replay RUN_ID
--export out.mp4` makes it a video at those rates, with Playwright's Chromium
and ffmpeg installed ([replay a finished run](remote.md#replay-a-finished-run)).

### Real cells

When the plan drives a real robot cell (its `runtime.kind` is a real-cell
backend, today `yam`, and its `runtime.rig_mode` is `read` or `on`), the
terminal adds commands that act on the arms:

| Command | Runs | Confirm first |
|---|---|---|
| `/home` | `ora cell home` | no |
| `/hold` | `ora cell hold` | no |
| `/float` | `ora cell float` | yes (arms will be guided by hand) |
| `/release` | `ora cell release` | yes (arms go limp) |
| `/estop` | `ora cell estop` | yes (every motor disabled, arms fall) |
| `/diagnose` | `ora cell diagnose` | no |
| `/halt` | `ora cell stop-motion --by "play terminal"` | no (the motion in progress ends where the arms are and they stay held; the agent gets a fresh observation) |
| `/approve [value]` | `ora cell approve --id ID [--value V]` | yes (shows the robot's waiting question and its id) |
| `/decline [reason]` | `ora cell decline --id ID --reason …` | yes (shows the question and the reason) |
| `/yolo on` · `/yolo off` · `/yolo` | `ora cell auto-approve on\|off\|status` | `on` only (motions will run without asking) |
| `/mode ask\|safe\|auto` · `/mode` · Shift+Tab | `ora cell mode ask\|safe\|auto` (`/mode` alone reads the mode, runs nothing) | `auto` only (every motion will run without asking) |

`/halt` is the way to change course mid-motion: without it the Supervisor
reads a stop (Esc) or a message only once the motion it executes has ended,
and a motion may take several seconds. The worker reads the stop at every
control tick and ends the motion within a few milliseconds; between motions it
does nothing. Esc while the robot moves calls it before it stops the
instruction. Sent while the agent plans (Esc, or the page's *Stop & send*), the
same stop refuses the motion the agent's reply proposes at its approval; when
the reply proposes none, the front end withdraws it once your message reaches
the agent (or the instruction ends) with no motion started in between.

They act at once, also while the robot executes a motion: the terminal runs
`scripts/ora cell <verb> --json` itself (with `--cell` when the plan has
`runtime.cell_config`), the same command the cell board's buttons and the shell
use, instead of putting them in the play inbox, which the Supervisor reads only
between model decisions. A command never reaches the agent. A command that
asks first runs only after `y` (any other key cancels it) and then passes
`--yes`. The terminal says *running…*, then shows the result or the error in
the conversation as an operator entry (*✓ /home · ora cell home · done in 38s*),
and another command can start meanwhile (`/estop` while `/home` runs). The
agent trace records each one as an operator shell command and its output, kept
out of the conversation, as a `!!` line is. In a run that drives no real cell
(`sim`, or a simulator) they are not offered: typed, they are answered that
they apply to real cells only.

`/approve` and `/decline` answer the question the worker waits on (enable the
arms, a motion, release arms that could not get home): the terminal reads the
cell's approval broker itself, shows the oldest waiting question with its id,
plain words and seconds left, and after `y` answers that id; approving is the
confirmation, nothing asks again. A guided calibration's questions are
answered the same way: `/approve` when a `task` is done, `/approve <value>` for a
`value` (a value is refused for any other question, and a `value` question
without one), `/approve` or `/decline` to save a `write` result or not. On rig
mode `on` it also says once, as soon as
a question appears, *左臂动作等待批准（id）· 输入 /approve 或 /decline 答复*. A
question the board or the shell answered first is refused by id
([real cells](real-cells.md)).

`/yolo on` turns on auto-approve for the worker driving the arms: after one
`y`, each motion it plans runs at once instead of waiting for `/approve`.
Enabling the arms and releasing them are still asked. `/yolo off` turns it off
again, and `/yolo` alone says whether it is on. Starting the session with
`--auto-approve` turns it on from the first motion. Each auto-approved motion
is still recorded in the broker as an answer (`by: "auto-approve (play
terminal)"`), and the agent gets the same receipt it gets for a motion you
approved ([real cells](real-cells.md#auto-approve)).

The footer starts with the cell's permission mode: *▸▸ ask · every motion
asked* (每个动作都问), *safe · small motions run* (小动作自动: a motion inside the
safe limits runs without asking, any other waits) or *auto · every motion runs*
(全部自动, in bold red; the same as `/yolo on`). `/mode` says it in full, with
`safe`'s limits and who set it; `/mode ask|safe|auto` switches it; Shift+Tab
cycles ask → safe → auto → ask (on rig mode `read`, ask ↔ safe), and switching
to `auto` asks y/N either way. On a real cell Shift+Tab is taken for the mode,
not Pi's thinking-level cycle; `/thinking` still sets the reasoning effort.
Each motion that answered itself is said once as a notice, with the numbers
safe mode checked ([real cells](real-cells.md)).

#### On the web page

`ora web` has the same controls for a run on a real cell, and none for any
other run ([web view](remote.md#web-view)). Its gateway reads the cell's
own files for the state, as the board and the terminal do, and runs the
terminal's own command lines for the actions (`scripts/ora cell <verb>
--json`, with `--by "web page"` for an answer, the switch, the mode or a stop), so a command
from either front end does the same thing:

| Real-cell feature | Play terminal | Web page |
|---|---|---|
| A waiting question appears: its title, plan in plain words, id and seconds left | ✓ a notice (rig mode `on`), then the y/N prompt of `/approve` | ✓ a card above the box you type in, for as long as it waits |
| Approve or decline it, by its id, after a confirmation; a decline carries a reason | ✓ `/approve`, `/decline [reason]` | ✓ the card's *Approve…* and *Decline…* (reason field), or typed `/approve`, `/decline [reason]` |
| Auto-approve: its state, on after a confirmation, off at once | ✓ `/yolo`, `/yolo on`, `/yolo off` | ✓ the switch in the cell bar (rig mode `on`), or typed `/yolo …` |
| The permission mode: shown always, `auto` after a confirmation | ✓ the footer, `/mode …`, Shift+Tab | ✓ the footer: a click opens its menu (`safe`'s limits shown), or typed `/mode …` |
| A motion that ran without asking (safe mode or auto-approve) | ✓ a notice, once each | ✓ the latest, in the cell bar |
| Home, Hold, Diagnose at once | ✓ `/home`, `/hold`, `/diagnose` | ✓ the cell bar's buttons (Home first), or typed |
| End the motion in progress where the arms are, at once | ✓ `/halt`; Esc while the robot moves | ✓ the cell bar's *Stop here*, typed `/halt`, *Stop & send*; Esc and *Stop instruction* while the robot moves |
| Refuse the motion the agent is still planning, before it moves (withdrawn when none comes) | ✓ Esc while the agent plans | ✓ *Stop & send*, Esc and *Stop instruction* while the agent plans |
| Float, Release, E-stop after a confirmation | ✓ y/N (only `y` confirms) | ✓ a dialog (only its *Yes* button confirms; Enter and Esc say no), or typed |
| A command runs while another one does (`/estop` while `/home` runs) | ✓ | ✓ |
| Each outcome, the other front end's too | ✓ an entry in the conversation; the page's is marked *from the web page* | ✓ the cell bar's list, marked *terminal* or *web page* |
| Recorded in the agent trace as an operator shell command and its output | ✓ | ✓ the gateway hands each outcome to the run's agent process, which records it the same way (`via: "web page"`) |
| The held arms' alerts | ✓ once each as a notice, critical ones as errors (rig mode `on`) | ✓ a banner each while it lasts, critical ones as errors |
| The worker that held the arms has ended | ✓ an error notice (rig mode `on`) | ✓ an error banner, until another worker holds the arms |
| Not offered without a real cell; typed, answered that they apply to real cells only | ✓ | ✓ |

The page asks first exactly where the terminal asks y/N, and its request must
say so (`confirmed: true`): the gateway refuses an unconfirmed E-stop, Float,
Release, answer, *auto-approve on* or *mode auto*. The page's controls keep working after
the run's agent process has ended, as long as `ora web` runs: the cell
outlives the run. Two tests keep the front ends together: a command added to
the terminal's table (`ora-pi/src/play/cell.ts`) without the page's
counterpart fails `ora-pi/web/test/cell.test.ts`, and without the gateway's
route fails `tests/ora/app/test_web_cell.py`.

## Free play and benchmark runs compared

| Command or capability | Free play | Benchmark and scored runs | Why |
|---|---|---|---|
| Typed text | the next instruction, or a message to the running one | a steer, marked as human intervention | Free play has no task; a benchmark run's task is fixed by its plan. |
| `/model`, Ctrl+P, `/scoped-models` | available | refused: the plan binds the model | Switching changes what produces every decision. |
| `/thinking` | available | refused: the plan binds the reasoning effort | It changes what the model is asked to do. |
| `/login`, `/logout` | available | available | Credentials live in the user-level ORA store, outside every run and its evidence. |
| `/compact` | Pi's own compaction, plus automatic compaction | refused | ORA's context policy owns a benchmark run's context. |
| `/tree`, `/fork`, `/new`, `/resume`, `/clone`, `/import` | available between instructions; the conversation moves, the world does not | refused | A benchmark run's history must be the one the robot lived. |
| `!` shell, `@path` | available | refused | The agent's inputs are fixed by the plan. |
| Pi's coding tools (read, write, edit, bash) | available, in the run's workspace | not advertised | The ToolSurface is part of the plan's identity. |
| Pi's `codemode`, `ora-robot` from the shell | available; a script's robot calls go through the Supervisor (with `--code-mode`, ORA's own code mode instead) | not available | A benchmark run's actions are the model's own tool calls. |
| Skills, prompt templates, `AGENTS.md`, your extensions | loaded from your Pi config | not loaded | They would change what the model sees. |
| `/reload`, `/trust`, `/debug` | available | refused | They reload or reveal resources a benchmark run does not use. |
| `/share`, `/bug` | after a confirmation | refused | They upload the session outside the run. |
| Esc | stops the running instruction | stops the run (operator stop) | A free-play session continues; a benchmark episode does not. |
| `/quit` | ends the session; ORA evaluates and seals it | detaches the terminal | The session is the operator's to end. |

## Scenes

A scene is one reset of one environment task. `ora play` never loads a scene
this host has not qualified.

- **Synthetic** composes the quickstart's world.
- **LIBERO and RoboTwin** take their scene from
  `ora.orchestrator.play.scenes`, which draws a scene at random from what this
  host has qualified and loads it without its task, with the backend's play
  binding: the official success check is withheld, so nothing ends a scene as
  a success. The scenes `/reset` may load are drawn from the same task's (and
  RoboTwin scene configuration's) scenes that the same qualification record
  admits. `--service-report` binds the perception services, as `ora run`
  does.
- **RoboDojo** offers only ORA's own Isaac Sim tabletops
  ([below](#isaac-sim-tabletops-robodojo-scene-sets)); its official tasks are
  not loaded in free play.

A worker cannot change task, so every scene of one session is the same task,
with another seed or initial state. `--scenes N` (default 8, at most 32) sets
how many are drawn, and `--draw-seed` repeats a draw. The plan records the
current scene (`play.scene`), the scenes (`play.scenes`) and the draw
(`play.draw`). A real environment's RunSpec is written under
`outputs/ora-play/specs/` (with the RoboTwin seed manifest it names);
`--prepare-only` stops there, and `ora play --env ENV --spec FILE
--confirm-launch` launches it as it is.

`--scene NAME` plays one task (a LIBERO or RoboTwin task name, or a scene of
a scene set) and still draws its seeds or initial states.

No success condition is latched in free play, and no verdict is recorded. If
the environment still ends a scene (its step limit), the running instruction
ends, and only `/reset` or `/quit` can follow.

### Tables with many objects (RoboTwin scene sets)

A benchmark task's scene holds the few objects its task needs. For open-ended
instructions ("pick up the cup and put it on the plate, then put the plate
where the cup was"), ORA also has its own RoboTwin scenes: tables with many
objects and no task at all. `--scene-set tabletop-v1` (or `household-v1`) draws
among a set's scenes instead of the 50 official tasks, and `--scene NAME` names
one (it implies its set).

| Set | Scene | What is on the table |
|---|---|---|
| `tabletop-v1` | `table_setting_sequence` | a plate, a bowl and a cup in a row near the robot; a fork, a knife and a coaster beyond them |
| `tabletop-v1` | `shell_game_memory_3` | three upright cups in a row and an orange table-tennis ball in front of one of them |
| `tabletop-v1` | `obstacle_rearrangement_6` | six coloured blocks near the robot, two low walls with a gap and a divider, and six matching flat pads beyond the walls (the seed shuffles the blocks' colours) |
| `household-v1` | `breakfast` | a plate, a bread basket and a bowl at the back; a loaf of bread, a mug, a bottle of juice and a box of coffee in front; a jar of jam on the robot's left and an apple on its right |
| `household-v1` | `kitchen` | a cutting board, a bamboo steamer and a red kettle at the back; a tomato, a bottle of soy sauce, an onion and a blue tub in front; a jar of sauce on the robot's left |

The scenes are rebuilt from the asset lists and camera images of three of the
six `robotwin-long-horizon-v1` prototypes, whose source was lost; they look
like the prototypes, not identical to them, and each seed moves every object a
little. They use objects from RoboTwin's asset library (plate, bowl, cups,
fork, knife, coaster, ball) and RoboTwin's box primitive for the blocks, walls
and pads; the walls, divider and pads are fixed to the table. The embodiment,
cameras, table and scene configuration (`demo_clean`) are RoboTwin's own, and
the RoboTwin checkout and its pinned revision are not changed: the worker loads
the scene from ORA (`ora/environment/backends/robotwin/play_scenes/`) instead
of from RoboTwin's `envs/`. A scene has no instruction, success check or
expert, and the agent is told nothing about it.

`household-v1` is ORA's own: a breakfast table and a kitchen counter, each with
containers to put things in or on (the plate, basket and bowl; the cutting
board, steamer and tub) and eight or nine objects in all. The big ones stand at
the back, the small ones (at most 8 cm across where a gripper closes, nothing
thin or flat) in front and at both sides, 15 cm or more apart, all where an arm
reaches them from above, so both arms are used and an arm carrying something
has a lane. Its objects weigh what they would (a mug 300 g, an apple 180 g, the
kettle 600 g) with the inertia of that mass and a friction of 0.6/0.5:
RoboTwin builds every library object at 10 g with the rotational inertia of a
solid body at 1000 kg/m³ and a friction of 0.3, so pushed it slides and spins slowly instead
of tipping, and pinched it can be thrown across the room. Nothing global
changes: the robot, the table and every other scene keep RoboTwin's physics.
Jaws that close on a thin part and open again can still throw an object (no
mass prevents a deep penetration from resolving fast): in the first breakfast
session they closed on the milk carton's sealed top fin and threw the carton
off the table, so a bottle of juice replaced it. Each scene frames the operator
camera on its whole table and both arms.

As with every scene, `ora play` loads a scene of a set only at a seed this host
has qualified: a play-scene audit must admit it. The audit builds each scene
in the RoboTwin worker on the host's RoboTwin slot and records, for each seed,
that the scene builds, settles (every object at rest, on the table, inside the
arms' workspace and not interpenetrating anything) and renders every camera.
Run it once per host, and again after a change to the scene modules:

```bash
./scripts/ora play audit --env robotwin --scene-set tabletop-v1            # seeds 0, 1, 2
./scripts/ora play audit --env robotwin --scene-set tabletop-v1 \
    --scene shell_game_memory_3 --seed-list 0 1 2 3 4 5 --resource-profile SLOT
```

It needs `ROBOTWIN_ROOT` and `ROBOTWIN_PYTHON` (the RoboTwin checkout at its
pinned revision, clean) and a registered execution host, holds the slot's
resource lease while it runs, and builds one scene at a time. Records and
camera images land under
`outputs/ora-play/qualification/play-scene-audits/<UTC time>/tabletop-v1/`. A
record stops admitting its seeds when the scene modules change, so `ora play`
then asks for a new audit. The scenes `/reset` may load are other audited seeds
of the same scene.

**Operator camera.** A scene of a set has ORA's own camera for you,
`operator_camera`: 1280 × 720, ray-traced like RoboTwin's cameras (8 samples
per pixel), from your side of the table at three quarters, with both arms and
both rows of objects in frame (`play_scenes/base.py`, `OPERATOR_VIEW`). `/view`
and the web page show it instead of RoboTwin's 320 × 240 observer camera, and
the page plays the robot's motion through it. It is added to the SAPIEN scene
after the objects and never to RoboTwin's camera manager, so the agent's
cameras, observations and point clouds are unchanged (the four policy cameras'
images and depth stay byte for byte what they were without it). The play-scene
audit renders it with the other cameras.

**Recorded motion.** A scene-set plan records its execution capture at
30 fps of simulated time (`runtime.execution_capture` `{enabled: true, fps: 30}`,
`robotwin_play_scenes.PLAY_EXECUTION_CAPTURE`): a frame every ninth 4 ms
physics step, each with the head and front cameras (320 × 240) and the
operator camera (1280 × 720). Only those three cameras are rendered for a frame
(RoboTwin's `update_picture` also renders both wrist cameras), and a frame is
rendered only once the capture's writer has caught up, so no frame is dropped.
Benchmark plans, and play on an official RoboTwin task, keep 10 fps and the
head and front cameras.

**Settling after each action.** RoboTwin advances physics only inside an
action, so an object the arm knocked over used to hang in mid-fall while the
model thought (in a recorded session a knife and a plate thrown off the table
stayed in the air for 11 s). After each action of a scene-set scene the world
keeps running with every arm and gripper holding its last command, 10 physics
steps at a time, until over one such window no object moved more than 1 mm or
turned more than 1°, at most 2 s of simulated time
(`robotwin_play_scenes.PLAY_POST_ACTION_SETTLE`); an object that fell to the
floor no longer keeps it going. The settle is part of the action's recorded
motion (its frames carry the action's id), changes no receipt, does not count
toward the scene's step limit, and is logged per action in
`native/post-action-settle.jsonl`.

What it costs, measured on an H200 with a scripted sweep through the cup of
`table_setting_sequence` (four arm moves, about 5 s of simulated time): a frame
takes about 0.39 s of wall time (0.26 s to ray-trace the three cameras, the
rest the writer's PNG encoding, which shares the interpreter with the
simulation), so the moves took 102 s instead of 57 s at 10 fps; the capture
holds 25.6 MB per simulated second instead of 1.7 MB (the operator camera's
frames are about 0.7 MB each). In a live session (65 actions, 75 s of simulated
motion) an action took 12.5 s of wall time at the median, against 5.1 s in one
recorded before the change (83 actions, 85 s), and the capture held 2.0 GB
against 0.16 GB. The replay and its video play that motion by simulated time,
so only a live session's waiting grows.

### Isaac Sim tabletops (RoboDojo scene sets)

RoboDojo's free play is ORA's own scene set `isaac-tabletop-v1`, the default
of `ora play --env robodojo` (`--scene-set isaac-tabletop-v1` names it, and
`--scene NAME` names one scene, which implies its set). Each scene is
RoboDojo's standard table, with its two ARX X5 arms and three cameras, set with
objects from RoboDojo's asset library. There is no task, instruction, reward or
success check, and the agent is told nothing about the scene.

| Scene | What is on the table | Physics |
|---|---|---|
| `breakfast` | a toaster and an egg carton with two eggs at the back, a plate with two slices of bread, a mug, a juice carton, a yogurt and a bowl of grapes | CPU |
| `desk` | a keyboard, a mouse, a mug, a pen holder with a pen, a stapler, a tape dispenser, a phone and a book | CUDA |
| `toys` | a teddy bear, a chick, a wooden toy, an owl, matryoshka dolls, three building blocks (two stacked), two toy cars and a big toy block | CUDA |
| `kitchen_tidy` | a kettle, a pot and an open box at the back, two bowls, a mug, a pack of noodles, a peeler and an oil bottle, and a small plate on a large plate | CUDA |
| `blocks` | nine blocks in nine colours in a loose 3 x 3 grid: seven 35 mm (red, orange, yellow, green, cyan, blue, magenta) and two small, about 20 mm (white, dark olive) | CUDA |

`ora play` prints the chosen scene's description for you; it is not part of the
plan and never reaches the agent. The toaster and the egg carton are fixed to
the table; everything else can be picked up.

**Layouts.** Public RoboDojo replays saved layouts and has no layout sampler,
so ORA generates them: each scene is a YAML in RoboDojo's task-config format
(`ora/environment/backends/robodojo/play_scenes/scenes/<set>/<scene>.yml`;
the set's `index.yml` gives each scene's label, physics device, settle time,
assets and description), and
`ora/environment/benchmarks/robodojo_play_layouts.py` turns it and a seed into a saved layout that
RoboDojo's own scene manager loads. The same YAML, seed and pinned assets give
the same layout on every host. Objects keep 1.5 cm from each other and the
`gripper_keep_out` clearance (8 cm) from the footprints of the two grippers'
start poses.

**Physics device.** A scene with an articulated object (the toaster and the
egg carton in `breakfast`) runs on IsaacLab's `cpu` physics, as RoboDojo's
own `make_toast` does: on `cuda:0` RoboDojo reads the articulation's joint
properties through numpy and crashes. Rigid-only scenes run on `cuda:0`. The
plan binds the scene's device. After each reset the worker settles the scene
before the first observation (12 s of simulated time for `breakfast`, whose
toaster timer knob relaxes slowly, 6 s otherwise: a pen leaning in its holder
still turns after 2 s).

**Operator view.** A RoboDojo scene has its own third-person camera for you
(`operator_camera`, 1280 × 720: the table at three quarters from its far side,
both arms in frame). `/view` and the web page show it after every observation,
and the page plays the robot's motion through it, recorded at 30 frames per
simulated second; the agent never receives it.

**Between actions.** After each action the scene runs on, the arms holding
still, until everything on the table is at rest (at most 2 s of simulated
time), so a cup the arm knocks over falls and rolls to a stop on camera and in
the next observation, instead of hanging in mid-air until the next action. The
recording of an action includes this settle.

**Reset.** `/reset` loads another audited seed of the same scene. The worker
closes the simulation and builds the next layout in the same Isaac Sim process,
on CPU and CUDA physics alike, which takes about a minute (longer for
`breakfast`).

**Tools.** The agent gets the default RoboDojo tools without `check_task` and
`finish_episode`, so `--service-report` (the perception services) is required,
as for `ora run --env robodojo`. The moves are planned and timed as in a
benchmark episode: the worker starts the shared motion planner on its card
(free moves take its collision-free route in the depth the cameras last saw,
and the reach notes of `move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`
and `grasp_plan` come from its IK) and times every move with the timing
worker, so `CUROBO_V2_PYTHON`, `CUROBO_V2_ROOT` and `ORA_TIMING_PYTHON` must be
set, and launch preflight checks them. A planner that does not start stops the
session at its start. The scene runs on its own worker backend
(`robodojo_play`) and does not touch the RoboDojo worker that benchmark runs
and their qualifications use.

**Audit.** `ora play` loads a scene only at a seed this host has audited. The
audit starts the play session itself in the RoboDojo worker on the host's
RoboDojo slot, loads each seed through the same reset `/reset` uses, and
records for each seed that:

- every object exists, rests on the table and stayed within 3 cm (and 20° of
  tilt) of its generated pose while the scene settled;
- over a further 2 s nothing moves more than 5 mm or turns more than 2°, and no
  articulation joint drifts;
- every object's centre is in the arms' reach envelope (the official
  RoboDojo layouts' object envelope, within 0.613 m of the nearer arm base); an
  object resting in or on another one (a pen in its holder, grapes in a bowl) is
  checked through the object it rests on;
- each movable object can be approached the way the agent's tools approach
  it: from at least one of `align_plan`'s sides (top, left, right, front,
  back), either arm reaches `align_plan`'s own pose for that side (link6 at
  the default 10 cm `standoff_m` from the object's box centre, the jaw at
  any of 12 rolls every 30°; read-only IK, asked after the seed's
  observation). The record keeps every side's reachable arms and rolls;
- every object on the table keeps 3 cm from the measured footprint of each
  gripper at its start pose;
- no object interpenetrates another or the table by more than 3 mm, and none
  touches the robot (PhysX contact reports, which only CPU physics delivers: a
  CUDA scene gets a second pass on CPU physics);
- the three cameras render.

```bash
./scripts/ora play audit --env robodojo                         # every scene, seeds 0-4
./scripts/ora play audit --env robodojo --scene desk --seed-list 0 1 2 \
    --resource-profile SLOT
```

It needs `ROBODOJO_ROOT`, `ROBODOJO_ASSET_ROOT` and `ROBODOJO_PYTHON` (the
pinned, clean RoboDojo installation), `CUROBO_V2_PYTHON` and `CUROBO_V2_ROOT`
(the session starts the shared motion planner, as a play session does), a
registered execution host with a
RoboDojo slot, and a committed checkout; it holds the slot's lease and runs one
Isaac Sim process at a time (about 2 minutes for a scene's first seed and
1 to 1.5 minutes for each further seed and pass). Records, worker reports and
camera images land under
`outputs/ora-play/qualification/play-scene-audits/<UTC time>/isaac-tabletop-v1/`.
A record binds the play kernel it ran (the RoboDojo worker code and the play
code at that commit) and the scene's own sources: its entry in `index.yml`
(everything but the description), its YAML and the layout generator. Editing
the play code or the generator retires every scene's records; editing one
scene's YAML or index entry retires only that scene's, so after a scene edit
`ora play` asks for a new audit of that scene alone
(`--scene NAME`). Editing a description retires nothing.

### The YAM station

`ora play --env yam` plays on the bimanual I2RT YAM station. A real cell has no
scene draw: the scene is whatever stands on the table, and `/reset` starts a
new round from the arms' rest pose (the worker's `restart_round`) rather than
loading another scene. `--rig-mode` is required and picks the cell:

| `--rig-mode` | What runs | What a motion does |
|---|---|---|
| `sim` | the station model in MuJoCo (`YAM_PYTHON`, `MUJOCO_GL=egl`) | executes on the model |
| `read` | the real cameras and joint encoders; the motors are never enabled | is answered with its plan and refused as `dry_run_read_only` |
| `on` | the real arms | waits for the operator's approval (board, `ora cell approve`, `/approve`); the operator may decline it with a reason. With `--auto-approve` (or `/yolo on` during the run) it executes without asking |

`read` and `on` touch the real cell, so they launch only with
`--confirm-launch` and name the registered execution host (`--host`, else
`ORA_EXECUTION_HOST`); `on` also needs `--operator-present`, declaring a person
beside the cell with a hand on the stop. `--cell CELL.json` passes the real
cell's config (`ora.cell-config.v1`) to the worker as `runtime.cell_config`.
`--auto-approve` (`on` only) sets `runtime.auto_approve_motions`. Motions then
run without asking, but the operator still approves enabling the arms once, and
a blocked way home still asks before releasing the arms in place. A person must
still stand beside the cell with a hand on the stop.

The station is no baseline instance: the backend composes the whole RunSpec
(`ora/orchestrator/backends/yam/play.py`). The policy semantics are the default
baseline's and the prompt layers those of the surface's bundle with the
free-play episode task. A session names its perception services or declines
them, so one command line always resolves one surface:

| Option | ToolSurface | Prompt bundle |
|---|---|---|
| `--service-report REPORT` (the default surface) | `yam_perception_v1`: observe, mark_point, geometry_query, compute, `vision_segment` and `vision_ground` (SAM3), `grasp_plan` (Contact-GraspNet), set_gripper, move_to, move_pose, recover_pose, wait_ticks, report_issue | `current_v4/yam-station-perception-native_api-v4.yaml` |
| `--no-perception` | `yam_direct_v1`: the same without the three perception tools | `current_v4/yam-station-native_api-v4.yaml` |

`REPORT` is a readiness report of the resident services (`ora services
qualify`), at most 24 hours old, naming the endpoints `ORA_SAM_PORT` and
`ORA_GRASP_PORT` give on this machine: services on another host reach the cell
over an SSH tunnel to its loopback ports. `grasp_plan` fuses the target's camera
with `overhead` and the planning arm's wrist camera and compiles each proposal
to that arm's TCP (`yam_linear_4310_tcp`: the centre of the two fingertip
facets; Contact-GraspNet's 103.4 mm origin-to-fingertip depth lands on it; 93.9
mm opening ceiling). It has no `grasp_execute` and no planning-time IK
annotation: the agent grasps with `move_pose` and `set_gripper`, and the
station's executor plans and admits every motion. On `yam_perception_v1` each
observation also delivers the public robot facts (each camera's calibration and
TCP pixels, each arm's TCP pose, world axes and measured jaw separation), and
the worker verifies the loaded station model's finger geometry against the
public descriptor at every round's start. A fixed camera the cell has not
calibrated has no extrinsics, so the perception tools refuse it.

A station session runs under the `realtime` [latency profile](../reference/latency-profile.md),
simulated cell included, since it rehearses the real one: no execution capture during
motions (the world view's camera panels then show the observations only) and no preview
picture with a dry run or a refused move. `--latency-profile sim_benchmark` turns both back on;
latency measured that way is not the cell's.

Four station behaviours are switches, each off by default while its effect on
the cell is measured:

| Option | What changes |
|---|---|
| `--scripted-moves` | Adds the episode-task layer `ora/policy/environments/yam/play-scripted-moves-v1.md`: a pick and place the evidence already fixes runs as one `codemode` script that checks every receipt and stops at the first surprise, instead of a turn per move. It names the perception tools, so it needs `--service-report`. |
| `--reach-query` | Sets `runtime.reach_query`: a `target_ik_unreachable` refusal also names `nearest_reachable_position_m`, the closest position at the target's height and approach that solves (within 20 cm, up to about 0.8 s of IK on the worker), and `nearest_reachable_offset_m`. Adds the layer `play-reach-query-v1.md`, which tells the model to read it rather than search a grid of dry runs. |
| `--motion-timing` | Sets `runtime.motion_timing`: ORA's shared timing worker (TOPP-RA + Ruckig, under `ORA_TIMING_PYTHON`, which preflight starts) times each arm path under the station's own speed caps, with acceleration limits and a start and stop at rest, instead of the station's fixed-speed timing; moves take about 1.2 to 1.3 times as long ([motion speed](../reference/motion-speed.md#yam)). Adds no layer. |
| `--motion-planner` | Sets `runtime.motion_planner`: ORA's shared cuRobo V2 planner (under `CUROBO_V2_PYTHON` with its source `CUROBO_V2_ROOT`, which preflight starts) plans the arm paths as on the simulators: a free-space move is one collision-free plan in the depth of the station's calibrated cameras, with the other arm in it, instead of the station's lift, traverse and descend; a precision or contact move stays a straight line, solved by the shared IK and checked against that depth. The station's timing, per-tick checks and refusal words stay ([planners](../../ora/environment/planners/README.md#the-yam-station)). Adds no layer, and the tools' descriptions still describe the station's own planning. |

The agent receives the four cameras (`overhead`, `front`, `wrist_left`,
`wrist_right`) and both arms' measured TCP poses with each observation. There
is no evaluator: the verdict is null and the operator judges each
instruction.

You see the same four cameras in the world view (`/live`) and on `ora web`'s
page: the worker records their original pixels for you, never for the agent
(`native/execution-capture`, `ora/environment/backends/yam/capture.py`). On
`sim` the station model renders them at the scene's start, at the start and
end of each action, ten times a second of control time while the arms move,
and at the end of each round. On `read` and `on` nothing reads the real
cameras while the arms can move: the frames are the pixels each observation
read (the agent observes after every action and at every new instruction),
the scene's start and an action's end are the next observation's, and only
the end of a round reads the cameras once more. A real cell's camera panel
therefore changes when the agent looks, not during a motion; the cell board's
live view streams them.

Each image the model receives is named in the text right before it:

- An observation's images come in the order `overhead`, `front`,
  `wrist_left`, `wrist_right`. Each one follows a line such as
  `INITIAL ACTUAL · sequence 4 · overhead · latest at this message`
  (`ACTION RESULT ACTUAL` after an action). These are the cameras' raw images,
  with nothing drawn on them.
- `observe` returns one image per requested view, in the order of its `views`
  (all four in the order above when `views` is omitted). In free play each
  image follows its label, for example `observe image 2 of 4: front camera.
  Marks: L = left arm TCP, R = right arm TCP, each drawn only where it falls
  inside this view; names from history_point_ids are drawn the same way.`
  `L` and `R` are the left and right arms' tool centre points, drawn on every
  view they fall inside, so a wrist camera often shows only its own arm's mark.
  A historical `observe` (with `step`) returns the saved images, unmarked, and
  its labels say so (`… front camera, the saved image of observation step 7; no
  marks drawn.`).

## Scripting the robot

The robot is a device the agent can program, in two ways. Both reach the
robot through the run's robot socket, so the Supervisor admits and executes
every call of a script inside the running instruction, one action at a time,
and returns the receipt and the new observation to the script. Each call is in
the agent trace with its origin.

- **Codemode (recommended).** The model writes a short JavaScript script that
  Pi runs in its QuickJS sandbox, with no file system, network or shell: it
  reaches the world only through `tools.<name>(...)`, for example
  `JSON.parse(await tools.robot_submit_action({...}))`. A robot call resolves
  to the Supervisor's answer (its `ok`, `receipt` and `observation`, or the
  refusal). The trace marks these calls `origin: codemode` with the codemode
  call that made them. Pi's `models` global (classifiers and image models on
  your credentials) is not offered.
- **`ora-robot` from the shell (full capability).** The agent's shell has
  `ora-robot` on its `PATH` and the Python client `ora.robot_client` on its
  `PYTHONPATH`, so a script in any language can look and act
  (`ora-robot call robot.submit_action --json '{...}'`). The trace marks these
  calls `origin: script` with the client and command line.

Once a script has acted in an instruction, a world-changing call the model
then makes itself in the same turn is refused (`world_changed_by_script`): its
turn began from an observation the world has left. The model can keep acting
through scripts, or end its turn and act on a fresh observation.

### Code mode

`ora play --code-mode` plays in ORA's code mode instead: the environment
baseline's `policy.code_mode` (RoboDojo's default baseline has one; the YAM
station names the same contract itself, `library: extended`, `prompt: v3`;
elsewhere a `--spec` names it), the plan's `play.code_mode: true`. The model's tools are
ORA's `read`, `write`, `edit` and `bash`, and nothing of Pi's: no Pi built-in
tool, no `codemode`, no `ora-robot`, no direct robot tool. A `bash` command runs
in ORA's script sandbox, and its Python programs reach the robot through the
`robot` library (`tools.move_pose(...)`, `dry_run=True` for a plan without
motion), as in a benchmark episode in code mode. The prompt is the code planner
with a free-play layer (`code-mode-play-v1.md`: the operator's instruction is
the objective, a reply in words ends it, no task check or closing step).

- Every robot call of a program goes through the Supervisor inside the
  command's call, one action at a time, and is in the agent trace with origin
  `code_mode` under that call; the command's call carries the Supervisor's
  execution ledger.
- A command whose programs executed an action ends the agent's turn, and the
  instruction goes on with the next turn on the new observation (as in an
  episode); such a command is one robot decision of the instruction.
- Esc ends a running command between its calls (`stopped`: its sandbox is
  killed, its later calls are refused) and then the instruction.
- On a real cell (YAM `--rig-mode read` or `on`), a stop at the cell ends the
  whole program, not only its current motion: once a motion's receipt says the
  operator or a guard stopped it (`guard_stop`: `/halt`, the page's stop, `ora
  cell stop-motion`, a guard) or the operator declined it
  (`operator_declined`), the program's later world-changing calls in that turn
  are refused (`script_stopped_by_operator`); the next turn reads the fresh
  observation and the operator's message. Pi's own `codemode` scripts and
  `ora-robot` follow the same rule there.
- The terminal and the page draw each command as written, its robot calls as
  they happen (a dry run marked as a plan, with its picture), how it ended and
  the end of what it printed; Ctrl+O (the page: open the call) shows all of it.
- Not on the YAM cell yet: a real cell's Esc stops only the current motion.

## What the agent receives, and what it does not

The scene's task language, its public completion contract and its success
predicate are withheld. The system prompt is Pi's own, followed by ORA's
layers for the environment, with the `episode-task-play-v2` layer in place of
the benchmark's episode-task layer. The agent's task is the play framing
("carry out the operator's instructions, one at a time; the operator may also
talk to you while you work"). `check_task` and the lifecycle tools
(`finish_episode`) are not advertised. Each instruction, and each message you
send while one runs, reaches the agent as your message, delivered before its
next model call. The layer tells the agent to answer a question or remark in a
sentence and keep working in the same reply, to take a changed request as the
new objective, and that a reply without a tool call ends the instruction.
Sessions planned before this layer bound `episode-task-play-v1`, in which a
newer instruction replaced the running one.

The agent process runs with your environment and privileges, minus every
credential variable: names containing `API_KEY`, `TOKEN`, `SECRET`,
`PASSWORD`, `CREDENTIAL` or `AUTH` are removed before Pi or its shell starts.
ORA's own provider key is passed to the agent process and removed from its
environment before anything else runs. Its working directory is the run's
workspace, and nothing gives it the simulator's internal state. Because it
runs as you, the shell can still read any file you can read, including the
credential store that `/login` writes. ORA does not add a sandbox of its own:
Pi's isolation options run the shell in a local micro-VM (Gondolin), which
needs hardware virtualization, or run all of Pi in Docker Sandboxes, which
needs Docker access; where they run, they can be added as Pi extensions.
Codemode is the sandboxed way to script.

## Budgets

Two layers bound a session. Each instruction may spend at most 24 robot
decisions, 80 model calls and 20 minutes (`play.instruction_budgets`). The
model calls and time are checked before every model call, and the robot
decisions before every action. When one runs out, that instruction ends and
the session waits. The whole session may spend 400 robot decisions (100 on
RoboDojo's default surface, [synchronous toolkit](../reference/toolkit-execution.md)),
1600 model calls and 4 hours (`budgets`); the agent sees the session's remaining
budget with each observation.

A model request carries the images the session has not pruned yet
(`play.context_images`). They all stay until a new instruction begins with 20
or more observations (or tool results with images) still carrying theirs; then
all but the latest three of that kind leave at once. Older images stay in the
conversation and its evidence but are sent as one line each naming their step
and camera, so a long session stays under the provider's request size limit.
Nothing sent earlier changes while an instruction runs, so its calls share one
prefix the provider can cache; a window that slid at every act made 62% of a
YAM session's calls recompute more than 8k tokens (2026-11-14). A RunSpec that
names `context_images` without `prune_at` keeps that sliding window.

On a self-hosted server with priority scheduling, `--request-priority
'{"correction": -1, "default": 0}'` gives the call that first reads a message
you added to the running instruction its own priority
(`play.request_priority`; the server decides whether lower or higher runs
first). Without it, requests carry no `priority` field.

On a real cell a motion takes seconds and the model waits for it.
`--pipeline-predict` (off by default; not with `--code-mode`) asks the model
for its next step while the motion runs, assuming it succeeds. The answer is
used only when the receipt that comes back confirms the motion (it reached its
target, or a closing gripper stopped on an object) and no message of yours
arrived meanwhile; otherwise the model is asked again with the fresh
observation. Every prediction, used or not, is in the agent trace
(`play.pipeline_predict`,
[configuration and evidence](../reference/configuration-and-evidence.md)).

## Evidence

Runs land under `outputs/ora-play/`. The plan carries `episode_mode: "play"`.

A session is one run. `/quit`, the session budget or running out of scenes
completes it, and it is evaluated and sealed like any other run. It is never
scored: `official_success` is `null` and `behavioral_sample_valid` is `false`.
Learning eligibility excludes it (`free_play_session`, and
`human_intervention` once an instruction was sent), so it reaches no score,
Memory or SFT export.

Robot cameras and the operator view never mix: the agent receives its
observation cameras only; the operator view is stored beside the evidence for
you, and the web page and the terminal show it.

The agent trace records:

- every model call, with the provider and model it used;
- every tool call and result, including the robot tools and Pi's coding tools, and every call a codemode script or an `ora-robot` script made, with its origin;
- every shell command and its output;
- every change of model or reasoning effort;
- every compaction and every move of the conversation.

The Supervisor records each instruction as `play_instruction_started` and
`play_instruction_finished` events (the started record's `observation` says
whether the instruction's fresh observation was taken: `{"fresh": true,
"observation_step": N}`, or `fresh: false` with the last step and the error;
the observation's policy-state snapshot has cause `instruction_start`), operator controls as `play_control` and
scene changes as `play_scene_reset`. When the environment has an operator
view (a camera for you, never for the agent: a RoboTwin scene set's
`operator_camera` or RoboTwin's observer camera, a RoboDojo scene's
`operator_camera`, Synthetic's drawn front view), each observation's view is stored under
`native/operator-view/` with an `operator_view_frame` event; the agent never
receives it. The evaluation checks the play trace for
integrity and completeness: the hash chain, artifacts, a response for every
request, a result for every tool call, and a receipt for every action. It does
not apply the benchmark's context and request checks. [Configuration and
evidence](../reference/configuration-and-evidence.md#free-play) lists the
fields.

The GPU environments run on a registered execution host with their simulator
and perception services, as `ora run` and `ora interactive --spec` do.
