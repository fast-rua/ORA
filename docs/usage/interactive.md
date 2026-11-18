# Interactive sessions

`ora interactive` runs ORA as the policy of a robot episode and shows the
agent at work in its own terminal UI: the Pi agent core's interactive mode,
with ORA's observations, tool calls and results, and the camera images the
model receives. You can steer it, check its budget, stop it, leave it running
and come back.

To give a robot instructions one at a time in a scene with no benchmark task,
use [free play](play.md) (`ora play`).

```bash
./scripts/ora interactive                                   # choose a provider profile, then run
./scripts/ora interactive --provider deepseek_flash_openai_high_v1 --actions 4
./scripts/ora interactive --provider env                    # ORA_BASE_URL, ORA_MODEL, ORA_API_KEY
./scripts/ora interactive --spec path/to/run-spec.yaml --confirm-launch --ui pi
```

Without `--spec` the run is a Synthetic episode (no simulator): the
quickstart's task, longer and steerable (`--actions` robot decisions, default 3;
`--task` replaces the instruction). With `--spec` and `--ui pi`, a prepared
Native benchmark RunSpec (several `--spec`: choose one) is composed exactly as
the task picker composes it, an `ora-interactive` plan on a fresh admission
cell, and runs the same way. Either way the run goes through the
normal path: the plan is resolved, preflight prints one line per check, the
resource lease is taken and a detached Supervisor starts the episode. The agent
core runs headless under the Supervisor, as in any run, and serves its
terminal UI; this terminal attaches to it. `--no-attach` launches and prints
the `ora attach` command instead.

The UI is the Pi agent core's own, so it needs only the core's Node.js and
packages ([installation](../getting-started/installation.md#nodejs-and-the-agent-core)).
A plan resolved for the retired Python Native core does not start; resolve its
RunSpec again, or follow its sealed run with `ora watch`.

## World view

While a terminal is attached, `ora interactive`, `ora play` and `ora attach`
also serve the run's cameras on a read-only browser page on this host's
loopback address: the original frames the simulator records (the Synthetic
runs draw a top view of their end effectors). `world view:
http://127.0.0.1:PORT/live` is printed before the TUI starts, and the TUI
header shows the address and what the browser shows (`browser shows frame 12`
once a browser has decoded one). On a desktop the browser opens by itself; in
an SSH session the header says to forward the port. `--world-port N` fixes
the port, `--world-browser off` keeps the browser closed and `--world-view
off` turns the page off. From your own computer, `ora-remote` does all of it
over one SSH connection: see [Using ORA on a remote GPU server](remote.md).
For the conversation itself in a browser, beside the cameras, see
[Web view](remote.md#web-view) (`ora web`).

## Model and credentials

`--provider` names a provider profile from `ora providers` (your own
included); without it, a terminal lists the live-admitted profiles with where
each one's key comes from, *+ add an endpoint*, and `env`, the quickstart's
`ORA_PROVIDER`/`ORA_BASE_URL`/`ORA_MODEL`/`ORA_API_KEY`. Choosing a profile
that *needs* its key asks for the key (not shown), keeps it in your user-level
ORA store and runs the provider probe, whose result shows before the run
starts ([keys](../getting-started/model-apis.md#keys-configured-once-in-the-terminal)).
The key stays with ORA: preflight qualifies a profile's key, the Supervisor
hands it to the agent process through its environment, and no key reaches the
plan or its evidence. The model is bound by the RunPlan, so inside a run
`/model` only shows it, and `/thinking` and `/scoped-models` are refused;
`/login` and `/logout` manage the same user-level store, for later runs and
free play, never the key the run is bound to.

## Keys and commands

| Key or command | Effect |
|---|---|
| Enter | Steer: the Supervisor admits the message and delivers it before the next model call. The run is then human-assisted. |
| Esc | Interrupt the model call in flight and stop the run (operator stop). |
| `/stop` | Stop the run after a confirmation. A model call in flight is recorded first, so the trace stays complete; also works while the robot acts, when Esc has nothing to interrupt. |
| `/status` | Step, phase, model calls, robot decisions, tokens, remaining budget (wall time counts down), operator messages, the last action with its refusal or environment receipt. |
| `/export [PATH]` | Write the session as Pi's HTML transcript, or the raw session with a `.jsonl` path, in the directory where `ora attach` runs (default `RUN_ID.html`). |
| `/model` | The model, protocol and provider profile the run is bound to. |
| `/detach`, Ctrl+D, Ctrl+Z, `/quit` | Leave the terminal; the run continues. |
| Ctrl+O | Show the raw observation and tool JSON instead of the summaries. |
| Ctrl+\\ | Leave at once from the attach client, whatever the UI is doing. |
| `/help` | These keys, and the Pi commands that are not available. |

The transcript reads as a summary: each observation shows its cameras and
images, the last action's effect (for example `after move_to: right arm 179 mm
from target, planner failed, not moved`), the arms and the remaining budget;
each tool call shows the tool and its key arguments, and its result shows
`✓ accepted`, `✗ refused` with the reason, or its salient fields. Ctrl+O shows
the raw JSON the model received. The summaries are drawn from the run's own
records; they change nothing the model sees.

The footer shows the step, model calls, robot decisions, phase (model working,
agent turn, robot acting) and the remaining budget. Pi's own commands that
would rewrite a live episode (`/new`, `/resume`, `/fork`, `/clone`, `/tree`,
`/import`), compact its context (`/compact`), reload extensions, upload the
session (`/share`, `/bug`) or run shell lines (`!`) are refused with the
reason; an unknown `/command` is refused rather than sent to the model.

Esc and `/stop` end the episode with the operator as termination authority:
the run is `incomplete`, then evaluated and sealed. Esc cuts a model call in
flight short (the request is recorded as interrupted); `/stop` lets it finish
and runs no further call or tool.

## Detach, re-attach, recover

Ctrl+D (or `/detach`) returns the shell and prints the `ora attach RUN_ID`
command; the episode goes on without a terminal. Attach again with:

```bash
./scripts/ora attach RUN_ID     # or a run directory
./scripts/ora attach            # choose among this host's running runs
./scripts/ora attach --last
```

Re-attaching redraws the whole transcript, including what happened while no
one watched. One terminal is attached at a time; a new one takes over. The
terminal endpoint is a unix socket on the host that runs the Supervisor, so
attach on that host. On a cluster the Supervisor lives in the job step that
launched it: detaching the terminal is fine, ending that step (closing the
tmux pane that runs `srun`, for example) ends the run.

If the agent process dies, the Supervisor restarts it and resumes the turn
(each tool still runs once); the attached terminal reconnects by itself and the
transcript marks the resumed turn. A run whose Supervisor is gone cannot be
resumed: its world state is lost. `ora retry` starts a new attempt when the
run published no robot decision.

When the episode ends, the terminal shows the termination and waits for ORA to
evaluate and seal the run, then prints its state, decisions, model calls,
official verdict, evaluation and seal. `ora attach` on a run in that window
says that the run is being evaluated and sealed, waits for it, and then opens
the landed run read-only; on a run that has landed it opens it at once.

## Sessions and export

```bash
./scripts/ora sessions                         # runs with a Pi session: state, attachable, counts
./scripts/ora sessions --json
./scripts/ora sessions export RUN_ID --output run.html    # or run.jsonl
```

The Pi session (`native/pi-session/*.jsonl`) is the run's conversation record;
the HTML export is Pi's transcript viewer. Exports can contain the task,
prompts and tool output: review them before sharing.

## Images

Observation and tool images are drawn inline on terminals with an image
protocol (Kitty, iTerm2, WezTerm, Ghostty and others pi-tui supports). Elsewhere,
including tmux by default, each image is a line with its artifact path, type
and size (`native/agent-artifacts/<sha256>.png [image/png] 640x480`).
`PI_IMAGE_PROTOCOL=kitty|iterm2` forces a protocol for the attaching terminal.
The images are the bytes the model received, read from the run's own
evidence.

## What a message changes

A run that received a message is human-assisted. Its official verdict is
kept, but it is marked `human_intervention: true` with an
`operator_message_count`, excluded from autonomous success rates, and not used
for automatic learning or SFT export. A stopped run ends `incomplete` with
the operator as termination authority. Use `ora watch RUN_ID` to inspect any
run without being able to send input.

## The task picker (`--spec`)

`ora interactive --spec PATH [--spec PATH ...] --confirm-launch` (also opened
by `ora local` and `ora run --env`) offers prepared benchmark tasks in a picker
and follows the selected run as ORA's transcript. Each spec must be a Native
RunSpec for LIBERO, RoboTwin or RoboDojo; the task keeps its benchmark
instruction, official evaluator, seed or layout, provider, resources and
policy. Selecting a task and passing `--confirm-launch` are both required
before the normal preflight and launch.

| Key | Effect |
|---|---|
| ↑/↓, Enter in the picker | Select and start a task |
| Enter in the editor | Steer the active run: the message reaches the model before its next call |
| Alt+Enter | Queue a follow-up for the next task you start |
| Tab | Switch between the picker and the editor when no run is active |
| Esc | Stop the active run through the normal stop path |
| Ctrl-C / Ctrl-D | Leave the interface; the run keeps going |

After the environment ends the episode, ORA writes metrics, evaluates, seals
and releases the run's resources. The status line shows `Landing…` until
that is done. Keep the terminal (and, on a cluster, the job step that runs
ORA) open until the run is reported finished: closing it earlier can stop
the landing midway.
