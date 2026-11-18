# Using ORA on a remote GPU server

The simulator usually runs on a GPU server without a display while you work
on your own computer. `ora-remote` runs one ORA command on the server over
SSH and gives you both halves of the session locally: the Pi terminal UI in
your terminal, and the robot's cameras in your browser. One SSH connection
carries both, with a single forwarded loopback port.

```bash
./scripts/ora-remote gpu-host --repo /srv/ORA play --env synthetic --provider env
./scripts/ora-remote gpu-host --repo /srv/ORA play --env libero --spec prepared.json --confirm-launch
./scripts/ora-remote gpu-host --repo /srv/ORA interactive --ui pi --provider PROFILE
./scripts/ora-remote gpu-host --repo /srv/ORA attach RUN_ID
./scripts/ora-remote gpu-host --repo /srv/ORA watch RUN_ID
```

Everything after the command word is passed to `ora` on the server unchanged;
see `ora play --help` and [interactive sessions](interactive.md) for those
commands. ORA's own launch gates still apply there: a prepared
benchmark RunSpec launches only with `--confirm-launch`, after its normal
preflight.

## What you need

- **On your computer:** OpenSSH, a browser and an ORA checkout (or install).
  `ora-remote` uses only the Python standard library, so the simulator stacks
  are not needed locally. `gpu-host` is anything `ssh` accepts: a host name,
  `user@host` or an alias in `~/.ssh/config` (with its keys, `ProxyJump` and
  so on).
- **On the server:** an ORA checkout set up for the environment you run
  (`ora setup --env ...`), its Node.js agent core, and the provider
  credentials ORA reads there. `--repo` names that checkout (or set
  `ORA_REMOTE_REPO` once); without either, `ora-remote` runs the `ora`
  command that the server's login shell finds on `PATH`.
- **SSH port forwarding** permitted for your account on the server (sshd's
  `AllowTcpForwarding`, on by default). Some clusters turn it off on their
  compute nodes. `ora-remote` then says, as soon as it connects, that the
  host refused the forward; the terminal UI still works, without the
  cameras. It does not work around the refusal: run ORA on a host that
  permits forwarding, or ask its administrator.

## What happens

1. `ora-remote` picks a free local port and runs
   `ssh -tt -L 127.0.0.1:PORT:127.0.0.1:PORT gpu-host '<the ora command>'`,
   adding `--world-port PORT --world-browser off --world-public-url
   http://127.0.0.1:PORT/live` to the command.
2. On the server the command launches the run (or attaches to it) and, while
   your terminal is attached, serves the world view on the server's loopback
   port. Pi's terminal UI draws in your terminal; its header shows
   `world view http://127.0.0.1:PORT/live` and what your browser shows
   (`browser shows frame 12`, or why nothing is shown yet). `/status`
   repeats it.
3. Once the forwarded page answers, your browser opens it (`--no-browser`
   prints the address instead). `ora-remote` keeps waiting for the page as
   long as the session lasts, since a real launch can preflight for minutes,
   and keeps ssh's messages about the not-yet-listening port off your
   terminal. The page shows the run's original camera
   frames, as the simulator records them, and tells the terminal which frame
   it decoded.

The world view needs the run's execution capture: RoboTwin, LIBERO and
RoboDojo record it by default, and the Synthetic runs of `ora play --env
synthetic` and `ora interactive` draw a top view of their end effectors.

## Detach, disconnect, come back

Ctrl+D (or `/detach`) closes your terminal and leaves the run going on the
server. A dropped connection (a laptop that sleeps, a network change) does
the same: the run continues under its detached Supervisor, and `ora-remote`
says so. To come back, **run the same command again**: while that run is
going, the server attaches to it instead of launching another (`--new`
launches another anyway). Once the run has finished, the same command starts
a new one. `ora-remote gpu-host attach RUN_ID` attaches to any run on that
server.

The server remembers which run a command left in
`.harness-runtime/ora-remote/<session>.json` of its workspace, outside run
evidence. The session name is derived from the host, `--repo` and the command
line, so it is the same each time you run the same command; the record names
the run and the server it runs on, nothing else.

On a cluster, the Supervisor lives in whatever started it. A command that
`ora-remote` runs over SSH on a compute node in your allocation keeps its run
when the connection drops, as long as the allocation lasts; a run started
inside an interactive job step (`srun --pty ...`) ends with that step.

## Images in the terminal

`ora-remote` recognises your terminal as pi-tui does (Kitty, Ghostty,
WezTerm, Warp, iTerm2 and others) and passes the result to the server as
`PI_IMAGE_PROTOCOL`, `PI_TRUE_COLOR` and `PI_HYPERLINKS`, since SSH does not
forward the variables pi-tui recognises a terminal by. The image escape
sequences then pass through SSH unchanged, so observation and tool images
draw inline as they do locally. Under tmux or screen on your computer they
stay text lines, as locally. Set `PI_IMAGE_PROTOCOL=kitty|iterm2|none` before
`ora-remote` to override the detection. ORA's own terminal words follow your
terminal's language: a Chinese locale here (or `ORA_UI_LANG=zh`) passes
`ORA_UI_LANG=zh` to the server.

## Web view

`ora web` shows a live run in the browser, on ORA's own page: the
conversation on the left, the robot's cameras on the right, from the moment
the page opens.

- **Conversation.** Drawn *Compact* by default, as a coding agent's
  transcript: each thought is one line (*Thought for 12 s*, the model call's
  time), each tool call one line with its name and a short line of its
  arguments, each observation one line with its image count; a click opens
  one (the whole thinking, or the call's arguments, result and image, its
  step selected in the camera panel). *Detailed*, switched above the
  conversation, draws each observation with the images the model received
  and each tool call as a card with its step and that step's image (the image
  the tool returned, or the step's view through the tool's camera). *Expand
  all thinking* opens every thought in either; the browser remembers both
  choices. The agent's replies stream in as it writes them. In free play each instruction is
  numbered; a message you send while one runs is shown joined to it, *queued*
  until the Supervisor delivers it and then *delivered at step N*, the agent's next words
  marked as its *Answer*; where an instruction ended is marked (*Instruction 2
  stopped (Esc) · at step 9*). A reply the model's output limit cut off is
  marked *Cut off: the reply reached the model's output limit (N tokens)*, and
  ORA's request to continue follows it (*ORA · continue*). When your message
  arrives just as the agent proposes a robot action, Pi starts a reply to it
  that ORA stops so the action runs first; that call is shown as *Superseded
  by your message*, not as an error, and the agent reads your message at its
  next step. The list follows the newest item until you scroll up yourself
  (wheel, touch, keys or its scrollbar); content that grows by a lot at once
  never leaves the end by itself, and scrolling back to the end follows again.
- **What the agent does now.** The header and the box you type in say it,
  with the seconds it has taken: *Thinking… 4 s*, *Writing… 6 s*, *Running
  vision_ground… 1 s*, *Robot moving… 2 s*. The page never sits still without
  a status.
- **Cameras.** *Live* shows the world now: the robot's recorded motion as it
  happens (the run's execution capture, played at its frame rate a moment
  behind, resting on the newest motion's last frame between actions), or,
  without recorded motion, the run's world view (as `/live` shows it). When
  the environment gives an operator view (a camera for you, never for the
  agent: a RoboTwin or RoboDojo scene set's operator camera, RoboTwin's
  observer camera, Synthetic's drawn front view), it is the
  main picture and the robot's cameras are a small picture to switch to; the
  camera chips switch too. Clicking a tool card or an observation moves the
  panel to that step: the motion its action ran, played once with a scrubber
  (*Motion*), or what the model saw then (*What the model saw*), through the
  tool's camera, with a `vision_ground` or `vision_segment` call's candidate
  boxes and a `mark_point` call's points drawn over it, exactly where the
  call's pixel coordinates fall on the image. The images the call returned are
  in a strip over the picture. The picture sits in one box, as large as the
  panel allows, in the shape of the run's operator camera: 16:9 for a 1280×720
  one (RoboDojo's and RoboTwin's free play), 4:3 for the older 640×480 one or
  when the run has no operator view. A live page takes the shape from the
  run's first operator view and keeps it; a replay from the run's evidence when
  it opens. Every image is fitted into the box whole, on black only where an
  image of another shape is letterboxed, whatever its source and size (a head
  or wrist camera, the operator view, a motion frame, a returned image); the
  bars around it keep their heights, so the picture never jumps when its
  source changes. With a 16:9 box the cameras take three fifths of the page's
  width and the conversation two fifths (about 750 pixels at 1920 wide, its
  text at its usual size). A motion's play bar lies over the picture's foot.
- **Speed (free play).** One line at the foot of the camera panel: the
  model's tokens per second, its time to the first token and the loop between
  robot actions against a target (*45 tok/s · TTFT 0.8 s · loop 6.2 s (target
  3.0 s)*, the medians of the last 10). A click opens it over the cameras
  (they do not move): each model call's tok/s (output tokens over the time
  from its first token to its end), TTFT (request to first token), call time
  and output tokens, the last one and the last 10, and the calls per action;
  then each loop as a bar on one scale with the target marked, the latest and
  the median of the last 10 per segment: *model thinking* (every model call
  of the act), *tools & overhead* (the rest until the action starts),
  *waiting for approval* (on a real cell, each motion question from when it
  was posted to its answer, and who answered: a person, *safe mode (…)* or
  *auto-approve (…)*), *planning + motion* (the action's time without that
  wait) and *next observation*. Under them it says how far the median loop
  is from the target, which segment dominates and what it would need, for
  example *Model thinking is 77% of the loop; TPS must go from 45 to 90 tok/s
  to meet 3.0 s*, assuming the other segments stay as they are and
  generation time scales with 1/TPS. The target is editable there (3 s by
  default; this browser remembers it). The times are the agent process's:
  it times each call (request sent, first content, end) and stamps the
  Supervisor's action started and finished announcements when they arrive.
  An approval wait is known only for a question the page saw waiting (the
  gateway reads the cell's questions while a page is open); otherwise it is
  counted in planning + motion, and the panel says so. A replay shows the
  panel from the evidence: the agent trace's model calls (no first-token
  time, so tok/s counts the whole call) and the Supervisor's actions, without
  approval waits.
- **Timeline.** One tick per observation step under the cameras (marked where
  a call drew something or failed, or its action's motion was recorded), each
  with its thinking and moving time; previous, next, and a replay that walks
  the steps with the conversation scrolling along.
- **Language.** English or Chinese: the page starts in your browser's
  language, and the header's *EN | 中文* switch changes it (remembered by
  this browser).
- **Messages and controls.** Enter sends: the first message of an idle agent
  as Pi's prompt, a message while it works as a steer (Alt+Enter: a
  follow-up). In an episode a message is an operator message, delivered
  before the next model call; *Stop run* asks for confirmation in the page.
  In free play a message is the next instruction (the first one too: the page
  works before any instruction exists). While one runs, the switch at the
  right of the status line says what Enter does with a message:
  - *Steer* (the default for each instruction): it joins the running
    instruction; the agent reads it before its next step, answers a question
    and keeps working, or takes a changed request as the new goal. A motion
    in progress finishes first: the agent reads the message with the
    observation after it.
  - *Queue* (also Alt+Enter, as in the terminal): it goes to the run at
    once as a `queued_instruction`; the Supervisor holds it until the running
    instruction ends, and it opens the next one. It is listed above the box
    until then (the run holds it, so it cannot be taken back).
  - *Stop & send*: on a real cell, the motion in progress ends at once where
    the arms are, held (`/halt`); while the agent is still planning, the same
    stop refuses the motion its reply proposes before anything moves (and is
    withdrawn when the reply proposes none, once your message reaches the
    agent). The message is a steer the agent reads with the fresh
    observation; the instruction goes on with your change. Without a real
    cell nothing can cut a motion short, so the message is queued and the
    instruction is stopped (Esc; a motion in progress finishes): the message
    opens the next instruction.

  With no instruction running the switch is gone and Enter sends. *Stop
  instruction* (Esc), *Next scene* and *End session* are the terminal's Esc,
  `/reset` and `/quit`; on a real cell, Esc and *Stop instruction* end the
  motion in progress first, or refuse the one the agent still plans, as the
  terminal's Esc does. The
  header says where free play is, as the Supervisor announces it: waiting for
  an instruction, an instruction running, or ended, with how the last
  instruction ended (reported, stopped, or out of budget).
- **Commands.** Typing `/` at the start of the box lists the commands the page
  takes, each with one line of what it does, as Pi's terminal lists its own:
  `/reset`, `/quit` and `/stop`, and on a real cell its commands (`/halt`,
  `/approve [value]`, `/decline [reason]`, `/mode [ask|safe|auto]`,
  `/yolo [on|off]`, `/home` ...). The list narrows as you type; ↑/↓ choose,
  Tab or a click completes, Enter completes a partial name and runs a command
  typed in full, and Esc closes the list (it does not stop the instruction).
  After `/mode ` and `/yolo ` it lists their words.
- **A real robot cell.** A free-play run on a real cell (rig mode `read` or
  `on`) adds what its terminal has: each question the robot waits on as a
  card above the box you type in (approve, or decline with a reason, after a
  confirmation), the auto-approve switch, *Stop here* (`/halt`: the motion
  in progress ends where the arms are), *Home*, *Hold* and *Diagnose* at
  once, *Float*, *Release* and *E-stop* after a confirmation only its *Yes*
  button gives, the outcome of each command given here or in the terminal,
  and the held arms' alerts as banners. The terminal's
  `/halt`, `/approve`, `/decline`, `/mode`, `/yolo`, `/home` ... typed into the box do the same
  ([free play on a real cell](play.md#on-the-web-page), [from your
  laptop](real-cells.md#the-web-page-from-your-laptop)).

The page offers what the run's mode allows and nothing else. An episode binds
the model, reasoning effort and context policy in its RunPlan and its session
is evidence, so the page has no model, reasoning, compaction, rename, new
session or branch controls (and the agent process refuses those commands
from any client). Free play opens them under the header's session menu, as
its terminal does. Shell commands are not available in the browser in either.

Install and build the page once on the server:

```bash
./scripts/ora web setup     # npm ci in ora-pi/web, then the page's build
```

The build records which sources it was built from; after an update that
changes the page, `ora web` asks for `ora web setup` again. A plain `npm ci`
in `ora-pi/web` changes nothing the page needs.

From your computer, one command opens the terminal UI and the web view
together, or the web view alone:

```bash
./scripts/ora-remote gpu-host --repo /srv/ORA --web play --env synthetic --provider env
./scripts/ora-remote gpu-host --repo /srv/ORA web RUN_ID
```

On the server itself, `./scripts/ora web [RUN_ID]` prints a link with the
launch token (and opens it outside an SSH session). Without a run ID it picks
the live run on the host (`--last` when several are).

Some things to know:

- A dialog appears where its command was given: *Stop run* in the page asks in
  the page, `/stop` typed in the terminal asks there.
- Older steps' images load on demand from the run's stored images; the latest
  steps come with the conversation. Motion frames and operator views are the
  run's own files, served as they were written.
- `ora web` keeps its state in a private temporary directory and writes
  nothing into the run; a run that is not live is refused, so no Pi other
  than the run's own ever opens its session file.
- The page is Apache-2.0 code seeded from the MIT-licensed
  [Watercol/pi-web-ui](https://github.com/Watercol/pi-web-ui)
  (`ora-pi/web/LICENSE`); its file, directory and working-directory features
  are gone.

### A run's scene for the 3D replay

The run dashboard (`ora-viewer dashboard`, described in `viewer/README.md`)
draws a RoboTwin, LIBERO-Pro, RoboDojo or RoboLab run's simulated objects from its scene trace, and a LIBERO-Pro or RoboLab run's robot.
The dashboard never starts a simulator, so the trace is exported first, on a machine that can run the run's simulator:

```bash
./scripts/ora replay RUN_ID --scene                       # scene-trace/RUN_ID/ beside the run's runs/ directory
./scripts/ora replay RUN_ID --scene --scene-traces DIR    # DIR/RUN_ID/; give the dashboard the same --scene-traces DIR
./scripts/ora replay RUN_ID --scene --resource-profile P  # RoboDojo: on a qualified slot, as its --check
```

`--scene` re-executes the run's replay program as `--check` does, with the same
needs (`ROBOTWIN_PYTHON` and `ROBOTWIN_ROOT`, or `LIBERO_PRO_PYTHON`,
`LIBERO_PRO_ROOT` and `LIBERO_PRO_DATA_ROOT`, with the caller's
`CUDA_VISIBLE_DEVICES`; RoboDojo: its interpreter and roots, a clean checkout,
the run's host marker and a card that runs Isaac), and keeps the scene of that re-execution only when
every action's end pose matches its receipt bit for bit, the verdict matches
and every action spans the same simulator steps as the execution capture. A run
that diverges gets no trace: the command exits 1 and prints the report's path,
in a work directory beside the traces. A verdict that alone reads differently is
no divergence: the trace is kept and the result says `verdict_differs`. Nothing is
written into the run. An
existing trace is not overwritten; remove it to export again. A trace of a
short run is 8 to 20 MB, nearly all of it meshes, and holds private simulator
state: it is for a developer and reaches no policy.

### Replay a finished run

`ora replay` shows a finished run on the same page, rebuilt read-only from the
run's evidence and played back on the run's own recorded time:

```bash
./scripts/ora replay RUN_ID                 # prints a link with a launch token; 16x, the robot's motion in real time
./scripts/ora replay RUN_ID --speed 4 --view detailed --lang zh
./scripts/ora replay RUN_ID --motion-speed same   # one speed throughout
```

The replay has two rates. The waiting and thinking between the robot's
actions play at `--speed` (16× by default; 1× to 64× on the page), never
faster through quiet stretches. Each action, from its submission to its
receipt with the settling window the environment recorded after it (frames
with the action's id; the window's physics steps in
`native/post-action-settle.jsonl`, shown as *the scene settling after the
action*), plays its execution capture by the frames' simulated time at
`--motion-speed` (1× by default: real simulated time, so a cup swept off the
table is seen flying; 0.25× to 4× on the page), and never shorter than the
action would play at `--speed`. Only captured frames are shown: on screen is
always the last frame taken by that simulated time, held until the next is
due; nothing is interpolated or made up, and a frame repeating an earlier
moment (an action's end and the next action's start share one) adds no time.
A capture taken sparsely (a frame every 0.1 s of simulated time) moves in
steps; RoboDojo's free play captures the operator camera at 1280×720 every
0.036 s of simulated time. `--motion-speed same` (*motion: same speed*
on the page) plays everything at `--speed`. The conversation follows the
recorded clock in both, at the motion's pace during a motion; cut the result
later if at all. The bar below the conversation says what the agent was doing
at that moment (*Thinking… 4 s*, *Robot moving… 1 s*, counted on the
recorded time), with play and pause, a scrubber over the recorded time marked
where each instruction began, the speed and the motion's speed; the header's
corner says what plays and how fast (*robot motion · real time*, *thinking ·
16×*, *waiting · 16×*, or *recorded · 16× speed* at one speed). *Now* shows
the world at the clock: each camera's newest captured frame (cameras taken at
different rates keep their own), the operator's camera through its sharpest
capture camera (a higher-resolution operator camera when the capture has one,
else the usual one), and the operator view when it was taken later and is as
sharp. A step's tool calls and motion open as on the live page. The page has no message box, controls,
session menu, notices or dialogs: nothing reaches the run, no Pi RPC is
attached, nothing in the run is written, and a live run is refused (`ora web`
shows it). `--publish` leaves the run's ID out of the page and shortens
absolute paths in what it shows (`…/name`).

What it is rebuilt from, and what it cannot show:

| Shown | From |
|---|---|
| Instructions, observations with their images, replies with their thinking, tool calls with their arguments, results and images, joined messages and their answers | Pi's session (`native/pi-session/`), each entry at the moment it was appended; images by the run's stored copies (`native/agent-artifacts/`) |
| When a model call began (*Thinking…*) and a tool ran | The reply's own start time and the result's duration in Pi's session |
| Instructions started and ended, a message typed while one ran (queued, then delivered at its step), the robot's actions and receipts, controls, the session's end | The Supervisor's `events.jsonl`; an action's step from `agent-events.jsonl` |
| The robot moving, and the cameras between actions | The execution capture (`native/execution-capture/`): every frame of every camera with its capture time, simulated time and size |
| The scene settling after an action | `native/post-action-settle.jsonl`: each action's settling window, by physics step |
| The operator view | `native/operator-view/` at each observation |

Not in the evidence, and so not shown: a reply's tokens as they streamed (it
appears whole when the model finished it), the page's own notices and
dialogs, and a message a page sent that the run refused. An image the run did
not store is left out and counted in the replay's notes. An episode's
observation images are in ORA's own history rather than Pi's session, so an
episode's observations show no images.

`--export OUT.mp4` draws the replay into a video and exits:

```bash
./scripts/ora replay RUN_ID --export out.mp4 --speed 16 --size 1920x1080 --publish
```

A headless Chromium opens the page and its playback is set to each frame's
moment in turn (a virtual clock), so the video keeps the replay's rates
whatever the machine's speed and is the same every time: frame *i* of a 30
fps video shows the playback moment *i*/30 s, which in a waiting stretch at
16× is 0.533 s of recorded time per frame and in a motion at 1× is 1/30 s of
simulated time per frame. Each frame is drawn at
`--scale` device pixels per CSS pixel (2 by default), scaled down to
`--size` with Lanczos and encoded as H.264 (yuv420p, CRF 16); a frame the page
draws exactly as the one before reuses its screenshot, and `--workers` pages
(4 by default) draw stretches of frames side by side. The page's layout is
`--size` CSS pixels (`--zoom 1.25` draws the page larger: its layout is the
size divided by the zoom). The video's page shows no controls, no language
switch and no conversation switch; its corner says what plays and how fast.
`--motion-speed`, `--view`, `--expand-thinking`, `--lang`, `--from` and `--to`
(seconds into the recorded time) choose what it shows. A report beside the
video (`OUT.mp4.json`, `ora.replay-video.v1`) records its size, frame rate,
frame count, duration, the camera box's shape (`camera_aspect`), its speed
and motion speed, the seconds of robot motion
in it, and every stretch (`segments`: waiting or motion, its rate, its start
and end in the video, in recorded time and, for a motion, in simulated time,
with the action and its step), and the replay's notes; it names the run
unless `--publish`.

The export needs two tools ORA does not install: Playwright for Python with
its Chromium (`pip install playwright==1.56.0` and `playwright install
chromium`; or `uv` on PATH, which runs it as `uv run --no-project --with
playwright==1.56.0`; or `--export-python` naming a Python that has it,
`--chromium` another Chromium), and ffmpeg with libx264 (on PATH, `ORA_FFMPEG`
or `--ffmpeg`; a static build will do). ORA's `uv.lock` does not include
either.

Any client of Pi's RPC protocol can attach the same way: give it
`ora-pi/bin/ora-rpc-attach` as its `pi` executable (or `ora-pi/src/rpc/attach.ts`
as Pi's RpcClient `cliPath`) with `--session <run>/native/pi-session/<file>`
or `--ora-run RUN_DIR`. The first client controls the run and later ones
observe unless they attach as controllers (`--ora-role`), as a new
`ora attach` takes over a terminal. ORA's own commands (`ora_get_run`, `ora_get_frames`,
`ora_play_control`, `ora_play_queue`) and events (`ora_play_state`, `ora_play_message`, `ora_model_call`) are described in the
[public interfaces](../reference/public-interfaces.md).

## Security

- The world view listens on the server's loopback address only, and refuses
  requests whose `Host` is not a loopback name (DNS rebinding). `ora-remote`
  binds the local end of the tunnel on `127.0.0.1` only. Nothing else is
  forwarded.
- The page is read-only: it shows recorded camera frames and the run's
  phase, and its only request back is the acknowledgement of a decoded frame.
  There is no web terminal, and no input reaches the run from the browser.
- Credentials stay on the server with ORA's provider profile; `ora-remote`
  passes none. Other users of a shared server can connect to a loopback port
  there, so treat the camera frames of the terminal flow's world view as
  visible to them.
- The web view is closed to them: its gateway listens on a 0600 unix socket
  in your runtime directory, `ssh -L` forwards that socket, and the browser
  needs a cookie it gets once for a launch token (a wrong try spends it).
  `ora-remote` generates the token on your computer and hands it to the
  server in a 0600 file on a first SSH command that shares one connection
  with the session (one sign-in); the token never appears on a command
  line or in a log. Requests must name a loopback host, state changes must
  come from the page's own origin, and only the page's own routes pass the
  gateway. On a real cell the gateway's own `/ora/cell` routes run the play
  terminal's `ora cell` command lines and nothing else (a fixed set of verbs,
  an answer by a question's id, the auto-approve switch); one that lets the
  arms move or fall is refused unless the request says the operator
  confirmed it. There is still no web terminal: the page has no shell, file,
  directory or working-directory route, it shows the model's text as Markdown
  without HTML, and it loads nothing from elsewhere (a strict
  Content-Security-Policy).

## Without `ora-remote`

If you SSH in yourself, `ora interactive --ui pi`, `ora play` and `ora attach`
print `world view: http://127.0.0.1:PORT/live` before attaching, and the TUI
header says to forward that port. Choose the port and forward it yourself:

```bash
ssh -L 127.0.0.1:8765:127.0.0.1:8765 gpu-host
./scripts/ora attach RUN_ID --world-port 8765     # then open http://127.0.0.1:8765/live locally
```

`--world-view off` turns the world view off.
