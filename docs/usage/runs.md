# Run and inspect

Start with the [Synthetic first episode](../quickstart.md), or compose a robot
RunSpec using an [environment guide](../environments/README.md). Operator
commands share the plan, preflight, Supervisor, evaluation, and seal boundary.

## Inspect before launching

From the repository root:

```bash
./scripts/ora plan validate path/to/run-spec.yaml
./scripts/ora plan show path/to/run-spec.yaml
./scripts/ora plan context path/to/run-spec.yaml
```

These inspect the specification, resolved plan, and policy context. The
[command guide](commands.md) describes preflight and launch,
including live provider checks and qualification. Synthetic's bounded quickstart
performs its normal launch sequence.

## Follow one run

Replace `RUN_ID` with the exact ID printed at launch:

```bash
./scripts/ora status RUN_ID
./scripts/ora watch RUN_ID
./scripts/ora metrics RUN_ID
```

A run ID resolves under this workspace's outputs root: the checkout's
`outputs/` (shared by its linked worktrees), or `ORA_OUTPUTS_ROOT` or
`ORA_WORKSPACE_ROOT` when set, plus every root listed in
`ORA_EXTRA_OUTPUT_ROOTS` (`ora workspace` prints them). A run anywhere else, a
copied run or another workspace's, is not found by its ID: pass its directory
instead, `./scripts/ora watch path/to/<experiment>/runs/<run-id>`, or add its
outputs root to `ORA_EXTRA_OUTPUT_ROOTS`. `ora replay` and `ora web` take a
directory the same way.

`ora watch` shows the run as a terminal transcript: what the model said,
which tools it called with condensed results, refusals, action receipts and
each round's official verdict. It follows a live run and replays a finished
one on recorded time (`--speed 0` prints it at once). The pi-tui front end
ships with the agent core (`ora-pi/src/watch/`; Node.js 22.19+ and
`npm ci --prefix ora-pi`); without those packages `ora watch` uses its
built-in terminal view. Both views head the transcript with
ORA's mark, a pixel robot arm, in teal (`#4ecdc4`), the accent that also marks
the spinner and the `▸▸` footer. `ora watch RUN_ID --raw` prints the
canonical causal events as JSON lines instead.

A live `ora watch` in a terminal also opens a read-only browser window bound
to the run's recorded camera frames (`--world-view off` disables it; a
finished-run replay leaves it off). When the run executes on another machine,
`./scripts/ora-remote HOST --repo /path/on/HOST watch RUN_DIR` runs ORA there
over SSH and forwards the world view to a local browser ([remote GPU
server](remote.md)). To steer a run while it works, see [interactive
sessions](interactive.md).

Use `ora stop RUN_ID` to stop that active run. Avoid broad process kills that
could stop unrelated sessions or services.

## Replay a run in the browser

```bash
./scripts/ora web setup                          # once per checkout: builds ORA's web page (npm ci and a build in ora-pi/web)
./scripts/ora replay RUN_ID                      # a finished run on that page; prints a link with a launch token
./scripts/ora replay RUN_ID --export out.mp4     # the replay as a video
./scripts/ora replay RUN_ID --scene              # RoboTwin, LIBERO-Pro, RoboDojo: the run's scene trace, for the run dashboard's 3D replay
```

`ora replay` refuses until `ora web setup` has built the page, and says so.
The page and the viewer below listen on 127.0.0.1 of the machine they run on;
from another computer, forward that port with `ssh -L` ([remote GPU
server](remote.md)). `--export` needs two tools ORA does not install:
Playwright for Python with its Chromium, and ffmpeg with libx264 ([replay a
finished run](remote.md#replay-a-finished-run)).

## Read the evidence

| File / directory | Records |
|---|---|
| `plan.json` | Resolved configuration and source/asset identities |
| Canonical events and provider evidence | Requests, replies, tools, proposals, and receipts |
| `result.json` | Episode outcome and counters |
| `evaluation.json` | Deterministic validation and environment verdict |
| `seal.json` | Hash-bound inventory of accepted evidence |

Synthetic lands under `outputs/quickstart/runs/<run-id>/` by default. Robot
placement follows the workspace and execution namespace. See
[configuration and evidence](../reference/configuration-and-evidence.md).

## Open the viewer

```bash
uv run --locked ora-viewer --open-browser
```

The local read-only viewer joins the plan, conversation, trajectory, original
media, and official verdict. Diagnostic reconstructed media is labeled separately.

[Viewer guide](../../viewer/README.md)

## Export a sealed run

```bash
./scripts/ora trace-validate RUN_ID
./scripts/ora export RUN_ID
./scripts/ora export-sft RUN_ID
```

The transfer archive and training projection serve different purposes.
Canonical evidence and the seal remain authoritative; export does not change
the verdict or original evidence. [Connect a training loop](rl-interface.md#read-the-reward-and-export-the-calls)
lists the training projection's files.
