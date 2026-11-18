# Run ORA safely

ORA puts a language model in control of a simulated robot. This page says
what the model can do in each mode, what the Supervisor enforces on its
behalf, what the model is never shown, and what ORA does not protect. Read it
before you run free play on a machine you share, point ORA at a new model
endpoint, or hand run evidence to someone else.

For the overall structure, see [how ORA works](../how-ora-works.md). The
exact inputs and outputs of the policy are in [policy I/O](../reference/policy-io.md).

## What the model can do

In every mode the model works through the tools its ToolSurface advertises.
The surface is part of the plan, so a run cannot gain a tool while it is
running.

| Tool kind | Examples | Effect |
|---|---|---|
| Read-only | `observation.dump_state`, perception, geometry and `compute` tools | Read public observations and derived values. They cannot change robot state. |
| Proposal | `robot.submit_action`, or the surface's own motion, gripper or skill names | The only way to change the world. Each call is a proposal that the Supervisor admits or refuses. |
| Lifecycle | `check_task`, `finish_episode` | `check_task` returns the official boolean for the current state. `finish_episode` ends the episode with the model's claimed outcome. |

On the default benchmark surfaces, `check_task` returns only `task_success:
true` or `false`, read from a record the Supervisor writes from the
environment's official boundary. The model gets no checker code, predicate or
reason. A `finish_episode` call that claims success is refused unless
`check_task` reads true, and neither tool can set the official verdict.

In free play ([`ora play`](play.md)) the agent is Pi working as a coding
agent, and it has more:

- Pi's read, write, edit and bash tools, working in the run's workspace;
- `!command` lines you type, which run in that workspace too;
- Pi's `codemode` tool, which runs short JavaScript in a QuickJS sandbox with
  no file system, network or shell, reaching the robot only through its tools;
- `ora-robot` and the `ora.robot_client` Python client on the shell's path,
  so a script in any language can call the robot tools.

Free play does not advertise `check_task` or the lifecycle tools, and scripts
may not call lifecycle tools. Every robot call a script makes still goes
through the Supervisor, one action at a time.

## What the Supervisor enforces

The Supervisor is the trusted part of ORA that sits between the model and the
simulator. The model cannot bypass it, in any mode.

- **Action validation.** Every world change crosses the Supervisor as a
  versioned action proposal. Arguments are checked against the tool's closed
  schema and against the current observation before anything is dispatched.
- **Typed refusals.** A refused proposal returns a stable code, the failed
  stage and the public constraint it broke, for example a bound and its
  units. Private paths, object state and stack traces stay out of the
  refusal. Repeating the same invalid request is refused again.
- **Workspace envelopes.** LIBERO and RoboDojo bind a hard action envelope per
  embodiment. A target outside it is refused. The envelope is a command
  bound, not a promise that the target is reachable or free of collisions.
- **Motion planning.** The planners check that a target is reachable before
  the robot moves, and the dual-arm RoboTwin profile keeps a minimum clearance
  between the two arms. The RoboTwin and RoboDojo planners model the robot,
  not the scene, so a planned path can still touch objects or the table. The
  LIBERO planner's global routes also avoid a collision world built from the
  current depth of both cameras (`world_model: public_depth_voxel_v1`), but
  that world cannot see behind objects and clears the region around the goal
  hand, and the short local OSC moves do not check it, so contact remains
  possible there too. Contact shows up in measured state and in the receipt.
- **One action at a time.** Each admitted action returns a measured receipt
  and a fresh observation before the next one. In free play, once a script
  has acted in a turn, the model's own world-changing call in that turn is
  refused.
- **Budgets.** Robot decisions, model calls, simulation work and wall time
  are bounded per run. Free play adds per-instruction limits (24 decisions,
  80 model calls, 20 minutes by default).
- **Operator control and stop.** Operator messages are admitted only at a
  model-call boundary; they cannot alter a request in flight, run a tool or
  override admission. Esc, `/stop` and `ora stop RUN_ID` end a run through the
  normal stop path, and `ora stop` checks the run's process identity before
  signalling it.
- **Verdict.** Only the environment's official boundary supplies success
  (RoboTwin `check_success`, LIBERO termination, RoboDojo's task verdict).
  The model's own claim, in text or through `finish_episode`, cannot change it.

## What the policy never receives

The model sees public camera images, proprioception, task language, the robot
manual and its tool results. It never receives:

- object ground truth (poses, identities or states read from the simulator);
- segmentation masks or labels from the simulator;
- contact state;
- the task source, expert code or demonstrations;
- the success predicate or the checker behind `check_task`.

Perception tools such as segmentation and grounding run models on the public
camera images; they do not read simulator labels. In free play the scene's
task language and completion contract are withheld as well, and the official
success check is disabled for the scene.

## What ORA does not protect

- **Free play runs as you.** Pi's bash and file tools, and your `!` lines, run
  on the host with your user's permissions. ORA removes environment variables
  whose names look like credentials before Pi starts, but the shell can still
  read any file you can, including the key store that `/login` writes. ORA
  adds no sandbox of its own; codemode is the sandboxed way to script. Pi's
  own isolation options (a micro-VM or Docker Sandboxes) can be added as Pi
  extensions where the host supports them.
- **Your Pi configuration is loaded.** In free play your Pi skills, prompt
  templates, `AGENTS.md` files and extensions are loaded. Treat them as code
  that runs with the agent.
- **Model API keys.** Keys live in environment variables or the user-level
  store (`pi-auth.json`, mode 0600, outside the repository). A plan records
  the profile and endpoint, never the key, and sealing refuses files whose
  names look like credentials (outside a code-mode run's sandboxed workspace,
  whose names the model chooses). Sealing does not scan file contents, so a key
  printed by a free-play shell command would be kept in the trace.
- **Data sent to the model provider.** Every request carries the prompt, tool
  results and camera images to the endpoint you chose. Pick an endpoint whose
  data policy you accept.
- **Benchmark runs are confined, not sandboxed.** The default agent core runs
  under Node's permission model, limited to its own sources, the run's agent
  evidence and the key store. Setting `ORA_PI_NO_PERMISSION=1` removes that
  limit. The simulator workers and perception services run as your user.
- **Shared servers.** The terminal world view listens on the server's
  loopback address; other users of that server can connect to it and see the
  camera frames. The web view is protected by a 0600 socket and a one-time
  launch token ([remote use](remote.md#security)).
- **Uploads.** In free play, `/share` and `/bug` upload the conversation
  after a confirmation that says what is sent and where.

## Modes at a glance

| Mode | Who acts | Tools | What is recorded | Scored |
|---|---|---|---|---|
| Benchmark run (`ora run`, `ora launch`) | The model alone | The plan's ToolSurface: read-only, proposal and lifecycle tools | Plan, requests and responses, tool calls, actions, receipts, images; evaluated and sealed | Yes, by the official boundary. Synthetic records `official_success: null`. |
| Interactive (`ora interactive`, the task picker) | The model; you can steer and stop | The same ToolSurface; Pi's shell, file and session-rewriting commands are refused | As a benchmark run, plus each operator message as an event | The verdict is kept, but a run with any message is marked human-assisted and left out of autonomous success rates |
| Free play (`ora play`) | You instruct; the model acts | Robot tools without `check_task` or lifecycle; Pi's coding tools, codemode, `ora-robot`, `!` lines | Everything above, plus shell commands and output, script calls with their origin, model switches and compactions | Never; `official_success: null` |
| Remote and web (`ora-remote`, `ora web`, `ora watch`) | Whoever acts in the underlying run | The underlying mode's; the browser has no shell or file route, and the world view is read-only | Nothing new in the run; `ora web` writes nothing into it | As the underlying mode |

See [interactive sessions](interactive.md#what-a-message-changes) and
[evaluation protocol](../evaluation/protocol.md) for how human-assisted runs
are counted.

## Before sharing evidence

A run directory is a full record of what happened. Before you send one, or
an export, to anyone:

- [ ] Read the provider requests and responses: they contain the full prompt,
      tool output and every image the model received.
- [ ] For free play, read the shell commands and their output in the agent
      trace and `native/robot-socket.jsonl`; they can include file contents
      and absolute paths from your machine.
- [ ] Check `plan.json` for the provider endpoint and host profile names you
      do not want to publish.
- [ ] Check HTML or JSONL session exports (`ora sessions export`) the same
      way; they carry the task, prompts and tool output.
- [ ] Confirm no key appears anywhere: search the run for your key prefixes,
      since sealing checks file names, not contents.
- [ ] Share the sealed run as it is. Redact a copy if you must, and say that
      it is no longer the sealed evidence.

## Real robots

ORA's supported runtimes are Synthetic, LIBERO and LIBERO-Pro, RoboTwin and
RoboDojo, all simulated ([supported configurations](../support.md)). ORA has
no real-robot runtime, operator approval flow or emergency-stop integration,
so real robots are outside the supported scope. A passing `ora
doctor` or preflight is not a physical safety certification. If you connect
ORA to hardware yourself, keep a person at an independent emergency stop and
treat the workspace envelopes and planners described above as simulation
limits, not physical guarantees.

To report a credential exposure, a Supervisor bypass or evidence tampering,
follow `SECURITY.md` in the repository root.
