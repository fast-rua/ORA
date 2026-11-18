# Run ORA safely

A language model controls a simulated robot. What it can do, what the
Supervisor enforces, what it never sees, and what ORA does not protect.

## What the model can do

- Read-only tools (observation.dump_state, perception, geometry, compute)
  read public observations and change nothing.
- Proposal tools (robot.submit_action, or a surface's motion, gripper or
  skill names) are the only way to change the world; the Supervisor admits or
  refuses each proposal.
- check_task returns only the official true or false; finish_episode ends the
  episode. A claimed success is refused unless check_task reads true.
- In free play the agent is Pi: read, write, edit, bash, codemode and
  ora-robot in the run's workspace, and no lifecycle tools.

## What the Supervisor enforces

- Every world change is a versioned proposal, checked against the tool's
  closed schema and the current observation. A refusal names its code, stage
  and public constraint, never private state.
- LIBERO and RoboDojo bind a hard workspace envelope per embodiment: a command
  bound, not a promise of reach or of no collision. The planners model the
  robot, not the scene, so a planned path can still touch the table.
- One action at a time, each with a measured receipt and a fresh observation.
- Budgets bound robot decisions, model calls, simulation work and wall time.
- Operator messages enter only at a model-call boundary. Esc, /stop and
  `ora stop RUN_ID` end a run through the normal stop path.
- Only the environment's official boundary supplies success.

## What the policy never receives

Object ground truth, simulator segmentation or labels, contact state, the task
source, expert code or demonstrations, and the success predicate or checker.

## What ORA does not protect

- Free play runs as you: Pi's bash and file tools and your ! lines can read
  any file you can, including the key store /login writes. ORA removes
  credential-like variables before Pi starts but adds no sandbox; codemode is
  the sandboxed way to script. Your Pi extensions and skills load too.
- Keys live in environment variables or the user store (~/.config/ora,
  mode 0600). A plan records the profile and endpoint, never the key. Sealing
  refuses files named like credentials but does not scan file contents.
- Every request carries the prompt, tool results and images to the endpoint
  you chose.
- Benchmark runs are confined by Node's permission model
  (ORA_PI_NO_PERMISSION=1 lifts it), not sandboxed; simulator workers and
  perception services run as your user.
- On a shared server, other users can connect to the loopback world view.
- ORA has no real-robot runtime, approval flow or emergency stop. A passing
  doctor or preflight is not a physical safety certification.

## Before you share a run

Read its provider requests and responses (prompt, tool output, images), free
play's shell commands and `native/robot-socket.jsonl`, the endpoint and host
names in `plan.json`, and any session export; search the run for your key
prefixes. Share a sealed run as it is, or say that a redacted copy is no
longer the sealed evidence. Report a credential exposure, Supervisor bypass
or evidence tampering as SECURITY.md says.

More: docs/usage/safety.md.
