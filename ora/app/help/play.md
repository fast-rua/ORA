# Free play

`ora play` loads a random scene this host has qualified, with no task, and
opens the run's Pi terminal UI. You give the robot instructions one at a
time; the agent carries out each one, reports what it did and sees, and waits
for the next. A message sent while it works joins the running instruction.

    ora play --env synthetic                      choose a provider profile, then play
    ora play --env synthetic --provider env       ORA_BASE_URL, ORA_MODEL, ORA_API_KEY
    ora play --env libero --provider PROFILE --service-report REPORT --confirm-launch
    ora play --env robotwin --scene-set tabletop-v1 ...   one of ORA's RoboTwin scene sets
    ora play --env robodojo --scene kitchen_tidy ...      one Isaac Sim tabletop
    ora play --env ENV --prepare-only             write the RunSpec under outputs/ora-play/specs/
    ora play --env ENV --spec FILE --confirm-launch       launch a prepared one
    ora play audit --env robotwin|robodojo ...    qualify ORA's own play scenes on this host

## Keys

    Enter             the next instruction, or a message to the running one
    Esc               stop the running instruction at its next boundary
    /reset            load another of the scenes drawn for this session
    /view             the operator view: a camera for you, never for the agent
    /quit             end the session; ORA evaluates and seals the run
    /stop             operator stop; the run stays incomplete
    !command          a shell command in the run's workspace (!! keeps it out of the conversation)
    /model, Ctrl+P    choose or cycle the model; /thinking sets the reasoning effort
    Ctrl+D, /detach   leave the terminal; `ora attach` comes back

## What the agent has

The agent is Pi as a coding agent with the robot as a device: read, write,
edit and bash in the run's workspace, codemode (JavaScript in a QuickJS
sandbox that reaches the robot only through its tools), your Pi skills,
prompt templates, AGENTS.md files and extensions, and `ora-robot` with the
`ora.robot_client` Python client on its shell's path. Every robot call, from
the model or from a script, goes through the Supervisor one action at a time,
with its receipt. Free play offers no check_task or lifecycle tool, and the
scene's task language and success predicate are withheld.

Pi's shell runs as you and can read any file you can (`ora help safety`).

## Budgets

Each instruction: 24 robot decisions, 80 model calls, 20 minutes. The whole
session: 400 robot decisions, 1600 model calls, 4 hours. When an
instruction's budget runs out, it ends and the session waits.

## Evidence

A session is one run under `outputs/ora-play/`, evaluated and sealed at
`/quit`. It is never scored (`official_success: null`) and never learnable
(`free_play_session`), so it reaches no score and no SFT export. The agent
trace records every model call, tool call, script call with its origin,
shell command, model change and compaction.

More: docs/usage/play.md. Every option: `ora play --help`.
