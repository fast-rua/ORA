# Runs: plan, launch, follow, finish

One run is one episode of one plan. `ora run` does every step for you; the
commands below are the same steps one at a time, for a RunSpec you composed
(`ora spec ENV compose ...`) or were given.

## Quick runs

    ora run --env synthetic         a bounded episode: no simulator, GPU or download
    ora run --env libero|robotwin|robodojo --task TASK
                                    one GPU task on the single-gpu host, in Pi's terminal UI
    ora local                       the CPU MuJoCo LIBERO task picker on this computer

## A RunSpec, step by step

    ora plan validate SPEC          parse, validate and bind it; print the plan identity
    ora plan show SPEC              the resolved RunPlan and its launch confirmation code
    ora plan context SPEC           the exact model context and tools, without an episode
    ora plan preflight SPEC         source, host, environment, provider and agent-core gates
    ora launch SPEC                 preflight again, ask for the confirmation code, launch

`launch` is the only step that starts an episode. A preflight can call the
provider, which may be billed. `--yes` answers the confirmation for a plan
you already reviewed.

## Follow and stop

    ora status [RUN_ID]             one run, or every lease and indexed run here
    ora watch RUN_ID                the transcript; a finished run replays (--speed 0: at once)
    ora attach [RUN_ID]             a running Pi-core run's terminal UI; Ctrl+D detaches
    ora web [RUN_ID]                the same run in the browser (ora web setup first)
    ora replay RUN_ID               a finished run in the browser (ora web setup first)
    ora stop RUN_ID                 stop that run only, after checking its lease and process

A RUN_ID resolves under this workspace's outputs root (ora workspace prints
it) and the roots in ORA_EXTRA_OUTPUT_ROOTS. For a run anywhere else, pass its
directory: ora watch path/to/<experiment>/runs/<run-id>.

Stop a run with `ora stop`, never with a broad process kill: other runs and
the perception services share the host. A stopped run stays incomplete.

## Finish

When a run lands, its Supervisor rebuilds the metrics, evaluates the run and
seals it; a refused seal is written to `seal-refusal.json`. By hand:

    ora metrics RUN_ID              rebuild the Runtime metrics and context audit
    ora evaluate RUN_ID             the official verdict and the evidence checks
    ora seal RUN_ID                 make the evaluated run immutable
    ora retry RUN_ID                a new run for an unsealed attempt that sent no robot action

A sealed run is never retried or rewritten. `ora revalidate` reviews finished
runs into separate reports and changes nothing in them.

## Where runs are

`ora workspace` prints the evidence and state roots (`ORA_OUTPUTS_ROOT`,
`ORA_WORKSPACE_ROOT`). Each run is a directory named by its exact run ID;
`ora help evidence` says what it holds.

More: docs/usage/runs.md. Every option: `ora COMMAND --help`, or
docs/usage/cli-reference.md.
