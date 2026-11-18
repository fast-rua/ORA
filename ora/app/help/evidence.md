# Evidence: what a run records

A run is a directory named by its exact run ID under the outputs root
(`ora workspace` prints it). The Synthetic quickstart lands in
`outputs/quickstart/runs/`, free play in `outputs/ora-play/`, and a composed
RunSpec's run in its experiment's namespace, `outputs/<experiment>/`.

## The files

    plan.json             the resolved RunPlan: configuration, source commit, asset hashes
    preflight.json        the admission gates the launch passed
    launch.json           the confirmed launch
    events.jsonl          the Supervisor's events: proposals, receipts, terminations
    agent-events.jsonl    the agent trace (ora.agent-event.v1), hash-chained: every
                          model call, tool call and result, by model_call_id
    native/               provider requests and other content-addressed artifacts,
                          the Pi session, the execution capture
    result.json           termination, counters and official_success
    evaluation.json       the deterministic evaluation and learning eligibility
    seal.json             the hash-bound inventory of the accepted evidence

After sealing, the provider requests are packed losslessly into
`native/provider-requests.tar.zst`; every reader still sees the original bytes.

## Who decides success

Only the environment's official boundary: RoboTwin's check_success, LIBERO
termination, RoboDojo's task verdict. The model's own claim never counts.
Synthetic and free play record `official_success: null`. A run with any
operator message is marked human-assisted.

## Inspect a run

    ora metadata RUN_ID         the context and environment manifest
    ora state RUN_ID [--step N] the public policy state, its hash chain checked
    ora metrics RUN_ID          context, token, cache, image and latency measurements
    ora ownership RUN_ID        who decided what: model, robot actions, planners
    ora trace-validate RUN_ID   the agent trace, request hashes and artifact graph
    ora report RUN ...          stop reasons, validity, tokens and cost of several runs
    ora-viewer --open-browser   the read-only viewer: plan, conversation, trajectory, media

Derived reports never replace the raw evidence they are rebuilt from.

## Take it elsewhere

    ora export RUN_ID           a transfer archive of one sealed run, inventory verified
    ora export-sft RUN_ID       the model calls as OpenAI tool-call JSONL, for training
    ora sessions export RUN_ID  the Pi session as HTML or JSONL

Export never changes the run or its verdict. `export-sft` refuses a run
without a valid evaluation, and one that is not learning-eligible
(incomplete, human-assisted, free play, a runtime fault). Before you share a
run, read `ora help safety`.

More: docs/usage/runs.md and docs/reference/configuration-and-evidence.md.
