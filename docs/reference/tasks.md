# Task files

A task file says what a run does, once; each run of it says only where it
executes. `ora task` resolves a task with overrides, runs it on a binding, or
submits one run of it to a scheduler.

## The task file (`ora.task.v1`)

```yaml
schema: ora.task.v1
simulator: robodojo
experiment: robodojo-toolkit-v6         # the output namespace
task: stack_bowls
seed: 0
layout: 0
model: deepseek-ai/DeepSeek-V4.1-Flash-1017   # by name; the bound replica must serve it
policy:                                  # optional, see context schemes and policy selection
  tool_surface: robodojo_placement_toolkit_v6
  context_scheme: context_keep_written   # RoboDojo's default; name another to change it
  # baseline, toolkit_grants, toolkit_receipt_delivery, task_file
  # reasoning_effort, max_output_tokens: the model request, over the Provider profile's values
record_toolkit_io: true
ground_truth_capture: true
save_sandbox: true                       # false: sandbox/ keeps only its hashes after sealing
budgets: {}                              # max_robot_decisions, max_model_calls, wall_time_s, max_sim_steps
```

Required: `schema`, `simulator`, `experiment`, `task`, `model`. Unknown keys and
wrong types are refused (`ora/orchestrator/task/task_file.py`). The file never
names the card, the resource profile, the model replica or its endpoint, the
perception services, the execution host or an admission cohort: those belong
to a run. Nor does it name a code version: a run uses the checkout it runs in.
[`examples/tasks/robodojo-toolkit-v6.yaml`](../../examples/tasks/robodojo-toolkit-v6.yaml)
is a complete example. RoboDojo runs evaluation seed 0 on the reviewed layouts; with
`policy.task_file` (a task file under `ora/policy/task_files/` in place of the reviewed
completion contract) `seed` may be 0-2 and `layout` any official layout. Otherwise choose the cell by
`layout`.

A RoboDojo task on the default selection (no `tool_surface`, or
`robodojo_placement_toolkit_v6`) runs code mode, its baseline's default since
2026-11-05 ([code mode by default](context-schemes.md#code-mode-by-default)), so
its run binding's host needs the code-mode sandbox (bubblewrap, user namespaces
and the interpreter `ora setup --env robodojo` builds). A task file has no
`code_mode` field: `robodojo_placement_toolkit_v5` (baseline
`robodojo_toolkit_v6`) keeps tool calls, and so does a RunSpec composed from the
task with `policy.code_mode: null`.

`policy.reasoning_effort` and `policy.max_output_tokens` set the model request
over the values of the Provider profile the run binding names (for example
`medium` and `32768` where the profile says `high` and `16384`). The RunSpec and
the plan carry the values sent; the task record keeps both under
`model_request_overrides` (`{field: {profile, run}}`). Preflight probes the
Provider with the run's values, and the context window must still leave an input
budget.

While it waits for its run, `ora task run` also prints the run's plain transcript (`ora watch --plain`) on
its own output: the task instruction, the model's replies and each tool call with its result summary, images
only as a count and no reasoning. A scheduler that keeps a task's output (ora-router's task log) shows it as the
run goes. `ORA_TASK_TRANSCRIPT=off` leaves it out.

The Supervisor runs in its own session, so a scheduler that stops the task by signalling its process group would
not reach it. `ora task run` therefore passes SIGTERM, SIGINT and SIGHUP on as a graceful stop of its run
(`ora stop`): the run seals as interrupted and releases its lease and GPU, and `ora task run` waits for that
before it exits. The task record notes the signal under `stopped_by_signal`.

## Overrides

Every `ora task` command takes `--task NAME`, `--seed N`, `--layout N` and any
number of `--set KEY=VALUE`, where KEY is a task key (`experiment`,
`policy.context_scheme`, `budgets.max_robot_decisions`, ...) and VALUE is read
as YAML (`3`, `true`, `{"tools": ["check_task"], ...}`). A key the schema does
not define is refused, so a run-bound value cannot be overridden.

```bash
./scripts/ora task resolve examples/tasks/robodojo-toolkit-v6.yaml \
  --task align_blocks --layout 3 --set budgets.max_robot_decisions=40
```

## The run binding (`ora.run-binding.v1`)

`ora task run` takes `--binding FILE`, or reads the allocation file a
scheduler names in `ORA_ROUTER_ALLOCATION` (its `lane.card`, `build_spec` and
`llm` fields). A binding file:

```json
{"schema": "ora.run-binding.v1", "gpu": 3, "resource_profile": "robodojo-gpu3",
 "provider_catalog": "/path/catalog.yaml", "provider_profile": "my-replica",
 "service_report": "/path/readiness.json", "cohort": "optional-cohort",
 "env": {"ORA_SAM_PORT": "8214", "ORA_GRASP_PORT": "8215", "ORA_EXECUTION_HOST": "my-host"},
 "models": ["short-model-name"]}
```

The task's `model` must be one of `models` or the Provider profile's `model`.
The run's environment is the process environment plus `env`, plus the
profile's `base_url_env` and `model_env` set to its `endpoint` and `model`
where they are unset. `ORA_WORKSPACE_ROOT`, `ORA_OUTPUTS_ROOT` and
`ORA_HOSTS_DIR` are read when `ora` starts, so a binding may restate them but
not change them.

## `ora task run`

```bash
./scripts/ora task run TASK.yaml [overrides] [--binding FILE] [--summary FILE] [--no-wait]
./scripts/ora task run --resolved '<task JSON>' ...
```

The simulator's backend composes the RunSpec (`OrchestratorBackend.compose_task`;
RoboDojo: `ora spec robodojo` with the task's policy selection), written to
`outputs/<experiment>/specs/<config>--<stamp>-<pid>.json`. The run is then
resolved, preflighted and launched as `ora launch --yes` does, and the command
waits for its Supervisor. The task record
`outputs/<experiment>/tasks/<run_id>.json` (`ora.task-run.v1`; also `--summary`)
holds the task, the overrides, the binding, the RunSpec, the plan hash, the
source commit and a summary: `run_id`, `run_dir`, `state`, `official_success`,
`termination`, `decisions`, `model_calls`, `evaluation_status`, `sealed` and
`ground_truth_dir`. The run directory itself is not touched. A blocked
preflight writes `tasks/blocked--<spec>.json` and exits 2.

### Start pacing

`ora task run` launches at once. Limiting how many simulators start together
on one host is the scheduler's job: a router admits a task only when its node
has a start quota free, so a waiting task holds no card.

## `ora task submit`

```bash
./scripts/ora task submit TASK.yaml [overrides] [--defaults FILE] [--owner O] [--print] \
  [--scheduler-url URL] [--priority P] [--window W] [--workspace DIR] [--cpu] \
  [--env NAME=VALUE] [--env-file NAME=FILE]
```

Builds a scheduler request (`type: submit`, `owner`, `name`, `ora` with the
cell, `resources`, and `run` with `argv`, `cwd`, `env`, `env_files`) whose
command is `scripts/ora task run --resolved '<task>'` in this checkout, and
POSTs it to `<url>/api/requests` (`--scheduler-url`, `ORA_SCHEDULER_URL`, or
`scheduler_url` in the defaults file). `--print` prints the request instead.
`--summary PATH` makes the run also write its task record at PATH (on storage
the run's node shares), so the submitter knows where to read the outcome.
The request carries no code version; to pin one, submit from a worktree of
that commit and leave it alone until the run ends. `--defaults FILE` (YAML)
holds standing fields: `scheduler_url`, `owner`, `priority`, `job_type`,
`window`, `workspace`, `cpu`, `pin`, `env`, `env_files`; command-line flags win.
