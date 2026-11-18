# Task files

## Responsibility

`ora task`: a task file (`ora.task.v1`) says what runs: simulator, task, seed,
layout, the model by name, the experiment, the policy selection and budgets.
A run binding (`ora.run-binding.v1`, or a scheduler's allocation file) says
where one run executes. `ora task run` composes the RunSpec through the
simulator's backend, launches it as `ora launch --yes` does, waits for it and
writes a task record with the result summary.

## What belongs here

- `task_file.py`: the schema, its validation and the overrides (`--task`,
  `--seed`, `--layout`, `--set KEY=VALUE`).
- `binding.py`: run bindings, the allocation-file reader, the model check
  and the run environment.
- `run.py`: compose, preflight, launch, wait, task record.
- `submit.py`: a scheduler request that runs `ora task run` in this checkout.

Not here: how a simulator's RunSpec is composed (the backend's
`compose_task`), scheduling, and evaluation.

## Dependency direction

Imports `ora.orchestrator.backends.registry`, `launch`, `paths` and `spec`;
never a backend module directly.

## Entry points

`ora task resolve|run|submit` (`ora/app/task.py`).

## Documentation

[Task files](../../../docs/reference/tasks.md).
