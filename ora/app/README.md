# ora/app

## Responsibility

The `ora` command. `main.py` holds the one root parser, the command table
`COMMANDS` (command -> `"ora.app.<module>:register"`), the groups `ora --help`
prints (`GROUPS`, in English or Chinese by `ORA_UI_LANG` or the locale) and the
dispatch. Every other module owns one group of commands: `register(subparsers)`
adds them to the root parser and `run(args)` carries one out. The app parses,
asks, prints and draws; the work itself (resolving a plan, launching,
evaluating, sealing) is the orchestrator's, which a command imports inside
`run`. So `ora --help` and `ora status` load the app modules and never the
evaluation or the episode coordinator (`tests/ora/app/test_import_budget.py`).

## What belongs here, and what does not

The commands, in the order and groups of `ora --help`:

| Group | Commands | Module |
|---|---|---|
| Run an episode | `setup` | `setup.py` |
| | `doctor` | `doctor.py` |
| | `run` | `run.py` |
| | `play` | `play.py` |
| | `replay` | `replay.py` |
| | `web` | `web.py` |
| | `cameras` | `cameras.py` |
| | `interactive`, `attach`, `sessions`, `local` | `interactive.py` |
| Formal runs and evidence | `plan` (`validate`, `show`, `context`, `preflight`) | `plan.py` |
| | `launch`, `retry`, `status`, `stop`, `watch`, `workspace`, `task` (`resolve`, `run`, `submit`) | `runs.py` (`task`: `task.py`) |
| | `evaluate`, `seal`, `revalidate`, `export`, `export-sft`, `export-traces` | `evidence.py` |
| | `report` | `report.py` |
| | `metadata`, `state`, `metrics`, `ownership`, `trace-validate` | `evidence.py` |
| Environment qualification and suites | `spec` | `spec.py` |
| | `qualify` | `qualify.py` |
| | `suite` | `suite.py` |
| | `play audit` | `play.py` |
| Configuration | `providers` (`key`, `add`, `remove`, `test`) | `providers.py` |
| | `services` | `services.py` |
| | `cell` (`init`, `check`, `read`, `record-rest`, `grippers`, `calibrate`, `board`, `depth-service` and the operator verbs) | `cell.py` |
| | `baselines`, `lock` | `catalog.py` |
| Developer | `toolkit` (`replay`, `provider-replay serve/compare`) | `toolkit.py`, `toolkit_provider_replay.py` |
| | `dev` (`embodiment`, `prompt-bundle`, `task-contracts`, `cli-reference`) | `dev.py`; `cli-reference` is `cli_reference.py` |
| | `help` | `main.py`; the topic pages are `help/*.md` |

`experiments` (`catalog.py`), a development view of the experiment registry,
runs but is listed in neither `ora --help` nor the command reference.

The modules beside them:

| Module | What it does |
|---|---|
| `passthrough.py` | The leaves that hand the rest of the line, `--help` included, to an operator tool's own `main(argv)`: `report`, `suite`, `ladder`, the `dev` tools, and `spec ENV TOOL`, `qualify ENV TOOL` and `play audit --env ENV`, whose tools the environment's backend declares (`spec_tools`, `qualify_tools`, `play_audit` in `ora/orchestrator/backends/registry.py`), so no app module names a backend module |
| `help/` | `ora help TOPIC`: `runs`, `evidence`, `play`, `remote`, `safety`, `rl-interface`, short pages shipped as package data, each naming the documentation it is distilled from |
| `cli_reference.py` | `ora dev cli-reference`: renders `docs/usage/cli-reference.md` from the parser (`--write`, `--check`) |
| `local_setup.py`, `gpu_setup.py`, `gpu_recipes.py`, `env_file.py` | `ora setup`: the CPU LIBERO installation, the single-GPU recipes and the `.env` keys they own |
| `quickstart.py`, `run_services.py` | `ora run`: the Synthetic quickstart, and the confirmations and perception services of a GPU task |
| `local.py`, `task_catalog.py`, `backend.py` | `ora local`'s pinned CPU inputs; the prepared tasks of `ora interactive --spec`; the JSON-lines backend the pi-tui front end starts over them |
| `replay_export.py` | `ora replay --export`: draws the replay page into a video; standard library and Playwright only, run by path |
| `web_gateway.py`, `world.py` | `ora web`'s one origin behind a launch token; the read-only world view beside a terminal |
| `web_cell.py` | `ora web`'s real-cell routes (`/ora/cell`): the robot's questions, the auto-approve switch and the permission mode (`/ora/cell/mode/<mode>`), the held arms' alerts and the terminal's `ora cell` commands, for a run on a real cell |
| `remote.py`, `remote_session.py` | `ora-remote`, its own console script (standard library only, on the laptop), and the run an `ora-remote` command left going on the host |

Not here: anything a run's plan, evidence or verdict depends on. Resolving,
preflight, leases, launch, supervision, evaluation and sealing are
`ora/orchestrator/`; what a policy sees is `ora/agent/`. A new environment's
`spec`, `qualify` or play-audit tool is declared by its backend, not added
here.

## Dependency direction

`ora.app` is the top layer: it imports `ora.orchestrator`, `ora.agent`,
`ora.environment`, `ora.contracts` and the real-cell kit `ora.cell` (`ora cell`),
and nothing imports it except its own
entry points (`ora`, `ora-remote`, `python -m ora.app`). A command module keeps
its module-level imports to the standard library and light app or orchestrator
modules, and imports the implementation inside `run`. `remote.py` and
`replay_export.py` import only the standard library, and the package module
imports nothing.

## Entry points

- `ora` (`ora.app.main:main`), `python -m ora.app`, `scripts/ora`.
- `ora-remote` (`ora.app.remote:main`), `scripts/ora-remote`.
- `python -m ora.app.backend {watch,interactive} --emit jsonl`: started by
  `ora-pi/src/watch/index.mjs`, never by hand.

Every operator and authoring tool is an `ora` command; the Viewer is its own
console script, `ora-viewer` (`viewer/app.py`, with `ora-viewer compile
videos|previews`). The modules below keep a `python -m` entry point (or run as
a file by path) because a process starts them; none is an operator command.
They are the only `__main__` guards in `ora/`, besides three in pinned
inputs that may not change a byte:
`ora/orchestrator/robodojo_play_audit.py` (also `ora play audit --env
robodojo`), `ora/environment/backends/robodojo/motion_qualification.py`
(started by nothing) and `ora/agent/model_services/resident.py` (started in
process by `grounding_grasp`; its bytes are part of the perception adapter
identity every running service is bound by). `tests/ora/app/test_main.py` checks that this table and
the guards agree.

| Process entry | Started by |
|---|---|
| `ora.app` | `scripts/ora`; `examples/quickstart_smoke.py` |
| `ora.app.backend` | `ora-pi/src/watch/index.mjs` (the pi-tui front end of `ora interactive --spec` and `ora watch --ui pi`) |
| `ora/app/replay_export.py` (by path, standard library only) | `ora replay --export`, under a Python with Playwright |
| `ora.orchestrator.supervise` | `orchestrator/launch.py`: one launch's supervisor |
| `ora.orchestrator.backends.robotwin.evaluation_seeds` | `ora run --env robotwin` (its `qualify`, through `registry.Tool.command`); also `ora qualify robotwin` |
| `ora.orchestrator.backends.robodojo.lane_qualification` | `ora run --env robodojo` (its `card` and `task`, through `registry.Tool.command`); also `ora qualify robodojo` |
| `ora.environment.worker` | `environment/rpc.py`: the isolated simulator worker, under the simulator's interpreter |
| `ora.environment.replay_worker` | `viewer/replay.py` |
| `ora.environment.incident_replay_worker` | `ora replay --incident` (`orchestrator/evaluation/incident_replay.py`), under the plan's interpreter |
| `ora/environment/graphics_probe.py` (by path) | LIBERO launch preflight (`orchestrator/backends/libero_pro/preflight.py`) |
| `ora.environment.backends.libero_pro.probe` | LIBERO launch preflight |
| `ora.environment.backends.libero_pro.preview` | `ora-viewer compile previews` (`viewer/libero_previews.py`) |
| `ora.environment.planners.curobo_v2.worker` | `environment/controllers/motion_planner.py`: the cuRobo V2 planner worker |
| `ora.environment.planners.timing.worker` | `environment/controllers/motion_retime.py`: the TOPP-RA + Ruckig timing worker |
| `ora.environment.backends.robodojo.control_replay` | `ora qualify robodojo motion-replay` |
| `ora.environment.backends.robodojo.toolkit_worker` | `ora toolkit replay --live`, through the backend registry; independent diagnostic entry |
| `ora.environment.backends.robodojo.physics_scene_probe` | `ora qualify robodojo physics-probe` |
| `ora.environment.backends.robodojo.play_scenes.audit` | `ora play audit --env robodojo` |
| `ora/environment/backends/robodojo/graphics_compat.py` (by path) | the RoboDojo installer and `ora doctor --env robodojo` |
| `ora/environment/backends/robodojo/installation/setup_runtime.py` (by path, standard library only) | `scripts/setup-robodojo`, `ora setup --env robodojo` |
| `ora/environment/backends/robodojo/installation/rebuild_sdk_wheel.py` (by path) | the RoboDojo installation guide, under the Isaac interpreter |
| `ora.environment.backends.robotwin.play_scene_audit` | `ora play audit --env robotwin` |
| `ora.environment.backends.robotwin.qualification` | each `ora qualify robotwin qualify` lane and its own re-execution; console `robotwin-qualify` |
| `ora.environment.backends.robotwin.vendored_install` | `ora setup --env robotwin` |
| `ora.cell.depth.service` | `ora cell depth-service`; by hand under whatever interpreter holds the depth model, on any machine the cell reaches |
| `ora.agent.model_services.grounding_grasp` | `ora services start` (`agent/model_services/operator.py`) |
| `ora.agent.providers.recovery_worker` | `agent/providers/recovery.py` |
| `ora.agent.toolkit.host.compute.evaluator --worker` | itself, for one sandboxed computation |
| `ora/agent/evidence/openai_package_tool.py` (by path, standard library only) | copied into every `ora export-sft --view openai` package as `ora_package.py`; whoever receives the package runs it there |
| `ora/agent/evidence/trace_dataset_tool.py` (by path, standard library only) | copied into every `ora export-traces` dataset as `ora_traces.py`; whoever receives the dataset runs it there |
| `ora.robot_client` | the policy's sandbox shim (`agent/pi/robot_socket.py`, `python3 -S -m ora.robot_client`); console `ora-robot` |

## Documentation

- [Command reference](../../docs/usage/cli-reference.md), generated from the
  parser by `ora dev cli-reference --write` (`tests/ora/app/test_cli_reference.py`
  fails on a stale page)
- [Commands and what they write](../../docs/usage/commands.md)
- [Runs](../../docs/usage/runs.md), [play](../../docs/usage/play.md),
  [interactive](../../docs/usage/interactive.md), [remote](../../docs/usage/remote.md),
  [safety](../../docs/usage/safety.md), [training loops](../../docs/usage/rl-interface.md):
  the pages the `help/` topics are distilled from
- [Repository structure](../../docs/developer/repository-structure.md), section 6
