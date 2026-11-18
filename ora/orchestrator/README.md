# ora/orchestrator

## Responsibility

The trusted side of a run. A RunSpec becomes a content-addressed RunPlan
(`spec/`); the plan passes the launch gates (`preflight/`); one supervised
episode binds it to an environment session, a tool runtime and a policy
session and runs the observe -> policy turn -> execute loop (`episode/`); the
finished run is evaluated, sealed and reported (`evaluation/`, `reports/`).
The Supervisor in `episode/` is the only path from a policy's proposal to a
state change, and the only reader of the environment's official verdict.

## What belongs here, and what does not

| Directory | Holds |
|---|---|
| `spec/` | RunSpec -> RunPlan: `run_spec.py` (`resolve_run_spec`, `load_run_spec`, `validate_run_plan`) and its section resolvers (`policy`, `native_policy`, `native_provider`, `runtime`, `composition`, `context_compactor`, `context_pack`, `operator_control`, `play`, `partition`, `result_tier`, `source`, `environment_context`, `latency`), the shared value rules and hashes (`values.py`), the benchmark cell builders (`benchmark.py`, `inputs.py`), the default instance (`defaults.py`), pinned policy data (`policy_pins.py`, `policy_lock.py`), the experiment registry (`experiment_identity.py`, `yaml_cache.py`) and `ora baselines --explain` (`baseline_explain.py`, `baseline_tool_routes.py`) |
| `preflight/` | The launch gates (`gates.py`, `checks.py`), execution resources and host profiles (`resources.py`, `resource_inventory.py`, `resource_admission.py`), host and output storage (`host_storage.py`) and the provider probe (`provider_probe_*.py`, `provider_image_answer.py`, `provider_gateway.py`) |
| `episode/` | One episode: `coordinator.py` (`run_episode`), `loop.py`, `actions.py` and `action_executor.py` (one admitted action or closed skill sequence), observation, rounds, terminal and result handling, free play (`play.py`, its operator view `play_operator_view.py`; a real cell's setup before its worker starts, `cell_setup.py`), the trusted stops (`termination.py`, `stagnation.py`, `operator_stop.py`, `wait_accounting.py`, `native_allowance.py`), live interaction and operator control, the policy session factory and the agent core selection (`agent_core.py`) |
| `evaluation/` | `evaluate.py` (`evaluate`, `seal`, `export`) and the checks it runs (`checks.py`, `learning.py`, `lossless.py`, `play.py`, `receipts.py`, `verdict_guard.py`), the records a launch derives (`metrics*.py`, `context_audit.py`, `context_manifest*.py`, `ownership_audit.py`, `visual_usage.py`, `decision_accounting.py`, `artifact_provenance.py`), the model-visible identity (`model_visible.py`), recorded-provider tape and raw request/image comparison (`provider_replay.py`, `request_comparison.py`, via `ora toolkit provider-replay`), evidence reading (`evidence_io.py`, `run_artifacts.py`) and `ora revalidate` (`revalidation.py`) |
| `reports/` | The read-only stop-reason and cost report (`run_report.py`) and its price table |
| `suite/` | The official-suite cell lists (`cells.py`) and their single-attempt launch queue (`queue.py`) |
| `play/` | Free-play scenes (`scenes.py`) and their plan binding (`binding.py`) |
| `toolkit/` | Offline script experiments at a recorded parent call; public evidence and external-only service assembly ([README](toolkit/README.md)) |
| `backends/` | Everything that belongs to one `runtime.kind`, behind `backends/registry.py` ([README](backends/README.md)) |
| `hosts/` | The execution host profiles ORA ships (`local-cpu`, `single-gpu`) |

The root keeps what every sub-package shares or what runs a process: `paths.py`
(workspace and output roots), `leases.py` (resource leases), `launch.py`
(launch, supervisor spawn and landing seal), `supervise.py` (the supervisor
process), `faults.py`, `import_closure.py`, `progress.py`, `world_view.py` with
`world_view_assets/`, and `identity-registry.json`. The `ora` command is not
here: it is `ora/app/` ([README](../app/README.md)), which parses, prints and
confirms and calls this package to do the work.

**Pinned in the root.** Seven RoboDojo backend modules stay at their paths
because qualification records and historical commits name them:
`robodojo_kernel.py` and `robodojo_qualification.py` (the owner paths of the
`WORKER_KERNEL_PATHS` manifest, read from past commits by path),
`robodojo_play_kernel.py` (the owner of `PLAY_KERNEL_PATHS`),
`robodojo_revalidation.py` (the lane revalidation checker),
`robodojo_scene_smoke.py` and `robodojo_layout_smoke.py` (the scene-smoke
`CHECKER_FILES`) and `robodojo_play_audit.py` (the play-scene audit launcher
the play kernel's files name). They are backend modules in all but location:
the import rules treat them as part of the RoboDojo backend, and they may be
edited only as section 12 of the structure contract allows. Their companion in
the environment layer, `ora/environment/robodojo_play_factory.py`, is a
`PLAY_KERNEL_PATHS` member: the coordinator swaps it in for a RoboDojo
free-play run, and renaming it re-runs the RoboDojo play audit.
`episode/action_executor.py` is the one orchestrator member of
`robodojo_revalidation.DEPENDENCIES`; it moved there from the root in the
restructure, so a lane revalidation certificate made before that move does not
load and the lane is re-signed (scene smoke, then revalidate).

Not here: simulator code (`ora/environment/backends/`, run in the isolated
worker), anything the policy sees or calls (prompts, tools, composition and
context are `ora/agent/`), and backend-specific rules in the core modules
(they are hooks in `backends/`).

## Dependency direction

`ora.orchestrator` imports `ora.agent`, `ora.environment` and `ora.contracts`,
never `viewer` or `ora.app`. The core reaches a backend module
only through `backends/registry.py`, which imports a backend on first use from
a `"module:attribute"` string; a backend imports the core freely. Package
modules import nothing eagerly: `ora/orchestrator/__init__.py` and
`spec/__init__.py` are lazy facades over the names they export.

The core sub-packages are not layered among themselves yet. `episode/` records
what `evaluation/` checks (`context_manifest`, `verdict_guard`,
`run_artifacts`) and `evaluation/` reads episode vocabularies
(`policy_finish`, `wait_accounting`, `operator_control`, `play`); `spec/` and
`preflight/` read `episode/agent_core.py` (standard library only) and
`spec/` reads `preflight/resources.py`. Code inside the orchestrator imports
the module that defines a name, never a package facade.

## Entry points

- `ora` (`ora.app.main:main`, in `ora/app/`) drives this package: `ora plan`,
  `launch`, `status`, `evaluate`, `seal`, `report`, `spec`, `qualify`, `suite`,
  `ladder` and the product commands.
- `from ora.orchestrator.spec import load_run_spec, resolve_run_spec, validate_run_plan`.
- `from ora.orchestrator import run_episode, EpisodeCoordinator` (the episode
  coordinator, without the CLI's launch admission).
- Operator tools, each an `ora` command: `spec/benchmark.py` (`ora spec
  robotwin|libero`: RoboTwin and LIBERO(-Pro) cells), `suite/queue.py` (`ora
  suite`: the official-suite queue), `reports/run_report.py` (`ora report`)
  and the backends' spec,
  qualification and play-audit tools (`ora spec|qualify ENV`, `ora play audit
  --env ENV`; [backends/README.md](backends/README.md)).
- `python -m ora.orchestrator.supervise`: started by `launch.py`, never by hand.
  The other process entries are listed in [ora/app/README.md](../app/README.md).

## Documentation

- [Runtime lifecycle](../../docs/reference/runtime-lifecycle.md) and
  [configuration and evidence](../../docs/reference/configuration-and-evidence.md)
- [Public interfaces](../../docs/reference/public-interfaces.md) and
  [plans API](../../docs/api/plans.md)
- [Code map](../../docs/code-map.md) and [architecture](../../docs/architecture.md)
- [Repository structure](../../docs/developer/repository-structure.md),
  sections 2, 5 and 12

`episode/execution_state.py` owns RoboDojo command and recovery state via
`agent/toolkit/state`, binds the RPC exchange and exports complete snapshots
at existing public publication boundaries. The toolkit-live host uses the
same owner; the worker never imports the state layer.

Synchronous toolkit-v2 binds `episode/tool_execution.py` beside the free-play script executor. `tool_budget.py` accounts actual native spend per parent; `tool_execution_log.py` persists intent, receipt/publication links and recovery evidence. `ExecutedActions` settles without dispatching twice. See [execution contract](../../docs/reference/toolkit-execution.md).

With observation receipts enabled, the same owner binds grasp feedback to its
completed actions and post-close publication. Derived provenance lives under
`native/tool-observation-feedback/`, separate from real worker receipts and
`native/tool-executions/`; it does not advance the physical or observation clock.

The episode `tool_execution` owner and `tool_execution_log` implement synchronous
parent-call budgets and durable recovery. `tool_execution_boundary` settles their
outcomes; `receipt_accounting` shares receipt accounting with the released proposal
path. Neither path repeats a completed primitive on a policy-process restart.
