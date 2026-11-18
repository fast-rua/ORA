# Runtime lifecycle

## Plan

`load_run_spec` and `resolve_run_spec` in `ora.orchestrator.spec` validate inputs, select defaults, bind assets and produce a RunPlan. A plan binds the exact source commit, policy composition, benchmark revisions, provider settings, resources and budgets. `validate_run_plan` checks persisted identity before launch.

The Native policy sees public observations, task instructions, robot manuals and advertised tools. It never receives object ground truth, contact state, task source, expert code or the private success predicate. Perception results are computed from public camera data.

## Admit and launch

`ora plan preflight RUN_SPEC` checks the clean source, execution host, simulator/runtime, provider, resident services, resources, required qualification and that the plan's agent core can start. A plan whose rollout compactor is bound to its own Provider profile ([context compactor](configuration-and-evidence.md#context-compactor)) adds one check after `provider_marker`: `compactor_provider_marker` probes the binding's endpoint with its own key, running the marker and image stages and no tool-continuity stage (the compactor reads images and calls no tools), with one rerun on a response-level miss and the key redacted from the detail. Without a binding the check list is unchanged. `ora launch RUN_SPEC` repeats those checks and requires the plan's confirmation code; `--yes` records explicit non-interactive confirmation.

The launch path atomically claims the exact execution resource and any bound cell admission. Qualified simulator slots may coexist when their source-tracked resource bindings allow it; unprofiled plans are host-exclusive. The supervisor rechecks source identity before opening sessions. A newer Git commit alone does not establish or invalidate physical qualification: its declared dependencies determine reuse.

`ora run --env synthetic` invokes that same launch path for a bounded simulator-free demonstration. The command itself authorizes this one Synthetic episode; it does not launch a benchmark task.

## Execute

The episode loop opens one policy session and one environment session. On the default Pi agent core the policy session starts `node ora-pi/src/main.ts` with a socket for JSON-lines RPC; the Node process may write only the run's agent evidence (`agent-events.jsonl`, `native/`), receives images as sha256 references, and calls back into Python for every tool, observation and static context slot. If it exits, the policy session restarts it a bounded number of times and resumes from the Pi session file without running any tool twice. Requests and tools use the resolved contracts. Read-only tools cannot mutate robot state. Every mutation crosses the Supervisor as a versioned action proposal before the environment executes it. The environment returns measured receipts and fresh observations; the policy cannot turn an attempted action into an official success claim.

The v3 atomic action envelope separates pose motion from gripper motion. Target-state v4 accepts pose/gripper independently and stages a combined proposal as pose-first, conditional-gripper execution. Motion-action v3 supports the declared motion interface. Physical contact can change measured state without itself being an infrastructure fault.

With `operator_control`, the episode loop checks the run's inbox before each Native model call. A steering message enters the conversation at that boundary; it cannot alter a request in flight, execute a tool or override Supervisor admission. An abort uses the normal stop path below.

A free-play plan (`ora play`, `episode_mode: "play"`) runs the same loop one operator instruction at a time, and each act is Pi's own agent run (Pi's context, provider code and coding tools, with the robot tools in Python). Between instructions the interaction waits in `awaiting_instruction`. An instruction ends with one of: the agent's text reply (its report); the operator's `stop_instruction`, applied before the next model call and never inside one; its per-instruction budget; or the end of the scene. A message the operator sends while an instruction runs joins it (delivered before the next model call), and the agent decides whether it answers it and continues or takes it as the new objective; a `queued_instruction` instead waits for the running instruction to end and then opens the next one. `reset` restarts the world on the plan's next scene, and `quit` ends the session. Free play latches no success and persists no verdict.

Budgets bound decisions, model calls, simulation work and wall time. The Supervisor checks a round's budgets before every policy turn, and the agent core checks the model-call budget before every model call inside a turn, continuations after a reply without a tool call included: a turn is capped by the smaller of `max_model_calls_per_observation` and the round's model calls left when the turn starts (the turn's own budget, not the one published with the observation, which is stale on a further turn on the same observation). A turn stopped by that cap returns `model_call_limit`, and the next pre-turn check ends the round on `model_call_budget` at exactly the budget. Should a turn still spend past the budget, the post-turn guard ends the round on `model_call_budget`; an action that turn proposed is not executed and the agent trace closes it with `action_discarded` ([episode evidence](configuration-and-evidence.md#episode-evidence)). A persistent multi-round session must be selected explicitly. Cross-episode Task Memory is disabled; recoverable context artifacts remain episode-local evidence.

## Land, inspect and seal

The environment's official boundary supplies success: RoboTwin `check_success`, LIBERO termination or RoboDojo's task verdict. Synthetic reports `official_success: null`, as does a [play episode](configuration-and-evidence.md#play-episodes), whose LIBERO or RoboTwin scene loads with that boundary withheld.

An action receipt's `terminal_after_action` is the environment session's terminal status after the action: official success, the benchmark's own limit, the run's `max_sim_steps`, an abort or a control fault, on LIBERO, RoboTwin, RoboDojo, RoboCasa365, RoboLab and YAM alike. An action that runs in several segments stops at the segment that ended the episode and returns that segment's receipt; the loop then meets the terminal status and ends the round on its reason, persisting the verdict.

On landing the supervisor derives metrics and context audits, evaluates evidence and attempts sealing. An invalid trace receives a seal refusal rather than a fabricated seal. Provider request storage can be compacted losslessly after sealing. Resource leases and per-run scratch are released for this exact run. The Supervisor belongs to the process (or cluster job step) that launched it; an interactive client reports the run finished only after the Supervisor has exited, so keep that process open until then.

`ora status`, `watch`, `metadata`, `state`, `metrics` and `ownership` inspect recorded state. `ora stop RUN_ID` verifies the selected run's process identity before signaling it. A stop (or Esc in the Pi TUI) that lands while a model request is in flight answers that request with a `provider_error` marked `interrupted: true`; the trace verifier accepts the mark only after the session's `session_interrupted` event, so the stopped run evaluates incomplete without a fault and seals, while an unmarked failed request stays `request_without_response`. Retry is restricted to an unsealed incomplete attempt before any robot action was dispatched. Sealed evidence is never rewritten by revalidation; reports are written separately.

[Configuration and evidence](configuration-and-evidence.md) describes paths and identities. The [command reference](../usage/cli-reference.md) lists every command and option.

## Single-GPU entry

`ora run --env libero|robotwin|robodojo --task TASK` (`ora/app/run.py`) runs these
stages for one task on the shipped `single-gpu` host and stops at the first
failure:

1. `env_file.load_into` (`ora/app/env_file.py`) loads `.env` without overriding or blanking the
   process environment; the host marker is bound for this process only (a
   different marker is refused), an unset `DEEPSEEK_BASE_URL` defaults to the
   profile endpoint and `preflight.provider_marker_configuration` runs offline.
   A dirty checkout is refused.
2. The backend's run cell (`backends/<backend>/run_cell.py`) decides reuse with preflight's own validators:
   `spec.benchmark.admitted_seed_manifest` for a RoboTwin seed manifest, and
   `robodojo_qualification.load_qualification` / `load_card_qualification` for
   a RoboDojo card/task pair, plus GPU 0's `nvidia-smi` UUID and the worker
   interpreter readback. Missing records are produced by the backend's qualification tools
   (`ora qualify robotwin prepare|qualify`, `ora qualify robodojo card|task`, started as
   subprocesses with `registry.Tool.command`) after an explicit confirmation (`--yes` off a
   terminal).
3. `run_services.ensure_services` (`ora/app/run_services.py`) starts the resident services only when
   no listener is ready, refuses a ready listener whose pid and process
   identity do not match this checkout's `model-services` record
   (`foreign_listeners`), and binds the newest readiness report whose recorded
   service health equals the running instances and is under 20 h old
   (`REPORT_REUSE_MAX_AGE_S`), else writes a new one. It runs after stage 2, so
   a long scan cannot age the report out. A report binds when its services
   report this checkout's wire `protocol_version`
   (`resident.PROTOCOL_VERSION`, bumped by hand when a request or response
   format or meaning changes); the adapter content hash of the service and of
   the checkout is recorded in each service's `qualification.protocol` as
   evidence, never compared. `ORA_SERVICE_PROTOCOL_CHECK=off` binds a service
   of any version (or one that predates the field), and the binding records
   `check: off`.
4. The RunSpec has experiment `ora-interactive`, the host's `<env>-gpu0`
   slot, the readiness report and a new single-attempt cohort
   `ora-run-<env>-<UTC time>`. `ora interactive` repeats full preflight and
   launches only with `--confirm-launch` and a picker selection.

`ora setup --env` (`ora/app/gpu_setup.py`) prepares the inputs: a step is done when
its real state holds (a clean checkout at the pinned revision) or its marker
records the same recipe digest; package markers also bind the target
virtualenv's `pyvenv.cfg`, so a recreated environment is reinstalled. RoboDojo's
`graphics-compat` is done when its receipt exists, and a failed readback
withdraws the receipt. When `libGLU.so.1` or `libvulkan.so.1` is not loadable,
the `system-libraries` step unpacks `libglu1-mesa` / `libvulkan1` (`apt-get
download`, `dpkg -x`) into `<prefix>/syslibs`; `ora setup` and `ora run` (from
`ORA_GPU_PREFIX`) prepend its library directory to `LD_LIBRARY_PATH`, which
the isolated workers inherit. Each command runs in its own
process group, killed as a group on timeout.

## RoboDojo physics device

`ora.environment.benchmarks.robodojo_physics` maps each reviewed task to its physics device. Articulation, conveyor and fluid scenes in its CPU table use CPU physics; other reviewed scenes use `cuda:0`. Rendering still has its own bound GPU. Composers, scene probes and qualification must use the same task/device/layout mapping; a rigid CUDA scene's qualification does not establish CPU-scene correctness.
