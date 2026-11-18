# ora/agent

## Responsibility

The policy side of a run. The agent loop of the default `native_api` policy runs
in the TypeScript Pi core (`ora-pi/`); this package is its Python half, a
scripted policy for tests and the quickstart, and everything a policy sees or
calls: the tool runtimes, ToolSurface / prompt / embodiment composition, the
context projection rules, recoverable Context artifacts, the provider adapters
preflight and the compactor use, the resident model-service clients, and the
agent trace's evidence formats.

| Unit | What it holds |
|---|---|
| `contracts.py`, `error_classes.py` | PolicySession contracts; the `ToolError` envelope, its argument rules and which tool-error codes are infrastructure failures |
| `evidence/` | Agent trace and artifacts (`canonical`, `agent_events`, `trace_verify`, `trace_privacy`, `sft_export`, `wire_export`, `token_ids`, `host_paths`, `openai_package`, `openai_package_tool`, `trace_dataset`, `trace_dataset_tool`), image and receipt binding checks, public controller, motion and plan feedback, policy-state snapshots, the free-play trace, the provider-request archive, the sandbox retention (`save_sandbox: false`: hashes only) |
| `sessions/` | The delegating PolicySession facade and the scripted adapter |
| `providers/` | OpenAI-compatible and Anthropic Messages adapters (`openai_compatible`, `anthropic_*`), `fake`, `factory.build_provider`, profiles, credentials, recovery |
| `model_services/` | SAM3 and Contact-GraspNet resident services, their loopback clients (`clients.py`, which the perception tools and preflight share), endpoint binding, identity and lifecycle (`operator.py`, behind `ora services`) |
| `composition/` | ToolSurface resolution, capability binders, embodiment cards, prompt sources, task contracts ([README](composition/README.md)) |
| `tools/` | Tool runtimes and the ten tool families ([README](tools/README.md)) |
| `memory/` | Recoverable Context artifacts: markers, `artifact_read`, the session catalog |
| `code_mode/` | Code mode (`policy.code_mode`): its contract, the code tools, the bubblewrap sandbox and the `robot` library, the persistent workspace ([README](code_mode/README.md)) |
| `context/` | What an observation and a tool result deliver to the policy ([README](context/README.md)) |
| `session_core/`, `scripted.py` | The Python half of a `native_api` session (tool gateway, projections, receipts); the scripted policy |
| `pi/`, `tui/` | The Pi core's process, RPC and robot socket; the `ora watch` transcript |

## What belongs here, and what does not

Here: code that decides or shapes what the policy sees and calls, and the
formats of what it leaves behind. Not here: simulator code (it runs in the
isolated environment worker, `ora/environment/backends/`), run orchestration
(plans, preflight, leases, evaluation, sealing: `ora/orchestrator/`), command-line
entry points, and policy data files (`ora/policy/`, locked by `policy.lock`).

## Dependency direction

`ora.agent` imports `ora.contracts` and the policy data, never
`ora.environment.backends`, `ora.orchestrator` or the CLI (rule L5). Inside the
package each unit imports only units below it (rule L6):

```text
pi, tui > session_core, scripted > context > memory, code_mode > tools > toolkit/host
  > composition, toolkit > providers, model_services > evidence, sessions
  > contracts, error_classes
```

`tests/repository/test_structure.py` checks both rules, and rule L7 inside
`tools/`.

## Entry points

- `pi.factory.build_pi_policy_session`: the default policy session.
- `scripted.py`: the scripted policy of the Synthetic quickstart and tests.
- `tools.build_tool_runtime`: the ToolRuntime a resolved ToolSurface names.
- `composition.resolve_tool_surface`: a profile's resolved ToolSurface.
- `providers.build_provider`: a Python provider for a profile's protocol.
- `evidence.trace_verify.verify_agent_trace`, `evidence.sft_export.export_sft`:
  check and export a sealed agent trace (`evidence/trajectory.py` re-exports
  them for the orchestrator); `ora trace-validate` and `ora export-sft` run them.
  `evidence.wire_export.export_wire` / `check_wire_export`: every call as sent,
  each image once, rebuilt against the archive (`ora export-sft --view wire`),
  with each call's token ids from `evidence.token_ids` (a stream that carries
  them, or an engine's tokenize endpoint), and with `evidence.host_paths` the
  host paths in it redacted and recorded (`--redact-host-paths`).
  `evidence.openai_package.export_openai` packages it as standard OpenAI
  conversations with a standard-library verify tool
  (`evidence.openai_package_tool`, shipped as `ora_package.py`).
  `evidence.trace_dataset.export_dataset` exports many runs as one dataset
  (`ora export-traces`): one record per episode with its context segments in
  order, each image once across the dataset, an index, and a standard-library
  reader (`evidence.trace_dataset_tool`, shipped as `ora_traces.py`).
- `context.observation.contact_points.project_policy_public_facts`: the
  public-facts projection the orchestrator injects into the tool runtime
  (`build_tool_runtime(..., public_facts_projector=...)`), so `tools/` never
  imports `context/`.
- Process entries: `python -m ora.agent.model_services.grounding_grasp` (the
  resident services, started by `ora services start`) and `python -m
  ora.agent.providers.recovery_worker`; [ora/app/README.md](../app/README.md#entry-points)
  lists them all.

## Documentation

- [Architecture](../../docs/architecture.md) and [code map](../../docs/code-map.md)
- [Context assembly](../../docs/context-assembly.md) and
  [policy I/O](../../docs/reference/policy-io.md)
- [Tool-surface authoring](../../docs/developer/authoring/tool-surface.md)
- [The agent core](../../ora-pi/README.md), the TypeScript half
- [Repository structure](../../docs/developer/repository-structure.md), rules L5-L7

The Pi bridge accepts synchronous toolkit execution metadata only from the Supervisor. Its RPC cache keeps that evidence even when the tool raises; the core persists it as hidden `_execution` history metadata and returns an `ExecutedActions` outcome.
