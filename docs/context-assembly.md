# Native context assembly

The Native policy owns a canonical message history and translates it at the provider boundary. ORA supplies project instructions, the task, robot manual, advertised tools, current public state and selected tool history. No managed coding-agent scaffolding is part of this distribution.

## Composition

Policy layers are source files bound by path and hash in the RunPlan. The typed v3 context channels are `harness_system`, `project_instructions`, `episode_task` and `tool_manual`. The retained legacy context channels are `codex_project`, `episode_prefix` and `tool_manual`; the first name remains a serialization field and does not select a Codex adapter.

The default robot compositions use the versioned Native causal slots. `native_slots_v4` adds protected round static facts to v3. The active baseline catalog determines the prompt bundle, ToolSurface, embodiment and context retention for each environment. Resolution validates their compatibility together.

## History and artifacts

Context projection retains causal action/observation relationships. Large tool results and media can be stored as recoverable context artifacts, with content hashes and bounded readers. The lossless retention policy distinguishes recoverable evidence from what is presently inlined into a request. Compaction and rollover are recorded, including the exact delivered request and usage evidence.

The rollout compactor that writes episode and emergency checkpoints is a separate model call and never receives an official outcome. It calls the policy's provider and model unless `policy.context_compactor` binds it to its own Provider profile ([context compactor](reference/configuration-and-evidence.md#context-compactor)); a binding changes the endpoint, model, effort and stream of its calls, never its packet, and its events then record `provider_binding`. Its packet is the policy-visible history with the task checker's verdict withheld in the two places the policy saw it: a `check_task` result is replaced by `{"status": "task_check_result_withheld_from_compactor", "reason": "official_outcome_reserved"}`, and a `finish_episode` refusal coded `task_success_not_confirmed` or `task_success_latched` keeps its other fields with `error.constraints.task_success` replaced by `verdict_withheld_from_compactor: "official_outcome_reserved"`. The policy's own context keeps both results as delivered. Any other reserved field in the packet (`task_success`, `official_success`, `official_verdict`, object ground truth and the like) still rejects the compaction (`rollout_compaction_private_evidence`). The Pi core applies the rule (`compactorHistoryMessage`) with the refusal codes Python sends it (`ora/agent/context/lossless/compactor_privacy.py`); the Python statement of the rule that the tests check against is `tests/support/agent/compactor_verdicts.py`.

The context budget must leave room for model output. Settings that alter projection, protected facts, thresholds or tool delivery change model-visible behavior and require a deliberate comparison. Do not silently summarize away a refusal's constraints or an action's units.

## Code and verification

- Composition: `ora/agent/composition/`, `ora/orchestrator/spec/composition.py`.
- Projection and request assembly: `ora/agent/context/` (the shared projection rules), `ora-pi/src/context/` (the pipeline, `native.ts`; the stage driver and stages, `stages/`; and `request_assembly.ts`).
- Artifact catalog and readers: `ora/agent/memory/catalog.py`, `markers.py`, `tool_runtime.py`, `artifact_reader.py`.
- Delivery trace: `ora/agent/evidence/agent_events.py`, `trace_verify.py`, `ora/orchestrator/evaluation/context_manifest.py`.

Use `ora plan context RUN_SPEC` to inspect the resolved context without launching. `scripts/dev/compare-model-visible BASE CANDIDATE` compares advertised tools, prompt text, embodiment facts, observation projection and settings. A package/schema migration can change identities without changing task behavior; report those changes explicitly rather than calling the raw digests equal.
