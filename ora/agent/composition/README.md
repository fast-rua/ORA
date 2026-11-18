# ora/agent/composition

## Responsibility

Resolve what a run's policy is composed of, deterministically and with every
input hashed: the ToolSurface the policy is offered, the embodiment facts and
cards it reads, its prompt layers and context packs, and the reviewed task
contracts.

| Unit | What it holds |
|---|---|
| `surface/` | ToolSurface resolution: profile catalog, manifests and components, schema validation, canonical tool names, parameter normalization, and the shared vocabularies (units, view config, motion modes, the `compute` grammar, the `robot_runtime` definitions) |
| `schemas/` | The constants a capability's tool text shares with its runtime (orders, limits, claim texts, capability names) and the finish contract; the binders that wrote the text were retired with the layered resolver |
| `embodiment/` | Embodiment descriptors, jaw models, control and geometry cards, environment context |
| `prompt/` | Prompt bundles, the hash-bound policy layers, context packs, context slots |
| `task_contract/` | Public completion contracts (generic and RoboDojo) and their layout compatibility |
| `baseline.py`, `context_scheme.py`, `ora_local.py`, `locked_assets.py`, `inspection.py` | The default baseline catalog, the context schemes (discovered from `ora/policy/context_schemes/`) that baselines and `ora local` name and a RunSpec may select instead of its baseline's, `ora local`'s composition, path-to-bytes asset resolution, read-only inspection of a resolved configuration |

## What belongs here, and what does not

Here: host-safe reading, validation, resolution and hashing of policy assets.
The assets themselves are data under `ora/policy/`, read by path and sha256 and
locked by `policy.lock`. Not here: running a tool (`tools/`), delivering a
result or an observation to the model (`context/`), or expanding a RunSpec into a
RunPlan (`ora/orchestrator/`).

## Dependency direction

Composition sits below `tools`, `memory` and `context` and may import only
`providers`, `model_services`, `evidence`, `sessions`, `contracts` and
`error_classes` (rule L6). `__init__.py` is a lazy facade: importing one module does not load
the others.

## Entry points

- `resolve_tool_surface(profile_id)` and `PROFILE_CATALOG`
  (`surface/tool_surface.py`, `surface/profile_catalog.py`).
- `load_prompt_bundle`, `materialize_policy` (`prompt/`).
- `baseline.load_native_baseline`: registered baseline compositions. Each RoboDojo
  toolkit surface brings its own baseline (`TOOLKIT_BASELINES` in
  `surface/toolkit_profiles.py`, which also lists the first toolkit tool cohort;
  see [toolkit surfaces](../../../docs/reference/toolkit-surface.md)).
- `inspect_run_configuration` (`inspection.py`), behind `ora plan context`.

## Documentation

- [Tool-surface authoring](../../../docs/developer/authoring/tool-surface.md),
  [prompt-bundle authoring](../../../docs/developer/authoring/prompt-bundle.md)
  and [task contracts](../../../docs/developer/authoring/task-contract.md)
- [Context packs](../../../docs/reference/context-packs.md) and
  [context assembly](../../../docs/context-assembly.md)
- [Policy assets](../../policy/README.md), the data this package resolves

The provider-tape baseline `native_full_explicit_motion_direct_replay_v1` was retired
on 2026-11-01, and the legacy-visible toolkit baseline and its tape twin on
2026-10-31; [provider tape replay](../../../docs/reference/provider-replay.md) now
only serves and compares recorded runs.
