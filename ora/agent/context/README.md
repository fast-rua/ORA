# ora/agent/context

## Responsibility

What the policy is shown: one observation's projection, the capability-trimmed
delivery of tool results and facts, and the rules for what a tool result may
carry into a provider request. The Pi core assembles the request and runs the
context management pipeline (`ora-pi/src/context/`); the rules and constants
Python owns, and sends to it at session start (`ora/agent/pi/domain.py`), are
here. Python statements of the pipeline that only tests check against (request
sizing, inline media, historical observations, the rollout compactor call, the
compactor's verdict masking, the L4 share) are not: they are in
`tests/support/agent/`.

| Unit | What it holds |
|---|---|
| `observation/` | One observation: current and post-action views, public facts, contact points, calibration, skill views, the projector |
| `delivery/` | Capability-trimmed delivery: the trims, de-duplication, round static facts, tool-result delivery |
| `lossless/` | Context management (v4-v8) as Python owns it: the management schemas and their fields (`management_fields.py`) and their validation (`management.py`), the L5 and L6 triggers sealing reads (`thresholds.py`), the compactor's system prompt (`compactor.py`) and the finish refusals whose verdict it never sees (`compactor_privacy.py`) |
| `causal_value.py` | What a tool result may carry into the provider request, and the privacy rules Python owns |

## What belongs here, and what does not

Here: projection and delivery rules, and their schemas. Not here: running a
tool (`tools/`), storing a recoverable artifact (`memory/`), the agent loop,
request assembly and the context pipeline (the Pi core), test-only mirrors of
that pipeline (`tests/support/agent/`), or the slot vocabulary the policy
layers produce (`composition/prompt/slots.py`; the facade re-exports it).

## Dependency direction

Context may import the units below it (`memory`, `tools`, `composition`,
`providers`, `model_services`, `evidence`, `contracts`, `error_classes`), never
`session_core`, `pi` or `tui` (rule L6). `__init__.py` is a lazy facade.

## Entry points

- `build_context_policy`, `ContextProjector` (`observation/projector.py`).
- `observation.contact_points.project_policy_public_facts`: the public-facts
  projection the orchestrator injects into the tool runtime.
- `causal_provider_value` (`causal_value.py`).
- `validate_context_management` (`lossless/management.py`), used when a plan is resolved.

## Documentation

- [Context assembly](../../../docs/context-assembly.md) and
  [context packs](../../../docs/reference/context-packs.md)
- [Policy I/O](../../../docs/reference/policy-io.md)
- [The agent core's context pipeline](../../../ora-pi/README.md) (`ora-pi/src/context/`)
