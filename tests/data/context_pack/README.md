# Synthetic context pack

A small context pack (`docs/reference/context-packs.md`) and one recorded
history, for `tests/agent_core/test_context_pack_view.py` and
`tests/ora/orchestrator/spec/test_context_pack.py`. Everything here is made up:
the core text, the embodiment card, the three tools and their overlay.

- `synthetic-pack/`: `manifest.json` (its own `ora_profile`), `CORE.md` and
  `tool-overlay.json` (three overlay entries, and one with `passed: false` that
  the loader skips). The tests pin the manifest's sha256.
- `synthetic-replay.json`: four requests of one short history as the Context
  pipeline projected them (the harness system message, project instructions
  holding a compiled embodiment card, the task with the template prose and a
  completion contract, observations with images, five turns covering every fold
  status), the tools the plan advertised, and, for each request, the messages
  and tools an independent reference implementation of `ora.compact-view.v1`
  produced with this pack's parameters. Images are compared by the sha256 of
  their delivered bytes, which includes a downscaled one.

Regenerate the expected views only together with a reviewed change to the view
contract; never to make a failing comparison pass.
