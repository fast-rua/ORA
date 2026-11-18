# Context packs

A context pack changes what a model receives on each request without changing
what the robot does. It is a directory owned outside ORA, usually by a project
that trains a model on a compact context and needs the requests at inference
to match its training samples byte for byte. ORA provides the mechanisms; the
pack provides the words and parameters:

| Mechanism | What the pack supplies |
| --- | --- |
| Compact system message: the pack's core text plus named sections of the run's compiled embodiment card. It replaces the harness system message and drops the project-instructions message | The core text file and the section names |
| Bare episode task: the task sentence followed by the episode's public completion contract, without the template's prose | Only the choice `task_and_contract_v1` |
| Tool-description overlay, looked up by the hash of the original function | The overlay file |
| Withheld tools | Their names |
| Finish semantics: a `finish_episode` success claim is decided by the official verdict instead of being refused for want of `check_task` | Only the choice `official_verdict` |
| Compact history view (`ora.compact-view.v1`) | Every parameter and text |
| Switches for the Context layers that act on token usage | Which layers to turn off |

The tools, the Supervisor, actions, observation and feedback formats, round
static facts, budgets and the official verdict are not changed. The first
consumer is a compact profile for a small open model; any project can define its
own pack.

## Selecting a pack

A RunSpec names the pack by location and manifest hash under
`policy.context_pack`:

```yaml
result_tier: zero_shot_no_check        # required when the pack withholds check_task
policy:
  kind: native_api
  context_pack:
    path: /absolute/path/to/pack        # or path_env: NAME_OF_AN_ENVIRONMENT_VARIABLE
    manifest_sha256: <sha256 of the pack's manifest.json>
    profile: {...}                      # only for a manifest without ora_profile
```

Exactly one of `path` (absolute) and `path_env` (a variable holding the
absolute path) names the directory; no pack location is written into the
repository. To start from a task's usual RunSpec, compose it with
`./scripts/ora run --env ... --task ... --prepare-only` (it prints the path),
add the field and the tier, check it with `./scripts/ora plan show RUNSPEC` and
launch it with `./scripts/ora launch RUNSPEC`. The pack applies on top of the default composition chosen for the
environment, so a pack plan differs from the default plan only by the pack.
A plan without the field carries nothing new, so the default delivery is
unchanged byte for byte.

Resolution (`ora/orchestrator/spec/context_pack.py`) refuses the field when:

- the manifest's sha256 or any listed file's sha256 differs, or a file is
  missing;
- the profile is invalid, or both the manifest and the RunSpec state one;
- the plan names another agent core than Pi (only the Pi core delivers packs);
- the profile withholds a mutation or lifecycle tool;
- the overlay has no version of an advertised tool;
- the profile needs a compiled embodiment card or a typed slot composition
  that the plan does not have;
- the plan is free play or uses a scripted policy.

Withholding `check_task` makes the plan a `zero_shot_no_check` plan, which the
RunSpec must declare ([result tiers](../evaluation/protocol.md#result-tiers)).

## The plan binding

The resolved plan records `policy.context_pack` (schema
`ora.context-pack-binding.v1`):

| Field | Content |
| --- | --- |
| `name` | The manifest's `name` |
| `manifest` | `{path, sha256}` of `manifest.json` |
| `path_env` | The variable that named the directory, or `null` |
| `files` | Every listed file and its sha256 |
| `profile_source` | `manifest` or `run_spec` |
| `profile` | The validated delivery profile |
| `system` | `{text, embodiment_card, embodiment_sections_kept}`: the composed system text and the card layer it came from |
| `episode_task` | `{template, contract_layers}`: `{{TASK}}` followed by the contract layers |
| `delivered_tools_sha256` | Canonical sha256 of the overlaid tools |

The tool overlay also appears as a derived layer of the advertised ToolSurface
(`context_pack_tool_overlay_v1`, with the source and delivered tool hashes).
The plan hash covers the binding. `model_visible_sha256` changes when the
pack's files, profile or composed texts change, and does not change when only
the directory path changes.

Before a session starts, `ora/orchestrator/episode/policy_factory.py` loads the bound
pack again and refuses to start if any file changed after planning, or if the
plan names another agent core than Pi.

## Manifest

```json
{
  "name": "example-compact-v1",
  "files": {"CORE.md": "<sha256>", "tool-overlay.json": "<sha256>"},
  "ora_profile": {...}
}
```

`name` and `files` are required. File names are relative paths inside the
pack. ORA reads only `name`, `files` and `ora_profile`; other keys are the
owner's notes. A manifest without `ora_profile` takes the profile from the
RunSpec. Once a manifest states one, the RunSpec cannot override it, so one
manifest hash determines the whole delivery.

## Delivery profile (`ora.context-pack-profile.v1`)

Each part is optional. An absent part leaves that part of the request as it was.

| Field | Type | Values | Default |
| --- | --- | --- | --- |
| `schema` | string | `ora.context-pack-profile.v1` | required |
| `system.core_file` | string | a file listed in the manifest | required with `system` |
| `system.embodiment_sections` | list of strings | `##` headings of the compiled embodiment card | required with `system` |
| `episode_task` | string | `template`, `task_and_contract_v1` | `template` |
| `tools.overlay_file` | string | a file listed in the manifest | no overlay |
| `tools.drop` | list of strings | tool names to withhold | `[]` |
| `finish.success` | string | `task_check`, `official_verdict` | `task_check` |
| `compact_view` | object | `ora.compact-view.v1`, below | full history |
| `context_management.pressure_externalization` | boolean | `false` turns off L4 | `true` |
| `context_management.rollout_compaction` | boolean | `false` turns off L5 and L6 | `true` |

**System message.** The text is the core file's text (stripped), a blank line,
`# Embodiment: <card title>`, a blank line, then the named `##` sections of the
plan's compiled embodiment card, verbatim, in the card's order and separated by
blank lines. The card is the last project-instructions layer. Its title is the
`# <Name> compiled embodiment vN` line. The project-instructions message is
not delivered.

**Episode task.** `task_and_contract_v1` delivers the task sentence followed by
each completion-contract layer of the episode, joined by blank lines, without
the episode-task template's prose.

**Tool overlay.** The file is a JSON list of `{"key", "spec"}` entries. `key` is
the first 12 hex digits of `sha256(json.dumps(function, sort_keys=True))`,
computed over the original function object the Provider would receive (`name`,
`description`, `parameters`). `spec` is the function object delivered in its
place. An entry with `"passed": false` is ignored. The overlay may change only
string `description` fields; names, types, enums, required fields, ranges and
defaults must be the ToolSurface's own. The runtime therefore validates
arguments exactly as before. If an advertised tool has no entry, the plan is
refused: a pack must be regenerated when its ToolSurface changes.

**Finish semantics.** `official_verdict` binds the ToolSurface capability
`finish_success_official_verdict_v1` into the runtime surface:

- `finish_episode(outcome="success")` is admitted without a same-boundary
  `check_task=true`, and the environment's official verdict alone decides
  success;
- the latched rule (only success after `check_task` confirmed it) is off, so
  no refusal reveals the hidden task-check record;
- a refused success reason lists its evidence terms without `check_task`.

The other outcome, reason and category rules stay. The official verdict never
enters anything the policy can see.

**Context management.** Lossless retention (`action_step_lossless_v4`) shapes
the projected history in layers. Some follow the history's own steps:
read-only results externalized once they age, oversized results, and media
outside the recent tail (L3). These run the same way on any model and also
shaped the requests that training data is built from. Two follow token usage
instead, so they depend on the model and the provider:

- L4 pressure externalization fires on inline-image counts, estimated tokens or
  provider-reported input tokens;
- L5 episode capsules and L6 emergency checkpoints call the rollout compactor
  when the provider-reported input crosses its watermarks.

`pressure_externalization: false` turns L4 off. `rollout_compaction: false`
turns L5 and L6 off, so the compactor is never called and each projection's
admission records give the reason `rollout_compaction_disabled`. In the Pi
core, where v6 and v8 run as context stages (`ora-pi/src/context/stages/`),
each switch is the `enabled` parameter of its stages (`pressure_externalize`;
`episode_capsule` and `emergency_checkpoint`): the stages stay in the scheme,
render as before and keep their manifest fields, and are never admitted. The plan's own
`policy.context_management` keeps its values; the profile switches are part of
the binding and of `model_visible_sha256`. A switch set to `false` needs
`action_step_lossless_v4` (refused otherwise, never ignored). Every
`context_projected` record of the pack then names the effective switches
(`context_pack.context_management`).

With both switches off, nothing shrinks the projected history. Only the compact
view bounds what is sent: the prefix, one fold line per older turn, the recent
turns with capped tool text, and at most `image_window` downscaled images. The
request therefore grows by about one line per turn. The other invariants do not
depend on L4–L6: the exact causal tail, the lossless semantics check, the
monotonic watermarks (they stay unset) and the lossless evaluation all hold.

A window that fills up ends the episode cleanly. Once the provider reports more
input tokens than the hard limit (`context_window_tokens - max_output_tokens -
2048`), the next turn ends with `context_exhausted`, with no summary written in
the policy's place. Under a v6 object with the hard backstop (`context_lossless_v6`
since 2026-11-14) a pack does not turn the backstop off: a request estimated
at or above L6's watermark (0.8 of the hard limit) first has all but its latest
2 exchanges aged, and ends the turn with `context_exhausted` before it is sent
only if it still exceeds the hard limit. The margin of `max_output_tokens + 2048` tokens covers the
growth from one request to the next. A request the server still rejects
(HTTP 400 for exceeding its maximum model length) is a provider failure of the
run. Give the provider profile the context window the server actually serves
(vLLM's `--max-model-len`); a small-window model is better served by smaller
`recent_turns`, `image_window` and caps than by compaction it was never trained
on.

## Compact history view (`ora.compact-view.v1`)

The Pi core applies the view to each projected request
(`ora-pi/src/context/compact_view.ts`):

1. The prefix, meaning every message before the first assistant message, is
   kept verbatim after the static replacements. It holds the system message,
   the task, the initial observation and the round static facts.
2. A turn is one assistant message and the tool and user messages after it,
   up to the next assistant message.
3. The last `recent_turns` turns are kept with three changes: the assistant
   text is stripped, any provider continuation state is dropped (the history
   carries no reasoning), and tool text longer than `tool_text_cap_chars`
   code points is cut and followed by `texts.tool_text_omission`, where
   `{omitted}` is the number of characters cut.
4. Every older turn becomes one line of a single user message that starts with
   `texts.earlier_steps_header` and a line break. The line format
   `turn_summary_v1` is:
   `{n}. {text} -> {tool}({arguments}) -> {status}; ...; effect: {summary}`.
   - `text` is the assistant's stripped text, cut to `rationale_cap_chars`
     with `…`. When the text is empty, it is left out together with its arrow.
   - `arguments` is compact JSON (`separators=(",", ":")`, non-ASCII kept),
     cut to `arguments_cap_chars` with `…`.
   - `status` is the call's result: `error {code or type}`,
     `task_success {value}`, `status {value}`, `success {value}` or `ok` for a
     JSON object; `returned a value` for other JSON; `returned text`
     otherwise; `no result` when the call has none.
   - A turn without calls shows `(no tool call)`.
   - `effect` is the last `last_action_effect.action_summary` among the turn's
     user messages.

   Values are written as Python's `str()` writes them.
5. Only the newest `image_window` images of the whole request are kept, prefix
   included. Each earlier image becomes the text `texts.image_placeholder`,
   whose `{label}` is the first 100 code points of the text part before it in
   the same message. A `<recoverable_artifact>` reference never serves as a
   label. A kept image whose long side exceeds `image_long_edge_px` is
   downscaled by Python: Pillow bicubic, each side `round(side * scale)`, RGB,
   PNG with Pillow's defaults. A smaller image is delivered unchanged.

The view never keeps an older turn because a later call might need it: it does
not know the next call.

| Field | Type | Values | Default |
| --- | --- | --- | --- |
| `schema` | string | `ora.compact-view.v1` | required |
| `recent_turns` | integer | >= 0 (0 folds every turn) | required |
| `tool_text_cap_chars` | integer or null | >= 1; null: no cap | required |
| `image_window` | integer or null | >= 0; null: every image | required |
| `image_long_edge_px` | integer or null | >= 1; null: no downscaling | required |
| `fold.line_format` | string | `turn_summary_v1` | required |
| `fold.rationale_cap_chars` | integer or null | >= 0; null: whole text | required |
| `fold.arguments_cap_chars` | integer or null | >= 0; null: whole arguments | required |
| `fold.latest_result_tools` | list of strings | reserved: must be empty in v1 | `[]` |
| `texts.earlier_steps_header` | string | non-empty | required |
| `texts.image_placeholder` | string | contains `{label}` once | required |
| `texts.tool_text_omission` | string | contains `{omitted}` once | required |

`fold.latest_result_tools` is reserved for a later rule that keeps the latest
result text of named tools. A non-empty list is refused, never ignored.

## Delivery and evidence

The history ORA keeps and projects is unchanged. The Context pipeline runs as
for any plan: composition, lossless retention, artifact references, round
static facts and compaction triggers. The pack's view is the last step before
request assembly (`ora-pi/src/context/context_pack.ts`), so every retention
invariant and the lossless evaluation still hold.

- **Replacements.** The Python half of the Pi core
  (`ora/agent/pi/context_pack_rpc.py`) answers two RPCs:
  - `context_pack.delivery` names each static message to replace by the sha256
    of its exact text and gives its replacement (`null` drops it), along with
    the overlaid tools, checked against `delivered_tools_sha256`. Each target
    must occur exactly once in the request.
  - `context_pack.images` downscales the kept images.
- **Record.** Every `context_projected` event carries a `context_pack` record:
  replaced and dropped messages, folded and kept turns, kept, omitted and
  resized images, the delivered tools' hash, and the context-management
  switches when the profile sets them. Because the view rewrites the
  delivered history on every request, the request assembler records such a
  request as `request_relation: context_pack_view` instead of refusing it as a
  cache-prefix rewrite.
- **Wire request.** The stored wire request is exactly what the model
  received.

## Serving a model trained on a pack

A pack trained on Qwen3.5-style chat templates with thinking disabled and XML
tool calls needs an OpenAI-compatible server that parses those calls. ORA reads
them as standard `tool_calls` (streamed by index, `finish_reason: tool_calls`).
With vLLM:

```bash
vllm serve Qwen/Qwen3.5-4B --served-model-name student-4b --host 127.0.0.1 --port 30000 \
    --max-model-len 262144 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

`qwen3_coder` parses the `<tool_call><function=...><parameter=...>` format. The
template switch travels on every request as the profile's request options:

```bash
./scripts/ora providers add student-4b --protocol openai_compatible \
    --endpoint http://127.0.0.1:30000/v1 --model student-4b --images \
    --context-window 262144 --max-output 4096 \
    --request-options '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Pass `--provider-profile student-4b` when composing the RunSpec with
`ora run`.

The reasoning effort stays `provider_default`, so no `reasoning_effort` field is
sent. See [model APIs](../getting-started/model-apis.md) for the rest of a
provider profile.

## Verification

- `tests/agent_core/test_context_pack_view.py` checks three things on CI:
  - the Pi core's view reproduces, byte for byte, the views an independent
    reference implementation produced for the synthetic pack and recorded
    history in `tests/data/context_pack/`;
  - a Synthetic Pi episode with a pack delivers it and evaluates valid;
  - a pack plan naming the retired Native core is refused.
- `tests/agent_core/test_context_pack_acceptance.py` is opt-in. It replays a
  pack owner's reference examples (sealed runs' recorded requests, the
  rationale of each turn, the expected messages and tools) through the same
  view. Set `ORA_TEST_CONTEXT_PACK_DIR`, `ORA_TEST_CONTEXT_PACK_MANIFEST`,
  `ORA_TEST_CONTEXT_PACK_EXAMPLES` and, for a manifest without a profile,
  `ORA_TEST_CONTEXT_PACK_PROFILE`. Examples whose reference view depended on
  the example's own next call are skipped and counted.
- `tests/ora/orchestrator/spec/test_context_pack.py` covers loading, profiles,
  composition, plans, the session RPC and finish semantics.

Packs have no agent-core goldens: those were recorded from the retired Native
core, which never delivered packs.

## Limits

- Packs are checked for single-round episodes. In a multi-round session, a
  later round's static facts appear mid-history and fold with their turn once
  that turn ages out.
- With the context-management switches left on, the rollout compactor still
  sees the full history and calls the policy's provider when its triggers
  fire. Pressure externalization then depends on the compact request's usage,
  so it can differ from the requests that training data was built from. Turn
  both off for a pack whose model must see exactly its training view.
- `ora plan context` shows the composition before the pack's replacements.
