# Context schemes and policy selection

A baseline fixes a tool surface, a prompt bundle, an action codec, a runtime
template and a context scheme as one unit. A **context scheme** is the context
half on its own, under a name, so a RunSpec can keep its baseline (or the one
its tool surface brings) and choose its context separately.

## Schemes

`ora/policy/context_schemes/<name>.json` (`ora.context-scheme.v1` or
`ora.context-scheme.v2`), pinned by `policy.lock`. There is no registry:
`context_schemes` (`ora/agent/composition/context_scheme.py`) discovers every
`*.json` file of the directory and knows it by the `id` inside it. A file that
is not an `ora.context-scheme.v1` or `v2` object with a string id is refused,
and so is a second file with an id already seen; an unknown id is refused with
the list of known ones. Adding a scheme is adding its file (and running
`./scripts/ora lock`).

### Scheme format v2: a scheme lists its stages

An `ora.context-scheme.v2` scheme lists the stages the agent core runs
(`ora-pi/src/context/stages/`) instead of a management object. Its `policy`
holds the other context fields (profile, composition, retention, summary
policy, memory, continuation replay, initial observation, context tools), and
beside it:

| Key | What it holds |
|---|---|
| `segments` | `unit` (`action_step`, or `exchange` for v9's stages, which age by exchange), `exact_tail`, what the summaries keep exact, and optionally `exact_tail_unit`: absent or `action_step` (completed ActionSteps), or `exchange` (the latest exchanges plus the latest robot action's exchange, however old; with `emergency_checkpoint.unit: exchange`) |
| `limits` | `reserve_tokens`: hard input limit = window − max output − reserve |
| `estimator` | `image_tokens`: what the request estimate charges per inline image |
| `references` | `pointers` (whether recoverable-artifact Pointers are shown), `advertise_over_bytes`, `preview.head_bytes` |
| `stages` | ordered `{stage, params}` entries |

Each stage's parameters have one closed schema,
`ora/policy/context_stages/<stage>.schema.json`, and the four settings
`scheme.schema.json`. Python validates a scheme with them, filling the
defaults they declare (`ora/agent/composition/schemas/context_stages.py`),
and sends them to the core at session start (`domain.context_stage_schemas`,
`ora/agent/pi/domain.py`); the core validates every stage it builds with the
same files and keeps no rules of its own.

A v2 scheme resolves to two plan fields: `policy.context_stages`
(`ora.context-stages.v1`, the settings and the complete stage list) and
`policy.context_management`, the management object the list is the stage list
of. The projection manifest, the evaluation and the plan's Context checks still
read that object, so a stage list resolves only when it has one (v6's, v8's,
v9's or v10's stages, with their values for the settings those versions fix;
v10's are v9's that state `exchange_aging.result_preview`, see
[keep-written context](keep-written-context.md)); the plan resolution
validates the object and checks the list is its stage list, and the core runs
the list after checking it equals its own translation of the object
(`policy.context_stages is not the stage list of policy.context_management`
otherwise). A context pack's switches turn the listed stages' `enabled` off as
they do the translated ones. The model-visible comparison reads
`context_management` and leaves `context_stages` out (`EXPANSION_KEYS`,
`ora/orchestrator/evaluation/model_visible.py`). Plans that carry only a
management object (v1 schemes, sealed plans) keep being translated by the core
as before.

`context_lossless_v6` is a v2 scheme since 2026-11-04: it resolved to its v1
file's management object byte for byte, its other policy fields are the v1
file's, and replaying the context goldens with the core running its stage list
reproduces every request, manifest and state.

### v6's hard backstop (2026-11-14)

Since 2026-11-14 `context_lossless_v6` (patched in place) runs the hard
backstop as its guard and sends a truncated reply without its reasoning: its
stage list ends with `hard_backstop` (`exact_tail_exchanges: 2`, `aging:
own`, `trigger_ratio` = L6's `emergency_fallback_ratio`, 0.8) instead of
`over_limit_guard` (`refuse`), and `materialize` has
`truncated_reply_reasoning: withheld`. Its management object is v6's plus two
optional fields, `backstop_exact_tail_exchanges: 2` and
`truncated_reply_reasoning: "withheld"` (`management_fields.py`
`CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS`); a v6 object without them is
the v6 every earlier plan carries and keeps running as it ran.

Why: v6 guarded the hard limit (window − max output − 2,048) only after the
Provider's report. Its L6 is admitted when the Provider reported the request
before at 0.8 of the hard limit (181,862 tokens on a 262,144 window with
32,768 output tokens) and summarizes the completed ActionSteps before a
3-ActionStep exact tail. On RoboLab tool-call runs (Qwen3.8-27B,
2026-11-14) that was too late and too narrow: a call adds the previous reply
to every later request, up to the whole 32,768-token output limit when the
reply was cut off before any tool call (38 such replies in six of the seven
runs), and inside one open ActionStep there is nothing L6 may absorb
(ea103b427290: 19 calls before its one robot action, L6 never admitted). Seven
runs sent requests their own estimate put 9,000–36,000 tokens over the limit
(236,040–263,710) and ended `context_exhausted` on the report, while the agent
was still working. The backstop estimates every request after every other
stage (the estimate of [exchange aging's backstop](exchange-aging-context.md#hard-backstop)),
so L6's watermark now holds inside an open ActionStep too: when the request's
own estimate reaches 0.8 of the hard limit and L6 cannot act on it (nothing
before its exact tail to absorb, or no Provider report coming), every exchange
but the latest 2 is aged in one lossless rewrite (transition `hard_backstop`,
reason `estimated_input_at_or_above_watermark`), the open ActionStep's and the
exact ActionStep tail's included: reasoning withheld, read-only tool results to the
scheme's 2,048-byte head preview and Pointer, tool-returned images to their
markers; observations stay whole. While L6 has ActionSteps to absorb and only
waits for the report of a request at its watermark, the backstop leaves the
request to L6 (aging it would keep every report under L6's watermark, so the
history before the exact tail would never be summarized), and within the
limit it rewrites only when at least 2 more exchanges can age. A request whose
latest 2 exchanges alone reach the watermark is sent as it is while it fits; one still over the hard
limit ends the turn with `context_exhausted` before it is sent ("... after the
hard backstop kept only the latest 2 exchanges exact ..."). L6 itself is
unchanged: it still summarizes completed ActionSteps on the Provider's report,
and runs first. Replayed through the context manager with the runs' recorded
histories, every request of the seven runs stays within the limit (see the
pull request for the per-run peaks). Besides the truncated replies and the
aging from the watermark on, L1–L6 keep their v6 rules.

| Id | Composition | Retention | Management | Named by |
|---|---|---|---|---|
| `context_lossless_v6` | `native_slots_v4` | `action_step_lossless_v4`, rollout checkpoints | `ora.native-context-management.v6` with the hard backstop (`backstop_exact_tail_exchanges: 2`) and truncated replies without their reasoning ([since 2026-11-14](#v6s-hard-backstop-2026-11-14)) | `native_full_explicit_motion_direct_v4` (the default baseline: LIBERO-Pro, RoboTwin, RoboCasa365, RoboLab) for tool calls, and `ora local` |
| `context_keep_written` | `native_slots_v4` | `action_step_lossless_v4`, rollout checkpoints | `ora.native-context-management.v10` ([keep what the model wrote](keep-written-context.md)): scripts, messages and reasoning never age (`reasoning_replay: all`), data does from 100,000 tokens (at most half the hard limit), L6 by exchange at 0.6 of the hard limit, prompt `keep-written.md`, `context_tools: ["artifact_read"]` | `robodojo_toolkit_v7` (RoboDojo's default baseline) and `robodojo_toolkit_v6`, both modes; `native_full_explicit_motion_direct_v4` for code mode (`code_mode_context_scheme`) |
| `context_keep_written_unaged_reasoning` | as `context_keep_written` | as `context_keep_written` | v10 with `reasoning_replay: unaged_exchanges` (arm H): an aged exchange's reasoning leaves with it | explicit `policy.context_scheme` (an option, the default on 2026-11-04 and 05; it has a retention entry) |

Retired on 2026-11-05: the v9 schemes `context_rolling_tail_v3`,
`_camera_tail`, `_recent_reasoning` and `_unaged_reasoning` (added on
2026-11-04 from the reference line `1a9c1b117`, never a default), the v8
scheme `context_rolling_tail_v2` (RoboDojo's default until 2026-11-04) and
`context_age_scripts` (arm B), with their prompts. Sealed runs keep their own
commits. Management v8 and v9 stay: the v10 schemes build on them and the
context goldens' v9 cases replay the reference runs with their own settings.
Their stage lists are test fixtures (`tests/data/context_scheme_fixtures`).

`context_keep_written_unaged_reasoning` became the default on 2026-11-04, after the context experiment's
30 runs per arm: it matched `context_keep_written` (arm A) on success, 12 to
12, with a lower peak input (median 205,000 against 291,000 tokens) and 22%
less input per run; `context_age_scripts` (arm B) saved little and was retired
the next day. On 2026-11-05 the user chose the experiment's best arm instead:
P, 15 of 30, which is `context_keep_written` with preview left optional, so `context_keep_written`
is the default and `context_keep_written_unaged_reasoning` an option ([keep what the model
wrote](keep-written-context.md)). Forced preview was retired the same day
(`policy.code_mode.preview` is refused, [public interfaces](public-interfaces.md)); the dry run stays optional.

### Code mode by default

A baseline may also name the code mode a RunSpec gets when it names no
`policy.code_mode` (`"code_mode"` in the catalog, a `policy.code_mode` value,
applied by `ora/orchestrator/spec/defaults.py` before the scheme is chosen).
`robodojo_toolkit_v7`, RoboDojo's default, names
`{"library": "extended", "prompt": "v3", "source": true}` since 2026-11-05 (the
decision from the code-mode comparison on 41 RoboDojo tasks and the context experiment): the extended library, the v3
planner with its optional `dry_run` layer (`code-mode-coding-v3.md`, then
`code-mode-dry-run-v1.md`), ORA's source under `/ora/src`, no forced preview,
and the scheme the baseline selects for code mode,
`context_keep_written`. A plan resolved from it equals, field
for field, the plan of a RunSpec that names that code mode itself, except the
baseline catalog's own sha256.

A RunSpec's own `policy.code_mode` wins. `"code_mode": null` (or `false`)
keeps tool calls, and resolves to the plan tool-call RunSpecs resolved to before
the default (no `code_mode` key). Free play gets no default (it has Pi's own
codemode), and neither does a plan with a context pack, which composes with tool
calls. `robodojo_toolkit_v6` (and its surface `robodojo_placement_toolkit_v5`)
names none, so its plans stay tool calls; naming `robodojo_placement_toolkit_v6`
selects `robodojo_toolkit_v7`, code mode included. The other backends' default,
`native_full_explicit_motion_direct_v4`, names none. The choice reads the
catalog alone, never the host: whether the host can run the sandbox is
preflight's `code_mode_sandbox` check ([public interfaces](public-interfaces.md),
code mode). `tests/data/default-expectations.json` (`code_mode`) records each
backend's default.

### Code mode's scheme

A baseline may name a second scheme for code mode, `code_mode_context_scheme`:
a code-mode policy's calls complete no ActionStep for long stretches, the case
L6 by exchange answers. A RunSpec that sets `policy.code_mode` and names no
`policy.context_scheme` (and no context pack, which composes with the
baseline's own) gets that scheme; `native_full_explicit_motion_direct_v4` names
`context_keep_written` there, so code mode on LIBERO-Pro, RoboTwin, RoboCasa365 and RoboLab
runs management v10 while tool calls keep v6. In code mode the scheme's prompt
is delivered after the code planner even on a bundle without a
`context_prompt` layer, and the round's static facts are then stated once, by
that prompt's last paragraph, instead of by their own layer after the code
core (`ora/orchestrator/spec/composition.py`: when the plan delivers its own
scheme's prompt). A dry run stays optional; forced preview (`policy.code_mode.preview`) was retired on 2026-11-05.
`tests/data/default-expectations.json` (`context_schemes`) records each
backend's scheme per mode.

Every baseline names its context half by id (`"context_scheme"` in the
catalog); none carries the fields inline. `context_lossless_v6` is the default
baseline's context written out as a scheme on 2026-11-03. Its file is byte for
byte the one retired under that id on 2026-11-01 (sha256 `be11a138…`), whose
policy was the default baseline's inline context, field for field; the first
version (2026-10-26) differed only in its `source_baseline` note. Plans resolved from the default baseline since
then gain the record `policy.context_scheme` (`{id, path, sha256}`) and keep
every other value; `ora local` writes the scheme's fields into its RunSpec,
which has its own tool surface and so takes no `policy.context_scheme`.

`context_rewrite_compact_v1`, `context_rolling_tail_v1` (the policy of
`context_rolling_tail_v2`, retired in turn on 2026-11-05, without its prompt) and
`context_append_until_compact_v1` were retired on 2026-10-31, and
`context_lossless_v7` (`native_slots_v3`, management v7, taken from
`robodojo_toolkit_v3`), with that baseline, on 2026-11-01. Sealed runs keep
their recorded definitions. The code of rewrite-compact retention (also the
context of a plan that names no retention, such as the Synthetic quickstart's)
and of the `inline_until_compaction_v1` camera mode stays, so those runs still
evaluate and replay.

The agent core runs context management v6, v8, v9 and v10 only (as context stages,
`ora-pi/src/context/stages/`). It refuses a plan with v4, v5 or v7 at session
start (`context_management ora.native-context-management.v7 is retired from
the agent core ...`); Python still validates them and evaluates the sealed runs
that used them. v4 and v5's behaviour stays testable as two stage parameters,
`stride_aging.strided` (unstrided aging) and `episode_capsule.trigger_ratio`
(an L5 watermark that scales with the window), which their recorded
differential fixtures run with.

Management v7 (the retired `context_lossless_v7`'s) and v8 set `image_estimate_tokens` to 530: the request estimate
charges that per inline image, where management v6 charges a fixed 4096. Fitting DeepSeek-V4.1-Flash
input usage on 4,186 recorded RoboDojo lossless requests against text bytes
(base64 image payloads removed) and the inline image count gave about 530
tokens per camera image and 3.5 text bytes per token (median error 1.8%); 4096
overstated a request by about 1.5x before usage calibration. Text keeps its
byte estimate.

A scheme's `policy` holds only `CONTEXT_FIELDS`: `context_profile`,
`context_composition`, `context_retention`, `context_token_estimator`,
`summary_policy`, `memory_policy`, `provider_continuation_replay`,
`initial_observation`, `context_management`, `context_tools` (a v2 scheme
names neither `context_management` nor `context_stages`: its stages resolve to
both). A consumer that
composes a policy from a baseline outside the RunSpec expansion (YAM free play)
reads the baseline's own fields plus its scheme's with `baseline_policy`
(`ora/agent/composition/baseline.py`). The v10 schemes are append-only context
([append-only context](append-only-context.md)) with exchange aging ([exchange
aging](exchange-aging-context.md)), and `context_keep_written` and its option keep what the
model wrote ([keep-written context](keep-written-context.md)).

## Context tools

`context_tools` lists the Context tools a scheme provides. The only one today
is `artifact_read`, the recoverable-artifact reader, which also gives
`observe` its `ref` mode. A policy without the field keeps the old rule:
lossless retention provides the reader, so `context_lossless_v6` advertises
it.
`[]` provides none; v8 management requires the field and cannot provide the
reader, since it shows no Pointer to read; v9 management requires it and must
provide the reader, since an aging leaves Pointers. The reader is therefore a property
of the context scheme a run used, recorded in the plan, not of its tool
surface. `observe(step=...)`, a saved observation of the current round, is the
tool surface's own mode and is unaffected.

## Selecting one

`policy.context_scheme: "<id>"` on a RunSpec that selects a baseline (by
`policy.baseline`, by a tool surface that has one, or the default). The scheme
replaces every context field of the baseline, so a field the scheme omits is
not inherited. A
RunSpec that also writes one of those fields must agree with the scheme. The
plan records `policy.context_scheme` as `{id, path, sha256}`, the scheme the
RunSpec named or else its baseline's; a RunSpec without the field gets its
baseline's scheme.

The record is not part of `policy_identity.model_visible_sha256`
(`PROVENANCE_KEYS` in `ora/orchestrator/evaluation/model_visible.py`): the
scheme's fields are written out in the plan's policy, its prompt is a prompt
layer and its context tools are advertised tools, and each of those is in the
digest; the scheme's name is not something the model reads. Naming the
baseline's own scheme changes nothing, and a plan of the default baseline kept
its digest when that baseline's inline context became `context_lossless_v6`. (RoboDojo plans, whose baselines named their scheme
before 2026-11-03, recorded a digest that included the scheme id; plans
resolved since then do not.) Another scheme can change what the model is shown beyond the
context: the recoverable-artifact tools (`artifact_read`, `observe(ref=...)`)
follow `context_tools`: the v10 schemes advertise them, `context_lossless_v6`
leaves them to its retention (and the retired v8 scheme did not).

A patch edits a scheme in place; a sealed run records its scheme's path and
sha256 and keeps what it read. A new id is for a changed context contract
([policy README](../../ora/policy/README.md), "Versions, patches and retirement").

## Composing a RoboDojo cell

`ora spec robodojo compose|spec TASK ...` takes the selection on its command line:

| Option | Writes |
|---|---|
| `--baseline ID` | `policy.baseline` |
| `--tool-surface ID` | `policy.tool_surface.profile` (a toolkit surface brings its baseline, `TOOLKIT_BASELINES`) |
| `--context-scheme ID` | `policy.context_scheme` |
| `--toolkit-grants JSON` | `toolkit_grants`; default: the `required_grants` of a file-backed surface |
| `--toolkit-receipt-delivery observation\|tool_output` | `toolkit_receipt_delivery`; it may only restate a file-backed surface's own delivery (`tool_output` on v5) |
| `--record-toolkit-io` | `record_toolkit_io: true` |
| `--ground-truth-capture` | `runtime.ground_truth_capture: true` |

For example, the toolkit v6 surface with `context_keep_written`, the option
that keeps every exchange's reasoning:

```bash
./scripts/ora spec robodojo compose stack_bowls --experiment my-study --gpu 0 \
  --tool-surface robodojo_placement_toolkit_v6 --context-scheme context_keep_written \
  --record-toolkit-io --ground-truth-capture
```

Resolution (`ora plan show|preflight`, `launch`) checks the combination like any
other RunSpec. `--service-report` binds the resident services after the
selection (`bind_services`; `ora task run` does the same), so every service
binding names the selected surface, and its capability scope is that of the
baseline the surface brings; launch preflight refuses a binding to any other
surface.
