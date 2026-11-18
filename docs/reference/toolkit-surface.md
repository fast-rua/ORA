# File-backed toolkit surfaces

Every ToolSurface is plain files. Each tool is one directory holding the code
it runs and what the model is shown for it, written exactly as delivered; the
surface selects tools and may carry its own prompt and its own tool files.

```text
ora/agent/toolkit/tools/<name>/
  tool.py             run(api, args); ora/agent/toolkit/registry.py finds it by name
  description.json    ora.toolkit-tool.v1: description, inputSchema, resultSchema

ora/policy/tool_surfaces/<profile_id>/
  surface.json        ora.toolkit-surface.v1: tool order, roles, capabilities,
                      view config, required grants, receipt delivery,
                      tool_descriptions
  prompt.md           the surface prompt (may be empty)
  <file>.json         ora.toolkit-tool.v1 for one tool on this surface
```

A surface names a tool that has both files, or one whose description it keeps
itself. `tool_descriptions` maps a tool to a description file beside the
manifest, which replaces that tool's shared `description.json` on this surface
alone. Such a file may leave out `resultSchema`: the tool's own result schema
then applies, and the resolved surface records the file it came from as
component `result:<name>`. `"resultSchema": null` says the tool has no public
result schema (the `robot_runtime_v2` and Synthetic Native tools). A tool name
is snake case, or dotted segments of it (`robot_runtime_v2`'s
`observation.dump_state`). A `tool_descriptions` entry is a path string: an
inline `{ref, sha256}` pin is refused.

## Surfaces written out from layered manifests

Until 2026-11-03 most surfaces were layered manifests: a base catalog, ordered
description, schema and result layers, and capability binders that rewrote the
composed text. Each was written out in this form in place, by a tool removed
with the layered resolver: the registered manifest kept its path and became the
surface file, with its profile id, revision, runtime, tool order, roles,
capabilities and view config, and a `<tool>.json` beside it holds exactly the
definition its layers and binders composed wherever that is not the tool's own
`description.json`. The path stayed because sealed plans, documents and the
embodiment descriptors' provenance name it; `provenance.flattened_from` is the
commit whose layered manifest was written out. Only the component list, the
manifest and composition hashes and the provenance changed;
`compare-model-visible` was the same in every cell. No binder runs: the text is
in the files. `dry_run` stays the resolver's to add to the moving tools.

Written out: `yam_direct_v1` (YAM free play with `--no-perception`; its
successor `yam_perception_v1`, YAM's default, was authored as files too),
`robolab_direct_v1`,
`robocasa_mobile_direct_v1` (`robocasa_mobile_v1/`),
`libero_placement_direct_overhead_v1` (`explicit_motion_v3/`),
`robotwin_toolchain_b2_closed_noservo_v2` (`toolchain_b2_noservo_v1/`),
`robot_runtime_v2`, `native_atomic_causal_v1` and `local_atomic_cpu_v1`. The
last three also declared `toolkit_only_v1` then, which the host had added to
them at run time before; their plans name it, nothing the model reads changed.
A sealed plan keeps the definitions it was shown and the component paths it
read; evaluation and replay read neither from the tree.

## Receipts in the tool result

Each action tool's `description.json` describes its native result: the tool
returns its completed execution receipt and `remaining_budget`, and the
current camera images come with the next observation. A tool's own images
cannot ride in a tool message: a result names them, and they are delivered in
the user message right after the batch's results, before the next model call.
One description serves every surface; a surface that delivers receipts another
way says so at the surface level.

## The current one: `robodojo_placement_toolkit_v6`

`robodojo_placement_toolkit_v6` is v5 without the official check, RoboDojo's
default since 2026-11-02 (baseline `robodojo_toolkit_v7`, bundle `toolkit_v7`,
context profile `robodojo_public_explicit_v7` with `boundary-v3.md`). The simulator's official predicate is
all a success needs: it ends the episode, and no tool reads it.

- Nineteen tools: no `check_task`, no `required_grants`, no `faithful_check_task`
  capability, so the Runtime writes no `policy-task-check.json`.
- `finish_episode` is `finish_episode.json` on this surface: `give_up`, and
  `abort` on recorded infrastructure evidence. The model cannot declare success.
- The round ends at the official boundary with no terminal turn, so the model
  gets no message about the verdict.
- `prompt.md` is v5's without the rule on spending native actions, because
  [no native-action count ends a fresh RoboDojo run](configuration-and-evidence.md#official-limit-and-the-extended-budget).

Everything else, tool by tool, is v5, below. v5, baseline `robodojo_toolkit_v6`
and bundle `toolkit_v6` stay registered so the plans sealed on them resolve.

## `robodojo_placement_toolkit_v5`

`robodojo_placement_toolkit_v5` is `robodojo_placement_toolkit_v3`'s tool set
plus `return_home`, with the `check_task` grant and `tool_output` receipts,
written out once per tool. v3, the layered legacy-visible surface it was
written from, was retired on 2026-10-31; the comparisons with v3 below describe
v3 as it was. Its roles, capabilities and view config are v3's
under `toolkit_receipt_delivery: tool_output`; the action tools' descriptions
are rewritten for the receipt they return instead of v3's "explicitly selects
tool_output" addendum. A RunSpec on it resolves
`toolkit_receipt_delivery: tool_output` without stating it.

Where it departs from v3 (all recorded in
`tests/ora/agent/composition/surface/test_toolkit_surface.py`):

- Moves run as joint trajectories: `move_pose`, `move_to`, `rotate_wrist`,
  `rotate_pitch` and the arm stages of `grasp_execute` each run as one screened,
  smooth joint trajectory instead of the released primitives.
- New optional inputs, each defaulting to v3's behavior: `speed`
  (`fast` | `normal` | `careful`), `slow_approaching`, `expect_contact` and
  `exact_orientation` on `move_pose`; `expect_contact` on `move_to`;
  `expect_contact` and `exact_orientation` on `rotate_wrist` and
  `rotate_pitch`. v3's inputs stay accepted unchanged. `speed` sets only the
  pace: the interaction label picks the route at every speed
  (`lib.motion_plan.route`). `expect_contact: true` is the one straight line past
  50 mm / 15 degrees, refused rather than replaced by another path. A controlled
  descent is `contact_approach` or a planned path.
- Where the shared planner runs, the straight part of a `contact_approach` or an
  `expect_contact` move is checked before it runs: the
  fingers may touch, and any link behind them that would go into what the cameras
  see refuses the move (`contact_path_blocked`). A `precision` line inside the
  local envelope has no such check.
- `move_pose`'s result gains a `motion` field: how the trajectory ran (route,
  speed, planned and executed knots, stop, settle, per-arm errors, `joint6`
  (per arm, the wrist roll's start, end, min, max, travel and least limit
  margin from the start joints through every planned knot), and
  `budget_limited` when its budget covered only part of the move). A budget
  cut never stops a joint-line bridge, or a joint-space move that turns joint 6
  by 90 degrees or more, part way: the covered part ends before it, or the move
  is refused before anything moves (`paced_motion_budget` with `unsplit_span`)
  when it starts with one.
- `recover_pose` takes `arms` instead of v3's `targets`: each named arm undoes
  its latest arm-moving tool call along that call's measured path, one per
  call. `return_home {arms}` lifts, then unwinds to the reset joints.
  [Policy I/O](policy-io.md#action-contracts) has the recovery semantics.
- `finish_episode`'s `give_up` adds the `insufficient_budget` category and
  `report_issue` an optional `kind` (`defect` | `confusion`); see
  [Lifecycle](policy-io.md#lifecycle).

Execution budgets differ from the released primitives' too. They are offline
implementation configuration in `ora/agent/toolkit/tools/budgets.py`:
every tool that moves the arm 250, what one joint trajectory may use, so a move
finishes unless the episode runs out; `set_gripper` 30, `wait_ticks` 20 and 810
for `grasp_execute`, whose three arm stages and two jaw commands each fit their
own limit; read-only and lifecycle tools spend 0. One call can spend 250 of the
episode, and its plan and dry run report what it will cost. The
model cannot supply `native_action_budget`, including zero, and the official
remaining allowance can reduce admission.

It was RoboDojo's default surface from 2026-10-30 to 2026-11-02. Naming it
resolves `robodojo_toolkit_v6`, the baseline the surface brings, with the
`check_task` grant the surface requires. That baseline uses the
`toolkit_v6` bundle (planner `policy-v6.md`, core `evidence-action-loop-v9.md`,
this surface's `prompt.md` in the system channel) and the
`context_keep_written` scheme ([keep-written context](keep-written-context.md); `context_keep_written_unaged_reasoning` on 2026-11-04 and 05, before that `context_rolling_tail_v2`, [append-only context](append-only-context.md), retired on 2026-11-05).
Code mode runs on it from `prompt: v2` (see [public interfaces](public-interfaces.md)).
`robodojo_toolkit_v5` (the same surface with `context_rolling_tail_v1` and no
surface prompt) was retired on 2026-10-31; sealed runs keep their recorded
definitions. Naming v5 brings its baseline and required grants:

```json
{
  "policy": {"tool_surface": {"profile": "robodojo_placement_toolkit_v5"}},
  "record_toolkit_io": true,
  "toolkit_grants": {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
}
```

## Retired surfaces

`robodojo_placement_toolkit_v4`, the file-backed surface that delivered
receipts with the next observation (an action tool's own output was its
submitted confirmation, and a `result_overrides` file restored the
observation-delivery result schemas), and its baseline `robodojo_toolkit_v4`
were retired on 2026-10-31, together with the layered v1 and v3 surfaces. No
registered surface delivers receipts with the observation any more. Sealed runs
keep their recorded definitions; evaluation, trace verification and export do
not need the repository's copy.

**In-place change (2026-10-28):** v4 and v5 gained everything listed above
(`return_home`, the new optional inputs, the `motion` result field,
`recover_pose {arms}` and joint-trajectory moves) in place, with no revision
bump. A run sealed on either keeps the surface of its plan's commit (the plan
pins each file's sha256). These applied only to v4 and v5: the layered v2
keeps the released primitives, inputs and budgets. Later on 2026-10-30 RoboDojo's default
moved from `robodojo_placement_closed_skills_v2` to v5 with
`robodojo_toolkit_v6` (user direction); the closed-skills surface and the v4 baseline's RoboDojo
instance were retired on 2026-11-01.

`toolkit_receipt_delivery` may only restate a file-backed surface's own
delivery. Component overrides are refused: change the files instead.

## Resolution

`resolve_tool_surface` reads the manifest (`surface.json`, or the manifest file a
written-out surface kept), then each listed tool's
`description.json` in order, through `policy.lock` (which covers
`ora/agent/toolkit/tools/*/description.json` besides `ora/policy/`). It checks each file's schema and name,
lints the input schemas, validates the tool-help contract when the surface
declares `policy_tool_help_v1`, and resolves the view config, tool roles and
effect profile. Nothing is composed or rewritten. The ResolvedToolSurface records one component per file
(`tool:<name>`, and `surface_prompt`), so a RunPlan pins every byte it used.

Grants are the surface's own: `required_grants` must equal the RunSpec's
`toolkit_grants`, and `validate_grants` applies its capability rules (a
`toolkit_legacy_projection_v1` surface requires both `check_task` grants).

## Surface prompt

`prompt.md` holds notes on the tool set as a whole. When it has text, the
baseline adds it as the last `tool_manual` layer, after the prompt bundle's
manual; when it is empty (as for v5), no layer is added and the prompt layers
are byte-identical to a surface without one.

**Correction (2026-10-26):** `tool_manual` is a manifest-only channel
(`context_audit.py`: `not_delivered`), so a surface prompt is pinned in the
plan but never delivered to the model. Earlier text here said it was
delivered. Delivering surface and context-scheme prompts through the system
channel is in progress (R0022 D4).

## Changing a surface

A description belongs to its tool: editing `tools/<name>/description.json`
changes what every surface listing that tool shows. Sealed runs are unaffected
(their plan records the commit and each file's sha256). Keeping two versions
of one tool's description selectable side by side is not supported yet.

To change the tool set or the prompt, copy the surface directory to a new
profile id, edit the copy, register it in
`ora/agent/composition/surface/profile_catalog.py` and add a baseline if runs
should select it by baseline. After any change run `./scripts/ora lock` and
compare what the model sees (`scripts/dev/compare-model-visible`, or a test
like the one above).

## Resident services

Toolkits v5 and v6 bind their resident services the way every other surface does: the
`ora.model-service-set-binding.v2` prerequisite names the plan's own surface in
each child's `tool_surface`, and preflight and the runtime both verify each
child against that surface. A child bound to another surface is refused.
Endpoint, instance/process, model, adapter and qualification checks run through
the unchanged service clients. There is no cross-profile reuse.

Until 2026-11-01 v5 declared `resident_service_binding` so that a service set
bound to `robodojo_placement_closed_skills_v2` could serve it, and plans recorded
that as `plan.json.service_binding_resolution`. Both left with the closed-skills
surface. The field stays in plans already sealed and nothing reads it.

## Branching on a surface

Runtime code asks the resolved surface for its capabilities
(`toolkit_only_v1`, `toolkit_synchronous_execution_v1`,
`toolkit_legacy_projection_v1`, `faithful_check_task`), never for its name, so a
new surface inherits exactly the behavior it declares.
`ora/agent/composition/surface/toolkit_profiles.py` only records the first toolkit
tool cohort and the default baseline of each RoboDojo toolkit surface. Every
registered surface declares `toolkit_only_v1`, and the host installs the toolkit
tools for exactly those (`served`).
