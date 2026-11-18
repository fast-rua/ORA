# Synchronous toolkit execution

RoboDojo's default surface `robodojo_placement_toolkit_v6` (baseline
`robodojo_toolkit_v7`) is the synchronous toolkit surface; v5 (baseline
`robodojo_toolkit_v6`) is the same surface with `check_task`. The v1 surface, which
kept the proposal-based behaviour (rejecting a second proposal in one parent
invocation), was retired on 2026-10-31 with its baseline and prompt assets; the
layered v2 (baseline `robodojo_toolkit_v3`, eighteen released robot tools, each
budget taken from the call) on 2026-11-01. v6 has nineteen tools; v5 has twenty,
`check_task` among them, which its required grants unlock (below); lossless
Context supplies the artifact reader separately. It runs in the existing process;
these API capabilities are not a Python security sandbox.

The toolkit `act` capability exposes only synchronous execution, with no legacy
`submit_action` adapter. A legacy `action_submitted` tool directive is rejected,
and the Supervisor refuses the old proposal fallback on this surface.

Each tool's parent budget is offline configuration
(`ora/agent/toolkit/tools/budgets.py`), never a tool argument: a call that
supplies `native_action_budget` is refused `invalid_arguments`. `move_to`,
`rotate_wrist`, `rotate_pitch` and `wait_ticks` get 20, `set_gripper` 30,
`move_pose`, `recover_pose` and `return_home` 50, `grasp_execute` 100;
read-only and lifecycle tools 0. Each `api.act.execute_action(action, budget)`
admits the minimum of its request, the parent balance, the episode balance and
the action's own cap (`joint_trajectory.native_action_cap`: 20 for a released
primitive, up to 250 for a joint trajectory). Only measured native actions are
charged. The response has `receipt` and
`remaining_budget`, never an observation. Read state separately after it returns.
The receipt is the public action-effect projection, plus action identity, actual
native consumption and explicitly projected controller stage-admission evidence
(consumed, reached/refused status, command verification and jaw directives).
No raw worker object is serialized into the public response. Default tools remove
`stage_evidence` before returning their results to the model; the full semantic
receipt remains in toolkit-io and the trusted ledger. See
[the field contract](toolkit-data.md#synchronous-execution-receipts-step-3).

Each primitive publishes an observation, policy-state entry and complete state
snapshot before returning. `grasp_execute` performs open, pregrasp, approach,
close and lift through separate calls. Verify uses the last publication's
measurements and public RGB-D; it spends no native actions and renders nothing.
Its closure, jaw hold and lift comparison predicates match the released path.

Positive parent consumption ends the model turn even if the tool later raises.
Zero-spend refusals remain in the ledger but allow another tool call. One parent
with positive spend increments `exec.tool_calls_with_actions` once; the round
limit is `budgets.max_robot_decisions` such calls, 100 when the RunSpec names
none (`SYNCHRONOUS_TOOLKIT_DECISIONS`, `ora/orchestrator/spec/runtime.py`). A
named budget is honoured as given (until 2026-11-14 it was cut to 100); a
free-play session keeps at most 100 (`ora play` names 400). The first phase still requires each action tool
to be alone in its assistant response; mixed read/action batches are refused.
The final observation is the last actual publication, not an action-count guess.
Intermediate publications never become extra Pi observation turns.

A surface without `faithful_check_task`, v6 among them, has no terminal lifecycle
turn: the round ends at the official boundary, as any terminal does, and the
model gets no further turn. On a surface with it, at an environment terminal
boundary, a read-only `check_task` does not end the terminal lifecycle turn: the
model can still call `finish_episode`. Pi still
refuses new physical actions there. If an execution attempt reaches the
Supervisor at that boundary, it is refused, recorded with zero consumption,
and ends the turn; positive consumption retains its `actions_executed` outcome.

Publications made inside a synchronous tool include every model request already
made in the current `act`. Pi sends its cumulative `model_calls_in_turn` as
trusted RPC metadata, separate from model-supplied tool arguments. The visible
balance is the limit minus settled round calls minus that in-flight count.
Multiple primitives or tools in one reply do not charge it again. `act` clears
the in-flight count when it returns (or raises); the Supervisor still settles
the returned count once. Existing published snapshots and their hashes are not
rewritten.

## A surface whose environment plans its moves

The host also runs a surface that declares `toolkit_only_v1` without
`toolkit_synchronous_execution_v1`, as `yam_direct_v1` and the defaults of LIBERO-Pro,
RoboTwin, RoboCasa365 and RoboLab (`libero_placement_direct_overhead_v1`,
`robotwin_toolchain_b2_closed_noservo_v2`, `robocasa_mobile_direct_v1`, `robolab_direct_v1`)
do; a RunSpec may name one on any backend, and only a synchronous toolkit surface requires
RoboDojo. So do Synthetic Native's and `ora local`'s atomic surfaces and
`robot_runtime_v2` (below): every registered surface declares it, and the host installs the
toolkit tools for exactly the surfaces that do (`host/runtime/dispatch.served`).
There the environment plans each move when it executes it, so the tool context's `motion`
section carries `environment_plans: true` and the tools hand over one proposal
instead of executing. Every tool the surface advertises runs through the toolkit
dispatch, and the tool state exists as it does on RoboDojo (`tool.marks` and
`tool.held.*`, seeded at install and cleared at each round reset). It belongs to
the host, which journals it in `state/tool-deltas.jsonl` as RoboDojo's owner does:
an `initialize` boundary at install, a `reset` boundary at each round reset and a
`parent` boundary around every call, named by the call's recorded proposal. A call
with no recorded proposal runs as before; its first write freezes the journal with
a gap. No worker reports an execution state here, so there is no
`state/exec-ledger.jsonl` and no snapshot chain (`state/steps/`), which only
RoboDojo's execution-state owner writes. `ora toolkit replay RUN --at N --tool T`
therefore takes the state before call N from the journal alone
(`deltas.journal_state`: the rows from `initialize` to the call's `begin`, on a
private copy for the tool's own writes), and reports 未记录 without the journal;
`--snapshot` stays 未记录 on such a run. A RoboDojo replay is unchanged: its tool
context has no `motion.environment_plans`, so it keeps the empty store it always had.

The context takes what the surface binds, whatever its embodiment: `jaw_offsets`
are the jaw centre in the controlled frame (zero on a grip-centre frame, empty on
a frame with no rule, as YAM has), `local_axes` are empty and `grasp` absent on a
surface with no controlled frame, `grounding_cameras` are the surface's observe
views, on a `single_arm_flat_targets_v1` surface `grasp` binds the right arm's
compiler and review cameras, and without `dual_arm_sparse_targets_v1` a surface
of this kind has `arms: ["right"]` in `observe`.

The tools take the arguments such a surface publishes and hand over the action the
runtime handlers hand over, byte for byte. On a `single_arm_flat_targets_v1` surface
the arguments are the right arm's request (`scalar_key` for the one value
`set_gripper`, `recover_pose` and the rotations take), and a `targets` wrapper is
refused. `speed`, `speed_m_s` and `slow_approaching` go in the action where the
surface declares `end_effector_speed_v1`; elsewhere they are unknown fields, as are
`expect_contact` and `exact_orientation` everywhere here, and as `target_ref` is
without `public_rgbd_retention_advisory_v1`. A compiled target carries
`observed_clearance` only under `observed_clearance_reference_v2`. A tool returns
`status: submitted` with no receipt, which arrives with the next observation, and
the host keeps what a closed jaw holds for the next move's clearance, as it does for
a grasp. `recover_pose` hands over its strategies alone: the environment holds the
anchors, so no `exec.*` state is read. `move_base` (`ora/agent/toolkit/tools/move_base/`)
runs only here, on a surface declaring `mobile_base_navigation_v1` (RoboCasa): it hands
over one `move_base` action whose world target it computes as the handlers did, and only
such a surface's tool context names it in `motion.mutation_refusals`. A dry run goes
through the gateway as the handlers' does: a refusal stays a refusal, the call is
recorded, and it stops at the action the real call would admit, which is what the
environment answers for.

RoboTwin's `grasp_execute` commits its sequence and does not run it: it writes the
ledger and the commitment the handlers wrote (`control/manipulation-skill-ledger.json`,
`control/manipulation-skill-commitment.json`) and hands over one
`ora-skill-sequence-action.v1`. The Supervisor drives the stages through the host's
`drive_skill_stage` and `build_skill_receipt` (`ora/agent/toolkit/host/skills/driver.py`), which
stay as they are until a synchronous study replaces them; the stages read the target and
grasp-plan artifacts the toolkit published, in the handlers' layout, and the staged receipt
arrives with the next observation. Its precommit check's motion probe keeps, for a variant the
worker's planner failed, the public planner error object (`stage`, `type`, `message`) the
RoboTwin and LIBERO-Pro workers publish (`interface/projections.motion_probe`). `align_plan`
sends the worker's read-only bridge the request the handlers sent.

The Viewer's images come from what the handlers logged, so such a surface logs it
too, on every call: `vision_ground` a `grounding_query_materialized` and `grasp_plan`
a `grasp_plan_materialized`, read from the query the result names (a repeated query
comes from its cache and publishes nothing), and `mark_point` an
`openeta_view_derived` for its annotated view and for its sheet, under the handlers'
identities and overlay paths. RoboDojo's toolkit surfaces log none of the three
(`mark_point` tests `motion.environment_plans`, and only this path calls
`host/runtime/tool_data.log_query`), so a sealed context replays what it recorded. What
else the handlers wrote has no reader and no toolkit counterpart: the
`segmentation_batch_materialized` event of a `vision_segment`, the point-cloud store
([toolkit data](toolkit-data.md#keys-in-the-tool-context-for-the-other-backends)) and
the mark store `control/openeta-points.json`, whose marks are `tool.marks` here.

Code mode's card does not change with the move: a contract's `script_reads` names
`query` and `source` only on a synchronous surface, and `robot.query` is refused
`query_unavailable` here as on the handlers.

The only new keys are `motion.environment_plans` and `observe.arms`, and only such
a surface has them. A RoboDojo tool context is byte for byte what it was, and a
context without them, which is every sealed one, means what it always meant. That
absence is what keeps `ora toolkit replay --tool` working on sealed runs: a branch
that is not RoboDojo's reads nothing on RoboDojo's path.

On such a surface `toolkit_only_v1` is left out of `policy_identity.model_visible_sha256`
(`ora/orchestrator/evaluation/model_visible.py`): it says which code answers a call, not
what the model reads, so a surface keeps its digest when it moves to the toolkit. Such a
run writes `native/toolkit-io` as a RoboDojo run does. A code-mode contract on
such a surface lists no `query` or `source` read: those need the worker's reach and
kinematics answers, which only a synchronous surface's worker gives; the host refuses
a script's query there too (`query_unavailable`). `lifecycle.outcomes` is the surface's
v2 contract or, where it declares none (`yam_direct_v1`, `native_atomic_causal_v1`,
`local_atomic_cpu_v1`), the released `success`, `failure` and `abort`, with
`released_arguments`, and `finish_episode` applies the released rules there: a failure
needs a reason and its attempts, and an abort no infrastructure evidence. Free play
withholds YAM's `finish_episode`, so no model reaches it on YAM.

## Synthetic Native, `ora local` and `robot_runtime_v2`

These three surfaces were the last layered manifests. Written out as files on
2026-11-03, they declare `toolkit_only_v1` in their manifests, which the host had added
to them at run time before; their `model_visible_sha256` and what the model reads are what
they were, and only their plans' component paths, hashes and capability list changed (the
code-mode pin in `tests/ora/agent/code_mode/test_planner.py` was re-pinned once for it).

`native_atomic_causal_v1` and `local_atomic_cpu_v1` declare
`atomic_pose_gripper_v1`. Their context has an `atomic` section, which no other
context has: the RGB resolution of their views, whether `view_dimensions` is
published, the bound end effector, the point-cloud views and whether the call is
one of several parallel `mark_point` calls. `observe` answers in `grip_xyz_m` and
the aperture. `mark_point` takes one pixel (`point_id`, `view`, `x`, `y`), also on
`candidate_agentview` and on a point-cloud view, whose pixel fixes two world axes:
the point stays `pending` until a complementary view fixes the third, and two
clicks of one axis disagreeing by more than 2 cm are refused `inconsistent_mark`.
Parallel calls of one turn share one image (`batched_mark_overlay_v1`), which the
host renders after the last of them as a `mark_point` run with no proposal of its
own. `preview_move` keeps its candidate in `tool.previews` (seeded and cleared at
each round reset with the other tool state) for a `move_pose` that names its
`preview_id` alone. `move_pose`, `close_gripper` and `open_gripper` hand over one
`ora-action.v3` action, a pose in `xyz_m`/`delta_mm` with axes or, under
`pose_frame_contract_v2`, the frame-explicit request, and a jaw opening of 0 or 1
for the bound arm. `finish_episode` takes the released outcomes (above).

`robot_runtime_v2`'s dotted tools run as `observation_dump_state`,
`vision_inspect`, `geometry_query`, `robot_read_manual` and `robot_submit_action`
on a `RobotToolRuntime` host (`toolkit/host/robot.py`). Each call's context has its own
section: the snapshot and publication `step` selects, the manual's text and
sections, and for `robot.submit_action` the generation and the workspace. None of
them needs an open observation: a saved step, the manual and the snapshot stay
readable while an action is pending, and the current observation is refused as
it was. `robot.submit_action` hands over one `ora-action.v4` `target_state`; the
host validates and persists it, and a second one on the same observation is
refused `duplicate_action`.

## Code mode's parent invocation

A code-mode script (`policy.code_mode`, see [public interfaces](public-interfaces.md)) runs as one parent
invocation marked `reentrant`: a synchronous tool the script calls runs inside that parent and spends its budget
instead of opening a parent of its own. `CodeModeActionExecutor`
(`ora/orchestrator/episode/code_mode_execution.py`) binds the executor on every backend through the methods of
`ToolActionExecutor` that it overrides: the episode's native actions remaining, the native actions a receipt used,
the cap of one primitive, the execution-state owner, the actions one execution accepts and what an interruption
records of its cause. A backend without native-action accounting (LIBERO-Pro,
RoboTwin, RoboLab) counts one per executed action; one that counts native actions but has no Supervisor-owned
execution state (RoboCasa365) lets one action use what the episode has left and charges its measured spend. On
RoboDojo the worker caps each action at the action's own cap (`joint_trajectory.native_action_cap`, the cap
tool-call mode lends too), and the script's parent lends each action it may run the most one action may use: 250,
since the surface's moves are joint trajectories (v5, `toolkit/host/motion.joint_trajectory_surface`; a surface of
released primitives would lend 20). A tool running inside the parent reads
what the parent has left (`nested_allowance`); Dispatch hands it to the tool as `script_native_actions_remaining`
(absent in tool-call mode), and the motion tools fit their plan to the
least of their own budget, the episode's remaining and the script's remaining, so the parent's admission never
stops a trajectory mid-path. Two other fields are now recorded in each call's tool context (the `configuration`
of `native/toolkit-io/tool-inputs.jsonl`), so v1-v3 contexts no longer keep their earlier bytes:
`episode_native_actions_remaining`, the episode's public allowance (an int, or null when unknown), written on
every `toolkit_synchronous_execution_v1` surface (v5) in both modes; and `lifecycle.categories`, the
`give_up` categories the surface's `finish_episode` declares, written on every `finish_episode` call. While a tool runs inside the
parent, the parent records that call (`Invocation.nested`: its `<call>#NNN` id and tool name), and each action
lends it to the worker (`limited_action`) with the ordinal of the parent's robot decision (`decision_counted`
once the parent's first spend counted it, also for a tool-call parent's later actions): the worker keeps one
`recover_pose` undo entry per arm-moving call of the script. A legacy tool's committed skill sequence (`ora-skill-sequence-action.v1`,
RoboTwin's closed-skill `grasp_execute`) is accepted where no execution state caps one action: the parent runs its
stage loop (`dispatch_action`) as one action, charged one native action even when it dispatched no stage. It is one
of the script's at most 12 actions; the script as a whole is one robot decision, charged at the parent's first spend.
On RoboDojo a skill sequence is refused before it starts; one predicate (`execution_state.per_action_capped`) decides
both where the coordinator binds execution state and where resolution refuses a closed-skill surface in code mode.
`bind_script_actions` installs the executor on the innermost tool
runtime, the one toolkit Dispatch reads, below every wrapper the coordinator adds (lossless retention's Artifact
reader, then code mode's tools): a wrapper forwards reads to its base, never writes.

## `grasp_plan` and the reach reference off the X5

`grasp_plan` and the reach reference of the moving tools answer as the handlers do on every embodiment, and every
backend's default surface runs them on the toolkit.

On a `single_arm_flat_targets_v1` surface `grasp_plan` takes no `arm`, plans for the right arm and publishes none. A
grip-site compiler (Panda, Robotiq TCP) publishes its complete identity, and its candidates carry the grip-site handoff
of `lib.compilation`. The tool context's `grasp` section has `selection` only where the hybrid is contact-first, which
is the Panda fallback family of `panda_public_rgbd_topdown_fallback_v1`: top-down candidates are shown only when the
learned pool is empty, and a target too sparse for that family costs only that family. A context without the key, every
RoboDojo and every sealed one, fills up to two top-down candidates first. The section has `pose_world_axes` (the
controlled point's local `approach` and `closing` axes) only on a `grasp_pose_world_axes_v1` surface (RO-27: every
default that has `grasp_plan`, on LIBERO, RoboTwin, RoboDojo, RoboCasa365 and RoboLab, since 2026-11-06); there
`public_result` adds `approach_world` and `jaw_line_world` to each candidate's `pregrasp_pose` and `contact_pose`, the
pose's rotation applied to those axes, rounded to 6 decimals, and each surface's `grasp_plan` result schema requires
both. A context without the key (a sealed run's) publishes the poses unchanged.

The v2 reach generations (`move_pose_reach_along_request_v2`, `pose_primitive_reach_along_request_v2`,
`reach_nearest_orientation_v2`) are `tools/move_pose/_reference.py`: every annotation names the IK that answered, and
the unsolved form carries the jaw centre's nearest turn. A backend's `EnvironmentRefusal` on the bridge keeps its code in
an annotation's `failure`, as the handlers report it; any other fault keeps its class name.

## Independent task-check grants

v5 requires both independent grants (its `required_grants`), and a RunSpec naming
it gets them with its baseline; v6 has neither the tool nor the grants:

```json
{"toolkit_grants":{"tools":["check_task"],"state_keys":["privileged.task_check"]}}
```

Either grant alone, or none, is rejected. `native/toolkit-state-grants.json` records the
assembly. Robot assembly has no privileged category, and this single-key grant
does not grant general privileged reads. `finish_episode` can cite public
completion evidence; only the benchmark boundary determines official success.

`toolkit_observation_receipts` controls visible receipt and receipt-derived
comparison/view summaries in subsequent observations. On v5 it follows
`toolkit_receipt_delivery`: false with the declared `tool_output`, true with
`observation`.
Receipts remain in tool results and trusted evidence regardless of this flag.
The flag does not suppress the current measured robot state or camera images.

## Durable execution evidence and recovery

`native/tool-executions/<identity_sha256>.json` uses
`ora.tool-execution.v1`. The identity is the SHA-256 of UTF-8 compact JSON
`[model_call_id,tool_call_id]` (no spaces, unescaped Unicode). Both IDs are stored
in the ledger and checked by Pi; a provider reusing a tool-call ID in another
model reply creates a fresh parent invocation. The Supervisor owns this file and atomically replaces it
with file and directory fsync at intent, completion and parent-finalization
boundaries. A Pi durability barrier precedes completion, so the receipt and
publication event prefix is durable before the host closes its intent.
Before each dispatch, the previous pending receipt is cleared. Completion must
bind the current action ID, never a previous action's receipt. Missing or stale
receipt delivery leaves the intent unclosed and terminates as world-state unknown.
It records requested/admitted budgets, actual spend/balance,
initial/final `{observation_id,step}`, and ordered actions. Each action has its
identity and envelope, admitted budget, before/after publication identities,
actual spend, public receipt, `receipt_event_id`, immutable `receipt_ref` with
SHA-256, and an optional refusal reason. `pending` holds an unclosed intent;
`finished` marks parent finalization. An action identity repeats only after a
Supervisor refusal, in tool-call and code mode alike: a refused action
(`ora.tool-action-refusal.v1`, zero native actions) spent nothing and published
nothing, so the next action bound to the same observation carries its identity,
whether it is a code-mode script's next call or, in tool-call mode, the action
of the model's next tool call (a zero-spend refusal leaves the turn open). An
identity that executed never repeats: the ledger refuses it within one
invocation, and the live world view refuses a proposal that repeats it. Recovery
and Pi's check of the trusted summary apply the same rule, and require a
refusal to spend nothing and keep its observation. The trace verifier does too,
in both modes: a refused action of an open invocation is closed by its refusal
receipt, with no fresh observation due, so the next request and action on the
observation it kept are no second action on it, provided the invocation's ledger
records the refusal ([evidence](configuration-and-evidence.md#episode-evidence)).

Python supplies the trusted summary to Pi independently of the tool result.
Pi records it in tool-result event payloads and the tool message's hidden
`_execution` field. It never reaches provider messages. Pi verifies receipt
hash/parent links and actual publication events, protects custom results based
on actual spend through L1–L4, and permits the next observation to span multiple
steps. Only positive-spend executions bind consecutive observation turns; a
zero-spend lifecycle/read-only call does not bind the next round's reset
observation. During restart, earlier replies in the same turn may contain
zero-spend primitives only when their completed trusted receipt evidence resolves.
Missing completion evidence retains the strict recovery refusal. The outcome is `actions_executed` / `ExecutedActions`, separate from a
proposal and independent of whether the tool result reports an error.

Interrupted execution has two terminal outcomes (Q45):

- An unclosed or undurable intent means the worker may have spent actions without
  a durable receipt and matching publication. Terminate with infrastructure
  `world_state_unknown`; never retry or assume the old state still holds.
- When every issued intent has a durable receipt and matching publication, the
  world state is known. Terminate with infrastructure `tool_execution_interrupted`;
  a closed ledger does not authorize recovering a cached parent result or continuing.
  This includes single-action tools. A read-only tool's process crash/disconnection
  has the same termination, with `last_action_id: null`.

Both outcomes preserve all evidence, reject subsequent physical actions, and set
`behavioral_sample_valid: false`. Once the current parent has attempted execution,
any uncaught exception or worker/RPC/publication failure terminates. Normal
`ToolRefusal` and budget-refusal results are not crashes. Ordinary Python errors
in tools that have not attempted execution still become tool errors for the model.

`native/tool-execution-interruption.json` (`ora.tool-execution-interruption.v1`)
is mirrored as `tool_execution_interruption` in `result.json` and `status.json`:

```json
{
  "schema": "ora.tool-execution-interruption.v1",
  "reason": "tool_execution_interrupted",
  "model_call_id": "model-call-000001",
  "tool_call_id": "parent-1",
  "last_action_id": "action-1",
  "world_state_certainty": "known"
}
```

`reason` can also be `world_state_unknown`, with certainty `unknown`.
`last_action_id` names the pending action if present, otherwise the last completed
one, and is null if no action was issued by this parent. A parent, a tool call's
or a code-mode script's, also records `cause` when an exception
interrupted it: `{type, message, stage}`, with `stage` `action` (an action the
ledger holds as begun and not completed), `tool` (the script or a tool, no action
pending), `ledger_completion` or `ledger_recovery`. A RoboDojo task that rules its
scene invalid shows here as `official environment became unstable`: upstream does
not count such a layout, so leave the cell out of a batch's denominator. Only the first interruption is
kept, so the ledger completion that fails after an action's failure ("execution
ledger completion is not durable") does not replace its cause. An interruption no
exception raised records none: an agent core restarted with the parent's result
unsettled or replaying its request (`tool_rpc.py`, `core_process.py`), or a closed
ledger found when the parent starts again. A tool call's record never has `cause`.
These are trusted owner
fields, not model/tool-supplied authority. A fault finalizer preserves this reason
instead of replacing it with the generic `trusted_fault`.

The synchronous host tracks result delivery separately from execution completion.
Pi receives a hidden `execution_delivery_guard`, persists the complete read-only
or zero-spend reply, and acknowledges those parents through `tools.settled` before
its next provider request. A positive-spend parent remains pending until its `act`
response reaches the Supervisor. Process loss before confirmation terminates,
even if the result or finished turn was already persisted. Any attempt to recover
that parent's cached result also terminates without running the tool body again.
Already confirmed historical calls do not make a later provider-only restart
fatal. Legacy v1 process recovery remains unchanged.

This does not claim exactly-once execution across an uncertain worker boundary.
Versioned evidence is separate from best-effort `toolkit-io` instrumentation:
recording gaps remain fail-open, while an interrupted execution is fatal.

## Intentional model-visible changes and acceptance

Each surface's model-visible text is its own files (`ora/policy/tool_surfaces/robodojo_placement_toolkit_v6/`,
[toolkit surface](toolkit-surface.md)): synchronous result schemas with receipts and balances, and the turn,
count and budget rules of its `prompt.md`. A change to them is a new surface revision; existing surfaces are
checked against the prior model-visible matrix (`scripts/dev/compare-model-visible`).

Live review should inspect ordered `action_proposed` / `environment_receipt` /
`fresh_observation_published`, parent `tool_result.payload.execution`, ledger
requested/admitted/used/remaining and pending, the final observation identity,
`exec.tool_calls_with_actions`, `exec.last_receipt`, native counters, per-action state
snapshots and absence of an old composite `execute_skill_sequence` on v5. The
reviewer runs arrange_largest_number and fasten_screws; CPU tests do not replace
those simulator checks.

## Reviewer launch and inspection

A qualified or warning-admitted RoboDojo task/seed/layout RunSpec that names
no baseline gets v6 and baseline `robodojo_toolkit_v7`; add
`"record_toolkit_io": true`. Each tool's budget is the fixed one above.

For each parent in arrange_largest_number and fasten_screws, check:

1. `tool_result.payload.execution` and the matching `native/tool-executions/`
   ledger agree on requested/admitted/used/remaining, action order and final
   publication. Each receipt SHA resolves through its environment_receipt event.
2. Every completed primitive has its publication and state snapshot; verify has
   none of its own. `exec.native_actions` advances by actual use and
   `exec.tool_calls_with_actions` increments only once for a positive-spend parent.
3. Default model tool results omit `stage_evidence`; subsequent observations omit
   receipts/comparisons/historical receipt images while retaining current sensors.
   Full semantic receipts remain in toolkit-io and the trusted ledger.
4. An all-zero refusal leaves the model turn open. An interrupted executing
   parent terminates with `tool_execution_interrupted` when its ledger is closed,
   or `world_state_unknown` when completion is missing or undurable. Both exclude
   the run as a behavior sample and never repeat the parent or its actions.
