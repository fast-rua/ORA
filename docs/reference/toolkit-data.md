# Categorized toolkit data API

Architecture v1 phase 1 step 2 assembles immutable capabilities for tools.
Tools and lib call the categories directly. No category caches observations,
receipts, controller holds or recovery anchors. The Supervisor owns the state
store; a data read resolves its current value on every call.

| Prefix | Package under `interface/` | Source | Without simulator |
| --- | --- | --- | --- |
| `perception` | `perception` | Explicit external model services | Recompute from verified published inputs |
| `state` | `state_access` | Current state owner; native publication references | Read loaded snapshot, original bytes |
| `planner` | `sim_query` | Worker planner: probe_motion, align_plan, reach_check, kinematics (`ora.kinematics-request.v1`: SDK CuRobo IK chains seeded from the previous solution, FK, sphere screens; the shared base's `plan_path`, `pace_path` and `contact_path`) | `SimulatorRequired` (需要仿真器), even with historical answers |
| `act` | `sim_exec` | Validated host execution boundary | Refused in phase one |
| `episode` | `lifecycle` | Supervisor lifecycle control | Replay output only |
| `privileged` | `privileged` | Published privileged state | RSI only |
| `artifacts` | `artifacts` | Granted source blobs and output artifact repository | Read source, write independent output |
| `compute` | `compute` | Bounded pure numeric evaluator | Recompute |
| `context` | `context` | Attachment declaration callback from context owner | Only if explicitly supplied |

`assemble_robot` returns eight categories and has no `privileged` attribute.
`assemble_rsi` includes that ninth category. Both accept only appropriately
scoped state accessors, never a supervisor writer. `state.read(key)` refuses
privileged keys; `state.write(key, value)` accepts finite JSON under `tool.*`
only. Loaded snapshots remain read-only. These are same-process API boundaries,
not Python isolation or protection against introspection.

The transitional `state.check_task()` entry requires both a `check_task` tool
grant and a `privileged.task_check` key grant. Runtime assembly obtains these
from the selected tool list and the `faithful_check_task` capability,
respectively. Either grant alone refuses. The trusted adapter reads only the
single state key; it exposes no general privileged namespace. Assembly writes
`native/toolkit-state-grants.json` (`ora.toolkit.state-grants.v1`) with
`reader`, `privileged_category`, `tool_grants`, `key_grants`, `effective_keys`,
`source`, and `transitional`. This evidence does not enter model content.
RoboDojo's default surface, `robodojo_placement_toolkit_v6`, has no `check_task`
tool, so neither grant exists there and the entry refuses.

Stateful owners are explicit:

- `toolkit/recording.py`: per-parent invocation ordinal and recording-failure flag;
  the shared Journal owns ordered durable events. All categories share one ordinal.
- `toolkit/replay_session.py`: historical cursor and diagnostic proposals/missing requests.
- `interface/tool_data.py`: the artifact repository owns immutable input/configuration
  bytes and derived-output lookup. Trusted host assembly generates and writes the
  input/configuration table before the tool runs; the category does not generate it.
- `toolkit/state/`: all changing environment and execution values. Production
  episodes inject their existing owner. Standalone tool hosts import their public
  publication into a host-owned store; that import never happens inside a category.

The flat methods on Runtime/Replay remain internal wire adapters for recorder
instrumentation and historical evidence tests. Live dispatch hands tools only the
classified facade. Library and tool source use category prefixes throughout.
The existing replay CLI keeps its diagnostic recorded/live semantics, including
explicit live execution, for existing consumers. New step/native-action snapshot
selectors are a separate path and cannot grant execution. `assemble_snapshot`
provides snapshot reads, current external services and explicit planner/action
refusals. Lifecycle intents use an explicitly supplied output-only recorder;
without one they refuse, never fall back to a live Supervisor. It does not grant
live simulator authority.

## Synchronous execution receipts (step 3)

The synchronous toolkit surface (v5; the layered v2 was retired on 2026-11-01)
uses `act.execute_action(action, budget)`; toolkit-v1 retained its frozen
submission path until it was retired on 2026-10-31. See [execution](toolkit-execution.md) for
budgets, independent check_task grants and the trusted execution ledger.
The data return has only `receipt` and `remaining_budget`; observations are read
separately after each publication.

`receipt` contains the original minimal action effect, `native_actions`,
`action_id`, and a narrow `stage_evidence` projection for tool-local predicates:

- `execution_status`: the controller endpoint status (also represented in the
  minimal effect; retained here for the original stage predicate).
- `consumed`: whether the controller consumed the request.
- `command_invariant_verified`: `command_invariant.verified`, as a boolean.
- `gripper_directives`: numeric `left`/`right` directives only.
- `failure_code`: the controller failure code, or null.

This is an explicit semantic projection, not serialization of a worker object.
Default tools use this evidence to preserve the original continue/stop and jaw
verification rules, then strip `stage_evidence` from their model-visible results.
The complete semantic receipt remains in toolkit-io and the execution ledger;
RSI-authored tools can use it through the same data API. Observation receipts and
their derived summaries have a separate Q18 delivery switch, default off in v2.
Attachment delivery retains the context owner's existing result projection.

With `toolkit_observation_receipts=true`, a synchronous `grasp_execute` also
projects the complete stage prefix and read-only verification into the next
observation's `last_action_effect`, using the same projection as the released
composite skill. If lift was attempted, up to two historical baseline images
come from the existing post-close publication. No extra render or physical
action occurs. The final current views and lossless input-view retention are
unchanged. With the switch off, neither this summary nor baseline images are
delivered. A zero-spend grasp does not create a synthetic action observation.

The tool's private `_runtime.skill_feedback` carries sequence/arm/verification
metadata and is removed by live dispatch before returning the public tool
result. The execution owner validates stage identities and receipt contents
against its own ledger; publication roots never come from the tool. This is
tool-derived semantic feedback, not a new worker execution receipt. Its
provenance is stored in `native/tool-observation-feedback/<parent-hash>.json`
(`ora.tool-skill-observation-feedback.v1`): model/tool call IDs, immutable
`feedback_ref`, and the contributing action IDs, receipt event IDs/references
and after-observation identities. The last real environment receipt binding
and all ledger actions remain unchanged. Refused stages remain visible in
the prefix; verification is present only if actually computed.

## Tool-owned holding intentions

Synchronous tools store `tool.held.left` / `tool.held.right` as null or an object
with `source`, `target_ref`, `attach_pose_xyz_xyzw` (episode-local metres / xyzw),
and `role: dispatch_intent_not_grasp_evidence`. These are intentions about jaw
contents, never contact, attachment, retention or official-success evidence.
`set_gripper` preserves the old annotation rules: opening clears the selected
arm; an annotated close on one arm stores its measured pose; an unannotated or
ambiguous two-arm close keeps prior intentions. Public target geometry supplies
clearance estimates without a hidden runtime cache. The geometry is read from
verified `files["manifest.json"].localization`, not artifact-index metadata.
Free-text annotations, candidate IDs, missing or failed target records keep an
unknown extent, as in the released tools; optional extent lookup cannot block
motion.

Tools write intentions **before dispatch**. Each action-completion snapshot
therefore contains dispatch-time intent. A rejected/zero-native-action request
restores the prior value. A pre-intent validation failure may also restore it:
the execution owner emits `ActionNotStarted`, projected in the data error as
`execution: {status: "not_started", native_actions: 0, stage: "before_intent"}`.
An intent-write failure before dispatch can carry the same evidence with
`stage: "begin"`; a partially persisted intent still blocks the session until
its evidence is resolved, even though the owner knows this request never ran.
Tools may not infer zero spend from an arbitrary exception. After begin returns,
errors keep the staged intention and incomplete execution is
terminated as `world_state_unknown`; no subsequent physical action is allowed.
Grasp clears before open, writes its commitment before close, and may clear after verify when existing jaw evidence says `unlikely`.
On a normal stopped grasp, any positive spend preserves the commitment intent,
including a stop before close; an entirely unspent grasp restores its original
intention. Unknown evidence is not a negative verdict. Verify changes no physical state
and adds no publication; that correction appears in the **next** completed
snapshot. Round reset clears both intentions.

## Explicit recovery anchors (step 5)

`recover_motion` in `ora-motion-action.v3` accepts an optional
`arguments.recovery_targets` object alongside `targets`. It has exactly the
same arms as `targets`; each value has `strategy` (`home` or
`backtrack_last_safe`), `arm_joint_qpos` (six finite radians), and
`pose_xyz_xyzw` (seven finite numbers, episode-local metres / xyzw). The strategy
must match `targets`. The new recover_pose tool reads
`exec.recovery.<arm>.home/previous` and restates those values explicitly.

Before any worker dispatch the execution owner compares them without tolerance
to its current selected anchors. A missing, changed or invented anchor produces
a completed zero-spend refusal, `stale_or_mismatched_recovery_target`, in the
trusted ledger; it consumes no action budget, publishes no new observation, and
does not poison the session as world-state-unknown. The owner checks again when
assembling the worker request. This is not an arbitrary joint-target capability.
Worker recovery and safety routes remain unchanged. Older action envelopes omit
the field and retain the original owner lookup; their schema and bytes do not
change. Explicit joint values are internal data, not added to model tool inputs
or public strategy/compilation results.

## Named public marks and geometry (step 5)

`tool.kept_plans` holds the plans a dry run made for the current observation, keyed by the observation and every input
of the plan, at most four (`lib/motion_plan/kept.py`): a real call with the same inputs runs the kept plan, so replay
reads it here and asks the kinematics nothing.

`tool.marks` is a JSON name-to-record map. Using a single map permits the old
point-name alphabet (including spaces and punctuation) without interpreting
names as state key paths. Each record retains its status, world `xyz_m` in
metres, observation ID, source cameras/pixels, depth-support quality, workspace
advisory and last cross-view verification. Repeating the exact same source
reuses its measurement; another view/pixel verifies the stored coordinate and
rejects a disagreement over 0.025 m. Individual failed pixels do not discard
successful entries. `observe(history_point_ids=...)` reads this map for overlays.
Round reset clears it. Tools use state accessors; the data layer caches no points.

Geometry queries grant the requested verified policy-state publication and read
its source depth/calibration, preserving current/saved-step hashes and refusal
fields. They never query the simulator or mutate named marks.

Both tools read a camera through `observation.load_pixels`, which assumes no pinhole
or rigid pose. Each checks the calibration fields it uses, so a camera with no
`extrinsic_cv` (a YAM camera with no calibration on file) answers a `region` query,
and `mark_point` refuses it with `invalid_camera_calibration` at the first pixel whose
depth it accepts. A view or camera the publication lacks is an argument refusal:
`view is not markable` on each `mark_point` entry, `camera is not policy-visible` on
`geometry_query`, `vision_ground` and `observe`.

Full snapshots retain the existing observation-completion schedule. A tool-state
write appears in the next completed snapshot. Exact replay *between* consecutive
read-only calls also requires the delta protocol below; a snapshot alone does
not reconstruct intermediate tool-state boundaries.

### Tool-state delta evidence (step 4)

`state/tool-deltas.jsonl` uses `ora.toolkit.tool-state-delta.v1`, independently
of the optional toolkit I/O recorder. One state owner holds the only writer:
RoboDojo's execution-state owner, or the tool host of a surface whose environment
plans its moves (`toolkit-execution.md`), whose journal is its only tool-state
evidence. `journal_state` gives the state before a parent from the journal alone,
since it starts at the owner's `initialize` boundary.
Each row contains `sequence`, `previous_sha256`, `sha256`, `kind` and `context`.
The hash covers canonical finite JSON without its own `sha256` field. `begin`
and `end` bracket every dispatched synchronous tool, including tools that do
not write state. Parent context binds `model_call_id`, `tool_call_id`, run and
round. Repeated provider tool IDs remain distinct by model call. Initialization
and round reset instead use `kind=owner, operation=initialize|reset` contexts.

`write` contains `changes`: each key has its previous value's canonical-byte
SHA (null means absent) and its new JSON value. Only `tool.*` is accepted.
`interface_ordinal` is the next toolkit I/O position, with
`interface_boundary=before`; it does **not** consume an I/O ordinal.
`state_write_ordinal` starts at zero inside each boundary and distinguishes
multiple writes between the same two I/O calls. Writes are flushed/fsynced.

Complete snapshots optionally include `tool_state_watermark` with schema,
count, last SHA and completeness. A snapshot can land in the middle of a
multi-action parent. Reconstruction checks the chain and previous-value hashes,
applies only rows after the watermark and before the selected parent's begin,
and creates a private tool-writable overlay. Source snapshots/files stay intact.
Older snapshots without a watermark cannot prove exact between-call state.

Recording failure leaves the live state write and original tool return/exception
unchanged, freezes the writer, and best-effort appends
`state/tool-deltas.gaps.jsonl` (`ora.toolkit.tool-state-gap.v1`: stage, sequence,
time_unix_s, model_call_id, tool_call_id, error.type/message). Even gap failure
is swallowed. No later begin is written by the frozen writer; exact restoration
of that interval reports missing evidence. Full snapshots still contain their
actual complete state, with a non-complete watermark. Never infer lost deltas.

### IK generator ledger

`state/exec-ledger.jsonl` (`ora.toolkit.exec-ledger.v1`) keeps every change of
`exec.ik_sampler_state`, the CuRobo seed generator the RoboDojo worker reports
in the execution state of every RPC response, whichever request solved IK
(`ik_chain`, `reach_check`, `align_plan`, `probe_motion` or a motion action),
independently of toolkit I/O and without per-method seeding fields. The owner
binds it with the tool-state journal; its first row (`kind: bound`) is the
value in effect then, and each later row (`kind: change`) the value after the
RPC `method` that changed it, with `previous_value_sha256`. Rows are
hash-chained like `tool-deltas.jsonl` (`sequence`, `previous_sha256`, `sha256`
over the canonical row body) and carry `key`, `value`, `value_sha256`, the
tool scope (`context`: the parent `model_call_id`/`tool_call_id`, an owner
boundary or `null`), `interface_ordinal` (the data-interface ordinal of the
call in progress inside a parent) and `tool_delta_count` (the
`tool-deltas.jsonl` row count when written, which orders the row against every
tool call's `begin`). A failed write freezes the writer and appends
`state/exec-ledger.gaps.jsonl` (`ora.toolkit.exec-ledger-gap.v1`). The seal
inventory covers both files.

`ora.agent.toolkit.state.exec_ledger.read_rows(run_root)` validates the
chain, the sequence and each value's hash, and refuses an absent ledger
(`未记录`) or a mismatch. The ledger is evidence: nothing restores from it (a
restore resumes the generators from the boundary snapshot's
`exec.ik_sampler_state`). The per-row seeding inside `ik_chain` answers stays as
it is.

### Independent point segmentation artifacts

`vision_segment` uses `perception.segment_points` with RGB/depth blobs from the
selected current-round publication. Candidate selection belongs to the tool;
the interface preserves the full service pool. `segmentation_query` artifacts
are identified by `segment-query-<24 hex>`, schema
`ora.point-segmentation-batch.v2`; allowed members are manifest.json,
raw-masks.npz, overview.png and candidate-T0.png through candidate-T4.png.
Runtime publishes these atomically under segmentation-batches/ and verifies
all manifest references when reusing them. Masks remain indexed public targets
for subsequent tools. Replay ignores the query cache for inference and records
fresh-service comparisons; no new simulator query is introduced.

### Keys in the tool context for the other backends

These keys are absent on RoboDojo and in every sealed tool context, and an absent key
means RoboDojo's behaviour:

| Key | Where | Set when | What it does |
|---|---|---|---|
| `overhead_view_pose: true` | Top level | The surface declares `overhead_footprint_localization_v1` | `vision_ground` and `vision_segment` read the current observation once more and add `overhead_view_pose` to each oblique footprint |
| `pointcloud` (`cameras`, `workspace_bounds_m`, `resolution`) | `observe` section | The surface declares `two_view_pointcloud_v1` | `observe` admits `pointcloud_top`, `pointcloud_front`, `pointcloud_side` on a current call, and refuses bad `views` as `views must be a non-empty unique list` or `unknown observe view` |
| `camera_list` | `observe` section | RoboTwin (`RoboTwin`), YAM (`station`) | Names the backend in `views must be a non-empty unique <name> camera list` |

A point-cloud view is recomputed from the published RGB-D on each call that names one
and is pure, so `ora toolkit replay --tool observe` needs no record of it beyond the
observation. Its image is published as an `observation_overlay`, at the overlay path and
with the identity the handlers used. The two-view store the handlers wrote beside it
(`points.npz`, the three bare views and their manifest, `pointcloud_materialized`,
`pointcloud_requested`) has no reader and no toolkit counterpart.

### Alignment, rotation, wait and issue tools

Alignment measures only the current public target's mask/RGB-D, validates its
source binding and depth quality, and submits all generated poses through
`planner.align_plan`. Refusals retain public field/allowed-value details. New
queries in recorded replay still fail NotRecorded. Rotations and hold-wait
submit exactly one action through `act.execute_action`; completion, budget
accounting and state publication remain owner responsibilities.

`episode.report_issue` validates its message and appends to the existing fixed
issue evidence file. It preserves the round lifecycle and finish/abort evidence
path. This is explicit episode evidence, not tool-local hidden memory. No new
cross-call state was found beyond tool.held, tool.marks, tool.kept_plans and already recorded
public artifacts/episode evidence in this migration wave.
