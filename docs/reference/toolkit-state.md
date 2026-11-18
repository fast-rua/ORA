# Toolkit state and worker execution context

The state-layer migration is incremental. This page records implemented
admission behavior; state ownership and snapshot details are added with their
implementation. No tool, prompt, context-retention or action-budget change is
part of the first step.

## Qualification warnings

RoboDojo run admission accepts absent qualification bindings and absent
referenced card/task/scene records. Compose accepts an omitted
`--qualification-reference` / `--task-qualification`. Preflight still checks
source/layout integrity, the interpreter, current GPU memory and driver
compatibility. A missing qualification cannot establish a historical GPU UUID
or graphics comparison; the report explicitly marks those as unavailable.
The ordinary `ora run --env robodojo` route does not automatically requalify a
missing record. An explicit `--requalify` still runs the requested qualification.

Run preflight uses the qualification readers with `allow_kernel_drift=True`.
Changed worker/control source digests become warnings naming both commits,
both worker digests and, for revalidated lanes, the changed paths. Historical
qualification creation and strict readers retain their checks. Record hashes,
JSON, schema, configuration, task/layout/device identities and raw evidence
are still checked. A corrupt record is not an absent record.

Evidence is in the run's `preflight.json`:

- `robodojo_infrastructure_qualification` has `passed: true` and
  `severity: warning` when execution is admitted without matching qualification.
- Missing evidence uses `ora.robodojo-unqualified-admission.v1`,
  `qualification_available: false`, the supplied reference (or null), plan
  commit/kernel hash, and `warnings[].code = qualification_missing`.
  It does not fabricate a passed qualification or a worker binding.
- Available records retain their original schema and contents; kernel admission
  includes `worker_kernel_mismatch` or `terminal_dependency_mismatch` warnings.

Nonzero layouts with layout-dependent language need no qualification job.
Without a gate readback, the plan binds `runtime.task_language_source` to
`live_reset_readback` and records a pending-language placeholder. After reset,
the Supervisor requires the environment's official instruction, writes
`native/task-language-binding.json` (`ora.task-language-binding.v1`), records
an event, and uses that exact instruction for the first and subsequent turns.
The sealed plan is not rewritten; the placeholder is never sent as the task.
The binding records `gate_verified: false`, source, task, layout and warning.
Fixed-language tasks continue using their reviewed instruction. Portable
layout/contract checks still apply independently of qualification.

## State namespaces and snapshots

`ora.agent.toolkit.state.store.StateStore` is the Supervisor-owned store.
Environment writers own `env.*` and `privileged.*`, execution writers own
`exec.*`, and tool writers own only `tool.*`. Values are copied on read and
write; JSON must be finite. Robot access denies privileged keys unless both
`check_task` and `privileged.task_check` are explicitly granted. RSI readers
can inspect all namespaces. This is a same-process boundary, not isolation.

`state/steps/step-NNNN/snapshot.json` (`ora.toolkit-state.v1`) contains `values`,
`policy_state` (path, bytes, file hash and state hash), `artifacts` (public
observation file references), and `previous_snapshot_sha256` / `snapshot_sha256`.
It aligns one-to-one with existing policy-state publications. Original images,
depth arrays and measurement files remain in `policy-observations/`; no copies
are made. Export is forbidden between an action and a complete publication.
`load_snapshot(run_root, step)` validates both hash chains and public artifact
bytes and returns read-only access without starting a simulator. Paths inside
tool JSON are not filesystem capabilities. Partial compound-action snapshots
are deferred to the later unified execution API.

`exec.recovery.<arm>.home` is measured each round; `exec.session_home.<arm>` is
fixed at the first reset. `exec.recovery.<arm>.previous` holds the joints and
link6 pose measured when the arm's last pose move started and, when that move
was a joint trajectory, `measured_path_rad`: the joints measured at the end of
every native action it ran, first row the anchor; a recovery never replaces
it. `exec.recovery.<arm>.undo` (`moves`, oldest first, at most 5) is the
undo stack recover_pose retraces on the trajectory surfaces: one entry per
arm-moving tool call of the round (`tool_call_id`, `tool`, `ordinal`, the
start `arm_joint_qpos` and `pose_xyz_xyzw`, the `measured_path_rad` of all its
trajectories and, after a cut retrace, `undo_progress`); a completed retrace
pops the top entry, recovery trajectories never push one. A restore rebuilds
both from the recorded receipts and traces. These values can differ slightly and must not be
collapsed. Pose targets use episode-local metres and xyzw quaternions, joints
use radians, and gripper directives use normalized opening.

## Execution wiring (architecture v1, step 1)

The RoboDojo coordinator creates one `ExecutionStateOwner` before worker start.
The regular and toolkit-live RPC routes send `execution_state` alongside each
request: `{schema: "ora.execution-state.v1", values: {exec.*}, recovery_targets}`.
Replies carry the measured/updated execution values, including on an ordinary
worker exception. The transport forwards this exchange to its owner; it does
not keep a second command cache. Other backends retain their existing wire.

`ExecutionRequest` installs detached command and anchor dictionaries only for
the duration of a worker request and removes them on exit. Reset measures the
new round's home, retaining the explicitly supplied first-reset session home.
Recovery takes per-arm `{strategy, arm_joint_qpos, pose_xyz_xyzw}` selected by
the host. Planning rejection keeps the previous anchor; the existing planning
success / pre-execution update remains unchanged. Motion, collision screens,
native stepping and rendering order are unchanged. Simulator/lifecycle and
artifact bookkeeping remain in the backend; the command/recovery memory now
belongs to the state layer.

Every `worker.execute` refreshes execution state. Complete state is exported
only after an existing observation and policy-state publication, including the
initial reset publication. No extra render or policy step is introduced. The
parent tool's native-action count determines whether it increments
`exec.tool_calls_with_actions`; a composite counts once, a zero-step refusal
counts zero. Existing episode budgets are unchanged in this step.

`privileged.task_check` comes from the same trusted status read as the existing
publication. When ground-truth capture is enabled, the last already-recorded
native tick for that round is copied into `privileged.ground_truth`. Neither
causes an extra simulator query; neither is included in robot state access or
policy-state/context. Tool-held-object write integration is part of later tool
migration, not inferred from old tool ledgers.

The standalone control-replay, physics-probe and play-audit workers use a
single temporary execution scope for their whole diagnostic request. Toolkit
live instead carries host-owned state between its start, restore, verification
and execution RPCs. The old standalone numerical motion-qualification `main`
now fails explicitly before boot, directing operators to the orchestrator
command `ora qualify robodojo motion`. Step 2 migrated its host driver;
its numerical cases and record schema remain available and tested.

For ungated language readback, the sealed plan stays unchanged. Terminal turns
use the bound language, the context manifest is rebuilt after reset, and the
audit task layer cites `native/task-language-binding.json:task_instruction`.

A simulator-free inspection example:

```python
from pathlib import Path
from ora.agent.toolkit.state.snapshot import load_snapshot

loaded = load_snapshot(Path("outputs/example/runs/run-id"), step=0)
public = loaded.access().values()
all_state = loaded.access(reader="rsi").values()
manifest = public["env.observation"]["manifest"]
original_observation_json = loaded.read_artifact(manifest["path"])
```

An unavailable or stale ground-truth tick is represented by `null`, never by
another round's tick. Ordinary tool access does not include that key.

The public `env.terminal` keeps the terminal Boolean but maps every `official_*`
reason to `official_boundary`. Only `privileged.task_check` retains the original
reason and official-success Boolean. This projection applies to the new state
layer; existing policy-state, tool output and trusted status evidence are not
rewritten.

Tools now access this store through the [categorized data API](toolkit-data.md).
The motion qualification host migration is complete: use
`ora qualify robodojo motion --runtime-config CONFIG --output DIR [--gate FILE]`.
It binds the same host execution-state owner before worker startup. The old
module invocation gives an explicit redirect error and starts no worker.
