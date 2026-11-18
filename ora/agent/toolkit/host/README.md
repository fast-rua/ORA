# ora/agent/toolkit/host

## Responsibility

The trusted host the toolkit's tools run under: it binds each call to its
public data, configuration and services, records the call's I/O, and hands a
proposal to the Supervisor. It holds what the tools themselves may not: files,
the control root, model-service clients, the restricted evaluator's process,
the episode's private state. It also holds the tool runtime frame those calls run in (`runtime/`), which
moved here from the retired `ora/agent/tools` package.

| Unit | What it holds |
|---|---|
| `runtime/` | The tool runtime frame: `ManipulationToolRuntime` (`core.py`, with the mixins `observe.py`, `grasp.py`, `skills.py`, `motion_submit.py`, `lifecycle.py` and what they share in `common.py`), `RobotToolRuntime` (`base.py`, `robot_tools.py`), the per-episode factory (`factory.py`), the toolkit dispatch (`dispatch.py`), the artifact publisher (`tool_data.py`), the input freezer (`inputs.py`) and the undo of a failed call's writes (`mutation_compensation.py`) |
| `adapter.py`, `observe.py`, `geometry.py`, `segment.py`, `motion.py`, `lifecycle.py`, `robot.py`, `state.py`, `visibility.py`, `query.py`, `numeric.py`, `failures.py` | What each tool's data interface reads through: trusted assembly, publication grants, the selected observation, motion configuration and proposal validation at the control edge, finish publication and the issue log, the `robot_runtime_v2` reads, the Supervisor state owner, robot history, `robot.query`'s interface, the evaluator binding, refusal projection |
| `recording.py`, `live.py`, `event_index.py`, `publication.py`, `execution.py`, `legacy_results.py`, `skill_feedback.py` | Toolkit I/O recording and the evidence around data calls, the event index, the immutable published-observation copy, the control-file boundary of the grasp tools, released confirmations, grasp stage feedback |
| `action_preview.py`, `backend_selection.py`, `observe_labels.py` | Dry runs on environment-planned backends; camera and grasp compiler selection and the closed result projection; the labels of a dual-arm observe's images |
| `compute/`, `finish/` | The restricted geometry evaluator (its worker runs in a process of its own); the `finish_episode` and `check_task` rules |
| `grasp/`, `perception/`, `reach/`, `motion_rules/`, `skills/`, `shared/` | What the adapters need: grasp proposers and compilers (the Panda top-down fallback family among them), the segmented-target store and localization, the segmentation backend protocols and the depth-prompt fallback segmenter, alignment and jaw-centre offsets, `move_pose`'s argument reading, refusals and observed clearance, `grasp_execute`'s stage loop, artifact I/O and typed refusals |

`grasp/arx_x5_gripper.py` is
the ARX X5 v5 descriptor's provenance source at its old path
`ora/agent/tools/grasp/`, which `ora/agent/composition/embodiment/moved_sources.py`
maps here while the file's bytes are the ones that moved.

Opt-in I/O recording: `ora/agent/toolkit/host/runtime/factory.build_tool_runtime(record_toolkit_io=True)`
installs `recording.py` around the existing clients, callbacks, observations and
submissions. Recording failures preserve the original result or exception and
write a best-effort gap; they never refuse or repeat the original call. A
recoverable recording gap stops only its parent; uncertain journal writes still
freeze that writer. `event_index.py` indexes appended parent proposals and
observation publications without rereading history. `publication.py` resolves
the immutable policy-observations copy by control/state identity even when the
current input mount is in the sandbox, and never expands the public BlobRef
allowlist. `skill_feedback.py` joins synchronous grasp stage interpretations
with the owner's action receipts for optional next-observation feedback, using
the released composite projector and owner-bound post-close views, without
creating a new action or changing any execution evidence.

## What belongs here, and what does not

Here: the trusted half of a toolkit call. Not here: a tool's own logic and what
it computes from public data (`../tools/`, `../lib/`), the data interface's
contract (`../interface/`).

## Dependency direction

The host imports the rest of the toolkit, `composition`, `providers`,
`model_services`, `evidence`, `contracts` and `error_classes` (rule L6).
`memory`, `code_mode`, `context`, `session_core`, `pi` and the orchestrator
import the host; `tools/`, `lib/` and `interface/` never do
(`tests/ora/agent/toolkit/tools/test_boundary.py`).

## Entry points

- `runtime.factory.build_tool_runtime`: the per-episode runtime the coordinator
  builds; `runtime.dispatch.install` and `runtime.dispatch.served`.
- `numeric.evaluate`: the restricted evaluator behind `compute` and
  `ora toolkit replay`; `python -m ora.agent.toolkit.host.compute.evaluator --worker`
  is its worker.
- `grasp.contact_graspnet_client.ContactGraspNetApiProposer`, which
  `ora/orchestrator/toolkit/services.py` builds for replay.
- `motion.joint_trajectory_surface`, `query.query_api`,
  `publication.published_manifest`: what the orchestrator, code mode and the
  Pi session read of the host.

## Documentation

- [Toolkit I/O](../../../../docs/reference/toolkit-io.md),
  [toolkit execution](../../../../docs/reference/toolkit-execution.md) and
  [the toolkit](../README.md)
- [Repository structure](../../../../docs/developer/repository-structure.md), rule L6
