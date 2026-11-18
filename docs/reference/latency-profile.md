# Latency profile

A simulation benchmark can pay for things on its control path that only serve people,
evaluation or evidence, because the simulated world waits for them. A real-time cell cannot:
the world keeps moving while the harness renders a video frame or writes a journal line. A
RunSpec's `latency_profile` names which of the two a run is, and the plan records the explicit
switches that profile resolved to (`ora/orchestrator/spec/latency.py`). The design note
[What a real-time cell turns off](../design/real-robot-cells.md#10-what-a-real-time-cell-turns-off)
classifies every measured cost; this page is the contract of the switches that exist.

## Profiles and switches

| Switch | `sim_benchmark` (default) | `realtime` | What it does | Changes what the model reads |
| --- | --- | --- | --- | --- |
| `execution_capture` | on | off | Records camera frames during motions (`runtime.execution_capture`, 10 fps by default) for people: `ora web`, the world view, `ora cameras`, the Viewer and `ora replay` video. No policy input and no verdict reads them. | No |
| `reach_reference` | on | off | Before a pose tool (`move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`) submits its move, it asks the planner's IK about the target for `compiled_pose.targets.<arm>.reach_along_request`. | Yes |
| `preview_images` | on | off | A dry run's record and a real call refused by its plan's verdict carry the rendered preview picture. | Yes |

`sim_benchmark` is today's behaviour: its plans resolve to the same runtime, tool
configuration and model-visible content as plans resolved before the switches existed. It
is the default for every RunSpec that names no profile. `ora play --env yam` names `realtime` (the station is a real-time
cell, and its simulated cell rehearses it); `ora play --latency-profile` sets either profile on
any free-play session.

## The RunSpec field

`latency_profile` is a top-level RunSpec field, either a profile name or an object that sets
switches one by one:

```yaml
latency_profile: realtime
```

```yaml
latency_profile: {profile: realtime, preview_images: true}   # an operator asked for pictures
```

A field other than `profile` and the three switches, an unknown profile or a switch that is
not a Boolean is refused. `runtime.execution_capture.enabled`, where the RunSpec names it,
sets the `execution_capture` switch: a free-play plan's 30 fps capture stays on under
`realtime`. Naming `execution_capture` in both places with different values is refused.
Under `realtime`, a capture setting that does not name `enabled` (for example `{fps: 5}`)
keeps its other fields and resolves with `enabled: false`.

The field is not under `runtime` for two reasons. RoboDojo's runtime field allowlist
(`validate_runtime_extension` in `ora/environment/benchmarks/robodojo.py`) is in a pinned
worker-kernel file, so a new runtime field would change the worker-kernel digest that RoboDojo's
qualification records bind. And the runtime block is part of `model_visible_sha256`'s settings, so a
runtime field on every plan would move the digest of every benchmark cell.

## The plan record

Every plan carries `latency_profile` (`ora.latency-profile.v1`):

| Field | Meaning |
| --- | --- |
| `profile` | `sim_benchmark` or `realtime` |
| `switches` | each switch's value for this run; `execution_capture` is what the resolved runtime records (`false` on a backend that records nothing unasked, such as Synthetic without a capture setting) |
| `set_by_spec` | the switches the RunSpec set itself rather than the profile |
| `model_facing_off` | the model-facing switches that are off |

A plan without the record (every plan resolved before it) is `sim_benchmark`.

`model_visible_sha256` adds `latency_model_facing_off` to its settings item only when that list
is non-empty, so a `sim_benchmark` plan keeps the digest it would have had without the record
and a plan that turns the reach reference or the pictures off has a different one.
`scripts/dev/compare-model-visible` therefore reports every benchmark cell unchanged.
The capture settings (`runtime.execution_capture`'s `enabled`, `fps` and `queue_capacity`) do
not enter the digest: a present capture config is compared as the recorder's default, so a plan
that only turns capture off, or records at 2 fps, has the digest of the same plan at 10 fps.

## What each switch changes at run time

**`execution_capture` off.** The worker renders no capture frame during a motion and writes
no `native/execution-capture/`. On RoboDojo a frame costs about 7.9 ms of main-thread wall time
per physics step at 10 fps (measured 2026-11-06, one lane alone on its card, 21 ms per 4 ms
physics step), so a motion runs about 37% faster. The live camera panels then have nothing to
show; recording for people on a real cell belongs to an independent process (the real YAM rig
already reads no extra camera frames while the arms can move, `backends/yam/capture.py`).
Not yet measured: whether RoboDojo's observation pixels depend on the capture's intermediate
renders (the observation synchronizes its own render products); a smoke compares the
observation image hashes of one seed with capture on and off before a realtime result is
compared with a benchmark one.

**`reach_reference` off.** The pose tools ask no IK question for the annotation, and each
target's `reach_along_request` is `{"status": "off", "reason": "this run asks no reach reference
before a move (latency profile)"}`. The released free-space jaw turn of an environment-planned
surface (`move_pose_jaw_symmetry_headroom_v1`) still asks the IK, since it chooses what is
submitted. The tool descriptions are unchanged, so they still describe the annotation. On
RoboDojo the reach check was about 2.2 s of the 4.8 s median before a move is proposed, mostly
because it is the first planner call after a new observation and pays the planner's full scene
rebuild; with the switch off that rebuild moves to the move's own first planner call
(`plan_path` or `contact_path`), so the switch alone saves the IK queries (about 0.45 s each),
not the rebuild. Prebuilding the scene is kernel work (below).

**`preview_images` off.** A dry run keeps every number of its record (plan, clearance,
segments, summary) and its `image` is `{"status": "off", "reason": "this run renders no preview
pictures (latency profile)"}`, with no `images_base64`; a refused dry run's record is its
refusal and that `image`. A real call refused by its plan's verdict carries no picture, and
nothing reads the cameras or asks the body FK for one.

## Evidence

- `plan.json` `latency_profile`, in every sealed run.
- Each tool call's recorded configuration (`native/toolkit-io/tool-inputs.jsonl`) carries
  `latency_switches` (the model-facing switches that are off) only when the plan turns one off,
  so a benchmark run's tool inputs are unchanged and a replay (`ora toolkit replay`, incident
  replay) runs a realtime call with the switches it ran with.
- `runtime.execution_capture` in the plan; no `native/execution-capture/` when it is off.

## Not switches yet

These costs are inside the RoboDojo worker kernel (`WORKER_KERNEL_PATHS`), so changing them
voids the RoboDojo qualification records and comes with re-qualification:

| Cost per 4 ms physics step or per move (2026-11-06) | Where | Plan |
| --- | --- | --- |
| Per-step journal `open("a")`/write/`close` of `controller/action-NNNN.jsonl`: 1.8 ms | `backends/robodojo/native_sequence.py`, `session.py` | keep each action's journal open, byte-identical output |
| Proprioception readback for the journal: 1.6 ms | the same per-step `record` callbacks | batch the readback |
| Control-queue deep copies: 1.1 ms | `native_sequence.py`, `bridge.py` | measured, not planned yet |
| Per-tick reward and official terminal check: 0.4 ms | `official_terminal.py` | stays in simulation: it decides the verdict; a real cell has none |
| Planner scene rebuild after each observation: most of the reach check's 2.2 s | `shared_motion.py`, planner worker | prebuild both arms' scenes when the observation is published; one cache entry per arm |
| Post-motion observation: about 0.8 s worker capture before the `stagnation_observation` event, then 0.5 s of publishing | `observation.py`, `episode/observation.py` | the stagnation record is a hash of that same observation's image, not a second observation; the worker's render synchronization is what costs |

PhysX (3.7 ms), the Isaac Lab actuator (1.5 ms) and the SDK control step (1.4 ms) are the
simulation itself.
