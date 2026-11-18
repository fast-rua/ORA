# Policy assets

This directory holds the policy assets this tree composes: prompt layers,
embodiment cards, descriptors and control profiles, environment context
profiles, ToolSurface manifests with their components, and the one baseline. It
is not a second Runtime and not a list of competing defaults. Every asset a
RunPlan selects is recorded there by path and sha256. A patch edits an asset in
place; a new version is only for an interface change (see
[Versions, patches and retirement](#versions-patches-and-retirement)).

## How one asset becomes active

```text
experiment config / RunSpec
  -> orchestrator validates omitted-field defaults
  -> policy_context and tool_surface profiles select exact resources
  -> composition resolves content and hashes
  -> expanded RunPlan records every selected identity
```

There is no implicit “highest version wins” rule. A file called `*-v3` is not
active merely because it has the largest suffix. Use `ora plan show` to inspect the
effective composition of a RunSpec.

## Directory ownership

| Directory | Content | Selected by |
|---|---|---|
| `core/` | Shared evidence/action-loop text: `evidence-action-loop-v8.md` (the default bundles), `-v2.md` (`ora local`), `thin-react-v1.md` (the Synthetic quickstart and the runtime_v1 ladder); `evidence-action-loop-v9.md` (the `toolkit_v6` bundle); code mode's core `code-policy-loop-v2.md` and `round-static-facts-v1.md` (after it where `native_slots_v4` sends that message), and `code-policy-loop-v3.md`, v9's code form | prompt bundle, `policy_context` or `policy.code_mode` |
| `prompt_bundles/` | Ordered prompt layers per backend (`current_v4/`, `local_cpu_v1/`), their schema and the episode-task templates (`episode-task-v2.md`, `episode-task-play-v2.md`, `tool-schema-delivery-v1.md`) | baseline instances, `ora local` and play |
| `planners/native/` | System and project layers: `policy-v5.md` (the default bundles), `policy-v3.md` (`ora local`), `project-v1.md` (the Synthetic quickstart and the runtime_v1 ladder); code mode's planner `code-mode-coding-v2.md` (`code-mode-coding-v3.md` for `prompt: v3`, which needs `library: extended`), `code-mode-trajectory-budget-v1.md` (toolkit v5, whose moves are joint trajectories), and with `prompt: v3` `code-mode-query-v1.md` and `code-mode-dry-run-v1.md` (what the surface serves, the contract's `script_reads`) | prompt bundle, `policy_context` or `policy.code_mode` |
| `embodiments/` | Robot descriptors, cards, wording reviews and control profiles, and the descriptor schemas (`descriptor-schema-v2.json`; `descriptor-schema-v1.json` for the v1 format the ARX X5 descriptor still uses) | embodiment/manual binding |
| `environments/` | Policy-safe environment boundaries, context profiles and task contracts | environment context profile |
| `tool_surfaces/` | Tool manifests, schemas and descriptions | `policy.tool_surface.profile` |
| `baselines/` | The provider-independent default composition | robot RunSpec default expansion |
| `context_schemes/` | Named context halves of the baselines, discovered by the `id` in each file ([context schemes](../../docs/reference/context-schemes.md)) | a baseline's `context_scheme`, `ora local`, `policy.context_scheme` |
| `context_stages/` | One parameter schema per context stage and `scheme.schema.json` for a v2 scheme's settings; Python validates v2 schemes with them and sends them to the agent core, which validates every stage it builds with them | v2 context schemes, `domain.context_stage_schemas` |

`embodiments/aloha-agilex/` (RoboTwin) holds one control profile,
`control-profile-v10.json`, which the RoboTwin bundle and the baseline bind, and
one hand-written card, `manual-v1.md`, which the Synthetic quickstart and the
runtime_v1 ladder bind; the default bundle uses the public-geometry card,
descriptor and wording review beside them. Control profiles v1-v9 and the
other numbered manuals were removed from this tree; their bytes remain in git
history and in the runs that bound them. This tree composes and executes
control profile v10 only (`OLDEST_COMPOSED_CONTROL_PROFILE`): a RunSpec or
RoboTwin session bound to v1-v9 is refused, and a recorded run bound to one is
evaluated at its source commit. The control profile carries a `source_binding`
that content-addresses the vendored RoboTwin planner chain
(`environment/backends/robotwin/vendored/`), so a planner or installer change
requires refreshing that binding through `./scripts/dev/refresh-vendored-pins`.
The other embodiment directories hold the public-geometry assets of their
backend (`arx-x5/` RoboDojo, `franka-panda/` LIBERO and `ora local`,
`panda-omron/` RoboCasa, `droid-franka-robotiq/` RoboLab) and
`synthetic-target-state/manual-v1.md` the Synthetic card.

Embodiment descriptors name the repository-local Runtime sources behind each
fact by path only (`local_path`, no `revision` or `sha256`); the RunPlan's
source commit identifies their bytes. An edit or a behavior-neutral split of a
cited source no longer touches the descriptor; only a renamed or removed source
needs its `local_path` updated, which the source audit reports. A descriptor
that runs have been sealed with keeps its bytes: its successor names the new
path, and `ora/agent/composition/embodiment/moved_sources.py` maps the old path,
with the sha256 of the bytes that moved, to the new one, so the old descriptor
still audits (and a binding to it still loads) while the file there has those
bytes. External sources (MJCF, URDF, USD, meshes, robot bindings) keep
`revision` and `sha256`.

## Naming assets by path; the lock

Assets this tree composes name each other by path only: a ToolSurface
component is `"observe.json"` beside its manifest, a context
profile's compiled descriptor and wording review are repository-relative
paths, a baseline catalog's `runtime.motion_profile` and
`runtime.control_profile` are paths, the RoboTwin registry names each
`semantic_review` by path, a RoboDojo registry entry is exactly
`{record, layer}`, a descriptor names its repository sources by path and a
wording review attests only the card text (`card_sha256`). None carries a
hand-written sha256 for another asset or a repository source.
`policy.lock` at the repository root holds one sha256 per tracked asset. A
loader reads only assets the lock lists and records the sha256 of the bytes it
read in the RunPlan; an inline `{path, sha256}` or `{ref, sha256}` is refused
([path-only asset references](../../docs/reference/configuration-and-evidence.md#path-only-asset-references)).

Hashes that remain inline are human-review pins of reviewed model-visible text
(`card_sha256` in profiles and wording reviews, task-contract layer-text
hashes), generated compiled card manifests and external identities (descriptor
external sources, task-contract source dependencies, model checkpoints,
simulator revisions, vendored-payload bindings). Frozen assets keep their old
inline pins: they compose only from their launch commit, and this tree refuses
them before loading.

To edit an asset:

1. Change the file, or add it and `git add` it.
2. Refer to it by path from the assets that use it. Their bytes do not change
   when it does.
3. Run `./scripts/ora lock` and commit `policy.lock` in the same
   commit. CI runs `lock --check`.
4. Run `scripts/dev/compare-model-visible <base> HEAD` to see what the model now
   receives.

`./scripts/ora lock --strip-pins` rewrites any composable asset that
still holds inline pins, for example after a merge from an older branch,
regenerates the compiled card manifest of any descriptor it rewrote, and then
regenerates the lock. `--strip-pins --check` only reports. A shared file is
named by every surface that uses it, so changing it changes all of them. Add a
layer when only one surface should change.

Shared Runtime code may read assets only through the composition layer under
`ora.agent.composition`. Experiment-specific prompts remain under
their experiment package and never become shared defaults by being copied here.

## Why historical names stay

Names such as `toolchain_b2_noservo_v1/`, `explicit_motion_v3/`,
`prompt_bundles/current_v4/` or the profile id
`robotwin_toolchain_b2_closed_noservo_v2` record the generation that introduced
an asset, not whether it is current or which experiment it belonged to. They
stay because they are identities. A resolved ToolSurface hashes its profile id,
manifest, provenance and component paths into `composition_sha256`; a RunPlan
records the profile id, every selected asset path with its sha256 and those
composition hashes, and its `plan_sha256` covers them; `policy.lock` and the
default-resolution expectations (`tests/data/default-expectations.json`) name
the same ids and paths. Renaming a profile or moving a component would change
those identities for unchanged model-visible text, so a fresh plan would no
longer match the sealed runs and qualifications that bound the old identity.
A name changes only with an interface change, as a new version that retires
its predecessor.

## Versions, patches and retirement

Decided 2026-10-31, after a cleanup that removed 53 dead description layers,
the RoboDojo toolkit legacy line and code-mode prompt v1:

- **Patches edit in place.** A wording fix, a corrected number, a new sentence
  or a changed default value edits the file that delivers it, not a new
  `-v<N+1>` sibling. Sealed runs keep what they recorded: a plan carries the
  effective tool definitions (`policy_composition.tool_surface.definitions`),
  the run keeps `compiled-policy/`, its context manifest and its provider
  requests, and the agent-core goldens replay a sealed run's recorded
  definitions (since PR #54). Evaluation, trace verification, export and the
  viewer do not read the tree's copy.
- **A new version is for an interface change only:** a tool added, removed or
  renamed, a schema a caller must adapt to, another action codec or context
  contract. The change that adds it retires the predecessor in the same PR
  (deletes it and moves its consumers), or records a dated retirement entry in
  `tests/data/policy-retention.json`.
- **Experiment variants live with the experiment.** Arms, comparison baselines
  and treatment surfaces go in the experiment's directory or pack, not in this
  default policy tree.
- **Nothing selectable is ownerless.** `tests/repository/test_policy_retention.py`
  lists every registered baseline, ToolSurface and context scheme that no
  default or maintained consumer reaches and fails unless
  `tests/data/policy-retention.json` gives it an owner, a reason and an expiry
  date, or once that date has passed.

Retired 2026-11-02, by human decision: the M0-M5 Runtime ladder (`ora ladder`,
`orchestrator/runtime_ladder/`), `robot_runtime_v1`, whose one consumer it was, and
the RoboTwin boundary prompt only the ladder read. 0 of 8,202 scanned sealed plans
name `robot_runtime_v1` or `robot_runtime_v2`. `robot_runtime_v2` stays all the same:
it serves Synthetic, the quickstart and `ora interactive`, which that count could
not see, because those plans hold no Synthetic run on any surface. Sealed runs need
neither: a plan carries its definitions, and evaluation of one that names no surface
gets the same five tools from `robot_runtime_v2`.

## Default baseline

New `native_api` RunSpecs for `libero_pro`, `robotwin`, `robocasa` or `robolab`
that omit `policy.tool_surface` select `native_full_explicit_motion_direct_v4`.
On `robodojo` they select `robodojo_toolkit_v7` instead
(`ENVIRONMENT_BASELINES` in `ora/agent/composition/baseline.py`, 2026-11-02):
the file-backed `robodojo_placement_toolkit_v6` surface (19 tools, moves as
joint trajectories, no `check_task`), the `prompt_bundles/toolkit_v7/` bundle
and the `context_keep_written` scheme (`context_keep_written_unaged_reasoning` on 2026-11-04 and 05; before that `context_rolling_tail_v2`, retired on 2026-11-05). The same selection can be made explicitly with
`policy.baseline: native_full_explicit_motion_direct_v4`. Its owner is
[`baselines/native-full-explicit-motion-direct-v4.json`](baselines/native-full-explicit-motion-direct-v4.json),
which names its context half, the scheme `context_lossless_v6`
([`context_schemes/lossless-v6.json`](context_schemes/lossless-v6.json),
management v6's stages in scheme format v2), instead of carrying it inline; `ora local` names the same
scheme. Its instances select:

- `libero_placement_direct_overhead_v1` on LIBERO/LIBERO-Pro (direct driver,
  17 tools, with `overhead_footprint_localization_v1`: `vision_ground` and
  `vision_segment` localizations add the footprint centre, view tilt and an
  overhead wrist pose), whose CuRobo planner plans in the public-depth
  collision world (`runtime.motion_planner.world_model:
  public_depth_voxel_v1`) that the surface's help describes;
- `robotwin_toolchain_b2_closed_noservo_v2` on RoboTwin (closed skill
  `grasp_execute` with probe IK counts, `align_plan` and the shared
  `wait_ticks`, 19 tools), under `demo_randomized` by default with
  `demo_clean` selectable;
- plus the derived `artifact.read` tool on every backend.

All three use `motion_action_v3`, causal `native_slots_v4` context (static
facts once per round), automatic current observation, `action_step_lossless_v4`
retention with exact provider continuation, the `prompt_bundles/current_v4/`
bundles (shared planner/core sources, environment-bound manuals and compiled
geometry cards), all current public cameras with measured-state and calibration
facts, same-credential provider recovery, Memory disabled, and the resident
segmentation and grasp prerequisites, unbound until an operator supplies
qualified service identities.

The Context catalog keeps the open visual step plus two completed steps of the
current episode. Its 24-image value is a soft L4 pressure trigger; the L5
watermark is an absolute 150,000 tokens, gated by exact provider usage.
Provider/model, endpoint, image capability, context window, task, seed,
budgets, resource assignment and live qualifications remain explicit.
Selecting the baseline is not permission to launch. Use `ora baselines --json`
to inspect its catalog and `ora plan show` or `ora plan context` for an exact
resolved task.

Baseline-defining overrides are rejected; custom treatments select an explicit
ToolSurface and context composition instead. Outside these robot defaults, the
fallback validation values are:

- tool surface: `robot_runtime_v2`;
- action codec: `target_state_v4`, unless the RunSpec names another;
- Memory: `disabled`;
- Native context profile outside the robot baseline: `native_minimal_v1`;
- provider/model, task, endpoint, context window and experiment treatment:
  always explicit.

## Registered surfaces

`agent/composition/surface/profile_catalog.py` registers every ToolSurface this tree
resolves:

| Profile | Role |
|---|---|
| `libero_placement_direct_overhead_v1`, `robotwin_toolchain_b2_closed_noservo_v2` | the default baseline's instances |
| `robodojo_placement_toolkit_v6` | RoboDojo's default (`robodojo_toolkit_v7`) |
| `robodojo_placement_toolkit_v5` | `robodojo_toolkit_v6`: v6 with `check_task`, kept so sealed plans resolve |
| `robot_runtime_v2` | Synthetic's surface: `ora run --env synthetic`, `ora interactive` and `ora play --env synthetic` name it (`orchestrator/spec/quickstart.py`). It is also the RunSpec fallback, which no maintained path reaches |
| `native_atomic_causal_v1` | the Synthetic Native test fixtures (`tests/support`) |
| `local_atomic_cpu_v1` | `ora local` |

Every surface is file-backed
([reference](../../docs/reference/toolkit-surface.md)): RoboDojo's
(`robodojo_placement_toolkit_v5/`, `v6/`) are `surface.json` and a surface
prompt, each tool's description `ora/agent/toolkit/tools/<name>/description.json`
beside its `tool.py`, read as written. The other backends' defaults were
layered manifests and were written out in that form (the tool that did it went
with the layered resolver on 2026-11-03): each keeps its profile id, its manifest's path
(`yam_direct_v1/` and its perception successor `yam_perception_v1/`, `robolab_direct_v1/`, `robocasa_mobile_v1/`,
`explicit_motion_v3/` for LIBERO, `toolchain_b2_noservo_v1/` for RoboTwin) and
the text it composed, with a `<tool>.json` beside the manifest wherever its
text is not the tool's own; Synthetic's (`robot_runtime_v2/`,
`native_atomic_causal_v1/`) and `ora local`'s (`local_atomic_cpu_v1/`) too. Edit
those files; there are no layers to edit.
The Franka Panda descriptor that cited the retired
`explicit_motion_v1/libero-definitions.json` keeps its bytes for the runs sealed
with it; its flat-surface successor cites the LIBERO manifest instead, and the
file is gone.

## Tool help contract and published limits

Every new ToolSurface declares `policy_tool_help_v1`. Its canonical definition
must tell the model when to use each tool, its actual effect, immediate and
deferred results, non-guarantees and error recovery; every nested input field
has a description and every tool has a public result schema. The resolver
enforces this contract before composition. See
[`docs/developer/authoring/tool-surface.md`](../../docs/developer/authoring/tool-surface.md).

The defaults' tool text is maintained in place in their tool files. It was
the published-limits layers (`published_limits_v1/<surface>-help.json` and
`-schemas.json`), the last description and schema layers of each layered
default, until the defaults were written out as files; git history keeps the
layers. `tests/data/published-limits.json` maps every number the delivered
text states to the owner the code enforces, and
`tests/ora/agent/composition/surface/test_published_limits.py` holds that text to
those owners
([reference](../../docs/reference/public-interfaces.md#published-limits-and-stated-refusals)).
The generator (`ora dev description-edits`) was retired with the dead layers it
read, and the review records of its sentence edits with it; git history keeps
them.

## End-effector speed

On each pacing default (LIBERO, RoboTwin, RoboCasa365, RoboLab), which
declares `end_effector_speed_v1`, `move_pose`, `move_to`, `rotate_wrist` and
`rotate_pitch` take the optional `speed`, `speed_m_s` and `slow_approaching` of
RoboDojo's toolkit `move_pose`, beside `targets` on RoboTwin's dual-arm tools
(`tests/ora/agent/toolkit/host/test_motion_speed_arguments.py`). The fields and
their help are in each surface's pose-tool files; they were the
`end_effector_speed_v1/<backend>-schemas.json` layers until the defaults were
written out as files ([motion speed](../../docs/reference/motion-speed.md)).

## Finish contract v2 capabilities

`policy_finish_outcomes_v2` and `policy_abort_infrastructure_round_v2`
(`agent/composition/schemas/finish_outcomes.py`) are declared by the v2 default
surfaces of RoboTwin, LIBERO and RoboDojo. They rewrite `finish_episode` to
`success` and `give_up` (reason required, give_up `category` from
`unreachable`, `object_lost`, `scene_damaged`, `repeated_failure`,
`other`), add `abort` admitted only on this round's recorded infrastructure
failure, and rewrite the `recover_pose` clauses that named the failure exit.
Every word of the new policy-visible text was generated by that module, with
the numbers taken from the constants the runtime enforces, and is now in each
default's `finish_episode.json`; the contract and
records are in the
[interface reference](../../docs/reference/public-interfaces.md).

## Closed-skills ports on the RoboTwin and LIBERO defaults

The RoboTwin and LIBERO defaults declare the RoboDojo closed-skills features
ported to them: `move_pose_reach_along_request_v2` and
`pose_primitive_reach_along_request_v2` (the shared planner's IK in its
depth world on both), `reach_nearest_orientation_v2`, `observed_clearance_reference_v2`,
`pose_orientation_from_world_axes_v1`, `grounding_candidate_depth_profile_v1`
(LIBERO) and `grasp_plan_ik_reachability_annotation_v1`.
`tests/ora/agent/composition/surface/test_closed_skills_ports_surfaces.py`
checks them. Their policy-visible text is in the two surfaces' tool files;
it was first written as the `closed_skills_ports_v1/` appends, which the
published-limits layers overwrote and which were deleted on 2026-10-31. `observed_extent_scene_review_v1`
needs no text.

## Compute usage and routing

The default surfaces share Compute v2 input/output selection, numeric helpers
and executable examples; the `compute` description and the calculation routing
of the defaults are in their tool files (`yam_direct_v1/compute.json` on
YAM, the tool's own `compute/description.json` elsewhere), and backend overlays own the planner
handoff examples. See the [usage guide](../../docs/usage/compute-usage.md).

## Local CPU ORA composition

The default local ORA entry selects `local_atomic_cpu_v1` and the
`local_cpu_v1/libero-panda-native-api.yaml` bundle explicitly through
`agent/composition/ora_local.py`. It reuses the atomic v3 OSC action compiler
with nine tools and no GPU planner, VLA or resident perception service.
`control-local-cpu-v1.json`, `manual-local-cpu-v1.md` and the
`libero_pro_local_cpu_v1` context profile bind the reviewed Panda/controller
facts; the legacy atomic manifests and the shared server baseline stay byte
stable. Context settings come from the current maintained baseline; local
provider, source, host and task inputs are checked by normal resolution and
preflight. See the [current local contract](../../docs/reference/public-interfaces.md#operator-and-composition)
and [source compatibility audit](../../docs/reference/ora-local-cpu-compatibility.md).

`baselines/robodojo-toolkit-v7.json`, RoboDojo's default, is
`robodojo_placement_toolkit_v6` (tool_output receipts, no `check_task`) with the
`toolkit_v7` bundle and the `context_keep_written` scheme, which `robodojo_toolkit_v6` names too.
It also names the code mode a RunSpec gets when it names none (`"code_mode"`: the extended library, prompt v3 and
ORA's source, 2026-11-05); `robodojo_toolkit_v6` names none and stays tool calls, the baseline of its sealed runs.
`native-full-explicit-motion-direct-v4.json` names it for code mode as well
(`code_mode_context_scheme`), beside `context_lossless_v6` for tool calls.
`baselines/robodojo-toolkit-v6.json` is `robodojo_placement_toolkit_v5` with its
`check_task` grants and the `toolkit_v6` bundle. The toolkit baselines v1, v2,
v4, v5, matched-context-v1, legacy-visible-v1 and legacy-replay-v1 were retired
on 2026-10-31, and v3 (the layered `robodojo_placement_toolkit_v2` with the
`toolkit_v3` bundle, the `boundary-toolkit-v2` environment boundary and the
`context_lossless_v7` context) on 2026-11-01; sealed runs keep their recorded
definitions.
