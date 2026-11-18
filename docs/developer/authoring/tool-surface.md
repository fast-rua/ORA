# ToolSurface authoring and model-visible help

Every ToolSurface is one manifest and the tool files it names, read as written
([file-backed surfaces](../../reference/toolkit-surface.md)); the help contract
below applies to those files. The layered manifests, their components and the
capability binders that rewrote composed text were retired on 2026-11-03, after
every surface had been written out as files with the same text.

ORA treats every policy-visible tool definition as executable model
documentation. A fixed tool catalog does not need a separate `help` tool: the
canonical description, input schema, public result schema and structured error
must together tell the policy when the tool is useful, what it changes, what it
returns and what its success cannot prove.

This guide is the required authoring contract for every new ToolSurface. The
source of truth remains the resolved ToolSurface, whose RunPlan records the
hash of every component it read; `CLAUDE.md`,
`AGENTS.md`, experiment prompts and transport adapters must not maintain a
second copy of individual tool syntax.

## Required model-visible description

Edit a surface's help in place, in the file that holds it, and introduce a new
version only for an interface change (see "Patches, versions and retirement"
below). A manifest lists each tool once in `tools`, in delivery order; a
duplicate or a name without a tool file fails resolution.

Every tool on a surface declaring the `policy_tool_help_v1` capability uses
these exact sections, in this order:

```text
<One-sentence purpose.>
Use when: <decision condition and useful stage>.
Effect: <read-only, world mutation, lifecycle, and persisted local state>.
Returns: <immediate public fields/images and any later receipt/observation>.
Does not: <important non-guarantees and forbidden inference>.
On error: <evidence-preserving correction or recovery>.
```

The purpose and `Use when` text must distinguish the tool from adjacent tools.
Review the final resolved definition and a captured provider projection when
available, rather than an unselected base file. Review producer/consumer pairs
together: a producer must name an available downstream tool, explain when to
choose it over an atomic alternative, and show a schema-valid complete handoff.
Keep trigger examples distinct from argument examples. Do not make every tool
mandatory; copying an existing valid value and an unnecessary repeated query
are useful negative selection examples. Apply shared rules once and bind the
actual backend cameras, frames, measurements and optional capabilities.
`Effect` must describe actual semantics rather than its name: a read-only tool
may persist an observation-bound measurement record, while a mutation tool
usually returns `submitted` before the Environment has executed it. `Returns`
must separate the immediate tool result from a later Supervisor receipt and
fresh observation. `Does not` carries authority boundaries such as “not object
ground truth,” “not reachability,” and “not task success.” `On error` must give
the policy a valid next step without teaching a benchmark solution.

Do not put general ReAct advice, task strategy, simulator-private facts,
successful trajectories or provider-specific syntax in a tool description.
Those belong to the core policy, episode task, Environment/embodiment contract
or provider adapter.

## Input schema contract

The input schema is the tool's machine-readable `--help`:

- the root is an object with `additionalProperties: false`;
- every property, including nested object properties, has a non-empty
  `description` stating semantics, units, frame and omission/default behavior;
- closed vocabularies use `enum`; do not advertise a free-form string and rely
  on a Runtime error to reveal the accepted values;
- bounded numbers and collections declare `minimum`, `maximum`, `minItems` and
  `maxItems` whenever the Runtime enforces them;
- tuple order is explicit, for example `[x,y,z]`, `[x,y,z,w]` or `[u,v]`;
- required fields are listed exactly, and mutually exclusive shapes use a
  schema branch rather than a hidden mode switch;
- on a surface declaring `optional_argument_null_as_omitted_v1` every
  optional member at every depth also accepts null, meaning omitted, and the
  delivered tool is one flat object; the surface's files and delivery carry it and the
  gateways drop the null, so do not hand-edit `"null"` into a tool file, give
  an optional member a null meaning of its own there, or rely on a root union
  for exclusivity (state it in the description and enforce it in the handler);
- no policy parameter may expose privileged state or an unenforced physical
  guarantee.

Schema and Runtime validation must accept the same values. Tightening only the
schema or only the handler is a contract defect. Composition refuses an input
keyword the Runtime validator does not enforce (`lint_input_schema`); a generic
schema mismatch is already reported with its field, rule and bound, so a
handler need not repeat a check the schema states.

## Public result and error contract

Every advertised tool owns one public result schema. The canonical ToolRuntime
validates the text/JSON result after private image payloads and Runtime
directives are removed. This prevents accidental field drift or privileged
data leakage. Image and other multimodal blocks remain separate artifacts; the
description states when they are returned.

Provider protocols do not all expose an output-schema field, so the
model-visible `Returns` sentence is mandatory even when a Runtime result schema
exists. Managed MCP, Native Anthropic and Native OpenAI projections must all
derive their name, description and input schema from the same resolved
ToolSurface.

Errors use the shared structured ToolError boundary
(`agent/error_classes.py`): stable `code`, concise message, `retryable`,
`retry_requires_changed_arguments` when known, and only relevant `field`,
`received` or `constraints`. `retryable` means a later call of this tool can
still succeed (after a corrected request or a changed public state); it is not
a resend instruction. An argument refusal code (`ARGUMENT_REFUSAL_CODES`)
additionally states that the unchanged request is refused again; pass
`retry_requires_changed_arguments` explicitly only when a raising site knows
better than its code. Express a mutually exclusive or unknown-argument rule with
the shared `require_*` helpers, which report argument names and the violated
rule, never values. An error should enumerate a closed choice or current bound
when that helps the policy correct the call. It must not reveal object
identities, hidden predicates, exact privileged contacts or simulator
internals. The gateway, not the tool, appends the repeated-identical-failure
count; see the
[structured error reference](../../reference/public-interfaces.md#structured-tool-errors-and-repeated-identical-requests).

## Effect, freshness and authority audit

For each tool, review all of the following before activation:

| Question | Required evidence |
|---|---|
| When should the policy select it? | Distinct `Use when` text and at least one selection test or fixture |
| What state can it change? | Manifest mutation/lifecycle/observation classification agrees with ToolRuntime dispatch |
| Which observation binds the call? | Current/stale behavior and artifact identity are explicit |
| What is returned immediately? | Public result schema and multimodal block contract |
| What arrives later? | Receipt/fresh-observation cadence for mutations |
| What cannot be inferred? | Explicit non-guarantees and authority owner |
| How does failure recover? | Structured error and valid retry/replan/stop path |
| Is transport parity preserved? | Resolved canonical definitions match the Native provider projection |

Proposal-producing perception tools must also document common
**successful-but-wrong** outcomes. Backend success only means that a valid
candidate or mask was produced. Text grounding is a recall-oriented candidate
set: rank and score order candidates within one call but do not certify rank
0. Point segmentation treats a click as a surface prompt rather than a
semantic label; a boundary, shadow or background click can validly return the
table, robot or neighbouring object. Help must identify the evidence to inspect
(overlay, prompt cross, box, mask area/shape and depth), the rejection rule and
a valid farther-interior/complementary-view retry. Dynamic-camera help must
also say that pixels and calibration are observation-bound: after a wrist
moves, old wrist pixels cannot be reused as current hand-eye geometry.

## Patches, versions and retirement

Since 2026-10-31 a surface's text is edited in place:

- **A patch edits the file that delivers the text.** A wording fix, a new
  sentence, a corrected number or a schema description changes the tool file
  the model reads it from (see "Every number in help has an owner"), not a new
  sibling file or profile. Sealed runs do not need the old bytes: a plan records the effective
  `policy_composition.tool_surface.definitions`, the run keeps
  `compiled-policy/` and its provider requests, and the agent-core goldens
  replay a sealed run's recorded definitions (since PR #54).
- **A new version is for an interface change only:** a tool added, removed or
  renamed, an input or result schema a caller has to adapt to, or another
  action codec or context contract. The change that adds it retires its
  predecessor in the same PR (deletes it and moves its consumers), or records a
  dated entry for it in `tests/data/policy-retention.json`.
- **Experiment variants live with the experiment.** An arm's surface, baseline,
  prompt or context scheme goes in its experiment directory or pack, not in
  `ora/policy/`. A selectable asset no default or maintained consumer reaches
  needs an owner, a reason and an expiry date in the retention registry;
  `tests/repository/test_policy_retention.py` fails without one and once the
  date has passed.

## Adding or changing a tool

1. Work on the owning feature/experiment branch. Change a maintained surface
   in place (see above); add a new profile only for an interface change, and
   retire its predecessor in the same change.
2. Add the tool once to the canonical ToolSurface and classify its effect in
   the manifest. Update the canonical definitions used by Native.
3. Write the five-section description and fully documented closed input schema.
4. Add the public result schema and validate every successful result path,
   including unavailable/pending variants and multimodal-field removal.
5. Add structured invalid, stale, unavailable and terminal-path tests. For a
   mutation, test proposal → Supervisor execution → receipt → fresh
   observation; tool submission alone is insufficient.
6. Check Native projection for both provider protocols. Bind exact definitions
   and hashes in the RunPlan.
7. Declare `policy_tool_help_v1`. Resolution fails if any section, parameter
   description or result schema is missing. Source-frozen profiles are listed
   explicitly as legacy exemptions; a new catalog entry cannot inherit that
   status implicitly.
8. Update the owning current reference, policy README and tests index as
   appropriate. Run the focused tests and the locked full suite before merging.
9. Keep the declared set and the dispatch set joined. A tool named in a
   profile's `tools` must be a toolkit tool, because `allowed_tools` defaults
   to every declared tool and the loader validates a manifest only against
   itself. Every registered surface declares `toolkit_only_v1`, so the toolkit
   serves it (`dispatch.served`): `dispatch.install`
   (`agent/toolkit/host/runtime/dispatch.py`) binds each advertised name to its
   tool under `agent/toolkit/tools/<name>/` and refuses a surface that names a
   tool the toolset does not have.
   `tests/ora/agent/composition/surface/test_declared_tool_handler_join.py` checks both across the whole
   catalog.

### Retiring a tool

Retiring a tool means no longer declaring it. The rules:

- **Undeclare it in the manifest.** Remove the tool from `tools`, and its file
  from `tool_descriptions`.
- **Delete a tool only when nothing declares it.** The loader never
  consults the toolset, so a tool still advertised without its directory
  resolves and composes normally and then fails when the runtime is built:
  `dispatch.install` (`agent/toolkit/host/runtime/dispatch.py`) refuses the surface. Run
  `tests/ora/agent/composition/surface/test_declared_tool_handler_join.py` before the
  deletion, not after; it names every profile that would still advertise the
  tool. If that set is not empty, stop, or retire those profiles as a
  separate, explicitly approved change.
- **Keeping bytes is not keeping a profile runnable.** Preserving a frozen
  manifest keeps its evidence identity valid; it does not keep the profile
  runnable without the tool. A retirement that relies on the first must say
  plainly that it gives up the second.
- **Retire the capabilities only the tool reads, and keep the rest.** Decide
  from the `supports(...)` call sites, not from a capability's name: a name can
  read like the retired tool's and still gate other tools, and one that reads
  general can be set only by the retired tool.
- **Deliver what the tool delivered first.** A quantity only the retired tool
  returned must reach the policy through a surviving tool before the tool goes.
- **Leave no mentions behind.** Check tool help, prompt layers, environment
  profiles and task contracts for the tool's name in either spelling;
  `tests/ora/agent/composition/surface/test_closed_skill_surface_contract.py` does this for
  the closed-skills surfaces.

A tool's description, input schema and result schema are the tool's own
`ora/agent/toolkit/tools/<name>/description.json`, or the surface's own file named
under `tool_descriptions` where this surface's text differs. Nothing is layered
or rewritten at resolution. Editing a tool's own `description.json` changes every
surface that uses it; text only one surface should carry goes in that surface's
file. A surface's file may leave out `resultSchema`, so the tool's own applies,
or set it to `null` for a tool with no public result schema (`robot_runtime_v2`
and Synthetic Native declare none). A tool name is snake case or dotted segments
of it (`observation.dump_state`). A new profile generation is for a changed tool
*set* or a changed identity, not for a changed schema or text.

When a shared runtime result changes shape, gate it on a capability instead of
replacing it. Every frozen surface that publishes the old result schema is the
identity of a sealed run and will reject the new shape on its first call, so the
runtime must be able to serve both: publish a second capability, keep the
released implementation reachable, select between them from the declared
capability alone, and prove the old path still returns the released bytes rather
than assuming the extraction preserved them. Record such evidence from a fixture
with no numerical ties: a fixture with square masks gives two equal in-plane
eigenvalues, so `principal_axis_world` becomes whichever axis the host's LAPACK
returns and the golden holds only on the host that recorded it.

Files are named by path only. Each `tool_descriptions` value is a path relative
to the manifest's directory, and `view_config.public_geometry_descriptor` is a
repository-relative path:

```json
"tool_descriptions": {
  "observe": "observe.json"
},
"view_config": {
  "public_geometry_descriptor": "ora/policy/embodiments/aloha-agilex/descriptor-public-geometry-v5-session-split-v1.json"
}
```

Do not write a sha256 beside a file: the loader refuses
`{"ref": ..., "sha256": ...}`. Integrity comes from `policy.lock`. The loader
reads each file only if the lock lists it, hashes the bytes it read, and
records `{slot, ref, sha256}` in the resolved surface and the RunPlan
([path-only asset references](../../reference/configuration-and-evidence.md#path-only-asset-references)).

To edit or add a tool file:

1. Edit the file, or create it and `git add` it.
2. Name it by path in the manifest.
3. Run `./scripts/ora lock` and commit `policy.lock` with the change.
   CI fails on a stale lock.
4. Run `scripts/dev/compare-model-visible <base> HEAD` to see which delivered tool
   definitions moved.

Changing a tool name, description, schema, public result, effect or error
meaning is a policy behavior change. It requires a new hash-bound composition
and cannot be retroactively attributed to an earlier run.


## Restricted geometry evaluator (`compute`)

`compute` is a read-only pure function over already-delivered numbers. Every
default surface declares it ([compute usage](../../usage/compute-usage.md)). It is recorded
here because its result shape is dictated by contracts this guide owns, and
because two of those contracts protect run validity rather than one call.

**It is not a substitute for delivery.** A quantity the Harness can compute
*without knowing the policy's intent* must be delivered by its owning tool;
`compute` is only for combinations the policy must first decide in order to
name. The review test: if a `compute` usage could be precomputed by a function
that does not know the policy's intent, it is a delivery gap — fix the delivery,
do not widen the evaluator. Without this line the accepted-call table grows
without bound.

Authoring requirements specific to this shape:

- **Fixed result keys.** `evidence/trace_privacy.py::_forbidden_path` scans key
  paths, so a `{identifier: value}` mapping lets a block that binds `contact`
  invalidate the run and block sealing. The result is
  `{"outputs": [{"name", "value", "kind"}]}` and the policy's identifier appears
  only as a value.
- **Never echo the source block.** `_forbidden_path` and
  `context/causal_value.py::_causal_projection_violations` both `json.loads`
  strings that begin with `{` or `[`. A returned source string could therefore
  be opened as an object carrying a forbidden key.
- **Name skipped bindings.** A non-numeric intermediate is reported in
  `intermediates_not_returned`, never dropped silently; silent omission is the
  defect class this unit exists to remove.
- **Declare every returned property at the result schema's top level.**
  `native/session.py::_declared_result_properties` projects only declared
  top-level properties, and drops the rest without an error.
- **Fit the decision-critical values in 2048 bytes at the top level**, because
  an externalised result is previewed by a byte prefix.
- **Every policy-fixable failure is `invalid_arguments`.** Syntax, rejected
  construct, unknown name, malformed reference, timeout, memory ceiling and
  oversized result all carry `field`, `received` and the violated bound.
  `internal_tool_error` is reserved for evaluator faults the policy cannot fix.
- **`Use when:` must require batching.** One call evaluates every quantity that
  step needs. Per-quantity round trips cost more than the hand derivation they
  replace, because each call re-reads the whole cached context.

Effect classification needs no code: a tool absent from the manifest's
`mutation_tools` and `lifecycle_tools` is machine-provably `read_only` under
`native_atomic_causal`, and a handler that returns no `_runtime.directive`
consumes a model call without consuming a robot decision.

### Enforced boundaries the kernel learned the hard way

- **Attributes are an allowlist, not a dunder denylist.** Layer 2 withholds
  `numpy.load` and `numpy.fromfile` by never exposing a module object, but
  `ndarray.tofile`, `ndarray.dump` and `ndarray.ctypes` are the same capability
  reached from a value the block already produced, and
  `Rotation.identity().random()` reaches a classmethod through an instance.
  Rejecting only names beginning with `_` left all of those reachable, and
  randomness in particular contradicts the receipt being re-executable offline.
  Empty builtins is not a substitute: numpy happens to fail the file writers
  with `KeyError: '__import__'` from its own internal import, which is an
  accident of the dependency and reaches the policy as a message it cannot act
  on. The allowlist is derived from the callable tables plus the numeric
  conversions of an already-produced value, so a table entry cannot be added and
  left unreachable.
- **Provenance belongs under `_runtime`, and the placement is forced.** `schema`
  and `sha256` are both in `context/causal_value.py::_CAUSAL_FORBIDDEN_KEYS`, so a
  receipt left in the visible half fails the causal projection audit.
  `native/session.py` pops `_runtime` before building the tool message and
  `skills/receipt.py` excludes it from the measured body, so the receipt is
  recorded as evidence without being shown to the model or charged against
  delivery. Audit a result the way session.py builds it; auditing the raw
  payload tests a string that is never put on the wire.
- **A reference form must not be mandatory.** A binding is either a bare value
  or `{"value": ..., "from": "<origin>"}`. Requiring the annotated form would
  stop the policy bringing its own constant, which is legitimate and common. The
  receipt records each input's declared origin and digest beside the digest of
  the block, which is what an offline recomputation needs; the block itself is
  already recorded verbatim in the tool call, so the receipt binds it rather
  than echoing it.
- **A tool that owns a subprocess needs a teardown hook that exists.** Neither
  `ManipulationToolRuntime` nor `RobotToolRuntime` has one and nothing calls
  `close()`, so a per-session worker leaks one process per episode. Until a
  session teardown hook exists, bind the process-wide instance closed at
  interpreter exit; per-call freshness comes from the per-request fork into a new
  empty namespace, not from the identity of the parent.

## Planning-time IK selection in `grasp_plan`

`grasp_plan` on the live dual-arm surfaces checks its whole compiled pool (up to
32) with the backend's internal read-only IK check before the display cap and
orders reachable, then within 30 degrees of vertical, then backend order. The
check is deliberately not a policy tool: no surface declares `reach_check`
and this tree has no handler for it, so the policy
sees only the annotation. Never drop a proposal below the cap and never block it on the
annotation: an unavailable IK bridge keeps the backend order and says
`status: unavailable`. The help must call the annotation an IK reference at
planning time, not a guarantee, and must not name the undeclared tool. Nor may
it keep the unannotated surfaces' "Reachability is delegated to move_pose or
grasp_execute; this call does not run a motion probe" beside that paragraph:
the Dojo surface replaced it with a pointer to the IK selection paragraph, and `max_candidates` names the IK-ordered pool. A grasp
result field that depends on a backend's refusal vocabulary is derived from the
closed per-backend set in `contracts/reach_check.py` or, for RoboTwin,
`contracts/robotwin_reach.py`, not restated.

LIBERO's annotation was written by a sibling of the released binder, keyed on
its identity capability `libero_pro_rgbd_v1`, so the surfaces that published the
released annotation kept their bytes; the two differ only in the reason enum
(`tests/ora/environment/backends/libero_pro/test_libero_reach_check.py`). Its help also
states that this backend's check is not `move_pose`'s admission and that their
disagreement has not been measured, instead of reusing another backend's figures.

Publish the member that will execute. An annotation that names a better member
while the pose fields keep another is a pose the policy, the copyable
`move_pose` target and `grasp_execute` will all run anyway: one run annotated
`member: exact` on a wall pinch whose compiled orientation solved 0.14 rad from
the X5 joint6 stop, and the precommit kept it because it was admitted. Under
`grasp_plan_jaw_symmetry_headroom_v1` the selection rewrites `pregrasp_pose` and
`contact_pose` to the chosen member's exact quaternions and says so in
`ik_reachability.jaw_symmetry`. A derived quantity a row does not carry (the
X5 joint6 margin) is published only with the argument that isolates it from
the row's own numbers, and is null when that argument does not hold; one that
would need a guessed IK branch (an elbow-extension margin) is left out rather
than estimated.

## Help that says what the handler does

Read a sentence against the handler before shipping it; an audit of the
current surfaces found these defect classes on the current surfaces
([corrections](../../reference/public-interfaces.md#tool-description-consistency)):

- **A per-entry outcome is not a refusal.** When a handler reports a problem
  inside a successful result (`geometry_query` points `valid: false`, a
  `mark_point` batch entry with `error_code: inconsistent_mark`), the help says
  which entry fails and what the call still returns; "refused" or "fails
  explicitly" belong only to errors that end the call.
- **State a default as its value.** On a surface declaring
  `optional_argument_null_as_omitted_v1` write "default 3", not "omit for
  three": null and omission both select it, and the composed "Null means
  omitted." then needs no second step. Advice that assumes another starting
  value ("raise it from one") contradicts the default and is task guidance.
- **One owner for a rule.** A selection rule that belongs to one tool lives in
  that tool's `Use when`; another tool points to it in one sentence rather than
  repeating it. Keep a repetition only when the tool needs it to be
  self-contained: its own result fields, a producer's complete handoff, its
  Returns section, or a pointer to the single owner of shared numbers (the
  workspace bounds live only in `move_pose`).
- **Capability-gated behaviour gates its wording.** A sentence true without a
  capability (no IK in `grasp_plan`, a failure exit in `finish_episode`) must
  not survive on a surface that declares it.

Correct one surface in its own tool file. A sentence in a tool's own
`description.json` reaches every surface that uses that file, so when only one
surface should change, give it its own file under `tool_descriptions`.

## A feature lands on every backend's default

A Harness behaviour that is not about one backend's identity, embodiment or
mechanism is declared on all three defaults (RoboTwin, LIBERO-Pro, RoboDojo).
`tests/data/backend-parity.json` classifies every capability and tool a default
declares, and `tests/ora/agent/composition/surface/test_backend_parity.py` (always selected) refuses:

- an unclassified name;
- a `generic` or `generic_needs_backend_code` name missing from a default
  without an exception that cites a decision record containing the exception
  id; a missing backend hook is named (`missing_hook`: a file that `defines` a
  function, a constant that `contains` a value, or a module glob that
  `mentions` an identifier) with its follow-up unit;
- an exception that has expired: the backend declares the name, or the hook it
  waited for now exists -- so whoever writes the backend code also declares the
  capability;
- a `backend_specific` name declared outside its `applies_to`;
- a default swap that drops a name of the default it replaced without a
  recorded retirement (`previous_defaults`, updated in the swap itself).

A successor capability covers its predecessor (`supersedes`), and a default may
not declare both. When a port is written for one backend first, keep it
capability-gated and add its exception in the same change, so the next default
swap cannot silently leave it behind.

## Every number in help has an owner

An enforced number that a tool's help states — a bound the handler refuses
past, a routing threshold, a fixed parameter such as a lift height or a tick
length — is one row of `tests/data/published-limits.json`. The row
names the owner the code enforces: a constant, a profile value by JSON pointer,
a source literal matched exactly once, or a value derived from those. It also
lists each sentence that states the number, per surface.
`tests/ora/agent/composition/surface/test_published_limits.py` composes the current defaults
and compares every stated number in the delivered text with its owner
after unit conversion. A census refuses any digit in a tool's composed text that
neither a row nor an informational entry covers
([reference](../../reference/public-interfaces.md#published-limits-and-stated-refusals)).

- Add the row in the change that adds the number. A measured figure (a
  latency, a median distance, a free-air timing) or an illustrative example is
  an `informational` entry with its reason, never a limit.
- A number no repository code enforces has no owner, so the help may not
  state it. An example is the RoboDojo SDK's IK seed count and residual
  tolerances, which are read from the SDK at run time.
- State refusals as rules. `On error:` opens with `Refused:` and lists:
  - what the handler refuses before anything is submitted, with the table's
    numbers;
  - what the backend rejects before any controller write.

  It never says what to try instead. A per-entry failure or a dropped
  candidate is not a refusal and says so.
- Edit the text the capability binders wrote (the compute grammar, ARX
  controller feedback, the abort class list and the workspace JSON) where it now
  is, in the surfaces' tool files.
- Edit a default surface's tool files in place (`<tool>.json` beside its
  manifest, or the tool's own `description.json` where the surface has none).
  They hold the text the published-limits layers
  (`published_limits_v1/<surface>-help.json` and `-schemas.json`) composed
  until the defaults were written out as files; those layers were generated
  from review records until 2026-10-31.

Phase 2 (not built) generates these sentences from the same owners, so a
changed constant changes the help in the same commit.

## Publishing a capability the runtime already has

Before writing a tool, check whether the handler exists and only the dispatch or
the declaration is missing. `RobotToolRuntime._geometry` has served world-to-pixel
projection, pixel-to-world back-projection and region depth all along, declared
as `geometry.query` by `robot_runtime_v1`, `robot_runtime_v2`,
`robot_runtime_v3`, `robot_runtime_native_prompt_v2` and
`robotwin_dual_layer_v1` -- five of the 89 registered profiles. (Three of 31
*directories*: `robot_runtime_v2/` registers three profiles. Count the catalog,
not the tree.) `ManipulationToolRuntime`
inherits the method but overrides `invoke` without delegating, so every surface
built on it could not dispatch the name however it was spelled.

The measurable consequence is the reason this section exists. Across 177 sealed
runs, 51 (29%) show the policy deriving `u = cx + fx * x / z` by hand against
intrinsics it read off calibration, and those are precisely the runs whose
surface omits the tool; the five runs that had it called it instead. A
capability gap and a publication gap look identical from the trace and are
fixed differently, so establish which one you have before building anything.

New declarations use the snake_case name, `geometry_query`. The three surfaces
that already froze `geometry.query` keep that spelling: a frozen surface is
evidence of what a past run was shown, and re-spelling it would rewrite that.

Both active `*_placement_explicit_motion_v3` surfaces now publish it, through an
`additional_definitions` component plus one `result_order` component each, so no
frozen catalog is touched. Two authoring decisions there are general.

**A new name is a new contract, so inherit the shape rather than the defects.**
The frozen input schema is a top-level `oneOf` whose branches carry `required`
alone. `agent/session_core/gateway.py::_openai_parameters` flattens a root union only
when every branch carries `properties`, so this one returns verbatim and reaches
the provider — and `validate_schema_instance` then answers a missing operation
with "must match exactly one branch", which is run H6-013. The published schema
declares the three operations as sibling properties with `required: ["camera"]`
and leaves exclusivity to the handler, which already raises a structured
`invalid_arguments` naming all three fields. The union enforced nothing the
handler does not, and cost the diagnosis. This is the same repair
`flat_placement_arguments_v1` made; reaching for `anyOf` instead would restore
the union under a different keyword. Because the new key is distinct, the three
frozen surfaces keep their `oneOf` and their bytes — re-spelling is what would
have made the change expensive, not flattening.

**A released result schema is proof the shape was accepted, not that it is
closed.** The frozen one declares `region_result` as a bare `{"type": "object"}`
and `points` as a bare array, which cannot catch field drift the contract exists
to catch. The published schema closes all three branches, with each branch's
`required` set as the intersection of the handler's own variants — a projected
point behind the camera returns no `pixel_uv` at all. A tightened result schema
is enforced on live calls, so it has to be exercised against real handler output
before it ships rather than asserted: `tests/ora/agent/toolkit/host/runtime/test_geometry_query_publication.py`
drives all three operations and both per-entry variants through `_geometry`.

## A tool name is prompt vocabulary

Prompt-bundle audits forbid every declared tool name, as a whole word and
case-insensitively, in the planner, core, environment, embodiment and
episode-task layers, so that tool syntax has one owner. A tool therefore cannot
take a common English word the layers already use. The shared motion-v3 `wait`
action is submitted by `wait_ticks`: every planner layer already says "wait for
the next actual observation", and naming the tool `wait` failed the
`planner_single_owner_vocabulary` check for every current bundle. The same rule
keeps task-contract notes tool-neutral: the RoboDojo `imitate_sorting_sequence`
note says the demonstration advances only while native actions are consumed and
asks the robot to wait, but names no tool, so it stays true on a surface
without `wait_ticks`. Check a proposed name against the rendered layers with the
prompt-bundle audit before publishing it.
