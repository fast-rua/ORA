# Experiment families, generations and runs

An experiment family identifies a continuing research or debugging question.
It does not mean one perfect execution: a family grows generations as its
treatment changes, and it can fork into branches. The source-tracked
[identity registry](../../ora/orchestrator/identity-registry.json)
records families, namespaces, lineage and historical aliases. It is not an
active-run or authorization registry, and it never grants launch authority.

The registry shipped with this repository is empty: the maintained package
`examples/official_suite` does not declare an identity, and its runs project
as `unmapped`. Register a family when a study
needs lineage across treatments, as described below.

## Identity levels

| Level | Example | Meaning |
|---|---|---|
| Family | `EXP90A` | Stable question or development line |
| Branch | `main`, `low-effort`, `vla` | Scientific or engineering fork within that family; independent of Git branch names |
| Generation | `EXP90A/low-effort/g001` | One named treatment generation, with zero or more explicit parent generation IDs |
| Execution namespace | `my-study` | Package, output and admission namespace (`RunSpec.experiment`) |
| Cohort / cell | `launch_admission.cohort_id` / `cell_id` | The authorized selection and attempt boundary |
| Run / round | `run_id` and persistent-session round | Exact source-bound execution and its per-round official verdict |

The canonical generation ID is `FAMILY/branch/gNNN`. Family IDs match
`EXP[0-9]{2,}[A-Z][0-9]*`; branch slugs use lowercase letters, digits and
hyphens and begin with a letter; generation numbers are positive with at least
three digits. Numbers are local to a branch. A fork starts at `g001`; its
parents identify where it came from. A generation may have multiple parents
when reviewed treatment components are combined. Never infer ancestry from
numeric ordering, directory names or a Git branch name alone.

A material change to the policy, prompt, tools, observation, Context, Memory,
execution semantics, controller or provider/effort creates a new generation.
A new test seed, interrupted process, retry, source documentation commit or
additional cell alone does not. Keep each actual source commit and treatment
hash even within a generation. A new scientific question creates a new family;
repairing or extending the same question normally keeps the family.

## Registry format

`identity-registry.json` has schema `ora.experiment-lineage.v1` and
these collections:

| Key | Content |
|---|---|
| `families` | `{FAMILY: {title, description}}` |
| `namespaces` | `{namespace: FAMILY}`; a namespace belongs to one family |
| `generations` | `{FAMILY/branch/gNNN: {status, label, identity}}`; `status` is `development` or `historical` |
| `legacy_aliases` | Exact `{experiment, cohort_id, source_commit, generation_id}` selectors for runs recorded before their plans carried an identity |
| `reviewed_migrations` | Optional; see [public registry](#public-registry) |

`load_catalog` rejects an unknown family, a generation key that differs from
its identity, an unknown parent, a lineage cycle, a namespace mapped to an
unknown family, and an ambiguous alias or migration. Commits are full
40-character hashes. The registry contains no generated trajectories,
credentials, run state or mutable outcome totals.

A minimal registry with one forked family:

```json
{
  "schema": "ora.experiment-lineage.v1",
  "families": {"EXP90A": {"title": "Pick and place", "description": "Tool-loop pick and place on RoboTwin."}},
  "namespaces": {"my-study": "EXP90A", "my-study-low": "EXP90A"},
  "generations": {
    "EXP90A/main/g001": {"status": "historical", "label": "First treatment", "identity": {
      "schema": "ora.experiment-identity.v1", "family": "EXP90A",
      "branch": "main", "generation": "g001", "parents": []}},
    "EXP90A/main/g002": {"status": "development", "label": "Revised prompt", "identity": {
      "schema": "ora.experiment-identity.v1", "family": "EXP90A",
      "branch": "main", "generation": "g002", "parents": ["EXP90A/main/g001"]}},
    "EXP90A/low-effort/g001": {"status": "development", "label": "Low-effort fork", "identity": {
      "schema": "ora.experiment-identity.v1", "family": "EXP90A",
      "branch": "low-effort", "generation": "g001", "parents": ["EXP90A/main/g001"]}}
  },
  "legacy_aliases": [],
  "reviewed_migrations": []
}
```

## Source declaration and resolution

Register the family, namespace and generation in the registry, then declare
the identity in `experiments/<namespace_with_underscores>/experiment.yaml`.
The repository ships no `experiments/` directory: create it for your own
study packages. A study that changes prompt, tools, observation projection,
context policy, sandbox, session cadence or execution semantics gets its own
branch, package and output workspace; maintained `ora/` modules never import
experiment code, run artifacts bind the exact experiment commit and generated
evidence never enters Git.

```yaml
experiment: my-study
experiment_identity:
  schema: ora.experiment-identity.v1
  family: EXP90A
  branch: main
  generation: g002
  parents: [EXP90A/main/g001]
```

The shared RunSpec resolver inherits this declaration automatically, before
computing the RunPlan hash. A generated RunSpec may repeat the same identity;
a conflicting declaration, an unregistered identity or a namespace registered
to another family is rejected. A standalone RunSpec can supply an explicit
registered identity without a package. Native and scripted
policies and every backend use this same resolver; identity does not depend on
a tool surface, simulator or provider.

Packages without a declaration remain resolvable without this field. A
declaration does not authorize another run or certify a qualified new
treatment.

Persisted plans validate their embedded identity structurally and include it
in their content hash. Validation never consults the registry, so later
registry additions cannot invalidate a sealed plan or change its bytes or
hash. Only an exact [reviewed migration](#public-registry) changes how a
recorded run is labeled, and its projection keeps the recorded identity beside
the new one. Plans without the field keep their exact original bytes and
hash. Identity metadata is operator evidence and is not injected into policy
context.

## Historical aliases

A run recorded before its package declared an identity can be assigned a
generation by an exact `(experiment, cohort_id, source_commit)` alias. Only
that exact triple is assigned: the same namespace at another source or cohort
stays `family_only` (when the namespace is registered) or `unmapped`. Aliases
and consumed generation identities are append-only; a correction needs a
reviewed migration, never an in-place rewrite of evidence.

An alias groups runs; it is not a treatment certificate. Use source and
treatment hashes, cohort selection and valid sealed verdicts for comparisons.
Family membership never licenses pooling success rates.

## Inspection and lifecycle

`./scripts/ora experiments [FAMILY] [--json]` lists generation IDs,
parents and development/historical status without launching anything; it is a
development command, left out of `ora --help`, the
[command guide](../usage/commands.md) and the generated
[command reference](../usage/cli-reference.md). Its `--json` view
(`ora.experiment-lineage-view.v1`) also lists the namespaces, legacy aliases
and `reviewed_migrations` of the selected generations. The Viewer does not
group runs by family or generation: its run registry filters by the RunPlan's
`experiment` field. `experiment_identity.project_identity` gives a run's
read-only projection with `binding: plan`, `historical_alias`,
`reviewed_migration`, `family_only`, `unmapped` or `invalid`. The binding
distinguishes an immutable recorded field from a read-only retrospective label.
A `reviewed_migration` projection also carries `recorded`, the `plan` or
`historical_alias` projection it replaces.

Creating a family, generation, branch or Git commit never resets a consumed
admission cell, bypasses its retry policy, changes resource leases or grants
launch authority. A generation becomes `historical` when its package leaves
the tree; its identity, parents and namespace stay registered and still
validate, and it composes only against its own launch source. A fresh attempt
that replaces an externally interrupted cell whose single-attempt claim was
consumed declares a successor generation; the interrupted prefix keeps its
identity and is not pooled.

## Public registry

The shipped identity registry is empty. Register your own families/namespaces/generations when using optional lineage declarations; an undeclared standalone RunSpec remains valid. Illustrative identities above are examples, not shipped experiment registrations.
