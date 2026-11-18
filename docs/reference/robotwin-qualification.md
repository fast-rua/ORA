# RoboTwin seed and scene qualification

Two RoboTwin tools qualify what a policy episode may start from. `ora qualify
robotwin` scans official evaluation seeds with RoboTwin's own expert and seals
seed manifests that a RunSpec names as its `qualification.manifest`; `ora play
audit --env robotwin` checks ORA's own free-play scenes on a host. Neither
calls a model, publishes a policy action or launches an episode. The RoboTwin
[environment page](../environments/robotwin.md#qualify-one-seed) shows a
one-seed scan; this page is the reference for both tools.

## Evaluation seeds: `ora qualify robotwin`

The maintained source-checkout entry is `ora qualify robotwin`
(`ora/orchestrator/backends/robotwin/evaluation_seeds.py`, the tools the
RoboTwin backend declares). It is separate from the generic episode commands
([commands](../usage/commands.md)):

- `configs [--outer-seed N] [--task-config demo_clean|demo_randomized]`
  prints the pinned 50-task evaluation protocol for one scene configuration
  without writing evidence, querying GPUs or starting a simulator.
- `prepare --robotwin-root PATH --robotwin-python PATH --tasks TASK...
  --slots SLOT... --candidate-count N [--accepted-per-task K]
  [--task-config demo_clean|demo_randomized] [--outer-seed N]
  [--start-offsets K... | --segments TASK:OFFSET[:COUNT]...]
  [--process-attempt-limit N] [--practice-start S]
  [--purpose targeted_regression|development_diagnostic]` checks the exact
  clean upstream revision (the vendored install the Runtime loads), the
  selected scene configuration's official settings, task roster, unseen
  texture pool and current capacity. It hashes declared dependencies and
  writes an immutable plan under the shared
  `outputs/runtime-v1/robotwin-evaluation/` workspace (under
  `ORA_OUTPUTS_ROOT` when set, not the node-local workspace). Tasks, slots and
  candidate count have no experiment defaults. It prepares 50 configurations
  but schedules only the explicitly named cases: per task, the first K seeds
  from the official start seed that the expert accepts (K defaults to 1; the
  official protocol is 100), trying at most N candidates. Accepted seeds are
  bound to the scene configuration and revision, so `demo_clean` and
  `demo_randomized` are separate scans. `--start-offsets K...` splits long
  tasks into segments: every (task, offset) is its own case and lane, scanning
  N seeds from the official start plus K (set `--accepted-per-task` equal to
  the candidate count to scan the whole segment). Case ids are `task` for
  offset 0 and `task@<first seed>` otherwise; they name the case's output and
  manifest. `--segments TASK:OFFSET[:COUNT]...` lists the cases explicitly
  instead, for tasks that need different numbers of segments; COUNT (at most
  N) sets one segment's length, so the unscanned remainder of stopped segments
  can be tiled exactly. `--process-attempt-limit` (default 8) is how many
  seeds one oracle process scans before re-executing itself; plans written
  before it existed use 1. One seed per process re-creates the CUDA context,
  renderer and planner warm-up every seed, which serialized on the NVIDIA
  driver at ~100 workers per node. The interpreter path is kept as given, not
  resolved (a venv's `bin/python` is a symlink), and a GPU whose `nvidia-smi`
  row reads `[N/A]` is left out of capacity. `--practice-start S` scans
  development seeds instead of evaluation seeds: offsets count from seed S
  rather than the official start, every case must end below 100000 (the
  lowest official start, `PRACTICE_SEED_LIMIT`; RoboTwin's own demonstration
  collection also draws from 0 upward), `--purpose` must be
  `development_diagnostic`, and `--outer-seed` stays 0. Practice case ids are
  always `task@<first seed>`, and the plan and every sealed manifest record
  `practice_start`. Such seeds suit practice episodes; no official trial uses
  them.
- `qualify --plan PATH --confirm QUALIFY-ROBOTWIN:HASH` checks the saved plan,
  exact clean Harness/upstream sources, dependency hashes and the host marker
  of the plan's `required_host` (set by `prepare --required-host`, default the
  current marker; a plan without one accepts any registered marker),
  rechecks GPU capacity, acquires resource leases and runs isolated oracle lanes
  in parallel. Lanes start 3 s apart, and a lease claim refused only because
  the GPU probe could not be read (`device_probe_unavailable`) is retried with
  backoff. Each lane records its commands, logs and private evidence. A
  candidate is rejected and the lane continues when the scene is unstable, the
  expert's plan or the official predicate fails (`oracle_failure`), or the
  expert's own task code raises (`oracle_exception`, innermost frame under
  RoboTwin `envs/`; upstream `collect_data.py` treats any `play_once` exception
  as a failed seed). An exception from anywhere else (CUDA, CuRobo, SAPIEN, the
  Harness) is `infra_failure` and stops that task's lane. A seed whose expert
  run gives no result for 2 h (`SEED_WATCHDOG_S`; 3 h, `SEED_WATCHDOG_FIRST_S`,
  for a worker process's first seed, which carries its start-up) is recorded in
  the worker's ledger as rejected with classification `oracle_timeout`
  (`recorded_by: robotwin_evaluation.seed_watchdog`, with `limit_s` and
  `first_in_process`); the worker is stopped and restarted and resumes at the
  next seed, and the lane result lists those seeds in `oracle_timeouts`. There
  is no other per-case timeout.
  It seals one `seed-manifests/<task>.json` per scanned task from the worker's
  ledger: accepted seeds and rejected candidates with their classification in
  requested-seed order, `complete` when the target was reached, per-seed
  evidence under `seeds[<seed>]` (`reset_predicate_false`, `oracle_success`,
  `post_success_physics_hold`, measured `terminal_stability_steps`),
  aggregates derived from those records (`unstable_accepted_seeds`,
  `required_stability_steps`) and `manifest_sha256`, so a RunSpec can name it
  as its `qualification.manifest`. Sealing leaves out `infra_failure` and
  `invalid_artifact` records (the worker resumes at that seed) and timeouts
  recorded under a shorter bound than the one now in force (earlier builds used
  30/90 min, which stopped sound slow seeds), so those seeds count as
  unscanned. Launch preflight admits a seed of this
  schema when the manifest's task, `task_config` and `robotwin_revision` match
  the RunSpec, the digest verifies, and the seed's own reset/oracle evidence
  holds; the post-success hold is evidence, never admission, because upstream
  collects such seeds. Exit 0 means every requested case
  passed; exit 2 reports incomplete/failing cases. This does not authorize or
  launch a policy episode.
- `merge --task-config C --inputs <qualification dirs or manifests> --output DIR
  [--accepted-per-task N] [--conflicting-outcomes refuse|reject] [--practice-start S]` joins each task's segments: walking seeds upward
  from the official start (from S with `--practice-start`, which merges only
  practice segments; official segments merge only without it), it keeps the first N accepted seeds of the
  contiguous scanned prefix, exactly what one sequential scan collects.
  Segments must share the RoboTwin revision and the task's own dependency
  closure (shared inputs, the checker and that task's `envs/<task>.py`, read
  from each segment's `prepared-plan.json`; a bare manifest falls back to the
  plan-wide producer hash, which differs whenever two plans schedule different
  tasks). A seed recorded with two outcomes is refused unless
  `--conflicting-outcomes reject` is given, which counts a seed accepted in one
  segment and rejected in another as rejected (classification
  `conflicting_outcomes`, listed in `conflicting_seeds`) and keeps the earliest
  segment's record when re-scans agree. Expert scene stability is not
  reproducible run to run, so the official suite manifests use `reject`. A gap before N leaves
  the manifest incomplete with `next_unscanned_seed`. The merged manifest keeps
  the seed-manifest schema, so `seed_admission` reads it like any `qualify`
  manifest, and lists its segments in `merged_from`; `merge-report.json`
  summarizes the tasks. Exit 2 if any task is incomplete.
- `seal --inputs <qualification dirs>` seals interim manifests from a scan that
  is still running into `seed-manifests-interim/` (the final `seed-manifests/`
  is untouched); a torn final ledger line is skipped. `merge` reads the interim
  folder when no final one exists, so a small `--accepted-per-task` gives a
  pilot list that is a prefix of the final one. Neither command authorizes or
  launches a policy episode.

Outer seed defaults to zero, hence requested seeds begin at 100000.
The exclusive limit is start plus the explicit candidate count. Preparation
reports the actual longest-lane estimate (540 seconds per candidate plus
90 seconds setup/finalization); estimates over 15 minutes are explicitly
`development_diagnostic`, never an implicit per-launch requirement.
Saved preparations do not automatically discover/reuse historical evidence.
Do not reschedule already valid cases solely because HEAD changed; identify
missing or dependency-invalidated cases before requesting a new preparation.
Historical experimental plan schemas/confirmation codes are not accepted by
this entry. No fixed GPU assignment, provider or policy binding is inherited.

## Play-scene audit: `ora play audit --env robotwin`

`ora play audit --env robotwin --scene-set SET [--scene NAME ...] [--seeds N | --seed-list S ...]
[--host HOST] [--resource-profile SLOT]` (`ora/orchestrator/backends/robotwin/play_audit.py`,
the RoboTwin backend's `play_audit`) qualifies ORA's own RoboTwin free-play scenes
(`ora play --scene-set`) on this host: for each seed (default 0, 1, 2) it
builds the scene in the `ROBOTWIN_PYTHON` worker on the slot's devices, one
scene at a time under the slot's resource lease, and writes one
`robotwin_play_scene_audit` record per scene with its camera images
([free play](../usage/play.md#tables-with-many-objects-robotwin-scene-sets),
[record and admission](configuration-and-evidence.md#play-episodes)).
It refuses a RoboTwin checkout that is not clean at the pinned revision and
uncommitted scene modules. It exits 0 when every scene has an accepted seed
and 1 otherwise; it never runs an expert, publishes a robot action or launches
an episode.
