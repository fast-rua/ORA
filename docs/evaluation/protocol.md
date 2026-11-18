# Evaluation protocol

How ORA decides, counts and reports a benchmark outcome. Where ORA has no
rule yet, the page says "not specified"; a reporting convention we recommend
but no tool computes yet is marked **recommended**. For
how a run executes end to end, see
[How ORA works](../how-ora-works.md); for what the policy sees and sends, see
[policy inputs and outputs](../reference/policy-io.md).

## What decides success

Only the selected environment's official boundary supplies the boolean verdict.
The model's self-assessment never does.

| Environment | Official verdict | Recorded as |
|---|---|---|
| RoboTwin | The simulator's `check_success`, latched as `eval_success` | `official_success` in `result.json` (per round in a session) |
| LIBERO / LIBERO-Pro | LIBERO termination; a truncated episode reports `false` | `official_success` |
| RoboDojo | The task's official result, latched inside the worker bridge | `official_success` |
| RoboCasa365 | RoboCasa's `_check_success`, latched at the first control step it holds | `official_success` |
| RoboLab | RoboLab's success termination; the episode freezes at success | `official_success` |
| Synthetic | None | `official_success: null` |
| Free play (`ora play`) | Withheld by design | `official_success: null` |

`official_success` always means a success inside the benchmark's official limit.
Fresh RoboCasa365 and RoboLab runs continue to three times that limit, and fresh
RoboDojo runs until a tool budget ends them; a success there is
`success_after_official_limit`, recorded with the native actions it took and
never counted as an official success. See
[the extended budget](../reference/configuration-and-evidence.md#official-limit-and-the-extended-budget).

A policy's `finish_episode` call, or any claim in its text, cannot override
official evidence. RoboDojo's default surface offers no `check_task` and no
`success` outcome: the simulator's official predicate ends a successful episode
by itself, and the model can only give up. On surfaces that advertise
`check_task`, the Supervisor
writes `policy-task-check.json` at every published observation from the
trusted environment boundary, and `check_task` returns only the boolean
`task_success` from it. The policy learns whether the task currently holds,
never the predicate, its source or object state. With the `faithful_check_task`
capability, `finish_episode` refuses `outcome: success` unless `check_task` is
true at the current boundary, and once success is confirmed it admits only
`success`. These refusals are policy-side task state, not infrastructure
faults. The file is per round and archived at each round reset.

## Units of evaluation

| Unit | Meaning |
|---|---|
| Task | A benchmark task identity (a RoboTwin task, a LIBERO suite and task index, a RoboDojo, RoboCasa365 or RoboLab task). |
| Case (cell) | One task on one seed, initial state or layout, with its scene configuration. A cell is the admission unit: `launch_admission.cell_id` inside a `cohort_id`. |
| Run | One launched execution, keyed by its exact `run_id`, under `outputs/<experiment>/runs/<run_id>/`. |
| Episode / round | One reset-to-end interaction with one official verdict. A single-episode run is its own round. |
| Session | A persistent run of K rounds (`session.cadence: persistent_agent_k_rounds_v1`, `rounds` 1 to 32, optional `round_seeds`). Each round has its own verdict; the session-level `official_success` stays `null`. Sessions are explicit inputs, never implied by task identity. |
| Attempt | The run report's unit: one round of a run. A session that stopped early keeps the round in flight as one more attempt. |
| Cohort / group | A cohort is the set of single-attempt cells; a queue group is one launch intent over a cell list under `outputs/<experiment>/groups/<group>/`. |

Three counters measure effort inside an episode and must not be confused:

| Counter | What one unit is |
|---|---|
| Robot decision | One executed robot action proposal. A refusal whose receipt certifies zero native execution and `state_changed: false` spends no decision; it is counted as `refused_proposals`. |
| Model call | One provider request. |
| Simulator step | Backend-specific: one LIBERO control step (`env.step`, 20 Hz); one RoboTwin action count (`take_action_cnt`); one RoboDojo physics step (0.004 s, ten per native action); one RoboCasa365 control step (20 Hz, base moves included); one RoboLab control step (15 Hz, eight physics steps); one Synthetic step. |

## Budgets and stop conditions

A RunSpec must give all four budgets as positive integers. They are ceilings,
not expected durations. In a session, budgets apply per round
(`budget_scope: per_round`).

| Budget | Benchmark builders (LIBERO-Pro, RoboTwin, RoboCasa365, RoboLab) | RoboDojo builder | Synthetic quickstart |
|---|---|---|---|
| `max_robot_decisions` | 80 | 80 | 1 |
| `max_model_calls` | 200 | 200 | 6 |
| `max_sim_steps` | 10,000 | Task ceiling (below) | 1 |
| `wall_time_s` | 21,600 (6 h) | 21,600 (6 h) | 120 |

The official suite uses the benchmark-builder values for both arms.

Environment step limits apply on top of the budgets:

| Environment | Native limit |
|---|---|
| LIBERO / LIBERO-Pro | `max_sim_steps` is passed to the environment as its horizon and shown to the policy as its native-action allowance, one per 50 ms control tick (a strict-limit run sets the suite's protocol limit: 220, 280, 300 and 520 for spatial, object, goal and libero_10). |
| RoboTwin | The task's own `step_lim`; `take_action_cnt` at or above it ends the episode (`robotwin_step_limit_exhausted`), as does reaching `max_sim_steps`. |
| RoboDojo | The task's official native-action limit (`step_lim`, 200 to 1,900 across the 54 admitted tasks; each `*_random` variant has its base task's). `max_sim_steps` must lie in `[10, limit x 10]` on a ten-step boundary and defaults to `limit x 10`. |
| RoboCasa365 | The task's evaluation horizon from RoboCasa's dataset registry (750, 750 and 450 control steps), passed to the environment and shown to the policy as its native-action allowance. |
| RoboLab | The upstream episode length at 15 Hz (750 or 900 control steps), shown to the policy as its native-action allowance. |

Only these principals may end a round: the environment, the operator, an
experiment budget, the policy's lifecycle request, or infrastructure. A round
stopped by a budget or by `finish_episode` still carries the standing official
predicate (normally `false`). Only an infrastructure fault, or a backend that
produced no verdict, leaves the boolean `null`.

## Seeds and case lists

| Benchmark | Case list | Seed or state selection |
|---|---|---|
| LIBERO | Spatial, Object, Goal and LIBERO-10 suites from the pinned code checkout | Fixed init-state files per task |
| LIBERO-Pro official core | 16 suites (Object, Goal, Spatial, LIBERO-10 x `_task`, `_swap`, `_lan`, `_object`, reported as Task, Pos, Sem, Obj) x 10 tasks x init states 0 to 49: 8,000 cells at the pinned code and data revisions | Init state `seed % 50`. Env perturbations are excluded because their states are generated at run time. |
| RoboTwin official | 50 tasks, each under `demo_clean` and `demo_randomized` | Per task and configuration, the first 100 expert-accepted seeds in ascending order from requested seed `100000 x (1 + outer_seed)`, unseen backgrounds. A seed is accepted when the expert plans and `check_success` holds after replay; expert rejections and infrastructure failures do not count as policy trials. Accepted seeds live in a sealed seed manifest from `ora qualify robotwin prepare` / `qualify`. Development (practice) seeds lie below 100000, scanned with `prepare --practice-start` ([commands](../reference/robotwin-qualification.md#evaluation-seeds-ora-qualify-robotwin)). |
| RoboDojo | The 54 settings at RoboDojo `f57e163` and assets `1127bda`: the 42 official tasks and the 12 `*_random` variants of the generalization tasks (task-contract registry-v10); each task's contract record is reviewed on its own layout 0, a variant's on `<task>_random_0.json`. `swap_blocks`, `press_by_number`, `organize_table` and `stack_blocks_by_language` results from the earlier pin (RoboDojo `ee67a146`) are not pooled with later ones ([pinned revision](../environments/robodojo.md#pinned-revision)) | Layout 0 of evaluation seed 0 by default. One episode can run on another official layout of evaluation seed 0 that the reviewed layout-portability table (`ora/policy/environments/robodojo/task-contracts/layout-portability-v5.json`) lists for the task: layouts 0 to 9 for every admitted task except `sweep_blocks`, whose contract holds on layouts 0, 3, 4 and 7, and `hang_mugs_random`, whose contract holds on layouts 0 to 4 and 6 to 9. A `*_random` variant's layout k is its own `<task>_random_<k>.json`. A session can use a reviewed layout pool (evaluation seed 0). The default K-round schedule is the first K pool entries, and each round's environment seed equals its layout ordinal. A RunSpec that names its own task file (`policy.task_file`, [task files](../../ora/policy/task_files/README.md)) delivers no reviewed contract and may run any official layout of evaluation seeds 0 to 2 (`EVAL_SEEDS`), the seeds the pinned Assets hold; its instruction is read back from the reset. |
| RoboCasa365 | 3 admitted tasks of the target split (target object instances, kitchen layout and style pairs 1 to 10) | The seed samples the kitchen, the target and the instruction. An official RoboCasa365 suite is not specified. |
| RoboLab | 3 admitted tasks | The seed is the environment seed. An official RoboLab suite is not specified. |
| Synthetic | One built-in world | Not applicable |

Cases are selected by explicit cell ids, never at random. The official-suite
example names them `lp-<suite>-t<task>-s<state>` and
`rt-<clean|random>-<task>-s<seed>`; `run.py cells` lists a selection and
`--accepted-limit` truncates each RoboTwin manifest. See
[examples/official_suite](../../examples/official_suite/README.md) and
[benchmark setup](../getting-started/official-suite.md).

An external case list, such as the RoboTwin leaderboard's 250 C2R pairs, is
not shipped. Such a list can name seeds the expert never solves. To keep the
same denominator, ORA runs those cells under a labelled
`robotwin_expert_unsolved_seed_v1` qualification exception that cites seed
manifests showing the expert failed. The exception stays in the plan, so
readers can separate those cells from admitted ones. A RoboDojo official suite
is not specified.

## Validity

Evaluation writes `evaluation.json` with one of three statuses:

| Status | Meaning | Sealable |
|---|---|---|
| `valid` | Every evidence check passed. | Yes |
| `incomplete` | The run stopped early for an expected reason only (`trusted_fault`, `causal_count_mismatch`), including an operator stop. | Yes |
| `invalid` | Any other evidence violation. | No: the seal is refused and recorded as a refusal. |

Each round also records `termination_authority` and `behavioral_sample_valid`.
The latter is `false` when infrastructure or the operator ended the round, or
when an operator message reached the policy (`human_intervention: true`).

**Infrastructure faults vs task failures.** A fault carries a layer (provider,
context, tool, protocol, environment, evidence, operator, orchestrator). A
trusted fault ends the session (or round) under infrastructure authority, as
does a policy abort the Harness admits on the round's recorded infrastructure
evidence; neither records a verdict. A task failure is an official `false`
reached through the environment, a budget or the policy's own finish. Physical
contact that changes measured state is not, by itself, an infrastructure fault.

**Retry and replacement.** Benchmark cells are `single_attempt_cell`
admissions with `retry_allowed: false`; launch consumes the claim. `ora retry`
applies only to an unsealed, incomplete attempt that dispatched no robot
decision and whose plan permits retry. It creates a new run and never resumes
or overwrites the old directory. A replacement for a consumed cell needs a new
cohort or cell id; a replacement for an externally interrupted cell declares a
successor generation ([experiment identity](../reference/experiment-identity.md)).
The original attempt stays in the evidence and is not pooled with its
replacement.

**Sealing.** `seal.json` binds the file inventory, the evaluation status and
the sealed source commit, and makes the files read-only. Later revalidation
(`ora revalidate`) writes separate reports and never changes the sealed run.
The run report also reads operator corrections under
`<namespace>/diagnostics/corrections/`. A correction overrides validity and
learning eligibility while the sealed evaluation is kept, and a hash mismatch
is flagged.

**Learning eligibility.** `evaluation.json` records
`learning_eligibility.eligible` with reasons. A run is ineligible when it did
not complete, had human intervention, had a trusted fault, is a free-play
session, has a receipt its backend labelled an execution fault, fails the
RoboDojo evidence checks, or, for LIBERO-Pro, lacks the qualified controller
evidence on every physical receipt. A behaviorally valid run can still be
ineligible.

## How to report results

The run report defines a **valid attempt** as one whose run validity is
`valid`, whose round has an official boolean and whose behavioral sample is
not marked invalid. Its `success_rate` is successes over valid attempts, and
it lists attempts, valid attempts, excluded attempts by reason, and every
ending class. Report the following with every result:

- every rate with its numerator and denominator;
- the observation/action abstraction, the task and seed protocol, attempts and
  budgets, and differences from the leaderboard protocol, as the setup guide
  requires;
- results per benchmark row, without pooling across renderers, treatments or
  families;
- the result tier, never pooled across tiers (below).

### Result tiers

Every plan records `result_tier`, what the policy carried into the episode
beyond its prompt and tools. The run report groups results by tier first and
counts each tier's `check_task` calls.

| Tier | Meaning | Produced here |
|---|---|---|
| `zero_shot` | The default: no experience from the benchmark, no per-task recipe in the prompt, `check_task` advertised where the surface offers it | Yes, the default |
| `zero_shot_no_check` | `zero_shot` with `check_task` removed from a surface that offers it: measures reliance on the official Boolean, which a real robot does not have. A surface with no `check_task`, RoboDojo's default, stays `zero_shot` | Yes, set `result_tier` in the RunSpec |
| `with_experience` | Experience gathered on the same benchmark's practice partition | No: refused until an experience library binding exists |
| `frozen_program` | A program compiled after practice and run without a model | No: refused; such results come from outside the Runtime |

Plans recorded before the field carry no tier and are reported as
`unlabelled`, apart from `zero_shot`. Report a `with_experience` result only
in its own row, with the practice episodes, tokens and GPU hours beside the
evaluation cost.

### Practice and evaluation partitions

Every plan records its partition: `evaluation` (the default) or `practice`.
A practice plan may not use an episode from any benchmark's evaluation set;
resolution refuses it, using the list in
`ora/environment/benchmarks/evaluation-sets-v1.json` (see
[practice initial states and evaluation sets](../reference/configuration-and-evidence.md#practice-initial-states-and-evaluation-sets)).
Practice runs are never evaluation results. The run report's first table
groups runs by partition and gives each group's runs, valid attempts,
successes, model calls, input and output tokens, API-equivalent cost and wall
hours, so the practice budget is reported beside evaluation; plans recorded
before the field are `unlabelled`.

**Recommended**, not yet computed by `ora report`:

- Show two rates side by side: successes over all selected cases, counting a
  case without a valid outcome as a failure, and the run report's rate over
  valid attempts.
- Report `success_after_official_limit` apart from both groups below.
- Split failures by ending class: task failure (`failure_latch`,
  `episode_end`, `environment_end`, `policy_finish`); timeout or budget
  (`native_action_limit`, `decision_budget`, `model_call_budget`,
  `wall_time_budget`, `simulation_step_cap`); infrastructure
  (`infrastructure`); and others (`operator_interrupt`, `in_progress`,
  `other`) listed separately.
- With repeats, give the mean and a 95% confidence interval and name the
  method. The run report gives means and percentiles for effort, not
  intervals.

## Checking that a suite run is complete

1. `./scripts/ora suite status --experiment NS --group G`
   prints `queue_status` (`complete` or `paused`), counts per cell state and
   the first error. A cell settles as `sealed_terminal`,
   `trusted_fault_recorded` or `unsealed_recorded`; a group is complete only
   when every cell settled.
2. `ora report --namespace outputs/NS --cohort ID` reports attempts,
   validity and endings per result tier, per cohort and per (model, task). Compare its attempt
   count with the selected cell list (`run.py cells`), and list cells with no
   run.
3. Rerun missing cells with `queue --resume` in a new group, which skips cells
   already sealed or claimed in the cohort.
4. `ora status` summarizes active leases and the indexed local run registry;
   `ora workspace` shows which evidence roots are indexed.

## Auditing one case

Open the run directory and read, in order: `plan.json` (task, seed, budgets,
source), `result.json` (termination, authority, counters, verdict, rounds),
`evaluation.json` (status, reasons, learning eligibility), `seal.json` or the
seal refusal, then the canonical events, receipts and provider evidence.
`ora watch RUN_ID` replays the transcript with per-round verdicts. File names
and schemas are described in
[configuration and evidence](../reference/configuration-and-evidence.md#episode-evidence)
and [run and inspect](../usage/runs.md#read-the-evidence).
