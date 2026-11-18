# Motion speed and wrist checks

Every arm move on a backend that adopts it is paced at the controlled point and
checked for an unrequested wrist turn before anything moves. The semantics are
Yulun's RoboDojo toolkit timing (`tools/move_pose/_smooth.py`), so a backend whose
global planner answers a joint path (any planner) times it the same way. Two implementations hold them:
the [TOPP-RA + Ruckig timing worker](#the-timing-worker), the default, and
`ora/environment/controllers/motion_pace.py`, the hand-written port, the
[fallback](#the-fallback-timing) for an environment without the timing
interpreter. Every paced move's receipt says which one ran.

## The request

`ora.contracts.motion_speed` holds the request both the agent and the
environment read. A policy call may name:

| Argument | Meaning |
|---|---|
| `speed` | `fast` (the default), `normal` or `careful` |
| `speed_m_s` | instead of `speed`: any top speed of the controlled point above 0 m/s |
| `slow_approaching` | `true` also holds the last 2 cm of each arm's path to `careful`, entered without a stop; refused with `careful` and with `speed_m_s` at or below 0.1 m/s |

| Preset | Controlled point | Reached in | Joints (every joint, the ceiling) |
|---|---|---|---|
| `fast` | no Cartesian cap | — | 4 rad/s, 40 rad/s² (RoboDojo, ARX X5 motion profile v2: 4.5 rad/s, 60 rad/s², certified at 5000 rad/s³; [profiles](#robodojo-motion-profiles)) |
| `normal` | 0.3 m/s, 60 deg/s | 1 m/s², 200 deg/s² | 2.5 rad/s, 15 rad/s² |
| `careful` | 0.1 m/s, 20 deg/s | 0.5 m/s², 100 deg/s² | 1.2 rad/s, 8 rad/s² |
| `speed_m_s` = v | v m/s, max(20, 200 v) deg/s | both in 0.25 s | fast's (the preset's 4 rad/s and 40 rad/s², whatever a robot profile names) |

A backend lowers the joint caps to what its executor can follow and never
raises them, so speed is limited at the controlled point first and each joint
stays the ceiling: a fast request may not be reached, a slow one practically
always is. A speed never picks a path: the interaction label does, at every
speed, so a slow `free_space` or a slow `precision` move past 5 cm or 15 degrees
takes the planned route, capped along the whole of it.

The motion action carries the request as the optional `speed` of an
`ora-motion-action.v3` `move_pose` (`{"speed": preset or "custom",
"slow_approaching": bool}`, plus `speed_m_s` with `custom`). A call that names
no speed option carries no `speed` and runs at the default.

## Pacing a planner's path

`motion_pace.pace_path(path, fk, ...)` is the one entry point, whichever
implementation runs. It takes the planner's joint path (start first) and the
caller's forward kinematics of the controlled point, and builds the grid:

1. Rows within 1e-6 rad of the row before them are dropped
   (`distinct_rows`); the first and last rows stay exact. A planner's timed
   output repeats its start and goal.
2. Consecutive rows are joined by straight joint lines (as an executor
   tracking them does) and nodes are spread evenly along them, at most
   0.02 rad apart (`joint_path_grid`). A row where the joint direction turns
   by more than 20 degrees is a corner: it stays a node and the move stops
   there. Smooth planner paths have none (at most 6 degrees on the recorded
   LIBERO plans).

The controlled point is measured at every node with the caller's forward
kinematics, so neither implementation knows a robot. `pace_path` takes an
optional `robot`, which names the jerk limits below; everything else in its
signature and its `PacedPath` is the same for both.

## The timing worker

`pace_path` uses the worker whenever the process has one
(`motion_timing.client`): a backend selects it with `motion_timing.configure`
(LIBERO, from `runtime.motion_planner.timing`; RoboTwin, RoboCasa365 and the YAM
station, from `runtime.motion_timing` through `motion_timing.bind_preflight`), or else
`ORA_TIMING_PYTHON` names its interpreter. The worker, `ora/environment/planners/timing/`, runs as its own
process under a small CPU interpreter holding toppra 0.6.10, ruckig 0.19.4 and
their dependencies (`requirements.py` there, about 250 MB; `ora setup` builds
it as the `motion-timing` component). Both libraries are MIT licensed. It takes
one `retime` request and answers the path sampled at the backend's control tick
([contract](configuration-and-evidence.md#timing-a-path-retime)).

1. **Pieces.** The path is cut at its corners into pieces that start and end
   at rest. Each is a cubic spline in joint space through the nodes,
   parameterized by joint arc length.
2. **TOPP-RA** finds the fastest progress along each piece under every cap at
   once, each at 97% (a cap holds at gridpoints only): every joint's speed and
   acceleration (the preset's, lowered to the backend's ceiling), and the
   controlled point's speed and turn rate, which are bounds on the path speed
   squared, `x <= (v / sigma'(s))^2`, and its ramp, the tangential acceleration
   `sigma'(s) u + sigma''(s) x`. `sigma` and `theta` are the controlled point's
   arc length and turn along the path, fitted through the caller's poses.
   `slow_approaching` adds the careful caps, joints included, from the
   gridpoint at or before 2 cm from the end; the profile slows into the zone
   without a stop. Ramps hold at both ends of every grid segment.
3. **From rest.** TOPP-RA has no jerk, and a joint at rest cannot take an
   acceleration step. Both ends of a piece therefore also stay below the speed
   a constant jerk (half the joints' limit) reaches from rest,
   `x <= 6^(4/3) / 4 * J^(2/3) * d^(4/3)` at `d` along the path from that end,
   on a grid spaced cubically there.
4. **Ticks.** The profile is sampled at whole ticks, stretched uniformly so
   the last tick ends exactly on the goal; a stretch only lowers every cap's use.
5. **Ruckig** (Community edition) certifies the rows at the joints' hardware
   jerk limits, per tick and state to state: from each row's velocity and
   acceleration, as the rows themselves give them, it must reach the next row
   in exactly one tick (`DurationDiscretization.Discrete`). A tick it cannot is
   timed again without the discretization: an executor taking Ruckig's shortest
   duration runs ahead of or behind the rows by that difference times the
   controlled point's speed there. Those distances may add up to 1 mm along a
   move (`max_deviation_m`), and no such tick may be off by more than a quarter
   of a tick. Past that the move is re-solved with every acceleration cap
   lowered by 20%, up to 8 times (`acceleration_scale`). The rows are always
   samples of the path, which is the path the planner checked for collisions.
6. **Wrist and budget** are `motion_pace`'s, on the timed rows. A cut re-solves
   node prefixes, each rest to rest, and never falls inside a large wrist turn.

Ruckig's `intermediate_positions` is never set: the Community wheel sends
waypoints to a remote API. A test holds this
(`tests/ora/environment/planners/timing/test_ruckig_guard.py` in CI, and a spy
on every Ruckig input under the timing interpreter). Lower jerk than the
hardware's has to come from lower acceleration caps; Ruckig cannot follow a path
smoothly at a jerk the timing does not respect.

### Jerk limits

`motion_timing.JOINT_JERK_LIMITS` holds each robot's hardware limit per joint,
and `pace_path(robot=...)` names one:

| Robot | rad/s³ per joint | Source |
|---|---|---|
| `franka_panda` | 7500, 3750, 5000, 6250, 7500, 10000, 10000 | [Franka, Robot and interface specifications](https://support.franka.de/docs/control_parameters.html) |
| `arx_x5`, `yam`, `aloha_agilex` | 1000 on every joint | `unverified`; no vendor document gives the X5's, and the 500 in the SDK's robot file is a planner setting |
| any other, or none | 1000 on every joint | `unverified` |

TODO: the ARX X5's, the YAM's and RoboTwin's ALOHA-AgileX arms' values are not published. Replace the
conservative default with a vendor value or a measurement before a real move
relies on them; receipts carry `jerk_source`, so `unverified` is visible on
every move until then.

A pace may name its own jerk (`joint_jerk_rad_s3` in explicit caps, the toolkit
`Pace.joint_jerk_rad_s3`); no preset does. `motion_timing.pace_jerk` certifies the
move at that jerk where the robot's is `unverified`, with `jerk_source: "pace"`,
and never above a documented one (the pace's is then a ceiling per joint and the
source stays the robot's). RoboDojo's ARX X5 motion profile v2 names 5000 rad/s³
for its `fast` ([profiles](#robodojo-motion-profiles)); every other move of the
X5 keeps the default below.

The YAM is the one real robot in this table. Its 1000 rad/s³ is a declared value,
not I2RT's: the i2rt source (v1.4.0) names no joint jerk limit (only the DM motors'
own acceleration-limit register, which ORA does not set), and no vendor figure was
found. The station's own acceleration caps bind long
before it ([YAM](#yam)): the rows the worker answers reach at most 169 rad/s³ by
finite differences at the 20 ms tick, so a real move is limited by those caps, not
by the declared jerk.

The X5 stays at 1000 because lowering it does not make the moves gentler. On 245
recorded paths (2026-10-31), 500 lengthens `fast` from a median of 9 ticks to 11
and sends 28 paths to the fallback timing instead of 16, and those have no jerk
limit at all: the largest TCP speed rises from 1.24 to 1.48 m/s. `normal` does not
notice (17 ticks at the median either way, every path certified).

### Changes from the fallback

The caps' semantics are the same. Three things differ on purpose:

- `ramp_free` is gone: every move keeps its ramps, however short.
- Caps hold at 97%, so a move's top speed is about 0.97 of the cap and
  `planned_peak_linear_m_s` says so.
- Moves carry a jerk-limited start and stop.

## The fallback timing

Without a timing interpreter, `pace_path` keeps the hand-written timing, a port
of the toolkit's. So does a path the worker answers `timing_infeasible`, which
no recorded LIBERO plan was; its receipt says why. A worker that is configured
but fails is an error, never a silent fallback.

With a Cartesian ramp, the nodes are spread again so that every segment has the
same length over the acceleration cap that binds it (linear length over the
linear ramp, turn over the turn ramp, joint step over the joint acceleration;
`paced_grid`).

The timing keeps one speed per node in grid segments per second. Speed caps
hold inside every segment, but the acceleration passes carry a node's speed
from one segment into the next. Adjacent segments of different binding length
(a repeated row is a segment of length zero) would therefore step the
controlled point's speed at the node and skip its ramp. The even and
cost-even grids prevent that.

A small forward-backward pass (a TOPP) then finds the fastest
progress under every cap at once, and `knots` samples it at the backend's
control tick, stretched to whole ticks so no cap is exceeded.
`slow_approaching` adds a careful zone that starts at the grid node at or
before 2 cm from the end, so the whole last 2 cm is slow. A segment that ends
short of the move's goal (a pre-contact segment) leaves it out. `peaks`
reports the top speed and turn rate the planned path reaches under every cap,
joints included.

`tests/ora/environment/controllers/test_motion_pace_parity.py` keeps these
presets and the toolkit's equal. The toolkit's own timing is gone: it asks this
fallback, or the worker, through `kinematics.pace_path` (see [RoboDojo](#robodojo)).

Its straight lines (3 mm / 1.5 degrees) were already even, and there its slow zone
held within 0.2%. On planner grids with repeated or uneven rows it showed the same
overshoot the scoping measured below.

A TOPP-RA scoping on the 219 recorded LIBERO plans (2026-10-31) found both
defects in the first version, which timed the planner's rows as they came.
With the fixes, every tick of every move of 2 cm or more keeps its ramp:

| Speed | Ramp, before (max) | Ramp, after (max) |
|---|---|---|
| `normal` | 3.6× | 1.03× |
| `careful` | 2.4× | 1.01× |
| `normal` + `slow_approaching` | 3.5× | 1.01× |
| `speed_m_s` 0.05 | 3.0× | 1.03× |

The ramp ratio is the per-tick tangential acceleration over the ramp; the
residual is tick discretization. The slow zone's top speed went from 1.25× to
1.00× of careful's. Moves got longer: `normal` +19% in total, `normal` +
`slow_approaching` +15%, `careful` and `speed_m_s` +2–3%, `fast` +1%.

The fallback still runs moves under 2 cm and 5 degrees at constant speed without
ramps.

## The two compared

Both implementations, on the 219 recorded V2 LIBERO plans and the 3 fixture
paths (2026-10-31, 50 ms tick, 1 rad/s joint ceiling, Panda jerk limits, offline).
Three plans are refused as unrequested wrist turns by both. Every figure is
measured at the tick from the answered rows with one independent Panda forward
kinematics, not the kinematics the timing was given. The ramp ratio includes the
first and last tick of the move, from and to rest.

| Speed | Timing | Duration p50 / p95 / max (s) | Total (s) | Top speed / cap (max) | Moves with a tick over cap | Ramp / limit (max) | Slow zone / careful (max) | Joint jerk p50 / max (rad/s³) |
|---|---|---|---|---|---|---|---|---|
| `fast` | fallback | 0.45 / 1.85 / 3.35 | 146.4 | — | — | — | — | 157 / 315 |
| `fast` | worker | 0.45 / 1.90 / 3.45 | 150.2 | — | — | — | — | 140 / 322 |
| `normal` | fallback | 0.85 / 2.45 / 3.55 | 239.6 | 0.999 | 0 | 5.909 | — | 33 / 100 |
| `normal` | worker | 0.90 / 2.50 / 3.65 | 245.7 | 0.970 | 0 | 0.969 | — | 34 / 104 |
| `careful` | fallback | 1.85 / 6.15 / 9.70 | 549.7 | 1.002 | 5 | 3.997 | — | 12 / 39 |
| `careful` | worker | 1.85 / 6.35 / 10.00 | 564.6 | 0.970 | 0 | 0.969 | — | 14 / 42 |
| `normal` + `slow_approaching` | fallback | 1.05 / 3.20 / 4.70 | 293.4 | 1.000 | 0 | 3.997 | 1.000 | 36 / 120 |
| `normal` + `slow_approaching` | worker | 1.05 / 3.30 / 4.85 | 297.2 | 0.971 | 0 | 0.970 | 0.971 | 38 / 121 |
| `speed_m_s` 0.05 | fallback | 3.40 / 9.75 / 12.90 | 911.4 | 1.004 | 28 | 4.998 | — | 6 / 16 |
| `speed_m_s` 0.05 | worker | 3.40 / 9.90 / 13.30 | 922.5 | 0.970 | 0 | 0.970 | — | 5 / 17 |

The worker kept every cap on every tick of all 1095 timed moves; each took one
attempt, and at most 2 ticks of a `fast` move needed Ruckig's shortest duration
(at most 0.47 mm). The fallback's largest ramp ratios are the moves
it runs without ramps. A solve takes 5 to 7 ms at the median and
at most 27 ms; the worker starts in about 2.5 s, once per run.

The evidence covers LIBERO's Franka paths only. At other settings, with `fast`
and the unverified 1000 rad/s³, the worker lowers its acceleration more often and
a few moves reach the fallback: none of 219 at a 50 ms tick and 1 rad/s, 10 of
219 at a 40 ms tick and 4 rad/s. The other presets take one attempt there too.

## The wrist

Before anything moves:

- **Unrequested turn.** The controlled point may turn at most 45 degrees more
  along the path than from its start to its planned end orientation. Beyond
  that the wrist is going the long way round, typically at its last joint's
  limit, and the move is refused with `paced_path_wrist_unwind`
  (`path_turn_deg`, `requested_turn_deg`, `unrequested_turn_deg` and the wrist
  report); the refusal asks for a target orientation nearer the current one, and
  the `move_pose` help of each environment-planned surface says what was tried
  before it (below). On 219 recorded LIBERO global plans the largest detour of an
  accepted plan was 31 degrees; the three refused plans each turned the hand
  65 to 69 degrees beyond the request, two of them with the wrist joint turning
  more than a half-turn.
- **Planner detour.** Before the pacing sees it, the cuRobo V2 worker reads
  each pose plan by the same rule and compares it with straight joint lines to
  the same pose: to the plan's own goal joints, when the plan turns the hand past
  the rule (trajectory optimisation detouring on a pure translation, as on
  RoboTwin, where such lines turned 0 to 7 degrees against the plans' 48 to
  156); and to the goal's nearest 2 pi equivalent inside the joint limits, for
  every plan whose goal lies more than half a turn away on a joint that has room
  for one (the ALOHA's +/-4.2 rad; its wrist is the joint that winds, while a
  Panda's or an X5's limits leave no such room); and, when the plan turns past
  the rule, to the other IK solutions of the pose the planner's attempts kept
  (the eight nearest the start, each at its nearest equivalent in range: at the
  Panda's last-joint limit another arm branch can hold the same orientation with
  the wrist in range). The lines and, when the rule accepts it, the plan are
  ranked as the pacing reads them, within the rule first and then the smallest
  path turn, the plan winning a tie, rather than taken in the order the planner
  found them; the best line
  that passes the planner's own checks (every row inside the joint limits and
  clear of the robot itself and of the goal's world, the whole line clear of the
  world a joint path to that goal is checked in) is answered instead of the
  plan. The pose is unchanged; the plan answer's `detail.wrist_detour` keeps
  each candidate's turns and outcome, `chosen` and `status` (`replaced`,
  `kept`). A goal at the far end of the last joint's range with no equivalent in
  range, where the lines turn the long way too, is kept for the pacing to refuse
  ([cuRobo V2 planner worker](configuration-and-evidence.md#curobo-v2-planner-worker)).
- **Jaw twin.** A pose move (`move_pose`, `move_to`, `rotate_*`) whose path is
  refused with `paced_path_wrist_unwind` on LIBERO, RoboCasa365 or RoboLab, with
  the jaws commanded open (the gripper's held directive opens; on LIBERO, whose
  directive ramps, the last accepted jaw target must not be `close` either), is
  admitted once more to its jaw twin: the target turned a half-turn about the
  hand's approach axis (local +z of the grip site or TCP), which a symmetric
  parallel jaw grips the same way and which lies half a turn away on the last
  joint, inside its range where the request was not
  (`ora/environment/controllers/jaw_flip_retry.py`). The twin runs only when the
  planner and the pacing admit it; the receipt's `motion.wrist.jaw_flip_retry`
  then says so (`applied: true`, `reason`, `turn_deg: 180`,
  `requested_orientation_xyzw`, `executed_orientation_xyzw`,
  `refused_unrequested_turn_deg`), and arrival and the pose error are judged at
  the twin; the action's `arguments` keep the request. When the twin is refused
  too, the original refusal stands and its `motion.refused.jaw_flip_retry` says
  the twin was tried (`applied: false`, `twin_refused` with the twin's code). A
  jaw that may hold something keeps the request, since the load would turn with
  the hand; a recovery's joint goal and an executed probe plan are never
  retried. A dry run asks the same retry and keeps the twin's plan for the real
  call, its path segment carrying `jaw_flip_retry` and its last segment ending
  at the twin, so a preview never promises what the real call refuses.
  RoboDojo's toolkit already picks the nearer jaw twin before planning
  (`lib.motion_plan.orientation`), and RoboTwin does not retry.
- **Budget.** A path needing more ticks than one action may spend is refused
  (`paced_motion_budget`, with `tick_budget` and `needed_ticks`), or, on a
  backend that cuts moves to their budget, cut to the longest first part that
  fits; a cut never falls inside a monotonic wrist-joint turn of 90 degrees or
  more, and a move that starts with one is refused with `unsplit_span`.
- **Report.** The wrist joint (the backend's last wrist joint) from the start
  through every planned knot: `start_deg`, `end_deg`, `min_deg`, `max_deg`,
  `travel_deg` (the sum of its turns, which both ends alone can hide) and
  `min_limit_margin_deg`, beside the path's `path_turn_deg` and
  `requested_turn_deg`.

## Receipts

A paced move's receipt reports `motion.speed` (`speed`, `speed_m_s` when given,
`slow_approaching`, the caps `linear_m_s`, `angular_deg_s`,
`joint_speed_deg_s`, `planned_peak_linear_m_s`, `planned_peak_angular_deg_s`,
and with `slow_approaching` `slow_approach_m` and the `slow` caps),
`motion.wrist`, `planned_ticks` and `duration_s`; the keys of `motion.speed`
are RoboDojo's. A move that ran its jaw twin adds `motion.wrist.jaw_flip_retry`,
and a wrist refusal whose twin was tried adds `motion.refused.jaw_flip_retry`
([the wrist](#the-wrist)); both reach the policy with the rest of `motion`.
On RoboTwin `motion.wrist` holds one report per moved arm, under
its name, the peaks are the largest over the arms, and `motion.timing` holds one
entry per moved arm. A move with no paced
route (a local route, RoboLab's servo) reports `motion.speed` with the request
and its caps only; RoboLab's leaves out the joint cap, which does not apply to
it.

The sealed receipt (`native/receipts/`) also carries `motion.timing` and each
segment's `pacing.timing`: `implementation`
(`toppra_ruckig` or `motion_pace`), and for the worker the library `versions`,
`jerk_source`, `attempts`, `acceleration_scale`, `ruckig_extended_segments`,
`max_deviation_m`, `solve_ms` and `verification`, each cap's largest ratio over
the ticks; for the fallback, `reason`. The run's metadata holds
`motion_timing_binding`.

Nothing the policy can read names a library. Every copy of a receipt in a place
the policy can read (the observation an environment publishes, the policy-state
chain, the context a session reads, a script's robot view built from it) is
`ora.contracts.policy_receipt` of it: without `motion.timing`, `pacing.timing`,
and the planner's own record, `planner.schema` and `planner.canonical_artifact`.
The public projections never read those keys, so the delivered text is the same.
A backend embeds `policy_receipt(receipt)` in its observation (LIBERO, RoboTwin,
RoboCasa365 and RoboLab do); the Supervisor
refuses an observation whose embedded receipt still holds timing evidence. The
copy the session keeps for its trace (`native/agent-artifacts/`, kind
`environment_receipt`) is not a Context Artifact, so `artifact_read` cannot open
it (`tests/ora/agent/evidence/test_policy_reachable_receipts.py`).

## Backends

| Backend | Adoption |
|---|---|
| RoboDojo (default) | Every route of the toolkit's motion tools (`move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`, `grasp_execute`'s arm stages, `recover_pose`, `return_home`) is timed through `kinematics.pace_path`, one row per native action, in the worker. The interaction label picks the route at every speed, as on every other backend: `speed` and `speed_m_s` change only the pace, and `expect_contact` is the whole straight line at any length; see below |
| LIBERO / LIBERO-Pro (default) | Every global planner route (`move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`, `recover_pose`, an executed probe plan) is paced in the worker, by the timing worker; see below |
| RoboTwin (default) | Every validated global plan (`move_pose`, `move_to`, `rotate_wrist`, `rotate_pitch`, `recover_pose`, the stages of `grasp_execute`) is paced in the worker; both arms of a two-arm move share one timeline; a local DLS stage spreads each servo step over the physics steps the caps need; see below |
| RoboCasa365 (default) | Every admitted arm path is paced in the worker, the interaction's per-tick steps staying the ceiling; `move_base` is outside it; see below |
| RoboLab (default) | A planned path is timed and wrist-checked in the worker. On the Cartesian servo (a contact approach, a short move) the speed caps the step per tick (`step_caps`); see below |
| YAM station (`runtime.motion_timing`, off by default) | Every arm motion (`move_pose`, `move_free_space`, `recover_motion`) is timed by the worker on the station's own IK path, under its interaction's caps, before the per-tick checks; without the switch the station's fixed-speed timing; see below |

### RoboDojo

The toolkit lib holds the path and the `Pace`; the environment times it.
`lib.motion_plan.timing.timed` sends each arm's joint path, one stretch between
forced stops at a time, as `kinematics.pace_path` (`speed` in the explicit wire form
`{"speed": "explicit", "caps", "slow_caps", "slow_approaching"}`, so a tool of its own
may use any `Pace`), and `environment/backends/robodojo/shared_motion.py` answers with
`motion_pace.pace_path`: 40 ms tick, wrist `joint6` limits +/-3.14 rad, robot `arx_x5`.
The answer is a recorded kinematics answer, so a replayed tool rebuilds its action
without a timing worker, and it names no library: the `timing` evidence of each move
stays in `native/motion-planner/timing.jsonl`. `ORA_TIMING_PYTHON` is bound by
preflight as on RoboTwin (`runtime.motion_timing`), for a free-play scene as for a
benchmark episode.

Two arms are timed one by one and put on one timeline: the shorter move is slowed
uniformly, which keeps every cap, and they start and stop together. Under a budget
each arm is cut on its own, so two arms may stop at different fractions of their
paths; `part_of_move` reports the smaller. The wrist check runs on every route but
`recover_pose` and `return_home`, whose paths unwind on purpose. A joint-line bridge
over an IK hole is no longer whole under a budget cut (large wrist turns are).

The hand-written timing the toolkit used to hold had two defects this removes: its
slow segment could run over speed on planner grids, and its ramp was skipped where
the grid was coarse. On 245 recorded paths of 114 toolkit runs (2026-10-31):

| Speed | Ticks, median (hand-written to library) | Peak TCP speed, largest | Joint speed p99 |
|---|---|---|---|
| `normal`, straight lines | 15 to 15 | 0.300 to 0.291 m/s | 2.43 to 2.23 rad/s |
| `fast`, straight lines | 5 to 9 | 1.61 to 1.03 m/s | 3.97 to 3.81 rad/s |
| `fast`, executed moves | 6 to 9.5 | 1.68 to 1.24 m/s | 3.98 to 3.98 rad/s |

`fast` takes about 1.7 times as long at the median: the X5 has no published jerk
limit, so the moves are certified at the unverified default (1000 rad/s^3,
[jerk limits](#jerk-limits)), and at a 40 ms tick that lowers the acceleration until
Ruckig can follow. Of the `fast` paths, 16 of 245 keep the fallback timing for that
move (`timing_infeasible`), as on RoboTwin.

#### RoboDojo motion profiles

What the toolkit runs at `fast`, and how its jaw commands move, is the ARX X5
motion profile the plan binds (`runtime.motion_profile`,
`ora/policy/embodiments/arx-x5/motion-profile-v<N>.json`,
[binding](configuration-and-evidence.md#robodojo-motion-profile)); `normal`,
`careful`, `speed_m_s` and the recovery retrace never follow it. A plan without
one runs v1, the toolkit's built-in values.

| | v1 (plans before 2026-11-15) | v2 (the default since) |
|---|---|---|
| `fast` joints | 4 rad/s, 40 rad/s², the X5's unverified 1000 rad/s³ | 4.5 rad/s (90% of the X5 actuator's 5 rad/s `velocity_limit_sim`), 60 rad/s², 5000 rad/s³ |
| `fast` settle | up to 5 native actions to 0.003 rad, 0.05 rad/s | the same |
| close | 2 openings/s, at most 5 mm lead, one settle look at the arm | the SDK limiter's full drive (10.8 mm lead), 5 mm once the finger stops (`hold_lead_m`), no look at an arm whose measured joint velocities are already below 0.05 rad/s (`arm_settle.first_reading`) |
| open releasing a close | 2 openings/s | full drive |
| end of a jaw command | within 0.03, or 3 still native actions | within 0.03, or 2 still native actions |

Why v2's `fast` names a jerk: on 12 sealed RoboDojo tool-call runs (2026-11-14/15,
7574 native actions, 78% of them `follow_joint_trajectory`), the fast moves'
planned peak joint speed was a median 1.9 rad/s and their peak acceleration a
median 14 rad/s²: the timing worker had backed the 40 rad/s² cap off to a median
0.51 of itself (4 attempts) before Ruckig certified the rows at 1000 rad/s³.
Re-timed offline by the same worker (each recorded path, `timing_infeasible`
ones left out), raising the caps alone barely helps: 244 paths need 2545 planned
native actions at v1's caps and 2479 at 4.5 rad/s and 60 rad/s² with the old
jerk (1.03x fewer); with 5000 rad/s³ as well, 258 paths need 1860 instead of
2728 (1.47x).
The X5 publishes no jerk; 5000 rad/s³ is within the 3750 to 10000 Franka documents
for its joints. Jaw commands spent their native actions travelling: a close at
2 openings/s took about 6 of its 10, then 3 to confirm a stall and 1 settle look,
and the arm was already at rest at every one of the 75 recorded closes.

Before and after, on the first 6 to 8 actions of five sealed RoboDojo runs
(`arrange_largest_number_random`, `arrange_largest_number`, `align_blocks`,
`general_pickup`, `fold_clothes`, layout 3, CUDA physics), each run once as
recorded and once under v2 from a fresh round of the same plant (2026-11-15).
Every recorded action replayed with its recorded native actions and final joints.

| Action | n | v1 native actions | v2 |
|---|---|---|---|
| fast move | 24 | 299 (12.5 each) | 218 (9.1 each) |
| close stopping on an object | 5 | 62 (12.4 each) | 28 (5.6 each) |
| close reaching the floor | 1 | 12 | 5 |
| open | 7 | 43 (6.1 each) | 23 (3.3 each) |

Every action ended as it did under v1 (the same execution status and failure
code), every close on an object stalled within 0.006 of v1's opening, and no
guard stopped a move. The largest per-move tracking error rose from 0.0021 rad
to 0.0094 rad, under the bound's 0.012 rad floor; a v2 move settles one native
action more often (the arm arrives with more lag).

### LIBERO

`planned_motion._pace_global_route` hands the planner's trajectory to
`pace_path` with forward kinematics of the grip site, read from a scratch copy
of the measured MuJoCo state (`harness_grip_site_fk`,
`robot_asset_geometry.panda_poses_at_joints`; the plant never moves), the
arm's joint limits from the loaded model and the robot `franka_panda`.
The default baseline's `runtime.motion_planner.timing` selects the timing worker
under `ORA_TIMING_PYTHON`, which `ora run --env libero` requires. Preflight starts
it (check `motion_timing_worker`), and the isolated LIBERO worker, whose
environment is allowlisted, gets the interpreter preflight started from the
preflight binding (`motion_timing`: its path and sha256, checked again when the
worker starts it). A run without the field keeps the fallback, whatever the
environment names. One knot per 50 ms control tick;
every joint at most what the executor's controller follows
(`hybrid_motion.libero_global_joint_ceilings`): under motion profile v3, the
default, 0.55 rad/s and 15 rad/s² at every speed (each preset's joint caps are
above or, for `careful`'s 8 rad/s², below them; v2 kept 1 rad/s); the wrist joint is
`panda_joint7`. The whole route must fit `max_waypoints` (2000) and
`max_execution_ticks` less the terminal settle; a longer one is refused
whole, never cut, so a move never stops part way through a wrist turn. The
`JOINT_POSITION` executor runs the knots `timed`: every knot gets at least one
tick, the 0.04 rad tolerance and the 8 ticks per knot still bound the lag, and
the terminal settle is unchanged. A route whose path does not move keeps the
untimed rule (no tick). A wrist refusal of a pose move's global segment retries
its jaw twin with the jaws open (`planned_motion._jaw_twin_route`,
the same plan request in the same world snapshot); the twin's plan is kept beside
the first as `curobo/plans/action-NNNN-jaw-twin.json`, and once it ran, the move's
settle, its local approach after a pre-contact pose and its pose error read the
twin of the target.

`move_pose`, `move_to`, `rotate_wrist` and `rotate_pitch` carry the request,
from the runtime handlers or from the toolkit tools, which send the same wire form
where the environment plans (`_smooth.proposed_speed`);
`recover_pose` and an executed probe plan run at the default. A move's
pre-contact segment (a staged `contact_approach`) leaves out the slow final
segment, which then belongs to its local approach. The bounded local OSC
segments are not timed but capped: each tick's commanded translation and
rotation are scaled to the caps times 50 ms (`step_caps`; within the last
2 cm the careful caps with `slow_approaching`), without ramps, and the
segment's loop budget grows by the fewest ticks the caps allow. At the default
speed the local segments are unchanged.

On the 219 recorded global plans of the planner comparison (Panda forward
kinematics, v2's 1 rad/s ceiling), the default pace takes 9 ticks (0.45 s) at the median and 69 at
most, `normal` 18 and 73, `careful` 37 and 200; at the default the grip site
peaks at about 1 m/s, which only the joints limit. Three plans are refused as
unrequested wrist turns.

#### The joint executor

The motion profile the plan binds (`runtime.motion_profile`) names the
RoboSuite `JOINT_POSITION` controller the global route runs on.
`hybrid_motion.libero_joint_controller` reads it and
`worker_environment.joint_position_controller_config` builds it, for a live
segment and for its replay alike:

| | v2 (frozen plans) | v3 (`motion-profile-v3.json`, the default) |
|---|---|---|
| Controller | RoboSuite's file: kp 50, damping ratio 1 | kp 50 (`joint_controller_kp`), damping ratio 1 |
| Reach: farthest the goal sits from the measured joints | 0.05 rad | 0.2 rad (`joint_command_step_rad`, RoboSuite's `output_max`) |
| A knot's first tick aims at | the knot | the knot plus `2 / sqrt(kp)` (0.283 s) times the knot's speed (`lead_joint_targets`: central differences, the last knot at rest) |
| Pacing ceiling | 1 rad/s (one 0.05 rad step a tick), no acceleration cap | `joint_follow_speed_rad_s` 0.55 rad/s, `joint_follow_accel_rad_s2` 15 rad/s² |
| Terminal settle ends | every joint within 0.005 rad of the goal and the grip site within 15 mm / 5 degrees | that, or no joint moved more than `terminal_settle_rest_step_rad` (0.002 rad) over the last tick and the grip site is within the local route's 5 mm / 0.05 rad (`libero_settle_at_rest`) |

Each command is the aim's offset from the measured joints divided by the reach
and clipped to ±1, so the controller's goal is the aim itself whenever it lies
within the reach. A critically damped controller follows a knot that moves at
`v` with a lag of `(2 / sqrt(kp)) v`; the lead removes that lag, and v3 admits a
follow speed only where the lead stays inside the reach (`2 / sqrt(kp)` times
0.55 rad/s is 0.156 rad of 0.2). The knot, not the aim, still gates progress
(0.04 rad, 8 ticks); a knot the arm has not met after its first tick is aimed at
itself, so the path holds there; the terminal settle aims at the goal. A settle
that ends at rest inside the local bound leaves nothing for the sparse move's
local settle (`global_route_met_the_local_bound`), so the move ends there; the
settle's `settled_by` and per-tick `trace` say which rule ended it.

Under v2 the pacing allowed 1 rad/s, but a goal at most 0.05 rad ahead holds a
kp 50 controller near 0.008 rad per tick, 0.16 rad/s: every knot waited about 6
ticks. On 18 LIBERO runs of ORA `559102a58` under PEAR's step limits, the first
move of an episode, from LIBERO's start posture, took 50 to 305 control steps
(median 164.5) for 18 to 58 paced knots, 5.3 times its plan, 10 of them in the
terminal settle. The same first moves, from the same starts, through the real
worker (`ora qualify libero motion`'s session, no policy):

| First move (seed 0) | Speed | v2: knots, steps (sealed run, replay) | v3: knots, steps (knot ticks + settle) | v3 end residual |
|---|---|---|---|---|
| `libero_spatial-t02` | normal | 24, 107 (107) | 33, 41 (33 + 8) | 2.3 mm, 0.002 rad |
| `libero_spatial-t03` | normal | 58, 305 (305) | 82, 101 (98 + 3) | 3.8 mm, 0.016 rad |
| `libero_object-t00` | fast (default) | 38, 248 (248) | 66, 73 (67 + 6) | 2.6 mm, 0.016 rad |
| `libero_goal-t04` | normal | 42, 221 (221) | 60, 75 (71 + 4) | 3.5 mm, 0.010 rad |
| `libero_spatial-t07` | normal | 19, 50 (50) | 19, 24 (19 + 5) | 4.9 mm, 0.002 rad |
| `libero_object-t09` | fast (default) | 35, 233 (233) | 61, 75 (69 + 6) | 2.3 mm, 0.015 rad |

v2 repeats the sealed runs' step counts and end poses exactly. The settle ended
at rest in all six v3 moves (3 to 8 ticks; under the v2 rule the same kp 50 moves
needed 7 to 10). The controller's unclipped torque, measured on every physics
substep (`torque` in `tracking/joint-action-NNNN.json`), stayed inside the
actuator range RoboSuite clips to (80 N m for joints 1 to 5, 12 N m for 6 and
7) on these moves and on the 14 qualification cases: at most 66.6 N m on joint 2
and 7.0 N m on joint 7, no substep clipped. kp 150 (1 rad/s) was measured and
refused: with 15 rad/s² it reached 90 N m on joint 2 and 20 N m on joint 7 and
stopped two first moves on `waypoint_tracking_timeout`; with 7.5 rad/s² it
clipped 1245 of 5700 qualification substeps. A v3 run records the gains in its
controller replay (`metadata.joint_controller`) and a replay loads exactly them;
a run bound to v2 records none and replays on RoboSuite's defaults, bit for bit
as before.

#### The straight OSC line (v4)

Motion profile v4, the default for new LIBERO plans, is v3 with an `osc_transit`
section. A planned `free_space` move whose plan keeps the grip site within
`max_path_deviation_m` (2 cm) of the straight line from the start to the target,
and whose hand turns at most `max_rotation_rad` (45 degrees), runs along that
straight line on LIBERO's OSC controller instead of the joint path
(`controllers/osc_transit.py`, `planned_motion._free_space_route_choice`). The
deviation is read at the paced knots, from the measured start, by the grip-site
forward kinematics the pacing uses. Every other planned move keeps the joint path:
a pre-contact segment or a recovery, a move with `slow_approaching`, an executed
`plan_ref` (it runs the plan its dry run showed), a larger turn, or a plan that
bows out further.

This changes what the plan vouches for. The planner checked its own joint path,
the whole arm on it; on the line the controller chooses the arm's posture, so of
the line only the gripper's path lies within 2 cm of what the planner checked.
The arm's links are not checked there. A force the wrist sensor measures above
the 40 N telemetry threshold (`sensing.force_telemetry_threshold_n`) stops the
line where it stands (`osc_transit_force_limit`); a contact on an arm link that
does not reach the wrist sensor is not seen.

The line is paced at the request's Cartesian caps, no faster than the OSC
controller follows: 0.25 m/s and 1.5 rad/s (`follow_speed_m_s`,
`follow_angular_rad_s`), reached at 2 m/s² and 12 rad/s². Each tick aims at the
line's pose `2 / sqrt(osc_kp)` (0.163 s, kp 150 of LIBERO's `osc_pose.json`) ahead
by its own speed, within the controller's 0.05 m / 0.5 rad reach, and the local
OSC loop then settles onto the target within the local 5 mm / 0.05 rad bound, at
most `settle_ticks` (20) more ticks. `fast` and `normal` therefore differ little
here (both are capped at 0.25 m/s; `normal` ramps at 1 m/s²), and on the joint path
both meet the 0.55 rad/s joint ceiling first; `careful` keeps 0.1 m/s on both.

The move's sparse receipt says which route ran and why (`arms.right.route_choice`),
its `motion` gives the line's `planned_ticks` and its controls are the
`libero_osc_pose` entries of the controller replay, so a replay is bit for bit as
for any local move. A dry run answers with the same choice: `route: osc_line`
and the line's ticks.

Measured through the real worker from the same six starts as above. Every action
each sealed run submitted, re-executed in order from the same start (open loop,
no policy), took:

| Run | Actions | v2 steps (= sealed) | v3 steps | v4 steps (OSC lines) |
|---|---|---|---|---|
| `libero_spatial-t02` | 6 | 235 | 160 | 118 (2) |
| `libero_spatial-t03` | 1 | 305 | 101 | 101 (0) |
| `libero_object-t00` | 4 | 334 | 118 | 117 (1) |
| `libero_goal-t04` | 4 | 325 | 139 | 137 (1) |
| `libero_spatial-t07` | 6 | 247 | 117 | 94 (3) |
| `libero_object-t09` | 4 | 291 | 121 | 121 (0) |

Of the six first moves, from LIBERO's start pose at 0.18 to 0.30 m, one ran as
the line (`libero_spatial-t07`, 0.18 m, no turn: 24 ticks); four turned the hand
102 to 145 degrees and one's plan bowed 2.3 cm off the line, so they kept the
joint path. Their median cost was 74 ticks at `fast`, 74 at `normal` and 113 at
`careful`, the numbers the `move_pose` text publishes
(`motion_settle.LIBERO_FIRST_MOVE_TICKS`). All six later free-space moves that
moved ran as the line, at 9 to 33 ticks where v3 took 11 to 73. The 14-case qualification
passed with the two free-space cases on the line (10 cm in 23 ticks, a 45-degree
combined move in 16; v3: 19 and 35).

### RoboTwin

`ora/environment/backends/robotwin/pacing.py` hands each validated plan to
`pace_path` with the public TCP's forward kinematics
(`kinematics._pinocchio_public_pose_xyzw`), before the dense control sequence is
built. One knot per physics step (4 ms); the wrist joint is `joint6` of the
moved arm; the worker certifies its jerk as `aloha_agilex`, at the unverified
default (`pacing.ROBOT`). The default baseline's `runtime.motion_timing` selects
the timing worker under `ORA_TIMING_PYTHON`, which `ora run --env robotwin` requires;
preflight starts it (`motion_timing_worker`), and the worker takes the interpreter
from that check in the run's `preflight.json` at its first round
(`motion_timing.bind_preflight`), refusing a changed sha256. The joint ceilings are
1.9 rad/s and 1.1 rad/s², set so that the
default is no faster than the planner's own timing it replaces (below). The
whole route must fit `plan_validation.maximum_waypoints`
(5000) and the transaction's physics steps less one waypoint's wait; a longer
one is refused whole. The measured-convergence transaction writes the knots as
its stages, each at least one physics step, so a stage that waits only slows
the move. The knots' velocity targets are their central differences: the
joint drives (stiffness 1000, damping 200) track a moving target through their
velocity term, so a wrong target would add lag.

Each arm of a two-arm move is paced alone, then the shorter move is slowed
uniformly onto the longer one's timeline, which keeps its caps; because pacing
changes when each arm is where, the paced pair goes through the inter-arm
clearance check again before anything moves. A pre-contact segment leaves out
the slow final segment, as on LIBERO. `recover_pose` and the stages of
`grasp_execute` run at the default.

A local DLS stage follows no planned path. The servo re-anchors on the measured
joints every iteration and the drives lag their targets, so shrinking the step
would only slow it without bound; instead each iteration's full step runs as a
straight joint line over as many physics steps as its predicted TCP translation
and turn need under the caps (`pacing.local_substeps`; within the last 2 cm the
careful caps with `slow_approaching`), with that line's velocity and no ramps. At
the default speed the local stages are unchanged.

On 6903 recorded arm moves of 394 sealed runs (planner rows through real
URDF kinematics, 0.6 µm from the recorded TCP; 2026-10-31), the planner's own
timing reaches 1.97 rad/s at the joints at p99 (3.1 at most) and 1.54 m/s at
the TCP at most. The vendored planner's bounds as ceilings (3 rad/s, 3 rad/s²)
made the default 0.79 of the planner's duration at the median with 1.53× its
peak speed, because the planner's profile stays below the acceleration it plans
with. At 1.9 rad/s and 1.1 rad/s² the default takes 347 physics steps (1.39 s)
at the median and 785 at p95, 1.31× the planner's duration (at least 1.23× on 95%
of moves); per move its peak TCP speed is 0.93× the planner's at the median and
1.0× at p95, and every percentile of joint and TCP speed stays below the
planner's (joint p50 0.78 against 0.81 rad/s, p99 1.85 against 1.97, TCP at most
1.16 against 1.54 m/s). `normal` peaks at 0.301 m/s, `careful` at 0.100 and
`speed_m_s` 0.05 at 0.051, within 2% of each cap. The linear ramp shows steps at
grid nodes on raw 4 ms ticks where the joint term binds; averaged over 40 ms it
holds at p99 (0.61 m/s² for `normal`'s 1, 0.50 for `careful`'s 0.5). 124 moves
(1.8%, in 70 runs) are refused as unrequested wrist turns, up to 457 degrees of
path turn; the largest accepted detour is 44 degrees. Knots lie on chords
between grid nodes up to 0.02 rad apart: 0.1 mrad at p95 from the planner's
polyline, 5 mrad and 0.84 mm of TCP at most, the goal exact. Tracking in
simulation is not measured by this offline pass.

Through the timing worker (`ORA_TIMING_PYTHON`, the same moves, `aloha_agilex`
at the unverified 1000 rad/s³) the default takes 1.41 s at the median, 1.33× the
planner's duration (at least 1.25× on 95% of moves); its per-move peak TCP speed
is 0.92× the planner's at the median and 0.98× at p95, joint p99 1.82 rad/s and
the TCP at most 0.85 m/s. `normal`, `careful` and `speed_m_s` 0.05 peak at 0.301,
0.1005 and 0.0501 m/s; the same 124 moves are refused as wrist unwinds and 6 at
0.05 m/s by the budget. 54 of the 6779 default moves (0.8%) are
`timing_infeasible` (Ruckig cannot follow them at that jerk) and keep the
fallback timing above, which says so in its `motion.timing`; over all five speeds
9 timed moves needed a lowered acceleration scale (0.8 or 0.64).

### RoboCasa365

`ora/environment/backends/robocasa/motion.py` (`pace_arm_path`) hands the
admitted joint path to `pace_path` with grip-site forward kinematics
(`kinematics.grip_poses`) at the plan's torso height, one knot per 50 ms control
tick, wrist joint `robot0_joint7`; the worker certifies its jerk at the Panda's
published limits (`franka_panda`), bound as on RoboTwin (`runtime.motion_timing`,
`motion_timing.bind_preflight` at `start`). Each joint keeps the interaction's per-tick
step as its ceiling (`ARM_JOINT_STEP_RAD`: 1, 0.5 or 0.24 rad/s). The
interaction's grip-site step (`ARM_STEP_TRANSLATION_M`, `ARM_STEP_ROTATION_RAD`)
holds too: a timing that breaks it is timed again at a lower joint ceiling, so
`joint_speed_deg_s` then reports the lowered cap. The path must fit 130 ticks,
the action's 150 less the 20-tick settle allowance; a longer one is refused
whole. `follow_path` streams one knot per tick, and arrival, blocked and contact
rules are unchanged.

Pacing runs before the torso moves, so a refusal spends no native action, and a
refused move does not replace `recover_pose`'s `previous`. A wrist refusal of a
pose move retries its jaw twin with the jaws commanded open (`motion.jaw_twin_path`):
the twin is admitted on the arm alone at the measured torso height and paced
like the request. After a torso move the
re-solved path is timed again at the measured height; if that timing is refused
the admitted timing stands, because the lift has already moved. `recover_pose`
runs at the default. `move_base` is outside the speed options.

The joint paths are not recorded, so the offline pass (2026-10-31) reconstructed
79 moves of 12 sealed runs as straight joint lines between the receipts' measured
joints, skipping 37 lift-and-traverse moves; straight-line IK paths are only
approximated by that. Speeds hold at 0.298, 0.100 and 0.050 m/s for `normal`,
`careful` and `speed_m_s` 0.05, ramps within 1%. Eight moves (10%) would be
refused as unrequested wrist turns, one of them an approximated IK line, so this
number is an estimate of the live rate. Through the timing worker (`franka_panda`)
every move certifies at the first attempt; peaks stay at 0.97 of each cap (the
worker holds caps at 97%), the default takes 0.70 s at the median, and the same
8 moves are refused as unrequested wrist turns (3 at `careful` and 5 at 0.05 m/s
by the budget).

### RoboLab

A move that plans (`environment/backends/robolab/shared_motion.py`: `free_space` or
`precision`, more than 50 mm or 15 degrees) is paced by `pace_path` at the 15 Hz tick
with the `franka_panda` jerk table, joint ceilings of 1.9 rad/s and 2.5 rad/s2 and the
wrist check on `panda_joint7`, one row per native action; the timing worker is bound
from the run's preflight, as on RoboTwin. A refusal (`paced_path_wrist_unwind`,
`paced_motion_budget` above 300 ticks) moves nothing; a wrist refusal of a pose
move with the jaws commanded open first retries its jaw twin
(`shared_motion.pace_or_twin`), and the servo then settles on the twin. The rows go to the arm as joint
targets, and the servo settles the last millimetres.

The servo, which takes a contact approach, a short move and a move the planner cannot
answer, has no joint path: RoboLab's differential IK solves inside the simulator, one
tick at a time. A request lowers the carrot step toward the next waypoint to the caps
over one 15 Hz tick (`step_caps`), below the interaction's own step and without ramps;
the slow zone counts the path left through the remaining waypoints, and the action's
tick budget grows by the fewest ticks the caps allow. At the default nothing is
capped. No path is timed there, so no wrist check runs on the servo route, and the
help says so.

### YAM

The bimanual I2RT YAM station (`environment/backends/yam/`) plans its own paths: each
arm's Cartesian line (a free-space move lifts, traverses and descends) sampled every
2 mm and 0.01 rad and solved by warm-started IK. Its own timing gives each sample the
time the strictest of the interaction's TCP speed, turn rate and the 0.6 rad/s joint cap
allows, and interpolates linearly at the 50 Hz control tick: a motion starts and stops at
full speed, with no acceleration or jerk limit. That timing stays the default.

`runtime.motion_timing` (`ora play --motion-timing`;
`{"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}`, nothing else
admitted) hands the same samples to the worker (`backends/yam/shared_motion.py`). The
request is the explicit wire form with the station's caps per interaction
(`controllers/yam_station.py`), each reached from rest in `TIMING_RAMP_S` (0.25 s, the
ramp of a numeric speed request, declared rather than measured on the arms):

| Interaction | TCP | Turn | TCP ramp | Joints |
|---|---|---|---|---|
| `free_space` | 0.12 m/s | 0.6 rad/s | 0.48 m/s², 2.4 rad/s² | 0.6 rad/s, 2.4 rad/s² |
| `precision` | 0.05 m/s | 0.3 rad/s | 0.2 m/s², 1.2 rad/s² | 0.6 rad/s, 2.4 rad/s² |
| `contact_approach` | 0.025 m/s | 0.2 rad/s | 0.1 m/s², 0.8 rad/s² | 0.6 rad/s, 2.4 rad/s² |

The jerk is certified at `yam`'s unverified 1000 rad/s³ ([jerk limits](#jerk-limits)).
Each straight piece of the path is timed on its own, rest to rest, so the arm stops at
the lift's and the traverse's corners and its rows pass through them exactly; timed as
one path, a lift-traverse-descend corner (60 to 150 degrees in joint space) is one
Ruckig cannot follow at that jerk. `recover_motion`'s joint line is timed for the arm
that travels furthest, at half the joint cap (0.3 rad/s, 1.2 rad/s²), and the other arm
follows the same progress. The wrist is reported, not refused: the path is the
station's own line to the requested orientation. A path needs at most
`TIMING_BUDGET_TICKS` (6000 ticks, two minutes), else `paced_motion_budget`.

The rows replace the fixed-speed ticks before the per-tick checks, so the joint limits
(widened to a start just past them, as at rest on the cell), the table clearance and the
inter-arm distance are checked on the ticks the arms will follow, and a dry run keeps
the timed schedule for the real call. The worker is bound as on RoboTwin: preflight
starts it (`motion_timing_worker`) and the session reads that check from the run's
`preflight.json` at start; a direct session takes `ORA_TIMING_PYTHON`, and without
either the station times its moves itself. `native/timing/binding.json` names what was
bound. A receipt then carries `motion`: `speed` and `wrist` per arm, `planned_ticks`,
`duration_s` and, in the sealed receipt only, `timing` per arm (`implementation`, and
the worker's evidence of each piece under `segments`, or `yam_fixed_speed` and its
`reason`). Without the switch no receipt has `motion`, so nothing the model reads
changes.

With `runtime.motion_planner` (`ora play --motion-planner`) the path is the shared
planner's instead (`backends/yam/shared_planner.py`), and either timing times it the same
way: a free-space plan is one piece, rest to rest, and a precision or contact line's
knots lie 1 cm (or 0.05 rad of turn) apart. A receipt then carries `motion` with or
without the timing switch, and its sealed copy alone holds `motion.planned_by`, which says
per arm what planned it (`curobo_v2` and its route `plan` or `line`, or `yam_mink_ik` and
why).

Measured on 150 random admitted moves on the station model (2026-11-14, both arms,
104 single-segment and 46 lift-traverse-descend, worker on this workstation's CPU),
every move timed by the worker at the first attempt, reaching the same final joints
exactly:

| | Fixed-speed timing | Shared timing |
|---|---|---|
| Duration, `free_space` p50 / p95 | 2.40 / 3.19 s | 3.16 / 4.07 s |
| Duration, `precision` p50 / p95 | 1.56 / 2.14 s | 1.86 / 2.46 s |
| Duration, `contact_approach` p50 / p95 | 1.15 / 1.71 s | 1.45 / 2.01 s |
| Duration ratio p50 (free / precision / contact) | 1 | 1.31 / 1.19 / 1.25 |
| TCP speed at the first tick, over the cap | 1.00 | 0.04 |
| Peak TCP speed over the cap, max | 1.005 | 0.970 |
| Joint acceleration p50 / max | 13.4 / 59.7 rad/s² | 1.10 / 2.33 rad/s² |
| Joint jerk p50 / max (finite differences) | 672 / 2985 rad/s³ | 38 / 169 rad/s³ |
| Joint speed max | 0.600 rad/s | 0.582 rad/s |

A solve takes 5 ms at the median and 15 ms at most; planning a whole motion with it,
IK included, 65 ms at the median and 1.3 s at most (the first, which starts the worker). Tracking on the
real arms is not measured yet; that measurement decides whether the switch becomes the
default.
