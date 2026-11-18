# move_pose

Measured-frame pose compilation, IK references and one action proposal.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool move_pose`.

The tool resolves flat sparse targets, world axes and jaw symmetry, reports
sampled reach and observed depth clearance, and runs the compiled target as one
smooth joint trajectory at the call's `speed` (`_smooth.py`: the interaction
picks the path through `lib.motion_plan.route` at every speed; fast, the default,
normal and careful set only the pace, fast's from the context's motion profile
(`_profile.py`: `runtime.motion_profile`, v1's built-in pace without one);
`speed_m_s` instead caps the controlled point at any m/s, with v1 fast's joint caps
as the only other limit, and
`motion.speed.planned_peak_linear_m_s` reports what the plan reaches;
`expect_contact` follows the whole straight line at any length, refused rather than
replaced by another path; `slow_approaching` slows the last 2 cm of each arm's
path to the careful pace inside the same trajectory). Planning, budget
fit, screening and execution are `lib.motion_plan.move`, which a tool of its
own can call with any `Pace`. Without a backend that executes joint trajectories, and on the
manifest surface v2, it submits the original move_pose action schema once. Coordinates are world metres and xyzw; axis-angle input is
degrees, joint limits are radians. Task policy stays in local helpers.

The reach reference (`_reach.py`) answers from the read-only reach check and never blocks a move. On a surface declaring the
v2 generation (`_reference.py`) every annotation names the IK that answered and whether it is this move's admission.

Replay consumes the recorded observation/IK sequence. A new target whose IK
was not recorded fails NotRecorded; a changed action at the submission cursor
fails Divergence. Recorded advisory IK failures keep their public failure type.
The trusted adapter alone validates and publishes submitted-action.json.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.

The toolkit-v2 execution surface calls `api.act.execute_action` synchronously and
returns the public receipt and remaining parent budget. Controller stage evidence
is stripped from the tool result. The toolkit-v1 submission path is unchanged.

`dry_run` (code mode, `library: extended`; `interface/dry_run.py`): `tool.run` under `dry_run.dry()` compiles the request and
plans it exactly as the real call does (`_smooth.plan_action`), then returns `tools/_dry_run.planned` instead of
executing: the plan's expected end pose, IK solution, route, screen and native actions. The tool's arguments and schema do
not change; `move_to`, `rotate_wrist` and `rotate_pitch` share the path (`tool._dry`). Where the surface does not run
joint trajectories (`_smooth.available`) the environment plans the move when it executes, the tool stops at
`dry_run.Proposed` and the environment answers for the action. A plan the real call would refuse is the dry run's
answer (`_dry_run.answered`).

Where the tool context says `motion.environment_plans` (`_context.environment_plans`) move_pose and the tools compiled
through it propose one action: flat arguments read as the right arm's request (`tool.surface_targets`), the speed options
go in the action (`_smooth.proposed_speed`), `expect_contact` and `exact_orientation` stay unknown fields, and the result
is `status: submitted`. A context without the key is RoboDojo's, every sealed run's among them, so this path adds no read
to the sequence those runs replay.
