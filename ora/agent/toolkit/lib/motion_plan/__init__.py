"""Arm motion planned in lib, executed by the environment as one ``follow_joint_trajectory``.

Parts (pure computation except ``kinematics``, which reads the backend's
recorded read-only kinematics query): ``paths`` (straight line, joint line,
grids), ``timing`` (speed and acceleration caps, knots every native action),
``kinematics`` (IK chains, FK and path screens through
``api.planner.kinematics``), ``checks`` (continuity, IK holes, budget fit),
``route`` (move_pose's interaction route as a planning rule), ``planner``
(one move from measured poses to targets), ``execute`` (the bound action
and its compact result; a rate-limited jaw command), ``move`` (a whole move
with any ``timing.Pace``: plan, fit, screen, run), ``recovery``
(recover_pose's plans: back along the last move's measured path, a screened
joint unwind, the home lift) and ``orientation`` (the jaw-symmetry rule: the
half-turn twin nearer the wrist). See ``lib/README.md``.
"""
