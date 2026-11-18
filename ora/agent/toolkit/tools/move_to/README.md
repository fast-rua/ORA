# move_to

Translation-only motion with the measured orientation held.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool move_to`.

This entry composes move_pose's local compilation with the move_to policy:
position_m is absolute world metres, interaction is free_space, and orientation
is the current measured quaternion. One parent call proposes one action.
Replay reads recorded observation/IK; missing queries fail, changed submissions
diverge. It never executes online motion.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.
