"""Why the YAM station still plans its own motions: its written, dated exception to the shared motion base.

docs/architecture.md; tests/repository/test_motion_exceptions.py reads this literal without importing the backend.
"""

MOTION_EXCEPTION = {
    "schema": "ora.motion-exception.v1",
    "backend": "yam",
    "what": "motion.py plans the station's arm motions itself (mink IK along a sampled Cartesian path, the station's own speed constants and per-tick checks) instead of the shared cuRobo V2 planner and TOPP-RA + Ruckig timing worker",
    "reason": "The real cell joined after the shared base and was never moved onto it. The shared timing and planner are being brought to the station behind default-off runtime switches (motion_timing, motion_planner), to stay off until they are qualified on the real cell with the operator present.",
    "owner": "freemty",
    "recorded": "2026-11-15",
    "expires": "2026-12-31",
    "ends_when": "both switches are qualified on the real cell and become the default; motion.py's planner is then removed",
}
