"""The move_pose mode tables.

A flat ``move_pose`` target names a position mode and an orientation mode, and
each mode requires its own companion fields. ``toolkit/host/motion_rules/arguments.py``
reads a target by these tables. The help that states them
(``motion_mode_companion_help_v1``) is in each surface's ``move_pose`` file, as
the binder that compiled it from these tables wrote it.
"""

from __future__ import annotations

#: Flat keys, grouped by the mode that requires them. Stated once so the reader,
#: the writer and the refusal cannot disagree about which field belongs to which
#: choice.
POSITION_FIELDS = {
    "hold_current": (),
    "absolute": ("position_m",),
    "delta": ("translation_m", "position_frame"),
}
ORIENTATION_FIELDS = {
    "hold_current": (),
    "absolute_quaternion": ("orientation_xyzw",),
    "axis_angle_delta": ("rotation_axis", "rotation_angle_deg", "orientation_frame"),
}
#: ``pose_orientation_from_world_axes_v1`` (``agent/toolkit/tools/move_pose/_axes.py``) adds
#: one orientation mode. The reader accepts it on every surface; only a
#: declaring surface advertises it and compiles it, and a schema advertises
#: exactly one of the two tables.
WORLD_AXES_MODE = "world_axes"
DECLARED_ORIENTATION_FIELDS = {
    **ORIENTATION_FIELDS, WORLD_AXES_MODE: ("approach_world", "jaw_line_world"),
}

__all__ = ["DECLARED_ORIENTATION_FIELDS", "ORIENTATION_FIELDS", "POSITION_FIELDS", "WORLD_AXES_MODE"]
