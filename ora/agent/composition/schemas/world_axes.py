"""The move_pose orientation mode and pose-target echo of ``pose_orientation_from_world_axes_v1``.

A declaring surface's ``move_pose`` file lists ``world_axes`` in every arm
target's ``orientation_mode`` enum with its two companion fields
``approach_world`` and ``jaw_line_world`` (described from the motion reader's own
mode table, ``surface/motion_modes.py``), names the mode where the description
lists the modes, and its motion files add ``target_axes_world`` to the
``compiled_pose.targets.<arm>`` result of ``move_pose``, ``move_to``,
``rotate_wrist`` and ``rotate_pitch``. The runtime half is
``agent/toolkit/tools/move_pose/_axes.py``; the policy-facing sentence is in the surface's own
tool files.

The axes are the surface's declared controlled-frame convention
(``view_config.controlled_frame_convention``, resolved by
``embodiment.geometry.public_local_axes``): link6 local +X and +Y on the ARX X5.
"""
from __future__ import annotations

import copy

from ..surface.schema_validation import ToolSurfaceError

CAPABILITY = "pose_orientation_from_world_axes_v1"
MODE = "world_axes"
COMPANIONS = ("approach_world", "jaw_line_world")
TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
_VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
TARGET_AXES_PROPERTY = {
    "type": "object", "additionalProperties": False, "required": ["approach", "jaw_line"],
    "description": "World directions of the approach axis and jaw line of the submitted target orientation, "
                   "computed from its compiled quaternion.",
    "properties": {"approach": copy.deepcopy(_VECTOR), "jaw_line": copy.deepcopy(_VECTOR)},
}


def _error(message: str) -> Exception:
    return ToolSurfaceError(f"{CAPABILITY} {message}")


__all__ = [
    "CAPABILITY", "COMPANIONS", "MODE", "TARGET_AXES_PROPERTY",
    "TOOLS",
]
