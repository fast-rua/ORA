"""``ora_lib``: frames, projections and geometry for code-mode scripts (plan ``policy.code_mode.library: extended``).

Inside the sandbox (NumPy and SciPy, nothing of ORA) this package is bound read-only at ``/ora/lib/ora_lib``
and the current published observation at ``/ora/observation``. Lengths are metres and angles radians; a name
says when it is otherwise (``angle_deg``).

- ``frames``: ``Point``, ``Pixel``, ``Pose``, ``Direction`` and ``x.to(frame)`` across ``world``,
  ``tcp:<arm>``, ``camera:<name>`` and ``pixel:<name>``; ``offset``, ``direction``, ``pixels_to_world``,
  ``project``;
- ``observation``: ``current()``, the published observation (cameras with calibration, depth and RGB, the
  measured end effectors, the public robot facts);
- ``card``: ``frames_card()``, the text of how this scene's frames relate (each camera's view of the world's axes,
  the arms' local axes), written from the current observation;
- ``geometry``: the plain-array arithmetic underneath (rigid transforms, projection, depth windows,
  ``mask_to_cloud``, ``depth_inliers``, ``principal_geometry``);
- ``vision``: ``object_geometry(target)``, the extent, principal axes, support height, height above support and
  mid height of a target ``vision_ground`` or ``vision_segment`` returned, and the plain-array measurements it is
  made of (``mask_summary``, ``localize_mask``).
"""

from . import geometry, observation, vision
from .card import frames_card
from .frames import (Direction, FrameError, Pixel, Point, Pose, direction, offset,
                     pixels_to_world, project)
from .observation import ObservationUnavailable, current
from .vision import ObjectGeometry, object_geometry

__all__ = ["Direction", "FrameError", "ObjectGeometry", "ObservationUnavailable", "Pixel", "Point", "Pose",
           "current", "direction", "frames_card", "geometry", "object_geometry", "observation",
           "offset", "pixels_to_world", "project", "vision"]
