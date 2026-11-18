"""What a surface's ``view_config`` may name, as literals and three predicates.

``tool_surface`` validates a manifest's ``view_config`` against these values
while it composes a surface; the observation projections (``context.observation``)
and the observe tool deliver by the same values. They live below both so a
surface is checked and served from one set, and composition reads them without
loading an observation projection.
"""

from __future__ import annotations

from typing import Any, Mapping

#: The derived contact sheet of auxiliary cameras: its view name, its default
#: source cameras and the layouts the projection can materialize, keyed by
#: camera count (the admitted ``auxiliary_contact_sheet_cameras`` counts).
AUXILIARY_CONTACT_SHEET_VIEW = "auxiliary_contact_sheet"
AUXILIARY_CONTACT_SHEET_CAMERAS = (
    "head_camera", "front_camera", "left_camera", "right_camera",
)
AUXILIARY_CONTACT_SHEET_LAYOUTS = {
    4: ("two_by_two_native_rgb_with_label_header", 2, 2),
    3: ("one_by_three_native_rgb_with_label_header", 3, 1),
}

#: ``current_observation_delivery`` under which current public robot facts are delivered.
PUBLIC_FACTS_DELIVERY = "all_public_v1"


def full_public_delivery(view_config: Mapping[str, Any] | None) -> bool:
    return isinstance(view_config, Mapping) and view_config.get("current_observation_delivery") == PUBLIC_FACTS_DELIVERY


GRIPPER_COMMAND_DELIVERY = "commanded_beside_measured_v1"
CONTROLLED_POINT_PROJECTION = "per_camera_pixel_v1"


def controlled_point_projection_delivery(view_config: Mapping[str, Any] | None) -> bool:
    """Whether the surface publishes each arm's TCP pixel in every camera."""
    return (full_public_delivery(view_config)
            and view_config.get("controlled_point_projection") == CONTROLLED_POINT_PROJECTION)


def gripper_command_delivery(view_config: Mapping[str, Any] | None) -> bool:
    """Whether the surface publishes the commanded jaw target beside the encoder."""
    return (full_public_delivery(view_config)
            and view_config.get("gripper_command_delivery") == GRIPPER_COMMAND_DELIVERY)


__all__ = [
    "AUXILIARY_CONTACT_SHEET_CAMERAS", "AUXILIARY_CONTACT_SHEET_LAYOUTS", "AUXILIARY_CONTACT_SHEET_VIEW",
    "CONTROLLED_POINT_PROJECTION", "GRIPPER_COMMAND_DELIVERY", "PUBLIC_FACTS_DELIVERY",
    "controlled_point_projection_delivery", "full_public_delivery", "gripper_command_delivery",
]
