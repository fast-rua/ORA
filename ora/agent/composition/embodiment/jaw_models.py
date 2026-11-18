"""Jaw models and jaw-centre rules, keyed by controlled-frame convention.

Two composition binders admit a surface by these tables
(``observed_clearance_reference_v2`` needs a controlled frame whose jaw model
is known, ``reach_nearest_orientation_reference_v1`` one whose jaw centre is
known), and the tool runtime measures with the same tables
(``tools/motion/clearance_geometry`` and ``clearance_runtime``,
``tools/reach/jaw_center``). They are literals, so the binders read them
without loading the measurement code, and a convention added here is admitted
and measured in one place.
"""
from __future__ import annotations

from typing import Any, Mapping

#: Finger opening read from the measured state (``clearance_runtime._qpos``).
OPENING_ENCODER = "finger_qpos_minus_reference"
OPENING_SIGNED = "signed_finger_qpos"
OPENING_SYMMETRIC = "half_measured_jaw_separation"
#: Symmetric jaws whose descriptor states the fingers at a non-zero reference opening.
OPENING_SYMMETRIC_FROM_REFERENCE = "half_measured_jaw_separation_minus_reference"
#: Holding test (``clearance_runtime.jaws_hold``).
HOLD_ENCODER = "normalized_opening_above_empty_close_reference"
HOLD_SEPARATION = "measured_jaw_separation_above_near_closed"

#: Controlled frame -> how its jaw opens and when it holds.
JAW_MODELS: dict[str, Mapping[str, Any]] = {
    "arx_x5_link6": {"loader": "v1"},
    "panda_grip_site": {"opening": OPENING_SIGNED, "finger_signs": (1.0, -1.0), "hold": HOLD_SEPARATION},
    "aloha_grip_center_tcp": {"opening": OPENING_SYMMETRIC, "finger_signs": (), "hold": HOLD_SEPARATION},
    # Robotiq 2F-85: symmetric 4-bar jaws; the observation publishes the measured pad separation.
    "robotiq_2f85_tcp": {"opening": OPENING_SYMMETRIC, "finger_signs": (), "hold": HOLD_SEPARATION},
    # I2RT YAM linear_4310: symmetric slides; the descriptor states the fingers fully open.
    "yam_linear_4310_tcp": {"opening": OPENING_SYMMETRIC_FROM_REFERENCE, "finger_signs": (),
                            "hold": HOLD_SEPARATION},
}

#: Grip-centre conventions and the controlled-point description their
#: descriptor must state for the zero offset to apply.
GRIP_CENTRE_CONVENTIONS = {
    "aloha_grip_center_tcp": "tool_center",
    "panda_grip_site": "grip_center",
    "robotiq_2f85_tcp": "grip_center",
    "yam_linear_4310_tcp": "grip_center",
}
#: Frames whose jaw centre comes from the descriptor's reviewed contact centre.
CONTACT_CENTRE_CONVENTIONS = ("arx_x5_link6",)
SUPPORTED_CONVENTIONS = (*GRIP_CENTRE_CONVENTIONS, *CONTACT_CENTRE_CONVENTIONS)


__all__ = [
    "CONTACT_CENTRE_CONVENTIONS", "GRIP_CENTRE_CONVENTIONS", "HOLD_ENCODER", "HOLD_SEPARATION",
    "JAW_MODELS", "OPENING_ENCODER", "OPENING_SIGNED", "OPENING_SYMMETRIC", "OPENING_SYMMETRIC_FROM_REFERENCE",
    "SUPPORTED_CONVENTIONS",
]
