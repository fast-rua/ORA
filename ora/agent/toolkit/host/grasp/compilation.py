"""Pure quaternion conversion and policy-safe compiler identity projection."""
from __future__ import annotations

from typing import Any, Mapping
from ora.agent.toolkit.lib._grasp_frames import _rotation_matrix_to_quaternion_xyzw


def _public_grasp_compiler_identity(
    compiler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project compiler provenance without private reconstruction constants."""

    public = {
        "id": str(compiler_identity["id"]),
        "embodiment_descriptor_id": str(
            compiler_identity["embodiment_descriptor_id"]
        ),
        "controlled_point": str(compiler_identity["controlled_point"]),
        "controlled_point_semantics": str(
            compiler_identity["controlled_point_semantics"]
        ),
    }
    arm = compiler_identity.get("arm")
    if arm is not None:
        if arm not in {"left", "right"}:
            raise ValueError("grasp compiler arm is invalid")
        public["arm"] = str(arm)
    if "physical_jaw_aperture_calibrated" in compiler_identity:
        public["physical_jaw_aperture_calibrated"] = bool(
            compiler_identity["physical_jaw_aperture_calibrated"]
        )
        if not public["physical_jaw_aperture_calibrated"]:
            public["aperture_caveat"] = str(compiler_identity["aperture_caveat"])
        # Where the admission ceiling came from, and where the published
        # per-candidate opening came from, are two different provenances and
        # only one of them is an embodiment fact. Publishing both stops the
        # caveat from being read as a claim about the predicted width.
        if "maximum_opening_source" in compiler_identity:
            public["maximum_opening_source"] = str(
                compiler_identity["maximum_opening_source"]
            )
        if "predicted_opening_source" in compiler_identity:
            public["predicted_opening_source"] = str(
                compiler_identity["predicted_opening_source"]
            )
    return public
