"""Pure embodiment frame compilation; no surface selection or I/O."""
from __future__ import annotations
import math
from typing import Any, Mapping

from ._pose_math import matrix_quaternion

PANDA_GRIP_SITE_COMPILER_ID = "contact_graspnet_to_robosuite_panda_grip_site_v2"
#: The controlled point is the centre of the fingertip pads, jaw x and approach z: the Panda's grip site, and the Robotiq
#: TCP, which is the same point and compiles through the same mapping.
GRIP_SITE_STYLE_COMPILER_IDS = frozenset({
    PANDA_GRIP_SITE_COMPILER_ID, "contact_graspnet_to_robolab_robotiq_2f85_tcp_v1",
})


def _rotation_matrix_to_quaternion_xyzw(matrix: Any) -> list[float]:
    import numpy as np

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("grasp orientation matrix is invalid")
    trace = float(np.trace(value))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
        w = 0.25 * scale
    else:
        index = int(np.argmax(np.diag(value)))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            x = 0.25 * scale
            y = (value[0, 1] + value[1, 0]) / scale
            z = (value[0, 2] + value[2, 0]) / scale
            w = (value[2, 1] - value[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            x = (value[0, 1] + value[1, 0]) / scale
            y = 0.25 * scale
            z = (value[1, 2] + value[2, 1]) / scale
            w = (value[0, 2] - value[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            x = (value[0, 2] + value[2, 0]) / scale
            y = (value[1, 2] + value[2, 1]) / scale
            z = 0.25 * scale
            w = (value[1, 0] - value[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("grasp orientation quaternion is degenerate")
    quaternion /= norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return [float(item) for item in quaternion]
def _compiled_grasp_geometry(
    candidate: Mapping[str, Any], *,
    compiler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one Contact-GraspNet source frame into one controlled point.

    Every registered embodiment compiles through this identity-parametrized
    mapping; the identity's constants are what differ.
    """

    import numpy as np

    convention = candidate.get("pose_convention")
    if convention == compiler_identity["source_pose_convention"]:
        source_origin = np.asarray(
            candidate["source_gripper_origin_xyz_m"], dtype=np.float64,
        )
        source_pregrasp = np.asarray(
            candidate["source_pregrasp_origin_xyz_m"], dtype=np.float64,
        )
        direct_grip_site = direct_pregrasp_grip_site = None
    elif convention == "pca_topdown_parallel_jaw_center_v1":
        direct_grip_site = np.asarray(
            candidate["grasp_center_xyz_m"], dtype=np.float64,
        )
        direct_pregrasp_grip_site = np.asarray(
            candidate["pregrasp_center_xyz_m"], dtype=np.float64,
        )
        source_origin = source_pregrasp = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError("grasp candidate pose convention is not compiler-qualified")
    approach = np.asarray(candidate["approach_world"], dtype=np.float64)
    jaw = np.asarray(candidate["jaw_world"], dtype=np.float64)
    if any(
        value.shape != (3,)
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate vectors must each contain three values")
    if not all(
        np.isfinite(value).all()
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate contains non-finite geometry")
    approach_norm = float(np.linalg.norm(approach))
    if not math.isfinite(approach_norm) or approach_norm <= 1e-6:
        raise ValueError("grasp approach axis is degenerate")
    approach_axis = approach / approach_norm
    jaw_axis = jaw - approach_axis * float(np.dot(jaw, approach_axis))
    jaw_norm = float(np.linalg.norm(jaw_axis))
    if not math.isfinite(jaw_norm) or jaw_norm <= 1e-6:
        raise ValueError("grasp jaw and approach axes are degenerate")
    jaw_axis /= jaw_norm
    lateral_axis = np.cross(approach_axis, jaw_axis)
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if not math.isfinite(lateral_norm) or lateral_norm <= 1e-6:
        raise ValueError("grasp axes cannot define a lateral direction")
    lateral_axis /= lateral_norm
    aloha = compiler_identity.get("controlled_point") == "end_effector"
    if aloha:
        x_axis, y_axis, z_axis = approach_axis, jaw_axis, lateral_axis
    else:
        x_axis, y_axis, z_axis = jaw_axis, lateral_axis, approach_axis
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(rotation)) < 0.999:
        raise ValueError("grasp axes do not form a right-handed orientation")
    source_to_grip_site = float(
        compiler_identity["source_origin_to_controlled_point_m"]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        source_origin = direct_grip_site - approach_axis * source_to_grip_site
        source_pregrasp = (
            direct_pregrasp_grip_site - approach_axis * source_to_grip_site
        )
    approach_distance = float(np.dot(
        source_origin - source_pregrasp, approach_axis,
    ))
    lateral_error = float(np.linalg.norm(
        (source_origin - source_pregrasp) - approach_axis * approach_distance,
    ))
    if (
        not math.isfinite(approach_distance)
        or not 0.005 - 1e-9 <= approach_distance <= 0.10 + 1e-9
        or lateral_error > 1e-4
    ):
        raise ValueError("grasp pregrasp is not one bounded axial servo away")
    gripper_depth = float(
        compiler_identity["contact_graspnet_gripper_depth_m"]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        grip_site = direct_grip_site
        pregrasp_grip_site = direct_pregrasp_grip_site
    else:
        grip_site = source_origin + approach_axis * source_to_grip_site
        pregrasp_grip_site = grip_site - approach_axis * approach_distance
    contact_baseline = source_origin + approach_axis * gripper_depth
    opening = float(candidate["gripper_opening_m"])
    contact_negative_jaw = contact_baseline - jaw_axis * opening / 2.0
    contact_positive_jaw = contact_baseline + jaw_axis * opening / 2.0
    return {
        "source_gripper_origin": source_origin,
        "source_pregrasp_origin": source_pregrasp,
        "grip_site": grip_site,
        "pregrasp_grip_site": pregrasp_grip_site,
        "contact_baseline": contact_baseline,
        "contact_negative_jaw": contact_negative_jaw,
        "contact_positive_jaw": contact_positive_jaw,
        "approach_distance": approach_distance,
        "rotation": rotation,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "jaw_axis": jaw_axis,
    }

def _grip_site_handoff(
    candidate: Mapping[str, Any], geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """The motion handoff of a grip-site compiler, whose candidates always carry their contact pose.

    Its own limits: 50 mm of approach and no opening check, which the proposal's
    admission owns. The Robotiq TCP reports the Panda mapping it compiles through.
    """
    if float(geometry["approach_distance"]) > 0.05:
        raise ValueError("grasp approach exceeds bounded servo distance")
    orientation_xyzw = matrix_quaternion(geometry["rotation"].tolist())
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": int(candidate["rank"]),
        "pregrasp_pose": {
            "position_m": [float(item) for item in geometry["pregrasp_grip_site"]],
            "orientation_xyzw": orientation_xyzw,
        },
        "contact_pose": {
            "position_m": [float(item) for item in geometry["grip_site"]],
            "orientation_xyzw": orientation_xyzw,
        },
        "approach_delta": {
            "frame": "grip_site",
            "translation_m": [0.0, 0.0, float(geometry["approach_distance"])],
            "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        },
        "gripper_opening_m": float(candidate["gripper_opening_m"]),
        "backend_score": float(candidate["backend_score"]),
        "compiler_checks": {
            "finite_geometry": "passed",
            "right_handed_grip_site_axes": "passed",
            "bounded_local_approach": "passed",
            "source_origin_to_grip_site": "passed",
            "contact_baseline_reconstruction": "passed",
            "embodiment_mapping": PANDA_GRIP_SITE_COMPILER_ID,
            "workspace": candidate["checks"]["workspace"],
            "aperture": candidate["checks"]["aperture"],
            "reachability": "delegated_to_move_free_space",
        },
    }


def _compile_grasp_candidate(
    candidate: Mapping[str, Any], *,
    compiler_identity: Mapping[str, Any],
    public_contact_pose: bool = False,
) -> dict[str, Any]:
    geometry = _compiled_grasp_geometry(
        candidate, compiler_identity=compiler_identity,
    )
    if compiler_identity["id"] in GRIP_SITE_STYLE_COMPILER_IDS:
        return _grip_site_handoff(candidate, geometry)
    maximum_approach = float(
        compiler_identity.get("maximum_approach_m", 0.05)
    )
    if (
        not math.isfinite(maximum_approach)
        or not 0.005 <= maximum_approach <= 0.10
    ):
        raise ValueError("grasp compiler approach limit is invalid")
    if float(geometry["approach_distance"]) > maximum_approach + 1e-9:
        raise ValueError("grasp approach exceeds bounded servo distance")
    opening = float(candidate["gripper_opening_m"])
    maximum_opening = float(compiler_identity.get("maximum_opening_m", 0.08))
    if opening > maximum_opening:
        raise ValueError("grasp opening exceeds embodiment limit")
    aloha = compiler_identity.get("controlled_point") == "end_effector"
    orientation_xyzw = _rotation_matrix_to_quaternion_xyzw(
        geometry["rotation"],
    )
    contact_pose = (
        # Same convention as the Panda compiler: the controlled point's
        # world pose at contact, closing the residual check against the
        # measured end-effector after the approach.
        {
            "contact_pose": {
                "position_m": [
                    float(item) for item in geometry["grip_site"]
                ],
                "orientation_xyzw": orientation_xyzw,
            },
        }
        if public_contact_pose else {}
    )
    # Only identities that declare the flag add the field, so frozen ALOHA
    # candidate records keep their exact shape.
    aperture_calibration = (
        {}
        if "physical_jaw_aperture_calibrated" not in compiler_identity
        else {
            "aperture_calibration": (
                "calibrated"
                if compiler_identity["physical_jaw_aperture_calibrated"]
                else str(compiler_identity["aperture_caveat"])
            ),
        }
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": int(candidate["rank"]),
        "pregrasp_pose": {
            "position_m": [
                float(item) for item in geometry["pregrasp_grip_site"]
            ],
            "orientation_xyzw": orientation_xyzw,
        },
        **contact_pose,
        "approach_delta": {
            "frame": "end_effector" if aloha else "grip_site",
            "translation_m": [
                float(geometry["approach_distance"]) if aloha else 0.0,
                0.0,
                0.0 if aloha else float(geometry["approach_distance"]),
            ],
            "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        },
        "gripper_opening_m": opening,
        "backend_score": float(candidate["backend_score"]),
        "compiler_checks": {
            "finite_geometry": "passed",
            "right_handed_controlled_point_axes": "passed",
            "bounded_local_approach": "passed",
            "source_origin_to_controlled_point": "passed",
            "embodiment_mapping": compiler_identity["id"],
            "workspace": candidate["checks"]["workspace"],
            "aperture": candidate["checks"]["aperture"],
            **aperture_calibration,
            "reachability": "delegated_to_move_free_space",
        },
    }
