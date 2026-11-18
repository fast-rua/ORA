"""Which grasp compilation an embodiment's surface selects.

Every registered compiler identity (Panda grip site, Robotiq TCP, the ALOHA
end-effector and grip-centre families, ARX X5 link6 and top-down, the YAM TCP), keyed by id,
and the pre-grasp hover distance a surface's grasp staging uses.
"""

from __future__ import annotations

from typing import Any

from ora.agent.error_classes import ToolError
from .arx_x5_gripper import ARX_X5_LINK6_COMPILER_IDENTITIES
from .x5_topdown import X5_TOPDOWN_COMPILER_IDENTITIES
from ora.agent.toolkit.host.grasp.compiler import PANDA_GRIP_SITE_COMPILER_IDENTITY, ROBOTIQ_TCP_COMPILER_IDENTITY
# The jaw geometry these identities are built from lives beside its source
# binding, because the published result schema is pinned to the same number.
from ora.agent.toolkit.host.grasp.aloha_gripper import ALOHA_NOMINAL_MAXIMUM_OPENING_M, ALOHA_SOURCE_BINDING

ROBOTWIN_ALOHA_HYBRID_GRASP_CAPABILITY = (
    "robotwin_aloha_public_rgbd_topdown_grasp_v1"
)


ALOHA_END_EFFECTOR_COMPILER_IDENTITIES = {
    arm: {
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_end_effector_v1",
        "embodiment_descriptor_id": "aloha_agilex",
        "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "controlled_point": "end_effector",
        "controlled_point_semantics": "RoboTwin_planner_pose",
        "contact_graspnet_gripper_depth_m": 0.1034,
        "source_origin_to_controlled_point_m": 0.0,
        "controlled_point_to_contact_baseline_m": 0.1034,
        "maximum_opening_m": ALOHA_NOMINAL_MAXIMUM_OPENING_M,
        "axis_mapping": {
            "model_approach_z": "end_effector_local_positive_x",
            "model_jaw_x": "end_effector_local_positive_y",
        },
        "arm": arm,
        "source_binding": ALOHA_SOURCE_BINDING,
    }
    for arm in ("left", "right")
}
ALOHA_GRIP_CENTER_COMPILER_IDENTITIES = {
    arm: {
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v2",
        "embodiment_descriptor_id": "aloha_agilex",
        "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "controlled_point": "end_effector",
        "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
        "contact_graspnet_gripper_depth_m": 0.1034,
        "source_origin_to_controlled_point_m": 0.12,
        "controlled_point_to_contact_baseline_m": -0.0166,
        "maximum_opening_m": ALOHA_NOMINAL_MAXIMUM_OPENING_M,
        "axis_mapping": {
            "model_approach_z": "grip_center_local_positive_x",
            "model_jaw_x": "grip_center_local_positive_y",
        },
        "arm": arm,
        "source_binding": ALOHA_SOURCE_BINDING,
    }
    for arm in ("left", "right")
}
ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES = {
    arm: {
        **identity,
        "id": f"contact_graspnet_to_robotwin_aloha_{arm}_grip_center_v3",
        # Contact-GraspNet reports the origin of its Panda gripper model.
        # Its own source-bound finger/contact depth is 103.4 mm.  The 120 mm
        # ALOHA link6 -> public-TCP transform is applied later by the RoboTwin
        # motion adapter and must not be reused here.
        "source_origin_to_controlled_point_m": 0.1034,
        "controlled_point_to_contact_baseline_m": 0.0,
        "maximum_approach_m": 0.10,
        # A deterministic public-RGB-D candidate family complements learned
        # oblique proposals.  The TCP target is a small qualified clearance
        # above the robust visible top surface; pre-grasp remains 100 mm back
        # along world-down approach.  These values are ALOHA compiler facts,
        # not task/object GT.
        "public_rgbd_topdown_profile": {
            "surface_percentile": 95.0,
            "contact_clearance_m": 0.008,
            "pregrasp_standoff_m": 0.10,
            "maximum_candidates": 2,
        },
    }
    for arm, identity in ALOHA_GRIP_CENTER_COMPILER_IDENTITIES.items()
}
# I2RT YAM linear_4310 (controllers/yam_station.py): the TCP is the centre of the two
# fingertip facets, local x the closing line and local z the approach, so the generic
# jaw-x / approach-z mapping applies. Contact-GraspNet's own origin-to-fingertip depth
# (103.4 mm) puts that fingertip point on the TCP, as on ALOHA v3; the linear_4310 opens
# to twice its 46.95 mm finger slide.
YAM_TCP_COMPILER_IDENTITIES = {
    arm: {
        "id": f"contact_graspnet_to_yam_{arm}_linear_4310_tcp_v1",
        "embodiment_descriptor_id": "yam_station_public_geometry_v1",
        "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "controlled_point": "yam_linear_4310_tcp",
        "controlled_point_semantics": "grip_center",
        "contact_graspnet_gripper_depth_m": 0.1034,
        "source_origin_to_controlled_point_m": 0.1034,
        "controlled_point_to_contact_baseline_m": 0.0,
        "maximum_opening_m": 0.0939,
        "maximum_approach_m": 0.10,
        "axis_mapping": {
            "model_jaw_x": "tcp_local_positive_x",
            "model_approach_z": "tcp_local_positive_z",
        },
        "arm": arm,
    }
    for arm in ("left", "right")
}
GRASP_COMPILER_IDENTITIES = {
    PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]: PANDA_GRIP_SITE_COMPILER_IDENTITY,
    ROBOTIQ_TCP_COMPILER_IDENTITY["id"]: ROBOTIQ_TCP_COMPILER_IDENTITY,
    **{
        identity["id"]: identity
        for identity in ALOHA_END_EFFECTOR_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ALOHA_GRIP_CENTER_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ALOHA_GRIP_CENTER_STANDOFF_COMPILER_IDENTITIES.values()
    },
    **{
        identity["id"]: identity
        for identity in ARX_X5_LINK6_COMPILER_IDENTITIES.values()
    },
    **{identity["id"]: identity for identity in X5_TOPDOWN_COMPILER_IDENTITIES.values()},
    **{identity["id"]: identity for identity in YAM_TCP_COMPILER_IDENTITIES.values()},
}


def _grasp_hover_offset_m(surface: Any, *, grasp_plan_enabled: bool) -> float:
    """Select the source-bound pre-grasp distance without changing old surfaces."""

    if surface.supports(ROBOTWIN_ALOHA_HYBRID_GRASP_CAPABILITY):
        if not grasp_plan_enabled:
            raise ToolError(
                "ALOHA public RGB-D grasp staging requires grasp_plan on the ToolSurface",
                code="tool_surface_capability_mismatch",
                retryable=False,
            )
        return 0.10
    return 0.04 if grasp_plan_enabled else 0.08
