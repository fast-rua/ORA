"""yam_rgbd_v1: the YAM station's four cameras reach observe through the dual-arm path."""

from __future__ import annotations


from ora.agent.composition.surface.tool_surface import resolve_tool_surface

CAMERAS = {"overhead", "front", "wrist_left", "wrist_right"}


def test_the_yam_surface_advertises_its_cameras_and_no_service_or_probe_tools():
    surface = resolve_tool_surface("yam_direct_v1")
    by_name = {item["name"]: item for item in surface.definitions}
    assert set(by_name) == {"observe", "mark_point", "geometry_query", "compute", "set_gripper", "move_to",
                            "move_pose", "recover_pose", "wait_ticks", "report_issue", "finish_episode"}
    assert set(by_name["observe"]["inputSchema"]["properties"]["views"]["items"]["enum"]) == CAMERAS
    assert set(by_name["mark_point"]["inputSchema"]["properties"]["view"]["enum"]) == CAMERAS
    assert set(by_name["geometry_query"]["inputSchema"]["properties"]["camera"]["enum"]) == CAMERAS
    assert surface.view_config["automatic_actual_views"] == ["overhead", "front", "wrist_left", "wrist_right"]
    # No other embodiment's cameras or planner leak into the model-visible text.
    text = " ".join(item["description"] for item in surface.definitions)
    for foreign in ("head_camera", "left_camera", "CuRobo", "RoboTwin", "rotate_wrist", "grasp_execute"):
        assert foreign not in text


def test_the_perception_surface_adds_sam3_and_contact_graspnet_on_the_station_cameras():
    surface = resolve_tool_surface("yam_perception_v1")
    by_name = {item["name"]: item for item in surface.definitions}
    assert set(by_name) == {"observe", "mark_point", "geometry_query", "compute", "vision_segment", "vision_ground",
                            "grasp_plan", "set_gripper", "move_to", "move_pose", "recover_pose", "wait_ticks",
                            "report_issue", "finish_episode"}
    for tool in ("vision_segment", "vision_ground"):
        assert set(by_name[tool]["inputSchema"]["properties"]["camera"]["enum"]) == CAMERAS
    assert by_name["grasp_plan"]["inputSchema"]["properties"]["arm"]["enum"] == ["left", "right"]
    assert surface.view_config["controlled_frame_convention"] == "yam_linear_4310_tcp"
    assert surface.supports("contact_graspnet_yam_linear_4310_tcp_compiler_v1")
    # No planning-time IK annotation (the YAM executor admits every motion) and no committed grasp skill.
    assert not surface.supports("grasp_plan_ik_reachability_annotation_v1")
    result = surface.result_schemas["grasp_plan"]
    assert "ik_reachability" not in result["properties"]
    compiler = result["properties"]["compiler"]["properties"]
    assert compiler["controlled_point"]["const"] == "yam_linear_4310_tcp"
    assert compiler["embodiment_descriptor_id"]["const"] == "yam_station_public_geometry_v1"
    assert result["$defs"]["candidate"]["properties"]["gripper_opening_m"]["maximum"] == 0.0939
    text = " ".join(item["description"] for item in surface.definitions)
    for foreign in ("head_camera", "left_camera", "front_camera", "CuRobo", "RoboTwin", "ALOHA", "rotate_wrist",
                    "grasp_execute", "IK selection"):
        assert foreign not in text, foreign


def test_a_grasp_plan_compiles_to_the_planning_arms_yam_tcp_with_the_station_cameras():
    from ora.agent.toolkit.host.backend_selection import grasp_compiler_id, grasp_scene_cameras
    from ora.agent.toolkit.host.grasp.compiler_registry import GRASP_COMPILER_IDENTITIES

    surface = resolve_tool_surface("yam_perception_v1")
    for arm in ("left", "right"):
        assert grasp_scene_cameras(surface, arm) == ("overhead", f"wrist_{arm}")
        identity = GRASP_COMPILER_IDENTITIES[grasp_compiler_id(surface, arm)]
        assert identity["arm"] == arm and identity["controlled_point"] == "yam_linear_4310_tcp"
        assert identity["source_origin_to_controlled_point_m"] == identity["contact_graspnet_gripper_depth_m"]
        assert identity["maximum_opening_m"] == 0.0939
