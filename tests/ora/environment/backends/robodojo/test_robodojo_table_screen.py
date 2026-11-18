"""The table check of ``screen_path``: SDK spheres against the planner's own table top.

A planar two-link arm in the x-z plane (both joints turn about y, so a positive
angle points the links down) stands 0.2 m above the table top; folded down it
reaches 0.35 m below it.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ora.environment.backends.robodojo import ik, kinematics_query
from ora.environment.controllers.joint_path_clearance import InterArmSphereScreen, TableSphereScreen
from ora.environment.controllers.joint_path_clearance import SphereSelfCollisionModel

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")
#: The real binding (the fake-plant fixture replaces ``kinematics_query.table_screen`` with a clear stand-in).
TABLE_SCREEN = kinematics_query.table_screen

URDF = """<robot name="planar">
  <link name="base"/><link name="link1"/><link name="link2"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>
</robot>"""
KINEMATICS = {
    "base_link": "base",
    "collision_spheres": {"base": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}],
                          "link1": [{"center": [0.15, 0.0, 0.0], "radius": 0.03}],
                          "link2": [{"center": [0.25, 0.0, 0.0], "radius": 0.04}]},
    "self_collision_ignore": {"base": ["link1"], "link1": ["link2"]},
}


def model():
    return SphereSelfCollisionModel.from_solver_kinematics(KINEMATICS, urdf_text=URDF, active_joint_names=["j1", "j2"])


def screen(buffer=0.0):
    base = InterArmSphereScreen.base_transform([0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    return TableSphereScreen({"left": model()}, {"left": base},
                             {"left": {"top_z_m": 0.0, "xy_extent_m": [[-1.5, 1.5], [-1.5, 1.5]]}},
                             sphere_buffers_m={"left": buffer})


def test_a_path_that_folds_an_arm_into_the_table_is_blocked():
    level = screen().clearance("left", [0.0, 0.0])
    assert level[0] == pytest.approx(0.15) and level[1] == "base"
    evidence = screen().straight_path("left", [0.0, 0.0], [math.pi / 2, 0.0])
    assert evidence["status"] == "table_collision" and evidence["limiting_link"] == "link2"
    assert evidence["min_clearance_m"] == pytest.approx(0.2 - 0.55 - 0.04)


def test_an_arm_already_below_the_top_may_rise_and_the_buffer_counts():
    assert screen().straight_path("left", [math.pi / 2, 0.0], [0.0, 0.0])["status"] == "clear"
    assert screen(buffer=0.01).clearance("left", [0.0, 0.0])[0] == pytest.approx(0.14)


def test_spheres_beyond_the_table_footprint_do_not_count():
    base = InterArmSphereScreen.base_transform([0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    narrow = TableSphereScreen({"left": model()}, {"left": base},
                               {"left": {"top_z_m": 0.0, "xy_extent_m": [[-0.1, 0.1], [-1, 1]]}})
    # Pointing straight down, every sphere hangs over the 0.2 m wide footprint; pointing straight up
    # (j1 = -90 degrees) the links are above it and only the base sphere is lowest.
    assert narrow.straight_path("left", [0.0, 0.0], [math.pi / 2, 0.0])["status"] == "table_collision"
    assert narrow.clearance("left", [0.0, -math.pi / 2])[0] == pytest.approx(0.15)


def test_the_binding_reads_the_planner_table_cuboid(monkeypatch):
    monkeypatch.setattr(ik, "self_collision_model", lambda session, arm, planner: model())
    planner = SimpleNamespace(frame_bias=[0.0, 0.0, 0.0],
                              scene_model={"cuboid": {"table": {"dims": [3.0, 3.0, 0.05],
                                                                "pose": [0.0, 0.0, -0.225, 1, 0, 0, 0]}}},
                              robot_cfg={"robot_cfg": {"kinematics": {"collision_sphere_buffer": 0.0}}})
    robot = SimpleNamespace(robot_name="arm", entity_origin_pose=[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    session = SimpleNamespace(robots={"left": robot},
                              environment=SimpleNamespace(robot_manager=SimpleNamespace(ik_solver={"arm": planner})))
    bound = TABLE_SCREEN(session)
    # As the SDK builds it: the top face at 0.2 m below the base in the planner frame is world z 0.
    assert bound.clearance("left", [0.0, 0.0])[0] == pytest.approx(0.15)
    assert TABLE_SCREEN(session) is bound
    del planner.scene_model
    session._table_screen = None
    with pytest.raises(Exception, match="table model is unavailable"):
        TABLE_SCREEN(session)


def test_screen_path_refuses_at_the_first_segment_into_the_table(sparse, monkeypatch):
    blocked = SimpleNamespace(straight_path=lambda arm, start, end: (
        {"status": "table_collision", "min_clearance_m": -0.01, "limiting_link": "link6"} if end[2] < 0.75
        else {"status": "clear"}))
    monkeypatch.setattr(kinematics_query, "table_screen", lambda session: blocked)
    home = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
    rows = [home, [0.0, 0.0, 0.78, 0, 0, 0], [0.0, 0.0, 0.74, 0, 0, 0]]
    result = kinematics_query.screen_path(sparse, {"knots_rad": {"left": rows}})
    assert result["status"] == "table_collision" and result["first_blocked_segment"] == 2
    assert result["arm"] == "left" and result["limiting_link"] == "link6"


def test_a_base_sphere_sitting_in_the_table_top_does_not_hide_a_link_going_down():
    base = InterArmSphereScreen.base_transform([0.0, 0.0, 0.04], [1.0, 0.0, 0.0, 0.0])
    low = TableSphereScreen({"left": model()}, {"left": base},
                            {"left": {"top_z_m": 0.0, "xy_extent_m": [[-1.5, 1.5], [-1.5, 1.5]]}})
    assert low.clearance("left", [0.0, 0.0])[0] == pytest.approx(-0.01)  # the base sphere, from the start
    # Folding 1 degree down takes link2 9.6 mm into the table: less deep than the base sphere sits, but
    # deeper than link2 itself started, so it is blocked.
    evidence = low.straight_path("left", [0.0, 0.0], [math.radians(1), 0.0])
    assert evidence["status"] == "table_collision" and evidence["limiting_link"] == "link2"
    assert low.straight_path("left", [0.0, 0.0], [-0.2, 0.0])["status"] == "clear"
