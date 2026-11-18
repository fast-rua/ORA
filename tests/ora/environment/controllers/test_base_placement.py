"""The base-placement search and the base's footprint check on the shared motion base (``base_placement.py``).

A toy holonomic base: one footprint box half a metre square, a column that rises with the lift, an arm mount on the
base origin 0.7 m up, and a planner whose IK reaches a target within a set horizontal distance of the mount. The
floor it stands on is seen (``SEEN``) unless a test says otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ora.environment.controllers import base_placement as B

SERVO = B.Servo(linear_gain=3.0, max_linear_m_s=0.6, yaw_gain=3.0, max_yaw_rad_s=1.0, tick_s=0.05,
                position_tolerance_m=0.03, yaw_tolerance_rad=0.04, max_ticks=400)
BASE = B.MobileBase(
    footprint=(B.Box((-0.25, 0.25), (-0.25, 0.25), (0.0, 0.4)), B.Box((-0.1, 0.1), (-0.1, 0.1), (0.4, 0.7), True)),
    servo=SERVO, torso_range_m=(0.0, 0.3), torso_step_m=0.1, reach_m=1.05, shoulder_m=0.33, floor_m=0.03)
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _here(x=0.0, y=0.0, yaw=0.0, torso=0.0):
    return B.Here(base=(x, y, yaw), torso_m=torso, mount=(x, y, 0.7 + torso, *IDENTITY))


def _wall(x, y_range=(-2.0, 2.0), height=0.6, step=0.02):
    """Depth points of a wall across the floor at ``x``."""
    ys = np.arange(y_range[0], y_range[1], step)
    zs = np.arange(0.05, height, step)
    return np.array([[x, y, z] for y in ys for z in zs])


def _reach_within(radius):
    """An IK that reaches the target from a stand whose mount is within ``radius`` of it, horizontally."""
    asked = []

    def solve(stands):
        asked.append(list(stands))
        return [math.hypot(stand.x_m - TARGET[0], stand.y_m - TARGET[1]) <= radius for stand in stands]
    return solve, asked


TARGET = (1.5, 0.0, 0.9)
#: The floor seen everywhere the toy base may stand.
SEEN = np.array([[x, y] for x in np.arange(-2.0, 3.0, 0.05) for y in np.arange(-2.0, 2.0, 0.05)])


def _points(obstacles, floor=SEEN, body=None):
    return B.PublicPoints(np.asarray(obstacles, dtype=float).reshape(-1, 3), floor=floor, body=body)


def _admit_all(kept):
    return lambda stand: kept.append(stand) or True


def test_stands_leave_out_where_the_base_is_and_are_cheapest_first():
    here = _here()
    found = B.stands(TARGET, here, BASE)
    assert found and all(math.hypot(s.x_m, s.y_m) <= B.SEARCH.max_travel_m for s in found)
    assert all(math.hypot(s.x_m, s.y_m) >= B.SEARCH.here_m or abs(s.yaw_rad) >= B.SEARCH.here_rad for s in found)
    assert [s.cost for s in found] == sorted(s.cost for s in found)
    assert all(math.dist((s.x_m, s.y_m, 0.7 + s.torso_m), TARGET) <= BASE.reach_m for s in found)
    heights = {round(s.torso_m, 6) for s in found}
    assert heights <= {0.0, 0.1, 0.2, 0.3} and 0.0 in heights


def test_a_base_without_a_lift_keeps_its_height():
    base = B.MobileBase(**{**BASE.__dict__, "torso_range_m": None})
    assert {s.torso_m for s in B.stands(TARGET, _here(torso=0.0), base)} == {0.0}
    assert {s.torso_m for s in B.stands(TARGET, _here(torso=0.12), base)} == {0.12}, "the measured height, whatever"


def test_a_nudge_in_place_ranks_after_a_real_move():
    """A turn of a few degrees where the base stands (RoboCasa smoke 11-06: 25 of 107 first suggestions) comes after
    every stand that really moves within ``nudge_cost_m``."""
    found = B.stands((1.5, 0.1, 0.9), _here(x=0.9), BASE)
    nudges = [s for s in found if math.hypot(s.x_m - 0.9, s.y_m) < B.SEARCH.nudge_m
              and abs(s.yaw_rad) < B.SEARCH.nudge_rad]
    moves = [s for s in found if math.hypot(s.x_m - 0.9, s.y_m) >= B.SEARCH.nudge_m]
    assert nudges and moves and min(s.cost for s in nudges) >= B.SEARCH.nudge_cost_m
    assert found.index(nudges[0]) > found.index(moves[0])


def test_the_robot_s_own_points_and_the_floor_are_not_obstacles():
    camera_points = np.array([[0.5, 0.0, 1.0], [0.0, 0.0, 0.2], [2.0, 0.0, 0.01], [2.0, 0.0, 0.5], [0.58, 0.0, 1.0],
                              [0.7, 0.3, 1.0]])
    hand = np.eye(4)
    hand[:3, 3] = (0.7, 0.3, 1.0)
    body = B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.05]]), 0.03, ((hand, ((-0.01, 0.01),) * 3),))
    assert body.holds(camera_points).tolist() == [True, False, False, False, True, True]
    assert not B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.05]]), 0.02).holds(camera_points[4:5]).any(), (
        "a sphere holds what lies within its radius and the margin, and no farther")
    inside = B.footprint_depth(camera_points, BASE, (0.0, 0.0, 0.0), 0.0, 0.0) >= 0
    assert inside.tolist() == [False, True, False, False, False, False]


def _cameras(points_world):
    """One camera per point, at the origin looking along +x, each pixel row ``[0, index]`` showing its point."""
    # OpenCV axes: x right, y down, z forward; world +x forward, +y left, +z up.
    cam2world = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    width, cameras = len(points_world), {}
    for column, point in enumerate(points_world):
        local = np.linalg.inv(cam2world) @ np.array([*point, 1.0])
        intrinsic = np.array([[1.0, 0.0, column - local[0] / local[2]], [0.0, 1.0, -local[1] / local[2]],
                              [0.0, 0.0, 1.0]])
        cameras[f"c{column}"] = {"depth_m": np.where(np.arange(width) == column, local[2], 0.0).reshape(1, width),
                                 "intrinsic_cv": intrinsic, "cam2world_cv": cam2world, "depth_range_m": (0.01, 10.0)}
    return cameras


def test_public_points_split_the_depth_and_the_robot_s_pixels_go_dark_for_a_world_seen_from_elsewhere():
    world = [[1.0, 0.1, 0.5], [0.5, 0.0, 1.0], [1.2, -0.2, 0.01]]
    cameras = _cameras(world)
    for index, camera in enumerate(cameras.values()):
        points, pixels = B.back_project(camera, 2.5)
        assert points[0] == pytest.approx(world[index]) and pixels.tolist() == [index]
    body = B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.05]]), 0.0)
    points = B.public_points(cameras, base=BASE, here=_here(), own=body, max_depth_m=2.5)
    assert len(points.obstacles) == 1 and points.obstacles[0] == pytest.approx([1.0, 0.1, 0.5])
    assert points.own_pixels["c1"].tolist() == [1] and points.own_pixels["c0"].tolist() == []
    def seen(x, y):
        return bool((np.hypot(points.floor[:, 0] - x, points.floor[:, 1] - y) < 1.5 * B.SEARCH.floor_cell_m).any())

    assert seen(1.2, -0.2) and seen(0.6, -0.1), "the floor a ray ended on, and the floor it passed over low enough"
    assert not seen(1.0, 0.1) and points.body is body, "a ray that ran above the base's height saw no floor there"
    dark = B.without_own(cameras, points)
    assert dark["c1"]["depth_m"][0, 1] == 0.0 and dark["c0"]["depth_m"][0, 0] > 0.0
    assert cameras["c1"]["depth_m"][0, 1] > 0.0, "the cameras themselves are untouched"


def test_the_servo_path_drives_the_turn_centre_straight_and_ends_on_the_goal():
    here = B.Here(base=(0.0, 0.0, 0.0), torso_m=0.0, mount=(0, 0, 0.7, *IDENTITY), turn_centre_m=(-0.2, 0.0))
    poses = B.servo_path(here, BASE, (1.0, 0.5, math.pi / 2))
    assert poses[-1] == (1.0, 0.5, math.pi / 2)
    centres = [(x - 0.2 * math.cos(yaw), y - 0.2 * math.sin(yaw)) for x, y, yaw in poses]
    start, end = np.array((-0.2, 0.0)), np.array(centres[-1])
    for centre in centres:
        offset = np.asarray(centre) - start
        assert abs(np.cross(end - start, offset)) / np.linalg.norm(end - start) < 1e-6, "the centre stays on its line"


def test_a_move_whose_footprint_meets_the_depth_is_refused_where_it_first_does():
    points = B.PublicPoints(_wall(1.0))
    blocked = B.sweep(points, BASE, _here(), (1.2, 0.0, 0.0))
    assert blocked is not None and 0.0 < blocked["path_fraction"] < 1.0 and blocked["robot_part"] == "base"
    assert blocked["obstacle_xy_m"][0] == pytest.approx(1.0) and blocked["obstacle_height_m"] > BASE.floor_m
    assert B.sweep(points, BASE, _here(), (0.5, 0.0, 0.0)) is None
    assert B.sweep(B.PublicPoints(np.zeros((0, 3))), BASE, _here(), (5.0, 0.0, 0.0)) is None


def test_the_arm_the_base_carries_is_checked_along_its_path():
    """A hand reaching into a shelf (RoboCasa smoke 11-06: three base moves stalled with the hand in a fridge): the
    footprint passes under the shelf, the carried hand does not, and the refusal names the arm or its load."""
    shelf = np.array([[x, y, 1.0] for x in np.arange(0.6, 1.2, 0.02) for y in np.arange(-0.4, 0.4, 0.02)])
    arm = np.array([[0.4, 0.0, 1.0, 0.05], [0.7, 0.0, 0.9, 0.04]])
    held = np.array([[0.5, 0.0, 1.0, 0.03]])
    assert B.sweep(_points(shelf), BASE, _here(), (0.4, 0.0, 0.0)) is None, "the footprint alone passes"
    blocked = B.sweep(_points(shelf, body=B.OwnBody(arm, 0.03)), BASE, _here(), (0.4, 0.0, 0.0))
    assert blocked["robot_part"] == "arm" and 0.0 < blocked["path_fraction"] <= 1.0
    assert blocked["obstacle_height_m"] == pytest.approx(1.0)
    clear_of_it = np.array([[0.3, 0.0, 0.9, 0.04]])
    load = B.sweep(_points(shelf, body=B.OwnBody(np.vstack([clear_of_it, held]), 0.03, held=1)), BASE, _here(),
                   (0.4, 0.0, 0.0))
    assert load["robot_part"] == "held_object"
    assert B.sweep(_points(shelf, body=B.OwnBody(arm, 0.03)), BASE, _here(), (-0.4, 0.0, 0.0)) is None, (
        "backing out of it is clear")


def _plate(y_start, y_end):
    """A load's depth held by its edge: a plate 5 mm under the jaws (0.6, 0, 1.0), from ``y_start`` to ``y_end``, one
    point at the centre of each 1 cm voxel."""
    return np.array([[x, y, 0.995] for x in np.arange(0.555, 0.65, 0.01) for y in np.arange(y_start, y_end, 0.01)])


#: A wall beside the load, 10 cm past a 0.22 m plate's edge and above the base's column.
LOAD_WALL = np.array([[x, 0.325, z] for x in np.arange(0.505, 0.70, 0.01) for z in np.arange(0.905, 1.10, 0.01)])
#: The arm (one sphere behind the jaws) and the planner's load: only the slice between the jaws.
HOLDING = B.OwnBody(np.array([[0.4, 0.0, 1.0, 0.04], [0.6, 0.0, 1.0, 0.03]]), 0.03, held=1)


def test_the_depth_linked_to_a_held_load_moves_with_it_and_a_real_obstacle_still_meets_it():
    """StoreLeftoversInBowl (smoke rc-ab-6864e57de): the planner's world kept only the jaws' slice of a bowl held by its
    rim as the load, the rest of the bowl stayed in the depth, and 66 base moves were refused as the load meeting it.
    The depth linked to the load is the load: it is the robot's own and is carried, and a wall 10 cm past it is not."""
    plate = _plate(0.005, 0.22)
    depth = np.vstack([plate, LOAD_WALL])
    as_scene = B.sweep(_points(depth[~HOLDING.holds(depth)], body=HOLDING), BASE, _here(), (0.0, 0.05, 0.0))
    assert as_scene["robot_part"] == "held_object" and as_scene["obstacle_xy_m"][1] < 0.22, (
        "with only the planner's slice carried, the load meets its own depth")
    seen = B.public_points(_cameras(depth), base=BASE, here=_here(), own=HOLDING, max_depth_m=2.5)
    assert seen.body.held > HOLDING.held and seen.body.held == len(seen.body.spheres) - 1
    assert seen.body.holds(plate).all() and not seen.body.holds(LOAD_WALL).any()
    assert len(seen.obstacles) == len(LOAD_WALL), "the wall stays, the plate is the robot's own"
    assert sum(len(pixels) for pixels in seen.own_pixels.values()) == len(plate), "a stand's world loses the load too"
    assert B.sweep(seen, BASE, _here(), (0.0, 0.05, 0.0)) is None, "a short move toward the wall is clear"
    assert B.sweep(seen, BASE, _here(), (0.0, -0.3, 0.0)) is None, "and so is one away from it"
    into = B.sweep(seen, BASE, _here(), (0.0, 0.2, 0.0))
    assert into["robot_part"] == "held_object" and into["obstacle_xy_m"][1] == pytest.approx(0.325), (
        "a move that carries the load into the wall is refused at the wall")


def test_depth_linked_to_a_held_load_that_runs_past_its_reach_is_split_and_what_runs_on_stays():
    """A plate running on 0.6 m from the jaws (a shelf, a door the jaws hold on to): the load keeps the part nearer its
    seeds than the depth past the reach, the rest stays in the depth beside it, and a move that would drag it is
    refused. Depth more than three voxels from the load's depth is not linked to it; nothing beside the load, or
    nothing held, links nothing."""
    shelf = _plate(0.005, 0.6)
    load = B.linked_load(shelf, HOLDING)
    assert len(load) and (load[:, 1] + load[:, 3]).max() < B.SEARCH.load_reach_m
    seen = B.public_points(_cameras(shelf), base=BASE, here=_here(), own=HOLDING, max_depth_m=2.5)
    assert seen.body.held > HOLDING.held and (seen.obstacles[:, 1] > B.SEARCH.load_reach_m).any()
    assert B.sweep(seen, BASE, _here(), (0.0, 0.05, 0.0))["robot_part"] == "held_object", "dragging it is refused"
    assert B.sweep(seen, BASE, _here(), (0.0, -0.1, 0.0)) is None
    near, far = _plate(0.005, 0.1), _plate(0.135, 0.22)
    linked = B.linked_load(np.vstack([near, far]), HOLDING)
    assert len(linked) and B.within_spheres(near, linked, 0.0).all()
    assert not B.within_spheres(far[far[:, 1] > 0.15], linked, 0.0).any(), "a 4 cm gap breaks the chain"
    assert B.linked_load(LOAD_WALL, HOLDING).shape == (0, 4), "nothing beside the load"
    assert B.linked_load(shelf, B.OwnBody(HOLDING.spheres[:1], 0.03)).shape == (0, 4), "nothing held"


def test_a_fragment_of_depth_within_three_voxels_of_the_load_joins_it():
    """StoreLeftoversInBowl step 2858 (smoke rc-ab-6d8eb1acf): 43 voxels that moved with the bowl lay 2.4 cm from the
    rest of it, and the carried bowl met them. A fragment within three voxels whose own chain stays within the reach
    is the load's."""
    near, far = _plate(0.005, 0.1), _plate(0.125, 0.22)
    linked = B.linked_load(np.vstack([near, far]), HOLDING)
    assert B.within_spheres(np.vstack([near, far]), linked, 0.0).all()
    seen = B.public_points(_cameras(np.vstack([near, far])), base=BASE, here=_here(), own=HOLDING, max_depth_m=2.5)
    assert len(seen.obstacles) == 0 and B.sweep(seen, BASE, _here(), (0.0, 0.05, 0.0)) is None


#: A wall 2 cm past the far edge of a 0.22 m plate, running on far past the load's reach.
BESIDE_WALL = np.array([[x, 0.235, z] for x in np.arange(0.305, 0.9, 0.01) for z in np.arange(0.705, 1.3, 0.01)])


def test_a_load_hanging_beside_a_wall_keeps_its_depth_and_the_wall_stays():
    """StoreLeftoversInBowl step 2700 (smoke rc-ab-6d8eb1acf): a bowl hanging within 2 cm of a wall linked to it, the
    chain ran on past the reach, and the whole of it, bowl included, was left to the scene, so three base moves were
    refused as the bowl meeting itself. Split by which it is nearer, the load keeps all of the plate and at most the
    wall where it touches; the rest of the wall stays, and a turn into it is refused at the wall."""
    depth = np.vstack([_plate(0.005, 0.22), BESIDE_WALL])
    load = B.linked_load(depth, HOLDING)
    assert B.within_spheres(_plate(0.005, 0.22), load, 0.0).all()
    touched = BESIDE_WALL[B.within_spheres(BESIDE_WALL, load, 0.0)]
    assert len(touched) < 0.01 * len(BESIDE_WALL) and (np.abs(touched[:, [0, 2]] - (0.6, 0.995)) <= 0.03).all()
    turn = (0.0, 0.0, math.radians(10.0))
    as_scene = B.sweep(_points(depth[~HOLDING.holds(depth)], body=HOLDING), BASE, _here(), turn)
    assert as_scene["robot_part"] == "held_object" and as_scene["obstacle_xy_m"][1] < 0.22, "it met itself"
    body = HOLDING.carrying(load)
    split = _points(depth[~body.holds(depth)], body=body)
    into = B.sweep(split, BASE, _here(), turn)
    assert into["robot_part"] == "held_object" and into["obstacle_xy_m"][1] == pytest.approx(0.235), "it meets the wall"
    assert B.sweep(split, BASE, _here(), (0.0, 0.0, -math.radians(10.0))) is None
    assert B.sweep(split, BASE, _here(), (0.0, -0.2, 0.0)) is None


def _edge_camera(depths):
    """One 4x4 camera 1 m up at the origin, looking along +x, each column of the image at its depth in ``depths``."""
    cam2world = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    return {"depth_m": np.tile(np.asarray(depths, dtype=np.float64), (4, 1)),
            "intrinsic_cv": np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]),
            "cam2world_cv": cam2world, "depth_range_m": (0.01, 10.0)}


def test_a_mixed_pixel_at_the_robot_s_edge_is_the_robot_s_own():
    """SteamInMicrowave [45] (smoke rc-ab-6d8eb1acf): two pixels at the bowl's edge in the over-shoulder camera read
    1.08 m, between the bowl's 0.96 m and the scene's 1.4 m, and the carried bowl met them. A pixel next to the robot's
    own, more than ``edge_step_m`` behind it and in front of another neighbour, is its edge; a surface that runs on
    behind the robot is not."""
    arm = B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.05]]), 0.03)
    seen = B.public_points({"c": _edge_camera([0.5, 0.5, 1.0, 2.0])}, base=BASE, here=_here(), own=arm,
                           max_depth_m=2.5)
    assert len(seen.obstacles) == 4 and seen.obstacles[:, 0] == pytest.approx(2.0)
    assert sorted(seen.own_pixels["c"].tolist()) == [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    behind = B.public_points({"c": _edge_camera([0.5, 0.5, 1.98, 2.0])}, base=BASE, here=_here(), own=arm,
                             max_depth_m=2.5)
    assert len(behind.obstacles) == 8, "a surface continuous with what lies behind it stays"


def test_a_suggested_stand_s_move_keeps_a_slack_to_move_base_s_own_check():
    """SteamInMicrowave [59] (smoke rc-ab-6864e57de): a move 3 mm off its suggestion was refused, its load 1.6 mm deeper
    than the tolerance. The search sweeps a stand's move ``suggestion_slack_m`` stricter than the admission."""
    slack = B.SEARCH.suggestion_slack_m
    wall = _wall(0.465, height=0.3)
    assert B.sweep(_points(wall), BASE, _here(), (0.2, 0.0, 0.0)) is None, "the footprint's 1 cm margin clears it"
    edge = B.sweep(_points(wall), BASE, _here(), (0.2, 0.0, 0.0), slack_m=slack)
    assert edge["robot_part"] == "base" and edge["penetration_m"] == 0.0
    arm = B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.05]]), 0.03)
    shelf = np.array([[0.745, y, 1.0] for y in np.arange(-0.2, 0.21, 0.01)])
    assert B.sweep(_points(shelf, body=arm), BASE, _here(), (0.2, 0.0, 0.0)) is None, "5 mm deep is within 1 cm"
    assert B.sweep(_points(shelf, body=arm), BASE, _here(), (0.2, 0.0, 0.0), slack_m=slack)["robot_part"] == "arm"
    solve, _ = _reach_within(0.65)
    found = B.search(TARGET, _here(), BASE, _points(_wall(2.0)), solve=solve, admits=lambda stand: True)
    for item in found["suggestions"]:
        for dx, dy in ((0.005, 0.0), (-0.005, 0.0), (0.0, 0.005), (0.0, -0.005)):
            assert B.sweep(_points(_wall(2.0)), BASE, _here(),
                           (item["x_m"] + dx, item["y_m"] + dy, math.radians(item["yaw_deg"]))) is None


def test_the_search_keeps_the_nearest_stands_whose_move_is_admitted():
    solve, asked = _reach_within(0.65)
    admitted = []
    found = B.search(TARGET, _here(), BASE, _points(_wall(2.0)), solve=solve, admits=_admit_all(admitted))
    assert found["schema"] == B.SCHEMA and 1 <= len(found["suggestions"]) <= B.SEARCH.suggestions
    best = found["suggestions"][0]
    assert math.hypot(best["x_m"] - TARGET[0], best["y_m"]) <= 0.65 + 1e-3
    assert best["arm_path"] == B.ADMITTED and len(admitted) == len(found["suggestions"])
    assert len(asked[0]) <= B.SEARCH.blank_queries, "one blank question for many stands"
    bands = asked[1:]
    assert len(bands) == len(admitted) and all(len(band) == 6 for band in bands), "each kept stand's arrival band"
    searched = found["searched"]
    assert searched["asked"] == len(asked[0]) and searched["checked_in_world"] == len(admitted)
    assert searched["arm_blocked"] == 0 and searched["floor_unseen"] == 0
    first, *rest = found["suggestions"]
    assert all(math.hypot(first["x_m"] - other["x_m"], first["y_m"] - other["y_m"]) >= B.SEARCH.separation_m
               or abs(first["yaw_deg"] - other["yaw_deg"]) >= math.degrees(B.SEARCH.separation_rad) for other in rest)


def test_a_stand_whose_footprint_or_path_meets_the_depth_or_whose_move_is_refused_is_not_offered():
    wall = _points(_wall(1.0, y_range=(-0.3, 0.3)))
    solve, _ = _reach_within(0.65)
    found = B.search(TARGET, _here(), BASE, wall, solve=solve, admits=lambda stand: True)
    for item in found["suggestions"]:
        assert B.sweep(wall, BASE, _here(), (item["x_m"], item["y_m"], math.radians(item["yaw_deg"]))) is None
    refused = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3))), solve=lambda stands: [True] * len(stands),
                       admits=lambda stand: False)
    assert refused["suggestions"] == [] and refused["searched"]["checked_in_world"] == B.SEARCH.observed_checks
    unknown = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3))), solve=lambda stands: [True] * len(stands),
                       admits=lambda stand: None)
    assert unknown["suggestions"] == [], "a stand whose admission the planner could not answer is not offered"


def test_a_stand_the_ik_leaves_at_the_edge_of_the_arrival_band_is_not_offered():
    """The servo stops within its tolerances of a stand, not on it (RoboCasa smoke 11-06: 1.2 to 1.5 cm off): a target
    the arm reaches only from the exact stand was out of reach on arrival."""
    def solve(stands):
        return [math.hypot(stand.x_m - TARGET[0], stand.y_m - TARGET[1]) <= 0.65 and stand.yaw_rad > -0.01
                for stand in stands]

    found = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3))), solve=solve, admits=lambda stand: True)
    for item in found["suggestions"]:
        assert math.radians(item["yaw_deg"]) - SERVO.yaw_tolerance_rad > -0.01
    assert found["searched"]["arrival_unreached"] > 0
    band = B.arrival_band(B.Stand(1.0, 2.0, 0.5, 0.1), SERVO)
    assert sorted((round(s.x_m - 1.0, 3), round(s.y_m - 2.0, 3), round(s.yaw_rad - 0.5, 3)) for s in band) == sorted(
        [(0.03, 0.0, 0.0), (-0.03, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, -0.03, 0.0), (0.0, 0.0, 0.04),
         (0.0, 0.0, -0.04)]) and {s.torso_m for s in band} == {0.1}


def test_a_stand_whose_carried_arm_meets_the_depth_on_the_way_is_counted_and_not_offered():
    shelf = np.array([[x, y, 1.0] for x in np.arange(0.6, 1.2, 0.02) for y in np.arange(-0.4, 0.4, 0.02)])
    body = B.OwnBody(np.array([[0.5, 0.0, 1.0, 0.04]]), 0.03)
    solve, _ = _reach_within(0.65)
    found = B.search(TARGET, _here(), BASE, _points(shelf, body=body), solve=solve, admits=lambda stand: True)
    assert found["searched"]["arm_blocked"] > 0
    for item in found["suggestions"]:
        assert B.sweep(_points(shelf, body=body), BASE, _here(),
                       (item["x_m"], item["y_m"], math.radians(item["yaw_deg"]))) is None


def test_a_stand_on_floor_no_camera_saw_is_not_offered():
    """DeliverStraw (smoke 11-06): every stand the footprint screen cleared lay behind a cabinet face, on floor no
    camera saw. Floor seen in front of the robot only: the stands there remain, those behind are left out."""
    far_side = SEEN[SEEN[:, 0] >= 1.2]
    solve, _ = _reach_within(1.0)
    found = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3)), floor=far_side), solve=solve,
                     admits=lambda stand: True)
    assert found["searched"]["floor_unseen"] > 0 and found["suggestions"]
    for item in found["suggestions"]:
        assert item["x_m"] >= 1.2 - 0.25 + B.SEARCH.floor_seen_fraction * 0.5 - B.SEARCH.floor_cell_m, (
            "enough of its floor was seen")
    unseen = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3)), floor=np.zeros((0, 2))), solve=solve,
                      admits=lambda stand: True)
    assert unseen["suggestions"] == [] and unseen["searched"]["footprint_clear"] == 0, (
        "no floor seen: only a stand mostly over the floor the base covers now could remain, and none is that close")


def test_nothing_reachable_says_so_with_what_was_asked():
    found = B.search(TARGET, _here(), BASE, _points(np.zeros((0, 3))), solve=lambda stands: [False] * len(stands),
                     admits=lambda stand: True)
    assert found["suggestions"] == [] and found["searched"]["ik_reached"] == 0 and found["searched"]["asked"] > 0


def test_a_stand_carries_the_mount_and_the_hand_with_the_base_and_the_lift():
    here = _here(x=1.0, y=2.0, yaw=0.0, torso=0.1)
    stand = B.Stand(1.5, 2.0, math.pi / 2, 0.25)
    mount = B.carried(here, stand, here.mount)
    assert mount[:3] == pytest.approx([1.5, 2.0, 0.95])
    hand = B.carried(here, stand, (1.6, 2.0, 1.0, *IDENTITY))
    assert hand[:3] == pytest.approx([1.5, 2.6, 1.15])
    assert B.carried(here, B.lifted(here, 0.3), (1.6, 2.0, 1.0, *IDENTITY))[:3] == pytest.approx([1.6, 2.0, 1.2])


def test_a_suggestion_is_a_ready_move_base_target():
    here = _here(x=1.0, y=1.0, yaw=0.0)
    item = B.suggestion(here, B.Stand(1.3, 1.4, math.radians(30), 0.12), B.ADMITTED)
    assert item == {"x_m": 1.3, "y_m": 1.4, "yaw_deg": 30.0, "torso_height_m": 0.12, "distance_m": 0.5,
                    "turn_deg": 30.0, "arm_path": "admitted"}
    assert "torso_height_m" not in B.suggestion(here, B.Stand(1.3, 1.4, 0.0, 0.12), B.ADMITTED, torso=False)
    lifted = B.MobileBase(**{**BASE.__dict__, "torso_range_m": None})
    found = B.search(TARGET, _here(), lifted, _points(np.zeros((0, 3))), solve=lambda stands: [True] * len(stands),
                     admits=lambda stand: True)
    assert found["suggestions"] and all("torso_height_m" not in item for item in found["suggestions"]), (
        "a stand that keeps the measured height names none")
