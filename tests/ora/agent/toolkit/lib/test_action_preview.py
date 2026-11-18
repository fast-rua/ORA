"""The one preview (``lib/action_preview.py``): a planned path over time, public calibration and the latest image.

It names no task, tool or backend: a camera is whatever the manifest lists, and the numbers come from the segments.
"""
import io
import json

import numpy as np
import pytest
from PIL import Image

from ora.agent.toolkit.lib import _clearance, _preview_render, action_preview
from ora.contracts.action_plan import body_points, decimated, segment

START = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
GOAL = [0.2, 0.0, 0.5, 0.0, 0.0, 0.7071068, 0.7071068]
IDENTITY = np.eye(4)


def _camera(name="anything", *, width=320, height=240, depth=None, z=1.0):
    """A camera looking along +z from z=-``z`` above the origin: world x right, y down."""
    extrinsic = np.eye(4)
    extrinsic[2, 3] = z
    intrinsic = np.asarray([[200.0, 0, width / 2], [0, 200.0, height / 2], [0, 0, 1]])
    return action_preview.Camera(name, np.full((height, width, 3), 90, dtype=np.uint8), depth, intrinsic, extrinsic)


def _path(arm="left", **fields):
    return segment("arm", arm, "planned path", planned=True, poses=[START, GOAL], native_actions=20, duration_s=0.8, **fields)


def _image(result):
    return Image.open(io.BytesIO(result["png"]))


def test_a_path_is_drawn_on_whatever_cameras_the_manifest_lists_whatever_they_are_called():
    first = action_preview.preview([_path()], manifest={}, cameras=[_camera("overhead")], hands={}, jaws=None, held=None)
    second = action_preview.preview([_path()], manifest={}, cameras=[_camera("a real robot's left eye")], hands={},
                                    jaws=None, held=None)
    assert first["image"]["cameras"] == ["overhead"] and second["image"]["cameras"] == ["a real robot's left eye"]
    assert np.asarray(_image(first)).std() > 0, "the path is on the picture"


def test_the_cameras_that_see_the_path_are_chosen_and_there_are_at_most_two():
    blind = _camera("behind", z=-5.0)
    cameras = [blind, _camera("first"), _camera("second"), _camera("third")]
    assert [camera.name for camera in _preview_render.seen_cameras(cameras, [[START[:3], GOAL[:3]]])] == ["first", "second"]
    assert [camera.name for camera in _preview_render.seen_cameras([blind], [[START[:3]]])] == ["behind"], "better than none"


def test_the_picture_is_bounded_whatever_the_cameras_size():
    result = action_preview.preview([_path()], manifest={}, cameras=[_camera(width=3840, height=2160)], hands={},
                                    jaws=None, held=None)
    assert max(_image(result).size) <= _preview_render.PANEL_WIDTH * _preview_render.MAX_PANELS
    assert result["image"]["size"] == list(_image(result).size)


def test_a_legend_wraps_to_the_picture_and_loses_nothing():
    words = "one two three four five six seven eight nine ten"
    assert " ".join(_preview_render._wrapped([words], 12)) == words
    assert all(len(line) <= 12 for line in _preview_render._wrapped([words], 12))


def test_the_numbers_of_an_arm_are_its_cost_its_travel_and_all_its_turning():
    numbers = action_preview.arm_numbers([_path()], "left")
    assert numbers["native_actions"] == 20 and "share_of_native_actions_left" not in numbers
    assert numbers["hand_travel_m"] == pytest.approx(0.2) and numbers["hand_turn_deg"] == pytest.approx(90.0, abs=0.01)
    assert numbers["servo_not_counted"] is False


def test_the_numbers_say_how_high_the_path_rises_above_its_higher_end_and_where():
    """A planner's path between two poses at one height may rise on the way (a pushed object then lifts with it)."""
    level = [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    arched = segment("arm", "left", "planned path", planned=True, native_actions=20, duration_s=0.8,
                     poses=[START, [0.05, 0.0, 0.51, 0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.519, 0.0, 0.0, 0.0, 1.0],
                            [0.15, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0], level])
    numbers = action_preview.arm_numbers([arched], "left")
    assert numbers["rise_above_ends_m"] == pytest.approx(0.019) and numbers["rise_at_fraction"] == 0.5
    summary = action_preview.preview([arched], manifest={}, cameras=[], hands={}, jaws=None, held=None)["summary"]
    assert "rises 19 mm above its higher end at 50% of the path" in summary
    # Measured from the higher end: a path that only climbs to its end never rises above it.
    flat = action_preview.preview([_path()], manifest={}, cameras=[], hands={}, jaws=None, held=None)
    assert flat["arms"]["left"]["rise_above_ends_m"] == 0 and "never rises above its higher end" in flat["summary"]


def test_the_rise_is_the_jaw_centre_s_where_the_hand_is_known_and_stops_at_the_first_contact():
    flipped = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]  # half a turn about x: a jaw 0.1 m out along z points down
    turning = segment("arm", "left", "planned path", planned=True, poses=[flipped, [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                                                                         [0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    hand = {"contact_center_local": [0.0, 0.0, 0.1], "opening_axis_local": None, "maximum_aperture_m": None}
    assert action_preview.arm_numbers([turning], "left")["rise_above_ends_m"] == 0, "the controlled point stays level"
    jaws = action_preview.arm_numbers([turning], "left", hand)
    assert jaws["rise_above_ends_m"] == pytest.approx(0.2) and jaws["rise_at_fraction"] == 0.5
    after = segment("arm", "left", "after contact: not predicted", planned=False, not_predicted=True,
                    poses=[GOAL, [0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0]])
    assert action_preview.arm_numbers([_path(), after], "left")["rise_above_ends_m"] == 0


def test_a_servo_has_no_cost_ahead_of_time_and_says_so():
    servo = segment("arm", "left", "servo", planned=False, poses=[START, GOAL])
    assert action_preview.arm_numbers([servo], "left")["native_actions"] is None
    assert action_preview.arm_numbers([_path(), servo], "left")["servo_not_counted"] is True


def test_nothing_after_the_first_contact_counts_or_is_called_planned():
    after = segment("arm", "left", "after contact: not predicted", planned=False, poses=[GOAL, START], not_predicted=True,
                    native_actions=50)
    numbers = action_preview.arm_numbers([_path(), after], "left")
    assert numbers["native_actions"] == 20 and numbers["hand_travel_m"] == pytest.approx(0.2)
    assert action_preview.path_of([_path(), after], "left") == [START, GOAL]


def test_a_refused_request_draws_what_was_asked_under_its_refusal():
    refused = segment("arm", "left", "refused: goal_unreachable", planned=False, poses=[START, GOAL])
    plain = action_preview.preview([refused], manifest={}, cameras=[_camera()], hands={}, jaws=None, held=None)
    headed = action_preview.preview([refused], manifest={}, cameras=[_camera()], hands={}, jaws=None, held=None,
                                    headline="refused: goal_unreachable; nothing moves")
    assert plain["png"] != headed["png"] and "servo" in plain["summary"]


def test_a_preview_without_any_camera_says_so_and_keeps_its_numbers():
    result = action_preview.preview([_path()], manifest={}, cameras=[], hands={}, jaws=None, held=None)
    assert result["image"]["status"] == "unavailable" and "png" not in result
    assert result["arms"]["left"]["native_actions"] == 20
    # The line reports the plan only: its cost, never against what is left.
    assert result["summary"].startswith("left: 20 native actions, hand travels") and "%" not in result["summary"].split(",")[0]


def test_the_jaw_centre_is_what_is_drawn_and_a_jaw_motion_shows_its_fingers():
    hands = {"left": {"contact_center_local": [0.0, 0.0, 0.1], "opening_axis_local": [0.0, 1.0, 0.0],
                      "maximum_aperture_m": 0.08}}
    jaw = segment("gripper", "left", "the jaws close", planned=True, poses=[START, START], jaw_width_from_m=0.08,
                  jaw_width_to_m=0.0)
    (path, fingers) = action_preview.drawn([_path(), jaw], hands)
    assert path["points"][0] == pytest.approx([0.0, 0.0, 0.6])
    assert fingers["fingers_from"][0] == pytest.approx([0.0, 0.04, 0.6]) and fingers["fingers_to"][0] == pytest.approx([0.0, 0.0, 0.6])


def _flat_view(z):
    """A depth camera 1 m above a plane at height ``z``, looking straight down."""
    height, width = 200, 200
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera_to_world[2, 3] = 1.0
    return _clearance.DepthView("above", np.full((height, width), 1.0 - z), np.asarray(
        [[100.0, 0, 100], [0, 100.0, 100], [0, 0, 1]]), camera_to_world)


FINGER = _clearance.FingerBounds(minimum=[-0.01, -0.01, -0.02], maximum=[0.01, 0.01, 0.0], opening_direction=[0, 1, 0])


def _line(*heights, x=0.0):
    return [[x + 0.1 * index, 0.0, height, 0, 0, 0, 1] for index, height in enumerate(heights)]


def test_clearance_is_the_least_distance_along_the_path_and_says_where():
    poses = [[0.0, 0.0, 0.3, 0, 0, 0, 1], [0.0, 0.0, 0.1, 0, 0, 0, 1], [0.0, 0.0, 0.3, 0, 0, 0, 1]]
    value = _clearance.along_path([_flat_view(0.0)], poses=poses, finger_qpos=[0.0], fingers=[FINGER])
    assert value["status"] == "measured" and value["clearance_m"] == pytest.approx(0.08)
    assert value["at_fraction"] == 0.5 and value["position_m"][2] == pytest.approx(0.0) and value["part"] == "fingers"


def test_a_path_whose_jaws_reach_into_what_the_depth_shows_says_how_far_and_where():
    """0 mm alone cannot tell a graze from a finger box deep in a surface: the deepest point inside is reported."""
    value = _clearance.along_path([_flat_view(0.0)], poses=_line(0.3, 0.008, 0.3), finger_qpos=[0.0], fingers=[FINGER])
    assert value["status"] == "measured" and value["clearance_m"] == 0.0
    assert value["inside_m"] == pytest.approx(0.008, abs=1e-6) and value["at_fraction"] == 0.5
    assert value["position_m"][2] == pytest.approx(0.0)
    line = action_preview._line("left", action_preview.arm_numbers([_path()], "left"), value)
    assert "the jaws reach 8 mm into what the depth shows at 50% of the path" in line and "least clearance" not in line
    clear = _clearance.along_path([_flat_view(0.0)], poses=_line(0.3, 0.03, 0.3), finger_qpos=[0.0], fingers=[FINGER])
    assert clear["clearance_m"] == pytest.approx(0.01) and "inside_m" not in clear


def test_a_path_s_ends_are_its_last_5_cm_along_it_in_one_place():
    assert _clearance.PATH_END_M == 0.05
    assert _clearance.near_path_ends(_line(0.3, 0.3, 0.3)) == [True, False, True]
    close = [[0.01 * index, 0, 0, 0, 0, 0, 1] for index in range(21)]
    assert sum(_clearance.near_path_ends(close)) == 10 and _clearance.near_path_ends(close)[5:16].count(False) == 11


def test_what_a_planner_clears_at_the_end_of_a_path_is_marked_not_measured():
    """The last pose sits in the surface: the planner removed it on purpose, and the clearance is the path's before it."""
    poses = [*_line(0.3, 0.2, 0.15), [0.3, 0.0, 0.01, 0, 0, 0, 1]]
    value = _clearance.along_path([_flat_view(0.0)], poses=poses, finger_qpos=[0.0], fingers=[FINGER])
    assert value["status"] == "measured" and value["clearance_m"] > 0.1 or value["status"] == "beyond"
    entered = value["enters_cleared_space"]
    assert entered["at_fraction"] == 1.0 and entered["part"] == "fingers" and entered["position_m"][2] == pytest.approx(0.0)
    assert "removed on purpose" in value["claim_boundary"]


def test_a_path_no_longer_than_its_ends_has_no_clearance_to_give():
    poses = [[0.0, 0.0, 0.1, 0, 0, 0, 1], [0.02, 0.0, 0.1, 0, 0, 0, 1]]
    value = _clearance.along_path([_flat_view(0.0)], poses=poses, finger_qpos=[0.0], fingers=[FINGER])
    assert value["status"] == "ends_only" and value["path_end_m"] == 0.05 and "clearance_m" not in value


def test_a_surface_beyond_the_reach_is_reported_as_beyond_it():
    value = _clearance.along_path([_flat_view(0.0)], poses=_line(0.5, 0.5, 0.5), finger_qpos=[0.0], fingers=[FINGER])
    assert value["status"] == "beyond" and value["clearance_at_least_m"] == _clearance.PATH_REACH_M


def test_no_depth_is_unknown_never_clear():
    empty = _clearance.DepthView("above", np.zeros((10, 10)), np.eye(3), np.eye(4))
    assert _clearance.along_path([empty], poses=_line(0.5, 0.5), finger_qpos=[0.0],
                                 fingers=[FINGER])["status"] == "unknown"


def test_the_moving_arm_itself_is_left_out_of_what_is_measured_against():
    poses = _line(0.1, 0.1, 0.1)
    measured = _clearance.along_path([_flat_view(0.0)], poses=poses, finger_qpos=[0.0], fingers=[FINGER])
    assert measured["clearance_m"] == pytest.approx(0.08)
    chain = [[0.1, 0.0, 0.0], [0.1, 0.0, 0.2]]  # a link standing on the plane under the hand
    without = _clearance.along_path([_flat_view(0.0)], poses=poses, finger_qpos=[0.0], fingers=[FINGER], body=chain)
    assert without.get("position_m", [9.0])[0] != pytest.approx(0.1, abs=0.05)


def test_a_chain_of_joints_comes_from_forward_kinematics_alone():
    """A planar two-link arm: its shoulder at the origin, its elbow 0.3 m along the first link."""
    def fk(rows):
        result = []
        for first, second, third in rows:
            elbow = np.asarray([0.3 * np.cos(first), 0.3 * np.sin(first), 0.0])
            hand = elbow + 0.2 * np.asarray([np.cos(first + second), np.sin(first + second), 0.0])
            result.append([*hand, 0, 0, 0, 1])
        return result

    chain = body_points(fk, [0.4, -0.3, 0.0])
    assert chain[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9), "the shoulder"
    assert chain[1] == pytest.approx([0.3 * np.cos(0.4), 0.3 * np.sin(0.4), 0.0], abs=1e-9), "the elbow"
    assert len(chain) == 3, "a joint that does not move the point adds nothing"


def test_a_long_path_keeps_its_ends_and_its_size():
    rows = [[float(index)] for index in range(1000)]
    kept = decimated(rows, 10)
    assert len(kept) == 10 and kept[0] == [0.0] and kept[-1] == [999.0]


def test_the_record_is_plain_json_but_for_the_picture():
    result = action_preview.preview([_path()], manifest={}, cameras=[_camera()], hands={}, jaws=None, held=None)
    json.dumps({key: value for key, value in result.items() if key != "png"}, allow_nan=False)
    assert "poses" not in result["segments"][0]


def test_a_preview_s_turn_does_not_read_a_rounded_quaternion_as_a_turn() -> None:
    """Recorded poses are rounded, so a held orientation's quaternion has a norm just under 1 (86 deg was reported for a
    straight descent)."""
    from ora.agent.toolkit.lib.action_preview import _turn_deg

    held = [0.7069, 0.0, 0.7069, 0.0]
    assert _turn_deg(held, held) == pytest.approx(0.0, abs=1e-3)


HAND = {"contact_center_local": [0.0, 0.0, 0.1], "opening_axis_local": [0.0, 1.0, 0.0],
        "approach_axis_local": [0.0, 0.0, 1.0], "maximum_aperture_m": 0.08}


def test_the_jaws_are_sketched_where_the_hand_is_and_where_it_ends_at_the_measured_opening():
    (path,) = action_preview.drawn([_path()], {"left": HAND}, {"left": 0.04})
    base, tip, other_tip, other_base = path["jaws_start"]
    assert tip == pytest.approx([0.0, 0.02, 0.6]) and other_tip == pytest.approx([0.0, -0.02, 0.6])
    assert base == pytest.approx([0.0, 0.02, 0.6 - action_preview.FINGER_DRAWN_M]), "fingers point along the approach"
    assert other_base == pytest.approx([0.0, -0.02, 0.6 - action_preview.FINGER_DRAWN_M])
    assert path["jaws_end"] != path["jaws_start"], "the hand ends turned"


def test_a_hand_without_an_opening_axis_has_no_jaws_and_the_widest_aperture_stands_in_for_an_unknown_opening():
    (bare,) = action_preview.drawn([_path()], {})
    assert bare["jaws_start"] is None and bare["jaws_end"] is None
    (wide,) = action_preview.drawn([_path()], {"left": HAND})
    assert wide["jaws_start"][1] == pytest.approx([0.0, 0.04, 0.6])


def test_without_a_declared_approach_the_jaw_centre_offset_points_the_fingers():
    hand = {key: value for key, value in HAND.items() if key != "approach_axis_local"}
    (path,) = action_preview.drawn([_path()], {"left": hand}, {"left": 0.04})
    assert path["jaws_start"][0] == pytest.approx([0.0, 0.02, 0.6 - action_preview.FINGER_DRAWN_M])


def test_the_measured_opening_is_the_published_fraction_of_the_widest_aperture():
    manifest = {"proprioception": {"end_effectors": {"left": {"gripper": 0.5}, "right": {"gripper": None}}}}
    assert action_preview.openings_m(manifest, {"left": HAND, "right": HAND}) == {"left": pytest.approx(0.04)}


def _red(result):
    pixels = np.asarray(_image(result).convert("RGB")).reshape(-1, 3).astype(int)
    return int(((pixels[:, 0] == _preview_render.REFUSED[0]) & (pixels[:, 1] == _preview_render.REFUSED[1])
                & (pixels[:, 2] == _preview_render.REFUSED[2])).sum())


def test_a_refused_request_is_marked_red_at_what_was_asked():
    asked = segment("arm", "left", "refused: goal_unreachable", planned=False, poses=[START, GOAL])
    kwargs = dict(manifest={}, cameras=[_camera()], hands={"left": HAND}, jaws=None, held=None,
                  headline="refused: goal_unreachable; nothing moves")
    marked = action_preview.preview([asked], refused="goal_unreachable", **kwargs)
    plain = action_preview.preview([asked], **kwargs)
    assert _red(marked) > 10 * max(1, _red(plain))


def test_every_drawn_line_has_a_dark_edge():
    result = action_preview.preview([_path()], manifest={}, cameras=[_camera()], hands={}, jaws=None, held=None)
    pixels = np.asarray(_image(result).convert("RGB"))[:240]
    assert (pixels.sum(axis=2) == 0).any(), "black edge under the path"


#: An ARX X5 hand as its public descriptor bounds it: fingers along +x, opening along +-y, empty close at 0.185.
X5_FINGERS = [_clearance.FingerBounds(minimum=[0.0716, 0.0004, -0.0306], maximum=[0.1576, 0.0378, 0.0304],
                                      opening_direction=[0.0, 1.0, 0.0]),
              _clearance.FingerBounds(minimum=[0.0716, -0.0378, -0.0306], maximum=[0.1576, -0.0004, 0.0304],
                                      opening_direction=[0.0, -1.0, 0.0])]
X5_JAWS = {"left": {"fingers": [{"minimum": list(finger.minimum), "maximum": list(finger.maximum),
                                 "opening_direction": list(finger.opening_direction)} for finger in X5_FINGERS],
                    "reference_finger_qpos": [0.0, 0.0], "empty_close_reference": 0.185185185,
                    "contact_center_local": [0.1501, 0.0, 0.0], "wrist_cameras": []}}
X5_HAND = {"contact_center_local": [0.1501, 0.0, 0.0], "opening_axis_local": [0.0, 1.0, 0.0],
           "approach_axis_local": [1.0, 0.0, 0.0], "maximum_aperture_m": 0.0888}


def _gripped(command="close", opening=0.3625, fingers=(0.0095, 0.0097)):
    """A RoboDojo left hand after set_gripper close stalled on something (push_T 0aac80dbb979, observation 7)."""
    return {"proprioception": {"end_effectors": {"left": {
        "pose_xyz_xyzw": [-0.42, -0.1165, 0.928, -0.6611, 0.2511, 0.661, 0.251], "gripper": opening,
        "gripper_command_target": command, "gripper_finger_qpos": list(fingers),
        "gripper_feedback": {"arrival_check": {"tolerance_normalized": 0.03}}}}}}


def test_jaws_that_closed_on_something_hold_it_whatever_closed_them():
    """set_gripper(close) with no target_ref names no target, yet the measured jaws say they hold a load between them."""
    from ora.agent.toolkit.lib import clearance
    from ora.agent.toolkit.lib.values import JsonValue

    holding = clearance.held(_gripped(), arm="left", jaws=JsonValue.of(X5_JAWS), held=JsonValue.of({})).json()
    assert holding["target_ref"] is None and holding["extent"] == "between_jaws"
    assert holding["jaw_gap_m"] == pytest.approx(0.0095 + 0.0097 + 0.0008, abs=1e-4)
    assert holding["box_size_m"] == pytest.approx([0.086, 0.02, 0.061], abs=1e-3)
    open_jaws = _gripped(command="open")
    assert clearance.held(open_jaws, arm="left", jaws=JsonValue.of(X5_JAWS), held=JsonValue.of({})).json() is None
    empty = _gripped(opening=0.19)
    assert clearance.held(empty, arm="left", jaws=JsonValue.of(X5_JAWS), held=JsonValue.of({})).json() is None


def test_a_preview_of_a_move_holding_something_reports_it_sweeps_it_and_draws_it():
    level = [-0.11, -0.196, 0.928, -0.6611, 0.2511, 0.661, 0.251]
    drag = segment("arm", "left", "planned path", planned=True, native_actions=17, duration_s=0.68,
                   poses=[_gripped()["proprioception"]["end_effectors"]["left"]["pose_xyz_xyzw"], level])
    result = action_preview.preview([drag], manifest=_gripped(), cameras=[], hands={"left": X5_HAND}, jaws=X5_JAWS,
                                    held={})
    assert result["arms"]["left"]["held_object"]["extent"] == "between_jaws"
    assert "local_corners" not in result["arms"]["left"]["held_object"], "the record keeps the box's size, not its corners"
    assert result["summary"].startswith("left: holds what its jaws closed on (86x20x61 mm between the jaws, 20 mm apart; "
                                        "what reaches past them is not measured), 17 native actions")
    free = action_preview.preview([drag], manifest=_gripped(command="open"), cameras=[], hands={"left": X5_HAND},
                                  jaws=X5_JAWS, held={})
    assert "held_object" not in free["arms"]["left"] and free["summary"].startswith("left: 17 native actions")
    box = _clearance.box_corners([0.0716, -0.01, -0.03], [0.1576, 0.01, 0.03]).tolist()
    (path,) = action_preview.drawn([drag], {"left": X5_HAND}, None, {"left": box})
    assert len(path["held_start"]) == 8 and path["held_end"][0] == pytest.approx(action_preview.jaw_point(level, box[0]))
    assert len(_preview_render.BOX_EDGES) == 12


def test_a_named_target_s_box_is_the_held_box_and_only_a_missing_one_falls_back_to_the_jaws():
    from ora.agent.toolkit.lib import _clearance_inputs as inputs

    geometry = {"left": inputs.JawGeometry(fingers=tuple(X5_FINGERS), reference_finger_qpos=(0.0, 0.0),
                                           empty_close_reference=0.185185185, contact_center_local=(0.1501, 0.0, 0.0),
                                           wrist_cameras=())}
    named = inputs.HeldExtent("cup_1", _clearance.box_corners([0.1, -0.05, -0.05], [0.2, 0.05, 0.05]), False)
    assert inputs.measured_held(_gripped(), geometry, {"left": named})["left"] is named
    unsized = inputs.measured_held(_gripped(), geometry, {"left": inputs.HeldExtent("cup_1")})["left"]
    assert unsized.target_ref == "cup_1" and unsized.source == inputs.BETWEEN_JAWS and unsized.local_corners is not None
    assert inputs.measured_held(_gripped(command="open"), geometry, {}) == {}


def test_the_held_box_is_on_the_picture():
    camera = _camera(z=2.0)
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    move = segment("arm", "left", "planned path", planned=True, native_actions=10, duration_s=0.4,
                   poses=[start, [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]])
    holding = _gripped()
    holding["proprioception"]["end_effectors"]["left"]["pose_xyz_xyzw"] = start
    kwargs = dict(cameras=[camera], hands={"left": X5_HAND}, jaws=X5_JAWS, held={})
    gripped = action_preview.preview([move], manifest=holding, **kwargs)
    released = action_preview.preview([move], manifest=_gripped(command="open"), **kwargs)
    assert gripped["image"]["status"] == "attached" and gripped["png"] != released["png"]
