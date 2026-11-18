"""``ora_lib``'s perception fragments (``geometry.depth_inliers``, ``geometry.dilate_disc``, ``vision``) compute what
``ora/agent/toolkit/lib`` computes, on the recorded public RoboDojo observation (``tests/data/toolkit-general-pickup``:
a real head RGB-D frame and calibration) and on random masks.

The sandbox interpreter has no OpenCV, so the two image operations the lib takes from it are written with SciPy and
held to it here: 8-connected components (including which of two components of the same size is the primary one) and
the elliptical dilation. ``object_geometry`` is held to the ``localization`` ``vision_ground`` returns for the same
mask, through the mount the sandbox reads. The localization composes the lib's pure steps in a different order
(``mask_to_cloud``, ``transform_points`` and ``principal_geometry`` against the lib's one inline formula), so its
floats agree to ``TOLERANCE_M`` rather than bit for bit; counts, status and strings are equal.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ora.agent.code_mode.library import observation_mount
from ora.agent.code_mode.ora_lib import geometry, object_geometry, vision
from ora.agent.code_mode.ora_lib import observation as library_observation
from ora.agent.toolkit.lib import geometry as lib_geometry
from ora.agent.toolkit.lib import vision as lib_vision
from ora.agent.toolkit.lib.values import Array, JsonValue
from ora.agent.toolkit.tools.vision_ground.tool import run as ground
from tests.support.agent.toolkit_tools import api_for

FIXTURE = Path(__file__).parents[4] / "data/toolkit-general-pickup/policy-observations"
#: The most a float of the localization may differ from the lib's, metres (or unitless for axes and fractions).
TOLERANCE_M = 1e-9
#: The overhead footprint fields the lib adds with ``extended_frame`` and ``localize_mask`` does not compute.
FOOTPRINT = {"footprint_center_xyz_m", "footprint_extent_m", "view_tilt_deg", "footprint_view"}
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def head() -> dict:
    """The recorded head camera: float32 depth (480x640, metres), ``K``, ``camera_to_world`` and its calibration."""

    root = next(FIXTURE.iterdir()) / "head"
    calibration = json.loads((root / "calibration.json").read_text())
    return {"depth": np.load(root / "depth-m.npy"), "calibration": calibration,
            "K": np.asarray(calibration["intrinsic_cv"]),
            "camera_to_world": np.linalg.inv(np.asarray(calibration["extrinsic_cv"]))}


def blobs(shape: tuple[int, int], count: int, size: int) -> np.ndarray:
    """Random rectangles, which overlap and touch at corners (8-connected components of every size)."""

    mask = np.zeros(shape, dtype=bool)
    for _ in range(count):
        top, left = RNG.integers(0, shape[0] - 1), RNG.integers(0, shape[1] - 1)
        mask[top:top + RNG.integers(1, size), left:left + RNG.integers(1, size)] = True
    return mask


# --- the two image operations the lib takes from OpenCV -----------------------------------------------------------

@pytest.mark.parametrize("radius", [1, 2, 3, 5, 8, 15, 30, 60])
def test_the_elliptical_dilation_equals_opencv_s(radius: int) -> None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    masks = [blobs((90, 120), count, 30) for count in (1, 3, 8)] + [RNG.random((90, 120)) > 0.97,
                                                                     np.zeros((90, 120), bool)]
    masks[0][0, :] = masks[1][-1, -1] = True  # at the border
    for mask in masks:
        expected = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        np.testing.assert_array_equal(geometry.dilate_disc(mask, radius), expected)


def test_the_dilation_by_a_disc_larger_than_the_image_equals_opencv_s() -> None:
    for shape, radius in (((9, 12), 30), ((1, 5), 3), ((5, 1), 3), ((40, 3), 20)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        for _ in range(10):
            mask = RNG.random(shape) > 0.8
            np.testing.assert_array_equal(geometry.dilate_disc(mask, radius),
                                          cv2.dilate(mask.astype(np.uint8), kernel).astype(bool))
    with pytest.raises(ValueError):
        geometry.dilate_disc(np.zeros((4, 4), bool), 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_depth_inliers_equal_the_lib_s(dtype) -> None:
    for trial in range(40):
        shape = (int(RNG.integers(20, 90)), int(RNG.integers(20, 90)))
        depth = RNG.uniform(0.4, 1.2, size=shape).astype(dtype)
        depth[blobs(shape, 3, 10)] += dtype(RNG.uniform(0.0, 0.5))  # steps in depth: the MAD gate cuts
        depth[RNG.random(shape) > 0.97] = np.nan
        depth[RNG.random(shape) > 0.97] = 0.0
        depth[RNG.random(shape) > 0.99] = 3.0
        mask = blobs(shape, int(RNG.integers(1, 6)), 40)
        for options in ({}, {"minimum_points": 1}, {"minimum_half_width_m": 0.0, "mad_scale": 2.0}):
            expected = lib_geometry.depth_inliers(Array.of(mask), Array.of(depth), **options)
            found = geometry.depth_inliers(mask, depth, **options)
            np.testing.assert_array_equal(found.pop("mask"), expected.mask.numpy())
            assert found == expected.diagnostics.json(), (trial, options)


def test_components_of_the_same_size_are_told_apart_as_opencv_numbers_them() -> None:
    """With ``minimum_points=1`` the primary component is one pixel, and which one is the tie's winner: raster order
    would pick the other here (the lone pixel at row 0 comes first by row, the one at row 1 by 2x2 block)."""

    depth = np.full((8, 12), 1.0)
    mask = np.zeros_like(depth, dtype=bool)
    mask[0, 9] = mask[1, 1] = True
    expected = lib_geometry.depth_inliers(Array.of(mask), Array.of(depth), minimum_points=1).mask.numpy()
    found = geometry.depth_inliers(mask, depth, minimum_points=1)["mask"]
    assert expected[1, 1] and not expected[0, 9], "OpenCV 4 numbers the 2x2 block holding (1, 1) first"
    np.testing.assert_array_equal(found, expected)
    ties = 0
    for _ in range(300):  # and in general
        shape = (int(RNG.integers(4, 30)), int(RNG.integers(4, 30)))
        mask = RNG.random(shape) > RNG.uniform(0.6, 0.9)
        depth = np.full(shape, 1.0)
        sizes = np.bincount(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[1].ravel())[1:]
        ties += int((sizes == sizes.max()).sum() > 1) if len(sizes) else 0
        np.testing.assert_array_equal(
            geometry.depth_inliers(mask, depth, minimum_points=1)["mask"],
            lib_geometry.depth_inliers(Array.of(mask), Array.of(depth), minimum_points=1).mask.numpy())
    assert ties > 30, "the random masks do exercise ties"


# --- the mask measurements, on the recorded head depth ------------------------------------------------------------

def head_masks(depth: np.ndarray) -> dict[str, np.ndarray]:
    """Masks over the recorded frame: the fixture grounder's box, a large one, a ragged one, one with a hole and a
    second component, one on the border and one with (almost) no depth."""

    boxes = {name: np.zeros(depth.shape, bool) for name in ("small", "large", "ragged", "two", "border", "void")}
    boxes["small"][180:210, 290:310] = True
    boxes["large"][150:300, 200:420] = True
    boxes["ragged"] = blobs(depth.shape, 12, 60) & (np.indices(depth.shape)[0] > 100)
    boxes["two"][160:260, 250:330] = True
    boxes["two"][200:220, 270:290] = False
    boxes["two"][300:330, 400:470] = True
    boxes["border"][400:480, 0:90] = True
    boxes["void"][5:25, 5:25] = True
    return boxes


def head_depth(head: dict) -> np.ndarray:
    """The recorded head depth (float64), with no measurement in the top-left corner (where the ``void`` mask lies)."""

    depth = head["depth"].astype(np.float64)
    depth[0:30, 0:30] = np.nan
    return depth


def close(found: dict, expected: dict) -> float:
    """Assert two localizations agree (the footprint fields aside); the largest float difference."""

    assert set(found) == set(expected) - FOOTPRINT
    worst = 0.0
    for name, value in found.items():
        other = expected[name]
        if isinstance(value, (float, list)) and value is not None and other is not None and name != "status":
            difference = np.max(np.abs(np.asarray(value, float) - np.asarray(other, float)))
            assert difference <= TOLERANCE_M, (name, difference)
            worst = max(worst, float(difference))
        else:
            assert value == other, name
    return worst


def test_mask_summary_equals_the_lib_s(head) -> None:
    depth = head_depth(head)
    for name, mask in head_masks(depth).items():
        expected = lib_vision.mask_summary(Array.of(mask), Array.of(depth)).json()
        assert vision.mask_summary(mask, depth) == expected, name
    with pytest.raises(ValueError, match="empty"):
        vision.mask_summary(np.zeros(depth.shape, bool), depth)


def test_localize_mask_equals_the_lib_s_on_the_recorded_frame(head) -> None:
    depth = head_depth(head)  # the tools measure in float64
    status, worst = {}, 0.0
    for name, mask in head_masks(depth).items():
        expected = lib_vision.localize_mask(Array.of(mask), Array.of(depth), JsonValue.of(head["calibration"]),
                                            extended_frame=True).json()
        found = vision.localize_mask(mask, depth, head["K"], head["camera_to_world"])
        status[name] = found["status"]
        worst = max(worst, close(found, expected))
    assert status["small"] == status["large"] == status["two"] == "localized"
    assert status["void"] == "insufficient_depth", status
    assert worst < TOLERANCE_M
    print(f"localize_mask: largest float difference from the lib {worst:.3g}")


def test_localize_mask_equals_the_lib_s_on_random_objects_standing_on_the_recorded_scene(head) -> None:
    """A rectangle made nearer than the recorded depth stands on it, so the support plane fit has the real table
    around it, and the ring and the gates meet their cases (noise, missing depth, steps)."""

    worst, supported = 0.0, 0
    for _ in range(16):
        depth = head["depth"].astype(np.float64) + RNG.normal(0, 0.0005, head["depth"].shape)
        mask = np.zeros(depth.shape, bool)
        top, left = RNG.integers(120, 300), RNG.integers(100, 450)
        mask[top:top + RNG.integers(20, 80), left:left + RNG.integers(20, 90)] = True
        depth[mask] -= RNG.uniform(0.02, 0.12)
        depth[RNG.random(depth.shape) > 0.995] = np.nan
        expected = lib_vision.localize_mask(Array.of(mask), Array.of(depth), JsonValue.of(head["calibration"]),
                                            extended_frame=True).json()
        found = vision.localize_mask(mask, depth, head["K"], head["camera_to_world"])
        supported += found["support_z_m"] is not None
        worst = max(worst, close(found, expected))
    assert supported >= 8, "most of these have a support to find"
    print(f"localize_mask on {supported}/16 supported random objects: largest float difference {worst:.3g}")


def test_a_malformed_calibration_or_mask_is_refused(head) -> None:
    mask = head_masks(head["depth"])["small"]
    with pytest.raises(ValueError):
        vision.localize_mask(mask[:10], head["depth"], head["K"], head["camera_to_world"])
    with pytest.raises(ValueError):
        vision.localize_mask(mask, head["depth"], np.zeros((3, 3)), head["camera_to_world"])
    with pytest.raises(ValueError):
        vision.localize_mask(mask, head["depth"], head["K"], np.diag([2.0, 1.0, 1.0, 1.0]))


# --- object_geometry, through the mount, against what vision_ground returned ---------------------------------------

def test_object_geometry_equals_the_localization_vision_ground_returned(published, tmp_path) -> None:
    api = api_for(published)
    grounded = ground(api, {"camera": "head", "text": "object", "max_candidates": 1})
    candidate = grounded["candidates"][0]
    expected = candidate["localization"]
    assert expected["support_z_m"] is not None, "the recorded frame has a support under the grounder's box"
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        scene = library_observation.load(str(mount.root))
        assert list(scene.targets) == [candidate["target_ref"]]
        found = object_geometry(candidate, observation=scene)
        assert object_geometry(candidate["target_ref"], observation=scene) == found
        assert object_geometry(scene.target(candidate["target_ref"]), observation=scene) == found, "the lib's own Target"
        worst = close({name: value for name, value in found.items()
                       if name not in ("target_ref", "camera", "observation_id")}, expected)
        assert (found.target_ref, found.camera, found.observation_id) == (
            candidate["target_ref"], "head", scene.observation_id)
        # typed values: world points and directions
        np.testing.assert_allclose(found.center.xyz, expected["center_xyz_m"], atol=TOLERANCE_M)
        assert found.center.frame == found.largest_axis.frame == "world"
        np.testing.assert_allclose(found.largest_axis.v, expected["largest_extent_axis_world"], atol=TOLERANCE_M)
        assert found.axes.v.shape == (3, 3) and found.extent_xyz_m == found["extent_xyz_m"]
        assert "height_above_support_m" in repr(found)
        print(f"object_geometry: largest float difference from vision_ground's localization {worst:.3g}")
        mask = scene.target(candidate["target_ref"]).mask  # the tool's own mask
        assert mask.shape == (480, 640) and int(mask.sum()) == expected["mask_pixel_count"]


def test_a_target_of_another_observation_or_a_wrong_argument_says_what_to_do(published) -> None:
    api = api_for(published)
    candidate = ground(api, {"camera": "head", "text": "object", "max_candidates": 1})["candidates"][0]
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        scene = library_observation.load(str(mount.root))
        with pytest.raises(KeyError, match="segment again"):
            object_geometry("target-" + "0" * 24, observation=scene)
        for bad in (3, {"result": {"candidates": []}}, {"rank": 0}):
            with pytest.raises(TypeError, match="target"):
                object_geometry(bad, observation=scene)
        assert candidate["target_ref"] in str(pytest.raises(KeyError, object_geometry, "target-" + "1" * 24,
                                                           observation=scene).value)
