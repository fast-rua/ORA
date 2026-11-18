"""RoboDojo three-camera perception path and ARX X5 grasp geometry."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ora.agent.composition.surface.tool_surface import (
    ToolSurfaceError, _resolve_view_config,
)
from ora.agent.composition.surface.view_config import AUXILIARY_CONTACT_SHEET_LAYOUTS
from ora.agent.context.observation.projection import (
    ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)
from ora.agent.toolkit.host import backend_selection
from ora.agent.toolkit.host.runtime import common as runtime_common
from ora.agent.toolkit.host.grasp.arx_x5_gripper import (
    ARX_X5_APERTURE_CAVEAT, ARX_X5_GRIPPER_GEOMETRY,
    ARX_X5_LINK6_COMPILER_IDENTITIES, ARX_X5_SOURCE_BINDING,
    ARX_X5_SOURCE_FILES, nominal_jaw_aperture_m,
)
from ora.agent.toolkit.host.shared.rgbd_fusion import fuse_rgbd_pointcloud_sources
from ora.agent.toolkit.host.grasp.compiler_registry import GRASP_COMPILER_IDENTITIES
from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT
from tests.support.robodojo import runtime as robodojo_runtime
from tests.support.robodojo.perception import BOUNDS, _sha256, _synthetic_robodojo_observation


ROBODOJO_CAMERAS = ("head", "left_wrist", "right_wrist")
# Exact key set RobodojoSession.observe writes per camera; the agreement test
# below re-derives it from the real writer through the fake plant.
ROBODOJO_CALIBRATION_KEYS = frozenset({
    "source_camera", "intrinsic_cv", "extrinsic_cv", "cam2world_cv",
    "depth_unit", "depth_array_order", "policy_quaternion_order",
    "world_frame", "depth_kind", "finite_depth_fraction",
    "renderer_calibration",
})
ROBODOJO_CAMERA_RECORD_KEYS = frozenset({
    "source_camera", "width", "height", "mount", "calibration_dynamic",
    "modalities", "files",
})
X5_ASSET_ROOT = robodojo_runtime.SOURCE_ROOT / (
    ARX_X5_SOURCE_BINDING["asset_root"]
)


def test_synthetic_layout_matches_the_real_writer(tmp_path: Path) -> None:
    """The synthetic key set equals what RobodojoSession.observe writes."""

    # Pytest puts tests/ on sys.path, so the backend module's fake plant is
    # importable without duplicating it.
    import tests.support.robodojo.backend as backend_fixture
    from ora.environment.backends.robodojo.session import RobodojoSession

    plant = backend_fixture.Plant()
    session = RobodojoSession(
        config={"workspace_bounds_m": BOUNDS}, output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None), {
            "schema": "ora.robodojo-source-manifest.v1",
            "source_revisions": {"robodojo": "1" * 40},
        }),
    )
    intrinsic = np.array([[50.0, 0.0, 5.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]])
    session._calibration = lambda name: (intrinsic, np.eye(4))
    for source in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        session.calibration_readbacks[source] = {
            "original_robodojo_intrinsic": intrinsic, "cam2world_cv": np.eye(4),
            "parameters": {"renderProductResolution": np.array([10, 8])},
        }
    session.start()
    real, _ = session.observe(tmp_path / "observations")
    real_root = tmp_path / "observations" / real["observation_id"]
    synthetic_root, synthetic = _synthetic_robodojo_observation(tmp_path / "synthetic")
    for public in ROBODOJO_CAMERAS:
        real_keys = set(json.loads((real_root / public / "calibration.json").read_text()))
        synthetic_keys = set(json.loads((synthetic_root / public / "calibration.json").read_text()))
        assert real_keys == synthetic_keys == ROBODOJO_CALIBRATION_KEYS
        assert set(real["cameras"][public]) == set(synthetic["cameras"][public])
        assert set(real["cameras"][public]) == ROBODOJO_CAMERA_RECORD_KEYS
        assert set(real["cameras"][public]["files"]) == {
            "rgb_lossless_png", "depth_numpy_m_f32", "calibration",
        }
        assert np.load(real_root / public / "depth-m.npy").dtype == np.float32
    assert set(real["cameras"]) == set(ROBODOJO_CAMERAS) == set(
        ARX_X5_CONTRACT["public_cameras"]
    )
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=real_root, observation=real,
        workspace_bounds_m=BOUNDS, camera_names=ROBODOJO_CAMERAS,
        voxel_size_m=0.005, max_depth_m=2.5,
    )
    assert [item["camera"] for item in fused["sources"]] == list(ROBODOJO_CAMERAS)
    assert fused["workspace_point_count"] == 3 * 8 * 10


def test_aloha_compiled_candidates_keep_their_frozen_shape() -> None:
    """The caveat field appears only for identities that declare the flag."""

    from ora.agent.toolkit.host.grasp.compiler import _compile_grasp_candidate
    from ora.agent.toolkit.host.grasp.compilation import _public_grasp_compiler_identity

    raw = {
        "candidate_id": "grasp-0123456789abcdef01234567", "rank": 0,
        "source_gripper_origin_xyz_m": [0.0, 0.0, 0.6966],
        "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.6566],
        "approach_world": [0.0, 0.0, 1.0], "jaw_world": [1.0, 0.0, 0.0],
        "gripper_opening_m": 0.05, "backend_score": 0.7,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed"},
    }
    aloha = GRASP_COMPILER_IDENTITIES["contact_graspnet_to_robotwin_aloha_left_grip_center_v3"]
    aloha_compiled = _compile_grasp_candidate(raw, compiler_identity=aloha)
    assert "aperture_calibration" not in aloha_compiled["compiler_checks"]
    assert set(_public_grasp_compiler_identity(aloha)) == {
        "id", "embodiment_descriptor_id", "controlled_point",
        "controlled_point_semantics", "arm",
    }
    x5 = GRASP_COMPILER_IDENTITIES["contact_graspnet_to_robodojo_x5_right_link6_v1"]
    x5_compiled = _compile_grasp_candidate(raw, compiler_identity=x5)
    assert x5_compiled["compiler_checks"]["aperture_calibration"] == ARX_X5_APERTURE_CAVEAT
    with pytest.raises(ValueError, match="exceeds embodiment limit"):
        _compile_grasp_candidate({**raw, "gripper_opening_m": 0.085}, compiler_identity=x5)


def test_x5_gripper_geometry_is_internally_consistent() -> None:
    geometry = ARX_X5_GRIPPER_GEOMETRY
    command = ARX_X5_CONTRACT["gripper_command"]
    assert geometry["command_joint_range_m"] == command["primary_joint_range"]
    assert geometry["mimic_multiplier"] == command["mimic_multiplier"]
    assert geometry["physical_jaw_aperture_calibrated"] is (
        command["physical_jaw_aperture_calibrated"]
    ) is False
    low, high = geometry["pad_inner_face_span_link6_x_m"]
    assert low < geometry["source_declared_gripper_bias_m"] < geometry[
        "contact_baseline_link6_x_m"
    ] < high
    assert nominal_jaw_aperture_m(1.0) == pytest.approx(
        geometry["nominal_maximum_aperture_m"], abs=1e-6,
    )
    assert nominal_jaw_aperture_m(0.0) == 0.0
    samples = [nominal_jaw_aperture_m(value / 10) for value in range(11)]
    assert samples == sorted(samples)
    with pytest.raises(ValueError):
        nominal_jaw_aperture_m(1.5)
    for arm, identity in ARX_X5_LINK6_COMPILER_IDENTITIES.items():
        assert GRASP_COMPILER_IDENTITIES[identity["id"]] is identity
        assert identity["arm"] == arm
        assert identity["source_origin_to_controlled_point_m"] + identity[
            "controlled_point_to_contact_baseline_m"
        ] == pytest.approx(identity["contact_graspnet_gripper_depth_m"], abs=1e-9)
        assert identity["maximum_opening_m"] < geometry["nominal_maximum_aperture_m"]
        assert identity["source_binding"] is ARX_X5_SOURCE_BINDING


def test_x5_geometry_provenance_matches_the_pinned_assets() -> None:
    if not X5_ASSET_ROOT.is_dir():
        pytest.skip("pinned RoboDojo x5 assets are not present on this host")
    for key, relative in ARX_X5_SOURCE_FILES.items():
        assert _sha256(X5_ASSET_ROOT / relative) == ARX_X5_SOURCE_BINDING[key], key
    urdf = (X5_ASSET_ROOT / "X5A.urdf").read_text(encoding="utf-8")
    origin = ARX_X5_GRIPPER_GEOMETRY["finger_joint_origin_link6_m"]
    assert f'<origin xyz="{origin["x"]} 0.024896 -0.0002436"' in urdf
    assert '<limit lower="0" upper="0.044"' in urdf
    config = (X5_ASSET_ROOT / "robot_config.yml").read_text(encoding="utf-8")
    assert "gripper_bias: 0.145" in config and "gripper_scale: [-0.01, 0.044]" in config


def test_three_camera_contact_sheet_is_one_labeled_row(tmp_path: Path) -> None:
    camera_paths: dict[str, Path] = {}
    for index, name in enumerate(ROBODOJO_CAMERAS):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (64, 48), (10 * index, 40, 60)).save(path)
        camera_paths[name] = path
    derived = materialize_auxiliary_contact_sheet(
        run_root=tmp_path, observation_step=2, camera_paths=camera_paths,
        cameras=ROBODOJO_CAMERAS,
    )
    assert (derived["width"], derived["height"]) == (64 * 3, 24 + 48)
    manifest = json.loads(Path(derived["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["identity"]["layout"] == "one_by_three_native_rgb_with_label_header"
    assert manifest["identity"]["camera_order"] == list(ROBODOJO_CAMERAS)
    with Image.open(derived["path"]) as sheet:
        # Each cell keeps its native pixels under the label header.
        assert sheet.getpixel((0, 24)) == (0, 40, 60)
        assert sheet.getpixel((64 * 2, 24)) == (20, 40, 60)
    assert AUXILIARY_CONTACT_SHEET_LAYOUTS[4][0] == "two_by_two_native_rgb_with_label_header"
    with pytest.raises(ObservationProjectionError, match="three or four unique"):
        materialize_auxiliary_contact_sheet(
            run_root=tmp_path, observation_step=3,
            camera_paths={name: camera_paths[name] for name in ROBODOJO_CAMERAS[:2]},
            cameras=ROBODOJO_CAMERAS[:2],
        )


def test_view_config_admits_three_sheet_cameras_or_no_sheet() -> None:
    base = {
        "schema": "ora.policy-view-config.v1",
        "rgb_resolution": "native", "derived_resolution": [256, 256],
        "default_observe_views": list(ROBODOJO_CAMERAS),
        "coordinate_convention": "opencv_top_left_xy",
    }
    capabilities = ["automatic_current_auxiliary_contact_sheet_v1"]
    with_sheet = _resolve_view_config({
        "view_config": {
            **base, "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
            "auxiliary_contact_sheet_cameras": list(ROBODOJO_CAMERAS),
        },
        "capabilities": capabilities,
    })
    assert with_sheet["auxiliary_contact_sheet_cameras"] == list(ROBODOJO_CAMERAS)
    without_sheet = _resolve_view_config({
        "view_config": {**base, "automatic_actual_views": ["head"]},
        "capabilities": [],
    })
    assert "auxiliary_contact_sheet_cameras" not in without_sheet
    with pytest.raises(ToolSurfaceError, match="three or four unique"):
        _resolve_view_config({
            "view_config": {
                **base, "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
                "auxiliary_contact_sheet_cameras": ["head", "left_wrist"],
            },
            "capabilities": capabilities,
        })


def test_tool_runtime_selects_robodojo_cameras_and_x5_compiler_by_capability() -> None:
    def surface(*capabilities: str) -> SimpleNamespace:
        return SimpleNamespace(supports=lambda name: name in capabilities)

    robodojo = surface(
        "dual_arm_grasp_plan_v1", backend_selection.ROBODOJO_RGBD_CAPABILITY,
        backend_selection.ARX_X5_GRASP_COMPILER_CAPABILITY,
    )
    assert backend_selection.dual_arm_grasp_scene_cameras(robodojo, "right") == ("head", "right_wrist")
    assert backend_selection.dual_arm_grasp_compiler_id(robodojo, "right") == (
        "contact_graspnet_to_robodojo_x5_right_link6_v1"
    )
    robotwin = surface(
        "dual_arm_grasp_plan_v1", "robotwin_rgbd_v1",
        "contact_graspnet_robotwin_aloha_grip_center_compiler_v3",
    )
    assert backend_selection.dual_arm_grasp_scene_cameras(robotwin, "left") == ("head_camera", "left_camera")
    assert backend_selection.dual_arm_grasp_compiler_id(robotwin, "left") == (
        "contact_graspnet_to_robotwin_aloha_left_grip_center_v3"
    )
    assert backend_selection.dual_arm_grasp_compiler_id(surface("dual_arm_grasp_plan_v1"), "left") == (
        "contact_graspnet_to_robotwin_aloha_left_end_effector_v1"
    )
    assert backend_selection.dual_arm_grasp_compiler_id(surface(), "left") is None
    assert set(ROBODOJO_CAMERAS) == runtime_common.ROBODOJO_POLICY_CAMERAS


# --------------------------------------------------------------------------
# grounding_candidate_depth_profile_v1: the candidate card's depth strip.
# --------------------------------------------------------------------------

def _profile_card(*, support_line: bool, localization: dict) -> tuple[np.ndarray, dict]:
    from ora.agent.toolkit.host.perception import grounding_preview

    rgb = np.full((48, 64, 3), 60, dtype=np.uint8)
    depth = np.full((48, 64), 1.2, dtype=np.float64)
    depth[14:34, 20:40] = 0.8
    mask = np.zeros((48, 64), dtype=bool)
    mask[14:34, 20:40] = True
    calibration = Path(localization.pop("_calibration"))
    image, metadata = grounding_preview.render_candidate_preview(
        rgb_u8=rgb, text="red mug",
        candidate={"rank": 0, "mask": mask, "bbox_xyxy": [20, 14, 40, 34],
                   "target_ref": "target-" + "0" * 24, "backend_score": 0.9},
        depth_profile=grounding_preview.profile_input(
            {"renderer": grounding_preview.PROFILE_RENDERER, "support_line": support_line},
            depth_m=depth, calibration_path=calibration, localization=localization),
    )
    return np.asarray(image), metadata["depth_profile"]


def test_the_support_line_follows_the_support_capability_and_the_window(tmp_path: Path) -> None:
    observation_root, _observation = _synthetic_robodojo_observation(tmp_path / "obs")
    calibration = str(observation_root / "head" / "calibration.json")
    localized = {"status": "localized", "center_xyz_m": [0.0, 0.0, 0.8],
                 "extent_xyz_m": [0.2, 0.2, 0.0], "support_z_m": 0.75}
    card, record = _profile_card(support_line=True, localization={**localized, "_calibration": calibration})
    assert record["support_z_m"] == 0.75 and record["views"][0]["support_line_drawn"] is True
    assert record["views"][1]["support_line_drawn"] is False, "the top view has no height axis"
    support_row = 580 + 22 + int((0.8 + 0.25 - 0.75) / record["metres_per_pixel"])
    assert np.all(card[support_row, 62:500] == (255, 176, 64), axis=-1).sum() > 100
    # A surface without target_support_height_v1 never draws the withheld number.
    _card, withheld = _profile_card(support_line=False, localization={**localized, "_calibration": calibration})
    assert withheld["support_z_m"] is None
    assert not any(view["support_line_drawn"] for view in withheld["views"])
    # Out of the window: recorded, not drawn.
    _card, outside = _profile_card(support_line=True, localization={
        **localized, "support_z_m": 0.2, "_calibration": calibration})
    assert outside["support_z_m"] == 0.2 and not outside["views"][0]["support_line_drawn"]
    # No localized depth: the strip says so instead of drawing an empty window.
    _card, missing = _profile_card(support_line=True, localization={
        "status": "insufficient_depth", "center_xyz_m": None, "_calibration": calibration})
    assert missing == {"renderer": "grounding-candidate-depth-profile.v1", "status": "unavailable",
                       "options": {"renderer": "grounding-candidate-depth-profile.v1",
                                   "support_line": True},
                       "reason": "candidate_not_localized"}


def test_only_a_surface_declaring_the_depth_profile_draws_it(tmp_path: Path) -> None:
    import base64
    import io
    from dataclasses import replace

    import tests.support.robodojo.closed_skills_default as parity
    from tests.support.agent.surface_catalog import surface_of
    from ora.agent.toolkit.host.perception import grounding_preview

    live = surface_of("robodojo_placement_toolkit_v5")
    assert grounding_preview.options(live) == {
        "renderer": "grounding-candidate-depth-profile.v1", "support_line": True}
    # The same Dojo surface without the capability keeps the released card.
    released = replace(live, capabilities=tuple(
        item for item in live.capabilities if item != grounding_preview.CAPABILITY))
    assert grounding_preview.options(released) is None
    sizes = {}
    for name, surface in (("released", released), ("live", live)):
        runtime = parity.tool_runtime(tmp_path / name)
        runtime.surface = surface
        runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
        result = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
        with Image.open(io.BytesIO(base64.b64decode(result["images_base64"][1]))) as image:
            sizes[name] = image.size
        assert result["candidate_previews"][0]["render_dimensions"] == list(sizes[name])
    assert sizes == {"released": (1024, 576), "live": (1024, 800)}


# --------------------------------------------------------------------------
# observed_extent_scene_review_v1: the global review's observed-extent cell.
# --------------------------------------------------------------------------
