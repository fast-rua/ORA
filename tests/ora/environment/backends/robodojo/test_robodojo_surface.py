"""RoboDojo/ARX X5 ToolSurface, observe gate and workspace profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.agent.composition.surface import tool_surface as surface_module
from ora.agent.composition.surface.tool_surface import (
    ToolSurfaceError, validate_schema_instance,
)
from tests.support.agent.surface_catalog import catalog_grants, surface_of
from tests.support.agent.toolkit_run import served
from ora.agent.toolkit.host.grasp.arx_x5_gripper import ARX_X5_APERTURE_CAVEAT
from ora.agent.toolkit.host.runtime.core import ManipulationToolRuntime
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.error_classes import ToolError
from ora.environment.workspace_profiles import (
    ROBODOJO_ARX_X5_TABLETOP_V1, resolve_workspace_profile,
)

import tests.support.robodojo.backend as backend
import tests.support.robodojo.perception as perception

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")
ROOT = Path(__file__).resolve().parents[5]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
ROBODOJO_CAMERAS = ["head", "left_wrist", "right_wrist"]
ROBOTWIN_CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
ROBODOJO = "robodojo_placement_toolkit_v5"
ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"


def _robodojo():
    return surface_of(ROBODOJO)


def _robotwin():
    return surface_of(ROBOTWIN)


def _x5_grasp_result(tmp_path: Path) -> dict[str, object]:
    """The public grasp_plan result the live RoboDojo runtime admits for the X5."""
    import tests.support.robodojo.closed_skills_default as parity
    from tests.support.agent.reach_check import _stub_bridge

    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = _robodojo()
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    runtime.align_plan_callback = _stub_bridge([])
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": "left"})
    # The delivered images ride beside the admitted result, outside its contract.
    result.pop("images_base64", None)
    return result


def test_x5_grasp_result_passes_the_robodojo_override_and_fails_robotwin(tmp_path: Path) -> None:
    from ora.agent.toolkit.host.grasp.aloha_gripper import ALOHA_NOMINAL_MAXIMUM_OPENING_M

    result = _x5_grasp_result(tmp_path)
    assert result["camera"] == "head" and result["arm"] == "left"
    assert result["compiler"]["embodiment_descriptor_id"] == "arx_x5_robodojo_v1"
    assert result["compiler"]["physical_jaw_aperture_calibrated"] is False
    assert result["compiler"]["aperture_caveat"] == ARX_X5_APERTURE_CAVEAT
    assert result["candidates"][0]["compiler_checks"]["aperture_calibration"] == (
        ARX_X5_APERTURE_CAVEAT
    )
    validate_schema_instance(result, _robodojo().result_schemas["grasp_plan"])
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(result, _robotwin().result_schemas["grasp_plan"])
    # The declared opening bound is the 80 mm X5 admission limit, not ALOHA's
    # nominal maximum; the compiler enforces it (test_robodojo_perception), the
    # result validator only checks types, enums, consts and required fields.
    opening = _robodojo().result_schemas["grasp_plan"]["$defs"]["candidate"][
        "properties"]["gripper_opening_m"]
    assert opening["maximum"] == 0.08
    assert _robotwin().result_schemas["grasp_plan"]["$defs"]["candidate"][
        "properties"]["gripper_opening_m"]["maximum"] == ALOHA_NOMINAL_MAXIMUM_OPENING_M
    compiler = _robodojo().result_schemas["grasp_plan"]["properties"]["compiler"]
    assert compiler["properties"]["controlled_point_semantics"]["const"] == "robodojo_link6_pose"
    assert "physical_jaw_aperture_calibrated" in compiler["required"]


def test_native_dual_arm_causal_accepts_a_robodojo_proprioception(
    tmp_path: Path,
) -> None:
    """The join the first live RoboDojo K3 launch failed before any model call.

    `dual_arm_sparse_targets_v1` requires a categorical command target for both
    arms in the causal actual observation; the RoboDojo session publishes it,
    so the Native check reads it rather than raising on the first observation.
    """
    from types import SimpleNamespace

    from ora.agent.pi.policy_session import PiPlanner

    session = backend.sparse_session(tmp_path / "backend")
    session.execute(backend.v3(
        session, tmp_path / "backend", kind="set_gripper",
        targets={"left": "close"},
    ))
    payload, _paths = session.observe(tmp_path / "backend" / "observations")
    effectors = payload["proprioception"]["end_effectors"]

    planner = object.__new__(PiPlanner)
    planner.run_root = tmp_path / "run"
    planner.tools = SimpleNamespace(surface=_robodojo())
    # The live surface's camera and public-fact delivery is covered by the
    # current-visual tests; this is the end-effector join it ends in.
    actual = planner._actual_end_effector_snapshot({
        "policy_state": {"step": 0, "path": "policy-state.json"},
        "proprioception": payload["proprioception"],
    }, [], {})
    assert actual["end_effectors"] == {
        arm: {
            "pose_xyz_xyzw": [round(float(value), 6)
                              for value in effectors[arm]["pose_xyz_xyzw"]],
            "gripper_command_target": "close" if arm == "left" else "open",
        }
        for arm in ("left", "right")
    }


def _runtime(tmp_path: Path, profile: str) -> ManipulationToolRuntime:
    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    return served(ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=perception.BOUNDS, state_root=None,
        tool_surface_profile=profile, toolkit_grants=catalog_grants(profile), controlled_end_effector="right",
        public_facts_projector=project_policy_public_facts,
    ))


def _publish_robodojo_observation(tmp_path: Path) -> str:
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA

    root, observation = perception._synthetic_robodojo_observation(
        tmp_path / "observations",
    )
    manifest = root / "observation.json"
    # The live surface delivers public robot facts only for geometry verified
    # against its descriptor.
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA, "scope": "loaded_robot_geometry_only", "passed": True,
        "descriptor_sha256": _robodojo().view_config["public_geometry_descriptor"]["sha256"],
    }
    manifest.write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    (control / "current-observation.json").write_text(json.dumps({
        "observation_id": observation["observation_id"],
        "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_root": str(root), "generation": 0, "open": True,
    }), encoding="utf-8")
    return str(observation["observation_id"])


def test_observe_serves_robodojo_cameras_only_under_its_capability(
    tmp_path: Path, ready_model_services: None,
) -> None:
    observation_id = _publish_robodojo_observation(tmp_path)
    runtime = _runtime(tmp_path, ROBODOJO)
    result = runtime.invoke("observe", {})
    assert result["observation_id"] == observation_id
    assert result["views"] == ROBODOJO_CAMERAS
    assert len(result["images_base64"]) == 3
    assert result["view_dimensions"] == {
        name: {"width": 64, "height": 48} for name in ROBODOJO_CAMERAS
    }
    effectors = result["public_robot_facts"]["end_effectors"]
    assert set(effectors) == {"left", "right"}
    # The published RoboDojo command is the executed normalized directive, so
    # the public read is the category it stands for, never "unknown".
    for arm in ("left", "right"):
        assert effectors[arm]["gripper"]["command_target"] == "open"
    wrist = runtime.invoke("observe", {"views": ["right_wrist"]})
    assert wrist["views"] == ["right_wrist"] and len(wrist["images_base64"]) == 1
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("observe", {"views": ["head_camera"]})
    assert rejected.value.as_dict()["constraints"] == {"enum": ROBODOJO_CAMERAS, "violated": "enum"}
    with pytest.raises(ToolError, match="items must be unique"):
        runtime.invoke("observe", {"views": ["head", "head"]})

    robotwin = _runtime(tmp_path, ROBOTWIN)
    with pytest.raises(ToolError) as unknown:
        robotwin.invoke("observe", {"views": ["head"]})
    assert unknown.value.as_dict()["constraints"] == {"enum": ROBOTWIN_CAMERAS, "violated": "enum"}
    assert "outside the input enum" in unknown.value.as_dict()["message"]


def test_robodojo_workspace_profile_covers_the_table_and_reset_poses() -> None:
    profile = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)
    assert profile["environment_id"] == "robodojo"
    assert profile["embodiment_id"] == "arx_x5"
    assert profile["frame"] == "robodojo_episode_local"
    assert profile["semantics"] == "hard_action_envelope_not_reachability"
    assert profile["provenance"]["backend_revision"] == (
        "ee67a1468510da7624a089164402359f2afc72c8"
    )
    bounds = profile["bounds_m"]
    # Table top: 1.4 x 1.1 m centred at (0, -0.05), surface z = 0.765.
    for x, y in ((-0.7, -0.6), (0.7, 0.5)):
        assert bounds["x"][0] <= x <= bounds["x"][1]
        assert bounds["y"][0] <= y <= bounds["y"][1]
    assert bounds["z"][0] < 0.765 < bounds["z"][1]
    for x, y, z in ((-0.3, -0.352, 0.922), (0.3, -0.352, 0.922)):
        assert bounds["x"][0] <= x <= bounds["x"][1]
        assert bounds["y"][0] <= y <= bounds["y"][1]
        assert bounds["z"][0] <= z <= bounds["z"][1]
    assert profile["profile_sha256"] == resolve_workspace_profile(
        ROBODOJO_ARX_X5_TABLETOP_V1,
    )["profile_sha256"]
    with pytest.raises(ValueError, match="unknown action workspace profile"):
        resolve_workspace_profile("robodojo_unknown_profile")


def test_sheet_camera_counts_follow_the_projection_layouts(monkeypatch) -> None:
    from ora.agent.composition.surface import view_config

    base = {
        "schema": "ora.policy-view-config.v1",
        "rgb_resolution": "native", "derived_resolution": [256, 256],
        "default_observe_views": ["head"],
        "coordinate_convention": "opencv_top_left_xy",
        "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
    }
    capabilities = ["automatic_current_auxiliary_contact_sheet_v1"]
    two = {"view_config": {**base, "auxiliary_contact_sheet_cameras": ["a", "b"]},
           "capabilities": capabilities}
    with pytest.raises(ToolSurfaceError, match="admitted counts: 3, 4"):
        surface_module._resolve_view_config(two)
    monkeypatch.setitem(
        view_config.AUXILIARY_CONTACT_SHEET_LAYOUTS, 2,
        ("one_by_two_test_layout", 2, 1),
    )
    assert surface_module._resolve_view_config(two)["auxiliary_contact_sheet_cameras"] == ["a", "b"]
