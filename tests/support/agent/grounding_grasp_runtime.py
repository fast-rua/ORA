"""Test helpers for grounding grasp runtime, shared by several test modules.

They were defined in
``tests/ora/agent/toolkit/host/grasp/test_grounding_grasp_runtime.py``; test modules import
them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from tests.support.agent.surface_catalog import catalog_grants, handlers_only, surface_of
import hashlib
import json
from pathlib import Path
from unittest import mock

import httpx

import numpy as np
from PIL import Image

from ora.agent.context import ContextProjector
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from tests.support.agent.toolkit_tools import grasp_service
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
# The maintained single-arm LIBERO-Pro and dual-arm RoboTwin defaults.
LIBERO_SURFACE = "libero_placement_direct_overhead_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_object_observation(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "environment", "grasp-plan", 0, max_actions=4,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    observation_root = source / observation["observation_id"]
    camera_root = observation_root / "agentview"
    rgb_path = camera_root / "rgb.png"
    depth_path = camera_root / "depth-m.npy"
    calibration_path = camera_root / "calibration.json"
    rgb = np.full((64, 64, 3), [48, 52, 60], dtype=np.uint8)
    rgb[18:44, 22:40] = [210, 48, 32]
    rgb[22:42, 44:58] = [48, 80, 220]
    Image.fromarray(rgb).save(rgb_path, format="PNG")
    depth = np.full((64, 64), 1.2, dtype=np.float32)
    depth[18:44, 22:40] = 0.8
    depth[22:42, 44:58] = 0.85
    np.save(depth_path, depth, allow_pickle=False)
    calibration_path.write_text(json.dumps({
        "intrinsic_cv": [
            [400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0],
        ],
        "extrinsic_cv": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "cam2world_gl": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "depth_unit": "metres",
    }, sort_keys=True), encoding="utf-8")
    record = observation["cameras"]["agentview"]
    record["width"] = 64
    record["height"] = 64
    for role, path in {
        "rgb_lossless_png": rgb_path,
        "depth_numpy_m_f32": depth_path,
        "calibration": calibration_path,
    }.items():
        record["files"][role] = {
            "path": path.relative_to(observation_root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    wrist_root = observation_root / "wrist"
    wrist_rgb_path = wrist_root / "rgb.png"
    wrist_depth_path = wrist_root / "depth-m.npy"
    wrist_calibration_path = wrist_root / "calibration.json"
    wrist_rgb = np.full((64, 64, 3), [76, 58, 112], dtype=np.uint8)
    Image.fromarray(wrist_rgb).save(wrist_rgb_path, format="PNG")
    wrist_depth = np.full((64, 64), 1.2, dtype=np.float32)
    wrist_depth[6:16, 6:16] = 0.95
    np.save(wrist_depth_path, wrist_depth, allow_pickle=False)
    wrist_calibration_path.write_text(
        calibration_path.read_text(encoding="utf-8"), encoding="utf-8",
    )
    wrist_record = observation["cameras"]["wrist"]
    wrist_record["width"] = 64
    wrist_record["height"] = 64
    for role, path in {
        "rgb_lossless_png": wrist_rgb_path,
        "depth_numpy_m_f32": wrist_depth_path,
        "calibration": wrist_calibration_path,
    }.items():
        wrist_record["files"][role] = {
            "path": path.relative_to(observation_root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    (observation_root / "observation.json").write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    control = tmp_path / "control"
    ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, PROJECT,
        expose_manual=False, main_camera="agentview",
    ).publish(
        observation=observation,
        observation_root=observation_root,
        task_instruction="pick the red object",
        remaining_budget={"robot_decisions": 4, "model_calls": 12},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    return control, observation["observation_id"]


class FakeSam3:
    backend_id = "sam3_text_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"fixture": "two_text_candidates"}

    def require_ready(self) -> dict[str, object]:
        return {"ready": True, "load_count": 1}

    def segment(self, **_kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        mask = np.zeros((64, 64), dtype=bool)
        mask[18:44, 22:40] = True
        return mask, {"fixture": "point"}

    def ground(
        self, *, rgb_u8: object, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert np.asarray(rgb_u8).shape == (64, 64, 3)
        assert text == "red object"
        first = np.zeros((64, 64), dtype=bool)
        first[18:44, 22:40] = True
        second = np.zeros((64, 64), dtype=bool)
        second[22:42, 44:58] = True
        values = [
            {"source_index": 0, "rank": 0, "mask": first,
             "backend_score": 0.82, "bbox_xyxy": [22, 18, 40, 44]},
            {"source_index": 1, "rank": 1, "mask": second,
             "backend_score": 0.41, "bbox_xyxy": [44, 22, 58, 42]},
        ][:max_candidates]
        return values, {
            "request_id": "ground-test",
            "returned_candidate_count": len(values),
        }


class FakeContactGraspNet:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"pregrasp_offset_m": 0.04}

    def require_ready(self) -> dict[str, object]:
        return {"ready": True, "load_count": 1}

    def propose(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        # A surface declaring grasp_plan_ik_reachability_annotation_v1 asks for
        # a pool of up to 32 and applies the display cap after IK selection.
        assert kwargs["max_candidates"] <= 32
        scene = np.asarray(kwargs["scene_points_camera_m"])
        target = np.asarray(kwargs["target_points_camera_m"])
        assert scene.ndim == 2 and scene.shape[1] == 3
        assert target.shape == (26 * 18, 3)
        # The single-view boundary used to be asserted here, which encoded a
        # transport decision as a test-double invariant: the moment the proposer
        # received the fused scene the canvas already showed, this double would
        # have rejected it. Record the fact instead of forbidding it.
        self.saw_second_camera_depth = bool(np.any(np.isclose(scene[:, 2], 0.95)))
        self.last_scene_points_camera_m = scene.copy()
        candidate = {
            "candidate_id": "grasp-0123456789abcdef01234567",
            "source_index": 0,
            "rank": 0,
            "source_gripper_origin_xyz_m": [0.0, 0.0, 0.9034],
            "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.9434],
            "approach_world": [0.0, 0.0, -1.0],
            "jaw_world": [1.0, 0.0, 0.0],
            "gripper_opening_m": 0.04,
            "backend_score": 0.75,
            "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
            "checks": {
                "workspace": "passed", "aperture": "passed",
                "ik": "not_checked", "collision": "not_checked",
            },
            "execution_ready": False,
        }
        pose = np.eye(4, dtype=np.float64)[None]
        pose[0, :3, 3] = [0.0, 0.0, 0.9034]
        return [candidate], {
            "request_id": "grasp-test",
            "rejected": [],
            "_backend_input_regime": {
                "schema": "ora.contact-graspnet-input-regime.v1",
                "policy": "distinct_shuffled_crop_points_v1", "status": "region",
                "crop_points": 2029, "unique_points": 2029,
                "configured_input_points": 20000, "input_points": 2029,
                "padding_factor": 1.0, "upstream_padding_factor": 9.8571,
            },
            "_canonical_backend_output": {
                "poses_camera": pose,
                "scores": np.asarray([0.75], dtype=np.float64),
                "contact_points_camera": np.asarray(
                    [[0.0, 0.0, 1.0068]], dtype=np.float64,
                ),
                "openings_m": np.asarray([0.04], dtype=np.float64),
            },
        }


def _select_correctness_surface(runtime, profile=LIBERO_SURFACE, *, serve=False):
    """``runtime`` on ``profile``, its motion probe finding nothing reachable. ``serve=True`` binds the surface as a
    run does and serves it from the toolkit (``toolkit_run``); otherwise the handlers serve it."""
    if serve:
        from tests.support.agent.toolkit_run import on_surface
        on_surface(runtime, profile)
    else:
        runtime.surface = surface_of(profile)
        runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    runtime._motion_probe = lambda arguments: {"results": [
        {"query_id": query["query_id"], "reachable": False, "variants": [], "selected_variant_id": None, "plan_ref": None}
        for query in arguments["queries"]
    ]}
    if serve:
        from tests.support.agent.toolkit_run import served
        served(runtime)
    return runtime


def _runtime(
    tmp_path: Path, *, grasp_backend: object | None = None,
    profile: str = LIBERO_SURFACE, toolkit: bool = False,
) -> tuple[ManipulationToolRuntime, str]:
    """The runtime over the object observation. ``toolkit=True`` constructs it for the toolkit, with the surface's
    own grasp client against the in-process fake service, for the caller to serve (``toolkit_run.served``); otherwise
    the handlers serve it, with the proposer test double."""
    control, observation_id = _publish_object_observation(tmp_path)
    sam = FakeSam3()
    with mock.patch.object(ContactGraspNetApiProposer, "require_ready", lambda self: {"fixture": True}):
        runtime = ManipulationToolRuntime(
            control_root=control,
            manual_path=MANUAL,
            event_path=tmp_path / "events.jsonl",
            workspace_bounds_m=BOUNDS,
            state_root=None,
            tool_surface_profile=profile, toolkit_grants=catalog_grants(profile),
            controlled_end_effector="right",
            segmentation_backend=sam,
            grounding_backend=sam,
            grasp_backend=grasp_backend or (None if toolkit else FakeContactGraspNet()),
            public_facts_projector=project_policy_public_facts,
        )
    if not toolkit:
        return handlers_only(runtime), observation_id
    contact = getattr(runtime.grasp_backend, "contact_backend", runtime.grasp_backend)
    if isinstance(contact, ContactGraspNetApiProposer):
        contact._client.close()
        object.__setattr__(contact, "_client", httpx.Client(transport=httpx.MockTransport(grasp_service([]))))
        object.__setattr__(contact, "require_ready", lambda: {"fixture": True})
    return runtime, observation_id
