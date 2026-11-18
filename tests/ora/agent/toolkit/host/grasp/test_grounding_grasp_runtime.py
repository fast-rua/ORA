"""Shared grounding/grasp Runtime contracts."""

from __future__ import annotations

import base64
import copy
import io
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ora.agent.composition import resolve_tool_surface
from ora.agent.model_services.identity import grounding_grasp_adapter_sha256
from ora.agent.toolkit.host.runtime import core as runtime_core
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.model_services.clients import (
    CONTACT_GRASPNET_MODEL_IDENTITY,
    SAM3_MODEL_IDENTITY,
    Sam3ApiSegmenter,
)
from ora.agent.toolkit.host.grasp.compiler import (
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    _compile_grasp_candidate,
    _compiled_grasp_geometry,
)
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.agent.toolkit.host.shared.rgbd_fusion import fuse_rgbd_pointcloud_sources
from ora.agent.error_classes import ToolError
from viewer.trajectory import compile_trajectory
from tests.support.agent.grounding_grasp_runtime import (
    ROOT, MANUAL, BOUNDS, LIBERO_SURFACE, _sha256, _publish_object_observation, _runtime,
)


ROBOTWIN_SURFACE = "robotwin_toolchain_b2_closed_noservo_v2"


def test_model_services_are_shared_and_experiment_free() -> None:
    shared = ROOT / "ora/agent/model_services"
    assert shared.is_dir()
    assert "experiments." not in (
        shared / "resident.py"
    ).read_text(encoding="utf-8")
    assert "experiments." not in (
        shared / "grounding_grasp.py"
    ).read_text(encoding="utf-8")
    identity_source = (shared / "identity.py").read_text(encoding="utf-8")
    for name in ("identity.py", "resident.py", "grounding_grasp.py"):
        assert f'"{name}"' in identity_source
    assert len(grounding_grasp_adapter_sha256()) == 64


def test_contact_graspnet_origin_compiles_to_panda_grip_site() -> None:
    approach = np.asarray([0.437330037083, 0.099535647523, -0.893775751258])
    approach /= np.linalg.norm(approach)
    jaw = np.asarray([-0.875623586418, -0.179426567165, -0.448429974359])
    source_origin = np.asarray([0.075219177673, 0.004896843306, 0.167504583204])
    candidate = {
        "candidate_id": "grasp-0123456789abcdef01234567",
        "source_index": 0,
        "rank": 0,
        "source_gripper_origin_xyz_m": source_origin.tolist(),
        "source_pregrasp_origin_xyz_m": (source_origin - approach * 0.04).tolist(),
        "approach_world": approach.tolist(),
        "jaw_world": jaw.tolist(),
        "gripper_opening_m": 0.008972176351,
        "backend_score": 0.2892,
        "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed"},
        "execution_ready": False,
    }
    geometry = _compiled_grasp_geometry(candidate)
    assert geometry["contact_baseline"] == pytest.approx(
        source_origin + approach * 0.1034,
    )
    assert geometry["grip_site"] == pytest.approx(
        source_origin + approach * 0.097,
    )
    assert np.linalg.norm(
        geometry["contact_baseline"] - geometry["grip_site"],
    ) == pytest.approx(0.0064)
    compiled = _compile_grasp_candidate(candidate)
    assert compiled["pregrasp_pose"]["position_m"] == pytest.approx(
        source_origin + approach * 0.057,
    )
    assert compiled["approach_delta"]["translation_m"] == pytest.approx(
        [0.0, 0.0, 0.04],
    )
    assert compiled["compiler_checks"]["embodiment_mapping"] == (
        PANDA_GRIP_SITE_COMPILER_IDENTITY["id"]
    )


def test_grasp_scene_fusion_is_stable_and_requires_both_public_cameras(
    tmp_path: Path,
) -> None:
    runtime, _observation_id = _runtime(tmp_path)
    _control, observation_root, observation, _pose, _gripper = runtime._current_grip_state()
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root,
        observation=observation,
        workspace_bounds_m=BOUNDS,
        camera_names=("agentview", "wrist"),
        voxel_size_m=0.005,
        max_depth_m=2.5,
    )
    assert set(fused["source_index"].tolist()) == {0, 1}
    assert fused["deduplication"] == "stable_first_by_camera_order"

    duplicate = copy.deepcopy(observation)
    duplicate["cameras"]["wrist"] = copy.deepcopy(
        duplicate["cameras"]["agentview"]
    )
    duplicate_fusion = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root,
        observation=duplicate,
        workspace_bounds_m=BOUNDS,
        camera_names=("agentview", "wrist"),
        voxel_size_m=0.005,
        max_depth_m=2.5,
    )
    assert set(duplicate_fusion["source_index"].tolist()) == {0}
    assert duplicate_fusion["sources"][1]["retained_voxel_points"] == 0

    missing = copy.deepcopy(observation)
    del missing["cameras"]["wrist"]
    with pytest.raises(ValueError, match="requires camera wrist"):
        fuse_rgbd_pointcloud_sources(
            observation_root=observation_root,
            observation=missing,
            workspace_bounds_m=BOUNDS,
            camera_names=("agentview", "wrist"),
            voxel_size_m=0.005,
            max_depth_m=2.5,
        )


def test_viewer_projects_grasp_plan_preview_and_supports_hash_only_runs(
    tmp_path: Path,
) -> None:
    run = tmp_path / "grasp-plan-run"
    state_root = run / "policy-state/steps"
    state_root.mkdir(parents=True)
    (state_root / "step-0000.json").write_text(json.dumps({
        "step": 0,
        "observation": {"id": "obs-0"},
        "cameras": {},
    }), encoding="utf-8")
    (run / "policy-state/manifest.json").write_text(json.dumps({
        "steps": [{"step": 0, "path": "policy-state/steps/step-0000.json"}],
    }), encoding="utf-8")
    query_ref = "grasp-query-0123456789abcdef01234567"
    overlay = run / "openeta-tool-artifacts/grasp-queries" / query_ref / "overlay.png"
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (10, 20, 30)).save(overlay)
    records = [
        {"type": "observation", "observation_id": "obs-0",
         "policy_state": {"step": 0}},
        {"type": "grasp_plan_materialized", "observation_id": "obs-0",
         "query_ref": query_ref, "candidate_count": 2,
         "overlay_sha256": _sha256(overlay)},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    trajectory = compile_trajectory(run)
    preview = next(
        image for image in trajectory["steps"][0]["images"]
        if image.get("kind") == "grasp_plan"
    )
    assert preview["label"] == "grasp.plan policy contact sheet · 2 candidates"
    assert preview["query_ref"] == query_ref
    assert preview["path"].endswith(f"{query_ref}/overlay.png")

    records[1]["overlay_sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert not any(
        image.get("kind") == "grasp_plan"
        for image in compile_trajectory(run)["steps"][0]["images"]
    )

    review = overlay.parent / "global-review.png"
    Image.new("RGB", (48, 36), (30, 40, 50)).save(review)
    records[1]["artifact"] = {
        "path": review.relative_to(run).as_posix(),
        "sha256": _sha256(review),
        "bytes": review.stat().st_size,
    }
    records[1]["policy_overlay_artifact"] = {
        "path": overlay.relative_to(run).as_posix(),
        "sha256": _sha256(overlay),
        "bytes": overlay.stat().st_size,
    }
    records[1].pop("overlay_sha256", None)
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    projected = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") == "grasp_plan"
    ]
    assert [image["visibility"] for image in projected] == [
        "policy_visible", "review_only",
    ]
    assert projected[0]["label"] == "grasp.plan policy contact sheet · 2 candidates"
    assert projected[0]["path"].endswith("/overlay.png")
    assert projected[1]["label"] == "grasp.plan global input review · 2 candidates"
    assert projected[1]["path"].endswith("/global-review.png")


def test_viewer_projects_hash_bound_grounding_overlay(tmp_path: Path) -> None:
    run = tmp_path / "grounding-run"
    state_root = run / "policy-state/steps"
    state_root.mkdir(parents=True)
    (state_root / "step-0000.json").write_text(json.dumps({
        "step": 0,
        "observation": {"id": "obs-0"},
        "cameras": {},
    }), encoding="utf-8")
    (run / "policy-state/manifest.json").write_text(json.dumps({
        "steps": [{"step": 0, "path": "policy-state/steps/step-0000.json"}],
    }), encoding="utf-8")
    grounding_ref = "grounding-query-0123456789abcdef01234567"
    overlay = (
        run / "openeta-tool-artifacts/grounding-queries"
        / grounding_ref / "overlay.png"
    )
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (20, 30, 40)).save(overlay)
    previews = []
    for rank in range(3):
        preview = overlay.parent / f"candidate-T{rank}.png"
        Image.new("RGB", (64, 48), (30 + rank, 40, 50)).save(preview)
        previews.append({
            "rank": rank,
            "target_ref": f"target-{rank:024x}",
            "artifact": {
                "path": preview.name,
                "sha256": _sha256(preview),
                "bytes": preview.stat().st_size,
            },
        })
    records = [
        {"type": "observation", "observation_id": "obs-0",
         "policy_state": {"step": 0}},
        {"type": "grounding_query_materialized", "observation_id": "obs-0",
         "grounding_ref": grounding_ref, "candidate_count": 3,
         "camera": "agentview", "overlay_sha256": _sha256(overlay),
         "candidate_previews": previews},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    grounding_images = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") in {"grounding", "grounding_candidate"}
    ]
    assert len(grounding_images) == 4
    projected = grounding_images[0]
    assert projected["label"] == "vision.ground · agentview · 3 targets"
    assert projected["visibility"] == "policy_visible"
    assert projected["grounding_ref"] == grounding_ref
    assert [image["rank"] for image in grounding_images[1:]] == [0, 1, 2]
    assert grounding_images[1]["label"] == (
        "vision.ground T0 candidate enlargement · agentview"
    )

    records[1]["candidate_previews"][1]["artifact"]["sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    grounding_images = [
        image for image in compile_trajectory(run)["steps"][0]["images"]
        if image.get("kind") in {"grounding", "grounding_candidate"}
    ]
    assert [image.get("rank") for image in grounding_images[1:]] == [0, 2]

    records[1]["overlay_sha256"] = "0" * 64
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert not any(
        image.get("kind") == "grounding"
        for image in compile_trajectory(run)["steps"][0]["images"]
    )


def test_sam3_text_grounding_client_validates_ranked_arrays() -> None:
    def encoded(value: np.ndarray) -> str:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return base64.b64encode(stream.getvalue()).decode("ascii")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "segmentation",
                "warmup_complete": True,
                "load_count": 1,
                "instance_id": "grounding-test-instance",
                "adapter_sha256": "a" * 64,
                "started_unix_s": 1.0,
                "model": dict(SAM3_MODEL_IDENTITY),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            assert self.path == "/v1/ground-text"
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            assert request["schema"] == (
                "ora.model-service.ground-text-request.v1"
            )
            assert request["text"] == "red object"
            masks = np.zeros((2, 8, 8), dtype=np.uint8)
            masks[0, 1:5, 1:4] = 1
            masks[1, 2:7, 5:7] = 1
            response = {
                "schema": "ora.model-service.ground-text.v1",
                "request_id": "grounding-test-request",
                "model": dict(SAM3_MODEL_IDENTITY),
                "service_identity": {
                    "schema": "ora.model-service-response-identity.v1",
                    "instance_id": "grounding-test-instance",
                    "pid": 123,
                    "started_unix_s": 1.0,
                    "adapter_sha256": "a" * 64,
                },
                "masks_npy_base64": encoded(masks),
                "scores_npy_base64": encoded(
                    np.asarray([0.8, 0.3], dtype=np.float32),
                ),
                "boxes_xyxy_npy_base64": encoded(
                    np.asarray([[1, 1, 4, 5], [5, 2, 7, 7]], dtype=np.float32),
                ),
                "metrics": {
                    "queue_ms": 0.1, "inference_ms": 9.0, "total_ms": 9.1,
                },
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Sam3ApiSegmenter(
        endpoint=f"http://127.0.0.1:{server.server_port}",
    )
    try:
        assert client.require_ready()["load_count"] == 1
        candidates, diagnostics = client.ground(
            rgb_u8=np.zeros((8, 8, 3), dtype=np.uint8),
            text="red object",
            max_candidates=2,
        )
        assert [item["rank"] for item in candidates] == [0, 1]
        assert candidates[0]["bbox_xyxy"] == [1, 1, 4, 5]
        assert candidates[0]["backend_score"] == pytest.approx(0.8)
        assert diagnostics["model"] == SAM3_MODEL_IDENTITY
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_contact_graspnet_binding_requires_the_exact_live_instance() -> None:
    class Handler(BaseHTTPRequestHandler):
        instance_id = "grasp-test-instance"

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({
                "schema": "ora.model-service.health.v1",
                "ready": True,
                "service": "grasp_proposals",
                "warmup_complete": True,
                "load_count": 1,
                "instance_id": type(self).instance_id,
                "pid": 456,
                "adapter_sha256": "c" * 64,
                "started_unix_s": 2.0,
                "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
        "endpoint": endpoint,
        "instance_id": "grasp-test-instance",
        "pid": 456,
        "started_unix_s": 2.0,
        "adapter_sha256": "c" * 64,
        "model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
        "load_count": 1,
        "warmup_complete": True,
        "qualification": {
            "sha256": "d" * 64,
            "status": "passed",
            "created_unix_s": 2.0,
        },
    }
    client = ContactGraspNetApiProposer(
        endpoint=endpoint, expected_binding=binding,
    )
    try:
        assert client.require_ready()["instance_id"] == "grasp-test-instance"
        Handler.instance_id = "replacement-instance"
        with pytest.raises(ValueError, match="bound instance_id"):
            client.require_ready()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_grasp_plan_runtime_splits_the_bound_service_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class BoundSam:
        def __init__(self, *, endpoint: str, expected_binding: object) -> None:
            seen["sam_endpoint"] = endpoint
            seen["sam_binding"] = expected_binding

        def require_ready(self) -> dict[str, object]:
            return {"service": "segmentation", "instance_id": "sam"}

        def ground(self, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
            return [], {}

    class BoundGrasp:
        def __init__(
            self, *, endpoint: str, hover_offset_m: float,
            expected_binding: object,
        ) -> None:
            seen["grasp_endpoint"] = endpoint
            seen["hover_offset_m"] = hover_offset_m
            seen["grasp_binding"] = expected_binding

        def require_ready(self) -> dict[str, object]:
            return {"service": "grasp_proposals", "instance_id": "grasp"}

    monkeypatch.setattr(runtime_core, "Sam3ApiSegmenter", BoundSam)
    monkeypatch.setattr(runtime_core, "ContactGraspNetApiProposer", BoundGrasp)
    control, _observation_id = _publish_object_observation(tmp_path)
    segmentation_binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
    }
    grasp_binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": LIBERO_SURFACE,
    }
    ManipulationToolRuntime(
        control_root=control,
        manual_path=MANUAL,
        event_path=tmp_path / "events-bound.jsonl",
        workspace_bounds_m=BOUNDS,
        state_root=None,
        tool_surface_profile=LIBERO_SURFACE,
        controlled_end_effector="right",
        prerequisite_binding={
            "schema": "ora.model-service-set-binding.v1",
            "tool_surface": LIBERO_SURFACE,
            "services": {
                "segmentation": segmentation_binding,
                "grasp_proposals": grasp_binding,
            },
        },
    )
    assert seen == {
        "sam_endpoint": "http://127.0.0.1:8114",
        "sam_binding": segmentation_binding,
        "grasp_endpoint": "http://127.0.0.1:8115",
        "hover_offset_m": 0.04,
        "grasp_binding": grasp_binding,
    }


def test_capability_scoped_service_binding_replaces_the_rename_hack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 set binding is valid for any surface supporting its scope."""

    seen: dict[str, object] = {}

    class BoundSam:
        def __init__(self, *, endpoint: str, expected_binding: object) -> None:
            seen["sam_binding"] = expected_binding

        def ground(self, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
            return [], {}

    class BoundGrasp:
        def __init__(
            self, *, endpoint: str, hover_offset_m: float,
            expected_binding: object,
        ) -> None:
            seen["grasp_binding"] = expected_binding

        def require_ready(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(runtime_core, "Sam3ApiSegmenter", BoundSam)
    monkeypatch.setattr(runtime_core, "ContactGraspNetApiProposer", BoundGrasp)
    control, _observation_id = _publish_object_observation(tmp_path)
    # Issued under a different profile name: previously this forced the
    # experiment-side _rebind_service_surface hack.
    segmentation_binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": ROBOTWIN_SURFACE,
    }
    grasp_binding = {
        "schema": "ora.grasp-proposal-service-binding.v1",
        "tool_surface": ROBOTWIN_SURFACE,
    }

    def build(capability_scope: list[str]) -> ManipulationToolRuntime:
        return ManipulationToolRuntime(
            control_root=control,
            manual_path=MANUAL,
            event_path=tmp_path / "events-scoped.jsonl",
            workspace_bounds_m=BOUNDS,
            state_root=None,
            tool_surface_profile=LIBERO_SURFACE,
            controlled_end_effector="right",
            prerequisite_binding={
                "schema": "ora.model-service-set-binding.v2",
                "capability_scope": capability_scope,
                "services": {
                    "segmentation": segmentation_binding,
                    "grasp_proposals": grasp_binding,
                },
            },
        )

    build(["contact_graspnet_panda_grip_site_compiler_v2"])
    assert seen["sam_binding"] == segmentation_binding
    assert seen["grasp_binding"] == grasp_binding

    with pytest.raises(ToolError, match="capability scope is not supported"):
        build(["capability_the_surface_does_not_declare_v1"])
    with pytest.raises(ValueError, match="set binding is invalid"):
        build([])


def test_the_grounding_request_bound_is_the_service_limit_not_the_palette():
    """Two ceilings were conflated and the wrong one bounded the request.

    The palette's length says how many regions the overview can draw in
    distinguishable colours; the resident service says how many it will answer
    at all, and `infer_ground` refuses anything outside [1,3] with a 422. One
    cohort asked for 4, 6 and 8 across eleven calls and every one came back as
    an HTTP status with a documentation link -- retryable, but naming nothing
    the policy could change -- while all 291 calls at three or fewer succeeded.
    """

    from ora.agent.model_services import grounding_grasp as service
    from ora.agent.model_services.clients import GROUNDING_REQUEST_MAX_CANDIDATES

    source = inspect.getsource(service.GroundingSegmentService.infer_ground)
    assert f"1 <= maximum <= {GROUNDING_REQUEST_MAX_CANDIDATES}" in source, (
        "the request bound must be the range the service actually validates"
    )
    for profile in (ROBOTWIN_SURFACE, LIBERO_SURFACE):
        for item in resolve_tool_surface(profile).definitions:
            if item.get("name") != "vision_ground":
                continue
            schema = (item.get("inputSchema") or {})["properties"]["max_candidates"]
            assert schema["maximum"] == GROUNDING_REQUEST_MAX_CANDIDATES
            assert "1 to 3" in schema["description"]
