"""Trusted grant/publish adapter for atomic toolkit tools' public artifacts.

The tools pass typed artifact identities and bytes, never filesystem paths.
Only this adapter translates them to the existing immutable public stores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

from ora.agent.toolkit.interface.tool_data import (
    ToolData,
    artifact_key,
    filename,
    verify_artifact,
)
from ora.agent.toolkit.interface.types import AccessDenied, canonical, digest
from ora.agent.toolkit.host.perception import (
    grounding_preview,
    mask_localization,
    overhead_view,
    target_support,
)
from ora.agent.toolkit.host.perception.target_store import _materialize_target_index
from ora.agent.toolkit.host.shared.artifact_io import _append, _artifact
from ora.agent.toolkit.host.backend_selection import ROBODOJO_RGBD_CAPABILITY

#: RoboDojo's cameras, in the order every sealed tool context records them; also what a surface with no views means.
ROBODOJO_CAMERAS = ["head", "left_wrist", "right_wrist"]


def grounding_cameras(surface):
    """The cameras text grounding takes: RoboDojo's three, and elsewhere the surface's own observe views."""
    views = (surface.view_config or {}).get("default_observe_views")
    return list(ROBODOJO_CAMERAS if views is None or surface.supports(ROBODOJO_RGBD_CAPABILITY) else views)


def configuration(host):
    """Freeze public tool-surface and service configuration at the parent boundary."""
    surface = host.surface
    backend = host.grounding_backend
    schema = surface.result_schemas.get(
        "vision_ground", surface.result_schemas.get("vision.ground", {})
    )
    withheld = []
    for capability, keys in (
        ("target_principal_frame_v1", mask_localization._EXTENDED_FRAME_KEYS),
        ("target_support_height_v1", target_support.KEYS),
        (overhead_view.CAPABILITY, (*overhead_view.KEYS, overhead_view.POSE_KEY)),
    ):
        if not surface.supports(capability):
            withheld.extend(keys)
    result = dict(
        schema="ora.toolkit.tool-configuration.v1",
        surface_id=surface.profile_id,
        grounding_cameras=grounding_cameras(surface),
        grounding_backend=(
            None
            if backend is None
            else dict(
                id=backend.backend_id,
                version=backend.backend_version,
                config_sha256=digest(canonical(dict(backend.configuration()))),
            )
        ),
        depth_profile=grounding_preview.options(surface),
        withheld_localization_fields=withheld,
        perception_evidence=surface.supports("public_rgbd_perception_evidence_v1"),
        perception_feedback=surface.supports("public_rgbd_perception_feedback_v2"),
        grounding_result_fields=(
            list(schema.get("properties", {}))
            if schema.get("additionalProperties") is False
            else None
        ),
    )
    if surface.supports(overhead_view.CAPABILITY):
        result["overhead_view_pose"] = True
    segmentation = getattr(host, "segmentation_backend", None)
    result["segmentation_available"] = callable(getattr(segmentation, "segment_points", None))
    result["segmentation_backend"] = None if segmentation is None else {
        "id": segmentation.backend_id, "version": segmentation.backend_version,
        "config_sha256": digest(canonical(dict(segmentation.configuration()))),
    }
    # The runtime builds a default grasp backend on every surface; one with no controlled frame plans no grasp.
    if (getattr(host, "grasp_backend", None) is not None
            and (surface.view_config or {}).get("controlled_frame_convention") is not None):
        result["grasp"] = grasp_configuration(host)
    return result


def grasp_configuration(host):
    """Public compiler/configuration identities only; no live callback or privileged state."""
    from ora.agent.toolkit.host.grasp.compiler_registry import GRASP_COMPILER_IDENTITIES
    from ora.agent.toolkit.host.backend_selection import grasp_compiler_id, grasp_scene_cameras
    from ora.agent.toolkit.host.reach.alignment import frame_local_axes
    from ora.agent.toolkit.host.motion_rules.arguments import surface_flat_targets
    from ora.agent.toolkit.host.skills.lift_evidence import LIFT_M, CAPABILITY

    surface, backend = host.surface, host.grasp_backend
    contact = getattr(backend, "contact_backend", backend)
    topdown = getattr(backend, "topdown_backend", None)
    arms = ("right",) if surface.supports("single_arm_flat_targets_v1") else ("left", "right")

    def identity(client):
        return dict(
            id=client.backend_id,
            version=client.backend_version,
            configuration=dict(client.configuration()),
        )

    result = dict(
        workspace_bounds_m=dict(host.workspace_bounds_m),
        backend=dict(
            id=backend.backend_id,
            version=backend.backend_version,
            config_sha256=digest(canonical(dict(backend.configuration()))),
        ),
        contact_backend=identity(contact),
        topdown_backend=None if topdown is None else identity(topdown),
        compilers={
            arm: dict(GRASP_COMPILER_IDENTITIES[grasp_compiler_id(surface, arm)])
            for arm in arms
        },
        scene_cameras={
            arm: list(grasp_scene_cameras(surface, arm))
            for arm in arms
        },
        approach_axis=frame_local_axes(
            str((surface.view_config or {}).get("controlled_frame_convention"))
        )["approach"],
        fused_backend=surface.supports("fused_backend_grasp_input_v1"),
        perception_feedback=surface.supports("public_rgbd_perception_feedback_v2"),
        public_contact_pose=surface.supports("public_grasp_contact_pose_v1"),
        observed_extent=surface.supports("observed_extent_scene_review_v1"),
        execution_tool=(
            "move_pose"
            if surface.supports("enhanced_move_pose_v1")
            else "move_free_space"
        ),
        reach_annotation=surface.supports("grasp_plan_ik_reachability_annotation_v1"),
        jaw_symmetry=surface.supports("grasp_plan_jaw_symmetry_headroom_v1"),
        controlled_frame=(surface.view_config or {}).get("controlled_frame_convention"),
        lift_m=LIFT_M if surface.supports(CAPABILITY) else None,
        copyable_approach=surface.supports("copyable_grasp_approach_target_v1"),
        flat_targets=surface_flat_targets(surface),
        semantic_interaction=any(
            surface.supports(c)
            for c in (
                "motion_intent_router_v1",
                "motion_intent_router_v2",
                "single_arm_flat_targets_v1",
            )
        ),
    )
    from ora.agent.toolkit.host.grasp.proposers import CONTACT_FIRST

    if getattr(backend, "selection", None) == CONTACT_FIRST:  # only such a surface has the key (``_proposals``)
        result["selection"] = CONTACT_FIRST
    if surface.supports("grasp_pose_world_axes_v1"):
        # RO-27: the controlled point's local axes, whose world directions each published pose carries
        # (``grasp_plan.tool.public_result``). Only such a surface has the key.
        axes = frame_local_axes(str((surface.view_config or {}).get("controlled_frame_convention")))
        result["pose_world_axes"] = dict(approach=list(axes["approach"]), closing=list(axes["closing"]))
    return result


def publish(host, kind, key, files, metadata):
    """Publish only the fixed compatibility stores; callers cannot select paths."""
    artifact_key(kind, key)
    for name in files:
        filename(kind, name)
    verify_artifact(kind, key, files, metadata)
    if kind == "observation_overlay":
        from ora.agent.toolkit.host.observe import publish as publish_overlay

        return publish_overlay(host, key, files, metadata)
    if kind == "skill_preparation":
        return  # CAS only; control files are installed by the validated submit adapter.
    if kind == "target":
        observed, camera = metadata["observation_id"], metadata["camera"]
        for component in (observed, camera):
            if (
                not isinstance(component, str)
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
            ):
                raise AccessDenied("invalid tool artifact observation/camera")
        root = (
            Path(host.artifact_root)
            / "policy-observations"
            / observed
            / camera
            / "segmentation-queries"
            / key
        )
    elif kind == "segmentation_query":
        root = Path(host.artifact_root) / "segmentation-batches" / key
    elif kind in {"grounding_query", "grasp_query"}:
        root = (
            Path(host.artifact_root)
            / ("grounding-queries" if kind == "grounding_query" else "grasp-queries")
            / key
        )
    else:
        raise AccessDenied("unsupported compatibility publication")
    resolved = root.resolve()
    if (
        not resolved.is_relative_to(Path(host.artifact_root).resolve())
        or resolved != root.absolute()
    ):
        raise AccessDenied("compatibility artifact root aliases another path")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=root.parent) as staging:
        staged = Path(staging) / "artifact"
        staged.mkdir()
        for name, data in files.items():
            path = staged / name
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o444)
        if not root.exists():
            # A complete directory becomes visible at once, including its manifest.
            os.rename(staged, root)
        else:
            for name, data in files.items():
                path = root / name
                if path.is_symlink() or path.exists() and path.read_bytes() != data:
                    raise ValueError("immutable tool artifact content drift")
            # Repair an interrupted unsealed store only after validating every
            # extant member. The manifest is the final publication boundary.
            order = [name for name in files if name != "manifest.json"] + [
                "manifest.json"
            ]
            for name in order:
                if not (root / name).exists():
                    os.replace(staged / name, root / name)
    if kind == "target":
        _materialize_target_index(
            artifact_root=host.artifact_root,
            target_root=root,
            manifest_path=root / "manifest.json",
            observation_id=metadata["observation_id"],
            camera=metadata["camera"],
            target_ref=key,
        )


def _grounding_event(root, result):
    manifest = json.loads((root / "grounding-queries" / result["grounding_ref"] / "manifest.json").read_bytes())
    identity = manifest["identity"]
    return dict(
        type="grounding_query_materialized",
        observation_id=identity["observation_id"],
        camera=identity["camera"],
        text=identity["text"],
        grounding_ref=manifest["grounding_ref"],
        backend=manifest["backend"],
        candidate_count=len(manifest["candidates"]),
        overlay_sha256=manifest["overlay"]["sha256"],
        candidate_previews=manifest["visualization"]["candidate_previews"],
    )


def _grasp_event(root, result, run_root):
    query = root / "grasp-queries" / result["grasp_query_ref"]
    manifest = json.loads((query / "manifest.json").read_bytes())
    identity = manifest["identity"]

    def artifact(record):
        return _artifact(query / record["path"], run_root)

    event = dict(
        type="grasp_plan_materialized",
        observation_id=identity["observation_id"],
        target_ref=identity["target_ref"],
        query_ref=manifest["query_ref"],
        backend=manifest["backend"],
        candidate_count=len(manifest["candidates"]),
        overlay_sha256=manifest["overlay"]["sha256"],
        policy_overlay_artifact=artifact(manifest["overlay"]),
        input_pointcloud_artifact=artifact(manifest["input_pointcloud"]["artifact"]),
        manifest_artifact=_artifact(query / "manifest.json", run_root),
        visualization=manifest["visualization"],
        review_visualization=manifest["review_visualization"],
    )
    if "review_source_binding" in identity:
        event["review_source_binding"] = identity["review_source_binding"]
    review = manifest["review_visualization"]
    if review.get("status") == "available":
        event["artifact"] = artifact(review["artifact"])
    if manifest["review_pointcloud"] is not None:
        event["review_pointcloud_artifact"] = artifact(manifest["review_pointcloud"]["artifact"])
    return event


def log_query(host, name, result):
    """Log the query a perception tool answered from, where the handlers' surface logged it, for the Viewer.

    ``publish`` runs only for a new query, and a repeated one comes from its cache, so this reads the query the result
    names: a step with no event shows no image. Nothing here reaches the tool or a replay."""
    if name not in {"vision_ground", "grasp_plan"}:
        return  # and a host with no perception stores (robot_runtime_v2's) has no artifact root
    root = Path(host.artifact_root)
    if name == "vision_ground":
        event = _grounding_event(root, result)
    else:
        event = _grasp_event(root, result, host.control_root.parent)
    _append(host.event_path, dict(event, time=time.time()))


def bind(host, api, *, inputs=None):
    api.tool_data = ToolData(
        store=api.store,
        configuration=configuration(host),
        inputs=inputs,
        publisher=lambda *args: publish(host, *args),
    )
    return api
