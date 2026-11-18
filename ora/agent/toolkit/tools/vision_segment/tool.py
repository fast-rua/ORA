"""Independent pixel queries with released per-point selection and public artifacts."""
import base64
import json
from typing import Any, Mapping

import numpy as np

from ...interface.types import InterfaceError, PublicToolFailure, canonical, digest
from ...lib import observation, perception, point_images, vision
from ...lib.values import Array, JsonValue
from ..errors import ToolRefusal
from ..vision_ground.tool import artifact, json_bytes, overhead_view_poses, source_identity, LOCALIZATION
from ._arguments import batch_point_arguments
from ._selection import (
    MAX_PREVIEW_IMAGES, MAX_RETURNED_CANDIDATES, MASK_IOU_THRESHOLD,
    POINT_COLOURS, UNANSWERED_COLOUR, RENDERER, SCHEMA, SELECTION_METHOD, select_masks,
)


def materialize_target(api, *, source, backend, observed, camera, pixels, item, position):
    mask = Array.of(item["mask"].astype(bool))
    identity = {
        "observation_id": observed, "camera": camera,
        "prompt_kind": "independent_point_candidate", "point_uv": item["point_uv"],
        "mask_sha256": digest(mask.numpy().astype(np.uint8).tobytes(order="C")),
        "source": source, "backend": backend, "localization_method": LOCALIZATION,
    }
    key = "target-" + digest(canonical(identity))[:24]
    summary = vision.mask_summary(mask, pixels.depth_m).json()
    localized = vision.localize_mask(mask, pixels.depth_m, pixels.calibration).json()
    cached = api.artifacts.tool_artifact(kind="target", key=key)
    if cached is not None:
        manifest = json.loads(api.artifacts.read_blob(cached["files"]["manifest.json"]))
        if manifest["identity"] != identity:
            raise ValueError("batch target source identity drift")
    else:
        files = {
            "mask.npy": vision.npy_bytes(Array.of(mask.numpy().astype(np.uint8))),
            "overlay.png": vision.png_bytes(point_images.point_mask_overlay(
                pixels.rgb, mask, point_uv=tuple(item["point_uv"]),
                bbox_xyxy=tuple(summary["bbox_xyxy"]))),
        }
        manifest = {
            "schema": "ora.segmented-target.v2", "target_ref": key, "identity": identity,
            "prompt": {"kind": "independent_point_candidate", "point_uv": item["point_uv"]},
            "backend": backend, "summary": summary, "localization": localized,
            "diagnostics": {"candidate_metadata_scope": "point_batch_query_manifest"},
            "private_mask": artifact("mask.npy", files["mask.npy"]),
            "overlay": artifact("overlay.png", files["overlay.png"]),
        }
        files["manifest.json"] = json_bytes(manifest)
        api.artifacts.publish_tool_artifact(
            kind="target", key=key, files=files,
            metadata={"observation_id": observed, "camera": camera})
    candidate = {
        "candidate_id": f"T{position}", "position": position, "target_ref": key,
        "backend_score": item["backend_score"], "source_queries": item["sources"],
        "shared_with_source_indexes": sorted({row["query_index"] for row in item["sources"]}),
        "representative_point_uv": item["point_uv"], "normalization": item["normalization"],
        "bbox_xyxy": summary["bbox_xyxy"], "mask_summary": summary, "localization": localized,
        "mask_touches_image_boundary": bool(item["mask"][0].any() or item["mask"][-1].any()
                                            or item["mask"][:, 0].any() or item["mask"][:, -1].any()),
        "overlay_sha256": manifest["overlay"]["sha256"],
    }
    drawing = perception.MaskCandidate(mask, JsonValue.of({**candidate, "rank": position}))
    return candidate, drawing


def query_outcomes(ledger, points, raw, diagnostics):
    queries = []
    backend_queries = diagnostics.get("query_results", [])
    if not isinstance(backend_queries, list) or len(backend_queries) != len(points):
        raise ValueError("batch backend omitted per-query completion evidence")
    for entry, point, query in zip(ledger, points, backend_queries):
        index = entry["query_index"]
        if query.get("query_index") != index or query.get("point_uv") != point:
            raise ValueError("batch backend query evidence identity drift")
        raw_count = sum(row["query_index"] == index for row in raw)
        backend_error = query.get("error_code")
        if query.get("raw_candidate_count") != raw_count or backend_error and raw_count:
            raise ValueError("batch backend query outcome contradicts its candidates")
        if backend_error:
            status, error_code = "failed", backend_error
        elif entry["candidate_ids"]:
            status, error_code = "candidates", None
        else:
            status = "no_candidate"
            error_code = "backend_returned_no_mask" if not raw_count else "mask_excludes_positive_point"
        queries.append({**entry, "raw_candidate_count": raw_count,
                        "candidate_count": len(entry["candidate_ids"]),
                        "status": status, "error_code": error_code})
    return queries


def materialize(api, obs, camera, pixels, *, points, maximum, config):
    refs = obs.camera(camera)
    observed = obs.identity.json()["observation_id"]
    source = source_identity(observed, camera, refs)
    backend = config["segmentation_backend"]
    profile = config.get("depth_profile")
    identity = {
        "observation_id": observed, "camera": camera, "points_uv": points,
        "max_candidates_per_point": maximum, "max_returned_candidates": MAX_RETURNED_CANDIDATES,
        "max_preview_images": MAX_PREVIEW_IMAGES, "source": source, "backend": backend,
        "selection": SELECTION_METHOD, "mask_iou_threshold": MASK_IOU_THRESHOLD,
        "renderer": RENDERER, **({} if profile is None else {"depth_profile": profile}),
    }
    key = "segment-query-" + digest(canonical(identity))[:24]
    cached = api.artifacts.tool_artifact(kind="segmentation_query", key=key, cache=True)
    if cached is not None:
        manifest = json.loads(api.artifacts.read_blob(cached["files"]["manifest.json"]))
        if manifest.get("identity") != identity or manifest.get("schema") != SCHEMA:
            raise ValueError("batch query identity drift")
        for candidate in manifest["candidates"]:
            if api.artifacts.tool_artifact(kind="target", key=candidate["target_ref"]) is None:
                raise ValueError("cached segmentation target unavailable")
        return manifest, cached, True
    segmented = perception.segment_points(
        api, rgb=JsonValue.of(api.artifacts.save_array(pixels.rgb.numpy())),
        depth_m=JsonValue.of(api.artifacts.save_array(pixels.depth_m.numpy())),
        points_uv=tuple(tuple(point) for point in points))
    raw = [{**item.fields.json(), "mask": item.mask.numpy()} for item in segmented.candidates]
    diagnostics = segmented.metadata.json()
    service_identity = diagnostics.pop("service_identity", None)
    selected, selection = select_masks(raw, points_uv=points, shape=pixels.depth_m.shape,
                                       max_candidates=maximum)
    ledger, preview_count = selection.pop("queries"), selection.pop("preview_count")
    files = {"raw-masks.npz": vision.npz_bytes(tuple(
        (f"q{row['query_index']}_c{row['source_index']}", Array.of(np.asarray(row["mask"], dtype=np.uint8)))
        for row in raw))}
    candidates, drawings = [], []
    for position, item in enumerate(selected):
        candidate, drawing = materialize_target(
            api, source=source, backend=backend, observed=observed, camera=camera,
            pixels=pixels, item=item, position=position)
        candidates.append(candidate)
        drawings.append(drawing)
    leading = tuple(drawings[int(row["candidate_ids"][0][1:])] if row["candidate_ids"] else None
                    for row in ledger)
    files["overview.png"] = vision.png_bytes(point_images.point_overview(
        pixels.rgb, points_uv=tuple(tuple(point) for point in points), candidates=leading,
        colours=POINT_COLOURS, unanswered_colour=UNANSWERED_COLOUR))
    names, previews = ["overview.png"], []
    for drawing in drawings[:preview_count]:
        fields = drawing.fields.json()
        preview = vision.candidate_inspection(
            pixels.rgb, drawing, text="independent point proposals", points_uv=points,
            **({} if profile is None else {
                "depth_m": pixels.depth_m, "calibration": pixels.calibration,
                "localization": JsonValue.of(fields["localization"]), "support_line": profile["support_line"]}))
        name = f"candidate-{fields['candidate_id']}.png"
        files[name] = vision.png_bytes(preview.rgb)
        names.append(name)
        previews.append(preview.metadata.json())
    queries = query_outcomes(ledger, points, raw, diagnostics)
    answered = sum(row["status"] == "candidates" for row in queries)
    manifest = {
        "schema": SCHEMA, "query_ref": key, "identity": identity, "backend": backend,
        "status": "candidates" if answered == len(queries) else "partial" if answered else "no_candidate",
        "candidates": candidates, "queries": queries, "selection": selection,
        "diagnostics": diagnostics, "response_service_identity": service_identity,
        "raw_masks": artifact("raw-masks.npz", files["raw-masks.npz"]),
        "raw_candidates": [{k: v for k, v in row.items() if k != "mask"} for row in raw],
        "images": [artifact(name, files[name]) for name in names], "candidate_previews": previews,
    }
    files["manifest.json"] = json_bytes(manifest)
    record = api.artifacts.publish_tool_artifact(kind="segmentation_query", key=key, files=files,
                                                metadata={"observation_id": observed, "camera": camera})
    return manifest, record, False


def run(api, arguments):
    """Replay: recorded RGB-D; re-request SAM3, then pure selection and drawing.

    No physical actions, hidden state or semantic identity oracle. Existing
    immutable query artifacts may be reused live, never as replay inference.
    """
    if not {"camera", "points_uv"} <= set(arguments) or set(arguments) - {"camera", "points_uv", "max_candidates", "step"}:
        raise ToolRefusal("vision_segment requires camera and points_uv; optional max_candidates and step")
    camera = arguments["camera"]
    if not isinstance(camera, str) or not camera:
        raise ToolRefusal("camera must be a public camera name", field="camera")
    config = api.artifacts.tool_context()
    selected = config["vision_segment"]
    if "error" in selected:
        raise ToolRefusal.from_public(selected["error"])
    obs = observation.observe(api, identity=JsonValue.of(selected["identity"]))
    if camera not in obs.manifest.json().get("cameras", {}):
        raise ToolRefusal("camera is not policy-visible")
    errors = obs.result.data.json().get("camera_errors", {})
    if camera in errors:
        raise ToolRefusal.from_public(errors[camera])
    try:
        pixels = observation.load_camera(api, obs.camera(camera))
    except ValueError as error:  # an uncalibrated camera: no world geometry for its masks
        raise ToolRefusal(str(error), field="camera", received=camera, retry_requires_changed_arguments=True) from error
    height, width = pixels.rgb.shape[:2]
    maximum = batch_point_arguments(arguments["points_uv"], arguments.get("max_candidates", 3),
                                    width=width, height=height)
    if not config["segmentation_available"]:
        raise ToolRefusal("selected segmentation backend does not support independent batch queries",
                          code="tool_capability_unavailable", retryable=False)
    try:
        manifest, record, cached = materialize(api, obs, camera, pixels,
            points=arguments["points_uv"], maximum=maximum, config=config)
    except PublicToolFailure as error:
        raise ToolRefusal.from_public(error.public()) from error
    except InterfaceError:
        raise
    except ValueError as error:
        raise ToolRefusal(str(error), code="segmentation_failed") from error
    result = public_result(manifest, cache_reused=cached)
    for candidate in result["candidates"]:
        for key in config["withheld_localization_fields"]:
            candidate["localization"].pop(key, None)
    overhead_view_poses(api, result, config)
    result["images_base64"] = [base64.b64encode(api.artifacts.read_blob(record["files"][row["path"]])).decode("ascii")
                               for row in manifest["images"]]
    return result


def public_result(manifest: Mapping[str, Any], *, cache_reused: bool) -> dict[str, Any]:
    identity = manifest["identity"]
    previews = manifest["candidate_previews"]
    return {"schema": SCHEMA, "status": manifest["status"], "query_ref": manifest["query_ref"],
        "observation_id": identity["observation_id"], "camera": identity["camera"],
        "points_uv": identity["points_uv"], "query_count": len(manifest["queries"]),
        "candidates": manifest["candidates"], "queries": manifest["queries"],
        "selection": manifest["selection"],
        "candidate_previews": [{"image_index": i, "position": record["rank"],
            **{k: record[k] for k in ("target_ref", "source_dimensions", "source_bbox_xyxy",
                "crop_xyxy", "render_dimensions", "resampling")}}
            for i, record in enumerate(previews, 1)],
        "image_order": ["overview", *[c["candidate_id"] for c in manifest["candidates"][:len(previews)]]],
        "cache_reused": cache_reused, "semantic_identity_verified": False,
        "claim_boundary": "independent point-prompt proposals with a per-point mask budget; "
            "one point never consumes another point's masks, an empty point means no mask at that "
            "pixel rather than an exhausted budget, and omitted_over_* counts every mask withheld; "
            "inspect the proposals and choose semantic identity; crops add no sensor detail"}
