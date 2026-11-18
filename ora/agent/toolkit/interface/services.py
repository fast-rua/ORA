"""Adapters for existing SAM3 and CGN clients, injectable without network use."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

from .blobs import BlobStore
from .projections import fields
from .types import InterfaceError, PublicData


def array_input(store: BlobStore, ref: dict) -> Any:
    import numpy as np
    if "array" in ref:
        return store.array(ref)
    if ref["media_type"] == "image/png":
        from PIL import Image
        return np.asarray(Image.open(BytesIO(store.read(ref))).convert("RGB"))
    if ref["media_type"] == "application/x-npy":
        return np.load(BytesIO(store.read(ref)), allow_pickle=False)
    raise InterfaceError("service input must be RGB PNG, NPY or a declared raw array")


def metadata(value: dict) -> dict:
    # These are public client semantic fields, never the entire result object.
    data = fields(value, "model request_id model_score returned_candidate_count image_encodings scene_point_count target_point_count raw_candidate_limit inference_seed seed_policy seed_identity_sha256", structured=("model",))
    if isinstance(value.get('_backend_input_regime'), dict):
        data['input_regime'] = fields(value['_backend_input_regime'],
            'schema policy status crop_points unique_points configured_input_points input_points padding_factor '
            'upstream_padding_factor point_order upstream_scene_order')
    if "_service_identity" in value:
        data["service_identity"] = value["_service_identity"]
    for key, allowed in (
        ("latency_ms", "roundtrip service_queue service_inference service_total"),
        ("mask_normalization", "method raw_component_count raw_pixel_count prompt_component_pixel_count retained_fraction"),
    ):
        if key in value:
            data[key] = fields(value[key], allowed)
    if "query_results" in value:
        data["query_results"] = [fields(row, "query_index point_uv raw_candidate_count error_code")
                                 for row in value["query_results"]]
    return data


def sam3_adapters(client: Any) -> dict[str, Callable]:
    """Use the existing validating client; fake clients implement the same calls."""
    def invoke(method: str, parameters: dict, inputs: dict, store: BlobStore) -> PublicData:
        rgb = array_input(store, inputs["rgb"])
        if method == "ground":
            candidates, meta = client.ground(rgb_u8=rgb, **parameters)
        elif method == "segment":
            mask, meta = client.segment(rgb_u8=rgb, depth_m=array_input(store, inputs["depth"]),
                                        point_uv=tuple(parameters["point_uv"]))
            return project_sam3(method, (mask, meta), store)
        else:
            candidates, meta = client.segment_points(rgb_u8=rgb, depth_m=array_input(store, inputs["depth"]),
                                                      points_uv=parameters["points_uv"])
        return project_sam3(method, (candidates, meta), store)

    return {name: (lambda params, inputs, store, method=name: invoke(method, params, inputs, store))
            for name in ("segment", "segment_points", "ground")}


def grasp_adapter(propose: Callable) -> Callable:
    """The trusted upper adapter supplies the existing proposer, with raw pool retained.

    Its compiler output is intentionally ignored here: task preferences and
    candidate selection remain above the data interface.
    """
    def invoke(parameters: dict, inputs: dict, store: BlobStore) -> PublicData:
        result = propose(parameters, array_input(store, inputs["scene"]), array_input(store, inputs["target"]))
        return project_grasp(result, store)
    return invoke


def project_sam3(method: str, result, store: BlobStore) -> PublicData:
    """Project the already executed service call without executing it again."""
    value, meta = result
    if method == "segment":
        return PublicData.from_json(dict(mask=store.put_array(value), metadata=metadata(meta)))
    rows = []
    for item in value:
        row = fields(item, "query_index source_index point_uv backend_score bbox_xyxy rank")
        row["mask"] = store.put_array(item["mask"])
        rows.append(row)
    return PublicData.from_json(dict(candidates=rows, metadata=metadata(meta)))


def project_grasp(result, store: BlobStore) -> PublicData:
    """Retain the full validated backend pool, never the compiler's selection."""
    _, meta = result
    raw = meta["_canonical_backend_output"]
    arrays = {name: store.put_array(raw[name]) for name in
              ("poses_camera", "scores", "contact_points_camera", "openings_m")}
    return PublicData.from_json(dict(arrays=arrays, metadata=metadata(meta),
                                     coordinate_frame="camera", length_unit="m",
                                     compiler_applied_to_output=False))
