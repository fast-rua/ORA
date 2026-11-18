"""Per-point mask ledgers, bounded delivery, source-bound artifacts and service wire."""

import base64
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from ora.agent.composition.surface import tool_names
from PIL import Image

from tests.support.agent.surface_catalog import surface_of
from ora.agent.model_services import clients as service_clients
from ora.agent.model_services.resident import SegmentService, _encode_npy


#: The kept default surfaces; each binds the v2 batch contract.
DEFAULT_PROFILES = ("robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                    "robodojo_placement_toolkit_v5")

def test_service_encodes_once_and_never_accumulates_point_prompts():
    calls, encodings = [], []
    torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *_a, **_k: nullcontext(),
        bfloat16="bf16", cuda=SimpleNamespace(synchronize=lambda: None))
    marker = object()

    def set_image(image):
        encodings.append(image.size)
        return marker

    def predict(state, **kwargs):
        assert state is marker
        points = kwargs["point_coords"]
        assert points.shape == (1, 2) and kwargs["point_labels"].tolist() == [1]
        assert "mask_input" not in kwargs
        calls.append(points.copy())
        if int(points[0, 0]) == 10:
            raise RuntimeError("one point failed")
        masks = np.ones((3, 32, 32), dtype=np.uint8)
        return masks, np.array([.6, .9, .8]), None

    service = object.__new__(SegmentService)
    service.torch, service.processor, service.model = torch, SimpleNamespace(set_image=set_image), SimpleNamespace(predict_inst=predict)
    service.model_identity, service.requests = {"test": "fake"}, 0
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    result = service.infer_batch({"schema": "ora.model-service.segment-points-request.v1",
        "image_png_base64": base64.b64encode(image.getvalue()).decode(), "points_uv": [[5, 5], [10, 10], [15, 15]]})
    assert len(encodings) == 1 and len(calls) == 3 and result["image_encodings"] == 1
    assert [len(q["candidates"]) for q in result["queries"]] == [3, 0, 3]
    assert result["queries"][1]["error"]["code"] == "point_inference_failed"
    assert service.requests == 1


def test_sam_batch_client_uses_one_request_and_validates_query_identity():
    requests, drift = [], []
    identity = {"schema": "ora.model-service-response-identity.v1",
                "instance_id": "fixture", "pid": 1, "started_unix_s": 1., "adapter_sha256": "a" * 64}

    def response(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/segment-points"
        queries = [{"query_index": q + int(bool(drift)), "point_uv": p, "error": None,
                    "candidates": [{"source_index": 0, "score": .9,
                        "mask_npy_base64": _encode_npy(np.ones((32, 32), dtype=np.uint8))}]}
                   for q, p in enumerate(body["points_uv"])]
        return httpx.Response(200, json={"schema": "ora.model-service.segment-points.v1",
            "request_id": "query-1", "model": service_clients.SAM3_MODEL_IDENTITY, "service_identity": identity,
            "queries": queries, "image_encodings": 1, "metrics": {"queue_ms": 0, "inference_ms": 1, "total_ms": 1}})

    segmenter = service_clients.Sam3ApiSegmenter()
    segmenter._client.close()
    object.__setattr__(segmenter, "_client", httpx.Client(transport=httpx.MockTransport(response)))
    try:
        args = dict(rgb_u8=np.zeros((32, 32, 3), np.uint8), depth_m=np.ones((32, 32)), points_uv=[[5, 5], [10, 10]])
        raw, metadata = segmenter.segment_points(**args)
        assert len(requests) == 1 and len(raw) == 2 and metadata["image_encodings"] == 1
        drift.append(True)
        with pytest.raises(ValueError, match="identity"):
            segmenter.segment_points(**args)
    finally:
        segmenter.close()


@pytest.mark.parametrize("profile", DEFAULT_PROFILES)
def test_max_candidates_help_states_the_budget_it_now_is(profile):
    """Its meaning changed from a pooled total to a per-point budget."""
    surface = surface_of(profile)
    definition = tool_names.definition_for(surface.definitions, "vision_segment")
    description = definition["inputSchema"]["properties"]["max_candidates"]["description"]
    assert "per-point" in description.lower() and "not a total" in description.lower()
    # The released text said "Total maximum distinct masks after pooling ... not a
    # per-point limit", which is now the opposite of what the runtime does.
    assert "pool" not in description.lower()
    assert "per-point" in definition["description"].lower()
    assert "TOTAL" not in definition["description"]
    positions = [definition["description"].index(marker) for marker in (
        "Use when:", "Effect:", "Returns:", "Does not:", "On error:")]
    assert positions == sorted(positions)
