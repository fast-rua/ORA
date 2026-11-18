"""Clients of the resident SAM3 and Contact-GraspNet loopback services.

The services themselves (``resident.py``, ``grounding_grasp.py``) run in their
own interpreters; this is the Harness side of the same wire: the loopback-only
endpoint rule, the qualified model identities, the readiness and
service-identity checks against a bound service, the array encoding, and the
failure classes a call can end in. The SAM3 client is whole here -- point,
point-batch and text-grounding requests with their response checks -- because
``operator.qualify`` exercises exactly those requests; Contact-GraspNet's is
the transport half, and ``toolkit.host.grasp.contact_graspnet_client.ContactGraspNetApiProposer`` adds the
proposal request and the embodiment compiler's admission. Nothing here imports
a tool module.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit

SAM3_MODEL_IDENTITY = {
    "family": "sam3",
    "source_commit": "6fe87d64a5beb9084923d7a9e002741178635b09",
    "checkpoint_sha256": (
        "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
    ),
}
CONTACT_GRASPNET_MODEL_IDENTITY = {
    "family": "contact_graspnet_pytorch",
    "source_commit": "da3dcfb2f53e43b186083ee4a9d1e232f73efc98",
    "checkpoint_sha256": (
        "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
    ),
    "config_sha256": (
        "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5"
    ),
}


#: ``transient``: the service did not answer (connection, timeout, 5xx).
#: ``permanent``: it answered as a different or misconfigured service (instance,
#: model or adapter identity drift, a malformed or unready health record); no
#: later call in this episode can match the bound identity again.
#: ``request``: it refused the request content with a 4xx status.
SERVICE_FAILURES = ("transient", "permanent", "request")


class ModelServiceError(ValueError):
    """One model-service call failed, with the class of failure and its detail."""

    def __init__(
        self, message: str, *, service: str, failure: str,
        status: int | None = None, detail: Any = None,
    ) -> None:
        if failure not in SERVICE_FAILURES:
            raise ValueError(f"unknown model-service failure class: {failure}")
        super().__init__(message)
        self.service = service
        self.failure = failure
        self.status = status
        self.detail = detail


def service_identity_drift(message: str, *, service: str) -> ModelServiceError:
    """A bound service answered with another identity or an unready health record."""

    return ModelServiceError(message, service=service, failure="permanent")


def http_failure(exc: Exception, *, service: str, what: str) -> ModelServiceError:
    """Classify one httpx failure, keeping the service's own ``detail`` text.

    ``str(HTTPStatusError)`` keeps only the status line and a documentation
    link; the 422 body that names the refused field was being dropped.
    """

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    detail = None
    if response is not None:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - any undecodable body keeps detail None
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
    failure = (
        "request" if isinstance(status, int) and 400 <= status < 500 else "transient"
    )
    text = f"{service} {what} failed"
    if isinstance(status, int):
        text += f" with HTTP {status}"
    if detail is not None:
        text += f": {detail}"
    elif not isinstance(status, int):
        text += f": {type(exc).__name__}"
    return ModelServiceError(
        text, service=service, failure=failure,
        status=status if isinstance(status, int) else None, detail=detail,
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """The canonical JSON sha256 ``toolkit.host.shared.artifact_io._canonical_hash`` computes.

    It names a bound service in ``Sam3ApiSegmenter.configuration``, which the
    segmentation tools hash into their backend identity, so it must stay the
    same function.
    """
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


#: The point batch the SAM3 service answers in one request; the service refuses
#: anything else (``resident.SegmentService``), and ``vision_segment`` publishes
#: the same bound (``segmentation_batch`` checks a call against it first).
MAX_POINTS = 8


def validate_points(points: Any, *, width: int, height: int) -> None:
    if (not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS
            or any(not isinstance(p, list) or len(p) != 2
                   or any(type(v) is not int for v in p) for p in points)):
        raise ValueError("points_uv must contain 1..8 independent integer [u,v] points")
    if any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
        raise ValueError("points_uv must lie inside the bound source image")


def _loopback_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint.rstrip("/"))
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model-service endpoint must be an unauthenticated loopback URL")
    return endpoint.rstrip("/")


def _encode_npy(value: Any) -> str:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_npy(value: Any) -> Any:
    import numpy as np

    if not isinstance(value, str):
        raise ValueError("model service returned a non-string array payload")
    try:
        raw = base64.b64decode(value, validate=True)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ValueError("model service returned an invalid array payload") from exc


def _verify_model_identity(
    received: Any, expected: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(received, Mapping):
        raise service_identity_drift("model service omitted model identity", service="model")
    for key, value in expected.items():
        if received.get(key) != value:
            raise service_identity_drift(f"model service {key} differs from the qualified identity", service="model")
    return dict(expected)


def _require_ready_service(
    *, client: Any, endpoint: str, service: str,
    expected_model: Mapping[str, str],
) -> dict[str, Any]:
    import httpx

    try:
        response = client.get(f"{endpoint}/healthz")
        response.raise_for_status()
        health = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise http_failure(exc, service=service, what="service readiness") from exc
    if not isinstance(health, Mapping):
        raise service_identity_drift(f"{service} service returned malformed readiness", service=service)
    if (
        health.get("schema") != "ora.model-service.health.v1"
        or health.get("ready") is not True
        or health.get("service") != service
        or health.get("warmup_complete") is not True
        or health.get("load_count") != 1
    ):
        raise service_identity_drift(f"{service} service is not warm and singly loaded", service=service)
    _verify_model_identity(health.get("model"), expected_model)
    if not isinstance(health.get("instance_id"), str) or not health["instance_id"]:
        raise service_identity_drift(f"{service} service omitted its model instance identity", service=service)
    if not isinstance(health.get("adapter_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", health["adapter_sha256"],
    ):
        raise service_identity_drift(f"{service} service omitted its adapter hash", service=service)
    started = health.get("started_unix_s")
    if not isinstance(started, (int, float)) or not math.isfinite(float(started)):
        raise service_identity_drift(f"{service} service omitted its start time", service=service)
    return dict(health)


def _normalize_expected_service_binding(
    binding: Mapping[str, Any], *, schema: str, endpoint: str,
) -> dict[str, Any]:
    value = dict(binding)
    required = {
        "schema", "tool_surface", "endpoint", "instance_id", "pid",
        "started_unix_s", "adapter_sha256", "model", "load_count",
        "warmup_complete", "qualification",
    }
    qualification = value.get("qualification")
    if (
        set(value) != required
        or value.get("schema") != schema
        or value.get("endpoint") != endpoint
        or not isinstance(value.get("tool_surface"), str)
        or not value["tool_surface"]
        or not isinstance(value.get("instance_id"), str)
        or not value["instance_id"]
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("started_unix_s"), bool)
        or not isinstance(value.get("started_unix_s"), (int, float))
        or not math.isfinite(float(value["started_unix_s"]))
        or not isinstance(value.get("adapter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["adapter_sha256"]) is None
        or not isinstance(value.get("model"), Mapping)
        or value.get("load_count") != 1
        or value.get("warmup_complete") is not True
        or not isinstance(qualification, Mapping)
        or qualification.get("status") != "passed"
        or not isinstance(qualification.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", qualification["sha256"]) is None
        or isinstance(qualification.get("created_unix_s"), bool)
        or not isinstance(qualification.get("created_unix_s"), (int, float))
        or not math.isfinite(float(qualification["created_unix_s"]))
    ):
        raise ValueError("expected model-service binding is invalid")
    return value


def _require_bound_service_identity(
    *, health: Mapping[str, Any], binding: Mapping[str, Any], service: str,
) -> None:
    for field in (
        "instance_id", "started_unix_s", "adapter_sha256", "model",
        "load_count", "warmup_complete", "pid",
    ):
        if health.get(field) != binding.get(field):
            raise service_identity_drift(
                f"{service} runtime service differs from bound {field}", service=service,
            )


def _require_response_service_identity(
    *, result: Mapping[str, Any], binding: Mapping[str, Any] | None,
    service: str,
) -> dict[str, Any]:
    identity = result.get("service_identity")
    required = {
        "schema", "instance_id", "pid", "started_unix_s", "adapter_sha256",
    }
    if (
        not isinstance(identity, Mapping)
        or set(identity) != required
        or identity.get("schema")
        != "ora.model-service-response-identity.v1"
        or not isinstance(identity.get("instance_id"), str)
        or not identity["instance_id"]
        or not isinstance(identity.get("pid"), int)
        or isinstance(identity.get("started_unix_s"), bool)
        or not isinstance(identity.get("started_unix_s"), (int, float))
        or not math.isfinite(float(identity["started_unix_s"]))
        or not isinstance(identity.get("adapter_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["adapter_sha256"]) is None
    ):
        raise service_identity_drift(f"{service} response omitted its service instance identity", service=service)
    if binding is not None:
        for field in ("instance_id", "pid", "started_unix_s", "adapter_sha256"):
            if identity.get(field) != binding.get(field):
                raise service_identity_drift(
                    f"{service} response differs from bound {field}", service=service,
                )
    return dict(identity)


#: What the resident SAM3 text-grounding service accepts, which is not what the
#: overlay palette can draw. ``infer_ground`` refuses anything outside [1,3] with
#: a 422, and the palette's 32 was being used as the request bound: eleven calls
#: in one cohort asked for 4, 6 or 8 and every one came back as an HTTP status
#: the policy could not act on, while all 291 calls at three or fewer succeeded.
#: The two limits are different questions and this is the one the wire answers.
GROUNDING_REQUEST_MAX_CANDIDATES = 3


@dataclass(frozen=True)
class Sam3ApiSegmenter:
    """Point-prompt SAM3 client for one already-warm loopback service."""

    endpoint: str = "http://127.0.0.1:8114"
    timeout_s: float = 30.0
    backend_id: str = "sam3_point_api"
    backend_version: str = "v1"
    expected_binding: Mapping[str, Any] | None = None
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import httpx

        object.__setattr__(self, "endpoint", _loopback_endpoint(self.endpoint))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("SAM3 service timeout must be positive")
        if self.expected_binding is not None:
            object.__setattr__(
                self, "expected_binding",
                _normalize_expected_service_binding(
                    self.expected_binding,
                    schema="ora.segmentation-service-binding.v1",
                    endpoint=self.endpoint,
                ),
            )
        object.__setattr__(
            self, "_client", httpx.Client(timeout=self.timeout_s),
        )

    def configuration(self) -> Mapping[str, Any]:
        return {
            "endpoint": self.endpoint,
            "timeout_s": self.timeout_s,
            "expected_model": dict(SAM3_MODEL_IDENTITY),
            "api_schema": "ora.model-service.segment-point.v1",
            "expected_binding_sha256": (
                None if self.expected_binding is None
                else _canonical_hash(self.expected_binding)
            ),
        }

    def close(self) -> None:
        self._client.close()

    def require_ready(self) -> dict[str, Any]:
        health = _require_ready_service(
            client=self._client, endpoint=self.endpoint,
            service="segmentation", expected_model=SAM3_MODEL_IDENTITY,
        )
        if self.expected_binding is not None:
            _require_bound_service_identity(
                health=health, binding=self.expected_binding, service="SAM3",
            )
        return health

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]:
        import cv2
        import httpx
        import numpy as np
        from PIL import Image

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        depth = np.asarray(depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("SAM3 requires aligned HxWx3 RGB and HxW depth")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {
            "schema": "ora.model-service.segment-point-request.v1",
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "point_uv": list(point_uv),
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/segment-point", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_failure(exc, service="SAM3", what="service request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.segment-point.v1":
            raise service_identity_drift("SAM3 service response schema mismatch", service="SAM3")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding, service="SAM3",
        )
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        mask = _decode_npy(result.get("mask_npy_base64")).astype(bool)
        if mask.shape != depth.shape:
            raise ValueError("SAM3 service mask shape differs from the source image")
        u, v = point_uv
        if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
            raise ValueError("SAM3 positive prompt is outside the source image")
        if not bool(mask[v, u]):
            raise ValueError("SAM3 service mask does not contain the positive prompt")
        component_count, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8,
        )
        prompt_label = int(labels[v, u])
        if component_count <= 1 or prompt_label == 0:
            raise ValueError("SAM3 service mask has no prompted foreground component")
        raw_pixel_count = int(np.count_nonzero(mask))
        raw_component_count = component_count - 1
        mask = labels == prompt_label
        prompt_component_pixel_count = int(np.count_nonzero(mask))
        service_metrics = result.get("metrics")
        if not isinstance(service_metrics, Mapping):
            raise ValueError("SAM3 service response lacks timing metrics")
        return mask, {
            "model": model,
            "_service_identity": service_identity,
            "model_score": float(result["score"]),
            "request_id": str(result["request_id"]),
            "mask_normalization": {
                "method": "positive_prompt_8_connected_component_v1",
                "raw_component_count": raw_component_count,
                "raw_pixel_count": raw_pixel_count,
                "prompt_component_pixel_count": prompt_component_pixel_count,
                "retained_fraction": (
                    prompt_component_pixel_count / raw_pixel_count
                ),
            },
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(service_metrics["queue_ms"]),
                "service_inference": float(service_metrics["inference_ms"]),
                "service_total": float(service_metrics["total_ms"]),
            },
        }

    def segment_points(
        self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        """Fetch independent alternatives; host-side selection owns the total K."""
        import httpx
        import numpy as np
        from PIL import Image

        rgb, depth = np.asarray(rgb_u8, dtype=np.uint8), np.asarray(depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("SAM3 requires aligned HxWx3 RGB and HxW depth")
        validate_points(points_uv, width=rgb.shape[1], height=rgb.shape[0])
        if self.expected_binding is not None:
            health = self.require_ready()
            if "segment_points_v1" not in health.get("capabilities", []):
                raise service_identity_drift("bound SAM3 service does not support batch point queries", service="SAM3")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {"schema": "ora.model-service.segment-points-request.v1",
                   "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
                   "points_uv": points_uv}
        started = time.perf_counter()
        try:
            response = self._client.post(f"{self.endpoint}/v1/segment-points", json=payload)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_failure(exc, service="SAM3", what="batch request") from exc
        if result.get("schema") != "ora.model-service.segment-points.v1":
            raise service_identity_drift("SAM3 batch response schema mismatch", service="SAM3")
        identity = _require_response_service_identity(result=result, binding=self.expected_binding, service="SAM3")
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        queries = result.get("queries")
        if not isinstance(queries, list) or len(queries) != len(points_uv):
            raise ValueError("SAM3 batch response lost query identities")
        raw_candidates, query_results = [], []
        for index, (query, point) in enumerate(zip(queries, points_uv)):
            if not isinstance(query, Mapping) or query.get("query_index") != index or query.get("point_uv") != point:
                raise ValueError("SAM3 batch response query identity drift")
            candidates, error = query.get("candidates"), query.get("error")
            if (not isinstance(candidates, list) or len(candidates) > 3
                    or error is not None and (not isinstance(error, Mapping) or candidates)):
                raise ValueError("SAM3 batch response has malformed candidates or failure")
            seen = set()
            for candidate in candidates:
                source_index, score = candidate.get("source_index"), candidate.get("score")
                if (type(source_index) is not int or source_index < 0 or source_index in seen
                        or type(score) not in (int, float) or not math.isfinite(score)):
                    raise ValueError("SAM3 batch candidate ordering/score is invalid")
                seen.add(source_index)
                mask = _decode_npy(candidate.get("mask_npy_base64"))
                if mask.shape != depth.shape or not np.isfinite(mask).all():
                    raise ValueError("SAM3 batch mask differs from the source image")
                raw_candidates.append({"query_index": index, "source_index": source_index,
                    "point_uv": point, "mask": mask.astype(bool), "backend_score": float(score)})
            query_results.append({"query_index": index, "point_uv": point,
                                  "raw_candidate_count": len(candidates),
                                  "error_code": "point_inference_failed" if error else None})
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping) or result.get("image_encodings") != 1:
            raise ValueError("SAM3 batch response lacks timing/encoding evidence")
        return raw_candidates, {"model": model, "_service_identity": identity,
            "request_id": str(result["request_id"]), "query_results": query_results,
            "image_encodings": 1,
            "latency_ms": {"roundtrip": (time.perf_counter() - started) * 1000,
                "service_queue": float(metrics["queue_ms"]),
                "service_inference": float(metrics["inference_ms"]),
                "service_total": float(metrics["total_ms"])}}

    def ground(
        self, *, rgb_u8: Any, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        import httpx
        import numpy as np
        from PIL import Image

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM3 text grounding requires one HxWx3 RGB image")
        prompt = text.strip() if isinstance(text, str) else ""
        if not 1 <= len(prompt) <= 120:
            raise ValueError("SAM3 text prompt must contain 1..120 characters")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= GROUNDING_REQUEST_MAX_CANDIDATES
        ):
            raise ValueError(
                "SAM3 text grounding max_candidates must be in "
                f"[1,{GROUNDING_REQUEST_MAX_CANDIDATES}]"
            )
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        payload = {
            "schema": "ora.model-service.ground-text-request.v1",
            "image_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "text": prompt,
            "max_candidates": max_candidates,
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/ground-text", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_failure(exc, service="SAM3", what="text-grounding request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.ground-text.v1":
            raise service_identity_drift("SAM3 text-grounding response schema mismatch", service="SAM3")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding, service="SAM3",
        )
        model = _verify_model_identity(result.get("model"), SAM3_MODEL_IDENTITY)
        masks = _decode_npy(result.get("masks_npy_base64")).astype(bool)
        scores = _decode_npy(result.get("scores_npy_base64")).astype(
            np.float64,
        ).reshape(-1)
        boxes = _decode_npy(result.get("boxes_xyxy_npy_base64")).astype(
            np.float64,
        )
        if masks.size == 0:
            masks = np.empty((0, rgb.shape[0], rgb.shape[1]), dtype=bool)
            boxes = np.empty((0, 4), dtype=np.float64)
        if (
            masks.ndim != 3
            or masks.shape[1:] != rgb.shape[:2]
            or boxes.ndim != 2
            or boxes.shape[1:] != (4,)
            or not len(masks) == len(scores) == len(boxes)
            or len(masks) > max_candidates
        ):
            raise ValueError("SAM3 text grounding returned unaligned candidates")
        candidates: list[dict[str, Any]] = []
        for source_index, (mask, score, box) in enumerate(
            zip(masks, scores, boxes, strict=True)
        ):
            if not np.all(np.isfinite(box)) or not math.isfinite(float(score)):
                raise ValueError("SAM3 text grounding returned non-finite values")
            if int(np.count_nonzero(mask)) < 1:
                continue
            clipped = [
                max(0, min(rgb.shape[1], int(round(float(box[0]))))),
                max(0, min(rgb.shape[0], int(round(float(box[1]))))),
                max(0, min(rgb.shape[1], int(round(float(box[2]))))),
                max(0, min(rgb.shape[0], int(round(float(box[3]))))),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError("SAM3 text grounding returned an invalid box")
            candidates.append({
                "source_index": source_index,
                "mask": mask,
                "backend_score": float(score),
                "bbox_xyxy": clipped,
            })
        candidates.sort(
            key=lambda item: (-float(item["backend_score"]), item["source_index"])
        )
        for rank, candidate in enumerate(candidates):
            candidate["rank"] = rank
        service_metrics = result.get("metrics")
        if not isinstance(service_metrics, Mapping):
            raise ValueError("SAM3 text-grounding response lacks timing metrics")
        return candidates, {
            "model": model,
            "_service_identity": service_identity,
            "request_id": str(result["request_id"]),
            "returned_candidate_count": len(candidates),
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(service_metrics["queue_ms"]),
                "service_inference": float(service_metrics["inference_ms"]),
                "service_total": float(service_metrics["total_ms"]),
            },
        }


@dataclass(frozen=True)
class ContactGraspNetServiceClient:
    """The loopback transport of one already-warm Contact-GraspNet service.

    Endpoint, bound service identity and readiness; the operator's readiness
    checks and the launch preflight need no more. ``toolkit.host.grasp.contact_graspnet_client`` extends it
    with the proposal request (``ContactGraspNetApiProposer``).
    """

    endpoint: str = "http://127.0.0.1:8115"
    timeout_s: float = 45.0
    expected_binding: Mapping[str, Any] | None = None
    _client: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import httpx

        object.__setattr__(self, "endpoint", _loopback_endpoint(self.endpoint))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Contact-GraspNet service timeout must be positive")
        self._check_settings()
        if self.expected_binding is not None:
            object.__setattr__(
                self, "expected_binding",
                _normalize_expected_service_binding(
                    self.expected_binding,
                    schema="ora.grasp-proposal-service-binding.v1",
                    endpoint=self.endpoint,
                ),
            )
        object.__setattr__(
            self, "_client", httpx.Client(timeout=self.timeout_s),
        )

    def _check_settings(self) -> None:
        """A subclass's own settings, checked before the binding and client exist."""

    def close(self) -> None:
        self._client.close()

    def require_ready(self) -> dict[str, Any]:
        health = _require_ready_service(
            client=self._client, endpoint=self.endpoint,
            service="grasp_proposals",
            expected_model=CONTACT_GRASPNET_MODEL_IDENTITY,
        )
        if self.expected_binding is not None:
            _require_bound_service_identity(
                health=health, binding=self.expected_binding,
                service="Contact-GraspNet",
            )
        return health


__all__ = [
    "CONTACT_GRASPNET_MODEL_IDENTITY", "ContactGraspNetServiceClient", "GROUNDING_REQUEST_MAX_CANDIDATES",
    "MAX_POINTS", "ModelServiceError", "SAM3_MODEL_IDENTITY", "SERVICE_FAILURES", "Sam3ApiSegmenter",
    "http_failure", "service_identity_drift", "validate_points",
]
