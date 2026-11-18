"""Method-specific semantic comparison; complete response hashes remain auditable."""
from __future__ import annotations

import json

import numpy as np

from .blobs import BlobStore
from .types import canonical, digest


# Exact public projection paths, never a recursive deletion by field name.
VOLATILE_FIELDS = {
    method: tuple(f"/data/metadata/{key}" for key in
                  ("request_id", "latency_ms", "timings", "queue"))
    for method in ("segment", "segment_points", "ground", "grasp_candidates")
}
ABSOLUTE_TOLERANCES = {
    "backend_score": 1e-6, "model_score": 1e-6, "scores": 1e-6,
    "poses_camera": 1e-5, "contact_points_camera": 1e-5, "openings_m": 1e-5,
}


def compare(method: str, reference: bytes, actual: bytes, *,
            reference_store: BlobStore, actual_store: BlobStore) -> dict:
    """Compare ordered public fields and verified arrays with rtol=0.

    Masks, identities, counts, array layouts and undeclared numeric fields are
    exact. Storage paths are not content. No assignment or candidate sorting is
    performed to improve a result.
    """
    volatile = VOLATILE_FIELDS[method]
    differences: list[dict] = []
    candidates: list[dict] = []

    def normalize(value, store, path=""):
        if isinstance(value, dict):
            if {"path", "sha256", "bytes", "media_type"} <= value.keys():
                store.register(value)
                store.read(value)
                return {k: v for k, v in value.items() if k != "path"}
            return {k: normalize(v, store, path + "/" + k)
                    for k, v in value.items() if path + "/" + k not in volatile}
        if isinstance(value, list):
            return [normalize(v, store, path + f"/{i}") for i, v in enumerate(value)]
        return value

    left, right = json.loads(reference), json.loads(actual)

    def changed(path, kind, **details):
        differences.append(dict(path=path, kind=kind, within_tolerance=False, **details))

    def numeric(a, b, path, field):
        if a.shape != b.shape or a.dtype != b.dtype:
            changed(path, "array_layout", reference_shape=list(a.shape), actual_shape=list(b.shape),
                    reference_dtype=str(a.dtype), actual_dtype=str(b.dtype))
            return
        if np.array_equal(a, b):
            return
        if field == "mask":
            intersection = int(np.count_nonzero(np.logical_and(a, b)))
            union = int(np.count_nonzero(np.logical_or(a, b)))
            changed(path, "mask", different_pixels=int(np.count_nonzero(a != b)),
                    pixel_iou=intersection / union if union else 1.0)
            return
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        finite = bool(np.all(np.isfinite(delta)))
        maximum = float(np.max(delta)) if finite and delta.size else None
        atol = ABSOLUTE_TOLERANCES.get(field, 0.0)
        differences.append(dict(path=path, kind="numeric", max_absolute_difference=maximum,
                                absolute_tolerance=atol, relative_tolerance=0,
                                within_tolerance=atol > 0 and finite and maximum is not None and maximum <= atol))

    def walk(a, b, path=""):
        if path in volatile:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if "array" in a and "path" in a and "array" in b and "path" in b:
                reference_store.register(a)
                actual_store.register(b)
                aa, bb = reference_store.array(a), actual_store.array(b)
                if a["array"] != b["array"] or a["media_type"] != b["media_type"]:
                    changed(path, "array_layout", reference=a["array"], actual=b["array"])
                numeric(aa, bb, path, path.rsplit("/", 1)[-1])
                if path.endswith("/poses_camera"):
                    # Exact row permutation is reported separately from positional pose errors.
                    ah = [digest(row.tobytes()) for row in aa]
                    bh = [digest(row.tobytes()) for row in bb]
                    candidates.append(dict(path=path, reference_count=len(ah), actual_count=len(bh),
                                           ordering_changed=False if ah == bh else True if sorted(ah) == sorted(bh) else None))
                return
            if {"path", "sha256", "bytes", "media_type"} <= a.keys() and {"path", "sha256", "bytes", "media_type"} <= b.keys():
                if normalize(a, reference_store) != normalize(b, actual_store):
                    changed(path, "blob")
                return
            for key in sorted(a.keys() | b.keys()):
                child = path + "/" + key
                if child in volatile:
                    continue
                if key not in a or key not in b:
                    changed(child, "missing_field", reference_present=key in a, actual_present=key in b)
                else:
                    walk(a[key], b[key], child)
            return
        if isinstance(a, list) and isinstance(b, list):
            if path.endswith("/candidates"):
                def identities(rows):
                    return [{k: row[k] for k in ("query_index", "source_index", "rank") if k in row}
                            for row in rows]
                candidates.append(dict(path=path, reference_count=len(a), actual_count=len(b),
                                       ordering_changed=identities(a) != identities(b)))
            if len(a) != len(b):
                changed(path, "count", reference_count=len(a), actual_count=len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + f"/{i}")
            return
        if type(a) in (int, float) and type(b) in (int, float) and a != b:
            numeric(np.asarray(a, dtype=float), np.asarray(b, dtype=float), path, path.rsplit("/", 1)[-1])
        elif type(a) != type(b) or a != b:
            changed(path, "value", reference=a, actual=b)

    normalized_left = normalize(left, reference_store)
    normalized_right = normalize(right, actual_store)
    walk(left, right)
    status = ("equal" if not differences else "within_tolerance"
              if all(d["within_tolerance"] for d in differences) else "different")
    return dict(status=status, scope="semantic_content", volatile_fields=list(volatile),
                differences=differences, candidates=candidates,
                reference_content_sha256=digest(canonical(normalized_left)),
                actual_content_sha256=digest(canonical(normalized_right)),
                reference_sha256=digest(reference), actual_sha256=digest(actual),
                envelope_equal=reference == actual)
